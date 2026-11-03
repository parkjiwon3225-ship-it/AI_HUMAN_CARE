# ===== cell P1 (notebook C, CPU runtime): export v1_speech for the Raspberry Pi + parity checks =====
import os, json
import numpy as np
import torch, torch.nn as nn
import onnxruntime as ort

SR, CROP = 16000, 40000
N_FFT, HOP, N_MELS, FMIN, FMAX = 512, 160, 64, 50.0, 8000.0
SAVE_DIR = "/content/drive/MyDrive/LAB/models"
NAME = "v1_speech"
OUT_DIR = SAVE_DIR + "/pi"
os.makedirs(OUT_DIR, exist_ok=True)

def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    h2m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    m2h = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    pts = m2h(np.linspace(h2m(fmin), h2m(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c))) * (2.0 / (r - l))
    return fb

class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("win", torch.hann_window(N_FFT))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank()))
    def forward(self, wav):
        spec = torch.stft(wav, N_FFT, hop_length=HOP, win_length=N_FFT, window=self.win,
                          center=True, pad_mode="reflect", return_complex=True)
        mel = torch.matmul(self.fb, spec.abs() ** 2)
        x = torch.log(mel + 1e-6)
        mu = x.mean(dim=(1, 2), keepdim=True)
        sd = x.std(dim=(1, 2), keepdim=True)
        return (x - mu) / (sd + 1e-5)

class Net(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.mel = LogMel()
        self.body = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(64, n_classes)
    def forward(self, wav):
        x = self.mel(wav)
        x = self.body(x.unsqueeze(1)).mean(dim=(2, 3))
        return self.fc(self.drop(x))

class Head(nn.Module):
    """log-mel (B,1,64,251) -> logits (B,2); this is the part that runs in onnxruntime on the Pi"""
    def __init__(self, net):
        super().__init__()
        self.body, self.fc = net.body, net.fc
    def forward(self, x):
        return self.fc(self.body(x).mean(dim=(2, 3)))

def logmel_np(x, fb):
    """numpy version of LogMel (what the Pi will run): x float32 (40000,) -> (64, 251)"""
    xp = np.pad(x.astype(np.float32), (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + len(x) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    win = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    spec = np.abs(np.fft.rfft(xp[idx] * win, axis=1)) ** 2
    m = np.log(fb @ spec.T + 1e-6)
    return ((m - m.mean()) / (m.std(ddof=1) + 1e-5)).astype(np.float32)

ck = torch.load("%s/urgent_%s.pt" % (SAVE_DIR, NAME), map_location="cpu", weights_only=False)
net = Net()
net.load_state_dict(ck["model"])
net.eval()
fb = net.mel.fb.numpy().copy()

head = Head(net).eval()
dummy = torch.zeros(1, 1, N_MELS, 1 + CROP // HOP)
onnx_path = "%s/urgent_%s.onnx" % (OUT_DIR, NAME)
kw = dict(input_names=["logmel"], output_names=["logits"], opset_version=17,
          dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}})
try:
    torch.onnx.export(head, dummy, onnx_path, dynamo=False, **kw)
except TypeError:
    torch.onnx.export(head, dummy, onnx_path, **kw)
print("exported:", onnx_path, "| size KB:", round(os.path.getsize(onnx_path) / 1024, 1))

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
t = np.arange(CROP) / SR
tests = [rng.standard_normal(CROP).astype(np.float32) * 0.05,
         (0.3 * np.sin(2 * np.pi * 440 * t) + 0.02 * rng.standard_normal(CROP)).astype(np.float32),
         (rng.standard_normal(CROP) * 0.2 * np.exp(-3 * t)).astype(np.float32),
         (rng.standard_normal(CROP) * 0.001).astype(np.float32)]
worst_f, worst_p = 0.0, 0.0
for x in tests:
    f_t = net.mel(torch.from_numpy(x[None])).numpy()[0]
    f_n = logmel_np(x, fb)
    worst_f = max(worst_f, float(np.abs(f_t - f_n).max()))
    with torch.no_grad():
        p_t = torch.softmax(net(torch.from_numpy(x[None])), dim=1).numpy()[0, 1]
    lg = sess.run(None, {"logmel": f_n[None, None].astype(np.float32)})[0][0]
    p_o = float(np.exp(lg[1]) / np.exp(lg).sum())
    worst_p = max(worst_p, abs(float(p_t) - p_o))
print("feature parity (torch vs numpy): max abs diff = %.2e" % worst_f)
print("probability parity (torch full model vs numpy features + onnx): max abs diff = %.2e" % worst_p)
print("RESULT:", "OK" if (worst_f < 1e-2 and worst_p < 1e-3) else "CHECK THE NUMBERS ABOVE")

np.save(OUT_DIR + "/mel_fb.npy", fb)
cfg = dict(model=NAME, sr=SR, crop_samples=CROP, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
           input_shape=[1, 1, N_MELS, 1 + CROP // HOP], classes=["ambient", "urgent"],
           norm="per-window mean/std (ddof=1) of log(mel+1e-6)")
rp = "%s/results_%s.json" % (SAVE_DIR, NAME)
if os.path.exists(rp):
    res = json.load(open(rp))
    cfg["thr_1pct_ambient"], cfg["thr_1pct_speech"] = res["thr"], res["thr_s"]
json.dump(cfg, open(OUT_DIR + "/urgent_%s_config.json" % NAME, "w"), indent=1)
print("saved to", OUT_DIR, ":", sorted(os.listdir(OUT_DIR)))
