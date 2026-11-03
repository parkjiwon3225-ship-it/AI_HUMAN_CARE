# ===== cell S1 (notebook D): helpers, data, ONNX model =====
import os, csv, json, time, collections
import numpy as np
import onnxruntime as ort
import soundfile as sf

SR, WIN = 16000, 40000                      # 16 kHz, 2.5 s window
D, P, OUT = "/content/data", "/content/pi", "/content/drive/MyDrive/LAB/streams"
cfg = json.load(open(P + "/urgent_v1_speech_config.json"))
FB = np.load(P + "/mel_fb.npy")
N_FFT, HOP = cfg["n_fft"], cfg["hop"]
HANN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
IDX = np.arange(N_FFT)[None, :] + HOP * np.arange(1 + WIN // HOP)[:, None]
sess = ort.InferenceSession(P + "/urgent_v1_speech.onnx", providers=["CPUExecutionProvider"])

def logmel_batch(xs):
    """numpy log-mel for a batch (B, 40000) -> (B, 64, 251); same math as the Raspberry Pi code"""
    xp = np.pad(xs, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
    spec = np.abs(np.fft.rfft(xp[:, IDX] * HANN, axis=2)) ** 2
    m = np.log(np.einsum("mf,btf->bmt", FB, spec) + 1e-6)
    mu = m.mean(axis=(1, 2), keepdims=True)
    sd = m.std(axis=(1, 2), ddof=1, keepdims=True)
    return ((m - mu) / (sd + 1e-5)).astype(np.float32)

def load_set(split, grp):
    with open("%s/meta_%s_%s.csv" % (D, split, grp), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    names = sorted(set(r["shard_file"] for r in rows))
    arrs, off, n = [], {}, 0
    for nm in names:
        a = np.load(os.path.join(D, nm))
        off[nm] = n
        n += a.shape[0]
        arrs.append(a)
    X = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    idx = np.array([off[r["shard_file"]] + int(r["row"]) for r in rows])
    return X, rows, idx

AX, Arows, Aidx = load_set("Validation", "ambient")        # background noise windows (4 s each)
UX, Urows, Uidx = load_set("Validation", "urgent")         # urgent clips
Ureal = np.array([int((min(float(r["clip_sec"]), 4.0) - 0.02) * SR) for r in Urows])
Ukeep = np.where(Ureal >= WIN)[0]                          # real sound >= 2.5 s
SX = np.load(D + "/speech_test.npy")                       # normal speech, unseen speakers
print("ambient windows:", len(AX), "| urgent clips:", len(Ukeep), "| speech windows:", len(SX))

def amb_window(i):
    return AX[Aidx[i]].astype(np.float32) / 32768.0

def urg_clip(k):
    j = Ukeep[k]
    return UX[Uidx[j]][:Ureal[j]].astype(np.float32) / 32768.0, Urows[j]

def speech_clip(i, rng):
    w = SX[i].astype(np.float32) / 32768.0
    n = int(rng.uniform(2.5, 4.0) * SR)
    s = int(rng.integers(0, len(w) - n + 1))
    return w[s:s + n]

def window_probs(x, hop_s=0.5, bs=64):
    """slide a 2.5 s window over x; returns window end times (s), urgent probability, window level (dBFS)"""
    hop = int(hop_s * SR)
    n = (len(x) - WIN) // hop + 1
    t = (np.arange(n) * hop + WIN) / SR
    p, r = np.empty(n, np.float32), np.empty(n, np.float32)
    for i in range(0, n, bs):
        j = min(i + bs, n)
        xs = np.stack([x[k * hop:k * hop + WIN] for k in range(i, j)]).astype(np.float32)
        lg = sess.run(None, {"logmel": logmel_batch(xs)[:, None]})[0]
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        p[i:j] = e[:, 1] / e.sum(axis=1)
        r[i:j] = 20 * np.log10(np.sqrt((xs.astype(np.float64) ** 2).mean(axis=1)) + 1e-9)
    return t, p, r
