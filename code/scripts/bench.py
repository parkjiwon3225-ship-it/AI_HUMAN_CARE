# bench.py - run on the Raspberry Pi:  python ~/urgent/bench.py
import json, os, subprocess, time
import numpy as np
import onnxruntime as ort

D = os.path.expanduser("~/urgent/models")
cfg = json.load(open(D + "/urgent_v1_speech_config.json"))
fb = np.load(D + "/mel_fb.npy")
N_FFT, HOP, CROP = cfg["n_fft"], cfg["hop"], cfg["crop_samples"]
WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
IDX = np.arange(N_FFT)[None, :] + HOP * np.arange(1 + CROP // HOP)[:, None]

def logmel_np(x):
    xp = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    spec = np.abs(np.fft.rfft(xp[IDX] * WIN, axis=1)) ** 2
    m = np.log(fb @ spec.T + 1e-6)
    return ((m - m.mean()) / (m.std(ddof=1) + 1e-5)).astype(np.float32)

def vc(arg):
    try:
        return subprocess.run(["vcgencmd", arg], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "n/a"

def stats(ms):
    a = np.array(ms)
    return "mean %.1f | median %.1f | p95 %.1f | max %.1f ms" % (a.mean(), np.median(a), np.percentile(a, 95), a.max())

rng = np.random.default_rng(0)
x = (rng.standard_normal(CROP) * 0.05).astype(np.float32)
print("python onnxruntime", ort.__version__, "| temp before:", vc("measure_temp"), "| throttled flags:", vc("get_throttled"))

t = []
for _ in range(210):
    t0 = time.perf_counter()
    f = logmel_np(x)
    t.append((time.perf_counter() - t0) * 1000)
feat_ms = t[10:]
print("features (numpy) :", stats(feat_ms))

f = logmel_np(x)[None, None]
for threads in (1, 2, 4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(D + "/urgent_v1_speech.onnx", so, providers=["CPUExecutionProvider"])
    for _ in range(10):
        sess.run(None, {"logmel": f})
    t = []
    for _ in range(200):
        t0 = time.perf_counter()
        lg = sess.run(None, {"logmel": f})[0][0]
        t.append((time.perf_counter() - t0) * 1000)
    total = np.array(feat_ms[:len(t)]) + np.array(t)
    p = float(np.exp(lg[1]) / np.exp(lg).sum())
    print("model, %d thread%s: %s" % (threads, "" if threads == 1 else "s", stats(t)))
    print("   features + model: %s | headroom for a 500 ms hop: x%.1f (p95)" % (stats(total), 500.0 / np.percentile(total, 95)))
print("sanity: urgent probability for white noise = %.3f" % p)
print("temp after:", vc("measure_temp"), "| throttled flags:", vc("get_throttled"))
