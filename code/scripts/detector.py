# detector.py - replay a wav file like a live microphone and run the urgent-sound detector (Raspberry Pi, no microphone needed)
# usage: python ~/urgent/detector.py ~/urgent/streams/stream_urg_first10min.wav --truth ~/urgent/streams/stream_urg_first10min_truth.json --compare ~/urgent/streams/probs_v1_speech_hop0.50.npz
import argparse, json, os, time
import numpy as np
import onnxruntime as ort
import soundfile as sf

SR, WIN = 16000, 40000                                  # 16 kHz, 2.5 s window
M = os.path.expanduser("~/urgent/models")

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--truth", help="truth json of the stream (optional)")
ap.add_argument("--compare", help="probabilities saved by Colab (optional)")
ap.add_argument("--theta", type=float, default=0.95)    # probability must be above this
ap.add_argument("--n", type=int, default=3)             # ... for this many windows in a row
ap.add_argument("--gate", type=float, default=-200.0)   # minimum window level in dBFS (-200 = off)
ap.add_argument("--cooldown", type=float, default=10.0) # seconds to stay quiet after an alarm
ap.add_argument("--hop", type=float, default=0.5)       # decision every 0.5 s
ap.add_argument("--threads", type=int, default=1)
args = ap.parse_args()

cfg = json.load(open(M + "/urgent_v1_speech_config.json"))
FB = np.load(M + "/mel_fb.npy")
N_FFT, HOP = cfg["n_fft"], cfg["hop"]
HANN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
IDX = np.arange(N_FFT)[None, :] + HOP * np.arange(1 + WIN // HOP)[:, None]
so = ort.SessionOptions()
so.intra_op_num_threads = args.threads
sess = ort.InferenceSession(M + "/urgent_v1_speech.onnx", so, providers=["CPUExecutionProvider"])

def logmel(x):
    xp = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    spec = np.abs(np.fft.rfft(xp[IDX] * HANN, axis=1)) ** 2
    m = np.log(FB @ spec.T + 1e-6)
    return ((m - m.mean()) / (m.std(ddof=1) + 1e-5)).astype(np.float32)

x, sr = sf.read(args.wav, dtype="float32")
if x.ndim > 1:
    x = x.mean(axis=1)
assert sr == SR, "the file must be 16 kHz"
hop = int(args.hop * SR)
ring = np.zeros(WIN, np.float32)                        # the last 2.5 s, like a live microphone buffer
times, probs, levels, ms, alarms = [], [], [], [], []
run, last, pos = 0, -1e9, 0
while pos + hop <= len(x):
    ring = np.concatenate([ring[hop:], x[pos:pos + hop]])   # a new 0.5 s chunk "arrives"
    pos += hop
    if pos < WIN:
        continue
    t = pos / SR
    t0 = time.perf_counter()
    lg = sess.run(None, {"logmel": logmel(ring)[None, None]})[0][0]
    p = float(np.exp(lg[1] - lg.max()) / np.exp(lg - lg.max()).sum())
    ms.append((time.perf_counter() - t0) * 1000)
    level = 20 * np.log10(float(np.sqrt(np.mean(ring.astype(np.float64) ** 2))) + 1e-9)
    times.append(t); probs.append(p); levels.append(level)
    run = run + 1 if (p > args.theta and level >= args.gate) else 0
    if run >= args.n and t - last >= args.cooldown:
        alarms.append(t)
        last, run = t, 0
        print("ALARM at %7.1f s  (p=%.3f, level %.0f dBFS)" % (t, p, level))

ms = np.array(ms)
print("\nwindows %d | audio %.0f s | per decision: mean %.1f ms, p95 %.1f ms | processing took %.1f%% of real time"
      % (len(ms), len(x) / SR, ms.mean(), np.percentile(ms, 95), 100.0 * ms.sum() / 1000.0 / (len(x) / SR)))
print("rule: p > %.3f for %d windows in a row, gate %s dBFS, cooldown %.0f s -> %d alarms"
      % (args.theta, args.n, "off" if args.gate < -100 else "%.0f" % args.gate, args.cooldown, len(alarms)))
np.savetxt(os.path.expanduser("~/urgent/probs_pi.csv"), np.c_[times, probs, levels], delimiter=",", header="time_s,prob,level_dbfs", comments="")

if args.compare:
    z = np.load(args.compare)
    k = min(len(probs), len(z["urg_p"]))
    print("\ncompare with Colab (first %d windows): max |prob difference| = %.2e, max |time difference| = %.2e s"
          % (k, np.abs(np.array(probs[:k]) - z["urg_p"][:k]).max(), np.abs(np.array(times[:k]) - z["urg_t"][:k]).max()))

if args.truth:
    ev = json.load(open(args.truth))["urgent"]
    hit, lat = [], []
    for e in ev:
        h = [a for a in alarms if e["t0"] <= a <= e["t1"] + 1.0]
        hit.append(bool(h))
        if h:
            lat.append(h[0] - e["t0"])
    fa = [a for a in alarms if not any(e["t0"] <= a <= e["t1"] + 3.0 for e in ev)]
    print("\nurgent events in this file: %d | alarm inside an event: %d (%.0f%%) | median delay %.1f s | alarms outside events: %d"
          % (len(ev), sum(hit), 100.0 * np.mean(hit) if hit else 0, np.median(lat) if lat else float("nan"), len(fa)))
