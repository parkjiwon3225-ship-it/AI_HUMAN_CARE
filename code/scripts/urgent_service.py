#!/usr/bin/env python3
"""urgent_service.py - urgent-sound detector: decision rules + 10 s before/after clip + event JSON (Raspberry Pi).
no microphone:  python urgent_service.py --wav FILE [--realtime] [--set theta=0.9 --set n=2]
microphone (NOT TESTED until the mic arrives):  python urgent_service.py --mic plughw:3,0 --channels 2
"""
import argparse, collections, datetime, json, os, shutil, subprocess, time
import numpy as np
import onnxruntime as ort
import soundfile as sf

SR, WIN = 16000, 40000                                   # 16 kHz, 2.5 s model window
KST = datetime.timezone(datetime.timedelta(hours=9))
DEFAULTS = dict(theta=0.95, n=3, gate_dbfs=-200.0, cooldown_s=10.0, hop_s=0.5, pre_s=10.0, post_s=10.0,
                device_id="pi-dev-01", model_name="urgent_v1", model_dir="~/urgent/models",
                out_dir="~/urgent/events", max_clips=200, min_free_mb=500, threads=1, model_file="urgent_v1_speech.onnx")

ap = argparse.ArgumentParser()
ap.add_argument("--wav", help="replay this 16 kHz wav like a live microphone")
ap.add_argument("--mic", help="ALSA device for arecord, e.g. plughw:3,0")
ap.add_argument("--channels", type=int, default=2)
ap.add_argument("--config", help="json file that overrides the defaults")
ap.add_argument("--set", action="append", default=[], help="override one value, e.g. --set theta=0.9")
ap.add_argument("--realtime", action="store_true", help="wait for real time between chunks (replay only)")
ap.add_argument("--start", help="wall-clock start of the replay, ISO 8601 (default: now)")
args = ap.parse_args()
cfg = dict(DEFAULTS)
if args.config:
    cfg.update(json.load(open(args.config)))
for kv in args.set:
    k, v = kv.split("=", 1)
    cfg[k] = json.loads(v)
M, OUT = os.path.expanduser(cfg["model_dir"]), os.path.expanduser(cfg["out_dir"])
os.makedirs(OUT, exist_ok=True)
HOP = int(cfg["hop_s"] * SR)

mc = json.load(open(M + "/urgent_v1_speech_config.json"))
FB = np.load(M + "/mel_fb.npy")
N_FFT, STEP = mc["n_fft"], mc["hop"]
HANN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
IDX = np.arange(N_FFT)[None, :] + STEP * np.arange(1 + WIN // STEP)[:, None]
so = ort.SessionOptions()
so.intra_op_num_threads = int(cfg["threads"])            # 1 thread = predictable timing next to the numpy features
sess = ort.InferenceSession(M + "/" + cfg["model_file"], so, providers=["CPUExecutionProvider"])

def prob_and_level(win):
    xp = np.pad(win, (N_FFT // 2, N_FFT // 2), mode="reflect")
    m = np.log(FB @ (np.abs(np.fft.rfft(xp[IDX] * HANN, axis=1)) ** 2).T + 1e-6)
    f = ((m - m.mean()) / (m.std(ddof=1) + 1e-5)).astype(np.float32)
    lg = sess.run(None, {"logmel": f[None, None]})[0][0]
    p = float(np.exp(lg[1] - lg.max()) / np.exp(lg - lg.max()).sum())
    return p, 20 * np.log10(float(np.sqrt(np.mean(win.astype(np.float64) ** 2))) + 1e-9)

def pipe_source(cmd, channels):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    need = HOP * channels * 2
    try:
        while True:
            b = proc.stdout.read(need)
            if len(b) < need:
                break
            a = np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
            yield a.reshape(-1, channels).mean(axis=1)       # stereo -> mono by averaging (same as training)
    finally:
        proc.terminate()

def wav_source(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == SR, "the file must be 16 kHz"
    for i in range(0, len(x) - HOP + 1, HOP):
        yield x[i:i + HOP]

def time_synced():
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], capture_output=True, text=True, timeout=3).stdout.strip()
        return {"yes": True, "no": False}.get(out)
    except Exception:
        return None

def prune():
    js = sorted(f for f in os.listdir(OUT) if f.startswith("evt_") and f.endswith(".json"))
    for f in js[:max(0, len(js) - cfg["max_clips"])]:
        for ext in (".json", ".wav"):
            try:
                os.remove(os.path.join(OUT, f[:-5] + ext))
            except OSError:
                pass

def save_event(ev, complete):
    if shutil.disk_usage(OUT).free / 1e6 < cfg["min_free_mb"]:
        print("  WARNING: free disk below %d MB, clip NOT saved" % cfg["min_free_mb"])
        return
    audio = np.concatenate([ev["pre"]] + ev["post"]) if ev["post"] else ev["pre"]
    base = ev["id"]
    wav = os.path.join(OUT, base + ".wav")
    sf.write(wav, (np.clip(audio, -1, 1) * 32767).astype(np.int16), SR, subtype="PCM_16")
    meta = {"event_id": base, "device_id": cfg["device_id"], "detected_at": ev["when"].isoformat(timespec="seconds"),
            "time_synced": time_synced(),
            "model": {"name": cfg["model_name"], "threshold": cfg["theta"], "consecutive": cfg["n"], "gate_dbfs": cfg["gate_dbfs"]},
            "score": {"max_prob": round(max(ev["probs"]), 4), "mean_prob": round(float(np.mean(ev["probs"])), 4)},
            "clip": {"file": base + ".wav", "sample_rate": SR, "channels": 1, "pre_sec": round(len(ev["pre"]) / SR, 2),
                     "post_sec": round((len(audio) - len(ev["pre"])) / SR, 2), "bytes": os.path.getsize(wav), "complete": complete},
            "speaker": None}
    json.dump(meta, open(os.path.join(OUT, base + ".json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "events.jsonl"), "a") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    prune()
    print("  saved %s  (%.1f s before + %.1f s after, complete=%s)" % (base, meta["clip"]["pre_sec"], meta["clip"]["post_sec"], complete))

src = wav_source(args.wav) if args.wav else pipe_source(["arecord", "-D", args.mic, "-f", "S16_LE", "-r", str(SR), "-c", str(args.channels), "-t", "raw", "-q"], args.channels)
start_wall = datetime.datetime.fromisoformat(args.start) if args.start else datetime.datetime.now(KST)
ring = np.zeros(int(cfg["pre_s"] * SR), np.float32)
n_seen, run, last, seq, pending = 0, 0, -1e9, 0, None
recent, ms, alarms = collections.deque(maxlen=int(cfg["n"])), [], 0
print("rule: p > %.3f for %d windows in a row, gate %s dBFS, cooldown %.0f s, clip %.0f s before + %.0f s after"
      % (cfg["theta"], cfg["n"], "off" if cfg["gate_dbfs"] < -100 else "%.0f" % cfg["gate_dbfs"], cfg["cooldown_s"], cfg["pre_s"], cfg["post_s"]))
t_begin = time.time()
for chunk in src:
    if args.realtime and args.wav:
        time.sleep(max(0.0, n_seen / SR + cfg["hop_s"] - (time.time() - t_begin)))
    ring = np.concatenate([ring[len(chunk):], chunk])
    n_seen += len(chunk)
    t = n_seen / SR
    if pending is not None:                              # still recording the 10 s after the alarm
        pending["post"].append(chunk)
        pending["got"] += len(chunk)
        if pending["got"] >= int(cfg["post_s"] * SR):
            save_event(pending, True)
            pending = None
    if n_seen < WIN:
        continue
    t0 = time.perf_counter()
    p, level = prob_and_level(ring[-WIN:])
    ms.append((time.perf_counter() - t0) * 1000)
    recent.append(p)
    run = run + 1 if (p > cfg["theta"] and level >= cfg["gate_dbfs"]) else 0
    if run >= cfg["n"] and t - last >= cfg["cooldown_s"] and pending is None:
        alarms, last, run, seq = alarms + 1, t, 0, seq + 1
        when = start_wall + datetime.timedelta(seconds=t)
        pre = ring[-min(len(ring), n_seen):].copy()
        pending = {"id": "evt_%s_%03d" % (when.strftime("%Y%m%d_%H%M%S"), seq), "when": when, "pre": pre, "post": [], "got": 0, "probs": list(recent)}
        print("ALARM at %7.1f s  p=%.3f  level %.0f dBFS" % (t, p, level))
if pending is not None:                                  # input ended before the 10 s after the alarm were recorded
    save_event(pending, False)
ms = np.array(ms) if ms else np.zeros(1)
print("\nwindows %d | audio %.0f s | alarms %d | per decision mean %.1f ms, p95 %.1f ms | events in %s" % (len(ms), n_seen / SR, alarms, ms.mean(), np.percentile(ms, 95), OUT))
