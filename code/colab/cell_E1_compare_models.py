# ===== cell E1 (notebook D): compare every exported model on the same test streams (CPU, about 3 min per new model) =====
import glob, shutil
MODELS_TO_TEST = None            # None = every urgent_*.onnx on Drive, or a list such as ["v1_speech", "v2_hard"]
GRID_TH = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
COOL, HIT_AFTER = 10.0, 1.0
TARGETS = {"strict": (1.0, 3.0), "relaxed": (5.0, 10.0)}     # allowed false alarms per hour: (noise only, with normal speech)
DRIVE_PI = "/content/drive/MyDrive/LAB/models/pi"
for f in glob.glob(DRIVE_PI + "/urgent_*.onnx"):
    shutil.copy(f, P + "/")
names = MODELS_TO_TEST or sorted(n for n in (os.path.basename(f)[7:-5] for f in glob.glob(P + "/urgent_*.onnx")) if "smoke" not in n)
TR = {n: json.load(open("%s/stream_%s_truth.json" % (OUT, n))) for n in ("neg", "speech", "urg")}

def get_streams(name):
    global sess
    cache = "%s/probs_%s_hop0.50.npz" % (OUT, name)
    if not os.path.exists(cache):
        sess = ort.InferenceSession(P + "/urgent_%s.onnx" % name, providers=["CPUExecutionProvider"])
        out = {}
        for n in ("neg", "speech", "urg"):
            x, _ = sf.read("%s/stream_%s.wav" % (OUT, n), dtype="float32")
            t, p, r = window_probs(x, 0.5)
            out.update({n + "_t": t, n + "_p": p, n + "_rms": r})
        np.savez(cache, **out)
    z = np.load(cache)
    return {n: (z[n + "_t"], z[n + "_p"]) for n in ("neg", "speech", "urg")}

def alarm_times(t, p, thr, n_need):
    ok, out, run, last = p > thr, [], 0, -1e9
    for i in range(len(t)):
        run = run + 1 if ok[i] else 0
        if run >= n_need and t[i] - last >= COOL:
            out.append(float(t[i]))
            last, run = float(t[i]), 0
    return out

def score(st, thr, n_need):
    fa_n = len(alarm_times(*st["neg"], thr, n_need)) / (st["neg"][0][-1] / 3600.0)
    fa_s = len(alarm_times(*st["speech"], thr, n_need)) / (st["speech"][0][-1] / 3600.0)
    al = alarm_times(*st["urg"], thr, n_need)
    ev = TR["urg"]["urgent"]
    hit = [any(e["t0"] <= a <= e["t1"] + HIT_AFTER for a in al) for e in ev]
    near = [h for e, h in zip(ev, hit) if e["gain_db"] >= -6]
    far = [h for e, h in zip(ev, hit) if e["gain_db"] < -6]
    return dict(fa_n=fa_n, fa_s=fa_s, rec=float(np.mean(hit)), near=float(np.mean(near)), far=float(np.mean(far)), thr=thr, N=n_need)

rows = {}
for name in names:
    t0 = time.time()
    st = get_streams(name)
    grid = [score(st, th, n_need) for n_need in (1, 2, 3, 4) for th in GRID_TH]
    rows[name] = {}
    for tname, (c_n, c_s) in TARGETS.items():
        ok = [g for g in grid if g["fa_n"] <= c_n and g["fa_s"] <= c_s]
        rows[name][tname] = max(ok, key=lambda g: g["rec"]) if ok else None
    print("%s: done in %.0f s" % (name, time.time() - t0))

print("\nbest event recall under a false-alarm limit (per hour: noise only / with normal speech), same 60-min streams for every model")
print("near = events played at 0 and -6 dB, far = -12 and -18 dB; rule = N windows in a row above the threshold")
print("%-14s | %-44s | %-44s" % ("model", "STRICT  (<= %g / <= %g per hour)" % TARGETS["strict"], "RELAXED (<= %g / <= %g per hour)" % TARGETS["relaxed"]))
for name in names:
    cells = []
    for tname in ("strict", "relaxed"):
        g = rows[name][tname]
        cells.append("recall %.2f (near %.2f / far %.2f)  N=%d th=%.4g" % (g["rec"], g["near"], g["far"], g["N"], g["thr"]) if g else "no rule meets the limit")
    print("%-14s | %-44s | %-44s" % (name, cells[0], cells[1]))
