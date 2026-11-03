# ===== cell S4 (notebook D): evaluate decision rules (threshold, consecutive N, loudness gate) =====
THETAS = [0.5, 0.7, 0.8, 0.882, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
NS = [1, 2, 3, 4]
GATES = [-200.0, -60.0, -50.0, -45.0, -40.0]         # window level in dBFS; -200 = no gate
COOLDOWN = 10.0                                      # seconds of silence after an alarm
TARGET_NEG, TARGET_SPEECH = 1.0, 3.0                 # allowed false alarms per hour (noise only / with normal speech)
HIT_AFTER = 1.0                                      # an alarm up to 1 s after the event end still counts

def alarms(t, p, r, theta, n_need, gate):
    ok = (p > theta) & (r >= gate)
    out, run, last = [], 0, -1e9
    for i in range(len(t)):
        run = run + 1 if ok[i] else 0
        if run >= n_need and t[i] - last >= COOLDOWN:
            out.append(float(t[i]))
            last, run = float(t[i]), 0
    return out

def hours(name):
    return (STREAMS[name][0][-1] + 0.0) / 3600.0

def evaluate(theta, n_need, gate):
    res = dict(theta=theta, N=n_need, gate=gate)
    a_neg = alarms(*STREAMS["neg"], theta, n_need, gate)
    a_sp = alarms(*STREAMS["speech"], theta, n_need, gate)
    a_ur = alarms(*STREAMS["urg"], theta, n_need, gate)
    res["fa_neg"], res["fa_neg_n"] = len(a_neg) / hours("neg"), len(a_neg)
    res["fa_speech"], res["fa_speech_n"] = len(a_sp) / hours("speech"), len(a_sp)
    ev = TRUTH["urg"]["urgent"]
    lat, hit = [], []
    for e in ev:
        h = [a for a in a_ur if e["t0"] <= a <= e["t1"] + HIT_AFTER]
        hit.append(bool(h))
        if h:
            lat.append(h[0] - e["t0"])
    fa_u = [a for a in a_ur if not any(e["t0"] <= a <= e["t1"] + 3.0 for e in TRUTH["urg"]["urgent"])]
    res["recall"] = float(np.mean(hit)) if hit else 0.0
    res["lat_med"] = float(np.median(lat)) if lat else float("nan")
    res["fa_urg_stream"] = len(fa_u) / hours("urg")
    res["hit"] = hit
    res["alarms_urg"], res["alarms_speech"] = a_ur, a_sp
    return res

GRID = []
t0 = time.time()
for g in GATES:
    for n_need in NS:
        for th in THETAS:
            GRID.append(evaluate(th, n_need, g))
print("evaluated %d rules in %.0f s" % (len(GRID), time.time() - t0))

print("\nfor each (N, gate): the LOWEST threshold that keeps false alarms <= %.1f/h (noise) and <= %.1f/h (with normal speech)" % (TARGET_NEG, TARGET_SPEECH))
print("%-3s %-8s %-7s %-8s %-9s %-8s %-9s %-7s" % ("N", "gate", "theta", "FA/h neg", "FA/h speech", "recall", "lat med s", "FA/h urg"))
CAND = []
for g in GATES:
    for n_need in NS:
        ok = [x for x in GRID if x["gate"] == g and x["N"] == n_need and x["fa_neg"] <= TARGET_NEG and x["fa_speech"] <= TARGET_SPEECH]
        if ok:
            x = min(ok, key=lambda z: z["theta"])
            CAND.append(x)
            print("%-3d %-8s %-7.3f %-8.1f %-11.1f %-8.3f %-9.1f %-7.1f" % (n_need, "none" if g < -100 else "%d" % g, x["theta"], x["fa_neg"], x["fa_speech"], x["recall"], x["lat_med"], x["fa_urg_stream"]))
        else:
            print("%-3d %-8s no threshold in the grid meets the targets" % (n_need, "none" if g < -100 else "%d" % g))
import csv as _csv
with open(OUT + "/rules_v1_speech.csv", "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["theta", "N", "gate_dbfs", "fa_per_h_noise", "fa_per_h_speech", "event_recall", "latency_median_s", "fa_per_h_urgent_stream"])
    for x in GRID:
        w.writerow([x["theta"], x["N"], x["gate"], round(x["fa_neg"], 2), round(x["fa_speech"], 2), round(x["recall"], 3), round(x["lat_med"], 2), round(x["fa_urg_stream"], 2)])
print("full grid saved to", OUT + "/rules_v1_speech.csv")
print("note: each stream is %s minutes long, so a rate below about 1 alarm/hour cannot be measured; 0 alarms means 'not seen', not 'never'" % MINUTES)
