# ===== cell S6 (notebook D): why so few detections? clipping, trade-off curve, response inside events =====
def read_i16(name):
    x, _ = sf.read("%s/stream_%s.wav" % (OUT, name), dtype="int16")
    return x

def clip_fraction(x, t):
    """share of samples at full scale inside every 2.5 s window (window ends at times t)"""
    flag = (np.abs(x.astype(np.int32)) >= 32767).astype(np.int64)
    c = np.concatenate([[0], np.cumsum(flag)])
    e = np.round(np.asarray(t) * SR).astype(int)
    return (c[e] - c[e - WIN]) / float(WIN)

print("(a) samples at full scale (= clipped) per stream")
CF = {}
for name in ("neg", "speech", "urg"):
    x = read_i16(name)
    CF[name] = clip_fraction(x, STREAMS[name][0])
    print("   %-7s %.4f%% of samples | windows with >=0.1%% clipped samples: %.1f%%" % (name, 100.0 * np.mean(np.abs(x.astype(np.int32)) >= 32767), 100.0 * np.mean(CF[name] >= 0.001)))
    del x

print("\n(b) windows with probability > 0.99 in the streams WITHOUT urgent events")
for name in ("neg", "speech"):
    t, p, r = STREAMS[name]
    hi = p > 0.99
    sp = TRUTH[name]["speech"]
    on_speech = np.array([any(e["t0"] <= tt and tt - 2.5 <= e["t1"] for e in sp) for tt in t])      # window overlaps a speech event
    print("   %-7s %4d windows > 0.99 | clipped (>=0.1%%): %.0f%% (all windows: %.0f%%) | overlapping a speech event: %.0f%% (all windows: %.0f%%) | median level %.0f dBFS (all: %.0f)"
          % (name, hi.sum(), 100 * np.mean(CF[name][hi] >= 0.001) if hi.any() else 0, 100 * np.mean(CF[name] >= 0.001),
             100 * np.mean(on_speech[hi]) if hi.any() else 0, 100 * np.mean(on_speech), np.median(r[hi]) if hi.any() else 0, np.median(r)))

print("\n(c) trade-off without a loudness gate: false alarms per hour vs recall (near = 0/-6 dB, far = -12/-18 dB)")
print("   %-2s %-6s %-9s %-11s %-7s %-7s %-7s %-8s" % ("N", "theta", "FA/h neg", "FA/h speech", "recall", "near", "far", "lat med"))
ev = TRUTH["urg"]["urgent"]
for n_need in (1, 2, 3):
    for th in THETAS:
        x = evaluate(th, n_need, -200.0)
        near = [h for e, h in zip(ev, x["hit"]) if e["gain_db"] >= -6]
        far = [h for e, h in zip(ev, x["hit"]) if e["gain_db"] < -6]
        print("   %-2d %-6.3f %-9.1f %-11.1f %-7.2f %-7.2f %-7.2f %-8.1f" % (n_need, th, x["fa_neg"], x["fa_speech"], x["recall"], np.mean(near), np.mean(far), x["lat_med"]))

print("\n(d) highest window probability while an urgent event is playing (windows holding >=1 s of the event)")
t, p, r = STREAMS["urg"]
g = collections.defaultdict(list)
for e in ev:
    k = (t >= e["t0"] + 1.0) & (t <= e["t1"] + 1.0)
    g[int(e["gain_db"])].append(float(p[k].max()) if k.any() else 0.0)
for k in sorted(g, reverse=True):
    v = np.array(g[k])
    print("   gain %+3d dB  n=%-3d  median %.3f | share > 0.882: %.2f | > 0.99: %.2f | > 0.999: %.2f" % (k, len(v), np.median(v), np.mean(v > 0.882), np.mean(v > 0.99), np.mean(v > 0.999)))
