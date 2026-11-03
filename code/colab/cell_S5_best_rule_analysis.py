# ===== cell S5 (notebook D): look at the best rule: missed events, latency, false alarms =====
CHOSEN = None                       # e.g. (0.95, 3, -50.0) = (theta, N, gate); None = pick the candidate with the highest recall
if CHOSEN is None:
    best = max(CAND, key=lambda z: (z["recall"], -z["lat_med"])) if CAND else None
else:
    best = evaluate(*CHOSEN)
if best is None:
    print("no rule met the targets; raise TARGET_* in S4 or lower the grid thresholds")
else:
    print("rule: theta=%.3f, N=%d, gate=%s dBFS | recall %.3f | median latency %.1f s | FA/h: noise %.1f, speech %.1f"
          % (best["theta"], best["N"], "none" if best["gate"] < -100 else int(best["gate"]), best["recall"], best["lat_med"], best["fa_neg"], best["fa_speech"]))
    ev = TRUTH["urg"]["urgent"]
    hit = best["hit"]

    def by(title, keyf):
        g = collections.defaultdict(list)
        for e, h in zip(ev, hit):
            g[keyf(e)].append(h)
        print(title)
        for k in sorted(g):
            print("   %-18s n=%-4d recall=%.2f" % (k, len(g[k]), np.mean(g[k])))

    by("recall by playback gain (quieter = farther away):", lambda e: "%+d dB" % int(e["gain_db"]))
    by("recall by class:", lambda e: e["cls"])
    by("recall by event length:", lambda e: "<3.0 s" if e["t1"] - e["t0"] < 3.0 else ("3.0-3.6 s" if e["t1"] - e["t0"] < 3.6 else ">=3.6 s"))
    miss = collections.Counter(e["note"] for e, h in zip(ev, hit) if not h)
    print("most often missed phrases:", ", ".join("%s x%d" % kv for kv in miss.most_common(8)) or "none")
    sp = TRUTH["speech"]["speech"]
    a = best["alarms_speech"]
    during = [x for x in a if any(e["t0"] <= x <= e["t1"] + 2.5 for e in sp)]
    print("false alarms in the speech stream: %d total, %d during/just after a normal-speech event, %d elsewhere" % (len(a), len(during), len(a) - len(during)))
    print("alarm times (s) in the speech stream:", [round(x) for x in a][:20])
