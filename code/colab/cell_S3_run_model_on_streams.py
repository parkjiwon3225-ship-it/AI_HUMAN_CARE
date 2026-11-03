# ===== cell S3 (notebook D): run the model over the streams (about 1-2 min per stream on CPU, estimate) =====
HOP_S = 0.5                                          # decision every 0.5 s (try 0.25 later)
STREAMS, TRUTH = {}, {}
for name in ("neg", "speech", "urg"):
    t0 = time.time()
    x, sr = sf.read("%s/stream_%s.wav" % (OUT, name), dtype="float32")
    assert sr == SR
    t, p, r = window_probs(x, HOP_S)
    STREAMS[name] = (t, p, r)
    TRUTH[name] = json.load(open("%s/stream_%s_truth.json" % (OUT, name)))
    print("%-7s windows %5d | prob mean %.3f p99 %.3f max %.3f | %.0f s" % (name, len(t), p.mean(), np.quantile(p, 0.99), p.max(), time.time() - t0))
    del x
np.savez(OUT + "/probs_v1_speech_hop%.2f.npz" % HOP_S, **{k + "_" + n: v[i] for k, v in STREAMS.items() for i, n in enumerate(("t", "p", "rms"))})
print("saved probabilities to", OUT)
