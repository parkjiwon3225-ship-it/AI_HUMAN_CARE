# ===== cell 8c: mining - find the sounds an earlier model gets wrong (CPU is fine, about 10 min; GPU is faster) =====
def mine(model, tag, n_neg=24000, n_pos=4000, keep_neg=4000, seed=7):
    """hard negatives: noise / normal-speech windows the model calls urgent. hard positives: urgent clips it misses in noise."""
    fneg, fpos = "%s/hardneg_%s.npy" % (HARD_DIR, tag), "%s/hardpos_%s.npy" % (HARD_DIR, tag)
    if os.path.exists(fneg) and os.path.exists(fpos):
        print("mining %s: already done" % tag)
        return np.load(fneg), np.load(fpos)
    r = np.random.default_rng(seed)
    t0 = time.time()
    def make_neg():
        base, u = rand_noise(Atr, r), r.random()
        if u < 0.3:
            return base                                              # noise only
        s = to_f32(S_train["X"][int(r.integers(0, len(S_train["X"])))])
        snr = float(r.uniform(-5.0, 25.0))
        if u < 0.8:
            return place_piece(piece_of(s, r), base, snr, r)         # noise + a part of a normal sentence
        return mix_snr(crop(s, CROP, r), base, snr)                  # noise + a whole normal sentence
    keep_x, keep_p = [], []
    for c in range(0, n_neg, 2000):
        xs = np.stack([np.clip(make_neg(), -1, 1).astype(np.float32) for _ in range(min(2000, n_neg - c))])
        p = prob_wavs(model, xs)
        sel = np.where(p > 0.3)[0]
        keep_x.append((xs[sel] * 32767).astype(np.int16))
        keep_p.append(p[sel])
    X = np.concatenate(keep_x) if keep_x else np.zeros((0, CROP), np.int16)
    P = np.concatenate(keep_p) if keep_p else np.zeros(0, np.float32)
    HN = X[np.argsort(-P)[:keep_neg]]
    print("mining %s: %d of %d negative windows scored above 0.3, kept the worst %d" % (tag, len(P), n_neg, len(HN)))
    ks = []
    for c in range(0, n_pos, 2000):
        kk, xs = [], []
        for _ in range(min(2000, n_pos - c)):
            k = int(Utr["keep"][int(r.integers(0, len(Utr["keep"])))])
            w, base, snr = urgent_wave(Utr, k), rand_noise(Atr, r), float(r.uniform(-10.0, 15.0))
            x = place_piece(piece_of(w, r, 1.0, 2.5), base, snr, r) if r.random() < 0.5 else mix_snr(crop(w, CROP, r), base, snr)
            kk.append(k)
            xs.append(np.clip(x, -1, 1).astype(np.float32))
        ks.extend(np.array(kk)[prob_wavs(model, np.stack(xs)) < 0.5].tolist())
    HP = np.unique(np.array(ks, dtype=np.int64))
    print("mining %s: %d distinct urgent clips were missed at least once (of %d tests) | %.0f s" % (tag, len(HP), n_pos, time.time() - t0))
    np.save(fneg, HN)
    np.save(fpos, HP)
    return HN, HP
print("mining function ready")
