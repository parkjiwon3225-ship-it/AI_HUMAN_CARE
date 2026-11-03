# ===== cell 6a: evaluation sets + robustness probe functions =====
import collections
EVAL_N = 1500

@torch.no_grad()
def prob_wavs(model, wavs):
    model.eval()
    out = []
    for i in range(0, len(wavs), 256):
        x = torch.from_numpy(np.ascontiguousarray(wavs[i:i + 256], dtype=np.float32)).to(device)
        out.append(F.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def rms_(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)

def add_at(amb, seg, start, snr_db):
    out = amb.copy()
    seg = seg * (rms_(amb) * 10 ** (snr_db / 20.0) / rms_(seg))
    out[start:start + len(seg)] += seg
    return np.clip(out, -1.0, 1.0).astype(np.float32)

def make_eval(n=EVAL_N, seed=3):
    r = np.random.default_rng(seed)
    iu = r.choice(len(Uva["keep"]), min(n, len(Uva["keep"])), replace=False)
    ia = r.choice(len(Ava["keep"]), min(n, len(Ava["keep"])), replace=False)
    real = []
    for i in iu:
        k = Uva["keep"][i]
        real.append(to_f32(Uva["X"][Uva["idx"][k]])[:Uva["real"][k]])
    urg = np.stack([crop(w, CROP, r, center=True) for w in real])
    amb = np.stack([crop(to_f32(Ava["X"][Ava["idx"][Ava["keep"][i]]]), CROP, r, center=True) for i in ia])
    spt = np.stack([crop(to_f32(w), CROP, r, center=True) for w in S_test])
    spn = np.stack([np.clip(mix_snr(spt[i], amb[i % len(amb)], 10.0), -1.0, 1.0) for i in range(len(spt))])
    return dict(real=real, urg=urg, amb=amb, spt=spt, spn=spn)

def probe(f, ev, name, verbose=True):
    r = np.random.default_rng(5)
    p_u, p_a, p_s, p_n = f(ev["urg"]), f(ev["amb"]), f(ev["spt"]), f(ev["spn"])
    thr = float(np.quantile(p_a, 0.99))
    thr_s = float(np.quantile(p_s, 0.99))
    res = dict(name=name, thr=thr, thr_s=thr_s, auc_amb=auc(p_u, p_a), auc_sp=auc(p_u, p_s),
               rec_thr=float(np.mean(p_u > thr)), rec_thr_s=float(np.mean(p_u > thr_s)),
               rec05=float(np.mean(p_u > 0.5)), far_amb=float(np.mean(p_a > 0.5)),
               far_sp=float(np.mean(p_s > 0.5)), far_spn=float(np.mean(p_n > 0.5)),
               far_sp_thr=float(np.mean(p_s > thr)), far_spn_thr=float(np.mean(p_n > thr)))
    real, amb = ev["real"], ev["amb"]
    P = lambda s: print("[%s] %s" % (name, s)) if verbose else None
    P("AUC urgent-vs-ambient=%.3f | urgent-vs-speech(unseen speakers)=%.3f" % (res["auc_amb"], res["auc_sp"]))
    P("@0.5: recall=%.3f | false alarm: ambient=%.3f speech=%.3f speech+noise=%.3f" % (res["rec05"], res["far_amb"], res["far_sp"], res["far_spn"]))
    P("@thr(1%% FAR on ambient)=%.3f: recall=%.3f | false alarm: speech=%.3f speech+noise=%.3f" % (thr, res["rec_thr"], res["far_sp_thr"], res["far_spn_thr"]))
    row = []
    for snr in (20, 10, 5, 0, -5):
        x = np.stack([add_at(amb[k % len(amb)], crop(w, CROP, r), 0, snr) for k, w in enumerate(real)])
        row.append("%ddB:%.2f" % (snr, np.mean(f(x) > thr)))
    P("recall vs SNR (speech over ambient) " + "  ".join(row))
    for L in (0.5, 1.0, 1.5, 2.0):
        n, row = int(L * SR), []
        for pos in ("start", "middle", "end"):
            st = {"start": 0, "middle": (CROP - n) // 2, "end": CROP - n}[pos]
            segs = [w[len(w) // 2 - n // 2: len(w) // 2 - n // 2 + n] for w in real]
            x = np.stack([add_at(amb[k % len(amb)], s, st, 10) for k, s in enumerate(segs)])
            row.append("%s:%.2f" % (pos, np.mean(f(x) > thr)))
        P("recall, %.1fs of speech at 10dB: %s" % (L, "  ".join(row)))
    row = []
    for g in (-40, -30, -20, -10, 0, 6):
        k = 10 ** (g / 20.0)
        row.append("%+d:%.2f/%.2f" % (g, np.mean(f(np.clip(ev["urg"] * k, -1, 1)) > thr), np.mean(f(np.clip(amb * k, -1, 1)) > thr)))
    P("gain dB -> recall/false alarm at thr: " + "  ".join(row))
    t = np.arange(CROP) / SR
    w20 = r.standard_normal((20, CROP))
    S_ = np.fft.rfft(w20, axis=1)
    fr = np.arange(S_.shape[1], dtype=np.float64)
    fr[0] = 1.0
    pink = np.fft.irfft(S_ / np.sqrt(fr), n=CROP, axis=1)
    clicks = np.zeros((20, CROP), np.float32)
    clicks[:, ::8000] = 0.8
    ctrl = {"silence": np.zeros((20, CROP), np.float32),
            "tiny noise": r.standard_normal((20, CROP)) * 1e-4,
            "white": r.standard_normal((20, CROP)) * 0.05,
            "pink": pink / pink.std() * 0.05,
            "1kHz tone": np.tile(0.2 * np.sin(2 * np.pi * 1000 * t), (20, 1)),
            "clicks": clicks}
    P("synthetic sounds, share above thr: " + "  ".join("%s:%.2f" % (k, np.mean(f(v.astype(np.float32)) > thr)) for k, v in ctrl.items()))
    return res
