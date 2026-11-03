# ===== cell 6b: breakdown by note + baseline v0 diagnosis (CPU or GPU) =====
def chunked(f, items, make, bs=512):
    out = []
    for i in range(0, len(items), bs):
        out.append(f(np.stack([make(it) for it in items[i:i + bs]])))
    return np.concatenate(out)

def group_report(title, rows, p, thr, field, higher_is_bad, min_n=15, top=8):
    g = collections.defaultdict(list)
    for r_, v in zip(rows, p):
        g[r_[field]].append(v > thr)
    items = [(k, len(v), float(np.mean(v))) for k, v in g.items() if len(v) >= min_n]
    items.sort(key=lambda t: t[2], reverse=higher_is_bad)
    print("   %s (n>=%d):" % (title, min_n))
    for k, n, m in items[:top]:
        print("      %-28s n=%-5d %.3f" % (k[:26], n, m))

def breakdown(f, name):
    ku, ka = list(Uva["keep"]), list(Ava["keep"])
    pu = chunked(f, ku, lambda k: crop(to_f32(Uva["X"][Uva["idx"][k]])[:Uva["real"][k]], CROP, None, center=True))
    pa = chunked(f, ka, lambda k: crop(to_f32(Ava["X"][Ava["idx"][k]]), CROP, None, center=True))
    thr = float(np.quantile(pa, 0.99))
    print("\n[%s] breakdown on all validation windows, threshold(1%% FAR on ambient)=%.3f, recall=%.3f" % (name, thr, np.mean(pu > thr)))
    ru, ra = [Uva["rows"][k] for k in ku], [Ava["rows"][k] for k in ka]
    group_report("ambient sounds with the HIGHEST false alarm, by note", ra, pa, thr, "note", True)
    group_report("urgent phrases with the LOWEST recall, by note", ru, pu, thr, "note", False)
    for fld in ("place", "dist"):
        group_report("urgent recall by " + fld, ru, pu, thr, fld, False, min_n=30)
        group_report("ambient false alarm by " + fld, ra, pa, thr, fld, True, min_n=30)

EV = make_eval()
print("eval sets:", {k: (len(v) if k == "real" else v.shape) for k, v in EV.items()})
v0 = Net().to(device)
v0.load_state_dict(torch.load(SAVE_DIR + "/urgent_v0.pt", map_location=device, weights_only=False)["model"])
RES = [probe(lambda w: prob_wavs(v0, w), EV, "v0")]
breakdown(lambda w: prob_wavs(v0, w), "v0")
MODELS = {"v0": v0}
