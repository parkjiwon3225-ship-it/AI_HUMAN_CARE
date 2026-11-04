# ===== cell 8c2: fix - define what cell 6b normally creates (MODELS, EV) and the result table, if they are missing =====
if "MODELS" not in globals():
    MODELS = {}
if "EV" not in globals():
    EV = make_eval()
    print("eval sets:", {k: (len(v) if k == "real" else v.shape) for k, v in EV.items()})
if "table" not in globals():
    def table(rs):
        cols = [("auc_amb", "AUCamb"), ("auc_sp", "AUCspeech"), ("rec_thr", "rec@1%amb"), ("rec_thr_s", "rec@1%sp"),
                ("far_sp_thr", "FARsp@thr"), ("far_spn_thr", "FARspN@thr"), ("hold_rec", "holdRec"), ("hold_rec_s", "holdRec_s")]
        uniq = {}
        for x in rs:
            uniq[x["name"]] = x
        print("\n%-20s" % "run" + "".join("%12s" % c[1] for c in cols))
        for x in uniq.values():
            print("%-20s" % x["name"] + "".join("%12s" % (("%.3f" % x[k]) if k in x else "-") for k, _ in cols))
print("ready: MODELS, EV, table")
