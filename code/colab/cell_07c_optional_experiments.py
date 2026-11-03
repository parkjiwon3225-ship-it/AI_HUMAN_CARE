# ===== cell 7c: OPTIONAL experiments (about 18 min, GPU needed; finished runs are skipped) =====
for nm, us, ho in [("v0_phrase_holdout", False, ("note", 0.2)),
                   ("v1_phrase_holdout", True, ("note", 0.2)),
                   ("v1_senior_holdout", True, ("gen", "senior"))]:
    RES.append(run_exp(nm, us, ho))
