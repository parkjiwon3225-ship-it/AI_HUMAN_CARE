# ===== cell 4b: load the training / validation sets (cell 4 is skipped: v0 is already saved) =====
for nm, sp, g in (("Utr", "Training", "urgent"), ("Atr", "Training", "ambient"),
                  ("Uva", "Validation", "urgent"), ("Ava", "Validation", "ambient")):
    if nm not in globals():
        globals()[nm] = build_set(sp, g, None)
print("sets ready:", len(Utr["keep"]), len(Atr["keep"]), len(Uva["keep"]), len(Ava["keep"]))
