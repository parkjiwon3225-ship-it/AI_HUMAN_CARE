# ===== cell 8d: train the v2 models one after another (GPU; safe to leave running for hours, it resumes after a disconnect) =====
SMOKE = False                  # True = about 3-minute test of everything with tiny numbers (names end with _smoke)
if device != "cuda":
    raise RuntimeError("No GPU attached. Switch the runtime to a T4 GPU, run cells 1-5, 6a, 7a, 8a, 8b, 8c again, then this cell.")
sfx = "_smoke" if SMOKE else ""
EP = 1 if SMOKE else 40
NM = (1000, 200, 200) if SMOKE else (24000, 4000, 4000)
U_RUN = {**Utr, "keep": Utr["keep"][:600]} if SMOKE else Utr

def banks(round2_from=None):
    HN, HP = mine(get_model("v1_speech"), "r1" + sfx, *NM)
    if round2_from:
        HN2, HP2 = mine(get_model(round2_from), "r2" + sfx, *NM, seed=8)
        HN, HP = np.concatenate([HN, HN2]), np.union1d(HP, HP2)
    return HN, HP

# name, epochs, seed, widths, share of partial sounds, hard negatives?, hard positives?, mine again with this model first?
RUNS = [("v2_hard", EP, 0, None, 0.4, True, True, None),
        ("v2_no_hn", EP, 0, None, 0.4, False, True, None),          # ablation: without hard negatives
        ("v2_no_part", EP, 0, None, 0.0, True, True, None),         # ablation: without partial sounds
        ("v2_wide", EP, 0, (32, 64, 128, 128), 0.4, True, True, None),
        ("v3_round2", EP, 0, None, 0.4, True, True, "v2_hard")]     # second mining round with the v2 model
if SMOKE:
    RUNS = [RUNS[0], RUNS[3]]
done = []
for name, ep, seed, ch, p_part, use_hn, use_hp, again in RUNS:
    name += sfx
    rp = "%s/results_%s.json" % (SAVE_DIR, name)
    try:
        if os.path.exists(rp):
            print("skip (already done):", name)
            done.append(json.load(open(rp)))
            continue
        t0 = time.time()
        HN, HP = banks(again + sfx if again else None)
        print("\n##### %s | %d epochs | width %s | partial %.1f | hard neg %s (%d) | hard pos %s (%d)" % (name, ep, ch or "base", p_part, use_hn, len(HN), use_hp, len(HP)))
        ds = Mix4(U_RUN, Atr, S_train, HN if use_hn else None, HP if use_hp else None, p_part=p_part)
        model = train_ckpt(name, ds, ep, seed, ch)
        done.append(finish_run(name, model))
        print("   done in %.0f s" % (time.time() - t0))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("!!!!! %s failed (%s) - continuing with the next run" % (name, e))
print("\nwindow-level summary (the streaming test in notebook D is the real judge):")
table(done)
