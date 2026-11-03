# ===== cell 7a: training definitions (nothing is trained in this cell) =====
def sample_speech(S, A, j, rng):
    x = crop(to_f32(S["X"][j]), CROP, rng)
    if rng.random() < 0.7:
        a = to_f32(A["X"][A["idx"][A["keep"][int(rng.integers(0, len(A["keep"])))]]])
        x = mix_snr(x, crop(a, CROP, rng), float(rng.uniform(0.0, 25.0)))
    return x

class Mix3(Dataset):
    """urgent (label 1) + ambient and optionally normal speech (label 0)"""
    def __init__(self, U, A, S):
        self.U, self.A, self.S = U, A, S
        self.n = len(U["keep"])
        self.n_sp = (self.n // 2) if S is not None else 0
        self.n_amb = self.n - self.n_sp
    def __len__(self):
        return 2 * self.n
    def __getitem__(self, i):
        if i < self.n:
            x, y = sample_urgent(self.U, self.A, i, RNG, True), 1
        elif i < self.n + self.n_amb:
            x, y = sample_ambient(self.A, int(RNG.integers(0, len(self.A["keep"]))), RNG, True), 0
        else:
            x, y = sample_speech(self.S, self.A, int(RNG.integers(0, len(self.S["X"]))), RNG), 0
        return torch.from_numpy(finish(x, RNG, True)), y

def train_model(ds, epochs, seed):
    torch.manual_seed(seed)
    model = Net().to(device)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True, worker_init_fn=worker_init)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * len(loader))
    for ep in range(1, epochs + 1):
        model.train()
        tot, cnt = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            cnt += len(y)
        if ep % 5 == 0 or ep == epochs:
            print("   epoch %d/%d loss=%.4f" % (ep, epochs, tot / cnt))
    return model

def get_model(name):
    if name not in MODELS:
        m = Net().to(device)
        ck = torch.load("%s/urgent_%s.pt" % (SAVE_DIR, name), map_location=device, weights_only=False)
        m.load_state_dict(ck["model"])
        MODELS[name] = m
    return MODELS[name]

def run_exp(name, use_speech, holdout=None, epochs=20, seed=0):
    rp = "%s/results_%s.json" % (SAVE_DIR, name)
    if os.path.exists(rp):
        print("skip (already done):", name)
        with open(rp) as fh:
            return json.load(fh)
    if device != "cuda":
        raise RuntimeError("No GPU attached. Training on CPU is too slow - run this cell later with a GPU.")
    t0 = time.time()
    print("\n##### %s | speech negatives: %s | holdout: %s" % (name, use_speech, holdout))
    rows, keep = Utr["rows"], Utr["keep"]
    r = np.random.default_rng(seed)
    held = np.zeros(len(keep), dtype=bool)
    if holdout and holdout[0] == "note":
        notes = sorted(set((rows[k]["cls"], rows[k]["note"]) for k in keep))
        pick = set(notes[i] for i in r.choice(len(notes), int(len(notes) * holdout[1]), replace=False))
        held = np.array([(rows[k]["cls"], rows[k]["note"]) in pick for k in keep])
    elif holdout and holdout[0] == "gen":
        held = np.array([rows[k]["generation"] == holdout[1] for k in keep])
    Utrain = dict(Utr)
    Utrain["keep"] = keep[~held]
    print("   train urgent clips: %d | held-out clips: %d" % (len(Utrain["keep"]), int(held.sum())))
    model = train_model(Mix3(Utrain, Atr, S_train if use_speech else None), epochs, seed)
    f = lambda w: prob_wavs(model, w)
    res = probe(f, EV, name)
    if held.any():
        hk = keep[held][:1500]
        hw = np.stack([crop(to_f32(Utr["X"][Utr["idx"][k]])[:Utr["real"][k]], CROP, r, center=True) for k in hk])
        ph = f(hw)
        res["hold_rec"] = float(np.mean(ph > res["thr"]))
        res["hold_rec_s"] = float(np.mean(ph > res["thr_s"]))
        print("[%s] recall on HELD-OUT clips (n=%d): at ambient thr=%.3f, at speech thr=%.3f" % (name, len(hk), res["hold_rec"], res["hold_rec_s"]))
    MODELS[name] = model
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, "%s/urgent_%s.pt" % (SAVE_DIR, name))
    with open(rp, "w") as fh:
        json.dump(res, fh)
    print("   done in %.0f s (model + results saved to Drive)" % (time.time() - t0))
    return res

def table(rs):
    cols = [("auc_amb", "AUCamb"), ("auc_sp", "AUCspeech"), ("rec_thr", "rec@1%amb"), ("rec_thr_s", "rec@1%sp"),
            ("far_sp_thr", "FARsp@thr"), ("far_spn_thr", "FARspN@thr"), ("hold_rec", "holdRec"), ("hold_rec_s", "holdRec_s")]
    uniq = {}
    for x in rs:
        uniq[x["name"]] = x
    print("\n%-20s" % "run" + "".join("%12s" % c[1] for c in cols))
    for x in uniq.values():
        print("%-20s" % x["name"] + "".join("%12s" % (("%.3f" % x[k]) if k in x else "-") for k, _ in cols))
