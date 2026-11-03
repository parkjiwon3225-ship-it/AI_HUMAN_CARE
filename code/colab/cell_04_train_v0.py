# ===== cell 4: v0 training (reference; the saved result is LAB/models/urgent_v0.pt) =====
# ---------------- run ----------------
t0 = time.time()
Utr, Atr = build_set("Training", "urgent", LIMIT), build_set("Training", "ambient", LIMIT)
Uva, Ava = build_set("Validation", "urgent", LIMIT // 2 if LIMIT else None), build_set("Validation", "ambient", LIMIT // 2 if LIMIT else None)
train_ds, val_ds = Pairs(Utr, Atr, True), Pairs(Uva, Ava, False)
print("train samples/epoch:", len(train_ds), "| val samples:", len(val_ds), "| load %.0fs" % (time.time() - t0))

model = Net().to(device)
print("parameters:", sum(p.numel() for p in model.parameters()))
loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True, worker_init_fn=worker_init)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(loader))

best_auc, best_state = -1.0, None
for ep in range(1, EPOCHS + 1):
    model.train()
    tot, cnt, t1 = 0.0, 0, time.time()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item() * len(y)
        cnt += len(y)
    p_u, p_a = predict(model, val_ds)
    a = auc(p_u, p_a)
    print("epoch %2d/%d  loss=%.4f  val AUC=%.3f  recall@0.5=%.3f  false alarm@0.5=%.3f  (%.0fs)" % (
        ep, EPOCHS, tot / cnt, a, np.mean(p_u > 0.5), np.mean(p_a > 0.5), time.time() - t1))
    if a > best_auc:
        best_auc = a
        best_state = {}
        for name, v in model.state_dict().items():
            best_state[name] = v.detach().cpu().clone()

model.load_state_dict(best_state)
final_report(model, val_ds)
if not QUICK:
    os.makedirs(SAVE_DIR, exist_ok=True)
    cfg = dict(sr=SR, crop_sec=CROP / SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, classes=["ambient", "urgent"])
    torch.save({"model": best_state, "config": cfg}, os.path.join(SAVE_DIR, "urgent_v0.pt"))
    with open(os.path.join(SAVE_DIR, "urgent_v0_config.json"), "w") as f:
        json.dump(cfg, f)
    print("saved to", SAVE_DIR)
print("total %.0f s" % (time.time() - t0))
