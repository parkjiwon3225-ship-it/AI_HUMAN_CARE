# ===== cell 8b: v2 training with checkpoints, ONNX export, result saving (nothing is trained here) =====
def train_ckpt(name, ds, epochs, seed, ch=None):
    """like train_model, but saves a checkpoint every epoch and resumes after a disconnect"""
    torch.manual_seed(seed)
    model = (Net() if ch is None else NetW(ch)).to(device)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True, worker_init_fn=worker_init)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * len(loader))
    cp, start = "%s/%s_ckpt.pt" % (CKPT_DIR, name), 1
    if os.path.exists(cp):
        ck = torch.load(cp, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start = ck["epoch"] + 1
        print("   resuming %s from epoch %d" % (name, start))
    for ep in range(start, epochs + 1):
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
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep}, cp)
        if ep % 5 == 0 or ep == epochs or ep == start:
            print("   epoch %d/%d loss=%.4f" % (ep, epochs, tot / cnt))
    return model

def export_onnx(model, name):
    try:
        import onnxruntime as ort_
        m = copy.deepcopy(model).cpu().eval()
        class Head(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.body, self.fc = net.body, net.fc
            def forward(self, x):
                return self.fc(self.body(x).mean(dim=(2, 3)))
        h = Head(m).eval()
        path = "%s/urgent_%s.onnx" % (PI_DIR, name)
        kw = dict(input_names=["logmel"], output_names=["logits"], opset_version=17, dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}})
        dummy = torch.zeros(1, 1, 64, 1 + CROP // HOP)
        try:
            torch.onnx.export(h, dummy, path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(h, dummy, path, **kw)
        x = m.mel(torch.from_numpy((np.random.default_rng(0).standard_normal((2, CROP)) * 0.05).astype(np.float32))).unsqueeze(1)
        with torch.no_grad():
            a = h(x).numpy()
        b = ort_.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"logmel": x.numpy()})[0]
        print("   onnx saved: %s | %.0f KB | max logit difference vs torch %.1e" % (path, os.path.getsize(path) / 1024, np.abs(a - b).max()))
    except Exception as e:
        print("   WARNING: ONNX export failed (%s). The .pt model is saved; export it later." % e)

def finish_run(name, model):
    res = probe(lambda w: prob_wavs(model, w), EV, name)
    MODELS[name] = model
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, "%s/urgent_%s.pt" % (SAVE_DIR, name))
    with open("%s/results_%s.json" % (SAVE_DIR, name), "w") as fh:
        json.dump(res, fh)
    export_onnx(model, name)
    return res
print("v2 training functions ready")
