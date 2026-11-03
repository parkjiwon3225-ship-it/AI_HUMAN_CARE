# ===== cell 3: model + evaluation functions =====
# ---------------- features + model ----------------
def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    h2m = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    m2h = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    pts = m2h(np.linspace(h2m(fmin), h2m(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c))) * (2.0 / (r - l))
    return fb

class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("win", torch.hann_window(N_FFT))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank()))
    def forward(self, wav):                                   # (B, T)
        spec = torch.stft(wav, N_FFT, hop_length=HOP, win_length=N_FFT, window=self.win,
                          center=True, pad_mode="reflect", return_complex=True)
        mel = torch.matmul(self.fb, spec.abs() ** 2)          # (B, mels, frames)
        x = torch.log(mel + 1e-6)
        mu = x.mean(dim=(1, 2), keepdim=True)
        sd = x.std(dim=(1, 2), keepdim=True)
        return (x - mu) / (sd + 1e-5)

class Net(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.mel = LogMel()
        self.body = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(64, n_classes)
    def forward(self, wav):
        x = self.mel(wav)
        if self.training:                                     # light SpecAugment
            x = x.clone()
            B, M, T = x.shape
            for b in range(B):
                f = np.random.randint(0, 9)
                f0 = np.random.randint(0, M - f)
                x[b, f0:f0 + f, :] = 0
                t = np.random.randint(0, 26)
                t0 = np.random.randint(0, T - t)
                x[b, :, t0:t0 + t] = 0
        x = self.body(x.unsqueeze(1)).mean(dim=(2, 3))
        return self.fc(self.drop(x))

# ---------------- metrics ----------------
def auc(pos, neg):
    s = np.concatenate([pos, neg])
    ranks = np.empty(len(s))
    ranks[s.argsort()] = np.arange(1, len(s) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))

def recall_at_fpr(pos, neg, fpr=0.01):
    thr = float(np.quantile(neg, 1.0 - fpr))
    return float(np.mean(pos > thr)), thr

@torch.no_grad()
def predict(model, ds):
    model.eval()
    out = []
    for x, y in DataLoader(ds, batch_size=256, shuffle=False, num_workers=0):
        out.append(F.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
    p = np.concatenate(out)
    return p[:ds.n], p[ds.n:]

def tercile_labels(v):
    q1, q2 = np.quantile(v, [1 / 3.0, 2 / 3.0])
    return np.where(v <= q1, "quiet", np.where(v <= q2, "mid", "loud"))

def final_report(model, ds):
    p_u, p_a = predict(model, ds)
    U, A, n = ds.U, ds.A, ds.n
    ru = [U["rows"][U["keep"][i]] for i in range(n)]
    ra = [A["rows"][A["keep"][i % len(A["keep"])]] for i in range(n)]
    print("\n===== final validation report =====")
    print("overall: recall@0.5=%.3f  false alarm@0.5=%.3f  AUC=%.3f" % (np.mean(p_u > 0.5), np.mean(p_a > 0.5), auc(p_u, p_a)))
    r1, thr = recall_at_fpr(p_u, p_a, 0.01)
    print("recall at 1%% false alarm = %.3f (threshold %.3f)" % (r1, thr))
    def show(title, keys, p, hit):
        print(title)
        for k in sorted(set(keys)):
            m = np.array([kk == k for kk in keys])
            print("   %-22s n=%-5d %s=%.3f" % (k, m.sum(), hit, np.mean(p[m] > 0.5)))
    show("urgent recall by class:", [r["cls"] for r in ru], p_u, "recall")
    cs = np.array([float(r["clip_sec"]) for r in ru])
    show("urgent recall by clip length:", ["<3.5s" if c < 3.5 else ("3.5-4s" if c < 4.0 else ">=4s") for c in cs], p_u, "recall")
    show("urgent recall by loudness:", list(tercile_labels(np.array([float(r["rms_dbfs"]) for r in ru]))), p_u, "recall")
    show("ambient false alarm by class:", [r["cls"] for r in ra], p_a, "false_alarm")
    show("ambient false alarm by loudness:", list(tercile_labels(np.array([float(r["rms_dbfs"]) for r in ra]))), p_a, "false_alarm")
