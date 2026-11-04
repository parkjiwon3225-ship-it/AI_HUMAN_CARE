# ===== cell 8a: v2 data and model definitions - partial sounds, hard examples, wider network (nothing is trained here) =====
!pip -q install onnx onnxruntime
import copy
HARD_DIR = "/content/drive/MyDrive/LAB/hard"
CKPT_DIR = "/content/drive/MyDrive/LAB/ckpt"
PI_DIR = SAVE_DIR + "/pi"
for _d in (HARD_DIR, CKPT_DIR, PI_DIR):
    os.makedirs(_d, exist_ok=True)
SNR_V2 = (-10.0, 20.0)                       # v1 used 0 to 20 dB; the streams showed that quieter sounds matter

def place_piece(piece, base, snr_db, rng):
    """add a piece of sound (0.8-2.5 s) at a random position of a 2.5 s window of noise"""
    L = len(piece)
    pos = int(rng.integers(0, CROP - L + 1))
    p = (piece * (rms(base) * 10 ** (snr_db / 20.0) / rms(piece))).astype(np.float32)
    f = max(1, min(320, L // 4))
    p[:f] *= np.linspace(0, 1, f, dtype=np.float32)
    p[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    out = base.copy()
    out[pos:pos + L] += p
    return out

def rand_noise(A, rng):
    return crop(to_f32(A["X"][A["idx"][A["keep"][int(rng.integers(0, len(A["keep"])))]]]), CROP, rng)

def urgent_wave(U, k):
    return to_f32(U["X"][U["idx"][k]])[:U["real"][k]]

def piece_of(w, rng, lo=0.8, hi=2.5):
    return crop(w, int(rng.uniform(lo, hi) * SR), rng)

class Mix4(Dataset):
    """label 1: urgent clips (some taken from clips the old model missed, some only partly inside the window)
       label 0: noise, normal speech (some only partly inside), hard negatives mined with an earlier model"""
    def __init__(self, U, A, S, HN=None, HP=None, p_part=0.4, hp_share=0.3, hn_share=0.25):
        self.U, self.A, self.S, self.HN, self.HP = U, A, S, HN, HP
        self.p_part, self.hp_share = p_part, hp_share
        self.n = len(U["keep"])
        self.n_hn = int(hn_share * self.n) if (HN is not None and len(HN)) else 0
        self.n_sp = (self.n - self.n_hn) // 2
        self.n_amb = self.n - self.n_hn - self.n_sp
    def __len__(self):
        return 2 * self.n
    def __getitem__(self, i):
        r = RNG
        if i < self.n:
            hard = self.HP is not None and len(self.HP) and r.random() < self.hp_share
            k = int(self.HP[int(r.integers(0, len(self.HP)))]) if hard else int(self.U["keep"][i])
            w, snr = urgent_wave(self.U, k), float(r.uniform(*SNR_V2))
            if r.random() < self.p_part:
                x = place_piece(piece_of(w, r), rand_noise(self.A, r), snr, r)
            else:
                x = crop(w, CROP, r)
                if r.random() < 0.8:
                    x = mix_snr(x, rand_noise(self.A, r), snr)
            y = 1
        elif i < self.n + self.n_hn:
            x, y = to_f32(self.HN[int(r.integers(0, len(self.HN)))]), 0
        elif i < self.n + self.n_hn + self.n_amb:
            x, y = sample_ambient(self.A, int(r.integers(0, len(self.A["keep"]))), r, True), 0
        else:
            s = to_f32(self.S["X"][int(r.integers(0, len(self.S["X"])))])
            if r.random() < self.p_part:
                x = place_piece(piece_of(s, r), rand_noise(self.A, r), float(r.uniform(-5.0, 25.0)), r)
            else:
                x = sample_speech(self.S, self.A, int(r.integers(0, len(self.S["X"]))), r)
            y = 0
        return torch.from_numpy(finish(x, r, True)), y

class NetW(Net):
    """wider network (about 4x the parameters); same forward pass and SpecAugment as Net"""
    def __init__(self, ch=(32, 64, 128, 128), n_classes=2):
        super().__init__(n_classes)
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.body = nn.Sequential(block(1, ch[0]), block(ch[0], ch[1]), block(ch[1], ch[2]), block(ch[2], ch[3]))
        self.fc = nn.Linear(ch[3], n_classes)
print("v2 data and model definitions ready")
