# ===== cell 2: settings + data functions =====
import os, csv, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA = "/content/data"
SAVE_DIR = "/content/drive/MyDrive/LAB/models"
QUICK = False             # True: 1-2 minute smoke test, False: real training
SR = 16000
CROP = int(2.5 * SR)      # model input: 2.5 s
MIN_REAL = CROP
P_MIX = 0.6               # prob. of adding ambient noise under an urgent clip
SNR_RANGE = (0.0, 20.0)
GAIN_DB = 12.0
N_FFT, HOP, N_MELS, FMIN, FMAX = 512, 160, 64, 50.0, 8000.0
EPOCHS = 2 if QUICK else 20
BATCH = 128
LR = 2e-3
SEED = 0
LIMIT = 600 if QUICK else None
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| QUICK:", QUICK)
torch.manual_seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

def worker_init(wid):
    global RNG
    RNG = np.random.default_rng((torch.initial_seed() + wid) % (2 ** 32))

# ---------------- data ----------------
def load_set(split, grp):
    with open(os.path.join(DATA, "meta_%s_%s.csv" % (split, grp)), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    names = sorted(set(r["shard_file"] for r in rows))
    arrs, off, n = [], {}, 0
    for nm in names:
        a = np.load(os.path.join(DATA, nm))
        off[nm] = n
        n += a.shape[0]
        arrs.append(a)
    X = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    idx = np.array([off[r["shard_file"]] + int(r["row"]) for r in rows])
    return X, rows, idx

def build_set(split, grp, limit=None):
    X, rows, idx = load_set(split, grp)
    d = {"X": X, "rows": rows, "idx": idx}
    if grp == "urgent":
        real = np.array([int((min(float(r["clip_sec"]), 4.0) - 0.02) * SR) for r in rows])
        keep = np.where(real >= MIN_REAL)[0]
        d["real"], d["keep"] = real, keep
        print("%s urgent: %d windows, dropped (real sound < 2.5 s): %.1f%%" % (split, len(rows), 100.0 * (1 - len(keep) / len(rows))))
    else:
        d["keep"] = np.arange(len(rows))
        print("%s ambient: %d windows" % (split, len(rows)))
    if limit:
        d["keep"] = d["keep"][:limit]
    return d

def to_f32(a):
    return a.astype(np.float32) / 32768.0

def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)

def crop(x, n, rng, center=False):
    if len(x) <= n:
        out = np.zeros(n, dtype=np.float32)
        out[:len(x)] = x
        return out
    s = (len(x) - n) // 2 if center else int(rng.integers(0, len(x) - n + 1))
    return x[s:s + n]

def mix_snr(sig, noise, snr_db):
    scale = rms(sig) / (rms(noise) * 10 ** (snr_db / 20.0))
    return sig + noise * scale

def sample_urgent(U, A, i, rng, train):
    k = U["keep"][i]
    w = to_f32(U["X"][U["idx"][k]])[:U["real"][k]]      # only the real (not repeated) part
    x = crop(w, CROP, rng, center=not train)
    if train and rng.random() < P_MIX:
        a = to_f32(A["X"][A["idx"][A["keep"][int(rng.integers(0, len(A["keep"])))]]])
        x = mix_snr(x, crop(a, CROP, rng), float(rng.uniform(*SNR_RANGE)))
    return x

def sample_ambient(A, j, rng, train):
    a = to_f32(A["X"][A["idx"][A["keep"][j]]])
    x = crop(a, CROP, rng, center=not train)
    if train and rng.random() < 0.3:
        b = to_f32(A["X"][A["idx"][A["keep"][int(rng.integers(0, len(A["keep"])))]]])
        x = mix_snr(x, crop(b, CROP, rng), float(rng.uniform(5.0, 25.0)))
    return x

def finish(x, rng, train):
    if train:
        x = x * (10 ** (float(rng.uniform(-GAIN_DB, GAIN_DB)) / 20.0))
    return np.clip(x, -1.0, 1.0).astype(np.float32)

class Pairs(Dataset):
    """first half: urgent (label 1), second half: ambient (label 0), same count"""
    def __init__(self, U, A, train):
        self.U, self.A, self.train = U, A, train
        self.n = len(U["keep"])
    def __len__(self):
        return 2 * self.n
    def __getitem__(self, i):
        if i < self.n:
            x, y = sample_urgent(self.U, self.A, i, RNG, self.train), 1
        else:
            j = int(RNG.integers(0, len(self.A["keep"]))) if self.train else (i - self.n) % len(self.A["keep"])
            x, y = sample_ambient(self.A, j, RNG, self.train), 0
        return torch.from_numpy(finish(x, RNG, self.train)), y
