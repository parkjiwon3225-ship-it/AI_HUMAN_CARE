# ===== cell 5 (v3, with progress): normal Korean speech (Zeroth-Korean, CC BY 4.0) -> 4 s windows  [notebook A and B] =====
import os, io, csv, time, shutil, tarfile, subprocess, collections, urllib.request
import numpy as np
import soundfile as sf

SP_DIR = "/content/data_speech"
SP_DRIVE = "/content/drive/MyDrive/LAB/speech16k"
TAR = "/content/zeroth_korean.tar.gz"
URLS = ["https://storage.googleapis.com/zeroth_project/zeroth_korean.tar.gz",
        "https://www.openslr.org/resources/40/zeroth_korean.tar.gz"]
CAP_UTT = 60                    # max utterances per training speaker
WIN4 = 4 * 16000
os.makedirs(SP_DIR, exist_ok=True)

def speech_windows(x, r):
    L = len(x)
    if L < WIN4:
        return []
    if L >= 2 * WIN4:
        starts = [int(r.integers(0, L // 2 - WIN4 + 1)), int(r.integers(L // 2, L - WIN4 + 1))]
    else:
        starts = [int(r.integers(0, L - WIN4 + 1))]
    return [x[s:s + WIN4] for s in starts]

def remote_size(u):
    try:
        req = urllib.request.Request(u, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        return 0

def have_size():
    return os.path.getsize(TAR) if os.path.exists(TAR) else 0

def download_with_progress(u, want):
    proc = subprocess.Popen(["wget", "-c", "-q", "-P", "/content", u])
    t0, b0, next_mark = time.time(), have_size(), 10
    while proc.poll() is None:
        time.sleep(2)
        have = have_size()
        speed = max((have - b0) / max(time.time() - t0, 1e-6) / 1e6, 0.001)
        if want:
            pct = 100.0 * have / want
            msg = "  downloading: %.2f / %.2f GB (%.1f%%) | %.1f MB/s | ETA %.1f min" % (
                have / 1e9, want / 1e9, pct, speed, max(want - have, 0) / (speed * 1e6) / 60.0)
        else:
            pct = 0.0
            msg = "  downloading: %.2f GB | %.1f MB/s" % (have / 1e9, speed)
        print("\r" + msg + "     ", end="", flush=True)
        if want and pct >= next_mark:
            print("\n" + msg)
            next_mark += 10
    print()
    return proc.returncode

def ensure_download():
    """resume the download until the file size matches the server; returns True if complete"""
    for u in URLS:
        want = remote_size(u)
        print("server:", u.split("/")[2], "| size GB:", round(want / 1e9, 2) if want else "unknown", "| have GB:", round(have_size() / 1e9, 2))
        for _ in range(4 if want else 1):
            if want and have_size() >= want:
                return True
            download_with_progress(u, want)
        if want and have_size() >= want:
            return True
    print("WARNING: could not confirm that the download is complete, have GB:", round(have_size() / 1e9, 2))
    return False

def build_speech():
    ensure_download()
    if have_size() == 0:
        raise RuntimeError("download failed")
    r = np.random.default_rng(1)
    wins = {"train": [], "test": []}
    meta = {"train": [], "test": []}
    per_spk, n_flac, complete = collections.Counter(), 0, True
    total, raw, last, next_mark = have_size(), open(TAR, "rb"), 0.0, 10
    print("reading the archive (percent = compressed bytes read):")
    try:
        with tarfile.open(fileobj=raw, mode="r|gz") as tf:
            for m in tf:
                if time.time() - last > 3:
                    last = time.time()
                    pct = 100.0 * raw.tell() / total
                    msg = "  reading: %.1f%% | flac files %d | windows train %d test %d" % (pct, n_flac, len(wins["train"]), len(wins["test"]))
                    print("\r" + msg + "     ", end="", flush=True)
                    if pct >= next_mark:
                        print("\n" + msg)
                        next_mark += 10
                if not (m.isfile() and m.name.endswith(".flac")):
                    continue
                sp = "test" if "test_data" in m.name else ("train" if "train_data" in m.name else None)
                if sp is None:
                    continue
                n_flac += 1
                utt = os.path.basename(m.name)[:-5]
                spk = utt.split("_")[0]
                if sp == "train" and per_spk[spk] >= CAP_UTT:
                    continue
                x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if sr != 16000:
                    continue
                for w in speech_windows(x, r):
                    wins[sp].append((np.clip(w, -1.0, 1.0) * 32767).astype(np.int16))
                    meta[sp].append((spk, utt))
                per_spk[spk] += 1
    except (tarfile.ReadError, EOFError) as e:
        complete = False
        print("\nWARNING: the archive ended early (%s). Keeping what was read so far." % e)
    finally:
        raw.close()
    print()
    print("windows: train", len(wins["train"]), "test", len(wins["test"]), "| speakers read:", len(per_spk))
    if len(wins["test"]) == 0 or len(wins["train"]) < 1000:
        raise RuntimeError("too little data was read; rerun this cell to resume the download")
    for sp in ("train", "test"):
        np.save("%s/speech_%s.npy" % (SP_DIR, sp), np.stack(wins[sp]))
        with open("%s/speech_%s.csv" % (SP_DIR, sp), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["speaker", "utt"])
            w.writerows(meta[sp])
    os.makedirs(SP_DRIVE, exist_ok=True)
    for fn in os.listdir(SP_DIR):
        shutil.copy(os.path.join(SP_DIR, fn), SP_DRIVE)
    if complete:
        os.remove(TAR)

if not os.path.exists(SP_DIR + "/speech_test.npy"):
    if os.path.exists(SP_DRIVE + "/speech_test.npy"):
        shutil.copytree(SP_DRIVE, SP_DIR, dirs_exist_ok=True)
    else:
        build_speech()

S_train = {"X": np.load(SP_DIR + "/speech_train.npy")}
S_test = np.load(SP_DIR + "/speech_test.npy")
with open(SP_DIR + "/speech_train.csv", encoding="utf-8") as f:
    spk_tr = set(r["speaker"] for r in csv.DictReader(f))
with open(SP_DIR + "/speech_test.csv", encoding="utf-8") as f:
    spk_te = set(r["speaker"] for r in csv.DictReader(f))
print("speech train windows:", len(S_train["X"]), "speakers:", len(spk_tr),
      "| test windows:", len(S_test), "speakers:", len(spk_te), "| overlap:", len(spk_tr & spk_te))
