import os, re, io, sys, csv, json, math, time, random, zipfile, collections

try:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
except ImportError as e:
    print("Missing package:", e)
    print("Run this, then run again:  pip install numpy scipy soundfile")
    raise SystemExit(1)

ROOT = r"C:\path\to\Deeplearning_Data\data"
OUT = r"C:\path\to\Deeplearning_Data\processed16k"

SR = 16000
WIN = 4 * SR
CAP_PER_NOTE = 40       # urgent: max clips per (class, note)
AMBIENT_RATIO = 1.0     # ambient windows per urgent window (per split)
MAX_WIN_URGENT = 2      # max windows from one urgent clip
WIN_PER_AMBIENT = 2     # max windows from one ambient clip
URGENT_MIN_SEC, URGENT_MAX_SEC = 1.5, 12.0
AMBIENT_MIN_SEC, AMBIENT_MAX_SEC = 4.0, 120.0
LEN_TOL = 0.15          # wav length must be within +-15% of json area length
SHARD = 2000
SEED = 0
URGENT_IDS = {"7", "10", "14"}
AMBIENT_IDS = {"15", "16"}

mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
rng = random.Random(SEED)
report = []

def log(s=""):
    print(s)
    report.append(str(s))

def num(x):
    try:
        return float(x)
    except Exception:
        return None

# ---------- find zips ----------
labels = {}
sources = collections.defaultdict(list)
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        if not f.lower().endswith(".zip"):
            continue
        p = os.path.join(dp, f)
        sp = "Training" if "Training" in p else ("Validation" if "Validation" in p else "?")
        m = re.match(r"^(.*)_(\d+)\.zip$", f)
        if m:
            sources[(sp, m.group(1))].append(p)
        else:
            labels[(sp, f[:-4])] = p

zcache = {}
def zopen(p):
    if p not in zcache:
        zcache[p] = zipfile.ZipFile(p)
    return zcache[p]

# ---------- candidates ----------
def build_candidates(split):
    cands = {"urgent": [], "ambient": []}
    skipped = collections.Counter()
    for (sp, cls), lp in sorted(labels.items()):
        if sp != split:
            continue
        cid = cls.split(".")[0]
        grp = "urgent" if cid in URGENT_IDS else ("ambient" if cid in AMBIENT_IDS else None)
        if grp is None or (sp, cls) not in sources:
            continue
        winfo = {}
        for sp_path in sources[(sp, cls)]:
            for i in zopen(sp_path).infolist():
                if i.filename.lower().endswith(".wav"):
                    st = os.path.splitext(os.path.basename(i.filename))[0]
                    if st.endswith("_label"):
                        st = st[:-6]
                    winfo[st] = (sp_path, i)
        zl = zopen(lp)
        for info in zl.infolist():
            if not info.filename.lower().endswith(".json"):
                continue
            stem = os.path.splitext(os.path.basename(info.filename))[0]
            if stem not in winfo:
                skipped[cls + ": no wav"] += 1
                continue
            try:
                d = json.loads(zl.read(info).decode("utf-8-sig"))
                an = (d.get("annotations") or [{}])[0]
                ar = an.get("area") or {}
                s, e = num(ar.get("start")), num(ar.get("end"))
                if s is None or e is None or e <= s:
                    skipped[cls + ": bad area"] += 1
                    continue
                alen = e - s
                lo, hi = (URGENT_MIN_SEC, URGENT_MAX_SEC) if grp == "urgent" else (AMBIENT_MIN_SEC, AMBIENT_MAX_SEC)
                if not (lo <= alen <= hi):
                    skipped[cls + ": length out of range"] += 1
                    continue
                if grp == "urgent" and an.get("audioType") != "voice":
                    skipped[cls + ": not voice"] += 1
                    continue
                cands[grp].append({
                    "split": sp, "group": grp, "cls": cls, "stem": stem, "alen": alen,
                    "note": str(an.get("note")), "situation": str((an.get("categories") or {}).get("category_03")),
                    "gender": str(an.get("gender")), "generation": str(an.get("generation")),
                    "place": str((d.get("audio") or {}).get("acquisitionPlace")), "dist": str(an.get("soundDistance")),
                    "zip": winfo[stem][0], "info": winfo[stem][1]})
            except Exception:
                skipped[cls + ": json error"] += 1
    return cands, skipped

def est_windows(c):
    L = c["alen"]
    if c["group"] == "urgent":
        return 1 if L <= 4.0 else min(MAX_WIN_URGENT, 1 + int((L - 4.0) // 2))
    return max(0, min(WIN_PER_AMBIENT, int(L // 4)))

def select(cands, split):
    by_key = collections.defaultdict(list)
    for c in cands["urgent"]:
        by_key[(c["cls"], c["note"])].append(c)
    urgent = []
    for k in sorted(by_key):
        v = by_key[k]
        rng.shuffle(v)
        urgent.extend(v[:CAP_PER_NOTE])
    quota = int(sum(est_windows(c) for c in urgent) * AMBIENT_RATIO)
    groups = collections.defaultdict(list)
    for c in cands["ambient"]:
        groups[(c["cls"], c["note"])].append(c)
    for k in groups:
        rng.shuffle(groups[k])
    keys = sorted(groups)
    idx = {k: 0 for k in keys}
    ambient, total, active = [], 0, keys[:]
    while active and total < quota:
        nxt = []
        for k in active:
            if total >= quota:
                break
            c = groups[k][idx[k]]
            idx[k] += 1
            ambient.append(c)
            total += est_windows(c)
            if idx[k] < len(groups[k]):
                nxt.append(k)
        active = nxt
    return urgent, ambient

# ---------- audio ----------
def load_16k(b):
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    ch = x.shape[1]
    x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    return x, sr, ch

def windows_urgent(x):
    L = len(x)
    if L < WIN:
        return [(0.0, np.tile(x, -(-WIN // L))[:WIN])]
    n = min(MAX_WIN_URGENT, 1 + int((L - WIN) // (2 * SR)))
    starts = [(L - WIN) // 2] if n == 1 else [int(s) for s in np.linspace(0, L - WIN, n)]
    return [(s / SR, x[s:s + WIN]) for s in starts]

def windows_ambient(x):
    L = len(x)
    if L < WIN:
        return []
    starts = []
    for _ in range(WIN_PER_AMBIENT * 10):
        if len(starts) >= WIN_PER_AMBIENT:
            break
        s = rng.randint(0, L - WIN)
        if all(abs(s - t) >= WIN for t in starts):
            starts.append(s)
    return [(s / SR, x[s:s + WIN]) for s in starts]

class Writer:
    def __init__(self, split, group):
        self.split, self.group = split, group
        self.buf, self.rows, self.shard, self.n = [], [], 0, 0
    def fname(self):
        return "%s_%s_%03d.npy" % (self.split, self.group, self.shard)
    def add(self, w16, row):
        row["shard_file"] = self.fname()
        row["row"] = len(self.buf)
        self.buf.append(w16)
        self.rows.append(row)
        self.n += 1
        if len(self.buf) >= SHARD:
            self.flush()
    def flush(self):
        if self.buf:
            np.save(os.path.join(OUT, self.fname()), np.stack(self.buf))
            self.buf = []
            self.shard += 1
    def close(self):
        self.flush()
        if self.rows:
            cols = ["shard_file", "row", "split", "group", "cls", "stem", "note", "situation", "gender",
                    "generation", "place", "dist", "src_sr", "src_ch", "clip_sec", "win_start_sec", "rms_dbfs"]
            with open(os.path.join(OUT, "meta_%s_%s.csv" % (self.split, self.group)), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.rows)

# ---------- main ----------
t0 = time.time()
plans = {}
for split in ("Training", "Validation"):
    cands, skipped = build_candidates(split)
    urgent, ambient = select(cands, split)
    plans[split] = (urgent, ambient)
    log("")
    log("===== %s =====" % split)
    for g, sel in (("urgent", urgent), ("ambient", ambient)):
        w = sum(est_windows(c) for c in sel)
        log("%s: candidates=%d selected_clips=%d est_windows=%d est_disk=%.2f GB" % (
            g, len(cands[g]), len(sel), w, w * WIN * 2 / 1e9))
        cnt = collections.Counter(c["cls"] for c in sel)
        log("   per class: " + ", ".join("%s:%d" % kv for kv in sorted(cnt.items())))
    if skipped:
        log("skipped before selection: " + ", ".join("%s=%d" % kv for kv in sorted(skipped.items())))

if mode != "run":
    log("")
    log("plan only. To convert, run again with:  run")
else:
    os.makedirs(OUT, exist_ok=True)
    fails = collections.Counter()
    for split in ("Training", "Validation"):
        urgent, ambient = plans[split]
        for grp, sel in (("urgent", urgent), ("ambient", ambient)):
            sel = sel[:]
            rng.shuffle(sel)
            wr = Writer(split, grp)
            for n, c in enumerate(sel, 1):
                try:
                    x, sr, ch = load_16k(zopen(c["zip"]).read(c["info"]))
                except Exception:
                    fails["decode error"] += 1
                    continue
                sec = len(x) / float(SR)
                if abs(sec - c["alen"]) > LEN_TOL * c["alen"]:
                    fails["wav length != area length"] += 1
                    continue
                wins = windows_urgent(x) if grp == "urgent" else windows_ambient(x)
                for start, w in wins:
                    rms = 20 * math.log10(float(np.sqrt(np.mean(w.astype("float64") ** 2))) + 1e-9)
                    w16 = np.clip(w, -1.0, 1.0)
                    w16 = np.round(w16 * 32767).astype("int16")
                    row = dict(c)
                    row.update(src_sr=sr, src_ch=ch, clip_sec=round(sec, 2), win_start_sec=round(start, 2), rms_dbfs=round(rms, 1))
                    wr.add(w16, row)
                if n % 500 == 0:
                    print("  %s %s: %d/%d clips, %d windows, %.0fs" % (split, grp, n, len(sel), wr.n, time.time() - t0))
            wr.close()
            log("done %s %s: windows=%d" % (split, grp, wr.n))
    if fails:
        log("dropped: " + ", ".join("%s=%d" % kv for kv in sorted(fails.items())))
    log("elapsed %.0f s. output: %s" % (time.time() - t0, OUT))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "convert_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
