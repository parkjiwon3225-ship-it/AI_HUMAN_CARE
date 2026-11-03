import os, csv, collections, random
import numpy as np
import soundfile as sf

OUT = r"C:\path\to\Deeplearning_Data\processed16k"
LISTEN = r"C:\path\to\listen16k"
os.makedirs(LISTEN, exist_ok=True)
rng = random.Random(0)
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))] if v else 0.0

sets = {}
for split in ("Training", "Validation"):
    for grp in ("urgent", "ambient"):
        p = os.path.join(OUT, "meta_%s_%s.csv" % (split, grp))
        if not os.path.exists(p):
            log("missing: " + p)
            continue
        with open(p, encoding="utf-8-sig") as f:
            sets[(split, grp)] = list(csv.DictReader(f))

shards = {}
def get_shard(name):
    if name not in shards:
        shards[name] = np.load(os.path.join(OUT, name), mmap_mode="r")
    return shards[name]

for (split, grp), rows in sets.items():
    log("")
    log("== %s %s ==  windows=%d" % (split, grp, len(rows)))
    names = sorted(set(r["shard_file"] for r in rows))
    total = sum(get_shard(n).shape[0] for n in names)
    s0 = get_shard(names[0])
    log("  shards=%d  rows_in_shards=%d  shape=%s dtype=%s  meta_matches=%s" % (len(names), total, tuple(s0.shape), s0.dtype, total == len(rows)))
    rms = [float(r["rms_dbfs"]) for r in rows]
    log("  rms dBFS: p5=%.1f p50=%.1f p95=%.1f | quieter than -60 dBFS: %.1f%%" % (
        pct(rms, .05), pct(rms, .5), pct(rms, .95), 100.0 * sum(1 for x in rms if x < -60) / len(rms)))
    fmt = collections.Counter("%s Hz/%s ch" % (r["src_sr"], r["src_ch"]) for r in rows)
    log("  source format: " + ", ".join("%s %.0f%%" % (k, 100.0 * v / len(rows)) for k, v in fmt.most_common(5)))
    if grp == "urgent":
        cs = [float(r["clip_sec"]) for r in rows]
        log("  repeated-to-fill windows (clip < 4 s): %.0f%% | clip < 3 s: %.0f%%" % (
            100.0 * sum(1 for x in cs if x < 4.0) / len(cs), 100.0 * sum(1 for x in cs if x < 3.0) / len(cs)))
    cls = collections.Counter(r["cls"] for r in rows)
    log("  classes: " + ", ".join("%s:%d" % kv for kv in sorted(cls.items())))
    sample = rng.sample(rows, min(500, len(rows)))
    clip = 0
    for r in sample:
        a = get_shard(r["shard_file"])[int(r["row"])]
        if int(np.abs(a.astype("int32")).max()) >= 32000:
            clip += 1
    log("  near-clipping windows (sample of %d): %.1f%%" % (len(sample), 100.0 * clip / len(sample)))

log("")
log("== listening samples -> %s ==" % LISTEN)
def pick(key, cond, n, tag):
    rows = [r for r in sets.get(key, []) if cond(r)]
    for r in rng.sample(rows, min(n, len(rows))):
        a = np.array(get_shard(r["shard_file"])[int(r["row"])])
        name = "%s_%s_%s_%s.wav" % (tag, key[1], r["cls"].split(".")[0], r["stem"].replace(".", "_"))
        sf.write(os.path.join(LISTEN, name), a, 16000, subtype="PCM_16")
        log("  %s | note=%s | src=%s Hz/%s ch | clip=%ss | rms=%s" % (name, r["note"], r["src_sr"], r["src_ch"], r["clip_sec"], r["rms_dbfs"]))
    if not rows:
        log("  (none for %s)" % tag)

T = ("Training", "urgent")
pick(T, lambda r: r["src_sr"] == "96000", 2, "hi96k")
pick(T, lambda r: r["src_sr"] == "48000", 1, "src48k")
pick(T, lambda r: r["src_ch"] == "1", 1, "mono")
pick(T, lambda r: float(r["clip_sec"]) < 3.0, 2, "short")
pick(T, lambda r: True, 1, "typical")
pick(("Training", "ambient"), lambda r: True, 3, "amb")
with open(os.path.join(LISTEN, "check_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved to:", LISTEN)
