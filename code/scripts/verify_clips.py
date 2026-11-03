import os, re, json, zipfile, random, struct, collections

ROOT = r"C:\path\to\Deeplearning_Data\data"
OUTDIR = os.path.join(os.path.expanduser("~"), "Desktop", "listen")
os.makedirs(OUTDIR, exist_ok=True)
rng = random.Random(1)
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else 0.0

def wav_info(b, total_len):
    if len(b) < 12 or b[0:4] != b"RIFF":
        return None
    pos, fmt, dsize = 12, None, None
    while pos + 8 <= len(b):
        cid = b[pos:pos + 4]
        size = struct.unpack_from("<I", b, pos + 4)[0]
        if cid == b"fmt " and pos + 24 <= len(b):
            tag, ch, sr, brate, align, bits = struct.unpack_from("<HHIIHH", b, pos + 8)
            fmt = (tag, ch, sr, brate, bits)
        elif cid == b"data":
            if size in (0, 0xFFFFFFFF) or pos + 8 + size > total_len + 1:
                size = total_len - (pos + 8)
            dsize = size
            break
        pos += 8 + size + (size & 1)
    if not fmt or dsize is None or fmt[3] == 0:
        return None
    return fmt[0], fmt[1], fmt[2], fmt[4], dsize / float(fmt[3])

def num(x):
    try:
        return float(x)
    except Exception:
        return None

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

WANT = {("Training", "10"): 3, ("Training", "14"): 3, ("Training", "7"): 3,
        ("Validation", "15"): 2, ("Validation", "16"): 2}

for key in sorted(labels):
    sp, cls = key
    if (sp, cls.split(".")[0]) not in WANT or key not in sources:
        continue
    n_listen = WANT[(sp, cls.split(".")[0])]
    zl = zipfile.ZipFile(labels[key])
    jinfo = {}
    for i in zl.infolist():
        if i.filename.lower().endswith(".json"):
            jinfo[os.path.splitext(os.path.basename(i.filename))[0]] = i
    wzips, winfo = {}, {}
    for sp_path in sources[key]:
        z = zipfile.ZipFile(sp_path)
        wzips[sp_path] = z
        for i in z.infolist():
            if i.filename.lower().endswith(".wav"):
                st = os.path.splitext(os.path.basename(i.filename))[0]
                if st.endswith("_label"):
                    st = st[:-6]
                winfo[st] = (sp_path, i)
    common = sorted(set(jinfo) & set(winfo))
    rng.shuffle(common)
    r_area, r_dur, secs, areas, durs, unparsed = [], [], [], [], [], 0
    rows = []
    for st in common[:200]:
        d = json.loads(zl.read(jinfo[st]).decode("utf-8-sig"))
        an = (d.get("annotations") or [{}])[0]
        ar = an.get("area") or {}
        s, e = num(ar.get("start")), num(ar.get("end"))
        dur = num((d.get("audio") or {}).get("duration"))
        sp_path, wi = winfo[st]
        with wzips[sp_path].open(wi) as f:
            head = f.read(4096)
        info = wav_info(head, wi.file_size)
        if info is None:
            unparsed += 1
            continue
        sec = info[4]
        secs.append(sec)
        if s is not None and e is not None and e > s:
            areas.append(e - s)
            r_area.append(sec / (e - s))
        if dur:
            durs.append(dur)
            r_dur.append(sec / dur)
        rows.append((st, info, s, e, dur, an, d))
    def share(v, lo=0.85, hi=1.15):
        return 100.0 * sum(1 for x in v if lo <= x <= hi) / len(v) if v else 0.0
    log("")
    log("[%s] %s  checked=%d unparsed=%d" % (sp, cls, len(secs), unparsed))
    log("  wav sec: mean=%.2f median=%.2f | json area len: mean=%.2f | json duration: mean=%.2f" % (
        sum(secs) / max(1, len(secs)), med(secs), sum(areas) / max(1, len(areas)), sum(durs) / max(1, len(durs))))
    log("  wav/area ratio: median=%.2f, within +-15%%: %.0f%%" % (med(r_area), share(r_area)))
    log("  wav/duration ratio: median=%.2f, within +-15%%: %.0f%%" % (med(r_dur), share(r_dur)))
    for st, info, s, e, dur, an, d in rows[:n_listen]:
        out = os.path.join(OUTDIR, "%s_%s_label.wav" % (sp[0], st))
        sp_path, wi = winfo[st]
        with open(out, "wb") as fo:
            fo.write(wzips[sp_path].read(wi))
        log("  listen: %s" % os.path.basename(out))
        log("     wav=%.1fs (%d Hz, %d ch, %d bit, tag %d) | area=%s-%s | json_duration=%s" % (
            info[4], info[2], info[1], info[3], info[0], s, e, dur))
        log("     note=%s | type=%s | %s/%s | place=%s | dist=%s" % (
            an.get("note"), an.get("audioType"), an.get("gender"), an.get("generation"),
            (d.get("audio") or {}).get("acquisitionPlace"), an.get("soundDistance")))

with open(os.path.join(OUTDIR, "listen_list.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nwav files + listen_list.txt saved in:", OUTDIR)
