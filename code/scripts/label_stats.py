import os, re, json, zipfile, random, struct, collections

ROOT = r"C:\path\to\Deeplearning_Data\data"
OUT = os.path.join(os.path.expanduser("~"), "Desktop", "label_stats.txt")
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

def num(x):
    try:
        return float(x)
    except Exception:
        return None

def pct(v, p):
    if not v:
        return 0.0
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]

def top(c, n=6):
    return ", ".join("%s:%d" % (k, v) for k, v in c.most_common(n))

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

log("===== 1. label stats (area is seconds inside each clip) =====")
label_stems = {}
for key in sorted(labels):
    sp, cls = key
    z = zipfile.ZipFile(labels[key])
    infos = [i for i in z.infolist() if i.filename.lower().endswith(".json")]
    areas, clips, nann = [], [], collections.Counter()
    cnt = collections.defaultdict(collections.Counter)
    bad = oor = perr = 0
    stems = set()
    for info in infos:
        try:
            d = json.loads(z.read(info).decode("utf-8-sig"))
        except Exception:
            perr += 1
            continue
        stems.add(os.path.splitext(os.path.basename(info.filename))[0])
        a = d.get("audio", {}) or {}
        dur = num(a.get("duration"))
        if dur is not None:
            clips.append(dur)
        cnt["device"][str(a.get("acquisitionDevice"))] += 1
        cnt["place"][str(a.get("acquisitionPlace"))] += 1
        cnt["sr_in_json"][str(a.get("sampleRate"))] += 1
        anns = d.get("annotations", []) or []
        nann[len(anns)] += 1
        for an in anns:
            ar = an.get("area") or {}
            s, e = num(ar.get("start")), num(ar.get("end"))
            if s is None or e is None or e <= s:
                bad += 1
            else:
                areas.append(e - s)
                if dur is not None and e > dur + 0.05:
                    oor += 1
            for k in ("audioType", "gender", "generation", "soundQuality", "soundDistance", "dialect"):
                cnt[k][str(an.get(k))] += 1
            cnt["note"][str(an.get("note"))] += 1
            cnt["situation"][str((an.get("categories") or {}).get("category_03"))] += 1
    z.close()
    label_stems[key] = stems
    log("")
    log("[%s] %s   json=%d  parse_err=%d  bad_area=%d  area_beyond_clip=%d" % (sp, cls, len(infos), perr, bad, oor))
    if clips:
        log("  clip sec: mean=%.1f p50=%.1f p90=%.1f max=%.1f" % (sum(clips) / len(clips), pct(clips, .5), pct(clips, .9), max(clips)))
    log("  annotations per file: %s" % top(nann, 5))
    if areas:
        log("  area sec: n=%d mean=%.2f p10=%.2f p50=%.2f p90=%.2f max=%.1f total_hours=%.1f" % (
            len(areas), sum(areas) / len(areas), pct(areas, .1), pct(areas, .5), pct(areas, .9), max(areas), sum(areas) / 3600.0))
    for k in ("audioType", "gender", "generation", "soundQuality", "soundDistance", "dialect", "device", "place", "sr_in_json"):
        log("  %s: %s" % (k, top(cnt[k])))
    log("  unique notes: %d" % len(cnt["note"]))
    log("  top notes: %s" % top(cnt["note"], 8))
    log("  top situations: %s" % top(cnt["situation"], 6))

log("")
log("===== 2. label <-> wav pairing (wav name = json name + _label.wav) =====")
rng = random.Random(0)
hdr_report = []
for key in sorted(set(labels) | set(sources)):
    sp, cls = key
    wav_stems, nwav = set(), 0
    for sp_path in sources.get(key, []):
        z = zipfile.ZipFile(sp_path)
        wavs = [i for i in z.infolist() if i.filename.lower().endswith(".wav")]
        nwav += len(wavs)
        for i in wavs:
            st = os.path.splitext(os.path.basename(i.filename))[0]
            if st.endswith("_label"):
                st = st[:-6]
            wav_stems.add(st)
        sr = collections.Counter()
        for i in rng.sample(wavs, min(300, len(wavs))):
            with z.open(i) as f:
                h = f.read(44)
            if len(h) >= 36 and h[0:4] == b"RIFF" and h[12:16] == b"fmt ":
                sr["%d Hz / %d ch / %d bit" % (struct.unpack_from("<I", h, 24)[0], struct.unpack_from("<H", h, 22)[0], struct.unpack_from("<H", h, 34)[0])] += 1
            else:
                sr["unparsed"] += 1
        z.close()
        hdr_report.append("[%s] %s (%s): %s" % (sp, cls, os.path.basename(sp_path), top(sr, 6)))
    lab = label_stems.get(key, set())
    log("[%s] %s: json=%d wav=%d matched=%d json_only=%d wav_only=%d" % (
        sp, cls, len(lab), nwav, len(lab & wav_stems), len(lab - wav_stems), len(wav_stems - lab)))
    if lab and wav_stems and not (lab & wav_stems):
        log("   no match. json example: %s | wav example: %s" % (sorted(lab)[:2], sorted(wav_stems)[:2]))

log("")
log("===== 3. real wav header sample rate (up to 300 random wav per zip) =====")
for r in hdr_report:
    log(r)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved:", OUT)
