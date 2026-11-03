# ===== cell S2 (notebook D): build the test streams (noise, noise+speech, noise+speech+urgent) =====
MINUTES = {"neg": 60, "speech": 60, "urg": 60}      # lengths of the three streams
GAINS_URG = [0, -6, -12, -18]                       # urgent events are played quieter to imitate distance
GAINS_SPEECH = [0, -6, -12]
SPEECH_GAP = {"speech": 8.0, "urg": 12.0}           # mean seconds between normal-speech events
XF, FADE = 1600, 800                                # 0.1 s cross-fade between noise windows, 50 ms event fades

def build_background(minutes, rng):
    notes = collections.defaultdict(list)
    for i, r in enumerate(Arows):
        notes[r["note"]].append(i)
    keys = sorted(notes)
    w = np.array([len(notes[k]) for k in keys], float)
    w /= w.sum()
    n_samples = int(minutes * 60 * SR)
    out = np.zeros(n_samples + 2 * 64000, np.float32)
    fin = np.linspace(0, 1, XF, dtype=np.float32)
    pos = 0
    while pos < n_samples:                           # one "scene" = 8-15 windows of the same noise type
        key = keys[int(rng.choice(len(keys), p=w))]
        for _ in range(int(rng.integers(8, 16))):
            x = amb_window(int(rng.choice(notes[key]))).copy()
            x[:XF] *= fin
            x[-XF:] *= fin[::-1]
            out[pos:pos + len(x)] += x
            pos += len(x) - XF
            if pos >= n_samples:
                break
    return out[:n_samples]

def add_clip(stream, clip, t0, gain_db):
    a = int(t0 * SR)
    b = min(a + len(clip), len(stream))
    c = clip[:b - a] * (10 ** (gain_db / 20.0))
    c[:FADE] *= np.linspace(0, 1, FADE, dtype=np.float32)
    c[-FADE:] *= np.linspace(1, 0, FADE, dtype=np.float32)
    stream[a:b] += c
    return a / SR, b / SR

def build_stream(kind, minutes, seed):
    rng = np.random.default_rng(seed)
    x = build_background(minutes, rng)
    dur = len(x) / SR
    ev = {"urgent": [], "speech": []}
    if kind == "urg":
        t = float(rng.uniform(10, 25))
        while t + 6 < dur:
            clip, row = urg_clip(int(rng.integers(0, len(Ukeep))))
            g = float(rng.choice(GAINS_URG))
            a, b = add_clip(x, clip, t, g)
            ev["urgent"].append(dict(t0=a, t1=b, gain_db=g, cls=row["cls"], note=row["note"], clip_sec=float(row["clip_sec"])))
            t += float(rng.uniform(25, 40))
    if kind in ("speech", "urg"):
        t = float(rng.uniform(3, SPEECH_GAP[kind]))
        while t + 4 < dur:
            clip = speech_clip(int(rng.integers(0, len(SX))), rng)
            t1 = t + len(clip) / SR
            if not any(t < e["t1"] + 1.5 and t1 > e["t0"] - 1.5 for e in ev["urgent"]):
                g = float(rng.choice(GAINS_SPEECH))
                a, b = add_clip(x, clip, t, g)
                ev["speech"].append(dict(t0=a, t1=b, gain_db=g))
            t += float(rng.exponential(SPEECH_GAP[kind])) + 1.0
    return x, ev

def save_stream(name, x, ev):
    peak = float(np.abs(x).max())
    x16 = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write("%s/stream_%s.wav" % (OUT, name), x16, SR, subtype="PCM_16")
    json.dump(ev, open("%s/stream_%s_truth.json" % (OUT, name), "w"), ensure_ascii=False)
    print("%-7s %5.1f min | urgent events %3d | speech events %3d | peak %.2f" % (name, len(x) / SR / 60, len(ev["urgent"]), len(ev["speech"]), peak))
    return x16

os.makedirs(OUT, exist_ok=True)
for i, kind in enumerate(("neg", "speech", "urg")):
    x, ev = build_stream(kind, MINUTES[kind], seed=100 + i)
    x16 = save_stream(kind, x, ev)
    if kind == "urg":                                   # a 10-minute cut that the Raspberry Pi can replay
        cut = 600 * SR
        evc = {k: [e for e in v if e["t1"] <= 600] for k, v in ev.items()}
        sf.write(OUT + "/stream_urg_first10min.wav", x16[:cut], SR, subtype="PCM_16")
        json.dump(evc, open(OUT + "/stream_urg_first10min_truth.json", "w"), ensure_ascii=False)
        print("pi test cut: first 10 min, urgent events", len(evc["urgent"]))
    del x, x16
