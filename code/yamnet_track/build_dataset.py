"""추출한 subset(16kHz mono)을 학습용 데이터셋으로 정리합니다.
- Training에서 좌우 위상이 반대였던 파일(lr_corr<0)은 제외합니다 (Validation은 수정 후 추출이라 유지).
- 정상 클립은 무작위 구간 최대 N초만 남겨 용량을 줄입니다.
- 1초보다 짧은 클립은 반복해서 1초로 맞춥니다 (YAMNet 입력 최소 길이).

사용법:  python build_dataset.py "<3_subset 폴더>" "<4_dataset 폴더>"
결과: <4_dataset>/train, <4_dataset>/val 폴더와 dataset.csv
"""
import argparse
import collections
import csv
import random
import statistics
import wave
from pathlib import Path

import numpy as np

SR = 16000
SPLITS = {"1.Training": "train", "2.Validation": "val"}

ap = argparse.ArgumentParser()
ap.add_argument("src", help="extract_subset.py 출력 폴더 (3_subset)")
ap.add_argument("dst", help="데이터셋 출력 폴더 (4_dataset)")
ap.add_argument("--max-normal-sec", type=float, default=10.0, help="정상 클립 최대 길이(초)")
ap.add_argument("--min-sec", type=float, default=1.0, help="최소 클립 길이(초)")
ap.add_argument("--drop-neg-corr-in", default="1.Training", help="lr_corr<0 파일을 제외할 split")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
rng = random.Random(args.seed)


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.ascontiguousarray(x).tobytes())


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) / 32768 + 1e-9))


def is_neg(r):
    return bool(r.get("lr_corr")) and float(r["lr_corr"]) < 0


out_rows, notes = [], []
for split, short in SPLITS.items():
    mpath = Path(args.src) / f"manifest_{split}.csv"
    if not mpath.exists():
        notes.append(f"[{split}] manifest 없음, 건너뜀")
        continue
    with open(mpath, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    dropped = 0
    if split == args.drop_neg_corr_in:
        keep = [r for r in rows if not is_neg(r)]
        dropped, rows = len(rows) - len(keep), keep
    else:  # 수정이 잘 됐는지 확인: 위상 반대 파일의 음량이 클래스 전체와 비슷해야 함
        for cls in sorted({r["class"] for r in rows}):
            rs = [r for r in rows if r["class"] == cls]
            neg = [float(r["rms_dbfs"]) for r in rs if is_neg(r)]
            if neg:
                all_med = statistics.median(float(r["rms_dbfs"]) for r in rs)
                notes.append(f"[{split}] {cls}: 위상 반대 {len(neg)}개, 음량 중앙 "
                             f"{statistics.median(neg):.1f} dBFS (클래스 전체 {all_med:.1f})")

    for r in rows:
        x = read_wav(Path(args.src) / split / r["class"] / f"{r['id']}.wav")
        if len(x) == 0:
            continue
        n_max = int(args.max_normal_sec * SR)
        if r["group"] == "normal" and len(x) > n_max:
            s = rng.randint(0, len(x) - n_max)
            x = x[s:s + n_max]
        n_min = int(args.min_sec * SR)
        if len(x) < n_min:
            x = np.tile(x, -(-n_min // len(x)))[:n_min]
        rel = Path(short) / r["group"] / f"{r['id']}.wav"
        write_wav(Path(args.dst) / rel, x)
        out_rows.append({
            "path": rel.as_posix(), "label": r["group"], "source": "aihub", "class": r["class"],
            "split": short, "category": r["category"], "duration_sec": round(len(x) / SR, 2),
            "rms_dbfs": round(rms_db(x), 1), "gender": r["gender"], "generation": r["generation"],
            "distance": r["distance"], "place": r["place"]})
    notes.append(f"[{split}] 사용 {len(rows)}개, 제외(위상 반대) {dropped}개")

if out_rows:
    with open(Path(args.dst) / "dataset.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)

agg = collections.defaultdict(lambda: [0, 0.0])
for r in out_rows:
    a = agg[(r["split"], r["label"])]
    a[0] += 1
    a[1] += r["duration_sec"]
print("\n".join(notes))
for (sp, lab), (n, sec) in sorted(agg.items()):
    print(f"{sp}/{lab}: {n}개, {sec / 3600:.2f}시간")
cls_count = collections.Counter((r["split"], r["class"]) for r in out_rows)
print("클래스별: " + ", ".join(f"{sp}/{c} {n}" for (sp, c), n in sorted(cls_count.items())))
print(f"dataset.csv 저장: {Path(args.dst) / 'dataset.csv'}")
