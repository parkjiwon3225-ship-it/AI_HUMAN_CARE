"""AI Hub 위급상황 음성/음향에서 클래스별로 일부만 골라 16kHz mono WAV로 저장합니다.
zip을 통째로 풀지 않고, 뽑은 파일만 읽어서 변환합니다.

사전 설치:  pip install numpy scipy
사용법 (처음엔 소량으로 테스트):
  python extract_subset.py "<01.데이터 폴더>" "<출력 폴더>" --split 2.Validation --per-class 20
"""
import argparse
import collections
import csv
import io
import json
import math
import random
import wave
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

GROUPS = {"10.낙상": "urgent", "14.도움요청": "urgent", "7.응급의료": "urgent",
          "15.실내": "normal", "16.실외": "normal"}
TARGET_SR = 16000

ap = argparse.ArgumentParser()
ap.add_argument("root", help="01.데이터 폴더")
ap.add_argument("out", help="출력 폴더")
ap.add_argument("--split", default="2.Validation", help="1.Training 또는 2.Validation")
ap.add_argument("--per-class", type=int, default=20, help="클래스당 뽑을 개수")
ap.add_argument("--normal-per-class", type=int, default=0,
                help="정상(실내·실외) 클래스 개수 (0이면 --per-class와 동일)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--class-count", nargs="*", default=[], help="클래스별 개수 지정. 예: 10.낙상=6000")
args = ap.parse_args()
random.seed(args.seed)
class_count = {k: int(v) for k, v in (s.split("=") for s in args.class_count)}


def fix(info):  # utf-8 표시가 없는 zip의 한글 이름 복원
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except Exception:
        return info.filename


def file_id(name):  # '10.낙상_123_label.wav' / '10.낙상_123.json' -> '10.낙상_123'
    s = Path(name).stem
    return s[:-6] if s.endswith("_label") else s


def to_mono_16k(raw):
    with wave.open(io.BytesIO(raw), "rb") as w:
        ch, sr, sw = w.getnchannels(), w.getframerate(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if sw == 1:  # 8bit (부호 없음)
        x = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128) * 256
    elif sw == 2:
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    elif sw == 3:  # 24bit: 3바이트를 직접 조립
        b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / 256
    elif sw == 4:  # 32bit 정수
        x = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 65536
    else:
        raise ValueError(f"지원하지 않는 비트 수: {sw * 8}bit")
    x = x[: len(x) // ch * ch].reshape(-1, ch)
    corr = None
    if ch == 2 and x[:, 0].std() > 0 and x[:, 1].std() > 0:
        corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])  # 음수면 좌우 위상이 반대
        if corr < 0:  # 위상이 반대면 평균(소리 상쇄)도 차이도 내용이 지워질 수 있어서, 더 큰 쪽 채널 하나만 씀
            x = x[:, [int(x[:, 1].std() > x[:, 0].std())]]
    x = x.mean(axis=1)
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g)
    return np.clip(x, -32768, 32767).astype(np.int16), ch, sr, sw, corr


rows, report = [], []
base = Path(args.root) / args.split
for cls, group in GROUPS.items():
    meta = {}
    with zipfile.ZipFile(base / "라벨링데이터_수정" / f"{cls}.zip") as lz:
        for info in lz.infolist():
            if not fix(info).lower().endswith(".json"):
                continue
            d = json.loads(lz.read(info).decode("utf-8-sig"))
            a = d["annotations"][0]
            meta[file_id(fix(info))] = {
                "category": a["categories"]["category_03"], "note": a.get("note"),
                "gender": a.get("gender"), "generation": a.get("generation"),
                "distance": a.get("soundDistance"), "place": d["audio"].get("acquisitionPlace"),
                "start": a["area"]["start"], "end": a["area"]["end"]}

    wav_index = {}
    zips = [zipfile.ZipFile(p) for p in sorted((base / "원천데이터" / cls).glob("*.zip"))]
    for zf in zips:
        for info in zf.infolist():
            n = fix(info)
            if n.lower().endswith(".wav"):
                wav_index[file_id(n)] = (zf, info)

    ids = [i for i in meta if i in wav_index]
    by_cat = collections.defaultdict(list)
    for i in ids:
        by_cat[meta[i]["category"]].append(i)
    for v in by_cat.values():
        random.shuffle(v)
    n_want = class_count.get(cls) or (
        args.normal_per_class if (group == "normal" and args.normal_per_class) else args.per_class)
    chosen = []  # 세부 분류(category_03)를 골고루 섞어서 뽑기
    while len(chosen) < n_want and any(by_cat.values()):
        for v in by_cat.values():
            if v and len(chosen) < n_want:
                chosen.append(v.pop())

    out_dir = Path(args.out) / args.split / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    ch_count, bit_count, diffs, skipped = collections.Counter(), collections.Counter(), [], []
    quiet = lowcorr = 0
    for k, i in enumerate(chosen, 1):
        if k % 500 == 0:
            print(f"  {cls}: {k}/{len(chosen)} 처리 중...", flush=True)
        zf, info = wav_index[i]
        try:
            x, ch, sr, sw, corr = to_mono_16k(zf.read(info))
        except Exception as e:  # 형식이 특이한 파일은 건너뛰고 기록
            skipped.append(f"{i}: {e}")
            continue
        bit_count[sw * 8] += 1
        with wave.open(str(out_dir / f"{i}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_SR)
            w.writeframes(x.tobytes())
        m = meta[i]
        dur = len(x) / TARGET_SR
        ch_count[ch] += 1
        diffs.append(abs(dur - (m["end"] - m["start"])))
        rms_db = 20 * math.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) / 32768 + 1e-9)
        quiet += rms_db < -50
        lowcorr += corr is not None and corr < 0.3
        rows.append({"id": i, "class": cls, "group": group, "split": args.split,
                     "duration_sec": round(dur, 2), "orig_channels": ch, "orig_sr": sr,
                     "rms_dbfs": round(rms_db, 1),
                     "lr_corr": None if corr is None else round(corr, 2), **m})

    close = sum(d < 0.15 for d in diffs)
    report.append(f"{cls}: 라벨 {len(meta)}개 / WAV 짝 {len(ids)}개 / 저장 {len(diffs)}개 | "
                  f"원본 채널 수 {dict(ch_count)} | 비트 수 {dict(bit_count)} | "
                  f"WAV 길이≈area 길이 {close}/{len(diffs)} | 건너뜀 {len(skipped)} | "
                  f"조용함(<-50dBFS) {quiet} | 좌우상관<0.3 {lowcorr}")
    report.extend(f"  건너뜀 예시 {s}" for s in skipped[:3])

if rows:
    with open(Path(args.out) / f"manifest_{args.split}.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
print("\n".join(report))
print(f"manifest 저장: {Path(args.out) / f'manifest_{args.split}.csv'}")
