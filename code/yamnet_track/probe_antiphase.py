"""위상이 반대인 낙상(Validation) 클립을 4가지 방식으로 변환해서 직접 들어볼 수 있게 저장합니다.
  L = 왼쪽 채널만 / R = 오른쪽 채널만 / mean = (L+R)/2 / diff = (L-R)/2
각 파일은 음량을 -20 dBFS로 맞춰 저장하므로, 소리 크기가 아니라 '무슨 소리가 들어 있는지'만 비교하면 됩니다.

사전 설치:  pip install numpy scipy
사용법:
  python probe_antiphase.py "<01.데이터 폴더>" "<manifest_2.Validation.csv 경로>" "<출력 폴더>" --n 5
"""
import argparse
import csv
import io
import math
import random
import wave
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TARGET_SR = 16000
CLS = "10.낙상"

ap = argparse.ArgumentParser()
ap.add_argument("root", help="01.데이터 폴더")
ap.add_argument("manifest", help="manifest_2.Validation.csv 경로")
ap.add_argument("out", help="출력 폴더")
ap.add_argument("--n", type=int, default=5, help="위상 반대 클립 개수")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
random.seed(args.seed)


def fix(info):  # utf-8 표시가 없는 zip의 한글 이름 복원
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except Exception:
        return info.filename


def file_id(name):
    s = Path(name).stem
    return s[:-6] if s.endswith("_label") else s


def read_stereo(raw):
    """WAV 바이트 -> (샘플 x 채널) float32 (int16 크기 기준), 샘플레이트"""
    with wave.open(io.BytesIO(raw), "rb") as w:
        ch, sr, sw = w.getnchannels(), w.getframerate(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    elif sw == 3:
        b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float32) / 256
    elif sw == 4:
        x = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 65536
    else:
        raise ValueError(f"지원하지 않는 비트 수: {sw * 8}bit")
    return x[: len(x) // ch * ch].reshape(-1, ch), sr


def dbfs(x):
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) / 32768 + 1e-9)


def save(path, x, sr):
    g = min(10 ** (-20 / 20) * 32768 / (float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-9),
            0.99 * 32768 / (float(np.abs(x).max()) + 1e-9))
    y = x * g
    if sr != TARGET_SR:
        d = math.gcd(sr, TARGET_SR)
        y = resample_poly(y, TARGET_SR // d, sr // d)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(np.clip(y, -32768, 32767).astype(np.int16).tobytes())


with open(args.manifest, encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["class"] == CLS and r["lr_corr"]]
anti = [r for r in rows if float(r["lr_corr"]) < -0.5]
same = [r for r in rows if float(r["lr_corr"]) > 0.5]
print(f"Validation 낙상: 위상 반대(상관<-0.5) {len(anti)}개, 같은 위상(상관>0.5) {len(same)}개")
picked = [("anti", r) for r in random.sample(anti, min(args.n, len(anti)))]
if same:
    picked.append(("ref", random.choice(same)))  # 비교용 정상 위상 클립 1개

index = {}
zdir = Path(args.root) / "2.Validation" / "원천데이터" / CLS
for z in sorted(zdir.glob("*.zip")):
    zf = zipfile.ZipFile(z)
    for info in zf.infolist():
        n = fix(info)
        if n.lower().endswith(".wav"):
            index[file_id(n)] = (zf, info)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
print(f"\n{'종류':<5}{'id':<16}{'상관':>6}  {'L':>7}{'R':>7}{'평균':>7}{'차이':>7}   (dBFS)")
for kind, r in picked:
    zf, info = index[r["id"]]
    x, sr = read_stereo(zf.read(info))
    if x.shape[1] != 2:
        continue
    L, R = x[:, 0], x[:, 1]
    variants = {"L": L, "R": R, "mean": (L + R) / 2, "diff": (L - R) / 2}
    for name, v in variants.items():
        save(out / f"{kind}_{r['id']}__{name}.wav", v, sr)
    print(f"{kind:<5}{r['id']:<16}{float(r['lr_corr']):>6.2f}  "
          + "".join(f"{dbfs(v):>7.1f}" for v in variants.values()))
    print(f"      세부 분류: {r['category']} | 대사: {r.get('note', '')}")
print(f"\n저장 위치: {out}  (파일 이름: 종류_id__변환방식.wav)")
