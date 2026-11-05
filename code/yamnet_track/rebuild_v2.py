"""낙상 위상 문제를 고친 변환으로 데이터셋을 처음부터 다시 만듭니다. (1시간 안팎 걸림)

사용법:  python rebuild_v2.py
  - extract_subset.py, build_dataset.py(수정본)와 같은 폴더(Desktop)에 두고 실행하세요.
  - 실행 중에는 터미널을 닫지 말고, PC가 절전 모드로 들어가지 않게 해 주세요.
  - 기존 3_subset, 4_dataset 폴더는 건드리지 않고 3_subset_v2, 4_dataset_v2에 새로 만듭니다.
  - 끝나면 LAB\\upload_v2 폴더에 드라이브에 올릴 파일 3개가 생깁니다.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = r"C:\path\to\LAB\2_raw_data\088.위급상황_음성-음향_데이터\01.데이터"  # 자기 경로로 수정
LAB = Path(r"C:\path\to\LAB")  # 자기 경로로 수정
SUB, DS, UP = LAB / "3_subset_v2", LAB / "4_dataset_v2", LAB / "upload_v2"
PY = sys.executable

steps = [
    ("Validation 추출", [PY, HERE / "extract_subset.py", RAW, SUB, "--split", "2.Validation",
                        "--per-class", "300", "--normal-per-class", "200"]),
    ("Training 추출", [PY, HERE / "extract_subset.py", RAW, SUB, "--split", "1.Training",
                      "--per-class", "3000", "--normal-per-class", "2000", "--class-count", "10.낙상=6000"]),
    ("데이터셋 정리", [PY, HERE / "build_dataset.py", SUB, DS, "--drop-neg-corr-in", ""]),
]

t0 = time.time()


def stamp():
    return f"[{(time.time() - t0) / 60:.0f}분 경과]"


for name, cmd in steps:
    print(f"\n=== {stamp()} {name} 시작 ===", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)

print(f"\n=== {stamp()} zip 만드는 중 ===", flush=True)
UP.mkdir(exist_ok=True)
shutil.make_archive(str(UP / "4_dataset_v2"), "zip", DS)
shutil.copy(SUB / "manifest_1.Training.csv", UP / "manifest_v2_train.csv")
shutil.copy(SUB / "manifest_2.Validation.csv", UP / "manifest_v2_val.csv")

print(f"\n=== {stamp()} 전부 완료 ===")
print(f"드라이브에 올릴 파일 ({UP}):")
for p in sorted(UP.iterdir()):
    print(f"  {p.name}  {p.stat().st_size / 1e9:.2f} GB")
