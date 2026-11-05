"""AI Hub 위급상황 음성/음향 데이터 폴더 구조 스캔 (외부 라이브러리 불필요)

사용법:  python scan_dataset.py "D:\\데이터\\위급상황" 
결과는 화면에 출력되고 scan_report.txt로도 저장됩니다.
"""
import collections
import json
import sys
import wave
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
DEPTH = 3  # 폴더를 몇 단계까지 묶어서 셀지

count = collections.defaultdict(collections.Counter)  # 폴더 -> 확장자별 개수
size = collections.Counter()  # 폴더 -> 바이트
samples = {"wav": {}, "json": {}}  # 확장자 -> {폴더: 첫 파일}

for p in root.rglob("*"):
    if not p.is_file():
        continue
    ext = p.suffix.lower().lstrip(".")
    folder = "/".join(p.relative_to(root).parts[:-1][:DEPTH]) or "(루트)"
    count[folder][ext] += 1
    size[folder] += p.stat().st_size
    if ext in samples and folder not in samples[ext] and len(samples[ext]) < 4:
        samples[ext][folder] = p

lines = [f"== 폴더별 파일 수 (기준: {root}) =="]
for folder in sorted(count):
    kinds = ", ".join(f"{e or '확장자없음'} {n}" for e, n in count[folder].most_common())
    lines.append(f"{folder}: {kinds} | {size[folder] / 1e9:.2f} GB")

lines.append("\n== WAV 샘플 정보 ==")
for folder, p in samples["wav"].items():
    try:
        with wave.open(str(p), "rb") as w:
            sr, ch, bits = w.getframerate(), w.getnchannels(), w.getsampwidth() * 8
            lines.append(f"{p.name}: {sr}Hz, {ch}ch, {bits}bit, {w.getnframes() / sr:.1f}초")
    except Exception as e:  # 파일 형식이 특이한 경우
        lines.append(f"{p.name}: 읽기 실패 ({e})")


def shape(o, depth=0):
    if isinstance(o, dict):
        return {k: shape(v, depth + 1) for k, v in o.items()} if depth < 3 else "{...}"
    if isinstance(o, list):
        return [shape(o[0], depth + 1), f"...총 {len(o)}개"] if o and depth < 3 else o[:2]
    return str(o)[:40]


lines.append("\n== JSON 샘플 구조 ==")
for folder, p in samples["json"].items():
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        lines.append(f"[{p.name}]\n" + json.dumps(shape(data), ensure_ascii=False, indent=1))
    except Exception as e:
        lines.append(f"[{p.name}] 읽기 실패 ({e})")

report = "\n".join(lines)
print(report)
Path("scan_report.txt").write_text(report, encoding="utf-8")
