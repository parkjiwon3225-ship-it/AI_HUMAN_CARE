"""zip을 풀지 않고 내용만 들여다보는 스크립트 (외부 라이브러리 불필요)

사용법:  python peek_zip.py "zip경로1" "zip경로2" ...
결과는 화면에 출력되고 peek_report.txt로도 저장됩니다.
"""
import collections
import json
import sys
import wave
import zipfile
from pathlib import Path

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def shape(o, depth=0):
    if isinstance(o, dict):
        return {k: shape(v, depth + 1) for k, v in o.items()} if depth < 3 else "{...}"
    if isinstance(o, list):
        return [shape(o[0], depth + 1), f"...총 {len(o)}개"] if o and depth < 3 else o[:2]
    return str(o)[:40]


def display_name(info):
    # utf-8 표시가 없는 zip은 한글이 깨지므로 cp949로 복원 시도
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except Exception:
        return info.filename


for path in sys.argv[1:]:
    out(f"\n===== {Path(path).name} =====")
    with zipfile.ZipFile(path) as zf:
        items = [(display_name(i), i) for i in zf.infolist() if not i.is_dir()]
        exts = collections.Counter(Path(n).suffix.lower() for n, _ in items)
        out(f"파일 {len(items)}개 | 확장자: {dict(exts)}")

        folders = collections.Counter("/".join(n.split("/")[:-1][:3]) or "(루트)" for n, _ in items)
        out("폴더별 파일 수 (최대 8개 표시):")
        for f, c in folders.most_common(8):
            out(f"  {f}: {c}")

        out("앞 6개 파일 이름:")
        for n, _ in items[:6]:
            out(f"  {n}")

        first_wav = next(((n, i) for n, i in items if n.lower().endswith(".wav")), None)
        if first_wav:
            n, i = first_wav
            try:
                with wave.open(zf.open(i), "rb") as w:
                    sr, ch, bits = w.getframerate(), w.getnchannels(), w.getsampwidth() * 8
                    out(f"WAV 샘플 [{Path(n).name}]: {sr}Hz, {ch}ch, {bits}bit, {w.getnframes() / sr:.1f}초")
            except Exception as e:
                out(f"WAV 샘플 [{Path(n).name}] 읽기 실패 ({e})")

        first_json = next(((n, i) for n, i in items if n.lower().endswith(".json")), None)
        if first_json:
            n, i = first_json
            try:
                data = json.loads(zf.read(i).decode("utf-8-sig"))
                text = json.dumps(shape(data), ensure_ascii=False, indent=1)
                out(f"JSON 샘플 [{Path(n).name}]:\n{text[:2500]}")
            except Exception as e:
                out(f"JSON 샘플 [{Path(n).name}] 읽기 실패 ({e})")

Path("peek_report.txt").write_text("\n".join(lines), encoding="utf-8")
