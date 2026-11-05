"""manifest CSV로 추출 결과의 품질을 점검합니다 (외부 라이브러리 불필요)

사용법:  python check_manifest.py "manifest_1.Training.csv"
결과는 화면에 출력되고 manifest_check.txt로도 저장됩니다.
"""
import collections
import csv
import statistics
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None


def top(rs, key, n=5):
    return ", ".join(f"{k} {v}" for k, v in collections.Counter(r[key] for r in rs).most_common(n))


by_cls = collections.defaultdict(list)
for r in rows:
    by_cls[r["class"]].append(r)

group_sec = collections.Counter()
for cls, rs in by_cls.items():
    dur = [num(r, "duration_sec") for r in rs]
    rms = sorted(num(r, "rms_dbfs") for r in rs)
    group_sec[rs[0]["group"]] += sum(dur)
    stereo = [(num(r, "lr_corr"), num(r, "rms_dbfs")) for r in rs if num(r, "lr_corr") is not None]
    neg = [m for c, m in stereo if c < 0]

    out(f"\n== {cls} ({rs[0]['group']}): {len(rs)}개, 총 {sum(dur) / 3600:.2f}시간, "
        f"길이 중앙 {statistics.median(dur):.1f}초 (최소 {min(dur):.1f} / 최대 {max(dur):.1f})")
    out(f"음량 dBFS: 중앙 {statistics.median(rms):.1f}, 하위 5% {rms[int(0.05 * (len(rms) - 1))]:.1f}")
    msg = f"스테레오 {len(stereo)}개 중 좌우상관 음수 {len(neg)}개"
    if neg:
        msg += f" (그 음량 중앙 {statistics.median(neg):.1f} dBFS)"
    out(msg)
    out("세부 분류 상위 5: " + top(rs, "category"))
    out("거리: " + top(rs, "distance") + " | 장소: " + top(rs, "place"))
    quietest = sorted(rs, key=lambda r: num(r, "rms_dbfs"))[:3]
    out("가장 조용한 파일 3개: " + ", ".join(f"{r['id']}({r['rms_dbfs']}dBFS, 상관 {r['lr_corr'] or '-'})" for r in quietest))

out("\n그룹별 총 길이: " + ", ".join(f"{g} {s / 3600:.1f}시간" for g, s in group_sec.items()))
Path("manifest_check.txt").write_text("\n".join(lines), encoding="utf-8")
