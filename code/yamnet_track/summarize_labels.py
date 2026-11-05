"""라벨 zip 안의 JSON을 전부 읽어 클래스별 통계를 요약합니다 (zip을 풀지 않음)

사용법:  python summarize_labels.py "라벨zip1" "라벨zip2" ...
결과는 화면에 출력되고 label_summary.txt로도 저장됩니다.
"""
import collections
import json
import statistics
import sys
import zipfile
from pathlib import Path

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def top(counter, n=8):
    return ", ".join(f"{k} {v}" for k, v in counter.most_common(n))


for path in sys.argv[1:]:
    cnt = collections.defaultdict(collections.Counter)
    notes = collections.Counter()
    durations, n_ann, errors = [], [], 0
    example = None

    with zipfile.ZipFile(path) as zf:
        infos = [i for i in zf.infolist() if i.filename.lower().endswith(".json")]
        for info in infos:
            try:
                d = json.loads(zf.read(info).decode("utf-8-sig"))
            except Exception:
                errors += 1
                continue

            audio = d.get("audio", {})
            for k in ("audioType", "acquisitionType", "acquisitionDevice",
                      "acquisitionPlace", "recodingType", "sampleRate"):
                cnt[k][audio.get(k)] += 1
            try:
                durations.append(float(audio.get("duration")))
            except (TypeError, ValueError):
                pass

            anns = d.get("annotations", [])
            n_ann.append(len(anns))
            for a in anns:
                for k in ("audioType", "gender", "generation", "soundQuality", "soundDistance"):
                    cnt["ann." + k][a.get(k)] += 1
                notes[a.get("note")] += 1
                cat = a.get("categories")
                if isinstance(cat, dict):
                    for k, v in cat.items():
                        cnt["categories." + k][str(v)[:30]] += 1
                else:
                    cnt["categories"][str(cat)[:30]] += 1
            if example is None and anns:
                example = {"area": anns[0].get("area"), "categories": anns[0].get("categories")}

    out(f"\n===== {Path(path).name} : JSON {len(infos)}개 (읽기 실패 {errors}) =====")
    if durations:
        out(f"원본 길이(초): 최소 {min(durations):.1f} / 중앙 {statistics.median(durations):.1f} / 최대 {max(durations):.1f}")
    out(f"annotation 개수: {top(collections.Counter(n_ann))}")
    for k, c in cnt.items():
        out(f"{k}: {top(c)}")
    out(f"note 상위 10개: {top(notes, 10)}")
    out(f"area/categories 예시: {json.dumps(example, ensure_ascii=False)[:600]}")

Path("label_summary.txt").write_text("\n".join(lines), encoding="utf-8")
