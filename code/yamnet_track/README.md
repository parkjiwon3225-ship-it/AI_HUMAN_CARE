# code/yamnet_track/ — YAMNet 임베딩 갈래 (11/05)

설명과 결과는 [../../docs/10_YAMNET_TRACK.md](../../docs/10_YAMNET_TRACK.md)를 보세요.
`C:\path\to\...`는 자리 표시입니다. 자기 경로로 바꿔 쓰세요. 데이터와 모델 파일은 저장소에 없습니다.

## PC 스크립트 (`numpy`, `scipy` 필요, 앞의 세 개는 표준 라이브러리만)

| 파일 | 하는 일 |
|---|---|
| `scan_dataset.py` | 데이터 폴더의 파일 수·WAV 헤더·JSON 구조 스캔 |
| `peek_zip.py` | zip을 풀지 않고 파일 구성, WAV 헤더, JSON 구조 확인 |
| `summarize_labels.py` | 라벨 zip의 JSON 전체를 읽어 클래스별 통계 요약 |
| `extract_subset.py` | 클래스별로 일부만 뽑아 16kHz 모노 WAV로 저장 + manifest. 8/16/24/32bit 처리, 위상 반대 스테레오는 더 큰 채널 하나 사용 |
| `check_manifest.py` | manifest로 길이·음량·좌우 위상·세부 분류 점검 |
| `probe_antiphase.py` | 위상 반대 클립을 L / R / 평균 / 차이로 저장 (직접 들어 보기용) |
| `build_dataset.py` | 학습용 폴더 정리 (정상 클립 10초 제한, 1초 미만 반복, `--drop-neg-corr-in`) |
| `rebuild_v2.py` | 추출 → 정리 → zip을 한 번에 실행 (경로 상수를 먼저 수정) |

## 코랩 노트북 (위에서부터 순서대로)

| 노트북 | 런타임 | 하는 일 |
|---|---|---|
| `01_extract_embeddings_v2.ipynb` | GPU 권장 | zip을 풀고 YAMNet 임베딩 추출(클립마다 음량 무작위) + 베이스라인 |
| `02_error_analysis_v2.ipynb` | CPU | 위상·채널·거리·말소리·세부 분류별 오분류 분석 |
| `03_train_classifier.ipynb` | CPU | 프레임 단위 MLP, N프레임 연속 규칙, 임계값 표 |
| `03b_train_classifier.ipynb` | CPU | 정상 가중치 3배, 말소리 없는 위급 프레임 제외 |

- 초기 버전 노트북(`01`, `02`의 v1)은 위상 처리 오류가 있던 데이터를 읽으므로 포함하지 않았습니다 ([09 #17](../../docs/09_CORRECTIONS_AND_LESSONS.md)).
- 노트북은 가짜 데이터로 흐름을 시험했고, 코랩에서의 실행 결과는 [10 §4](../../docs/10_YAMNET_TRACK.md)에 있습니다.
