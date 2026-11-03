# code/ — 재현용 코드

> 로컬 경로는 `C:\path\to\...`처럼 자리 표시로 바꿔 두었습니다. 자기 환경에 맞게 고쳐서 사용하세요.
> 데이터와 모델 파일은 이 저장소에 없습니다 ([../docs/03_DATA.md](../docs/03_DATA.md)).

## 1. `colab/` — 코랩 셀 (번호 순서대로)

재현 순서와 노트북 구성은 [../docs/06_COLAB_RUNBOOK.md](../docs/06_COLAB_RUNBOOK.md)를 보세요.
셀 하나를 코랩 셀 하나에 붙여 넣습니다 (한 번에 약 10,300자를 넘기면 잘린 적이 있어서 셀을 나눠 두었습니다).

| 노트북 | 셀 순서 |
|---|---|
| A (CPU) | `cell_A1_connect_drive.py` → `cell_05_speech_data_v3.py` |
| B (CPU 또는 GPU) | `cell_01` → `cell_02` → `cell_03` → `cell_04b` → `cell_05` → `cell_06a` → `cell_06b` → (GPU) `cell_07a` → `07b` → `07c` → `07d` |
| C (CPU) | `cell_P0_connect_install.py` → `cell_P1_export_onnx.py` |
| D (CPU) 스트리밍 평가 | `cell_S0` → `S1` → `S2` → `S3` → `S4` → `S5` → `S6` (`cell_S*_*.py`). S0, S1 다음 S2부터 순서대로 |

- `cell_A1`, `cell_P0`에는 코랩 전용 `!pip` 명령이 있어서 일반 파이썬으로는 실행되지 않습니다.
- `cell_04_train_v0.py`는 v0 학습 셀(참고용, 이미 저장된 결과가 있으면 실행하지 않음).

## 2. `scripts/` — 노트북(PC)과 파이에서 실행

| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `label_stats.py` | PC | zip을 풀지 않고 라벨 JSON의 통계, 라벨↔wav 짝, wav 헤더 샘플레이트를 점검 |
| `verify_clips.py` | PC | wav 길이를 `area`, `duration`과 비교하고 청취용 wav를 추출 |
| `convert16k.py` | PC | **데이터 변환** (16kHz 모노, 4초 창). `python convert16k.py plan`으로 규모 확인 후 `run` |
| `check_output.py` | PC | 변환 결과 점검(음량, 반복 채움 비율, 찌그러짐)과 청취용 샘플 추출 |
| `bench.py` | 라즈베리파이 | ONNX 모델의 처리 속도 측정 (`~/urgent/models/`에 모델 3개 필요) |
| `detector.py` | 라즈베리파이 | wav를 마이크처럼 흘려 보내 감지 시험 (규칙 옵션, 코랩 결과와 비교). 사용법 [05 §7](../docs/05_RASPBERRY_PI.md) |

필요한 패키지: `numpy`, `scipy`, `soundfile` (변환·점검), `onnxruntime`, `numpy`(파이).

## 3. 주의

- 코드는 작성 후 **실제 환경에서 단계별로 검산**했지만, 코랩 셀은 순서와 런타임(CPU/GPU)을 지켜야 합니다.
- 노트북과 스크립트에 **키, 비밀번호, 개인 경로를 넣지 마세요** (이 폴더는 공개 저장소입니다).
