# 05. 라즈베리파이: 환경, 속도 측정, SSH, 마이크 설치 계획

> 접속 정보(IP, 계정명, 와이파이 이름, 키)는 이 문서에 쓰지 않습니다. 아래의 `<...>`는 각자 값으로 바꿔 쓰는 자리 표시입니다.

## 1. 현재 장비 상태 (2026-11-02 점검)

| 항목 | 값 | 판단 |
|---|---|---|
| 기기 | Raspberry Pi 4 Model B Rev 1.5 (개발용), 최종 장비는 Raspberry Pi 5 | |
| OS / 커널 | Debian 13 (trixie) 기반 Raspberry Pi OS, 6.18.x, **aarch64(64비트)** | 모델 실행에 문제 없음 |
| 파이썬 | 3.13.5 (가상환경 사용) | |
| CPU / 메모리 | Cortex-A72 4코어 최대 1.8GHz, 메모리 3.7GiB 중 3.2GiB 여유 (스왑 2GiB) | 충분 |
| 저장소 | SD 29GB 중 약 19GB 여유 | 20초 클립 약 640KB(16kHz 모노 16bit) → 이론상 수만 개 |
| 전원·온도 | `throttled=0x0`, 아이들 약 43~47°C, 부하 후 55°C | 정상 |
| 펌웨어 | 부트로더 최신 | |
| 시간 | KST, NTP 동기화 `yes`, **RTC 없음** | 아래 참고 |
| 네트워크 | 5GHz 와이파이, 신호 약 -59dBm | 사용 가능 (설치 장소에서 재확인) |
| 녹음 장치 | 없음 (HDMI 2개, 3.5mm만) | 마이크 전이라 정상 |
| I2C / SPI | I2C 켜짐(`dtparam=i2c_arm=on`), SPI 켜짐 | 마이크 HAT 준비됨 |
| 패키지(가상환경) | numpy 2.5.4, scipy 1.18.1, soundfile 0.14.0, onnxruntime 1.31.1 | 설치 완료 |

**참고**
- RTC가 없어서 전원이 꺼지면 시계가 멈춘다. 켠 직후 인터넷에 붙기 전까지 알림에 찍는 시각이 틀릴 수 있다 → 시각이 중요하면 동기화 여부를 확인하는 코드를 넣는다.
- 5GHz 와이파이는 벽에 약하다. 실제 설치 장소의 신호를 따로 확인한다 (`iwgetid -r`, `cat /proc/net/wireless`).

## 2. 환경 구성 순서 (재현용)

```bash
# 1) 64비트인지 확인 (aarch64 가 나와야 함)
uname -m
python3 --version

# 2) 가상환경과 패키지
sudo apt update
sudo apt install -y python3-venv python3-pip libsndfile1 alsa-utils
mkdir -p ~/urgent/models
python3 -m venv ~/urgent/venv
source ~/urgent/venv/bin/activate        # 새로 접속할 때마다 필요, 줄 앞에 (venv)가 붙음
pip install --upgrade pip
pip install numpy scipy soundfile onnxruntime
```

- 모델 파일 3개(`urgent_v1_speech.onnx`, `mel_fb.npy`, `urgent_v1_speech_config.json`)는 코랩에서 만들고(`code/colab/cell_P1_export_onnx.py`), 노트북을 거쳐 파이로 복사한다 (`scp`, 아래).
- 속도 측정: `code/scripts/bench.py`를 `~/urgent/bench.py`로 두고 `python ~/urgent/bench.py`.

## 3. 속도 측정 결과 (모델 238.5KB)

| 방식 | 특징 추출 | 모델 | 합계 p95 | 0.5초 간격 여유 |
|---|---|---|---|---|
| 1스레드 | 8.2ms | 12.9ms | 21.3ms | 약 23배 |
| 2스레드 | 8.2ms | 7.5ms | 15.9ms | 약 31배 |
| 4스레드 | 8.2ms | 5.9ms | 14.4ms | 약 35배 |

- 0.5초마다 판단 시 코어 한 개의 약 4%, 0.1초마다는 약 21%. → **판단을 더 자주 하는 쪽으로 쓸 여유가 있다**(짧은 외침 대응).
- 마이크 녹음, 클립 저장과 함께 돌아야 하므로 1~2스레드가 무난. 특징 추출(numpy 8.2ms)은 필요하면 더 빠르게 고칠 수 있다.
- 흰 잡음에서 위급 확률 0.188 (코랩의 합성음 결과와 방향 일치).

## 4. SSH 사용법 (노트북에서 파이 원격 사용)

### 4-1. 준비 (한 번만)
- 파이에서 SSH 서비스가 켜져 있어야 한다: `sudo systemctl status ssh` → `active (running)`. 꺼져 있으면 `sudo systemctl enable --now ssh`.
- 노트북 `~/.ssh/config`에 접속 이름을 등록해 두면 `ssh <이름>`만으로 접속된다. (**이 파일과 키는 저장소에 올리지 않는다**)

```
Host <이름>
    HostName <파이의 IP 주소>
    User <파이 사용자>
    IdentityFile ~/.ssh/<키 파일>
    ServerAliveInterval 30
```
- 집 와이파이와 휴대폰 핫스팟처럼 **네트워크마다 IP가 다르므로 Host를 따로** 만든다.

### 4-2. 자주 쓰는 명령

| 하고 싶은 일 | 명령 (노트북 PowerShell) |
|---|---|
| 접속 | `ssh <이름>` (처음엔 `yes` 입력). 나올 때 `exit` |
| 명령만 보내고 결과 받기 | `ssh <이름> "uname -m; free -h"` |
| 노트북 → 파이 파일 복사 | `scp 파일 <이름>:~/urgent/models/` |
| 파이 → 노트북 파일 복사 | `scp <이름>:~/파일 .` |
| 파이 안전 종료 | 접속 후 `sudo shutdown -h now` (전원선을 바로 뽑지 않는다) |
| 긴 스크립트 보내기 | 터미널에 붙여 넣지 말고 파일로 만들어 `scp`로 전송 (붙여 넣으면 중간이 사라진 적이 있음) |

### 4-3. 막혔을 때

| 메시지 | 원인 / 해결 |
|---|---|
| `Connection timed out` | 응답이 전혀 없음. 파이가 꺼져 있거나 부팅 중, 파이의 IP가 바뀜, 서로 다른 네트워크, 공유기의 기기 간 통신 차단(게스트 와이파이/AP 격리), VPN. **순서**: ① 파이에서(VNC 등) `hostname -I`와 `iwgetid -r` ② 노트북에서 `ipconfig`로 IP 앞 3자리와 와이파이 이름 비교 ③ 다르면 `config`의 `HostName` 수정 ④ 같으면 `ping <파이 IP>` |
| `Connection refused` | 파이의 SSH 서비스가 꺼짐 → `sudo systemctl enable --now ssh` |
| `Permission denied` | 사용자나 비밀번호 틀림, 또는 키로만 접속하도록 설정됨 → `config`의 `IdentityFile` 확인 |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | OS를 다시 구워서 지문이 바뀜 → `ssh-keygen -R <IP>` 후 재접속 |

- 이번에 겪은 시간 초과는 파이가 막 켜진 상태였을 가능성이 크다 (SSH 서비스 시작 시각이 확인 8분 전). 원인을 확정하진 못했다(추정).
- 공유기에서 파이의 IP를 고정(DHCP 예약)하면 주소가 바뀌는 문제가 사라진다.

## 5. 마이크 설치 계획 (reSpeaker 2-Mics Pi HAT V2.0)

> Seeed 공식 위키(V2 문서, 2026-01-29 갱신)를 읽고 현재 파이 환경과 맞춰 정리했다. **아직 실행하지 않았다.** 설치하면서 한 단계씩 확인한다.

**경고**: 인터넷에 많은 **v1용 드라이버 설치법(install.sh 방식)은 쓰지 않는다.** 위키에 따르면 최신 Raspberry Pi OS에서는 v1용 설치법이 더 이상 동작하지 않고, 설치 후 데스크톱이 깨지거나 `aplay`/`arecord`에서 장치가 안 보일 수 있다. **V2 문서만** 따른다. (하드웨어 버전은 Seeed의 "하드웨어 버전 구별" 문서로 도착 후 확인.)

V2 방식은 Seeed의 `seeed-linux-dtoverlays` 저장소에서 `respeaker-2mic-v2_0` 오버레이를 컴파일해 `/boot/firmware/overlays`에 복사하고, `config.txt`에 `dtoverlay=respeaker-2mic-v2_0` 한 줄을 추가한 뒤 재부팅하는 것이다. (Pi Zero W 전용 드라이버 컴파일 단계는 Pi 4와 무관.)

### 설치 순서
1. 파이를 끄고 HAT 장착(핀 정렬 확인) → 켜기.
2. 백업: `sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.bak`
3. 위키의 오버레이 컴파일 명령 실행. `dtc: command not found`면 `sudo apt install -y device-tree-compiler`.
4. 재부팅 후 `arecord -l`에 `seeed2micvoicec` 같은 카드가 있는지 확인 (이름·번호는 기기마다 다름).
5. 5초 시험 녹음: `arecord -D plughw:<카드번호>,0 -f S16_LE -r 16000 -c 2 -d 5 test.wav` → `scp`로 노트북에 가져와 청취.
6. `alsamixer`(F6으로 카드 선택)에서 녹음 볼륨 조정.

### 참고
- 마이크는 2채널(스테레오). **모델은 모노로 학습했으니 채널을 평균 내서 쓴다** (학습 때 스테레오 데이터도 채널 평균).
- 위키의 예제 스크립트는 `RPi.GPIO`를 쓰는데 우리는 쓰지 않는다 (ALSA로 직접 녹음).
- 위키의 예시 출력은 Pi 5 기준으로 보이는 장치 이름이라 Pi 4에서는 다를 수 있다.
- 현재 `config.txt`: 오디오 켜짐, I2C 켜짐, SPI 켜짐 (HAT 준비됨). 카드 번호는 현재 0·1이 HDMI, 2가 3.5mm → HAT은 보통 그 다음 번호.

### 사전 확인 명령 (읽기 전용, 마이크 전에 실행 가능)
```bash
modinfo snd_soc_tlv320aic3x_i2c 2>&1 | head -3     # 코덱 드라이버가 커널에 있는지
ls /boot/firmware/overlays | grep -i -E "respeaker|seeed|aic3"
git --version; make --version | head -1; dtc --version
```

## 6. 파이 소프트웨어 설계 메모 (구현 예정)

| 요소 | 설계 |
|---|---|
| 입력 | 16kHz 모노 입력(스테레오는 평균), 최근 2.5초를 0.5초마다 판단 |
| 판단 | 확률 > 기준값이 **연속 N회**일 때만 경보, 소리가 너무 작으면 건너뜀(게이트), 알림 후 쉬는 시간 |
| 클립 | 최근 10초 링 버퍼 유지 → 경보 시 이후 10초를 더 받아 **20초 wav(약 640KB) + JSON** 저장 |
| 저장 관리 | 오래된 클립 삭제 규칙, 디스크 여유 점검 |
| 전달 | 앱 쪽 방식 확정 필요 ([08_TEAM_SHARING](08_TEAM_SHARING.md)) |
| 안정성 | 부팅 시 자동 시작, 전송 실패 시 재시도, 시각 동기화 확인 |

## 7. 가짜 실시간 감지기 (`code/scripts/detector.py`)

wav 파일을 마이크처럼 0.5초씩 흘려 보내고, 최근 2.5초를 판단하는 프로그램이다. 입력을 마이크로 바꾸면 그대로 실제 감지기의 뼈대가 된다.

```bash
source ~/urgent/venv/bin/activate
python ~/urgent/detector.py ~/urgent/streams/stream_urg_first10min.wav \
  --truth ~/urgent/streams/stream_urg_first10min_truth.json \
  --compare ~/urgent/streams/probs_v1_speech_hop0.50.npz
```
- 규칙은 `--theta`(기준값), `--n`(연속 횟수), `--gate`(소리 크기 하한 dBFS), `--cooldown`(쉬는 시간)으로 바꾼다.
- 스트림 파일(`stream_urg_first10min.*`, `probs_*.npz`)은 코랩 노트북 D가 드라이브 `LAB/streams`에 만든다.

### 실행 결과 (파이4, 1스레드, 2026-11-03)
| 항목 | 값 |
|---|---|
| 판단 한 번 | 평균 21.5ms, p95 22.5ms, 실시간의 4.3% |
| 코랩과의 확률 차이 | 최대 1.84e-03 (시각 차이 0) → 원인 미확인, [07](07_ROADMAP.md) A1b |
| 임시 규칙(기준값 0.95, 연속 3회, 쉼 10초) | 경보 8번, 위급 문장 18개 중 5개 감지(28%), 지연 중앙값 2.4초, 문장 밖 경보 2번 |

## 8. 감지 서비스 (`code/scripts/urgent_service.py`)

`detector.py`에 **판단 규칙, 앞뒤 10초 클립 저장, 앱용 JSON, 저장 관리**를 붙인 프로그램이다.

```bash
source ~/urgent/venv/bin/activate
python ~/urgent/urgent_service.py --wav ~/urgent/streams/stream_urg_first10min.wav            # 마이크 없이 파일 재생
python ~/urgent/urgent_service.py --mic plughw:<카드번호>,0 --channels 2                      # 마이크 (도착 후 확인)
```
- 값 바꾸기: `--set theta=0.99 --set n=2` 또는 `--config 파일.json`. 키: `theta, n, gate_dbfs, cooldown_s, hop_s, pre_s, post_s, device_id, model_name, model_file, model_dir, out_dir, max_clips, min_free_mb, threads`.
- 결과: `out_dir`(기본 `~/urgent/events`)에 `evt_<시각>_<번호>.wav`(20초, 16kHz 모노), `.json`, `events.jsonl`.
- JSON 항목: `event_id, device_id, detected_at(+09:00), time_synced, model{name, threshold, consecutive, gate_dbfs}, score{max_prob, mean_prob}, clip{file, sample_rate, channels, pre_sec, post_sec, bytes, complete}, speaker(null)`. 형식은 팀 협의 후 바뀔 수 있다 ([08](08_TEAM_SHARING.md)).
- 스트림 맨 앞이면 앞 구간이 10초보다 짧게, 입력이 먼저 끝나면 `complete: false`로 저장한다.
- 저장 개수 제한(`max_clips`)을 넘으면 오래된 것부터 지우고, 디스크가 `min_free_mb`보다 적으면 저장을 건너뛴다.
- **마이크 입력은 `arecord` 대신 파일을 파이프로 흘려서 확인**했고 실제 마이크는 도착 후 확인한다. 스테레오는 평균해서 모노로 쓴다.

### 후보 모델 실행 결과 (파이4, 1스레드, 10분 스트림)
| 모델 | 규칙 | 판단 한 번 | 경보 |
|---|---|---|---|
| v1 (기준값 0.95, 연속 3회) | | 42.1ms(4스레드 기본값) → 23.1ms(1스레드) | 8번 |
| v2_wide | 0.999, 연속 1회 | 54.7ms (실시간의 약 11%) | 9번 |
| v2_no_part | 0.99, 연속 2회 | 23.1ms (약 4.6%) | 8번 |

- 두 후보의 경보 시각이 거의 겹친다. 정식 비교는 60분 스트림의 [04 §9](04_MODEL_EXPERIMENTS.md).
- ONNX 스레드는 1개로 고정한다(4개가 기본값이면 판단이 약 2배 느렸다).
