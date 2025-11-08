# Football Play Analysis System

AI 기반 컴퓨터 비전 시스템으로 미식축구 경기 영상을 분석하여 자동으로 플레이 통계를 추출하고 구조화된 JSON 데이터를 생성합니다.

## 🎯 주요 기능

- **선수 및 공 자동 감지 및 추적** (YOLOv8 + CLIP)
- **팀 자동 분류** (CLIP 기반 유니폼 색상 감지 - 흰색/노란색)
- **사이드라인 인원 자동 제외** (호모그래피 + BEV 경계 필터링)
- **플레이 타입 자동 판정** (항상 RUN)
- **야드 게인 자동 계산** (BEV 변환 기반)
- **포즈 분석** (쪼그려 앉기, 움직임 감지)
- **JSON 출력** (구조화된 플레이 데이터)

---

## 🚀 빠른 시작

### 1. 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# CLIP 설치 (팀 분류용)
./install_clip.sh

# Tesseract OCR 설치 (저지 번호 인식용)
# macOS:
brew install tesseract
# Ubuntu:
sudo apt-get install tesseract-ocr
```

### 2. 호모그래피 캘리브레이션 (최초 1회)

```bash
python calibrate_homography.py
```

- 비디오 첫 프레임에서 필드의 4개 지점 클릭
- 's' 키를 눌러 저장
- `homography_matrix.npy` 파일 생성됨

### 3. 비디오 분석 실행

```bash
# 비디오를 input/ 폴더에 배치
cp your_video.mp4 input/video.mp4

# 분석 실행
python main.py
```

### 4. 결과 확인

- **output/result.mp4** - 바운딩 박스와 팀 색상이 표시된 영상
- **output/bev.mp4** - 평면도 (Bird's Eye View) 영상
- **output/clip_summary.json** - 구조화된 플레이 데이터

---

## 📊 시스템 아키텍처

### AI 모델 파이프라인

```
비디오 입력
    ↓
┌─────────────────────────────────────────┐
│  YOLOv8 Detection (선수/공 감지)        │
│  - 신뢰도: 선수 0.5 / 공 0.2            │
│  - 크기 필터링: 5-300px                 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  CLIP 분류 (팀/엔티티)                  │
│  - 팀: 흰색=Team A / 노란색=Team B      │
│  - 1프레임 즉시 고정 (변경 불가)        │
│  - 프레임 간격: 30 (1초당 1번)          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  호모그래피 변환 (BEV)                  │
│  - 사이드라인 필터링 (50-950, 50-450)   │
│  - 야드 계산 (20 pixels/yard)           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  PlayAnalyzer (플레이 분석)             │
│  - 플레이 타입: 항상 RUN                │
│  - 볼 캐리어 추적                       │
│  - 포즈 분석 (쪼그려 앉기/움직임)       │
└─────────────────────────────────────────┘
    ↓
결과 출력 (MP4 + JSON)
```

### 핵심 컴포넌트

| 컴포넌트 | 파일 | 기능 |
|---------|------|------|
| **DetectorTracker** | detector_tracker.py | YOLOv8 감지 + CLIP 분류 통합 |
| **CLIPClassifier** | clip_classifier.py | 팀/엔티티 분류 (1프레임 즉시 고정) |
| **ViewTransformer** | transformer.py | 호모그래피 변환 + BEV 경계 필터링 |
| **PlayAnalyzer** | play_analyzer.py | 플레이 분석 + 포즈 분석 |
| **Visualizer** | visualizer.py | 영상 렌더링 (bbox, 팀 색상, BEV) |
| **TeamClassifier** | team_classifier.py | 레거시 K-means 팀 분류 (백업) |

---

## ⚙️ 주요 설정 (config.py)

### 공 감지 설정

```python
BALL_CONFIDENCE = 0.2           # 공 감지 임계값 (매우 낮음 - 미식축구공 감지 어려움)
BALL_SIZE_MIN = 5               # 최소 크기 (픽셀)
BALL_SIZE_MAX = 300             # 최대 크기 (픽셀)
```

### 추적 지속성 설정 (한 번 디텍션되면 계속 유지)

```python
MAX_TRACKING_FRAMES = 300       # 10초 동안 감지 안 돼도 track 유지
TRACKING_IOU_THRESHOLD = 0.15   # IoU 임계값 (낮을수록 관대)

# detector_tracker.py에서:
tracker='bytetrack.yaml'        # ByteTrack 알고리즘 (안정적)
iou=0.3                         # 겹침 30%까지 허용
```

**효과**: 한 번 감지된 선수/공은 화면에서 잠깐 가려져도 계속 추적됨

### CLIP 팀 분류 설정

```python
CLIP_FRAME_INTERVAL = 30        # 30프레임마다 1번 = 1초당 1번 (30fps 기준)
CLIP_CONFIDENCE_THRESHOLD = 0.40

CLIP_TEAM_PROMPTS = [
    "an american football player wearing a white team jersey",     # Team A (흰색)
    "an american football player in a bright yellow team uniform", # Team B (노란색)
    "a football referee in black and white vertical striped shirt" # Referee
]
```

**팀 고정 메커니즘**:
- `freeze_threshold = 1` - 첫 감지 시 즉시 고정
- `confidence >= 0.3` - 30% 이상이면 고정
- 한 번 고정되면 **절대 변경 불가**

### BEV 필드 경계 (사이드라인 제외)

```python
BEV_FIELD_X_MIN = 50      # 왼쪽 사이드라인
BEV_FIELD_X_MAX = 950     # 오른쪽 사이드라인
BEV_FIELD_Y_MIN = 50      # 상단 엔드존
BEV_FIELD_Y_MAX = 450     # 하단 (벤치/사이드라인 제외)
```

---

## 🎮 사용 방법

### 기본 실행

```bash
python main.py
```

### 예상 로그

```
Loading CLIP model 'ViT-B/32' on device 'cuda'...
CLIP Team Classifier initialized with 3 team prompts
Team freeze mechanism: IMMEDIATE (1-frame freeze)

⚽ Ball detected! Track #5, conf=22%, size=25x28
🔒 Team FROZEN: Track #1 → Team A (conf: 45%)
🔒 Team FROZEN: Track #3 → Team B (conf: 38%)

Play started: State -> PLAY_ACTIVE (RUN), Passer: 3
Play ended: Carrier down. Tackler: 8, Yards gained: 5.20

============================================================
PLAY SUMMARY
============================================================
Play Type: RUN
Yards Gained: 5.20
Ball Carrier ID: 3
Final State: PLAY_ENDED
============================================================
```

### JSON 출력 형식

```json
{
  "gameKey": "GAME_2024_001",
  "Clips": [
    {
      "playType": "RUN",
      "gainYard": 5.2,
      "carrierInfo": {
        "trackId": 3,
        "jerseyNumber": "23"
      },
      "tacklerInfo": {
        "trackId": 8,
        "jerseyNumber": "Unknown"
      }
    }
  ]
}
```

---

## 🔧 문제 해결

### 문제 1: 공이 감지 안 됨

**해결책**:
```python
# config.py
BALL_CONFIDENCE = 0.15  # 0.2 → 0.15로 더 낮춤
```

### 문제 2: 팀이 계속 바뀜

**원인**: 신뢰도가 0.3 미만
**해결책**:
1. CLIP 프롬프트를 더 구체적으로 수정
2. 신뢰도 임계값 낮추기:
```python
# clip_classifier.py:315
if all(h[1] >= 0.2 for h in history):  # 0.3 → 0.2
```

### 문제 3: 사이드라인 인원이 감지됨

**해결책**: BEV 경계 좁히기
```python
# config.py
BEV_FIELD_X_MIN = 100  # 50 → 100
BEV_FIELD_X_MAX = 900  # 950 → 900
BEV_FIELD_Y_MAX = 400  # 450 → 400
```

### 문제 4: yards gained가 0으로 나옴

**원인**: 플레이가 끝나지 않음 (PLAY_ENDED 상태 도달 안 함)
**해결책**:
1. 비디오가 플레이 끝까지 포함하는지 확인
2. 포즈 감지 임계값 조정 (play_analyzer.py:185)

### 문제 5: 호모그래피 매트릭스 오류

```bash
FileNotFoundError: homography_matrix.npy
```

**해결책**:
```bash
python calibrate_homography.py
```

---

## 📖 핵심 개념

### 1. 팀 고정 메커니즘

```
프레임 1: Track #1 감지
  ├─ CLIP 분류: Team A (conf: 45%)
  ├─ 신뢰도 >= 0.3? YES
  └─ 🔒 즉시 고정! (1프레임)

프레임 2-끝: Track #1 재감지
  └─ frozen_teams에 있음 → 절대 변경 불가
```

### 2. 플레이 타입 판정

**현재 로직**: **항상 RUN**

```python
# play_analyzer.py
self.play_type = 'RUN'  # 기본값 (변경 불가)
```

모든 플레이는 RUN입니다. PASS 자동 감지는 제거되었습니다.

### 3. BEV 좌표계

```
BEV Canvas (1000x500 pixels)
┌─────────────────────────────┐ y=0   (상단)
│   x=50                x=950 │
│   ↓                      ↓  │
├───┐                    ┌───┤
│   │  경기장 (on-field) │   │
│   │  BEV_FIELD_X_MIN   │   │
│   │  ~                 │   │
│   │  BEV_FIELD_X_MAX   │   │
│   │                    │   │
├───┘                    └───┤ y=450 (BEV_FIELD_Y_MAX)
│  사이드라인 (제외)          │
└─────────────────────────────┘ y=500 (하단)
```

### 4. 야드 계산

```python
pixel_distance = euclidean(start_bev, end_bev)
yards_gained = pixel_distance / PIXELS_PER_YARD_BEV  # 20 pixels/yard

# 예시: 100 pixels 이동 = 5 yards
```

### 5. 포즈 분석

**쪼그려 앉기 감지**:
```python
hip_height = avg_hip_y - avg_shoulder_y
knee_bend = avg_knee_y - avg_hip_y

if hip_height < knee_bend * 0.7:
    stance = 'CROUCHED'  # PRE_SNAP 자세
```

**움직임 감지**:
```python
speed = distance_moved / PIXELS_PER_YARD_BEV  # yards/frame

if speed > 0.5:
    is_moving = True
```

---

## 🎓 고급 설정

### CLIP 프롬프트 커스터마이징

게임마다 유니폼 색상이 다르면 프롬프트 수정:

```python
# config.py
CLIP_TEAM_PROMPTS = [
    "an american football player wearing a bright red team jersey with white numbers",
    "an american football player in a dark blue uniform with yellow helmet",
    "a football referee in black and white vertical striped shirt"
]
```

### 포즈 분석 활용 (main.py 추가 가능)

```python
# 쪼그려 있는 선수 필터링
if pose_result:
    posture = analyzer.analyze_player_posture(pose_result)
    if posture['is_crouching']:
        print(f"Player {track_id} is in PRE_SNAP stance")

# 움직임 기반 분석
movement = analyzer.detect_player_movement(track_id, bev_pos)
if movement['is_moving']:
    print(f"Player {track_id} moving at {movement['speed']:.2f} y/f")
```

### 프레임 간격 조정

더 빠른 감지 (더 많은 계산):
```python
CLIP_FRAME_INTERVAL = 15  # 30 → 15 (0.5초당 1번)
```

더 느린 감지 (더 적은 계산):
```python
CLIP_FRAME_INTERVAL = 60  # 30 → 60 (2초당 1번)
```

---

## 📋 버전 히스토리

### V3 (현재) - 2024
**주요 변경사항**:
- ✅ **플레이 타입 항상 RUN** (PASS 자동 감지 완전 제거)
- ✅ **팀 1프레임 즉시 고정** (freeze_threshold: 2 → 1)
- ✅ **프레임 간격 30** (1초당 1번 감지)
- ✅ **공 임계값 0.2** (이전 0.3)
- ✅ **공 크기 범위 5-300px** (이전 10-200px)
- ✅ **포즈 분석 메서드 추가** (쪼그려 앉기, 움직임)
- ✅ **추적 지속성 강화** (한 번 디텍션되면 10초간 유지)

**핵심 원칙**:
- "던지지 않으면 모두 RUN"
- "처음 1번 감지 후 팀 고정"

### V2 (이전)
- 높이 기반 PASS 감지 (제거됨)
- 2프레임 팀 고정 (1프레임으로 변경)
- 프레임 간격 5 (30으로 변경)

### V1 (초기)
- K-means 팀 분류만 사용
- CLIP 미통합
- 수동 사이드라인 제외

---

## 🛠️ 개발자 가이드

### 주요 파일 구조

```
projects/
├── main.py                    # 메인 실행 파일
├── config.py                  # 중앙 설정 파일 ⭐
├── detector_tracker.py        # YOLO + CLIP 통합
├── clip_classifier.py         # CLIP 팀/엔티티 분류 ⭐
├── play_analyzer.py           # 플레이 분석 + 포즈 분석 ⭐
├── transformer.py             # 호모그래피 + BEV 필터링
├── visualizer.py              # 영상 렌더링
├── team_classifier.py         # 레거시 K-means (백업)
├── number_recognizer.py       # Tesseract OCR
├── calibrate_homography.py    # 캘리브레이션 도구
├── input/                     # 입력 비디오
├── output/                    # 출력 파일
│   ├── result.mp4            # 주석이 달린 영상
│   ├── bev.mp4               # 평면도 영상
│   └── clip_summary.json     # 플레이 데이터
└── homography_matrix.npy      # 변환 매트릭스
```

### 코드 수정 가이드

| 수정 목적 | 파일 | 라인 |
|-----------|------|------|
| 공 감지 임계값 | config.py | 36 |
| 팀 색상 프롬프트 | config.py | 181-184 |
| 프레임 간격 | config.py | 167 |
| 팀 고정 속도 | clip_classifier.py | 216 |
| 팀 고정 신뢰도 | clip_classifier.py | 315 |
| BEV 경계 | config.py | 134-137 |
| 플레이 타입 로직 | play_analyzer.py | 27, 132 |
| 포즈 임계값 | play_analyzer.py | 340 |

---

## ⚠️ 알려진 제약사항

1. **플레이 타입**: 항상 RUN (PASS 감지 안 됨)
2. **1플레이 처리**: 한 번에 1개 플레이만 처리 (배치 처리 불가)
3. **호모그래피 필수**: 실행 전 캘리브레이션 필수
4. **사이드뷰 영상**: 포즈 감지 정확도 낮을 수 있음
5. **저지 번호**: OCR 정확도 제한적 (측면 영상)
6. **팀 고정**: 첫 감지가 틀리면 영구적으로 잘못된 팀 할당

---

## 📚 참고 자료

### 사용 기술
- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **OpenAI CLIP**: https://github.com/openai/CLIP
- **OpenCV**: https://opencv.org/
- **Tesseract OCR**: https://github.com/tesseract-ocr/tesseract

### 핵심 알고리즘
- **호모그래피 변환**: `cv2.perspectiveTransform()`
- **CLIP Zero-Shot 분류**: 텍스트 프롬프트 기반 이미지 분류
- **COCO 키포인트**: 17개 신체 포인트 (YOLOv8-pose)
- **유클리드 거리**: `scipy.spatial.distance.euclidean()`

---

## 🤝 기여 및 지원

문제가 발생하면:
1. 로그 메시지 확인 (⚽ Ball detected, 🔒 Team FROZEN 등)
2. 설정 파일 (config.py) 검토
3. 호모그래피 매트릭스 재캘리브레이션
4. BEV 경계 조정

---

**시스템 버전**: V3 (2024)
**핵심 원칙**: "던지지 않으면 모두 RUN" + "처음 1번 감지 후 팀 고정"
**프레임 간격**: 30 (1초당 1번 감지)
**팀 고정**: 1프레임 즉시 고정 (변경 불가)
