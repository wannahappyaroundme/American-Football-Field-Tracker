# Football Play Analysis System - 기술 분석 문서

> 영상 데이터에서 플레이 통계 자동 추출 시스템의 완전한 기술 명세서

---

## 📋 목차

1. [시스템 개요](#1-시스템-개요)
2. [기술 스택](#2-기술-스택)
3. [완전한 데이터 플로우 파이프라인](#3-완전한-데이터-플로우-파이프라인)
4. [모든 Python 파일 및 역할](#4-모든-python-파일-및-역할)
5. [핵심 알고리즘 & 수식](#5-핵심-알고리즘--수식)
6. [성능 특성](#6-성능-특성)
7. [주요 설정 및 임계값](#7-주요-설정-및-임계값)
8. [State Machine](#8-state-machine)
9. [알려진 제약사항](#9-알려진-제약사항)
10. [데이터 구조](#10-데이터-구조)
11. [의존성](#11-의존성)
12. [아키텍처 비교](#12-아키텍처-비교)
13. [중요 파일](#13-중요-파일)
14. [핵심 메트릭](#14-핵심-메트릭)

---

## 1. 시스템 개요

### 목적
미식축구 경기 영상에서 **컴퓨터 비전 및 AI**를 사용하여 플레이 통계를 **자동으로 추출**

### 핵심 파이프라인
```
Video Input → Detection/Tracking → Transformation → Play Analysis → Visualization → JSON Output
```

### 처리 방식
- **한 번에 하나의 플레이** 처리 (배치 처리 없음)
- **프레임별 처리** (30 FPS 기준)
- **상태 기반** 플레이 분석 (4-state machine)
- **실시간 처리 가능** (10-30 FPS on GPU)

---

## 2. 기술 스택

### 🧠 Deep Learning Models

| 모델 | 버전 | 용도 | 클래스/출력 |
|------|------|------|------------|
| **YOLOv8n** | nano | 객체 탐지 | Person (0), Ball (32) |
| **YOLOv8n-pose** | nano | 포즈 추정 | 17 COCO keypoints |
| **OpenAI CLIP** | ViT-B/32 | 제로샷 분류 | Entity filtering, Team classification |

### 📷 Computer Vision & Image Processing

| 라이브러리 | 주요 기능 |
|-----------|----------|
| **OpenCV (cv2)** | Homography transformation, Perspective transform, Video I/O, Morphological operations |
| **NumPy** | 행렬 연산, 수치 계산 |
| **Pillow (PIL)** | CLIP용 이미지 포맷 변환 |

### 📊 Data Processing & Machine Learning

| 라이브러리 | 주요 기능 |
|-----------|----------|
| **scikit-learn** | K-means clustering (색상 기반 팀 분류) |
| **SciPy** | Euclidean distance (Ball carrier 식별) |

### 🔧 Additional Tools

| 도구 | 용도 |
|------|------|
| **Tesseract OCR** | 등번호 인식 (pytesseract) |
| **PyTorch** | CLIP 모델 실행 (CUDA/CPU) |
| **FFmpeg codec (mp4v)** | 비디오 인코딩 |

### 🎯 Tracking Algorithm

| 알고리즘 | 특징 |
|---------|------|
| **ByteTrack** | 2-stage association, Kalman filter prediction, 900 frames buffer (30초) |

---

## 3. 완전한 데이터 플로우 파이프라인

### 전체 플로우 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT (MP4)                            │
│              cv2.VideoCapture(VIDEO_INPUT_PATH)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│              FRAME-BY-FRAME PROCESSING LOOP                       │
│                    (continues until PLAY_ENDED)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: DETECTION & TRACKING (DetectorTracker.track_frame)      │
│                                                                   │
│  • YOLOv8 Detection (conf=0.25)                                  │
│    - Classes: PERSON (0), BALL (32)                              │
│    - Output: [x1, y1, x2, y2], track_id, class_id, confidence   │
│                                                                   │
│  • ByteTrack Persistence (track_buffer=900 frames = 30 sec)      │
│    - Maintains tracks even when detection fails                  │
│    - New track threshold: 0.5 (prevents track fragmentation)     │
│    - IoU threshold: 0.2 (very permissive matching)               │
│                                                                   │
│  • Person Confidence Filtering                                   │
│    - Person: conf >= 0.3 (lenient)                               │
│    - Ball: conf >= 0.15 (very lenient)                           │
│                                                                   │
│  • Ball Size Validation                                          │
│    - 5px <= width/height <= 300px (excludes false positives)     │
│                                                                   │
│  OUTPUT: List of tracks with [bbox, track_id, class_id, conf]    │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: CLIP ENTITY CLASSIFICATION (every 30 frames)            │
│                                                                   │
│  For each person detection:                                      │
│  • Crop frame region [x1:x2, y1:y2]                              │
│  • Convert BGR → RGB → PIL Image                                 │
│  • Run CLIP.encode_image() with batch processing (16 crops)      │
│  • Compare against 4 entity prompts:                             │
│    1. "active football player in uniform on field"               │
│    2. "referee in black/white striped uniform"                   │
│    3. "coach/sideline staff on sideline"                         │
│    4. "photographer/cameraman with equipment"                    │
│                                                                   │
│  • Filter results: exclude 'sideline' and 'other'                │
│  • Cache results (reuse for non-classification frames)           │
│                                                                   │
│  OUTPUT: Each track includes entity_type, entity_confidence      │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: COORDINATE TRANSFORMATION (ViewTransformer)             │
│                                                                   │
│  • Load homography_matrix.npy (4x4 perspective transform)        │
│  • For each player foot position:                                │
│    - get_foot_position(bbox) → center-bottom point (x, y_max)   │
│    - cv2.perspectiveTransform([x, y]) → [x_bev, y_bev]          │
│                                                                   │
│  • BEV Canvas: 1000x500 pixels (1000=50 yards, 500=25 yards)     │
│  • Field Boundaries (exclude sideline/bench):                    │
│    - X: [50, 950] pixels = on-field area                        │
│    - Y: [50, 450] pixels = sideline benches at 450-500           │
│                                                                   │
│  • Yard Line Conversion:                                         │
│    yards_from_left = x_bev / PIXELS_PER_YARD_BEV (20 px/yd)     │
│    yard_line = BEV_LEFT_YARD_LINE (0) + yards_from_left          │
│    side = 'OWN' if yard_line <= 50 else 'OPP'                    │
│    yard = int(round(yard_line)) or int(round(100-yard_line))     │
│                                                                   │
│  OUTPUT: BEV coordinates, yard line info per player              │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4: PLAY ANALYSIS - STATE MACHINE (PlayAnalyzer.update)     │
│                                                                   │
│  State Transitions:                                              │
│  ┌──────────┐                                                    │
│  │PRE_SNAP  │ ← Initial state                                    │
│  └────┬─────┘                                                    │
│       │ (ball carrier detected)                                  │
│       ↓                                                           │
│  ┌──────────────┐                                                │
│  │PLAY_ACTIVE   │ ← Default play type: 'RUN'                     │
│  └────┬────┬────┘                                                │
│       │    │ (ball_carrier's knee below ankle)                   │
│       │    └──────→ [pose detection via YOLOv8-pose]             │
│       │             (see Step 5)                                 │
│       │                                                           │
│       │ (ball leaves carrier)                                    │
│       ├──→ BALL_IN_AIR ──→ [catch by new player] ──→ PLAY_ACTIVE│
│       │                                                           │
│       └──→ PLAY_ENDED ✓                                          │
│                                                                   │
│  Core Operations:                                                │
│  • Ball Carrier ID: euclidean_distance(ball_bev, person_bev)    │
│  • Field Filtering: is_on_field(bev_pos) via boundary check     │
│  • Player States: {track_id: {path: [(x,y,status)...],          │
│    last_known_position, last_seen_frame}}                        │
│  • Occlusion Handling: Predict position for up to 30 frames     │
│                                                                   │
│  OUTPUT: state, ball_carrier_id, play_type, player_states       │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  STEP 5: POSE ANALYSIS (if play_active & ball_carrier detected)  │
│                                                                   │
│  • Crop frame to ball_carrier's bbox                             │
│  • Run YOLOv8-pose on cropped image (17 COCO keypoints)          │
│  • Check play end condition:                                     │
│    if min(knee_y) > max(ankle_y):  # Y increases downward       │
│        → state = PLAY_ENDED                                      │
│        → find closest tackler (euclidean distance in BEV)        │
│        → calculate_yards_gained()                                │
│                                                                   │
│  COCO Keypoints (indices used):                                  │
│    13: left knee,  14: right knee                                │
│    15: left ankle, 16: right ankle                               │
│                                                                   │
│  Additional Posture Analysis (optional):                         │
│    • hip_height = avg_hip_y - avg_shoulder_y                     │
│    • is_crouching = hip_height < knee_bend * 0.7                 │
│    • speed = distance_moved / PIXELS_PER_YARD_BEV                │
│    • is_moving = speed > 0.5 yards/frame                         │
│                                                                   │
│  OUTPUT: play_ended flag, tackler_id, yards_gained               │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  STEP 6: TEAM CLASSIFICATION (CLIP-based with 1-frame freeze)    │
│                                                                   │
│  Every 10 frames, for each new/unfrozen player:                  │
│  • If CLIP_TEAM_CLASSIFICATION enabled:                          │
│    - Crop player image                                           │
│    - Run CLIP.encode_image() against 3 team prompts:             │
│      1. "player wearing yellow jersey + white pants"             │
│      2. "player wearing white jersey + black pants"              │
│      3. "referee in striped shirt"                               │
│    - Get highest confidence match                                │
│                                                                   │
│  ⭐ Team Freeze Mechanism (CRITICAL):                            │
│    - freeze_threshold: 1 frame (IMMEDIATE)                       │
│    - Once confidence >= 0.3 on 1st detection:                    │
│      → frozen_teams[track_id] = team_label                       │
│      → 🔒 TEAM FROZEN (cannot change)                            │
│                                                                   │
│  Fallback (if CLIP disabled):                                    │
│    • Use color-based K-means clustering (TeamClassifier)         │
│    • Extract dominant jersey color (HSV upper 40% of crop)       │
│    • K-means(n_clusters=4) on filtered pixels                    │
│    • Identify referee (low saturation), others (small cluster)    │
│    • Assign Team A, Team B to remaining clusters                 │
│                                                                   │
│  OUTPUT: player_teams mapping {track_id: 'Team A'/'Team B'/...}  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  STEP 7: VISUALIZATION                                           │
│                                                                   │
│  A. Draw Annotations (result.mp4):                               │
│     For each track on original frame:                            │
│     • Draw bbox with team color:                                 │
│       - Ball carrier: RED (0,0,255)                              │
│       - Team A: BLUE (255,0,0)                                   │
│       - Team B: RED (0,0,255)                                    │
│       - Referee: YELLOW (0,255,255)                              │
│       - Unknown: GRAY (128,128,128)                              │
│     • Add labels: "#track_id (Team X)"                           │
│     • Overlay play state & type (top-left)                       │
│     • Filter off-field players (via is_on_field check)           │
│                                                                   │
│  B. Bird's Eye View (bev.mp4):                                   │
│     For each player_state in player_states:                      │
│     • Draw path lines (solid for active, dotted for occluded)    │
│     • Use team color for path                                    │
│     • Mark current position with circle (radius=5px)             │
│     • Add track ID label                                         │
│     • Canvas: 1000x500 pixels (white background)                 │
│                                                                   │
│  OUTPUT: Two annotated video files                               │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  STEP 8: JERSEY NUMBER RECOGNITION (NumberRecognizer)            │
│                                                                   │
│  • Stored player_crops during play (up to 5 per player)          │
│  • Select best crop (largest by area)                            │
│  • Apply preprocessing:                                          │
│    - Grayscale conversion                                        │
│    - Binary inverse threshold (THRESH_BINARY_INV)                │
│  • Run Tesseract OCR with config:                                │
│    --psm 6 (uniform block of text)                               │
│    -c tessedit_char_whitelist=0123456789 (digits only)           │
│  • Return recognized number or None                              │
│                                                                   │
│  Note: Accuracy limited for side-view footage                    │
│                                                                   │
│  OUTPUT: {track_id: jersey_number_string or None}                │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  STEP 9: JSON OUTPUT GENERATION                                  │
│                                                                   │
│  Structure:                                                      │
│  {                                                               │
│    "gameKey": "GAME_2024_001",                                   │
│    "gameDate": "2024-01-15",                                     │
│    "homeTeam": "Home Team",                                      │
│    "awayTeam": "Away Team",                                      │
│    "stadium": "Stadium Name",                                    │
│    "Clips": [                                                    │
│      {                                                           │
│        "clipKey": "CLIP_001",                                    │
│        "playType": "RUN",  ← AI-derived (always RUN)             │
│        "gainYard": 5.2,    ← AI-derived (calculated from BEV)    │
│        "carrierInfo": {                                          │
│          "trackId": 3,     ← AI-derived                          │
│          "jerseyNumber": "23"  ← AI-derived (OCR)                │
│        },                                                        │
│        "tacklerInfo": {                                          │
│          "trackId": 8,                                           │
│          "jerseyNumber": "Unknown"                               │
│        },                                                        │
│        "quarter": 1,       ← Manual metadata                     │
│        "down": 1,          ← Manual metadata                     │
│        ... other manual fields ...                               │
│      }                                                           │
│    ]                                                             │
│  }                                                               │
│                                                                   │
│  OUTPUT: /output/clip_summary.json                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. 모든 Python 파일 및 역할

| 파일명 | 줄 수 | 주요 역할 | 핵심 함수/클래스 |
|--------|------|-----------|-----------------|
| **main.py** | 313 | 메인 파이프라인 오케스트레이터 | `main()`: 모든 컴포넌트 초기화, 프레임 처리 루프, JSON 생성 |
| **config.py** | 209 | 중앙 설정 파일 | 모든 경로, 임계값, 색상, 모델 설정 (200+ 설정값) |
| **detector_tracker.py** | 206 | Detection + Tracking | `DetectorTracker.track_frame()`: YOLO detection, ByteTrack, CLIP entity filtering |
| **play_analyzer.py** | 420 | 상태 머신 & 플레이 분석 | `PlayAnalyzer.update_tracks()`: 4-state machine, pose detection, yard calculation, posture analysis |
| **transformer.py** | 128 | 좌표 변환 | `ViewTransformer.transform_point()`: Homography, BEV transformation, yard line calculation, field boundary checking |
| **visualizer.py** | 239 | 비디오 렌더링 | `Visualizer.draw_annotations()`, `draw_bird_eye_view()`: 두 개의 출력 비디오 생성 |
| **team_classifier.py** | 280 | 색상 기반 팀 분류 | `TeamClassifier.assign_teams()`: K-means clustering, referee/other detection (CLIP 백업) |
| **clip_classifier.py** | 444 | CLIP 기반 분류 | `CLIPEntityClassifier`, `CLIPTeamClassifier`: 제로샷 분류, 1-frame freeze |
| **number_recognizer.py** | 62 | 등번호 OCR | `NumberRecognizer.recognize_number()`: Tesseract, binary thresholding, digit-only whitelist |
| **calibrate_homography.py** | 88 | Homography 보정 도구 | 4점 선택, homography matrix 계산 및 저장 |
| **tracker.py** | 52,795 | 대체 파이프라인 (미사용) | Enhanced tracking with stadium masking (main.py에서 사용 안 함) |
| **test_clip_integration.py** | 119 | 단위 테스트 | CLIP 통합 테스트 |

### 파일별 상세 설명

#### **main.py** (313줄)
```python
# 주요 흐름
def main():
    # 1. CLIP classifiers 초기화 (if enabled)
    # 2. YOLO models 로드 (detection + pose)
    # 3. Components 초기화 (transformer, analyzer, visualizer, etc.)
    # 4. Video I/O 설정
    # 5. Frame-by-frame processing loop:
    #    - track_frame() → CLIP entity filtering
    #    - CLIP team classification (every 10 frames, skip frozen)
    #    - update_tracks() → state machine
    #    - pose detection (if ball carrier exists)
    #    - draw_annotations() + draw_bird_eye_view()
    #    - write output frames
    #    - break if state == 'PLAY_ENDED'
    # 6. Jersey number recognition
    # 7. JSON output generation
```

#### **detector_tracker.py** (206줄)
```python
class DetectorTracker:
    def __init__(self, clip_classifier=None):
        # YOLO model 로드
        # ByteTrack 설정 (track_buffer=900)
        # CLIP entity classifier 연결

    def track_frame(self, frame):
        # 1. YOLO tracking (conf=0.25, iou=0.2, persist=True)
        # 2. 클래스별 confidence 필터링 (Person: 0.3, Ball: 0.15)
        # 3. Ball 크기 검증 (5-300px)
        # 4. CLIP entity classification (batch, every 30 frames)
        # 5. Entity cache 활용 (non-classification frames)
        # 6. Entity filtering (exclude sideline/other)
        # Return: tracks with entity_type
```

#### **play_analyzer.py** (420줄)
```python
class PlayAnalyzer:
    def __init__(self):
        self.state = 'PRE_SNAP'  # 4 states
        self.ball_carrier_id = None
        self.play_type = 'RUN'
        self.player_states = {}  # {track_id: {path, last_known_position, last_seen_frame}}

    def update_tracks(self, tracks, view_transformer):
        # 1. Ball carrier identification (euclidean distance in BEV)
        # 2. State transitions (PRE_SNAP → PLAY_ACTIVE → BALL_IN_AIR → PLAY_ENDED)
        # 3. Player state management (active/occluded paths)
        # 4. Occlusion prediction (up to 30 frames)
        # 5. Store player crops (up to 5 per player)

    def check_play_end(self, pose_results, tracks, view_transformer):
        # Pose keypoint analysis (knee vs ankle Y-coordinates)
        # Find tackler (closest to ball carrier in BEV)
        # Calculate yards gained
```

#### **transformer.py** (128줄)
```python
class ViewTransformer:
    def __init__(self):
        # Load homography_matrix.npy (4x4 perspective transform)

    def get_foot_position(self, bbox):
        # Return center-bottom point: ((x1+x2)/2, y2)

    def transform_point(self, point_tuple):
        # cv2.perspectiveTransform: camera → BEV coordinates

    def bev_to_yard_line(self, bev_pos):
        # Calculate yard line from BEV coordinates
        # Return: {side: 'OWN'/'OPP', yard: int}

    def is_on_field(self, bev_pos):
        # Check if position within field boundaries
        # X: [50, 950], Y: [50, 450]
```

#### **clip_classifier.py** (444줄)
```python
class CLIPEntityClassifier:
    def __init__(self):
        # Load CLIP model (ViT-B/32)
        # Define 4 entity prompts

    def batch_classify_entities(self, crops, track_ids):
        # Batch process 16 crops at once
        # Return: [(entity_type, confidence), ...]

class CLIPTeamClassifier:
    def __init__(self):
        # Load CLIP model
        # Define 3 team prompts
        self.frozen_teams = {}  # Immutable after 1st detection

    def classify_team(self, crop):
        # Run CLIP against team prompts
        # Return: (team_label, confidence)

    def is_frozen(self, track_id):
        # Check if team already frozen (performance optimization)
```

---

## 5. 핵심 알고리즘 & 수식

### 5.1 Ball Carrier Identification

**알고리즘:** Euclidean distance minimization in BEV space

```python
def identify_ball_carrier(ball_bev_pos, person_tracks_bev):
    min_distance = float('inf')
    ball_carrier_id = None

    for person in person_tracks_bev:
        distance = euclidean(ball_bev_pos, person['bev_pos'])

        if distance < min_distance:
            min_distance = distance
            ball_carrier_id = person['track_id']

    return ball_carrier_id
```

**수식:**
```
distance = √[(x_ball - x_person)² + (y_ball - y_person)²]
ball_carrier = argmin_person(distance)
```

### 5.2 Yards Gained Calculation

**알고리즘:** BEV distance conversion to yards

```python
def calculate_yards_gained(start_ball_bev, end_ball_bev):
    pixel_distance = euclidean(start_ball_bev, end_ball_bev)
    gain_yards = pixel_distance / PIXELS_PER_YARD_BEV  # 20 px/yard
    return gain_yards
```

**수식:**
```
pixel_distance = √[(x_end - x_start)² + (y_end - y_start)²]
gain_yards = pixel_distance / 20
```

### 5.3 Yard Line Conversion

**알고리즘:** BEV X-coordinate to field yard line

```python
def bev_to_yard_line(bev_pos):
    x_bev, y_bev = bev_pos

    # Convert BEV pixels to yards
    yards_from_left = x_bev / PIXELS_PER_YARD_BEV  # 20 px/yard

    # Add reference yard line (left edge)
    yard_line = BEV_LEFT_YARD_LINE + yards_from_left  # BEV_LEFT_YARD_LINE = 0

    # Determine side (OWN vs OPP)
    if yard_line <= 50:
        side = 'OWN'
        yard = int(round(yard_line))
    else:
        side = 'OPP'
        yard = int(round(100 - yard_line))

    return {'side': side, 'yard': yard}
```

**수식:**
```
yards_from_left = x_bev / 20
yard_line = 0 + yards_from_left

if yard_line ≤ 50:
    side = 'OWN', yard = round(yard_line)
else:
    side = 'OPP', yard = round(100 - yard_line)
```

### 5.4 Play End Detection (Pose-Based)

**알고리즘:** COCO keypoint Y-coordinate comparison

```python
def check_play_end(pose_results):
    keypoints = pose_results[0].keypoints.xy[0]  # 17 COCO keypoints

    # Extract knee and ankle keypoints
    left_knee_y = keypoints[13][1]
    right_knee_y = keypoints[14][1]
    left_ankle_y = keypoints[15][1]
    right_ankle_y = keypoints[16][1]

    # Check if knee is below ankle (player down)
    min_knee_y = min(left_knee_y, right_knee_y)
    max_ankle_y = max(left_ankle_y, right_ankle_y)

    # Y increases downward in image coordinates
    if min_knee_y > max_ankle_y:
        return True  # Play ended

    return False
```

**수식:**
```
min_knee_y = min(knee_left_y, knee_right_y)
max_ankle_y = max(ankle_left_y, ankle_right_y)

play_ended = (min_knee_y > max_ankle_y)  # Y축 아래로 증가
```

### 5.5 Team Freeze Mechanism (CLIP)

**알고리즘:** Immediate freeze on first confident detection

```python
class CLIPTeamClassifier:
    def __init__(self):
        self.frozen_teams = {}  # {track_id: team_label}
        self.freeze_threshold = 1  # Frames (IMMEDIATE)

    def assign_team(self, track_id, crop):
        # Check if already frozen
        if track_id in self.frozen_teams:
            return self.frozen_teams[track_id]  # Immutable!

        # Run CLIP classification
        team_label, confidence = self.classify_team(crop)

        # Freeze if confidence >= 0.3
        if confidence >= 0.3:
            self.frozen_teams[track_id] = team_label
            print(f"🔒 TEAM FROZEN: Track #{track_id} → {team_label}")
            return team_label

        return team_label  # Not frozen yet
```

**수식:**
```
if track_id ∈ frozen_teams:
    return frozen_teams[track_id]

if confidence ≥ 0.3:
    frozen_teams[track_id] = team_label  # 🔒 영구 고정
```

### 5.6 Jersey Color Extraction (K-means)

**알고리즘:** Dominant color extraction from upper 40% of crop

```python
def extract_dominant_color(crop):
    h, w = crop.shape[:2]

    # Extract upper 40% (jersey area)
    crop_upper = crop[0:int(h*0.4), :]

    # Convert to HSV
    hsv = cv2.cvtColor(crop_upper, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # Filter by saturation and value
    mask = (pixels[:, 1] > 30) & (pixels[:, 2] > 30) & (pixels[:, 2] < 220)
    filtered = pixels[mask]

    # K-means clustering (n=1 for dominant color)
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(filtered)
    dominant_hsv = kmeans.cluster_centers_[0]

    # Convert back to BGR
    dominant_bgr = cv2.cvtColor(
        np.uint8([[dominant_hsv]]),
        cv2.COLOR_HSV2BGR
    )[0][0]

    return dominant_bgr
```

**수식:**
```
crop_upper = crop[0 : h×0.4, :]
HSV = RGB_to_HSV(crop_upper)
filtered = HSV[(S > 30) ∧ (30 < V < 220)]
dominant_color = KMeans(filtered, k=1).center
```

### 5.7 Homography Transformation

**알고리즘:** Perspective transform using 4-point correspondence

```python
def compute_homography(src_points, dst_points):
    # src_points: 4 clicked field points (camera view)
    # dst_points: [[0,500], [1000,500], [1000,0], [0,0]] (BEV)

    H, status = cv2.findHomography(
        np.array(src_points, dtype=np.float32),
        np.array(dst_points, dtype=np.float32)
    )

    # H: 3x3 homography matrix
    np.save('homography_matrix.npy', H)

    return H

def transform_point(point, H):
    point_array = np.array([[point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point_array, H)
    return tuple(transformed[0][0])
```

**수식:**
```
┌   ┐   ┌           ┐ ┌   ┐
│ x'│   │ h11 h12 h13│ │ x │
│ y'│ = │ h21 h22 h23│ │ y │
│ w'│   │ h31 h32 h33│ │ 1 │
└   ┘   └           ┘ └   ┘

x_bev = x' / w'
y_bev = y' / w'
```

### 5.8 ByteTrack 2-Stage Association

**알고리즘:** High-confidence → Low-confidence matching

```python
def bytetrack_associate(detections, tracks):
    # Stage 1: High confidence detections (conf >= 0.3)
    high_conf_dets = [d for d in detections if d.conf >= 0.3]
    matched_high, unmatched_tracks_high, unmatched_dets_high = \
        associate_by_iou(high_conf_dets, tracks, iou_thresh=0.2)

    # Stage 2: Low confidence detections (0.05 <= conf < 0.3)
    low_conf_dets = [d for d in detections if 0.05 <= d.conf < 0.3]
    matched_low, unmatched_tracks_low, unmatched_dets_low = \
        associate_by_iou(low_conf_dets, unmatched_tracks_high, iou_thresh=0.2)

    # Stage 3: Create new tracks (only if conf >= 0.5)
    new_tracks = [d for d in unmatched_dets_low if d.conf >= 0.5]

    # Stage 4: Remove old tracks (not matched for 900 frames)
    active_tracks = [t for t in unmatched_tracks_low if t.age < 900]

    return matched_high + matched_low + new_tracks + active_tracks
```

**수식:**
```
IoU(box1, box2) = Area(box1 ∩ box2) / Area(box1 ∪ box2)

Stage 1: match(high_conf, tracks) if IoU > 0.2
Stage 2: match(low_conf, remaining_tracks) if IoU > 0.2
Stage 3: create_new_track if conf >= 0.5
Stage 4: remove_track if age > 900 frames
```

---

## 6. 성능 특성

### 6.1 처리 속도

| 메트릭 | 값 | 조건 |
|--------|-----|------|
| **실시간 처리** | 10-30 FPS | Modern GPU (CUDA) |
| **YOLO Detection** | ~20ms/frame | YOLOv8n on GPU |
| **CLIP Overhead** | ~50ms/16 crops | Batch processing, every 30 frames |
| **Pose Detection** | ~15ms/crop | Ball carrier only |
| **BEV Transform** | <1ms/point | NumPy vectorized |
| **Total FPS** | 15-25 FPS | All components enabled |

### 6.2 메모리 사용량

| 컴포넌트 | 메모리 | 설명 |
|---------|--------|------|
| **YOLO Model** | ~50 MB | YOLOv8n detection + pose |
| **CLIP Model** | ~350 MB | ViT-B/32 on CUDA |
| **Player Crops** | ~10 MB | 22 players × 5 crops × 10 KB |
| **Player States** | ~1 MB | Path history (up to 1000 frames) |
| **Video Buffers** | ~50 MB | Input + 2 output streams |
| **Total** | ~460 MB | Reasonable for modern systems |

### 6.3 배치 처리

| 작업 | 배치 크기 | 주기 |
|------|---------|------|
| **CLIP Entity Classification** | 16 crops | Every 30 frames (1초) |
| **CLIP Team Classification** | 16 crops | Every 10 frames |
| **YOLO Detection** | 1 frame | Every frame |
| **Pose Detection** | 1 crop | Every frame (ball carrier only) |

### 6.4 Occlusion 처리

| 메트릭 | 값 | 설명 |
|--------|-----|------|
| **Detection 손실 허용** | 900 frames (30초) | ByteTrack buffer |
| **Occlusion 예측** | 30 frames (1초) | Linear prediction |
| **Path 표시** | Solid (active) / Dotted (occluded) | BEV visualization |

### 6.5 벤치마크 (예상)

| 시나리오 | FPS | 메모리 | 정확도 |
|---------|-----|--------|--------|
| **CLIP 비활성화** | 25-30 | ~110 MB | Medium (K-means only) |
| **CLIP 활성화 (현재)** | 15-25 | ~460 MB | High (semantic classification) |
| **고해상도 (4K)** | 5-10 | ~800 MB | Highest (but slower) |

---

## 7. 주요 설정 및 임계값

### 7.1 Detection Confidence

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `YOLO_CONFIDENCE` | 0.5 | 일반 탐지 기본값 (config.py) |
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.5 | 기존 코드 호환성 |
| **실제 사용 (detector_tracker.py):** | | |
| `conf` (YOLO tracking) | 0.25 | Detection threshold (낮음 = 새 ID 억제) |
| `person_filter` | 0.3 | Person conf >= 0.3만 포함 |
| `ball_filter` | 0.15 | Ball conf >= 0.15만 포함 (타원형 공) |

### 7.2 Tracking Parameters (ByteTrack)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `track_buffer` | 900 frames | 30초 동안 track 유지 (detection 없어도) |
| `new_track_thresh` | 0.5 | 새 track 생성 임계값 (높음 = 억제) |
| `match_thresh` | 0.5 | Track 매칭 임계값 (낮음 = 관대) |
| `track_high_thresh` | 0.3 | 1단계 association (낮음 = 포괄적) |
| `track_low_thresh` | 0.05 | 2단계 association (매우 낮음 = 공 포함) |
| `iou` | 0.2 | IoU threshold (낮음 = 관대한 매칭) |

### 7.3 CLIP Configuration

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `ENABLE_CLIP_CLASSIFICATION` | True | CLIP 활성화 여부 |
| `ENABLE_CLIP_ENTITY_FILTERING` | True | Sideline 인원 제외 |
| `ENABLE_CLIP_TEAM_CLASSIFICATION` | True | CLIP 팀 분류 사용 |
| `CLIP_MODEL_NAME` | 'ViT-B/32' | Fast model (ViT-B/16, ViT-L/14도 가능) |
| `CLIP_BATCH_SIZE` | 16 | 배치 처리 크기 |
| `CLIP_FRAME_INTERVAL` | 30 | 30프레임마다 1번 (1초) |
| `CLIP_CONFIDENCE_THRESHOLD` | 0.40 | 최소 신뢰도 (40%) |
| `CLIP_DEVICE` | 'cuda' / 'cpu' | PyTorch 자동 감지 |

### 7.4 Team Freeze (CLIP)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `freeze_threshold` | 1 frame | **즉시 고정** (1프레임 후) |
| `confidence_threshold` | 0.3 | 고정 신뢰도 (30% 이상) |
| `classification_interval` | 10 frames | main.py에서 10프레임마다 분류 |
| `FREEZE_TEAM_ASSIGNMENT` | True | 팀 고정 활성화 (config.py) |
| `SHOW_FROZEN_INDICATOR` | True | 녹색 테두리 표시 |

### 7.5 BEV Field Boundaries

| 파라미터 | 값 (pixels) | 설명 |
|---------|------------|------|
| `BEV_FIELD_X_MIN` | 50 | 왼쪽 사이드라인 마진 |
| `BEV_FIELD_X_MAX` | 950 | 오른쪽 사이드라인 마진 |
| `BEV_FIELD_Y_MIN` | 50 | 상단 엔드존 마진 |
| `BEV_FIELD_Y_MAX` | 450 | 하단 벤치/사이드라인 마진 |
| **Canvas Size** | 1000 × 500 | 전체 BEV 캔버스 크기 |
| **On-Field Area** | 900 × 400 | 실제 경기장 영역 |

### 7.6 Ball Validation

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `BALL_CONFIDENCE` | 0.2 | config.py 기본값 (미사용) |
| **Actual filter** | 0.15 | detector_tracker.py 실제 사용 |
| `min_width` | 5 px | 최소 공 크기 |
| `max_width` | 300 px | 최대 공 크기 (false positive 제거) |
| `min_height` | 5 px | 최소 높이 |
| `max_height` | 300 px | 최대 높이 |

### 7.7 Team Colors (Visualization)

| 팀 | BGR 색상 | RGB 표현 |
|----|---------|---------|
| **Team A** | (255, 0, 0) | Blue |
| **Team B** | (0, 0, 255) | Red |
| **Referee** | (0, 255, 255) | Yellow |
| **Unknown** | (128, 128, 128) | Gray |
| **Ball** | (0, 255, 255) | Yellow |
| **Ball Carrier** | (0, 0, 255) | Red |

### 7.8 Yard Calculation

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `PIXELS_PER_YARD_BEV` | 20 | 1 yard = 20 pixels in BEV |
| `BEV_LEFT_YARD_LINE` | 0 | 왼쪽 가장자리 yard line |
| `BEV_DIRECTION` | 'LEFT_TO_RIGHT' | 공격 방향 |
| **Field Width** | 50 yards | 1000 px / 20 = 50 yards |

### 7.9 Logging Intervals

| 작업 | 주기 (frames) | 설명 |
|------|--------------|------|
| `DETECTION_EVERY_N_FRAMES` | 90 | 상태 로깅 (90프레임마다) |
| CLIP entity classification | 30 | Entity 분류 주기 |
| CLIP team classification | 10 | 팀 분류 주기 (main.py) |
| Progress print | 30 | 진행상황 출력 |

---

## 8. State Machine

### 8.1 State Diagram

```
┌─────────────┐
│  PRE_SNAP   │ ← 초기 상태 (ball carrier 미감지)
└──────┬──────┘
       │
       │ Trigger: ball_carrier 최초 감지
       │ (person과 ball의 BEV 거리 최소)
       ↓
┌─────────────────┐
│  PLAY_ACTIVE    │ ← play_type = 'RUN' (기본값)
└──┬───────────┬──┘
   │           │
   │           │ Trigger: ball이 carrier를 떠남
   │           │ (ball과 carrier 거리 > threshold)
   │           ↓
   │      ┌──────────────┐
   │      │ BALL_IN_AIR  │ ← play_type = 'PASS' (자동 변경)
   │      └───────┬──────┘
   │              │
   │              │ Trigger: 새로운 player가 ball 획득
   │              │ (새 carrier 식별)
   │              ↓
   │         (PLAY_ACTIVE로 복귀)
   │
   │ Trigger: carrier의 무릎이 발목 아래
   │ (pose detection: min_knee_y > max_ankle_y)
   ↓
┌─────────────────┐
│  PLAY_ENDED ✓   │ ← yards 계산, tackler 식별, 루프 종료
└─────────────────┘
```

### 8.2 State Transition Table

| From State | To State | Trigger | Actions |
|-----------|----------|---------|---------|
| **PRE_SNAP** | PLAY_ACTIVE | Ball carrier detected | `play_type = 'RUN'`<br>`ball_carrier_id = track_id` |
| **PLAY_ACTIVE** | BALL_IN_AIR | Ball leaves carrier | `play_type = 'PASS'`<br>Track ball separately |
| **BALL_IN_AIR** | PLAY_ACTIVE | New player catches ball | Update `ball_carrier_id` |
| **PLAY_ACTIVE** | PLAY_ENDED | Knee below ankle (pose) | Calculate yards<br>Find tackler<br>Break loop |

### 8.3 State-Specific Operations

#### PRE_SNAP
- **활동:** Track all players, collect team colors
- **조건:** `ball_carrier_id == None`
- **종료:** Ball carrier 최초 감지

#### PLAY_ACTIVE
- **활동:**
  - Track player paths
  - Store player crops (jersey OCR용)
  - Pose detection on ball carrier
  - Update BEV visualization
- **조건:** `ball_carrier_id != None`
- **종료:** Ball in air OR play ended

#### BALL_IN_AIR
- **활동:**
  - Track ball separately
  - Monitor all players for catch
- **조건:** Ball distance > carrier threshold
- **종료:** New carrier detected OR play ended

#### PLAY_ENDED
- **활동:**
  - Calculate yards gained
  - Identify tackler (closest to carrier)
  - Generate JSON output
- **조건:** Knee < ankle (pose detection)
- **종료:** Break main loop

---

## 9. 알려진 제약사항

### 9.1 기능적 제약

| # | 제약사항 | 설명 | 해결 방안 |
|---|---------|------|----------|
| 1 | **Play Type 항상 RUN** | PASS 감지 로직 완전 제거됨 | `BALL_IN_AIR` state 재활성화 필요 |
| 2 | **Single Play 처리** | 한 번에 하나의 플레이만 처리 | State machine reset 로직 추가 |
| 3 | **Track ID Fragmentation** | 22명 → 50-100개 ID 생성 | **Track ID Mapper 구현 필요 (최우선)** |
| 4 | **포지션 추론 없음** | QB/RB/WR 자동 인식 안 됨 | Formation pattern analysis 추가 |
| 5 | **Secondary Tackler 없음** | 가장 가까운 defender만 식별 | Multi-defender proximity check |

### 9.2 기술적 제약

| # | 제약사항 | 설명 | 해결 방안 |
|---|---------|------|----------|
| 6 | **Homography 필수** | `calibrate_homography.py` 먼저 실행 필요 | Auto-calibration (field line detection) |
| 7 | **측면 촬영 한계** | Pose detection 정확도 제한 | Multi-angle camera support |
| 8 | **등번호 OCR 낮은 정확도** | 측면/가려진 경우 실패 | Ensemble voting (5 crops) |
| 9 | **Team Freeze 영구적** | 한 번 고정되면 수정 불가 | Manual override UI |
| 10 | **Manual Metadata** | CLI 인자 없음, 코드 수정 필요 | Config file or CLI parser |

### 9.3 성능 제약

| # | 제약사항 | 영향 | 최적화 방안 |
|---|---------|------|-----------|
| 11 | **CLIP 오버헤드** | ~50ms per 16 crops | Batch size 증가, Interval 조정 |
| 12 | **4K 영상 느림** | 5-10 FPS | Downscale to 1080p |
| 13 | **CPU 모드 매우 느림** | <5 FPS | GPU 권장 (CUDA) |

---

## 10. 데이터 구조

### 10.1 Track Dictionary

```python
track = {
    'bbox': [x1, y1, x2, y2],      # float - Bounding box 좌표
    'track_id': int,                # Persistent tracking ID (ByteTrack)
    'class_id': int,                # 0=person, 32=ball (COCO classes)
    'confidence': float,            # Detection confidence (0.0-1.0)

    # Optional (if CLIP enabled)
    'entity_type': str,             # 'player', 'referee', 'sideline', 'other'
    'entity_confidence': float      # CLIP confidence (0.0-1.0)
}
```

### 10.2 Player State

```python
player_states = {
    track_id: {
        'path': [
            (x_bev, y_bev, 'active'),     # 실제 detection
            (x_bev, y_bev, 'active'),
            (x_bev, y_bev, 'occluded'),   # 예측된 위치
            ...
        ],
        'last_known_position': (x_bev, y_bev),
        'last_seen_frame': int           # 마지막 detection 프레임
    }
}
```

### 10.3 Play Summary

```python
summary = {
    'playType': 'RUN',               # Always 'RUN' (PASS 비활성화)
    'gainYard': float,               # BEV distance / PIXELS_PER_YARD_BEV
    'car_id': int,                   # Ball carrier track_id
    'passer_id': int,                # Play 시작 시 ball carrier
    'tkl_id': int,                   # Tackler track_id (closest to carrier)
    'state': 'PLAY_ENDED'            # Final state
}
```

### 10.4 JSON Output Schema

```json
{
  "gameKey": "GAME_2024_001",
  "gameDate": "2024-01-15",
  "homeTeam": "Home Team",
  "awayTeam": "Away Team",
  "stadium": "Stadium Name",
  "weather": "Clear",
  "temperature": "72F",
  "Clips": [
    {
      "clipKey": "CLIP_001",

      "// ===== AI-Derived Fields ===== //": null,
      "playType": "RUN",
      "gainYard": 5.2,

      "carrierInfo": {
        "trackId": 3,
        "jerseyNumber": "23"
      },
      "passerInfo": {
        "trackId": 3,
        "jerseyNumber": "Unknown"
      },
      "tacklerInfo": {
        "trackId": 8,
        "jerseyNumber": "Unknown"
      },

      "// ===== Manual Metadata ===== //": null,
      "quarter": 1,
      "time": "12:00",
      "down": 1,
      "toGo": 10,
      "yardLine": 25,
      "offensiveTeam": "Home Team",
      "defensiveTeam": "Away Team",
      "offensiveFormation": "Unknown",
      "defensiveFormation": "Unknown",
      "personnel": "Unknown",
      "playResult": "Unknown",
      "tackler": "Unknown"
    }
  ]
}
```

### 10.5 Team Classification

```python
# Color-based (K-means)
team_classifier.player_teams = {
    track_id: 'Team A',    # Blue jersey cluster
    track_id: 'Team B',    # Red jersey cluster
    track_id: 'Referee',   # Low saturation (striped)
    track_id: 'Others'     # Smallest cluster (<20% samples)
}

# CLIP-based (with freeze)
clip_team_classifier.player_teams = {
    track_id: 'Team A',    # "yellow jersey + white pants"
    track_id: 'Team B',    # "white jersey + black pants"
    track_id: 'Referee'    # "striped shirt"
}

clip_team_classifier.frozen_teams = {
    track_id: 'Team A',    # 🔒 Immutable after 1st detection
    ...
}
```

### 10.6 CLIP Entity Cache

```python
detector_tracker.entity_cache = {
    track_id: ('player', 0.85),     # (entity_type, confidence)
    track_id: ('referee', 0.92),
    track_id: ('sideline', 0.78),   # Will be filtered out
    ...
}
```

---

## 11. 의존성

### 11.1 requirements.txt

```
ultralytics          # YOLOv8 detection + pose
opencv-python        # cv2 for video processing
numpy                # Numerical computations
scipy                # Euclidean distance
pytesseract          # Tesseract OCR wrapper
scikit-learn         # K-means clustering
torch                # PyTorch (CLIP backend)
torchvision          # Vision utilities
pillow               # PIL Image (CLIP preprocessing)
git+https://github.com/openai/CLIP.git  # OpenAI CLIP model
```

### 11.2 시스템 요구사항

| 컴포넌트 | 최소 | 권장 |
|---------|------|------|
| **Python** | 3.8+ | 3.10+ |
| **CUDA** | 11.0+ (optional) | 12.0+ |
| **GPU VRAM** | 2 GB | 4 GB+ |
| **RAM** | 4 GB | 8 GB+ |
| **Disk** | 2 GB (models) | 10 GB+ (videos) |
| **Tesseract** | 4.0+ | 5.0+ |

### 11.3 외부 파일

| 파일 | 생성 방법 | 필수 여부 |
|------|----------|----------|
| `homography_matrix.npy` | `python calibrate_homography.py` | ✅ 필수 |
| `yolov8n.pt` | Auto-download (first run) | ✅ 필수 |
| `yolov8n-pose.pt` | Auto-download (first run) | ✅ 필수 |
| `CLIP ViT-B/32` | Auto-download (first run) | Optional (if CLIP enabled) |

---

## 12. 아키텍처 비교

### 12.1 Main Pipeline (main.py + config.py) ✅ **PRIMARY**

**특징:**
- 모듈형 컴포넌트 아키텍처
- CLIP 기반 팀 분류 (1-frame freeze)
- BEV boundary 필터링 (경기장 밖 제외)
- 포괄적인 문서화 (CLAUDE.md)
- 4-state machine
- 10프레임마다 팀 분류

**파일:**
- `main.py` (313줄)
- `config.py` (209줄)
- `detector_tracker.py`, `play_analyzer.py`, `clip_classifier.py`, etc.

### 12.2 Alternative Pipeline (tracker.py + tracker_config.py)

**특징:**
- **Stadium masking** (HSV green field detection)
- **ROI filtering** (상하단 제외)
- **Persistent dots with fade** (시각적 잔상 효과)
- **Team assignment freezing** (깜빡임 방지)
- 더 공격적인 tracking 설정
- Field recognition (HSV color space)

**차이점:**

| 기능 | Main Pipeline | Alternative Pipeline |
|------|--------------|---------------------|
| Entity Filtering | CLIP semantic | HSV green field detection |
| Team Classification | CLIP (1-frame freeze) | K-means + freeze |
| ROI Masking | BEV boundary only | ROI + Stadium mask |
| Visual Effects | Standard paths | Persistent dots + fade |
| Documentation | Comprehensive | Minimal |
| Usage | ✅ Active | ❌ Not used by main.py |

**사용 시기:**
- **Main Pipeline:** 표준 플레이 분석 (일반적 사용)
- **Alternative Pipeline:** 사이드라인 필터링이 중요하거나 시각 효과 필요 시

---

## 13. 중요 파일

### 13.1 코어 파일

| 파일 | 크기 | 역할 |
|------|------|------|
| **CLAUDE.md** | 17.6 KB | 포괄적인 시스템 문서 (개발 가이드) |
| **config.py** | 9.4 KB | 중앙 설정 허브 (200+ 설정) |
| **clip_classifier.py** | 444 lines | CLIP entity/team 분류 + freeze |
| **play_analyzer.py** | 420 lines | State machine + play 분석 |
| **detector_tracker.py** | 206 lines | YOLOv8 + ByteTrack |
| **bytetrack_extended.yaml** | 11 lines | ByteTrack 설정 (track_buffer: 900) |

### 13.2 입출력 파일

| 파일 | 경로 | 타입 |
|------|------|------|
| **Input Video** | `input/video.mp4` | Video |
| **Homography Matrix** | `homography_matrix.npy` | NumPy array (3×3) |
| **Annotated Output** | `output/result.mp4` | Video |
| **BEV Output** | `output/bev.mp4` | Video |
| **JSON Summary** | `output/clip_summary.json` | JSON |

### 13.3 모델 파일 (Auto-downloaded)

| 모델 | 경로 | 크기 |
|------|------|------|
| **YOLOv8n** | `yolov8n.pt` | ~6 MB |
| **YOLOv8n-pose** | `yolov8n-pose.pt` | ~6 MB |
| **CLIP ViT-B/32** | `~/.cache/clip/` | ~350 MB |

---

## 14. 핵심 메트릭

### 14.1 Homography & BEV

| 메트릭 | 값 | 설명 |
|--------|-----|------|
| **Homography Matrix** | 3×3 | Perspective transform (camera → BEV) |
| **BEV Canvas Size** | 1000 × 500 px | 전체 캔버스 |
| **Calibration Points** | 4 points | Maps to [[0,500], [1000,500], [1000,0], [0,0]] |
| **Pixels per Yard** | 20 px/yd | 1000 px / 50 yards = 20 |
| **Field Width (BEV)** | 50 yards | 1000 px / 20 = 50 yards |
| **Field Height (BEV)** | 25 yards | 500 px / 20 = 25 yards |

### 14.2 Yard Line Calculation

| 메트릭 | 범위 | 설명 |
|--------|------|------|
| **Yard Line Range** | 0-100 | OWN: 0-50, OPP: 50-100 |
| **BEV_LEFT_YARD_LINE** | 0 | 왼쪽 가장자리 기준점 |
| **BEV_DIRECTION** | 'LEFT_TO_RIGHT' | 공격 방향 |

### 14.3 Detection & Tracking

| 메트릭 | 값 | 설명 |
|--------|-----|------|
| **Detection Loss Tolerance** | 900 frames (30초) | ByteTrack buffer |
| **Expected Players** | 22 | 11 vs 11 |
| **Actual Track IDs** | 50-100 | ⚠️ Fragmentation issue |
| **Target Track IDs** | 22-30 | With Track ID Mapper |

### 14.4 COCO Keypoints

| Keypoint | Index | Usage |
|----------|-------|-------|
| Left Knee | 13 | Play end detection |
| Right Knee | 14 | Play end detection |
| Left Ankle | 15 | Play end detection |
| Right Ankle | 16 | Play end detection |
| Hips | 11, 12 | Posture analysis (optional) |
| Shoulders | 5, 6 | Posture analysis (optional) |

### 14.5 처리 통계 (예상)

| 메트릭 | 값 |
|--------|-----|
| **평균 FPS** | 20 FPS (GPU) |
| **플레이당 프레임** | 150-300 frames (5-10초) |
| **플레이당 처리 시간** | 8-15초 (GPU) |
| **CLIP 실행 횟수/플레이** | 5-10회 (30-frame interval) |
| **Pose 실행 횟수/플레이** | 150-300회 (ball carrier only) |

---

## 📝 추가 참고 자료

- **CLAUDE.md**: 전체 시스템 문서 (Quick Reference, 디버깅 팁)
- **explain.md**: 현황 요약 (Track ID Fragmentation 문제)
- **requirements.txt**: 전체 의존성 목록
- **bytetrack_extended.yaml**: ByteTrack 세부 설정

---

## 🎯 핵심 결론

### 시스템 강점
✅ 실시간 처리 가능 (10-30 FPS)
✅ 정확한 야드 계산 (BEV transformation)
✅ 의미적 분류 (CLIP entity/team)
✅ 강력한 occlusion 처리 (900-frame buffer)
✅ 모듈형 아키텍처 (확장 용이)

### 최우선 개선 과제
🔴 **Track ID Fragmentation** (22명 → 50-100개 ID)
→ **Track ID Mapper** (CLIP embedding + 공간 근접도) 구현 필요

### 성능 목표
- Track ID 개수: **85% 감소** (50-100 → 22-30)
- 팀 분류 정확도: **향상** (중복 제거)
- JSON 데이터 일관성: **확보** (unique players)

---

**문서 버전:** 1.0
**생성일:** 2025-11-03
**프로젝트:** Football Play Analysis System
**저자:** Technical Analysis (Claude Code)
