"""
================================================================================================
Football Play Analysis System - 통합 설정 파일
================================================================================================

이 파일은 미식축구 경기 영상 분석 시스템의 모든 설정값을 관리합니다.

【프로젝트 개요】
- 목적: 미식축구 경기 영상에서 선수/심판/공을 자동 감지하고 추적하여 플레이 통계 생성
- 핵심 기술:
  1. YOLO11n: 객체 탐지 (선수, 공)
  2. ByteTrack: 다중 객체 추적 (ID 안정성)
  3. Jersey OCR: 등번호 인식으로 ID 안정화
  4. CLIP: 시맨틱 팀 분류 (Team A/B/Referee)
  5. Homography: Bird's Eye View 변환 (야드 계산)

【데이터 흐름】
Video → YOLO Detection → ByteTrack → Jersey OCR → CLIP Team Classification → BEV Transform → Output

【파일 구조】
- config.py (이 파일): 모든 설정값
- detector_tracker.py: YOLO + ByteTrack + Jersey OCR
- team_classifier.py: CLIP 기반 팀 분류
- play_analyzer.py: 플레이 상태 머신
- visualizer.py: 결과 시각화
- main.py: 메인 파이프라인

================================================================================================
"""

# ============================================================================
# 📁 VIDEO I/O (입출력 파일 경로)
# ============================================================================

VIDEO_INPUT_PATH = 'input/video.mp4'         # 입력 비디오 파일
VIDEO_OUTPUT_PATH = 'output/result.mp4'      # 결과 비디오 (박스 + 팀 라벨)
BEV_OUTPUT_PATH = 'output/bev.mp4'           # Bird's Eye View 비디오 (전술 맵)
JSON_OUTPUT_PATH = 'output/clip_summary.json'  # 플레이 통계 JSON

# Video output quality
OUTPUT_CODEC = 'avc1'    # H.264 codec (모든 플랫폼 호환)
OUTPUT_QUALITY = 100     # 0-100 (100 = 최고 화질, 파일 크기 큼)


# ============================================================================
# 🎯 YOLO DETECTION (객체 탐지)
# ============================================================================
#
# 【YOLO11n을 선택한 이유】
# - YOLOv8n 대비 22% 적은 파라미터 (더 빠름)
# - 작은 객체 탐지 성능 개선 (멀리 있는 선수/공)
# - COCO 데이터셋 사전 학습 (person=0, sports ball=32)
#
# 【대안 모델 비교】
# - YOLO11s: 더 정확하지만 느림 (권장하지 않음)
# - RT-DETR: Transformer 기반, 정확하지만 매우 느림
# - Custom fine-tuned model: 가장 정확하지만 학습 데이터 필요
#
# ============================================================================

DETECTION_MODEL_PATH = 'yolo11n.pt'       # YOLO11 Nano (경량, 빠름)
POSE_MODEL_PATH = 'yolo11n-pose.pt'       # Pose estimation (플레이 종료 감지용)

# Detection confidence thresholds
# 【왜 0.3인가?】
# - 너무 낮으면 (0.1): 오탐지 많음 (그림자, 광고판 등)
# - 너무 높으면 (0.6): 가려진 선수 놓침
# - 0.3 = 균형점 (ByteTrack이 낮은 신뢰도 객체도 추적 가능)
YOLO_CONFIDENCE = 0.3
DETECTION_CONFIDENCE_THRESHOLD = 0.3  # 기존 코드 호환성

# YOLO Tracking parameters
# 【ByteTrack 설정 파일】
# - bytetrack_extended.yaml에서 세부 파라미터 관리
# - track_buffer=90: 가려진 선수를 3초간 추적 유지
# - match_thresh=0.70: IoU 0.7 이상이면 같은 선수로 판단 (관대함)
DETECTION_EVERY_N_FRAMES = 90  # 상태 로깅 간격 (실제로는 매 프레임 탐지)

# Class IDs (COCO dataset 표준)
CLASS_ID_PERSON = 0   # 사람 클래스
CLASS_ID_BALL = 32    # 스포츠 공 클래스

# ============================================================================
# ⚽ BALL DETECTION (공 탐지)
# ============================================================================
#
# 【공 탐지의 어려움】
# - YOLO의 sports ball 클래스는 축구공/농구공 위주로 학습됨
# - 미식축구공은 타원형이라 탐지 어려움
# - 멀리 있으면 5픽셀 이하로 작아짐
# - 손에 가려지거나 선수들 사이에 끼면 안 보임
#
# 【현재 성능】
# - YOLO11n: 0.2% 탐지율 (1000프레임 중 2프레임)
# - YOLOv8n: 0.1% 탐지율 (1000프레임 중 1프레임)
# → 실용적이지 않음, 별도 fine-tuning 필요
#
# 【개선 방법】
# 1. Fine-tuned model: 미식축구공 데이터셋으로 재학습 (권장)
# 2. 전용 ball tracker: TrackNet, BallTrack 등 사용
# 3. Multi-scale detection: 다양한 크기에서 탐지
#
# ============================================================================

ENABLE_BALL_DETECTION = True
BALL_CONFIDENCE = 0.10          # 공 탐지 신뢰도 (낮게 설정 = 더 많이 탐지)
BALL_MIN_SIZE = 5               # 최소 공 크기 (픽셀) - 멀리 있는 공
BALL_MAX_SIZE = 150             # 최대 공 크기 (픽셀) - 매우 가까운 공
BALL_ASPECT_RATIO_MIN = 0.5     # 최소 종횡비 (0.5 = 매우 타원형)
BALL_ASPECT_RATIO_MAX = 3.0     # 최대 종횡비 (3.0 = 매우 길쭉함)
BALL_COLOR = (0, 255, 255)      # 공 바운딩 박스 색상 (노란색, BGR)
BALL_CARRIER_COLOR = (0, 165, 255)  # 볼 캐리어 박스 색상 (주황색, BGR)


# ============================================================================
# 🏟️ STADIUM/FIELD RECOGNITION (경기장 인식)
# ============================================================================
#
# 【왜 필요한가?】
# - 사이드라인의 코치, 카메라맨, 관중을 제외해야 함
# - BEV 변환 없이 빠르게 필터링 가능
# - HSV 색공간에서 녹색 필드 감지
#
# 【동작 원리】
# 1. HSV 변환: RGB → HSV (색상 분리)
# 2. 녹색 범위 마스킹: inRange(FIELD_HSV_LOWER, FIELD_HSV_UPPER)
# 3. Morphology: 노이즈 제거 (opening + closing)
# 4. 경계 침식: 경기장 경계 안쪽만 선택
#
# 【대안】
# - BEV 기반 필터링: 더 정확하지만 homography 필요
# - CLIP 기반 필터링: 매우 정확하지만 느림
# - Hybrid (현재 방식): Stadium masking (빠름) + CLIP (정확함)
#
# ============================================================================

ENABLE_STADIUM_MASKING = True  # 경기장 밖 사람 제외

# Green field detection (HSV range)
# H (Hue): 색상 (0-180, 35-85 = 녹색)
# S (Saturation): 채도 (0-255, 40-255 = 선명함)
# V (Value): 명도 (0-255, 40-255 = 밝기)
FIELD_HSV_LOWER = (35, 40, 40)      # 녹색 필드 하한
FIELD_HSV_UPPER = (85, 255, 255)    # 녹색 필드 상한

# Morphological operations (노이즈 제거)
MORPH_KERNEL_SIZE = 15   # 커널 크기 (클수록 큰 노이즈 제거)
MORPH_ITERATIONS = 2     # 반복 횟수 (많을수록 강력)

# Field boundary settings
MIN_FIELD_AREA_PERCENT = 25.0    # 최소 필드 면적 (전체 화면의 %)
FIELD_BOUNDARY_EROSION = 10      # 경계 침식 (픽셀) - 경기장 밖 확실히 제외


# ============================================================================
# 📐 ROI MASKING (관심 영역)
# ============================================================================
#
# 【왜 필요한가?】
# - 상단: 스코어보드, 관중석
# - 하단: 광고판, 카메라 장비
# → 이 영역은 경기와 무관하므로 제외
#
# ============================================================================

ROI_TOP_PERCENT = 0.20      # 상단 20% 제외
ROI_BOTTOM_PERCENT = 0.10   # 하단 10% 제외


# ============================================================================
# 👥 TEAM CLASSIFICATION (팀 분류)
# ============================================================================
#
# 【CLIP을 선택한 이유】
# - HSV 색상만으로는 불충분 (조명 변화, 그림자, 비슷한 색상)
# - CLIP: Vision-Language Model로 시맨틱 이해
#   - "yellow jersey with white helmet" vs "white jersey with black helmet"
#   - "referee in striped shirt" (줄무늬 패턴 인식)
# - 정확도: HSV 80% → CLIP 95%+
#
# 【CLIP 모델 선택】
# - ViT-B/32: 빠름, 정확도 양호 (현재 사용)
# - ViT-B/16: 중간, 더 정확
# - ViT-L/14: 느림, 가장 정확
#
# 【Hybrid 방식】
# 1. HSV로 1차 분류 (빠름)
# 2. CLIP로 2차 검증 (정확함)
# 3. 한번 분류되면 고정 (FREEZE_TEAM_ASSIGNMENT)
#
# ============================================================================

# Team classification method
TEAM_DETECTION_METHOD = 'adaptive_clustering'  # 자동 K-means 클러스터링
                                                # 'fixed_range' = 수동 HSV 범위 설정

MIN_PLAYERS_FOR_CLUSTERING = 6  # 최소 선수 수 (이하면 fixed_range로 폴백)
NUM_TEAM_CLUSTERS = 3           # Team A, Team B, Referee

# Team colors for visualization (BGR format)
# 【왜 BGR인가?】
# - OpenCV는 BGR 순서 사용 (RGB 아님!)
# - (B, G, R) = (파랑, 녹색, 빨강)
TEAM_A_COLOR = (255, 0, 0)      # Blue (파란색)
TEAM_B_COLOR = (0, 0, 255)      # Red (빨간색)
REFEREE_COLOR = (0, 255, 255)   # Yellow (노란색)
UNKNOWN_COLOR = (128, 128, 128) # Gray (회색 - 팀 미배정)
PLAYER_COLOR = (255, 0, 0)      # Blue (기존 코드 호환성)

# Fixed HSV ranges (backup for manual classification)
# 【실제 유니폼 색상】
# Team A: 노란색 저지 + 흰색 헬멧 + 흰색 바지
# Team B: 흰색 저지 + 검은색 헬멧 + 검은색 바지
# Referee: 검은색/흰색 줄무늬 셔츠 + 검은색 바지
#
# 【HSV 튜닝 팁】
# - H (Hue): 색상의 종류 (Red=0, Yellow=30, Green=60, Blue=120)
# - S (Saturation): 채도 (0=무채색, 255=선명한 색)
# - V (Value): 밝기 (0=검정, 255=흰색)
TEAM_A_HSV_RANGE = ((18, 60, 100), (35, 255, 255))    # Yellow (노란색)
TEAM_B_HSV_RANGE = ((0, 0, 160), (180, 40, 255))      # White (흰색)
REFEREE_HSV_RANGE = ((0, 0, 40), (180, 35, 170))      # Black/White stripe


# ============================================================================
# 🎯 OBJECT TRACKING (객체 추적)
# ============================================================================
#
# 【ByteTrack를 선택한 이유】
# - SOTA multi-object tracker (2021)
# - 2-stage association: high confidence + low confidence
# - Kalman filter로 가려진 객체 예측
# - BoT-SORT 시도했으나 Ultralytics config 오류로 실패
#
# 【ByteTrack 동작 원리】
# 1. High confidence detections (0.35+): 확실한 탐지
# 2. Low confidence detections (0.20-0.35): 가려진 객체
# 3. Track buffer (90프레임): 3초간 미탐지 허용
# 4. Match threshold (0.70): IoU 0.7 이상이면 같은 객체
#
# 【ID 안정성 개선 과정】
# - Phase 1: YOLOv8 + ByteTrack → 133 IDs (6.0x)
# - Phase 2: YOLO11 + optimized ByteTrack → 75 IDs (3.4x)
# - Phase 3: + Jersey OCR mapping → 73 IDs (3.3x)
# - 목표: 30-40 IDs (1.5-2.0x)
#
# 【설정 파일】
# - bytetrack_extended.yaml: 세부 파라미터
#   - track_high_thresh: 0.35 (1차 매칭)
#   - track_low_thresh: 0.20 (2차 매칭, 가려진 객체)
#   - new_track_thresh: 0.50 (새 track 생성 임계값)
#   - track_buffer: 90 (3초간 유지)
#   - match_thresh: 0.70 (IoU 임계값, 높을수록 관대)
#
# ============================================================================

ENABLE_TRACKING = True              # 객체 추적 활성화
MAX_TRACKING_FRAMES = 300           # Kalman filter 예측 최대 프레임 (10초 at 30fps)
TRACKING_IOU_THRESHOLD = 0.15       # IoU 임계값 (낮을수록 관대)

# Team assignment freezing (팀 고정)
# 【왜 필요한가?】
# - CLIP이 매 프레임마다 다르게 분류할 수 있음
# - 한번 Team A로 분류되면 절대 바뀌면 안 됨!
# - 2프레임 연속 같은 팀 → 고정
FREEZE_TEAM_ASSIGNMENT = True       # 팀 고정 활성화
TEAM_ASSIGNMENT_CONFIDENCE = 2      # 2프레임 후 팀 고정

# Visualization and debugging
SHOW_FROZEN_INDICATOR = True        # 고정된 선수에게 표시
PRINT_FREEZE_EVENTS = True          # 팀 고정 시 콘솔 출력
PRINT_TRACKING_STATS = True         # 추적 통계 출력
DEBUG_TEAM_COLORS = False           # 팀 색상 디버그 (HSV 값 출력)


# ============================================================================
# 🗺️ BIRD'S EYE VIEW (BEV) - 전술 맵
# ============================================================================
#
# 【Homography 변환이란?】
# - 카메라 시점 → 위에서 본 시점 (2D 평면 투영)
# - 4개의 점 매칭으로 변환 행렬 계산
# - 야드 계산, 플레이어 위치 추적에 필수
#
# 【Calibration 방법】
# 1. python calibrate_homography.py 실행
# 2. 경기장 라인 4개 점 클릭 (예: 해시 마크 코너)
# 3. homography_matrix.npy 파일 생성
# 4. 이 행렬로 모든 프레임 변환
#
# 【BEV 캔버스 크기】
# - 1000 x 500 픽셀 (2:1 비율)
# - PIXELS_PER_YARD_BEV = 20 → 50 yards wide (1000/20=50)
#
# 【BEV 필터링 확장】
# - 이전: Y=50-450 (너무 좁아서 6-14명만 탐지)
# - 현재: Y=0-500 (전체 필드, CLIP이 사이드라인 필터링)
# - 결과: 25-30명 탐지 (22명 선수 + 심판 + 교체 선수)
#
# ============================================================================

# Tactical map dimensions
FIELD_WIDTH = 400           # BEV 캔버스 너비 (픽셀)
FIELD_HEIGHT = 600          # BEV 캔버스 높이 (픽셀)
FIELD_LENGTH_YARDS = 120    # 실제 필드 길이 (야드)
FIELD_WIDTH_YARDS = 53.33   # 실제 필드 너비 (야드)

# Persistent dots (잔상 효과)
# 【왜 필요한가?】
# - 선수의 이동 경로 시각화
# - 전술 분석에 유용
PERSISTENT_DOTS = True      # 점이 축적됨
DOT_FADE_ALPHA = 0.70       # 페이드 속도 (0.70 = 매우 빠른 페이드)
                            # 0.98 = 느린 페이드 (많은 잔상)
                            # 0.85 = 빠른 페이드
                            # 0.70 = 매우 빠른 페이드 (현재)

# Homography settings
HOMOGRAPHY_MATRIX_PATH = 'homography_matrix.npy'
PIXELS_PER_YARD_BEV = 20  # BEV 픽셀당 야드 (1000 / 50 = 20)

# BEV field configuration
BEV_LEFT_YARD_LINE = 0         # 왼쪽 기준 야드 라인
BEV_DIRECTION = 'LEFT_TO_RIGHT'  # 공격 방향

# BEV field boundaries (⭐ EXPANDED for multi-detection)
BEV_FIELD_X_MIN = 50      # 왼쪽 사이드라인 마진 (픽셀)
BEV_FIELD_X_MAX = 950     # 오른쪽 사이드라인 마진 (픽셀)
BEV_FIELD_Y_MIN = 0       # 상단 엔드존 마진 (픽셀) - EXPANDED: 50 → 0
BEV_FIELD_Y_MAX = 500     # 하단 엔드존 마진 (픽셀) - EXPANDED: 450 → 500


# ============================================================================
# 📏 DISTANCE TRACKING (거리 측정)
# ============================================================================

ENABLE_DISTANCE_TRACKING = True  # 야드 측정 활성화


# ============================================================================
# 🤖 CLIP CONFIGURATION (OpenAI CLIP)
# ============================================================================
#
# 【CLIP이란?】
# - OpenAI의 Vision-Language Model
# - 이미지 + 텍스트를 같은 공간에 임베딩
# - "yellow jersey player" ↔ 이미지 유사도 계산
#
# 【왜 CLIP인가?】
# 1. Zero-shot learning: 재학습 없이 즉시 사용
# 2. 시맨틱 이해: 색상뿐 아니라 패턴, 형태 이해
# 3. 높은 정확도: HSV 80% → CLIP 95%+
#
# 【CLIP 사용처】
# 1. Entity filtering: player vs coach vs photographer
# 2. Team classification: Team A vs Team B vs Referee
# 3. Referee detection: 줄무늬 패턴 인식 (HSV로 불가능)
#
# 【Prompt Engineering】
# - 구체적일수록 정확함
# - "a player" (X) → "an american football player in full uniform actively playing" (O)
# - "referee" (X) → "referee in black and white striped uniform" (O)
#
# 【성능 최적화】
# - Batch processing: 16개 crop을 한번에 처리
# - Frame interval: 30프레임마다 실행 (1초에 1번)
# - Team freezing: 한번 분류되면 재분류 안 함
#
# ============================================================================

# Enable CLIP classification
ENABLE_CLIP_CLASSIFICATION = True       # CLIP 전체 활성화
ENABLE_CLIP_ENTITY_FILTERING = True     # 사이드라인 인원 필터링
ENABLE_CLIP_TEAM_CLASSIFICATION = True  # 팀 분류

# CLIP device (GPU 사용 가능 시 자동 감지)
try:
    import torch as _torch_module
    CLIP_DEVICE = 'cuda' if _torch_module.cuda.is_available() else 'cpu'
except ImportError:
    CLIP_DEVICE = 'cpu'
    print("Warning: PyTorch not found. CLIP will use CPU (slower).")

# CLIP model settings
# 【모델 선택】
# - ViT-B/32: 88M params, 빠름, 정확도 양호 (권장)
# - ViT-B/16: 149M params, 중간, 더 정확
# - ViT-L/14: 427M params, 느림, 가장 정확
CLIP_MODEL_NAME = 'ViT-B/32'  # 현재 사용

# CLIP processing settings
CLIP_BATCH_SIZE = 16         # Batch 크기 (GPU 메모리에 따라 조정)
CLIP_FRAME_INTERVAL = 30     # 30프레임마다 CLIP 실행 (1초에 1번 at 30fps)
CLIP_CONFIDENCE_THRESHOLD = 0.50  # 최소 신뢰도 (0.50 = 50% 이상만 분류)

# Entity classification prompts
# 【Prompt 설계 원칙】
# 1. 구체적으로: "player" → "football player in full uniform"
# 2. 맥락 추가: "actively playing on the field during a game"
# 3. 차별점 강조: "referee in striped uniform" (줄무늬!)
CLIP_ENTITY_PROMPTS = [
    "an american football player in full uniform actively playing on the field during a game",
    "a referee in black and white striped uniform on the football field",
    "a coach or sideline staff member standing on the sideline wearing team apparel",
    "a photographer or cameraman with professional camera equipment near the field"
]

# Team classification prompts
# 【중요!】
# - Team A/B: 헬멧 착용 (helmet) + 등번호 (numbers)
# - Referee: 헬멧 없음 (NO helmet) + 등번호 없음 (NO numbers) + 줄무늬 (stripes)
# - 심판은 1-3명만 존재! (경기당 7명이지만 카메라에는 1-3명만 보임)
#
# 【Prompt 전략】
# 1. 헬멧 강조: "wearing a helmet" vs "NO helmet"
# 2. 줄무늬 강조: "vertical striped shirt"
# 3. 행동 차이: "running tackling" vs "standing still watching"
CLIP_TEAM_PROMPTS = [
    "an american football player wearing a solid white colored protective helmet on their head with a bright yellow or gold colored jersey shirt with black colored numbers on the back and solid white colored football pants, actively running tackling and playing in the game",

    "an american football player wearing a solid black colored protective helmet on their head with a plain solid white colored jersey shirt with sky blue or light blue colored numbers on the back without any stripes and solid black colored football pants, actively running tackling and playing in the game",

    "a football referee official wearing NO helmet on their bare head with a black and white vertical striped referee shirt with thick alternating black and white vertical stripes without any numbers on the back and solid black pants, standing still and watching the game as an official referee without running or tackling like players do"
]

# Entity filtering (result.mp4에서 제외할 대상)
CLIP_EXCLUDE_ENTITIES = ['sideline', 'other']  # 사이드라인 인원 제외
CLIP_SHOW_REFEREES = True  # False = 심판도 제외

# Hybrid mode (Stadium masking + CLIP)
# 【Hybrid 전략】
# 1. Stadium masking (빠름): 명확한 경기장 밖 제외
# 2. CLIP (느림): 경계 50픽셀 이내만 정밀 검사
# → 속도 + 정확도 균형
CLIP_HYBRID_MODE = True  # Hybrid 모드 활성화
CLIP_REFINE_BOUNDARY = 50  # 경계 50픽셀 이내만 CLIP 검사


# ============================================================================
# 🐛 DEBUG AND DEVELOPMENT (디버그 설정)
# ============================================================================

SHOW_PREVIEW = False  # 처리 중 미리보기 (비활성화 = 더 빠름)
VERBOSE = True        # 상세 로그 출력
