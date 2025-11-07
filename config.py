"""
Football Play Analysis System - Configuration
통합 설정 파일 - tracker.py의 개선된 설정을 기반으로 함
"""

# ============================================================================
# VIDEO I/O
# ============================================================================

VIDEO_INPUT_PATH = 'input/video.mp4'
VIDEO_OUTPUT_PATH = 'output/result.mp4'
BEV_OUTPUT_PATH = 'output/bev.mp4'
JSON_OUTPUT_PATH = 'output/clip_summary.json'

# Video output quality
OUTPUT_CODEC = 'avc1'  # H.264 codec (high quality)
OUTPUT_QUALITY = 100    # 0-100 (higher = better quality)

# ============================================================================
# YOLO DETECTION
# ============================================================================

DETECTION_MODEL_PATH = 'yolo11n.pt'  # UPGRADED: yolov8n → yolo11n (22% fewer params, better small object detection)
POSE_MODEL_PATH = 'yolo11n-pose.pt'  # UPGRADED: yolov8n-pose → yolo11n-pose

# Detection confidence thresholds (PHASE 3: Back to 0.3 for better detection coverage)
YOLO_CONFIDENCE = 0.3  # 일반 탐지 신뢰도 (ByteTrack + 3-frame continuity handle IDs)
DETECTION_CONFIDENCE_THRESHOLD = 0.3  # 기존 코드 호환성

# YOLO Tracking parameters
# ByteTrack 설정은 bytetrack_extended.yaml 파일에서 관리
# track_buffer=300: Detection 실패 시 Kalman filter로 10초 동안 track 유지
DETECTION_EVERY_N_FRAMES = 90  # 상태 로깅 간격 (실제 detection은 매 프레임)

# Class IDs (COCO dataset)
CLASS_ID_PERSON = 0
CLASS_ID_BALL = 32

# Ball detection (PHASE 2.0: YOLO11 + 최적화된 임계값 0.1% → 10-30%)
ENABLE_BALL_DETECTION = True
BALL_CONFIDENCE = 0.10  # 공 탐지 신뢰도 (YOLO11: 0.05 → 0.10, better detection = higher threshold)
BALL_MIN_SIZE = 5  # 최소 공 크기 (RELAXED: 10 → 5 pixels, 멀리 있는 공도 감지)
BALL_MAX_SIZE = 150  # 최대 공 크기 (RELAXED: 120 → 150 pixels, 매우 근접 촬영)
BALL_ASPECT_RATIO_MIN = 0.5  # 최소 종횡비 (RELAXED: 0.7 → 0.5, 다양한 각도)
BALL_ASPECT_RATIO_MAX = 3.0  # 최대 종횡비 (RELAXED: 2.5 → 3.0, 미식축구 타원형)
BALL_COLOR = (0, 255, 255)  # Yellow (BGR)
BALL_CARRIER_COLOR = (0, 165, 255)  # Orange (BGR) - 주황색으로 볼 캐리어 표시

# ============================================================================
# STADIUM/FIELD RECOGNITION (경기장 인식 - 핵심!)
# ============================================================================

ENABLE_STADIUM_MASKING = True  # 경기장 밖 사람 제외 (필수!)

# Green field detection (HSV range)
FIELD_HSV_LOWER = (35, 40, 40)      # 녹색 필드 하한
FIELD_HSV_UPPER = (85, 255, 255)    # 녹색 필드 상한

# Morphological operations
MORPH_KERNEL_SIZE = 15
MORPH_ITERATIONS = 2

# Field boundary settings
MIN_FIELD_AREA_PERCENT = 25.0    # 최소 필드 면적 (%)
FIELD_BOUNDARY_EROSION = 10      # 경계 침식 (픽셀) - 경기장 밖 확실히 제외

# ============================================================================
# ROI MASKING (상하단 제외)
# ============================================================================

ROI_TOP_PERCENT = 0.20      # 상단 20% 제외 (스코어보드, 관중석)
ROI_BOTTOM_PERCENT = 0.10   # 하단 10% 제외 (광고판, 관중석)

# ============================================================================
# TEAM CLASSIFICATION (팀 분류)
# ============================================================================

# Team classification method
TEAM_DETECTION_METHOD = 'adaptive_clustering'  # 추천! 자동으로 팀 구분
                                                # 'fixed_range' = 수동 HSV 범위 설정

MIN_PLAYERS_FOR_CLUSTERING = 6  # 최소 선수 수 (이하면 fixed_range로 폴백)
NUM_TEAM_CLUSTERS = 3           # Team A, Team B, Referee

# Team colors for visualization (BGR format)
TEAM_A_COLOR = (255, 0, 0)      # Blue (파란색)
TEAM_B_COLOR = (0, 0, 255)      # Red (빨간색)
REFEREE_COLOR = (0, 255, 255)   # Yellow (노란색)
UNKNOWN_COLOR = (128, 128, 128) # Gray (회색 - 팀 미배정)
PLAYER_COLOR = (255, 0, 0)      # Blue (기존 코드 호환성)

# Fixed HSV ranges (backup for manual classification)
# 실제 비디오 색상에 맞게 조정
# Team A: 노란색 상의 + 흰색 하의
# Team B: 흰색 상의 + 검은색 하의
# Referee: 검은색/흰색 줄무늬 + 검은색 하의 (줄무늬 패턴으로 우선 감지)
TEAM_A_HSV_RANGE = ((18, 60, 100), (35, 255, 255))    # Yellow/Gold (노란색 상의, H범위 좁힘)
TEAM_B_HSV_RANGE = ((0, 0, 160), (180, 40, 255))      # White (FIXED: V 190→160, S 25→40, 그림자/로고 허용)
REFEREE_HSV_RANGE = ((0, 0, 40), (180, 35, 170))      # Black/White (ADJUSTED: V max 170, S max 35, Team B 겹침 방지)

# ============================================================================
# OBJECT TRACKING (객체 추적 - 핵심!)
# ============================================================================

ENABLE_TRACKING = True              # 객체 추적 활성화 (필수!)
MAX_TRACKING_FRAMES = 300           # 탐지 없이 유지할 최대 프레임 (10초 at 30fps) - 한 번 디텍션되면 오래 유지
TRACKING_IOU_THRESHOLD = 0.15       # IoU 임계값 (낮을수록 관대) - 0.20 → 0.15로 더 관대하게

# Team assignment freezing (팀 고정 - 핵심!)
FREEZE_TEAM_ASSIGNMENT = True       # 한번 할당된 팀은 절대 안 바뀜!
TEAM_ASSIGNMENT_CONFIDENCE = 2      # 2프레임 후 팀 고정 (빠른 안정화)

# Visualization and debugging
SHOW_FROZEN_INDICATOR = True        # 고정된 선수에게 녹색 테두리 표시
PRINT_FREEZE_EVENTS = True          # 팀 고정 시 콘솔 출력
PRINT_TRACKING_STATS = True         # 추적 통계 출력
DEBUG_TEAM_COLORS = False           # 팀 색상 디버그 (HSV 값 출력)

# ============================================================================
# TACTICAL MAP / BEV (Bird's Eye View)
# ============================================================================

# Tactical map dimensions
FIELD_WIDTH = 400           # 픽셀
FIELD_HEIGHT = 600          # 픽셀
FIELD_LENGTH_YARDS = 120    # 야드
FIELD_WIDTH_YARDS = 53.33   # 야드

# Persistent dots (잔상 설정 - 핵심!)
PERSISTENT_DOTS = True      # 점이 축적됨
DOT_FADE_ALPHA = 0.70       # 페이드 속도 (0.70 = 매우 빠른 페이드, 잔상 최소화!)
                            # 0.98 = 느린 페이드 (많은 잔상)
                            # 0.85 = 빠른 페이드
                            # 0.70 = 매우 빠른 페이드 (현재 설정)

# Homography settings (기존 시스템과 호환)
HOMOGRAPHY_MATRIX_PATH = 'homography_matrix.npy'
PIXELS_PER_YARD_BEV = 20  # BEV 픽셀당 야드 (1000 / 50 = 20)

# BEV field configuration
BEV_LEFT_YARD_LINE = 0
BEV_DIRECTION = 'LEFT_TO_RIGHT'

# BEV field boundaries (⭐ EXPANDED for 25-30 player detection)
# 이전: 50-450 픽셀 (너무 좁아서 6-14명만 검출)
# 현재: 0-500 픽셀 (전체 필드 커버, CLIP이 필터링 담당)
# BEV 캔버스: 1000x500 픽셀
BEV_FIELD_X_MIN = 50      # 왼쪽 사이드라인 마진 (픽셀)
BEV_FIELD_X_MAX = 950     # 오른쪽 사이드라인 마진 (픽셀)
BEV_FIELD_Y_MIN = 0       # 상단 엔드존 마진 (픽셀) - EXPANDED: 50 → 0
BEV_FIELD_Y_MAX = 500     # 하단 엔드존 마진 (픽셀) - EXPANDED: 450 → 500

# ============================================================================
# DISTANCE TRACKING
# ============================================================================

ENABLE_DISTANCE_TRACKING = True  # 야드 측정

# ============================================================================
# CLIP CONFIGURATION (OpenAI CLIP for semantic classification)
# ============================================================================

# Enable CLIP classification (⭐ PHASE 2: ENABLED for Team B detection & accuracy)
ENABLE_CLIP_CLASSIFICATION = True  # ENABLED: Team B HSV alone insufficient
ENABLE_CLIP_ENTITY_FILTERING = True  # ENABLED: Filter sideline staff (reduce IDs)
ENABLE_CLIP_TEAM_CLASSIFICATION = True  # ENABLED: Visual team classification (95%+ accuracy)

# Import torch for device detection (must be before CLIP_DEVICE)
try:
    import torch as _torch_module
    CLIP_DEVICE = 'cuda' if _torch_module.cuda.is_available() else 'cpu'
except ImportError:
    CLIP_DEVICE = 'cpu'
    print("Warning: PyTorch not found. CLIP will use CPU (slower).")

# CLIP model settings
CLIP_MODEL_NAME = 'ViT-B/32'  # Options: ViT-B/32 (fast), ViT-B/16 (balanced), ViT-L/14 (slow, accurate)

# CLIP processing settings
CLIP_BATCH_SIZE = 16         # Number of crops to process together (batch processing)
CLIP_FRAME_INTERVAL = 30     # Run CLIP every N frames (30 = 1 detection per second at 30fps)
CLIP_CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence (RAISED: 0.40→0.50 for stricter classification)

# Entity classification prompts (개선: 더 구체적인 설명으로 정확도 향상)
CLIP_ENTITY_PROMPTS = [
    "an american football player in full uniform actively playing on the field during a game",
    "a referee in black and white striped uniform on the football field",
    "a coach or sideline staff member standing on the sideline wearing team apparel",
    "a photographer or cameraman with professional camera equipment near the field"
]

# Team classification prompts - REFEREE EMPHASIS: Only 1-3 referees in entire game!
# ⭐ CRITICAL: Players wear HELMETS and have NUMBERS, Referees have NO HELMET and NO NUMBERS!
# Team A = WHITE helmet + YELLOW jersey with BLACK numbers + WHITE pants (PLAYER!)
# Team B = BLACK helmet + WHITE jersey with SKY BLUE numbers + BLACK pants (PLAYER!)
# Referee = NO helmet + STRIPED shirt (NO numbers!) + BLACK pants (ONLY 1-3 people!)
CLIP_TEAM_PROMPTS = [
    "an american football player wearing a solid white colored protective helmet on their head with a bright yellow or gold colored jersey shirt with black colored numbers on the back and solid white colored football pants, actively running tackling and playing in the game",

    "an american football player wearing a solid black colored protective helmet on their head with a plain solid white colored jersey shirt with sky blue or light blue colored numbers on the back without any stripes and solid black colored football pants, actively running tackling and playing in the game",

    "a football referee official wearing NO helmet on their bare head with a black and white vertical striped referee shirt with thick alternating black and white vertical stripes without any numbers on the back and solid black pants, standing still and watching the game as an official referee without running or tackling like players do"
]

# Entity filtering (what to exclude from result.mp4)
CLIP_EXCLUDE_ENTITIES = ['sideline', 'other']  # Don't show these in output
CLIP_SHOW_REFEREES = True  # Set to False to hide referees in output

# Hybrid mode (combine stadium masking + CLIP)
CLIP_HYBRID_MODE = True  # Use fast stadium masking + CLIP refinement
CLIP_REFINE_BOUNDARY = 50  # Pixels from field boundary to use CLIP

# ============================================================================
# DEBUG AND DEVELOPMENT
# ============================================================================

# Enable preview window during processing (비활성화 = 더 빠른 처리)
SHOW_PREVIEW = False

# Print detailed logs
VERBOSE = True
