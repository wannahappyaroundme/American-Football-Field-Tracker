"""
Configuration parameters for Football Player and Ball Tracker.
Modify these values to tune the tracking behavior.
"""

# ============================================================================
# VIDEO INPUT/OUTPUT
# ============================================================================

# Path to input video file
VIDEO_INPUT_PATH = "zoomed_game.mp4"

# Path to save output video (set to None to disable saving)
VIDEO_OUTPUT_PATH = "tracked_output.mp4"  # Or None


# ============================================================================
# YOLO DETECTION SETTINGS
# ============================================================================

# YOLO model path/name
# Options: 'yolov8n.pt' (nano, fastest)
#          'yolov8s.pt' (small)
#          'yolov8m.pt' (medium)
#          'yolov8l.pt' (large)
#          'yolov8x.pt' (extra large, most accurate)
YOLO_MODEL_PATH = "yolov8n.pt"

# Confidence threshold for detections (0.0 - 1.0)
# Higher = fewer but more confident detections
YOLO_CONF_THRESHOLD = 0.5

# Target classes to detect (COCO dataset class IDs)
# 0 = person (players)
# 32 = sports ball (football)
YOLO_TARGET_CLASSES = [0, 32]


# ============================================================================
# ROI (REGION OF INTEREST) MASKING
# ============================================================================

# Percentage of frame height to exclude from top (0.0 - 1.0)
# This removes scoreboards, UI elements, upper crowd
ROI_TOP_PERCENT = 0.2  # Exclude top 20%

# Percentage of frame height to exclude from bottom (0.0 - 1.0)
# This removes lower crowd, advertisements
ROI_BOTTOM_PERCENT = 0.2  # Exclude bottom 20%


# ============================================================================
# SORT TRACKER SETTINGS
# ============================================================================

# Maximum number of frames to keep alive a track without associated detections
# Higher = tracks survive longer without detections (more robust to occlusions)
# Lower = tracks die faster (less false positives, but may lose legitimate tracks)
SORT_MAX_AGE = 1

# Minimum number of detections before a track is confirmed
# Higher = more conservative (fewer false tracks, but slower to establish new tracks)
# Lower = more aggressive (faster tracking, but more false positives)
SORT_MIN_HITS = 3

# Minimum IoU (Intersection over Union) for matching detections to tracks
# Higher = stricter matching (less ID switches, but may lose tracks)
# Lower = more lenient matching (maintains tracks better, but more ID switches)
SORT_IOU_THRESHOLD = 0.3


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Track bounding box color (BGR format)
TRACK_COLOR = (255, 0, 255)  # Magenta

# Track bounding box thickness (pixels)
TRACK_THICKNESS = 2

# Maximum width for display window (scales down if larger)
DISPLAY_MAX_WIDTH = 1600  # Set to None to disable scaling


# ============================================================================
# PERFORMANCE SETTINGS - DYNAMIC FRAME PROCESSING
# ============================================================================

# DEPRECATED: Static FRAME_SKIP replaced by dynamic processing
# Process every Nth frame (set to 1 to process all frames)
# Higher values = faster processing but choppier tracking
FRAME_SKIP = 2  # Process every 2nd frame for balance of speed and smoothness

# Enable dynamic frame processing (intelligent skip based on motion)
ENABLE_DYNAMIC_PROCESSING = False  # Disabled for speed - use static FRAME_SKIP

# Frame change detection method: 'optical_flow' or 'mse'
FRAME_CHANGE_METHOD = 'mse'  # MSE is faster, optical flow is more accurate

# Threshold for frame change detection (0.0 - 1.0)
# Lower = more sensitive (processes more frames)
# Higher = less sensitive (skips more frames)
FRAME_CHANGE_THRESHOLD = 0.015  # MSE threshold (normalized)

# Optical flow parameters (if using 'optical_flow')
OPTICAL_FLOW_THRESHOLD = 2.0  # Magnitude threshold for motion detection

# Enable debug output
DEBUG_MODE = True


# ============================================================================
# ADVANCED YOLO SETTINGS
# ============================================================================

# Image size for YOLO inference
# Larger = more accurate but slower
# Common values: 640, 1280
YOLO_IMG_SIZE = 640

# Device to run YOLO on
# 'cpu' or 'cuda' (GPU)
YOLO_DEVICE = 'cpu'


# ============================================================================
# POSE ESTIMATION SETTINGS
# ============================================================================

# Enable pose estimation for each tracked player
ENABLE_POSE_ESTIMATION = False  # Disabled for speed - enable later if needed

# Pose estimation model: 'mediapipe' or 'yolo_pose'
POSE_MODEL = 'mediapipe'  # MediaPipe is faster and more accurate

# Minimum pose detection confidence
POSE_CONFIDENCE = 0.5

# Pose estimation interval (process pose every N tracking updates)
# Higher = faster but less frequent pose data
POSE_ESTIMATION_INTERVAL = 1  # Process pose every frame where tracking occurs


# ============================================================================
# TEAM CLASSIFICATION SETTINGS
# ============================================================================

# Enable automatic team classification
ENABLE_TEAM_CLASSIFICATION = False  # Disabled for speed - enable later if needed

# Number of color clusters (teams + referees)
NUM_TEAMS = 3  # Team A, Team B, Referees

# Region of bounding box to sample for jersey color (percentage)
# Sample from upper portion to avoid legs/field
JERSEY_SAMPLE_TOP = 0.2    # Start at 20% from top
JERSEY_SAMPLE_BOTTOM = 0.6  # End at 60% from top

# K-Means clustering parameters
KMEANS_INIT_FRAMES = 30  # Frames to collect before initial clustering
KMEANS_UPDATE_INTERVAL = 60  # Re-cluster every N frames

# Formation detection parameters
CROUCH_THRESHOLD = 0.7  # Confidence threshold for crouching pose
LINE_OF_SCRIMMAGE_TOLERANCE = 5  # Pixels tolerance for line detection


# ============================================================================
# HOMOGRAPHY & BIRD'S EYE VIEW SETTINGS
# ============================================================================

# Enable homography-based bird's eye view
ENABLE_BIRDS_EYE_VIEW = False  # Disabled for speed - enable later if needed

# Field dimensions (in yards)
FIELD_LENGTH = 120  # Including end zones
FIELD_WIDTH = 53.33  # Standard width

# Bird's eye view output dimensions (pixels)
BIRDS_EYE_WIDTH = 400
BIRDS_EYE_HEIGHT = 600

# Field line detection parameters
FIELD_LINE_DETECTION_INTERVAL = 30  # Recalculate homography every N frames
USE_CACHED_HOMOGRAPHY = True  # Use previous homography if line detection fails

# Minimum confidence for field line detection
MIN_FIELD_LINES = 4  # Minimum lines needed for homography

# Player visualization on bird's eye view
PLAYER_DOT_RADIUS = 5
TEAM_A_COLOR = (0, 0, 255)    # Red (BGR)
TEAM_B_COLOR = (255, 0, 0)    # Blue (BGR)
REFEREE_COLOR = (255, 255, 255)  # White (BGR)


# ============================================================================
# DEEPSO RT / STRONGSORT SETTINGS
# ============================================================================

# Use DeepSORT instead of basic SORT
USE_DEEPSORT = False  # Disabled for speed - SORT is faster

# DeepSORT max age (frames to keep track alive)
DEEPSORT_MAX_AGE = 30  # Much higher than SORT

# DeepSORT appearance feature weight
DEEPSORT_MAX_IOU_DISTANCE = 0.7

# DeepSORT embedding distance threshold
DEEPSORT_MAX_COSINE_DISTANCE = 0.3


# ============================================================================
# TUNING TIPS
# ============================================================================

"""
Quick Tuning Guide:

1. TOO MANY FALSE DETECTIONS:
   - Increase YOLO_CONF_THRESHOLD (e.g., 0.6 or 0.7)
   - Increase SORT_MIN_HITS (e.g., 5)

2. MISSING PLAYERS:
   - Decrease YOLO_CONF_THRESHOLD (e.g., 0.3 or 0.4)
   - Adjust ROI_TOP_PERCENT and ROI_BOTTOM_PERCENT
   - Use larger YOLO model (yolov8s.pt or yolov8m.pt)

3. FREQUENT ID SWITCHES:
   - Increase SORT_MAX_AGE (e.g., 3 or 5)
   - Decrease SORT_IOU_THRESHOLD (e.g., 0.2)

4. TRACKS LOST DURING OCCLUSIONS:
   - Increase SORT_MAX_AGE (e.g., 5 or 10)
   - This keeps tracks alive longer without detections

5. SLOW PROCESSING:
   - Use smaller YOLO model (yolov8n.pt)
   - Increase FRAME_SKIP (e.g., 2 for every other frame)
   - Reduce YOLO_IMG_SIZE (e.g., 416)
   - Use GPU if available (YOLO_DEVICE = 'cuda')

6. TRACKS NOT ESTABLISHING QUICKLY:
   - Decrease SORT_MIN_HITS (e.g., 1 or 2)
   - Trade-off: more false positives

7. ROI NOT COVERING FIELD:
   - Adjust ROI_TOP_PERCENT and ROI_BOTTOM_PERCENT
   - View "1. ROI Mask Visualization" panel to see coverage
"""

