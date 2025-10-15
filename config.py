"""
Configuration parameters for the Football Field Tracker.
Modify these values to tune the line detection and tracking behavior.
"""

# ============================================================================
# VIDEO INPUT/OUTPUT
# ============================================================================

# Path to input video file
VIDEO_INPUT_PATH = "zoomed_game.mp4"

# Path to save output video (set to None to disable saving)
VIDEO_OUTPUT_PATH = "./zoomed_game_output.mp4"  # Example: "output_tracked.mp4"


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

# ============================================================================
# FIELD SEGMENTATION (HSV Color Masking)
# ============================================================================

# HSV range for green field detection
# These values isolate the green playing field from background
# HSV format: Hue (0-180), Saturation (0-255), Value (0-255)

# Lower bound for green field (adjust based on field conditions)
HSV_LOWER_GREEN = (35, 40, 40)      # (H_min, S_min, V_min)

# Upper bound for green field
HSV_UPPER_GREEN = (85, 255, 255)    # (H_max, S_max, V_max)

# Morphological operations to clean up the mask
# These remove small noise and fill gaps in the field mask
MORPH_KERNEL_SIZE = 15              # Size of morphological kernel
MORPH_CLOSING_ITERATIONS = 2        # Number of closing operations
MORPH_OPENING_ITERATIONS = 1        # Number of opening operations

# Minimum field area (as percentage of frame) to consider valid
# This helps filter out false positives
MIN_FIELD_AREA_PERCENT = 20.0       # Minimum 20% of frame should be field

# ============================================================================
# EDGE DETECTION (Applied to masked field)
# ============================================================================

# Gaussian blur parameters (for noise reduction before edge detection)
GAUSSIAN_KERNEL_SIZE = (5, 5)  # Kernel size (must be odd numbers)
GAUSSIAN_SIGMA = 1.5           # Standard deviation

# Canny edge detection parameters
CANNY_THRESHOLD_LOW = 50       # Lower threshold for edge detection
CANNY_THRESHOLD_HIGH = 150     # Upper threshold for edge detection
CANNY_APERTURE_SIZE = 3        # Aperture size for Sobel operator


# ============================================================================
# HOUGH LINE DETECTION
# ============================================================================

# Hough Transform parameters
HOUGH_RHO = 1                  # Distance resolution in pixels
HOUGH_THETA = 1                # Angular resolution in degrees (converted to radians)
HOUGH_THRESHOLD = 50           # Minimum number of intersections to detect a line
HOUGH_MIN_LINE_LENGTH = 50     # Minimum length of a line segment (pixels)
HOUGH_MAX_LINE_GAP = 10        # Maximum gap between segments to link them (pixels)


# ============================================================================
# LINE CLASSIFICATION
# ============================================================================

# Angle thresholds for classifying lines (in degrees)
# A line is horizontal if its angle is below this threshold
HORIZONTAL_ANGLE_THRESHOLD = 15.0

# A line is vertical if its angle is above this threshold
VERTICAL_ANGLE_THRESHOLD = 75.0


# ============================================================================
# LINE MERGING/FILTERING
# ============================================================================

# Maximum angle difference (in degrees) to consider two lines parallel
PARALLEL_ANGLE_THRESHOLD = 5.0

# Maximum perpendicular distance (in pixels) to merge parallel horizontal lines
HORIZONTAL_DISTANCE_THRESHOLD = 20.0

# Maximum perpendicular distance (in pixels) to merge parallel vertical lines
VERTICAL_DISTANCE_THRESHOLD = 30.0


# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Maximum width for display window (scales down if video is larger)
DISPLAY_MAX_WIDTH = 1600

# Colors for visualization (BGR format)
COLOR_HORIZONTAL_LINES = (0, 255, 0)    # Green for yard lines
COLOR_VERTICAL_LINES = (255, 0, 0)      # Blue for sidelines/hash marks
LINE_THICKNESS = 2
POINT_RADIUS = 5


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Process every Nth frame (set to 1 to process all frames)
# Higher values increase speed but reduce accuracy
FRAME_SKIP = 1

# Enable debug output
DEBUG_MODE = True


# ============================================================================
# FIELD MODEL (for future homography calculation)
# ============================================================================

# American football field dimensions (in yards)
FIELD_LENGTH = 120      # Total length including end zones (100 + 2×10)
FIELD_WIDTH = 53.33     # Width in yards (160 feet)

# Yard line spacing
YARD_LINE_SPACING = 5   # Yard lines every 5 yards

# Hash mark properties
HASH_MARK_LENGTH = 2    # Length in feet (converted to yards: 2/3)
HASH_MARK_WIDTH_NFL = 18.5  # Distance from sideline to hash marks (yards)


# ============================================================================
# HOMOGRAPHY SETTINGS (for Parts 3-4)
# ============================================================================

# Minimum number of reference points needed for homography
MIN_REFERENCE_POINTS = 4

# Output top-down view dimensions (in pixels)
OUTPUT_VIEW_WIDTH = 800
OUTPUT_VIEW_HEIGHT = 1200

# Pixels per yard in output view
PIXELS_PER_YARD = OUTPUT_VIEW_HEIGHT / FIELD_LENGTH


