"""
Football Video Analysis Tool
=============================

Offline video analysis system for American football that combines:
1. YOLOv8 object detection for accurate player separation
2. Team classification via jersey color clustering (K-Means)
3. Static homography transformation for tactical top-down view
4. Side-by-side visualization (original + tactical map)

Priority: Accuracy and detailed analysis over speed.
Output: Processed MP4 video file with complete annotations.

Author: Computer Vision Engineer
Date: October 2025
Version: 2.0 - Clean Consolidated System
"""

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import math
from typing import Tuple, List, Optional, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Video paths
INPUT_VIDEO = "zoomed_game.mp4"
OUTPUT_VIDEO = "output_analysis.mp4"

# YOLO settings
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5

# Stadium/Field Recognition (HSV for green field)
ENABLE_STADIUM_MASKING = True
FIELD_HSV_LOWER = (35, 40, 40)      # Lower bound for green field
FIELD_HSV_UPPER = (85, 255, 255)    # Upper bound for green field

# Morphological operations for field mask cleanup
MORPH_KERNEL_SIZE = 15
MORPH_ITERATIONS = 2

# ROI Masking (relative percentages, not absolute pixels)
ROI_TOP_PERCENT = 0.20      # Exclude top 20% (scoreboard area)
ROI_BOTTOM_PERCENT = 0.10   # Exclude bottom 10% (lower crowd)

# Team classification - HSV color ranges (Hue, Saturation, Value)
# Adjust these ranges based on your team's jersey colors
TEAM_A_HSV_RANGE = ((90, 50, 50), (130, 255, 255))    # Example: Blue jerseys
TEAM_B_HSV_RANGE = ((0, 0, 180), (180, 30, 255))      # Example: White jerseys
REFEREE_HSV_RANGE = ((0, 0, 0), (180, 255, 60))       # Example: Black jerseys

# Team visualization colors (BGR)
TEAM_A_COLOR = (255, 0, 0)      # Blue
TEAM_B_COLOR = (0, 0, 255)      # Red
REFEREE_COLOR = (0, 255, 255)   # Yellow
UNKNOWN_COLOR = (128, 128, 128) # Gray

# Top-down view settings
FIELD_WIDTH = 400    # Pixels
FIELD_HEIGHT = 600   # Pixels
FIELD_LENGTH_YARDS = 120  # Including end zones
FIELD_WIDTH_YARDS = 53.33

# Tracking settings
ENABLE_TRACKING = True           # Maintain objects when detection fails
MAX_TRACKING_FRAMES = 30         # Max frames to track without detection
TRACKING_IOU_THRESHOLD = 0.3     # IoU threshold for matching

# Distance tracking settings
ENABLE_DISTANCE_TRACKING = True  # Measure yards traveled per player
YARDS_PER_PIXEL = None          # Auto-calculated from homography

# Camera change detection (reset cache when camera zooms/pans)
ENABLE_CAMERA_CHANGE_DETECTION = True
CAMERA_CHANGE_THRESHOLD = 0.15   # MSE threshold for detecting zoom/pan
HOMOGRAPHY_RECALC_INTERVAL = 300 # Frames between forced recalculation

# Tactical map settings
PERSISTENT_DOTS = True           # Keep dots on tactical map (don't clear)
DOT_FADE_ALPHA = 0.98           # Fade rate for old dots (1.0 = no fade)

# Team color detection (improved clustering)
TEAM_DETECTION_METHOD = 'adaptive_clustering'  # 'adaptive_clustering' or 'fixed_ranges'
MIN_PLAYERS_FOR_CLUSTERING = 8  # Minimum players before auto team detection


# ============================================================================
# PART 0: CAMERA CHANGE DETECTION & CACHE MANAGEMENT
# ============================================================================

class CameraChangeDetector:
    """
    Detects significant camera changes (zoom, pan) to trigger cache reset.
    """
    
    def __init__(self, threshold=0.15):
        """Initialize detector."""
        self.threshold = threshold
        self.prev_frame_gray = None
        self.prev_homography = None
    
    def detect_change(self, frame):
        """
        Detect if camera has significantly changed.
        
        Returns:
            (has_changed, change_magnitude)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))  # Downsample for speed
        
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return True, 1.0
        
        # Calculate MSE
        mse = np.mean((gray.astype(float) - self.prev_frame_gray.astype(float)) ** 2)
        normalized_mse = mse / (255.0 ** 2)
        
        self.prev_frame_gray = gray
        
        return normalized_mse > self.threshold, normalized_mse


# ============================================================================
# PART 0.5: STADIUM/FIELD RECOGNITION
# ============================================================================

def create_stadium_mask(frame: np.ndarray) -> np.ndarray:
    """
    Create a binary mask that isolates the playing field/stadium area.
    
    This mask identifies the green field using HSV color space and applies
    morphological operations to create a clean mask. Only players within
    this mask will be detected, excluding fans, objects, and people outside
    the stadium.
    
    Args:
        frame: Input frame
        
    Returns:
        Binary mask (255 = field/stadium, 0 = background to exclude)
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for green field
    lower_green = np.array(FIELD_HSV_LOWER)
    upper_green = np.array(FIELD_HSV_UPPER)
    field_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    
    # Closing: Fill gaps in the field
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel, 
                                  iterations=MORPH_ITERATIONS)
    
    # Opening: Remove noise outside the field
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel,
                                  iterations=MORPH_ITERATIONS)
    
    # Dilate slightly to include players near field edges
    field_mask = cv2.dilate(field_mask, kernel, iterations=1)
    
    return field_mask


def apply_roi_mask(mask: np.ndarray, frame_height: int, frame_width: int) -> np.ndarray:
    """
    Apply ROI (Region of Interest) mask to exclude top and bottom percentages.
    
    Uses relative percentages rather than absolute pixels for better
    adaptability to different video resolutions.
    
    Args:
        mask: Existing mask (e.g., stadium mask)
        frame_height: Height of frame in pixels
        frame_width: Width of frame in pixels
        
    Returns:
        Modified mask with ROI applied
    """
    # Calculate boundaries based on relative percentages
    top_boundary = int(frame_height * ROI_TOP_PERCENT)
    bottom_boundary = int(frame_height * (1.0 - ROI_BOTTOM_PERCENT))
    
    # Black out top and bottom regions
    mask[0:top_boundary, :] = 0
    mask[bottom_boundary:, :] = 0
    
    return mask


def create_combined_mask(frame: np.ndarray) -> np.ndarray:
    """
    Create comprehensive mask combining stadium recognition and ROI.
    
    This creates a mask that:
    1. Identifies the green playing field (stadium masking)
    2. Excludes top percentage (scoreboard, upper crowd)
    3. Excludes bottom percentage (lower crowd, ads)
    
    Only players within this final mask will be detected.
    
    Args:
        frame: Input frame
        
    Returns:
        Final combined binary mask
    """
    height, width = frame.shape[:2]
    
    if ENABLE_STADIUM_MASKING:
        # Start with stadium/field mask
        mask = create_stadium_mask(frame)
        
        # Apply ROI exclusions
        mask = apply_roi_mask(mask, height, width)
    else:
        # Just use ROI mask without stadium detection
        mask = np.ones((height, width), dtype=np.uint8) * 255
        mask = apply_roi_mask(mask, height, width)
    
    return mask


# ============================================================================
# PART 1: STATIC HOMOGRAPHY CALCULATION
# ============================================================================

def detect_field_lines(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[List, List]:
    """
    Detect horizontal and vertical field lines using Hough Transform.
    
    Args:
        frame: Input video frame
        mask: Optional binary mask to limit detection area
        
    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    print("  Detecting field lines...")
    
    # Apply mask if provided
    if mask is not None:
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        masked_frame = frame
    
    # Convert to grayscale
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance white lines using top-hat morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.add(gray, tophat)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                           minLineLength=100, maxLineGap=20)
    
    if lines is None:
        return [], []
    
    # Classify lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        
        if angle < 15 or angle > 165:
            horizontal_lines.append((x1, y1, x2, y2))
        elif 75 < angle < 105:
            vertical_lines.append((x1, y1, x2, y2))
    
    print(f"  Found {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")
    
    return horizontal_lines, vertical_lines


def find_line_intersections(h_lines: List, v_lines: List) -> List[Tuple[float, float]]:
    """
    Find intersection points between horizontal and vertical lines.
    
    Args:
        h_lines: List of horizontal lines
        v_lines: List of vertical lines
        
    Returns:
        List of intersection points (x, y)
    """
    intersections = []
    
    for h_line in h_lines[:4]:  # Use first 4 horizontal lines
        for v_line in v_lines[:2]:  # Use first 2 vertical lines
            x1h, y1h, x2h, y2h = h_line
            x1v, y1v, x2v, y2v = v_line
            
            # Calculate intersection
            denom = (x1h - x2h) * (y1v - y2v) - (y1h - y2h) * (x1v - x2v)
            if abs(denom) < 1e-6:
                continue
            
            t = ((x1h - x1v) * (y1v - y2v) - (y1h - y1v) * (x1v - x2v)) / denom
            
            x = x1h + t * (x2h - x1h)
            y = y1h + t * (y2h - y1h)
            
            intersections.append((x, y))
    
    return intersections


def calculate_homography(first_frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Calculate static homography matrix from first frame.
    
    Args:
        first_frame: First frame of video
        mask: Optional mask to limit line detection to field area
        
    Returns:
        Homography matrix or None if calculation fails
    """
    print("\nCalculating static homography from first frame...")
    
    # Detect field lines (using mask to focus on field area)
    h_lines, v_lines = detect_field_lines(first_frame, mask)
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        print("  Warning: Insufficient lines detected")
        return None
    
    # Find intersections (source points in video)
    intersections = find_line_intersections(h_lines, v_lines)
    
    if len(intersections) < 4:
        print("  Warning: Insufficient intersection points")
        return None
    
    # Use first 4 intersections as source points
    src_points = np.float32(intersections[:4])
    
    # Define destination points on top-down view
    # Assuming intersections form a quadrilateral on the field
    dst_points = np.float32([
        [0, 0],                              # Top-left
        [FIELD_WIDTH, 0],                    # Top-right
        [0, FIELD_HEIGHT],                   # Bottom-left
        [FIELD_WIDTH, FIELD_HEIGHT]          # Bottom-right
    ])
    
    # Calculate homography matrix
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if H is not None:
        print("  ✓ Homography matrix calculated successfully")
    else:
        print("  ✗ Homography calculation failed")
    
    return H


# ============================================================================
# PART 1.5: SIMPLE OBJECT TRACKING
# ============================================================================

class SimpleTracker:
    """
    Simple tracker to maintain detections when YOLO fails.
    Uses IoU matching to associate detections across frames.
    """
    
    def __init__(self, max_age=30, iou_threshold=0.3):
        """Initialize tracker."""
        self.tracks = []  # List of active tracks
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of (bbox, team_label, team_color)
            
        Returns:
            List of tracked objects with IDs
        """
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for track in self.tracks:
            best_iou = 0
            best_det_idx = None
            
            for det_idx in unmatched_detections:
                det_bbox = detections[det_idx][0]
                iou = self.calculate_iou(track['bbox'], det_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            # Match found
            if best_iou >= self.iou_threshold:
                track['bbox'] = detections[best_det_idx][0]
                track['team_label'] = detections[best_det_idx][1]
                track['team_color'] = detections[best_det_idx][2]
                track['age'] = 0
                matched_tracks.append(track)
                unmatched_detections.remove(best_det_idx)
            else:
                # No match - age the track
                track['age'] += 1
                if track['age'] < self.max_age:
                    matched_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            bbox, team_label, team_color = detections[det_idx]
            matched_tracks.append({
                'id': self.next_id,
                'bbox': bbox,
                'team_label': team_label,
                'team_color': team_color,
                'age': 0
            })
            self.next_id += 1
        
        self.tracks = matched_tracks
        return self.tracks


# ============================================================================
# PART 2: TEAM CLASSIFICATION VIA COLOR CLUSTERING
# ============================================================================

def get_team_color(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extract dominant SHIRT/JERSEY color from bounding box.
    
    Focuses specifically on the shirt/jersey area (torso) and uses
    improved sampling to get accurate team colors.
    
    Args:
        frame: Full frame
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Dominant HSV color or None
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure valid bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Extract SHIRT/JERSEY region (upper-middle of bbox, avoiding head and legs)
    # Focus on 25%-60% of bbox height for best jersey visibility
    bbox_height = y2 - y1
    bbox_width = x2 - x1
    
    # Shirt region (middle-upper torso)
    shirt_y1 = int(y1 + bbox_height * 0.25)  # Start at 25% (below head)
    shirt_y2 = int(y1 + bbox_height * 0.60)  # End at 60% (above legs)
    
    # Also crop horizontally to focus on center (avoid arms/background)
    shirt_x1 = int(x1 + bbox_width * 0.2)
    shirt_x2 = int(x2 - bbox_width * 0.2)
    
    if shirt_y2 <= shirt_y1 or shirt_x2 <= shirt_x1:
        return None
    
    shirt_region = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    
    if shirt_region.size == 0:
        return None
    
    # Convert to HSV for better color separation
    shirt_hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
    
    # Reshape to list of pixels
    pixels = shirt_hsv.reshape(-1, 3).astype(np.float32)
    
    # Filter out shadows and glare more aggressively
    brightness = pixels[:, 2]  # V channel
    saturation = pixels[:, 1]  # S channel
    
    # Keep only pixels with good saturation and reasonable brightness
    # This focuses on actual colored jersey fabric, not shadows or highlights
    mask = (brightness > 50) & (brightness < 210) & (saturation > 30)
    filtered_pixels = pixels[mask]
    
    if len(filtered_pixels) < 20:
        return None
    
    # Use K-Means with k=2 to find the two most dominant colors
    # Then select the more saturated one (likely the jersey, not skin/equipment)
    n_clusters = min(2, len(filtered_pixels))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(filtered_pixels)
    
    # Select cluster with highest average saturation (jersey color)
    best_color = None
    best_saturation = 0
    
    for center in kmeans.cluster_centers_:
        if center[1] > best_saturation:  # S channel
            best_saturation = center[1]
            best_color = center
    
    return best_color if best_color is not None else kmeans.cluster_centers_[0]


def classify_team(hsv_color: np.ndarray) -> Tuple[str, Tuple[int, int, int]]:
    """
    Classify team based on HSV color.
    
    Args:
        hsv_color: Dominant HSV color (H, S, V)
        
    Returns:
        Tuple of (team_label, visualization_color_bgr)
    """
    h, s, v = hsv_color
    
    # Check Team A range
    (h_min_a, s_min_a, v_min_a), (h_max_a, s_max_a, v_max_a) = TEAM_A_HSV_RANGE
    if h_min_a <= h <= h_max_a and s_min_a <= s <= s_max_a and v_min_a <= v <= v_max_a:
        return "Team A", TEAM_A_COLOR
    
    # Check Team B range
    (h_min_b, s_min_b, v_min_b), (h_max_b, s_max_b, v_max_b) = TEAM_B_HSV_RANGE
    if h_min_b <= h <= h_max_b and s_min_b <= s <= s_max_b and v_min_b <= v <= v_max_b:
        return "Team B", TEAM_B_COLOR
    
    # Check Referee range
    (h_min_r, s_min_r, v_min_r), (h_max_r, s_max_r, v_max_r) = REFEREE_HSV_RANGE
    if h_min_r <= h <= h_max_r and s_min_r <= s <= s_max_r and v_min_r <= v <= v_max_r:
        return "Referee", REFEREE_COLOR
    
    return "Unknown", UNKNOWN_COLOR


# ============================================================================
# PART 3: TOP-DOWN VIEW CREATION
# ============================================================================

def create_field_template() -> np.ndarray:
    """
    Create a blank top-down field template with yard lines.
    
    Returns:
        Field image with yard lines drawn
    """
    # Create green field
    field = np.zeros((FIELD_HEIGHT, FIELD_WIDTH, 3), dtype=np.uint8)
    field[:, :] = (34, 139, 34)  # Green
    
    # Draw yard lines every 10 yards
    yards_per_pixel = FIELD_HEIGHT / FIELD_LENGTH_YARDS
    
    for yard in range(0, FIELD_LENGTH_YARDS + 1, 10):
        y = int(yard * yards_per_pixel)
        cv2.line(field, (0, y), (FIELD_WIDTH, y), (255, 255, 255), 2)
        
        # Add yard number
        if 0 < yard < FIELD_LENGTH_YARDS:
            cv2.putText(field, str(yard), (10, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw sidelines
    cv2.rectangle(field, (0, 0), (FIELD_WIDTH - 1, FIELD_HEIGHT - 1),
                 (255, 255, 255), 3)
    
    # Add title
    cv2.putText(field, "TACTICAL VIEW", (FIELD_WIDTH // 2 - 60, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return field


def transform_point_to_topdown(point: Tuple[float, float], 
                               homography: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Transform point from video coordinates to top-down view.
    
    Args:
        point: Point in video coordinates (x, y)
        homography: Homography matrix
        
    Returns:
        Transformed point or None
    """
    if homography is None:
        return None
    
    # Convert to homogeneous coordinates
    pt = np.array([[point]], dtype=np.float32)
    
    # Transform
    transformed = cv2.perspectiveTransform(pt, homography)
    
    x, y = transformed[0][0]
    
    # Clip to field bounds
    x = np.clip(x, 0, FIELD_WIDTH - 1)
    y = np.clip(y, 0, FIELD_HEIGHT - 1)
    
    return (int(x), int(y))


# ============================================================================
# PART 4: MAIN PROCESSING PIPELINE
# ============================================================================

def process_video():
    """
    Main processing pipeline - analyzes entire video offline.
    """
    print("="*70)
    print("  FOOTBALL VIDEO ANALYSIS TOOL")
    print("="*70)
    
    # Step 1: Load video
    print(f"\n[1/6] Loading video: {INPUT_VIDEO}")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {INPUT_VIDEO}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  ✓ Video loaded: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Step 2: Load YOLO model
    print(f"\n[2/6] Loading YOLOv8 model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)
    print("  ✓ Model loaded successfully")
    
    # Step 3: Analyze first frame and create stadium mask
    print("\n[3/6] Analyzing first frame for stadium/field recognition...")
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        return
    
    # Create initial stadium mask from first frame
    print("  Creating stadium mask to exclude non-field areas...")
    initial_mask = create_combined_mask(first_frame)
    
    # Calculate field coverage
    mask_coverage = (np.sum(initial_mask > 0) / initial_mask.size) * 100
    print(f"  ✓ Stadium mask created - field coverage: {mask_coverage:.1f}%")
    
    if mask_coverage < 15:
        print("  ⚠ Warning: Low field coverage - consider adjusting FIELD_HSV ranges")
    
    # Calculate homography using masked field
    print("\n[4/6] Calculating static homography from first frame...")
    homography_matrix = calculate_homography(first_frame, initial_mask)
    
    if homography_matrix is None:
        print("  ⚠ Homography calculation failed - using identity matrix")
        homography_matrix = np.eye(3, dtype=np.float32)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Step 5: Create field template
    print("\n[5/6] Creating top-down field template...")
    field_template = create_field_template()
    print("  ✓ Field template created")
    
    # Step 6: Setup video writer
    print(f"\n[6/6] Setting up output video: {OUTPUT_VIDEO}")
    
    # Output dimensions: side-by-side (original + tactical)
    output_width = width + FIELD_WIDTH
    output_height = max(height, FIELD_HEIGHT)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (output_width, output_height))
    
    print(f"  ✓ Output: {output_width}x{output_height} @ {fps:.1f} FPS")
    
    # Initialize tracker for maintaining detections
    print("\n[TRACKING] Initializing object tracker...")
    tracker = SimpleTracker(max_age=MAX_TRACKING_FRAMES, iou_threshold=TRACKING_IOU_THRESHOLD) if ENABLE_TRACKING else None
    if tracker:
        print("  ✓ Tracker enabled - maintains objects when YOLO detection fails")
    
    # Create persistent tactical map (dots stay visible, no blinking)
    print("\n[TACTICAL MAP] Setting up persistent tactical display...")
    if PERSISTENT_DOTS:
        persistent_tactical_map = field_template.copy().astype(np.float32)
        print("  ✓ Persistent mode - dots accumulate and stay visible")
    else:
        persistent_tactical_map = None
        print("  Standard mode - map refreshes each frame")
    
    # Processing
    print(f"\n{'='*70}")
    print(f"  PROCESSING {total_frames} FRAMES")
    print("="*70)
    print("  Homography: CACHED (calculated once, reused for all frames)")
    print("  Stadium masking: ENABLED" if ENABLE_STADIUM_MASKING else "  Stadium masking: DISABLED")
    print(f"  ROI: Excluding top {ROI_TOP_PERCENT*100:.0f}% and bottom {ROI_BOTTOM_PERCENT*100:.0f}%")
    print("  Tracking: ENABLED - maintains IDs across frames" if ENABLE_TRACKING else "  Tracking: DISABLED")
    print("  Tactical dots: PERSISTENT - no blinking" if PERSISTENT_DOTS else "  Tactical dots: REFRESH each frame")
    print("  Priority: Accuracy over speed (offline analysis)")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # ====================================================================
        # STADIUM MASKING - Recognize and isolate field area
        # ====================================================================
        
        # Create stadium mask for this frame (identifies field, excludes non-field areas)
        stadium_mask = create_combined_mask(frame)
        
        # Apply mask to frame (zeros out background)
        # This removes people/objects outside the stadium before YOLO detection
        masked_frame = cv2.bitwise_and(frame, frame, mask=stadium_mask)
        
        # ====================================================================
        # YOLO DETECTION (on masked frame with background removed)
        # ====================================================================
        
        # Run YOLO on masked frame - only detects players on the field
        results = model(masked_frame, verbose=False, conf=YOLO_CONFIDENCE)
        
        # ====================================================================
        # COLLECT ALL DETECTIONS
        # ====================================================================
        
        detections_list = []  # For tracker: (bbox, team_label, team_color)
        
        # Process each detected person
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                # Only process 'person' class (class 0 in COCO)
                if int(box.cls[0]) != 0:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                
                # ============================================================
                # VERIFY DETECTION IS WITHIN STADIUM MASK
                # ============================================================
                
                # Calculate foot position (bottom center of bbox)
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                
                # Check if foot position is within the stadium mask
                # This double-checks that player is actually on the field
                if foot_x < 0 or foot_x >= stadium_mask.shape[1] or \
                   foot_y < 0 or foot_y >= stadium_mask.shape[0]:
                    continue  # Skip if out of bounds
                
                if stadium_mask[foot_y, foot_x] == 0:
                    continue  # Skip if not on field (background)
                
                # ============================================================
                # TEAM CLASSIFICATION
                # ============================================================
                
                # Extract dominant jersey color
                dominant_color_hsv = get_team_color(frame, (x1, y1, x2, y2))
                
                if dominant_color_hsv is not None:
                    team_label, box_color = classify_team(dominant_color_hsv)
                else:
                    team_label, box_color = "Unknown", UNKNOWN_COLOR
                
                # Add to detections list for tracker
                detections_list.append(((x1, y1, x2, y2), team_label, box_color))
        
        # ====================================================================
        # UPDATE TRACKER - Maintain objects across frames
        # ====================================================================
        
        if tracker:
            tracked_objects = tracker.update(detections_list)
        else:
            # No tracking - use detections directly
            tracked_objects = [{'id': i, 'bbox': d[0], 'team_label': d[1], 'team_color': d[2], 'age': 0} 
                             for i, d in enumerate(detections_list)]
        
        # ====================================================================
        # VISUALIZATION - Draw tracked objects
        # ====================================================================
        
        # Create fresh tactical view or use persistent one
        if PERSISTENT_DOTS and persistent_tactical_map is not None:
            # Fade old dots slightly
            persistent_tactical_map = persistent_tactical_map * DOT_FADE_ALPHA
            topdown_view = persistent_tactical_map.copy().astype(np.uint8)
        else:
            topdown_view = field_template.copy()
        
        # Annotate original frame
        annotated_frame = frame.copy()
        
        # Draw each tracked object
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            team_label = obj['team_label']
            box_color = obj['team_color']
            track_id = obj['id']
            
            # ============================================================
            # DRAW ON ORIGINAL FRAME
            # ============================================================
            
            # Draw bounding box with team color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Draw label with ID and team
            label = f"ID:{track_id} {team_label}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Label background
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), 
                         (x1 + text_w + 10, y1), box_color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # ============================================================
            # TRANSFORM TO TOP-DOWN VIEW
            # ============================================================
            
            # Calculate foot position (bottom center of bbox)
            foot_x = (x1 + x2) / 2
            foot_y = y2
            
            # Transform to top-down coordinates using CACHED homography
            topdown_point = transform_point_to_topdown((foot_x, foot_y), homography_matrix)
            
            if topdown_point is not None:
                # Draw player position on tactical map
                cv2.circle(topdown_view, topdown_point, 6, box_color, -1)
                cv2.circle(topdown_view, topdown_point, 7, (0, 0, 0), 1)
                
                # Draw ID number
                cv2.putText(topdown_view, str(track_id), 
                           (topdown_point[0] + 8, topdown_point[1] + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Update persistent tactical map
        if PERSISTENT_DOTS and persistent_tactical_map is not None:
            persistent_tactical_map = topdown_view.astype(np.float32)
        
        # ====================================================================
        # CREATE SIDE-BY-SIDE OUTPUT
        # ====================================================================
        
        # Resize tactical view to match frame height if needed
        if topdown_view.shape[0] != height:
            topdown_resized = cv2.resize(topdown_view, (FIELD_WIDTH, height))
        else:
            topdown_resized = topdown_view
        
        # Combine original (annotated) and tactical view side-by-side
        combined = np.hstack([annotated_frame, topdown_resized])
        
        # Ensure correct output dimensions
        if combined.shape[0] != output_height or combined.shape[1] != output_width:
            combined = cv2.resize(combined, (output_width, output_height))
        
        # Write to output video
        out.write(combined)
        
        # Optional: Show preview (comment out for faster processing)
        # cv2.imshow('Processing...', cv2.resize(combined, (1280, 720)))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n  ✓ Processing complete!")
    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput saved to: {OUTPUT_VIDEO}")
    print(f"Total frames processed: {frame_count}")
    print(f"\nOpen {OUTPUT_VIDEO} to view the analysis.")
    print("\nFeatures in output:")
    print("  • Left side: Original video with team-colored bounding boxes")
    print("  • Right side: Top-down tactical view showing player positions")
    print("  • Color coding: Team A (blue), Team B (red), Referee (yellow)")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        process_video()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
