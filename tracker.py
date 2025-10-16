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


# ============================================================================
# PART 1: STATIC HOMOGRAPHY CALCULATION
# ============================================================================

def detect_field_lines(frame: np.ndarray) -> Tuple[List, List]:
    """
    Detect horizontal and vertical field lines using Hough Transform.
    
    Args:
        frame: Input video frame
        
    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    print("  Detecting field lines...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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


def calculate_homography(first_frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate static homography matrix from first frame.
    
    Args:
        first_frame: First frame of video
        
    Returns:
        Homography matrix or None if calculation fails
    """
    print("\nCalculating static homography from first frame...")
    
    # Detect field lines
    h_lines, v_lines = detect_field_lines(first_frame)
    
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
# PART 2: TEAM CLASSIFICATION VIA COLOR CLUSTERING
# ============================================================================

def get_team_color(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extract dominant jersey color from bounding box (torso region).
    
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
    
    # Extract torso region (upper 40% of bounding box)
    bbox_height = y2 - y1
    torso_y1 = int(y1 + bbox_height * 0.2)
    torso_y2 = int(y1 + bbox_height * 0.6)
    
    if torso_y2 <= torso_y1:
        return None
    
    torso_region = frame[torso_y1:torso_y2, x1:x2]
    
    if torso_region.size == 0:
        return None
    
    # Convert to HSV
    torso_hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
    
    # Reshape to list of pixels
    pixels = torso_hsv.reshape(-1, 3).astype(np.float32)
    
    # Remove very dark pixels (shadows) and very bright pixels (glare)
    brightness = pixels[:, 2]  # V channel
    mask = (brightness > 40) & (brightness < 220)
    filtered_pixels = pixels[mask]
    
    if len(filtered_pixels) < 10:
        return None
    
    # Use K-Means to find dominant color (k=1)
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
    kmeans.fit(filtered_pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    return dominant_color


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
    
    # Step 3: Calculate static homography from first frame
    print("\n[3/6] Analyzing first frame for field geometry...")
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        return
    
    homography_matrix = calculate_homography(first_frame)
    
    if homography_matrix is None:
        print("  ⚠ Homography calculation failed - using identity matrix")
        homography_matrix = np.eye(3, dtype=np.float32)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Step 4: Create field template
    print("\n[4/6] Creating top-down field template...")
    field_template = create_field_template()
    print("  ✓ Field template created")
    
    # Step 5: Setup video writer
    print(f"\n[5/6] Setting up output video: {OUTPUT_VIDEO}")
    
    # Output dimensions: side-by-side (original + tactical)
    output_width = width + FIELD_WIDTH
    output_height = max(height, FIELD_HEIGHT)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (output_width, output_height))
    
    print(f"  ✓ Output: {output_width}x{output_height} @ {fps:.1f} FPS")
    
    # Step 6: Process each frame
    print(f"\n[6/6] Processing {total_frames} frames...")
    print("  This may take a while - accuracy over speed!")
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
        # YOLO DETECTION
        # ====================================================================
        
        # Run YOLO to detect all persons
        results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)
        
        # ====================================================================
        # PROCESS DETECTIONS
        # ====================================================================
        
        # Create fresh top-down view for this frame
        topdown_view = field_template.copy()
        
        # Annotate original frame
        annotated_frame = frame.copy()
        
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
                # TEAM CLASSIFICATION
                # ============================================================
                
                # Extract dominant jersey color
                dominant_color_hsv = get_team_color(frame, (x1, y1, x2, y2))
                
                if dominant_color_hsv is not None:
                    team_label, box_color = classify_team(dominant_color_hsv)
                else:
                    team_label, box_color = "Unknown", UNKNOWN_COLOR
                
                # ============================================================
                # DRAW ON ORIGINAL FRAME
                # ============================================================
                
                # Draw bounding box with team color
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
                
                # Draw label above box
                label = f"{team_label} ({conf:.2f})"
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
                
                # Transform to top-down coordinates
                topdown_point = transform_point_to_topdown((foot_x, foot_y), homography_matrix)
                
                if topdown_point is not None:
                    # Draw player position on tactical map
                    cv2.circle(topdown_view, topdown_point, 6, box_color, -1)
                    cv2.circle(topdown_view, topdown_point, 7, (0, 0, 0), 1)
        
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
