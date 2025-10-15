"""
Football Field Tracker - Dynamic Homography for Top-Down View
==============================================================

This system analyzes zoomed-in and panning American football footage to create
a consistent top-down tactical map by computing a new homography matrix for 
each frame based on detected yard lines and hash marks.

Author: Computer Vision Engineer
Date: October 2025
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import math
import config  # Import configuration parameters


# ============================================================================
# PART 1: MAIN VIDEO PROCESSING LOOP
# ============================================================================

def load_video(video_path: str) -> cv2.VideoCapture:
    """
    Load the input video file.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        VideoCapture object
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file: {video_path}")
    
    # Print video properties for debugging
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video loaded successfully:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {frame_count}")
    
    return cap


# ============================================================================
# PART 2: FIELD SEGMENTATION (HSV Color Masking)
# ============================================================================

def create_field_mask(frame: np.ndarray) -> np.ndarray:
    """
    Create a binary mask that isolates the green playing field using HSV color space.
    
    This is the first stage of our two-stage approach:
    1. Identify the green field area
    2. Perform line detection only on that area
    
    Process:
    1. Convert frame from BGR to HSV color space
    2. Use cv2.inRange to isolate green colors
    3. Apply morphological operations to clean up the mask
    4. Validate that the mask covers a reasonable field area
    
    Args:
        frame: Input BGR frame
        
    Returns:
        Binary mask where white (255) = field, black (0) = background
    """
    # Convert BGR to HSV color space
    # HSV is better for color segmentation because it separates color (Hue)
    # from intensity (Value), making it more robust to lighting changes
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask using the configured HSV range for green
    # This isolates all pixels that fall within the green color range
    lower_green = np.array(config.HSV_LOWER_GREEN)
    upper_green = np.array(config.HSV_UPPER_GREEN)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    # This removes noise and fills small gaps
    kernel = np.ones((config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8)
    
    # Closing: removes small holes inside the field (dilation followed by erosion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                           iterations=config.MORPH_CLOSING_ITERATIONS)
    
    # Opening: removes small noise outside the field (erosion followed by dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                           iterations=config.MORPH_OPENING_ITERATIONS)
    
    # Validate mask - check if it covers a reasonable area
    mask_area_percent = (np.sum(mask > 0) / mask.size) * 100
    
    if config.DEBUG_MODE:
        print(f"  Field mask coverage: {mask_area_percent:.1f}% of frame")
    
    if mask_area_percent < config.MIN_FIELD_AREA_PERCENT:
        print(f"  Warning: Field mask only covers {mask_area_percent:.1f}% of frame")
        print(f"  This is below the minimum threshold of {config.MIN_FIELD_AREA_PERCENT}%")
        print(f"  Consider adjusting HSV_LOWER_GREEN and HSV_UPPER_GREEN in config.py")
    
    return mask


def apply_mask_to_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply the field mask to the frame, keeping only the field area.
    
    This zeros out (makes black) all pixels that are not part of the field,
    so line detection will only see the field area.
    
    Args:
        frame: Input BGR frame
        mask: Binary mask (white = keep, black = remove)
        
    Returns:
        Masked frame with only field area visible
    """
    # Use bitwise AND to keep only the pixels where mask is white (255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return masked_frame


def visualize_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a visualization showing the mask overlaid on the original frame.
    
    This is helpful for debugging and tuning the HSV parameters.
    The field area will be shown with a green tint overlay.
    
    Args:
        frame: Original BGR frame
        mask: Binary field mask
        
    Returns:
        Frame with semi-transparent green overlay on detected field
    """
    overlay = frame.copy()
    
    # Create a green overlay where the mask is white
    overlay[mask > 0] = cv2.addWeighted(
        overlay[mask > 0], 0.7,  # 70% original
        np.full_like(overlay[mask > 0], (0, 255, 0)), 0.3,  # 30% green
        0
    )
    
    return overlay


# ============================================================================
# PART 3: AUTOMATED LINE DETECTION AND FILTERING (on Masked Field)
# ============================================================================

def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the frame for line detection.
    
    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Apply Canny edge detection
    
    Args:
        frame: Input BGR frame
        
    Returns:
        Tuple of (grayscale image, edge-detected image)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    # Kernel size is configurable for different noise levels
    blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL_SIZE, config.GAUSSIAN_SIGMA)
    
    # Apply Canny edge detection
    # Thresholds are configurable based on field line contrast
    edges = cv2.Canny(blurred, config.CANNY_THRESHOLD_LOW, config.CANNY_THRESHOLD_HIGH, 
                     apertureSize=config.CANNY_APERTURE_SIZE)
    
    return gray, edges


def detect_lines_hough(edges: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect line segments using Hough Line Transform (Probabilistic).
    
    Args:
        edges: Edge-detected binary image
        
    Returns:
        Array of detected lines in format [[x1, y1, x2, y2], ...] or None
    """
    # HoughLinesP parameters (configurable for different field conditions):
    # - rho: Distance resolution in pixels
    # - theta: Angular resolution in radians
    # - threshold: Minimum number of intersections to detect a line
    # - minLineLength: Minimum length of a line segment
    # - maxLineGap: Maximum gap between line segments to treat them as one
    lines = cv2.HoughLinesP(
        edges,
        rho=config.HOUGH_RHO,
        theta=np.pi / 180 * config.HOUGH_THETA,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.HOUGH_MAX_LINE_GAP
    )
    
    return lines


def calculate_line_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the angle of a line segment in degrees.
    
    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        
    Returns:
        Angle in degrees (0-180), where 0° is horizontal
    """
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate angle in radians, then convert to degrees
    angle = math.atan2(abs(dy), abs(dx)) * 180 / math.pi
    
    return angle


def calculate_line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the Euclidean length of a line segment.
    
    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        
    Returns:
        Length of the line segment
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def extend_line(x1: float, y1: float, x2: float, y2: float, 
                frame_width: int, frame_height: int) -> Tuple[float, float, float, float]:
    """
    Extend a line segment to the frame boundaries.
    This is useful for finding virtual intersections outside the visible frame.
    
    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Extended line coordinates (x1_ext, y1_ext, x2_ext, y2_ext)
    """
    # Calculate line parameters (y = mx + b)
    if x2 - x1 == 0:  # Vertical line
        return x1, 0, x1, frame_height
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    # Find intersections with frame boundaries
    points = []
    
    # Intersection with left edge (x=0)
    y_left = intercept
    if 0 <= y_left <= frame_height:
        points.append((0, y_left))
    
    # Intersection with right edge (x=frame_width)
    y_right = slope * frame_width + intercept
    if 0 <= y_right <= frame_height:
        points.append((frame_width, y_right))
    
    # Intersection with top edge (y=0)
    if slope != 0:
        x_top = -intercept / slope
        if 0 <= x_top <= frame_width:
            points.append((x_top, 0))
    
    # Intersection with bottom edge (y=frame_height)
    if slope != 0:
        x_bottom = (frame_height - intercept) / slope
        if 0 <= x_bottom <= frame_width:
            points.append((x_bottom, frame_height))
    
    if len(points) >= 2:
        return points[0][0], points[0][1], points[1][0], points[1][1]
    else:
        return x1, y1, x2, y2


def classify_lines(lines: np.ndarray, 
                   horizontal_angle_threshold: float = None,
                   vertical_angle_threshold: float = None) -> Tuple[List, List]:
    """
    Classify detected lines into horizontal (yard lines) and vertical (sidelines/hash marks).
    
    A line is considered:
    - Horizontal if its angle is < horizontal_angle_threshold degrees
    - Vertical if its angle is > vertical_angle_threshold degrees
    
    Args:
        lines: Array of detected lines [[x1, y1, x2, y2], ...]
        horizontal_angle_threshold: Maximum angle for horizontal classification
        vertical_angle_threshold: Minimum angle for vertical classification
        
    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    # Use config defaults if not specified
    if horizontal_angle_threshold is None:
        horizontal_angle_threshold = config.HORIZONTAL_ANGLE_THRESHOLD
    if vertical_angle_threshold is None:
        vertical_angle_threshold = config.VERTICAL_ANGLE_THRESHOLD
    
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle and length
        angle = calculate_line_angle(x1, y1, x2, y2)
        length = calculate_line_length(x1, y1, x2, y2)
        
        # Store line with its properties
        line_data = {
            'coords': (x1, y1, x2, y2),
            'angle': angle,
            'length': length
        }
        
        # Classify based on angle
        if angle < horizontal_angle_threshold:
            horizontal_lines.append(line_data)
        elif angle > vertical_angle_threshold:
            vertical_lines.append(line_data)
    
    return horizontal_lines, vertical_lines


def merge_similar_lines(lines: List[dict], 
                       parallel_threshold: float = 5.0,
                       distance_threshold: float = 20.0) -> List[dict]:
    """
    Merge lines that are close and parallel to each other.
    
    This reduces noise and creates more robust line representations by:
    1. Grouping lines with similar angles (parallel)
    2. Grouping lines that are spatially close
    3. Averaging their positions to create a single representative line
    
    Args:
        lines: List of line dictionaries with 'coords', 'angle', 'length'
        parallel_threshold: Maximum angle difference (degrees) to consider lines parallel
        distance_threshold: Maximum distance (pixels) to consider lines close
        
    Returns:
        List of merged line dictionaries
    """
    if not lines:
        return []
    
    # Sort lines by their average y-coordinate (for horizontal) or x-coordinate (for vertical)
    lines_sorted = sorted(lines, key=lambda l: (l['coords'][1] + l['coords'][3]) / 2)
    
    merged_lines = []
    used = [False] * len(lines_sorted)
    
    for i, line1 in enumerate(lines_sorted):
        if used[i]:
            continue
        
        # Start a new group with this line
        group = [line1]
        used[i] = True
        
        x1_1, y1_1, x2_1, y2_1 = line1['coords']
        angle1 = line1['angle']
        
        # Find similar lines to merge
        for j, line2 in enumerate(lines_sorted[i+1:], start=i+1):
            if used[j]:
                continue
            
            x1_2, y1_2, x2_2, y2_2 = line2['coords']
            angle2 = line2['angle']
            
            # Check if lines are parallel (similar angles)
            angle_diff = abs(angle1 - angle2)
            if angle_diff > parallel_threshold:
                continue
            
            # Check if lines are close to each other
            # Calculate average distance between the lines
            mid_y1 = (y1_1 + y2_1) / 2
            mid_y2 = (y1_2 + y2_2) / 2
            distance = abs(mid_y1 - mid_y2)
            
            if distance < distance_threshold:
                group.append(line2)
                used[j] = True
        
        # Merge the group into a single line
        if group:
            # Average all coordinates
            avg_x1 = np.mean([l['coords'][0] for l in group])
            avg_y1 = np.mean([l['coords'][1] for l in group])
            avg_x2 = np.mean([l['coords'][2] for l in group])
            avg_y2 = np.mean([l['coords'][3] for l in group])
            avg_angle = np.mean([l['angle'] for l in group])
            total_length = sum([l['length'] for l in group])
            
            merged_line = {
                'coords': (avg_x1, avg_y1, avg_x2, avg_y2),
                'angle': avg_angle,
                'length': total_length,
                'num_merged': len(group)
            }
            merged_lines.append(merged_line)
    
    return merged_lines


def filter_and_classify_lines(lines: np.ndarray, 
                               frame_width: int, 
                               frame_height: int) -> Tuple[List[dict], List[dict]]:
    """
    Main function to filter and classify detected lines.
    
    This is the primary line processing pipeline that:
    1. Classifies lines into horizontal and vertical
    2. Merges similar/parallel lines
    3. Filters out weak detections
    4. Returns clean lists of yard lines and sidelines
    
    Args:
        lines: Raw lines detected by HoughLinesP
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Tuple of (horizontal_lines, vertical_lines) - cleaned and merged
    """
    if lines is None or len(lines) == 0:
        return [], []
    
    # Step 1: Classify lines into horizontal and vertical
    horizontal_lines, vertical_lines = classify_lines(lines)
    
    print(f"  Detected {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")
    
    # Step 2: Merge similar lines in each category
    merged_horizontal = merge_similar_lines(horizontal_lines, 
                                           parallel_threshold=config.PARALLEL_ANGLE_THRESHOLD,
                                           distance_threshold=config.HORIZONTAL_DISTANCE_THRESHOLD)
    
    merged_vertical = merge_similar_lines(vertical_lines,
                                         parallel_threshold=config.PARALLEL_ANGLE_THRESHOLD, 
                                         distance_threshold=config.VERTICAL_DISTANCE_THRESHOLD)
    
    print(f"  After merging: {len(merged_horizontal)} horizontal, {len(merged_vertical)} vertical lines")
    
    # Step 3: Sort by position for easier identification
    # Sort horizontal lines by y-coordinate (top to bottom)
    merged_horizontal.sort(key=lambda l: (l['coords'][1] + l['coords'][3]) / 2)
    
    # Sort vertical lines by x-coordinate (left to right)
    merged_vertical.sort(key=lambda l: (l['coords'][0] + l['coords'][2]) / 2)
    
    return merged_horizontal, merged_vertical


def visualize_detected_lines(frame: np.ndarray, 
                             horizontal_lines: List[dict], 
                             vertical_lines: List[dict]) -> np.ndarray:
    """
    Draw detected lines on the frame for visualization.
    
    Args:
        frame: Input frame
        horizontal_lines: List of horizontal line dictionaries
        vertical_lines: List of vertical line dictionaries
        
    Returns:
        Frame with lines drawn
    """
    vis_frame = frame.copy()
    
    # Draw horizontal lines (yard lines) using configured color
    for line in horizontal_lines:
        x1, y1, x2, y2 = [int(coord) for coord in line['coords']]
        cv2.line(vis_frame, (x1, y1), (x2, y2), 
                config.COLOR_HORIZONTAL_LINES, config.LINE_THICKNESS)
        # Draw small circle at midpoint
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(vis_frame, (mid_x, mid_y), config.POINT_RADIUS, 
                  config.COLOR_HORIZONTAL_LINES, -1)
    
    # Draw vertical lines (sidelines/hash marks) using configured color
    for line in vertical_lines:
        x1, y1, x2, y2 = [int(coord) for coord in line['coords']]
        cv2.line(vis_frame, (x1, y1), (x2, y2), 
                config.COLOR_VERTICAL_LINES, config.LINE_THICKNESS)
        # Draw small circle at midpoint
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(vis_frame, (mid_x, mid_y), config.POINT_RADIUS, 
                  config.COLOR_VERTICAL_LINES, -1)
    
    # Add legend
    cv2.putText(vis_frame, f"Horizontal (Yard): {len(horizontal_lines)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                config.COLOR_HORIZONTAL_LINES, 2)
    cv2.putText(vis_frame, f"Vertical (Side/Hash): {len(vertical_lines)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                config.COLOR_VERTICAL_LINES, 2)
    
    return vis_frame


# ============================================================================
# MAIN VIDEO PROCESSING LOOP
# ============================================================================

def process_video(video_path: str, output_path: Optional[str] = None):
    """
    Main video processing loop that applies line detection to each frame.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path to save output video
    """
    # Load video
    cap = load_video(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Output is a 2x2 grid, so dimensions are (2*width, 2*height)
        writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                (frame_width * 2, frame_height * 2))
        print(f"Output will be saved to: {output_path}")
        print(f"Output format: 2x2 grid showing all processing stages")
    
    frame_count = 0
    
    print("\n" + "="*60)
    print("Starting video processing...")
    print("="*60 + "\n")
    
    # Main processing loop
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("\nEnd of video reached or error reading frame.")
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}...")
        
        # ====================================================================
        # PART 2: FIELD SEGMENTATION (HSV Masking)
        # ====================================================================
        
        # Step 1: Create green field mask using HSV color space
        # This isolates only the playing field, removing crowd/stadium/trees
        print("  Stage 1: Creating field mask...")
        field_mask = create_field_mask(frame)
        
        # Step 2: Apply mask to frame
        # This zeros out everything except the field
        masked_frame = apply_mask_to_frame(frame, field_mask)
        
        # ====================================================================
        # PART 3: LINE DETECTION AND FILTERING (on Masked Field)
        # ====================================================================
        
        # Step 3: Preprocess the MASKED frame
        # Now edge detection only sees the field, not the background
        print("  Stage 2: Detecting lines on masked field...")
        gray, edges = preprocess_frame(masked_frame)
        
        # Step 4: Detect lines using Hough Transform
        raw_lines = detect_lines_hough(edges)
        
        # Step 5: Filter and classify lines
        horizontal_lines, vertical_lines = filter_and_classify_lines(
            raw_lines, frame_width, frame_height
        )
        
        # Step 6: Visualize detected lines on ORIGINAL frame (not masked)
        # This shows lines overlaid on the full original image
        vis_frame = visualize_detected_lines(frame, horizontal_lines, vertical_lines)
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        
        # Create comprehensive visualization with 4 panels:
        # Top-left: Original frame with detected lines
        # Top-right: Field mask (binary, showing what was detected as field)
        # Bottom-left: Masked frame (field only, before edge detection)
        # Bottom-right: Edge detection result
        
        # Convert grayscale images to BGR for consistent stacking
        field_mask_colored = cv2.cvtColor(field_mask, cv2.COLOR_GRAY2BGR)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Add labels to each panel
        label_color = (255, 255, 255)
        label_bg_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Label each panel
        vis_frame_labeled = vis_frame.copy()
        cv2.rectangle(vis_frame_labeled, (5, 5), (350, 45), label_bg_color, -1)
        cv2.putText(vis_frame_labeled, "1. Original + Detected Lines", 
                   (10, 35), font, font_scale, label_color, thickness)
        
        field_mask_labeled = field_mask_colored.copy()
        cv2.rectangle(field_mask_labeled, (5, 5), (250, 45), label_bg_color, -1)
        cv2.putText(field_mask_labeled, "2. Field Mask (HSV)", 
                   (10, 35), font, font_scale, label_color, thickness)
        
        masked_frame_labeled = masked_frame.copy()
        cv2.rectangle(masked_frame_labeled, (5, 5), (280, 45), label_bg_color, -1)
        cv2.putText(masked_frame_labeled, "3. Masked Field Only", 
                   (10, 35), font, font_scale, label_color, thickness)
        
        edges_labeled = edges_colored.copy()
        cv2.rectangle(edges_labeled, (5, 5), (220, 45), label_bg_color, -1)
        cv2.putText(edges_labeled, "4. Edge Detection", 
                   (10, 35), font, font_scale, label_color, thickness)
        
        # Create 2x2 grid
        top_row = np.hstack([vis_frame_labeled, field_mask_labeled])
        bottom_row = np.hstack([masked_frame_labeled, edges_labeled])
        full_display = np.vstack([top_row, bottom_row])
        
        # Resize for display if needed (using configured max width)
        display_width = config.DISPLAY_MAX_WIDTH
        scale = display_width / full_display.shape[1]
        if scale < 1:
            new_width = display_width
            new_height = int(full_display.shape[0] * scale)
            full_display_resized = cv2.resize(full_display, (new_width, new_height))
        else:
            full_display_resized = full_display
        
        # Show frame
        cv2.imshow('Football Field Tracker - Two-Stage Detection', full_display_resized)
        
        # Write to output if specified (write the 2x2 grid)
        if writer:
            writer.write(full_display)
        
        # ====================================================================
        # USER CONTROLS
        # ====================================================================
        
        # Wait for key press (1ms) - allows video to play
        # Press 'q' to quit, 'p' to pause
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nUser requested quit.")
            break
        elif key == ord('p'):
            print("\nPaused. Press any key to continue...")
            cv2.waitKey(0)
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete. Processed {frame_count} frames.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Use paths from configuration file
    # Modify config.py to change input/output paths and other parameters
    
    try:
        process_video(config.VIDEO_INPUT_PATH, config.VIDEO_OUTPUT_PATH)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

