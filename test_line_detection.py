"""
Test script for line detection functionality.
Creates a synthetic football field image to test the line detection pipeline.
"""

import cv2
import numpy as np
from football_field_tracker import (
    create_field_mask,
    apply_mask_to_frame,
    preprocess_frame,
    detect_lines_hough,
    filter_and_classify_lines,
    visualize_detected_lines
)


def create_synthetic_field(width=1920, height=1080):
    """
    Create a synthetic football field image for testing.
    
    This simulates a zoomed-in view of a football field with:
    - Yard lines (horizontal white lines)
    - Hash marks (short vertical lines)
    - Green grass background
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Synthetic field image
    """
    # Create green background (grass)
    field = np.zeros((height, width, 3), dtype=np.uint8)
    field[:, :] = (34, 139, 34)  # Green color (BGR)
    
    # Add some texture/noise to simulate grass
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    field = np.clip(field.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw yard lines (horizontal white lines)
    line_color = (255, 255, 255)  # White
    line_thickness = 8
    
    # Draw 5 yard lines across the frame
    num_lines = 5
    spacing = height // (num_lines + 1)
    
    for i in range(1, num_lines + 1):
        y = spacing * i
        cv2.line(field, (0, y), (width, y), line_color, line_thickness)
    
    # Draw hash marks (short vertical lines)
    hash_thickness = 6
    hash_length = 60
    hash_spacing = width // 8
    
    for i in range(1, 8):
        x = hash_spacing * i
        for j in range(1, num_lines + 1):
            y = spacing * j
            cv2.line(field, (x, y - hash_length//2), 
                    (x, y + hash_length//2), line_color, hash_thickness)
    
    # Draw sidelines (vertical lines on left and right)
    sideline_x_left = width // 6
    sideline_x_right = width - width // 6
    cv2.line(field, (sideline_x_left, 0), (sideline_x_left, height), 
            line_color, line_thickness)
    cv2.line(field, (sideline_x_right, 0), (sideline_x_right, height), 
            line_color, line_thickness)
    
    return field


def test_line_detection_synthetic():
    """
    Test line detection on synthetic field image.
    """
    print("="*60)
    print("Testing Line Detection on Synthetic Field")
    print("="*60)
    
    # Create synthetic field
    field = create_synthetic_field()
    print("\nCreated synthetic field image")
    
    # Apply line detection pipeline with masking
    print("\n1. Creating field mask (HSV)...")
    field_mask = create_field_mask(field)
    
    print("2. Applying mask to frame...")
    masked_field = apply_mask_to_frame(field, field_mask)
    
    print("3. Preprocessing masked frame...")
    gray, edges = preprocess_frame(masked_field)
    
    print("4. Detecting lines with Hough Transform...")
    raw_lines = detect_lines_hough(edges)
    print(f"   Detected {len(raw_lines) if raw_lines is not None else 0} raw line segments")
    
    print("5. Filtering and classifying lines...")
    horizontal_lines, vertical_lines = filter_and_classify_lines(
        raw_lines, field.shape[1], field.shape[0]
    )
    
    print(f"\nResults:")
    print(f"  Horizontal lines (yard lines): {len(horizontal_lines)}")
    print(f"  Vertical lines (sidelines/hash marks): {len(vertical_lines)}")
    
    # Visualize results
    print("\n6. Visualizing results...")
    vis_frame = visualize_detected_lines(field, horizontal_lines, vertical_lines)
    
    # Show results in 2x2 grid
    field_mask_colored = cv2.cvtColor(field_mask, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    top_row = np.hstack([vis_frame, field_mask_colored])
    bottom_row = np.hstack([masked_field, edges_colored])
    comparison = np.vstack([top_row, bottom_row])
    
    # Resize for display
    scale = 1200 / comparison.shape[1]
    new_width = int(comparison.shape[1] * scale)
    new_height = int(comparison.shape[0] * scale)
    comparison_display = cv2.resize(comparison, (new_width, new_height))
    
    cv2.imshow('Line Detection Test - Synthetic Field', comparison_display)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


def test_line_detection_on_image(image_path):
    """
    Test line detection on a real image/frame.
    
    Args:
        image_path: Path to image file
    """
    print("="*60)
    print(f"Testing Line Detection on: {image_path}")
    print("="*60)
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"\nLoaded image: {frame.shape[1]}x{frame.shape[0]}")
    
    # Apply line detection pipeline with masking
    print("\n1. Creating field mask (HSV)...")
    field_mask = create_field_mask(frame)
    
    print("2. Applying mask to frame...")
    masked_frame = apply_mask_to_frame(frame, field_mask)
    
    print("3. Preprocessing masked frame...")
    gray, edges = preprocess_frame(masked_frame)
    
    print("4. Detecting lines with Hough Transform...")
    raw_lines = detect_lines_hough(edges)
    print(f"   Detected {len(raw_lines) if raw_lines is not None else 0} raw line segments")
    
    print("5. Filtering and classifying lines...")
    horizontal_lines, vertical_lines = filter_and_classify_lines(
        raw_lines, frame.shape[1], frame.shape[0]
    )
    
    print(f"\nResults:")
    print(f"  Horizontal lines (yard lines): {len(horizontal_lines)}")
    print(f"  Vertical lines (sidelines/hash marks): {len(vertical_lines)}")
    
    # Print details of detected lines
    print("\nHorizontal Lines:")
    for i, line in enumerate(horizontal_lines):
        x1, y1, x2, y2 = line['coords']
        print(f"  {i+1}. Y-position: ~{int((y1+y2)/2)}, Angle: {line['angle']:.1f}°")
    
    print("\nVertical Lines:")
    for i, line in enumerate(vertical_lines):
        x1, y1, x2, y2 = line['coords']
        print(f"  {i+1}. X-position: ~{int((x1+x2)/2)}, Angle: {line['angle']:.1f}°")
    
    # Visualize results
    print("\n6. Visualizing results...")
    vis_frame = visualize_detected_lines(frame, horizontal_lines, vertical_lines)
    
    # Show results in 2x2 grid
    field_mask_colored = cv2.cvtColor(field_mask, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    top_row = np.hstack([vis_frame, field_mask_colored])
    bottom_row = np.hstack([masked_frame, edges_colored])
    comparison = np.vstack([top_row, bottom_row])
    
    # Resize for display
    max_width = 1600
    if comparison.shape[1] > max_width:
        scale = max_width / comparison.shape[1]
        new_width = int(comparison.shape[1] * scale)
        new_height = int(comparison.shape[0] * scale)
        comparison_display = cv2.resize(comparison, (new_width, new_height))
    else:
        comparison_display = comparison
    
    cv2.imshow('Line Detection Test', comparison_display)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("Football Field Line Detection Test")
    print("="*60 + "\n")
    
    if len(sys.argv) > 1:
        # Test on provided image
        image_path = sys.argv[1]
        test_line_detection_on_image(image_path)
    else:
        # Test on synthetic field
        print("No image provided. Testing on synthetic field.")
        print("Usage: python test_line_detection.py [image_path]\n")
        test_line_detection_synthetic()


