import cv2
import numpy as np
from config import VIDEO_INPUT_PATH, HOMOGRAPHY_MATRIX_PATH

# Global list to store source points
src_points = []
# Store the original frame for redrawing
original_frame = None

def redraw_frame(frame):
    """
    Redraw the frame with all current points, connecting lines, and annotations.

    Args:
        frame: The original frame to draw on

    Returns:
        The annotated frame
    """
    display_frame = frame.copy()

    # Draw connecting lines between points
    if len(src_points) > 1:
        for i in range(len(src_points)):
            if i < len(src_points) - 1:
                # Draw line to next point
                cv2.line(display_frame, src_points[i], src_points[i + 1],
                        (0, 255, 255), 2)  # Yellow line

        # If we have 4 points, close the rectangle
        if len(src_points) == 4:
            cv2.line(display_frame, src_points[3], src_points[0],
                    (0, 255, 255), 2)  # Yellow line

            # Draw filled polygon with transparency to show transformation area
            overlay = display_frame.copy()
            points_array = np.array(src_points, dtype=np.int32)
            cv2.fillPoly(overlay, [points_array], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)

    # Draw circles and numbers for each point
    for i, point in enumerate(src_points):
        # Draw circle
        cv2.circle(display_frame, point, 8, (0, 255, 0), -1)  # Green filled circle
        cv2.circle(display_frame, point, 10, (255, 255, 255), 2)  # White border

        # Draw point number
        number_text = str(i + 1)
        cv2.putText(display_frame, number_text,
                   (point[0] + 15, point[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, number_text,
                   (point[0] + 15, point[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    # Draw instructions on the frame
    instructions = [
        "Click 4 field corners (e.g., hash marks)",
        f"Points: {len(src_points)}/4",
        "'s' = Save | 'r' = Reset | 'q' = Quit"
    ]

    y_offset = 30
    for instruction in instructions:
        # Draw text background
        (text_width, text_height), _ = cv2.getTextSize(
            instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(display_frame, (5, y_offset - 25),
                     (text_width + 15, y_offset + 5), (0, 0, 0), -1)
        # Draw text
        cv2.putText(display_frame, instruction,
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 35

    # Show destination coordinates preview
    if len(src_points) == 4:
        dest_text = "Dest: [0,500] [1000,500] [1000,0] [0,0]"
        cv2.putText(display_frame, dest_text,
                   (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return display_frame

def click_event(event, x, y, flags, params):
    """
    Mouse callback function to capture clicked points on the frame.

    Args:
        event: Mouse event type
        x: X coordinate of the mouse click
        y: Y coordinate of the mouse click
        flags: Additional flags
        params: Additional parameters (contains the frame)
    """
    global src_points, original_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # Only allow 4 points maximum
        if len(src_points) < 4:
            # Append the clicked point to src_points
            src_points.append((x, y))
            print(f"Point {len(src_points)} clicked: ({x}, {y})")

            # Redraw the frame with all annotations
            display_frame = redraw_frame(original_frame)
            cv2.imshow('Calibration', display_frame)

if __name__ == "__main__":
    # Open the video and read the first frame
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read video from {VIDEO_INPUT_PATH}")
        cap.release()
        exit(1)

    # Store the original frame globally for redrawing
    # Note: original_frame is already declared as global at module level (line 8)
    original_frame = frame.copy()

    # Create a window and set mouse callback
    cv2.namedWindow('Calibration')
    params = {'frame': original_frame}
    cv2.setMouseCallback('Calibration', click_event, params)

    # Display instructions
    print("=" * 60)
    print("HOMOGRAPHY CALIBRATION TOOL")
    print("=" * 60)
    print("Instructions:")
    print("  1. Click 4 corners of a known field area (e.g., hash marks)")
    print("  2. Points will be connected with yellow lines")
    print("  3. Press 's' to save when 4 points are selected")
    print("  4. Press 'r' to reset and start over")
    print("  5. Press 'q' to quit without saving")
    print("=" * 60)

    # Show initial frame with instructions
    display_frame = redraw_frame(original_frame)
    cv2.imshow('Calibration', display_frame)

    # Wait for keypress in a loop
    while True:
        key = cv2.waitKey(1) & 0xFF

        # If 's' is pressed and exactly 4 points have been selected
        if key == ord('s'):
            if len(src_points) == 4:
                # Define the 4 corresponding destination points (BEV coordinates)
                dst_points = np.array([[0, 500], [1000, 500], [1000, 0], [0, 0]], dtype=np.float32)

                # Convert src_points to numpy array
                src_points_array = np.array(src_points, dtype=np.float32)

                # Calculate the homography matrix
                matrix = cv2.getPerspectiveTransform(src_points_array, dst_points)

                # Save the matrix
                np.save(HOMOGRAPHY_MATRIX_PATH, matrix)

                print("\n" + "=" * 60)
                print("✓ Homography matrix saved successfully!")
                print("=" * 60)
                print(f"Source points: {src_points}")
                print(f"Destination points: {dst_points.tolist()}")
                print(f"Saved to: {HOMOGRAPHY_MATRIX_PATH}")
                print("=" * 60)
                break
            else:
                print(f"⚠ Please select exactly 4 points (currently {len(src_points)} selected)")

        # Press 'r' to reset points and start over
        elif key == ord('r'):
            src_points.clear()
            print("\n↺ Points reset. Click 4 new points.")
            display_frame = redraw_frame(original_frame)
            cv2.imshow('Calibration', display_frame)

        # Press 'q' to quit without saving
        elif key == ord('q'):
            print("\n✗ Calibration cancelled.")
            break

    # Close all windows and release video capture
    cv2.destroyAllWindows()
    cap.release()
