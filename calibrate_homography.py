import cv2
import numpy as np
from config import VIDEO_INPUT_PATH, HOMOGRAPHY_MATRIX_PATH

# Global list to store source points
src_points = []

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
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the clicked point to src_points
        src_points.append((x, y))

        # Draw a circle on the frame at the clicked point
        frame = params['frame']
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Display the updated frame
        cv2.imshow('Calibration', frame)

if __name__ == "__main__":
    # Open the video and read the first frame
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read video from {VIDEO_INPUT_PATH}")
        cap.release()
        exit(1)

    # Create a copy of the frame for display
    display_frame = frame.copy()

    # Create a window and set mouse callback
    cv2.namedWindow('Calibration')
    params = {'frame': display_frame}
    cv2.setMouseCallback('Calibration', click_event, params)

    # Display instructions
    instruction_text = "Click 4 points on the field (e.g., corners of the hash marks) and press 's' to save."
    print(instruction_text)

    # Put text on the frame
    cv2.putText(display_frame, "Click 4 points, then press 's'",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Calibration', display_frame)

    # Wait for keypress in a loop
    while True:
        key = cv2.waitKey(1) & 0xFF

        # If 's' is pressed and exactly 4 points have been selected
        if key == ord('s') and len(src_points) == 4:
            # Define the 4 corresponding destination points (BEV coordinates)
            dst_points = np.array([[0, 500], [1000, 500], [1000, 0], [0, 0]], dtype=np.float32)

            # Convert src_points to numpy array
            src_points_array = np.array(src_points, dtype=np.float32)

            # Calculate the homography matrix
            matrix = cv2.getPerspectiveTransform(src_points_array, dst_points)

            # Save the matrix
            np.save(HOMOGRAPHY_MATRIX_PATH, matrix)

            print("Homography matrix saved.")
            print(f"Source points: {src_points}")
            print(f"Destination points: {dst_points.tolist()}")
            break

        # Press 'q' to quit without saving
        elif key == ord('q'):
            print("Calibration cancelled.")
            break

    # Close all windows and release video capture
    cv2.destroyAllWindows()
    cap.release()
