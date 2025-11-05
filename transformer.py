import numpy as np
import cv2
from config import (
    HOMOGRAPHY_MATRIX_PATH,
    PIXELS_PER_YARD_BEV,
    BEV_LEFT_YARD_LINE,
    BEV_DIRECTION,
    BEV_FIELD_X_MIN,
    BEV_FIELD_X_MAX,
    BEV_FIELD_Y_MIN,
    BEV_FIELD_Y_MAX
)


class ViewTransformer:
    """
    A class to transform points from the original camera view to a bird's eye view (BEV)
    using a homography matrix.
    """

    def __init__(self):
        """
        Initialize the ViewTransformer by loading the homography matrix from file.
        """
        self.homography_matrix = np.load(HOMOGRAPHY_MATRIX_PATH)
        print(f"Loaded homography matrix from {HOMOGRAPHY_MATRIX_PATH}")

    def transform_point(self, point_tuple):
        """
        Transform a point from the original view to the bird's eye view.

        Args:
            point_tuple: A tuple (x, y) representing a point in the original view

        Returns:
            A tuple (x', y') representing the transformed point in the BEV
        """
        # Format the point as a numpy array with shape (1, 1, 2)
        point = np.array([[point_tuple]], dtype=np.float32)

        # Apply perspective transformation
        transformed_point = cv2.perspectiveTransform(point, self.homography_matrix)

        # Extract and return the transformed coordinates as a tuple
        x_prime = float(transformed_point[0][0][0])
        y_prime = float(transformed_point[0][0][1])

        return (x_prime, y_prime)

    def get_foot_position(self, bbox):
        """
        Get the foot position (center-bottom point) from a bounding box.

        Args:
            bbox: A list [x1, y1, x2, y2] representing the bounding box

        Returns:
            A tuple (x, y) representing the center-bottom point of the bounding box
        """
        x1, y1, x2, y2 = bbox

        # Calculate center-bottom point
        center_x = (x1 + x2) / 2
        bottom_y = y2

        return (center_x, bottom_y)

    def bev_to_yard_line(self, bev_pos):
        """
        Convert BEV position to yard line information.

        Args:
            bev_pos: A tuple (x, y) representing position in BEV coordinates

        Returns:
            A dictionary with 'side' ('OWN' or 'OPP') and 'yard' (int)
        """
        x_bev, y_bev = bev_pos

        # Clamp x_bev to valid BEV range [0, 1000]
        x_bev = max(0, min(1000, x_bev))

        # Calculate yard distance from left edge of BEV
        yards_from_left = x_bev / PIXELS_PER_YARD_BEV

        # Adjust based on direction and reference line
        if BEV_DIRECTION == 'LEFT_TO_RIGHT':
            # Offense moves from left to right
            # Left side is OWN territory, right side is OPP territory
            yard_line = BEV_LEFT_YARD_LINE + yards_from_left
        else:
            # Offense moves from right to left
            # Right side is OWN territory, left side is OPP territory
            total_yards = 1000 / PIXELS_PER_YARD_BEV  # 50 yards
            yard_line = BEV_LEFT_YARD_LINE + (total_yards - yards_from_left)

        # Clamp yard_line to valid field range [0, 100]
        yard_line = max(0, min(100, yard_line))

        # Determine side and yard number
        # Assuming 100-yard field: OWN 0-50, then OPP 50-0
        if yard_line <= 50:
            side = 'OWN'
            yard = int(round(yard_line))
        else:
            side = 'OPP'
            yard = int(round(100 - yard_line))

        # Ensure yard is within valid range
        yard = max(0, min(50, yard))

        return {'side': side, 'yard': yard}

    def is_on_field(self, bev_pos):
        """
        Check if a BEV position is within the valid field boundaries.

        Args:
            bev_pos: A tuple (x, y) representing position in BEV coordinates

        Returns:
            Boolean: True if position is on field, False otherwise
        """
        x, y = bev_pos

        return (BEV_FIELD_X_MIN <= x <= BEV_FIELD_X_MAX and
                BEV_FIELD_Y_MIN <= y <= BEV_FIELD_Y_MAX)
