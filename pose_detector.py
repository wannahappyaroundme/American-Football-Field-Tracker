"""
Pose Detector for Football Play Analysis

This module handles pose-based detection enhancements for crouched players:
1. Detects player stances (standing, crouched, 3-point, 4-point)
2. Expands bounding boxes to include lower body for crouched players
3. Verifies PRE_SNAP set positions
4. Improves detection of players in low stances at scrimmage line
"""

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    POSE_MODEL_PATH,
    ENABLE_POSE_DETECTION,
    POSE_CONFIDENCE_THRESHOLD,
    POSE_FRAME_INTERVAL,
    ENABLE_STANCE_DETECTION,
    CROUCHED_KNEE_ANGLE_MAX,
    STANDING_KNEE_ANGLE_MIN,
    THREE_POINT_STANCE_HAND_THRESHOLD,
    EXPAND_BBOX_FOR_CROUCHED,
    BBOX_EXPANSION_RATIO,
    KEYPOINT_LEFT_SHOULDER,
    KEYPOINT_RIGHT_SHOULDER,
    KEYPOINT_LEFT_HIP,
    KEYPOINT_RIGHT_HIP,
    KEYPOINT_LEFT_KNEE,
    KEYPOINT_RIGHT_KNEE,
    KEYPOINT_LEFT_ANKLE,
    KEYPOINT_RIGHT_ANKLE,
    KEYPOINT_LEFT_WRIST,
    KEYPOINT_RIGHT_WRIST,
    KEYPOINT_CONFIDENCE_MIN
)


class PoseDetector:
    """
    Detects player poses to improve detection of crouched players.
    """

    def __init__(self):
        """
        Initialize the PoseDetector with YOLO pose model.
        """
        self.pose_model = None
        self.frame_count = 0

        if ENABLE_POSE_DETECTION:
            print(f"Loading pose model from {POSE_MODEL_PATH}...")
            self.pose_model = YOLO(POSE_MODEL_PATH)
            print("✓ Pose model loaded successfully")
        else:
            print("Pose detection disabled in config")

    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points (in degrees).

        Args:
            point1: First point (x, y) - e.g., shoulder
            point2: Middle point (x, y) - e.g., hip (vertex)
            point3: Third point (x, y) - e.g., knee

        Returns:
            Angle in degrees (0-180)
        """
        # Create vectors
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

        # Calculate angle using dot product
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def detect_stance(self, keypoints, confidences):
        """
        Detect player stance based on keypoint positions.

        Args:
            keypoints: Array of 17 keypoints (x, y) coordinates
            confidences: Array of 17 confidence values

        Returns:
            Dictionary with stance information:
            {
                'type': 'standing' | 'crouched' | '3-point' | '4-point' | 'unknown',
                'knee_angle': float,
                'is_set': bool (in set position for PRE_SNAP)
            }
        """
        stance_info = {
            'type': 'unknown',
            'knee_angle': None,
            'is_set': False
        }

        # Extract keypoints with confidence check
        def get_keypoint(idx):
            if confidences[idx] >= KEYPOINT_CONFIDENCE_MIN:
                return keypoints[idx]
            return None

        # Get critical keypoints
        left_shoulder = get_keypoint(KEYPOINT_LEFT_SHOULDER)
        right_shoulder = get_keypoint(KEYPOINT_RIGHT_SHOULDER)
        left_hip = get_keypoint(KEYPOINT_LEFT_HIP)
        right_hip = get_keypoint(KEYPOINT_RIGHT_HIP)
        left_knee = get_keypoint(KEYPOINT_LEFT_KNEE)
        right_knee = get_keypoint(KEYPOINT_RIGHT_KNEE)
        left_ankle = get_keypoint(KEYPOINT_LEFT_ANKLE)
        right_ankle = get_keypoint(KEYPOINT_RIGHT_ANKLE)
        left_wrist = get_keypoint(KEYPOINT_LEFT_WRIST)
        right_wrist = get_keypoint(KEYPOINT_RIGHT_WRIST)

        # Calculate knee angle (hip-knee-ankle)
        # Use the side with better visibility
        knee_angle = None

        if left_hip is not None and left_knee is not None and left_ankle is not None:
            knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        elif right_hip is not None and right_knee is not None and right_ankle is not None:
            knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        stance_info['knee_angle'] = knee_angle

        if knee_angle is None:
            return stance_info

        # Determine stance type based on knee angle
        if knee_angle < CROUCHED_KNEE_ANGLE_MAX:
            # Check for 3-point or 4-point stance
            # 3-point: one hand on ground
            # 4-point: both hands on ground

            hands_on_ground = 0
            ground_threshold_y = None

            # Estimate ground level from ankles
            if left_ankle is not None:
                ground_threshold_y = left_ankle[1]
            elif right_ankle is not None:
                ground_threshold_y = right_ankle[1]

            if ground_threshold_y is not None:
                # Check if wrists are near ground level
                if left_wrist is not None and left_wrist[1] > (ground_threshold_y - 50):  # Within 50 pixels of ground
                    hands_on_ground += 1
                if right_wrist is not None and right_wrist[1] > (ground_threshold_y - 50):
                    hands_on_ground += 1

                if hands_on_ground >= 2:
                    stance_info['type'] = '4-point'
                    stance_info['is_set'] = True
                elif hands_on_ground == 1:
                    stance_info['type'] = '3-point'
                    stance_info['is_set'] = True
                else:
                    stance_info['type'] = 'crouched'
                    stance_info['is_set'] = True  # Also considered set
            else:
                stance_info['type'] = 'crouched'
                stance_info['is_set'] = True

        elif knee_angle > STANDING_KNEE_ANGLE_MIN:
            stance_info['type'] = 'standing'
            stance_info['is_set'] = False  # Standing, not in set position
        else:
            # In between - partially crouched
            stance_info['type'] = 'crouched'
            stance_info['is_set'] = True

        return stance_info

    def expand_bbox_for_stance(self, bbox, keypoints, confidences, frame_height):
        """
        Expand bounding box to include full body for crouched players.

        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            keypoints: Array of 17 keypoints
            confidences: Array of 17 confidence values
            frame_height: Height of the frame

        Returns:
            Expanded bounding box [x1, y1, x2, y2]
        """
        if not EXPAND_BBOX_FOR_CROUCHED:
            return bbox

        # Detect stance
        stance = self.detect_stance(keypoints, confidences)

        # Only expand for crouched stances
        if stance['type'] not in ['crouched', '3-point', '4-point']:
            return bbox

        x1, y1, x2, y2 = bbox

        # Find lowest visible keypoint (ankle or knee)
        lowest_y = y2

        for idx in [KEYPOINT_LEFT_ANKLE, KEYPOINT_RIGHT_ANKLE,
                    KEYPOINT_LEFT_KNEE, KEYPOINT_RIGHT_KNEE]:
            if confidences[idx] >= KEYPOINT_CONFIDENCE_MIN:
                kp_y = keypoints[idx][1]
                if kp_y > lowest_y:
                    lowest_y = kp_y

        # Expand bbox downward if needed
        bbox_height = y2 - y1
        expansion = int(bbox_height * BBOX_EXPANSION_RATIO)
        new_y2 = min(y2 + expansion, frame_height - 1)

        # Also expand if keypoints are below current bbox
        if lowest_y > y2:
            new_y2 = max(new_y2, int(lowest_y + 10))  # Add small margin

        return [x1, y1, x2, new_y2]

    def process_frame(self, frame, person_detections):
        """
        Process frame with pose detection to enhance person detections.

        Args:
            frame: Video frame (numpy array)
            person_detections: List of person detection dicts with 'bbox', 'track_id'

        Returns:
            Enhanced person detections with:
            - Expanded bboxes for crouched players
            - Stance information added to each detection
        """
        self.frame_count += 1

        # Skip pose detection based on frame interval
        if self.frame_count % POSE_FRAME_INTERVAL != 0:
            return person_detections

        if not ENABLE_POSE_DETECTION or self.pose_model is None:
            return person_detections

        # Run pose detection on full frame
        pose_results = self.pose_model(frame, conf=POSE_CONFIDENCE_THRESHOLD, verbose=False)

        if len(pose_results) == 0 or pose_results[0].keypoints is None:
            return person_detections

        # Extract keypoints
        keypoints_data = pose_results[0].keypoints.data.cpu().numpy()  # Shape: (N, 17, 3) - N people, 17 keypoints, (x, y, conf)

        # Match pose detections with person detections
        enhanced_detections = []

        for person_det in person_detections:
            bbox = person_det['bbox']

            # Find matching pose detection (closest bbox IoU)
            best_match_idx = None
            best_iou = 0.3  # Minimum IoU threshold

            for idx, kp_data in enumerate(keypoints_data):
                # Compute bbox from keypoints
                valid_keypoints = kp_data[kp_data[:, 2] >= KEYPOINT_CONFIDENCE_MIN]
                if len(valid_keypoints) == 0:
                    continue

                kp_x = valid_keypoints[:, 0]
                kp_y = valid_keypoints[:, 1]
                kp_bbox = [kp_x.min(), kp_y.min(), kp_x.max(), kp_y.max()]

                # Calculate IoU
                iou = self._calculate_iou(bbox, kp_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx

            # If matched, enhance the detection
            if best_match_idx is not None:
                kp_data = keypoints_data[best_match_idx]
                keypoints = kp_data[:, :2]  # (x, y)
                confidences = kp_data[:, 2]  # confidence

                # Detect stance
                stance = self.detect_stance(keypoints, confidences)

                # Expand bbox if crouched
                expanded_bbox = self.expand_bbox_for_stance(
                    bbox, keypoints, confidences, frame.shape[0])

                # Add enhanced info to detection
                person_det['bbox'] = expanded_bbox
                person_det['stance'] = stance
                person_det['keypoints'] = keypoints
                person_det['keypoints_conf'] = confidences

            enhanced_detections.append(person_det)

        return enhanced_detections

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            IoU value (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def verify_presnap_set(self, person_detections):
        """
        Verify if players are in set position for PRE_SNAP state.

        Args:
            person_detections: List of person detections with stance info

        Returns:
            Dictionary with:
            {
                'all_set': bool,
                'set_count': int,
                'total_count': int,
                'set_percentage': float
            }
        """
        total_count = 0
        set_count = 0

        for det in person_detections:
            if 'stance' in det:
                total_count += 1
                if det['stance']['is_set']:
                    set_count += 1

        set_percentage = (set_count / total_count * 100) if total_count > 0 else 0

        # Consider "all set" if 80% or more players are in set position
        all_set = set_percentage >= 80.0

        return {
            'all_set': all_set,
            'set_count': set_count,
            'total_count': total_count,
            'set_percentage': set_percentage
        }
