"""
Pose Estimation Module
======================

Provides unified interface for pose estimation using MediaPipe or YOLO-Pose.
Extracts keypoints for each tracked player to enable posture analysis
(e.g., crouching, running, standing).
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
import mediapipe as mp


class PoseEstimator:
    """
    Unified pose estimation interface supporting multiple backends.
    """
    
    def __init__(self, model: str = 'mediapipe', confidence: float = 0.5):
        """
        Initialize pose estimator.
        
        Args:
            model: Pose model to use ('mediapipe' or 'yolo_pose')
            confidence: Minimum confidence threshold
        """
        self.model = model
        self.confidence = confidence
        
        if model == 'mediapipe':
            self._init_mediapipe()
        elif model == 'yolo_pose':
            self._init_yolo_pose()
        else:
            raise ValueError(f"Unknown pose model: {model}")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Pose."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks=True,
            min_detection_confidence=self.confidence,
            min_tracking_confidence=self.confidence
        )
    
    def _init_yolo_pose(self):
        """Initialize YOLO-Pose (if implemented)."""
        # Placeholder for YOLO-Pose implementation
        raise NotImplementedError("YOLO-Pose not yet implemented")
    
    def estimate_pose(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Estimate pose for a single bounding box.
        
        Args:
            frame: Full frame image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Dictionary with pose data or None if detection failed
        """
        if self.model == 'mediapipe':
            return self._estimate_mediapipe(frame, bbox)
        elif self.model == 'yolo_pose':
            return self._estimate_yolo_pose(frame, bbox)
    
    def _estimate_mediapipe(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Estimate pose using MediaPipe.
        
        Args:
            frame: Full frame
            bbox: Bounding box coordinates
            
        Returns:
            Pose data dictionary or None
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop to bounding box with padding
        padding = 10
        x1_crop = max(0, x1 - padding)
        y1_crop = max(0, y1 - padding)
        x2_crop = min(frame.shape[1], x2 + padding)
        y2_crop = min(frame.shape[0], y2 + padding)
        
        crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if crop.size == 0:
            return None
        
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.pose.process(crop_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract keypoints (convert relative coordinates to absolute)
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            # Convert from crop coordinates to frame coordinates
            x = landmark.x * crop.shape[1] + x1_crop
            y = landmark.y * crop.shape[0] + y1_crop
            z = landmark.z  # Depth (relative)
            visibility = landmark.visibility
            
            keypoints.append({
                'x': x,
                'y': y,
                'z': z,
                'visibility': visibility
            })
        
        # Analyze pose
        pose_analysis = self._analyze_pose(keypoints)
        
        return {
            'keypoints': keypoints,
            'analysis': pose_analysis,
            'model': 'mediapipe'
        }
    
    def _estimate_yolo_pose(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """Estimate pose using YOLO-Pose (placeholder)."""
        raise NotImplementedError()
    
    def _analyze_pose(self, keypoints: List[Dict]) -> Dict:
        """
        Analyze pose to detect specific postures.
        
        Args:
            keypoints: List of keypoint dictionaries
            
        Returns:
            Analysis dictionary with detected postures
        """
        analysis = {
            'is_crouching': False,
            'is_standing': False,
            'is_running': False,
            'confidence': 0.0
        }
        
        # MediaPipe keypoint indices
        # 0: nose, 11/12: shoulders, 23/24: hips, 25/26: knees, 27/28: ankles
        
        if len(keypoints) < 33:  # MediaPipe has 33 landmarks
            return analysis
        
        # Get key points
        nose = keypoints[0]
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        left_knee = keypoints[25]
        right_knee = keypoints[26]
        
        # Calculate average shoulder and hip heights
        shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        hip_y = (left_hip['y'] + right_hip['y']) / 2
        knee_y = (left_knee['y'] + right_knee['y']) / 2
        
        # Calculate torso length
        torso_length = hip_y - shoulder_y
        
        # Calculate leg bend (hip to knee distance vs expected)
        upper_leg_length = knee_y - hip_y
        
        # Visibility checks
        min_visibility = 0.5
        all_visible = all([
            left_shoulder['visibility'] > min_visibility,
            right_shoulder['visibility'] > min_visibility,
            left_hip['visibility'] > min_visibility,
            right_hip['visibility'] > min_visibility,
            left_knee['visibility'] > min_visibility,
            right_knee['visibility'] > min_visibility
        ])
        
        if not all_visible:
            return analysis
        
        # Crouching detection
        # In a crouch, knees are bent and torso is compressed
        if torso_length > 0:
            leg_bend_ratio = upper_leg_length / torso_length
            
            # Crouching: legs are bent (ratio < 1.0)
            if leg_bend_ratio < 0.8:
                analysis['is_crouching'] = True
                analysis['confidence'] = min(left_knee['visibility'], right_knee['visibility'])
            
            # Standing: legs are relatively straight (ratio > 1.0)
            elif leg_bend_ratio > 1.2:
                analysis['is_standing'] = True
                analysis['confidence'] = min(left_knee['visibility'], right_knee['visibility'])
            
            # Running: detected by leg asymmetry (one leg forward, one back)
            # Calculate leg z-depth difference
            left_leg_z = left_knee['z']
            right_leg_z = right_knee['z']
            z_diff = abs(left_leg_z - right_leg_z)
            
            if z_diff > 0.1:  # Significant depth difference
                analysis['is_running'] = True
                analysis['confidence'] = min(left_knee['visibility'], right_knee['visibility'])
        
        return analysis
    
    def batch_estimate(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Optional[Dict]]:
        """
        Estimate poses for multiple bounding boxes.
        
        Args:
            frame: Full frame
            bboxes: List of bounding boxes
            
        Returns:
            List of pose dictionaries (None for failed detections)
        """
        results = []
        for bbox in bboxes:
            pose_data = self.estimate_pose(frame, bbox)
            results.append(pose_data)
        return results
    
    def draw_pose(self, frame: np.ndarray, pose_data: Dict, color: Tuple[int, int, int] = (0, 255, 0)):
        """
        Draw pose keypoints on frame.
        
        Args:
            frame: Frame to draw on
            pose_data: Pose data from estimate_pose()
            color: Drawing color (BGR)
        """
        if not pose_data or 'keypoints' not in pose_data:
            return
        
        keypoints = pose_data['keypoints']
        
        # Draw keypoints
        for kp in keypoints:
            if kp['visibility'] > 0.5:
                x, y = int(kp['x']), int(kp['y'])
                cv2.circle(frame, (x, y), 3, color, -1)
        
        # Draw skeleton connections (simplified)
        connections = [
            (11, 12),  # Shoulders
            (11, 13),  # Left arm
            (13, 15),  # Left forearm
            (12, 14),  # Right arm
            (14, 16),  # Right forearm
            (11, 23),  # Left torso
            (12, 24),  # Right torso
            (23, 24),  # Hips
            (23, 25),  # Left thigh
            (25, 27),  # Left shin
            (24, 26),  # Right thigh
            (26, 28),  # Right shin
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                
                if start_kp['visibility'] > 0.5 and end_kp['visibility'] > 0.5:
                    start_pos = (int(start_kp['x']), int(start_kp['y']))
                    end_pos = (int(end_kp['x']), int(end_kp['y']))
                    cv2.line(frame, start_pos, end_pos, color, 2)
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'pose'):
            self.pose.close()

