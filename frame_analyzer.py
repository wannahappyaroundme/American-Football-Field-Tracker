"""
Dynamic Frame Processing Module
================================

Implements intelligent frame skipping based on motion detection.
Only processes frames when significant motion is detected, achieving
real-time performance without sacrificing important data.

Methods:
- MSE (Mean Squared Error): Fast, simple pixel-wise comparison
- Optical Flow: More accurate motion detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class FrameChangeDetector:
    """
    Detects significant changes between consecutive frames to determine
    when to run expensive analysis (YOLO, pose estimation, etc.)
    """
    
    def __init__(self, method: str = 'mse', threshold: float = 0.015):
        """
        Initialize frame change detector.
        
        Args:
            method: Detection method ('mse' or 'optical_flow')
            threshold: Threshold for change detection
        """
        self.method = method
        self.threshold = threshold
        self.prev_frame_gray = None
        
        # Optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def detect_change(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if significant change occurred between current and previous frame.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            Tuple of (has_significant_change, change_magnitude)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame - always process
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return True, 1.0
        
        # Detect change based on method
        if self.method == 'mse':
            change_magnitude = self._detect_mse(gray)
        elif self.method == 'optical_flow':
            change_magnitude = self._detect_optical_flow(gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Update previous frame
        self.prev_frame_gray = gray
        
        # Determine if change is significant
        has_change = change_magnitude > self.threshold
        
        return has_change, change_magnitude
    
    def _detect_mse(self, gray: np.ndarray) -> float:
        """
        Detect change using Mean Squared Error (MSE).
        
        Fast method that compares pixel intensity differences.
        
        Args:
            gray: Current grayscale frame
            
        Returns:
            Normalized MSE value (0.0 - 1.0)
        """
        # Calculate MSE
        mse = np.mean((gray.astype(float) - self.prev_frame_gray.astype(float)) ** 2)
        
        # Normalize to 0-1 range (assuming max difference is 255)
        normalized_mse = mse / (255.0 ** 2)
        
        return normalized_mse
    
    def _detect_optical_flow(self, gray: np.ndarray) -> float:
        """
        Detect change using Dense Optical Flow.
        
        More accurate method that detects actual motion vectors.
        
        Args:
            gray: Current grayscale frame
            
        Returns:
            Average flow magnitude (normalized)
        """
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame_gray,
            gray,
            None,
            **self.flow_params
        )
        
        # Calculate flow magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get average magnitude
        avg_magnitude = np.mean(mag)
        
        return avg_magnitude
    
    def reset(self):
        """Reset detector (e.g., at scene changes)."""
        self.prev_frame_gray = None


class AdaptiveFrameProcessor:
    """
    Manages dynamic frame processing with motion-based skipping.
    
    This class decides when to run full analysis vs. when to reuse
    previous results, achieving real-time performance.
    """
    
    def __init__(self, 
                 enable_dynamic: bool = True,
                 method: str = 'mse',
                 threshold: float = 0.015,
                 min_process_interval: int = 1,
                 max_skip_frames: int = 10):
        """
        Initialize adaptive frame processor.
        
        Args:
            enable_dynamic: Enable dynamic processing
            method: Change detection method
            threshold: Change detection threshold
            min_process_interval: Minimum frames between processing
            max_skip_frames: Maximum consecutive frames to skip
        """
        self.enable_dynamic = enable_dynamic
        self.detector = FrameChangeDetector(method, threshold)
        self.min_process_interval = min_process_interval
        self.max_skip_frames = max_skip_frames
        
        self.frames_since_process = 0
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
    
    def should_process_frame(self, frame: np.ndarray) -> Tuple[bool, dict]:
        """
        Determine if current frame should be processed.
        
        Args:
            frame: Current frame
            
        Returns:
            Tuple of (should_process, info_dict)
        """
        self.total_frames += 1
        
        # If dynamic processing disabled, always process
        if not self.enable_dynamic:
            self.processed_frames += 1
            return True, {
                'reason': 'dynamic_disabled',
                'change_magnitude': 0.0
            }
        
        # Enforce minimum interval
        if self.frames_since_process < self.min_process_interval:
            self.frames_since_process += 1
            self.skipped_frames += 1
            return False, {
                'reason': 'min_interval',
                'change_magnitude': 0.0
            }
        
        # Enforce maximum skip (safety)
        if self.frames_since_process >= self.max_skip_frames:
            self.processed_frames += 1
            self.frames_since_process = 0
            return True, {
                'reason': 'max_skip_reached',
                'change_magnitude': 1.0
            }
        
        # Detect change
        has_change, magnitude = self.detector.detect_change(frame)
        
        if has_change:
            self.processed_frames += 1
            self.frames_since_process = 0
            return True, {
                'reason': 'motion_detected',
                'change_magnitude': magnitude
            }
        else:
            self.skipped_frames += 1
            self.frames_since_process += 1
            return False, {
                'reason': 'no_motion',
                'change_magnitude': magnitude
            }
    
    def get_statistics(self) -> dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        if self.total_frames == 0:
            return {
                'total_frames': 0,
                'processed_frames': 0,
                'skipped_frames': 0,
                'skip_ratio': 0.0,
                'speedup_factor': 1.0
            }
        
        return {
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'skipped_frames': self.skipped_frames,
            'skip_ratio': self.skipped_frames / self.total_frames,
            'speedup_factor': self.total_frames / max(self.processed_frames, 1)
        }

