"""
Ball Carrier Detection Module
==============================

Identifies which player has possession of the football using proximity analysis.
"""

import numpy as np
from typing import Optional, Tuple, List
import math


class BallCarrierDetector:
    """
    Detects which player is carrying the ball based on proximity.
    """
    
    def __init__(self, max_distance: float = 80.0):
        """
        Initialize ball carrier detector.
        
        Args:
            max_distance: Maximum distance (pixels) for ball carrier assignment
        """
        self.max_distance = max_distance
        self.current_carrier_id = None
        self.carrier_history = []  # Track ball carrier over time
    
    def get_bbox_centroid(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Calculate centroid of bounding box.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Centroid (cx, cy)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance in pixels
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def identify_ball_carrier(self,
                             player_tracks: np.ndarray,
                             ball_track: Optional[np.ndarray] = None) -> Optional[int]:
        """
        Identify which player is carrying the ball.
        
        Algorithm:
        1. If no ball detected, return None
        2. Calculate distance from ball to each player
        3. Player with minimum distance < threshold is the carrier
        
        Args:
            player_tracks: Array of player tracks [[x1, y1, x2, y2, track_id], ...]
            ball_track: Ball track [x1, y1, x2, y2, track_id] or None
            
        Returns:
            Track ID of ball carrier, or None if no carrier
        """
        # No ball detected
        if ball_track is None or len(player_tracks) == 0:
            self.current_carrier_id = None
            return None
        
        # Get ball centroid
        ball_bbox = ball_track[:4]
        ball_centroid = self.get_bbox_centroid(ball_bbox)
        
        # Calculate distances to all players
        min_distance = float('inf')
        carrier_id = None
        
        for track in player_tracks:
            # Extract bbox and ID
            player_bbox = track[:4]
            player_id = int(track[4])
            
            # Calculate centroid
            player_centroid = self.get_bbox_centroid(player_bbox)
            
            # Calculate distance
            distance = self.calculate_distance(ball_centroid, player_centroid)
            
            # Check if this is the closest player
            if distance < min_distance:
                min_distance = distance
                carrier_id = player_id
        
        # Only assign carrier if within threshold
        if min_distance <= self.max_distance:
            self.current_carrier_id = carrier_id
            self.carrier_history.append({
                'frame': len(self.carrier_history),
                'carrier_id': carrier_id,
                'distance': min_distance
            })
        else:
            self.current_carrier_id = None
        
        return self.current_carrier_id
    
    def get_current_carrier(self) -> Optional[int]:
        """
        Get current ball carrier ID.
        
        Returns:
            Track ID of current carrier or None
        """
        return self.current_carrier_id
    
    def get_carrier_history(self) -> List[dict]:
        """
        Get history of ball carrier assignments.
        
        Returns:
            List of carrier history records
        """
        return self.carrier_history
    
    def reset(self):
        """Reset detector state."""
        self.current_carrier_id = None
        self.carrier_history = []

