"""
Distance Tracking Module
========================

Calculates and tracks the total distance traveled by each player using
bird's eye view coordinates.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import math


class DistanceTracker:
    """
    Tracks cumulative distance traveled for each tracked object.
    """
    
    def __init__(self, pixels_per_yard: float = 5.0):
        """
        Initialize distance tracker.
        
        Args:
            pixels_per_yard: Calibration factor (pixels per yard in bird's eye view)
        """
        self.pixels_per_yard = pixels_per_yard
        
        # Store last position for each track ID
        self.last_positions: Dict[int, Tuple[float, float]] = {}
        
        # Store cumulative distance for each track ID (in yards)
        self.distances: Dict[int, float] = {}
        
        # Store position history
        self.position_history: Dict[int, list] = {}
    
    def update(self, track_id: int, position: Tuple[float, float]) -> float:
        """
        Update distance for a track.
        
        Args:
            track_id: Unique track identifier
            position: Current position (x, y) in bird's eye view coordinates
            
        Returns:
            Distance moved this frame (in yards)
        """
        # Initialize if new track
        if track_id not in self.distances:
            self.distances[track_id] = 0.0
            self.last_positions[track_id] = position
            self.position_history[track_id] = [position]
            return 0.0
        
        # Get last position
        last_pos = self.last_positions[track_id]
        
        # Calculate distance moved (in pixels)
        dx = position[0] - last_pos[0]
        dy = position[1] - last_pos[1]
        pixel_distance = math.sqrt(dx*dx + dy*dy)
        
        # Convert to yards
        yard_distance = pixel_distance / self.pixels_per_yard
        
        # Update cumulative distance
        self.distances[track_id] += yard_distance
        
        # Update last position
        self.last_positions[track_id] = position
        
        # Store in history
        self.position_history[track_id].append(position)
        
        return yard_distance
    
    def batch_update(self, tracks_positions: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
        """
        Update distances for multiple tracks.
        
        Args:
            tracks_positions: Dictionary mapping track_id to (x, y) position
            
        Returns:
            Dictionary mapping track_id to distance moved this frame
        """
        frame_distances = {}
        
        for track_id, position in tracks_positions.items():
            distance = self.update(track_id, position)
            frame_distances[track_id] = distance
        
        return frame_distances
    
    def get_distance(self, track_id: int) -> float:
        """
        Get cumulative distance for a track.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Total distance traveled in yards (0.0 if track not found)
        """
        return self.distances.get(track_id, 0.0)
    
    def get_all_distances(self) -> Dict[int, float]:
        """
        Get all tracked distances.
        
        Returns:
            Dictionary mapping track_id to cumulative distance
        """
        return self.distances.copy()
    
    def get_position_history(self, track_id: int) -> list:
        """
        Get position history for a track.
        
        Args:
            track_id: Track identifier
            
        Returns:
            List of (x, y) positions
        """
        return self.position_history.get(track_id, [])
    
    def reset_track(self, track_id: int):
        """
        Reset distance tracking for a specific track.
        
        Args:
            track_id: Track to reset
        """
        if track_id in self.distances:
            del self.distances[track_id]
        if track_id in self.last_positions:
            del self.last_positions[track_id]
        if track_id in self.position_history:
            del self.position_history[track_id]
    
    def reset_all(self):
        """Reset all distance tracking."""
        self.distances.clear()
        self.last_positions.clear()
        self.position_history.clear()
    
    def set_calibration(self, pixels_per_yard: float):
        """
        Update calibration factor.
        
        Args:
            pixels_per_yard: New calibration factor
        """
        # Recalculate existing distances with new calibration
        old_calibration = self.pixels_per_yard
        self.pixels_per_yard = pixels_per_yard
        
        # Adjust all stored distances
        ratio = old_calibration / pixels_per_yard
        for track_id in self.distances:
            self.distances[track_id] *= ratio
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get aggregate statistics.
        
        Returns:
            Dictionary with statistics (max, min, avg, total)
        """
        if not self.distances:
            return {
                'max_distance': 0.0,
                'min_distance': 0.0,
                'avg_distance': 0.0,
                'total_distance': 0.0,
                'num_tracks': 0
            }
        
        distances = list(self.distances.values())
        
        return {
            'max_distance': max(distances),
            'min_distance': min(distances),
            'avg_distance': np.mean(distances),
            'total_distance': sum(distances),
            'num_tracks': len(distances)
        }

