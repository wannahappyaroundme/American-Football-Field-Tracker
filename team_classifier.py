"""
Team Classification Module
===========================

Automatically classifies players into teams (Team A, Team B, Referees)
based on jersey color clustering and game context (formation analysis).
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from collections import Counter


class TeamClassifier:
    """
    Classifies players into teams using color clustering and contextual rules.
    """
    
    def __init__(self,
                 num_teams: int = 3,
                 jersey_sample_top: float = 0.2,
                 jersey_sample_bottom: float = 0.6,
                 init_frames: int = 30,
                 update_interval: int = 60,
                 crouch_threshold: float = 0.7,
                 los_tolerance: int = 5):
        """
        Initialize team classifier.
        
        Args:
            num_teams: Number of clusters (Team A, Team B, Referees)
            jersey_sample_top: Top percentage of bbox to sample
            jersey_sample_bottom: Bottom percentage of bbox to sample
            init_frames: Frames to collect before initial clustering
            update_interval: Re-cluster every N frames
            crouch_threshold: Confidence for crouching detection
            los_tolerance: Tolerance for line of scrimmage detection
        """
        self.num_teams = num_teams
        self.jersey_sample_top = jersey_sample_top
        self.jersey_sample_bottom = jersey_sample_bottom
        self.init_frames = init_frames
        self.update_interval = update_interval
        self.crouch_threshold = crouch_threshold
        self.los_tolerance = los_tolerance
        
        self.color_samples = []
        self.kmeans = None
        self.team_labels = {0: 'Unknown', 1: 'Unknown', 2: 'Unknown'}
        self.frames_processed = 0
        self.initialized = False
    
    def extract_jersey_color(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract dominant jersey color from bounding box.
        
        Args:
            frame: Full frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Dominant color as RGB array or None
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Ensure valid bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Calculate sampling region (upper body only)
        bbox_height = y2 - y1
        sample_y1 = int(y1 + bbox_height * self.jersey_sample_top)
        sample_y2 = int(y1 + bbox_height * self.jersey_sample_bottom)
        
        if sample_y2 <= sample_y1:
            return None
        
        # Extract region
        region = frame[sample_y1:sample_y2, x1:x2]
        
        if region.size == 0:
            return None
        
        # Convert to RGB
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        
        # Reshape to list of pixels
        pixels = region_rgb.reshape(-1, 3).astype(np.float32)
        
        # Remove very dark (shadows) and very bright (glare) pixels
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 30) & (brightness < 225)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            return None
        
        # Get dominant color using K-means (k=1)
        kmeans_color = KMeans(n_clusters=1, n_init=10, random_state=42)
        kmeans_color.fit(filtered_pixels)
        dominant_color = kmeans_color.cluster_centers_[0]
        
        return dominant_color
    
    def collect_colors(self, frame: np.ndarray, tracks: List[Tuple]) -> bool:
        """
        Collect jersey colors from current frame.
        
        Args:
            frame: Current frame
            tracks: List of tracks [(x1, y1, x2, y2, track_id), ...]
            
        Returns:
            True if ready to cluster, False otherwise
        """
        for track in tracks:
            if len(track) < 5:
                continue
            
            bbox = track[:4]
            color = self.extract_jersey_color(frame, bbox)
            
            if color is not None:
                self.color_samples.append(color)
        
        self.frames_processed += 1
        
        # Check if ready to initialize
        if not self.initialized and len(self.color_samples) >= self.init_frames * 5:
            return True
        
        return False
    
    def fit_clusters(self) -> bool:
        """
        Fit K-means clusters to collected color samples.
        
        Returns:
            True if clustering succeeded
        """
        if len(self.color_samples) < self.num_teams * 10:
            return False
        
        # Convert to array
        colors_array = np.array(self.color_samples)
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.num_teams, n_init=10, random_state=42)
        self.kmeans.fit(colors_array)
        
        self.initialized = True
        
        return True
    
    def label_clusters_by_formation(self,
                                    frame: np.ndarray,
                                    tracks: List[Tuple],
                                    pose_data: Dict[int, Dict]) -> bool:
        """
        Label clusters (Team A, Team B, Referee) using formation analysis.
        
        Detects pre-snap formation by finding crouching players,
        establishes line of scrimmage, and assigns teams based on position.
        
        Args:
            frame: Current frame
            tracks: List of tracks
            pose_data: Dictionary mapping track_id to pose data
            
        Returns:
            True if labeling succeeded
        """
        if not self.initialized or self.kmeans is None:
            return False
        
        # Extract track data with team predictions
        track_info = []
        for track in tracks:
            if len(track) < 5:
                continue
            
            x1, y1, x2, y2, track_id = track
            bbox = (x1, y1, x2, y2)
            
            # Get jersey color
            color = self.extract_jersey_color(frame, bbox)
            if color is None:
                continue
            
            # Predict team cluster
            team_cluster = self.kmeans.predict([color])[0]
            
            # Get pose data
            pose = pose_data.get(int(track_id))
            
            # Calculate foot position (bottom center of bbox)
            foot_x = (x1 + x2) / 2
            foot_y = y2
            
            track_info.append({
                'track_id': track_id,
                'bbox': bbox,
                'foot_pos': (foot_x, foot_y),
                'team_cluster': team_cluster,
                'pose': pose
            })
        
        if len(track_info) < 10:  # Need enough players
            return False
        
        # Find crouching players
        crouching_players = []
        for info in track_info:
            if info['pose'] and info['pose']['analysis']['is_crouching']:
                if info['pose']['analysis']['confidence'] >= self.crouch_threshold:
                    crouching_players.append(info)
        
        if len(crouching_players) < 5:  # Need offensive line
            return False
        
        # Find line of scrimmage (average Y position of crouching players)
        crouch_y_positions = [p['foot_pos'][1] for p in crouching_players]
        line_of_scrimmage = np.mean(crouch_y_positions)
        
        # Classify players by position relative to line of scrimmage
        offensive_players = []
        defensive_players = []
        neutral_players = []
        
        for info in track_info:
            foot_y = info['foot_pos'][1]
            
            if abs(foot_y - line_of_scrimmage) < self.los_tolerance:
                neutral_players.append(info)
            elif foot_y < line_of_scrimmage:
                offensive_players.append(info)  # Above line (closer to camera)
            else:
                defensive_players.append(info)  # Below line (farther from camera)
        
        # Count team clusters in each group
        if len(offensive_players) > 0 and len(defensive_players) > 0:
            offense_clusters = [p['team_cluster'] for p in offensive_players]
            defense_clusters = [p['team_cluster'] for p in defensive_players]
            
            # Most common cluster in offense = Team A (offense)
            offense_counter = Counter(offense_clusters)
            team_a_cluster = offense_counter.most_common(1)[0][0]
            
            # Most common cluster in defense = Team B (defense)
            defense_counter = Counter(defense_clusters)
            team_b_cluster = defense_counter.most_common(1)[0][0]
            
            # Remaining cluster = Referees
            all_clusters = set(range(self.num_teams))
            assigned = {team_a_cluster, team_b_cluster}
            referee_cluster = list(all_clusters - assigned)[0] if len(assigned) == 2 else 2
            
            # Update labels
            self.team_labels[team_a_cluster] = 'Team A'
            self.team_labels[team_b_cluster] = 'Team B'
            self.team_labels[referee_cluster] = 'Referee'
            
            return True
        
        return False
    
    def classify_tracks(self, frame: np.ndarray, tracks: List[Tuple]) -> Dict[int, str]:
        """
        Classify all tracks into teams.
        
        Args:
            frame: Current frame
            tracks: List of tracks
            
        Returns:
            Dictionary mapping track_id to team name
        """
        if not self.initialized or self.kmeans is None:
            return {}
        
        classifications = {}
        
        for track in tracks:
            if len(track) < 5:
                continue
            
            x1, y1, x2, y2, track_id = track
            bbox = (x1, y1, x2, y2)
            
            # Get jersey color
            color = self.extract_jersey_color(frame, bbox)
            if color is None:
                classifications[int(track_id)] = 'Unknown'
                continue
            
            # Predict team cluster
            team_cluster = self.kmeans.predict([color])[0]
            
            # Map to team label
            team_name = self.team_labels.get(team_cluster, 'Unknown')
            classifications[int(track_id)] = team_name
        
        return classifications
    
    def update(self, frame: np.ndarray, tracks: List[Tuple], pose_data: Dict[int, Dict]) -> Dict[int, str]:
        """
        Main update method for team classification.
        
        Args:
            frame: Current frame
            tracks: List of tracks
            pose_data: Dictionary of pose data
            
        Returns:
            Dictionary mapping track_id to team name
        """
        # Collect colors
        ready_to_cluster = self.collect_colors(frame, tracks)
        
        # Initial clustering
        if ready_to_cluster and not self.initialized:
            self.fit_clusters()
            # Try to label using formation
            self.label_clusters_by_formation(frame, tracks, pose_data)
        
        # Periodic re-clustering
        elif self.initialized and self.frames_processed % self.update_interval == 0:
            # Re-fit with recent samples
            recent_samples = self.color_samples[-500:]  # Last 500 samples
            if len(recent_samples) >= self.num_teams * 10:
                self.color_samples = recent_samples
                self.fit_clusters()
        
        # Classify all tracks
        return self.classify_tracks(frame, tracks)
    
    def get_team_color(self, team_name: str) -> Tuple[int, int, int]:
        """
        Get visualization color for team.
        
        Args:
            team_name: Team name
            
        Returns:
            BGR color tuple
        """
        color_map = {
            'Team A': (0, 0, 255),      # Red
            'Team B': (255, 0, 0),      # Blue
            'Referee': (255, 255, 255),  # White
            'Unknown': (128, 128, 128)   # Gray
        }
        return color_map.get(team_name, (128, 128, 128))

