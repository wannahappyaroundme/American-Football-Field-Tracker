import cv2
import numpy as np
from sklearn.cluster import KMeans
from config import ENABLE_CLIP_TEAM_CLASSIFICATION


class TeamClassifier:
    """
    A class to classify players into teams based on jersey color.
    Enhanced with optional CLIP-based classification for improved accuracy.
    """

    def __init__(self, clip_team_classifier=None):
        """
        Initialize the TeamClassifier.

        Args:
            clip_team_classifier: Optional CLIPTeamClassifier instance for semantic team classification
        """
        self.team_colors = {}  # Dict mapping team label to dominant color
        self.player_teams = {}  # Dict mapping track_id to team ('Team A', 'Team B', 'Referee')
        self.color_samples = []  # List to collect color samples for clustering
        self.clip_classifier = clip_team_classifier
        self.use_clip = ENABLE_CLIP_TEAM_CLASSIFICATION and clip_team_classifier is not None

        if self.use_clip:
            print("TeamClassifier initialized with CLIP-based classification")
        else:
            print("TeamClassifier initialized with color-based K-means clustering")

    def extract_jersey_color(self, crop_image):
        """
        Extract the dominant jersey color from a player crop image.

        Args:
            crop_image: Cropped image of a player (numpy array)

        Returns:
            Tuple (B, G, R) representing the dominant color
        """
        # Resize crop to speed up processing
        h, w = crop_image.shape[:2]

        # Focus on upper body (jersey area) - top 40% of the crop
        upper_body = crop_image[0:int(h*0.4), :]

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)

        # Reshape to list of pixels
        pixels = hsv.reshape(-1, 3)

        # Filter out very dark (shadows) and very bright (white) pixels
        # Keep only pixels with saturation > 30 and value between 30 and 220
        mask = (pixels[:, 1] > 30) & (pixels[:, 2] > 30) & (pixels[:, 2] < 220)
        filtered_pixels = pixels[mask]

        if len(filtered_pixels) == 0:
            # If no valid pixels, use all pixels
            filtered_pixels = pixels

        # Use K-means to find dominant color (k=1)
        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        dominant_hsv = kmeans.cluster_centers_[0]

        # Convert back to BGR
        dominant_hsv_img = np.uint8([[dominant_hsv]])
        dominant_bgr = cv2.cvtColor(dominant_hsv_img, cv2.COLOR_HSV2BGR)[0][0]

        return tuple(dominant_bgr)

    def collect_color_sample(self, track_id, crop_image):
        """
        Collect a color sample from a player for later clustering.

        Args:
            track_id: The track ID of the player
            crop_image: Cropped image of the player

        Returns:
            None
        """
        color = self.extract_jersey_color(crop_image)
        self.color_samples.append({
            'track_id': track_id,
            'color': color
        })

    def classify_teams(self, n_teams=4):
        """
        Classify all collected samples into teams using K-means clustering.

        Args:
            n_teams: Number of teams (default 4: Team A, Team B, Referee, Others)

        Returns:
            None
        """
        if len(self.color_samples) < 2:
            print(f"Warning: Not enough samples ({len(self.color_samples)}) to classify teams")
            return

        # Aggregate colors by track_id (average multiple samples per player)
        track_colors = {}
        for sample in self.color_samples:
            tid = sample['track_id']
            if tid not in track_colors:
                track_colors[tid] = []
            track_colors[tid].append(sample['color'])

        # Calculate average color per player
        player_avg_colors = {}
        for tid, colors in track_colors.items():
            avg_color = np.mean(colors, axis=0)
            player_avg_colors[tid] = avg_color

        # Extract colors for clustering
        track_ids = list(player_avg_colors.keys())
        colors = np.array([player_avg_colors[tid] for tid in track_ids])

        # Use fewer clusters if we don't have enough samples
        actual_n_teams = min(n_teams, len(track_ids))

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=actual_n_teams, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors)
        cluster_centers = kmeans.cluster_centers_

        # Identify special clusters
        referee_cluster = self._identify_referee_cluster(cluster_centers)
        others_cluster = self._identify_others_cluster(cluster_centers, colors, labels)

        # Assign team labels
        cluster_to_team = {}
        available_team_names = []

        # First assign special teams
        if referee_cluster is not None:
            cluster_to_team[referee_cluster] = 'Referee'

        if others_cluster is not None and others_cluster != referee_cluster:
            cluster_to_team[others_cluster] = 'Others'

        # Assign remaining clusters to Team A, Team B
        remaining_clusters = [i for i in range(actual_n_teams)
                            if i not in cluster_to_team]

        for idx, cluster in enumerate(remaining_clusters):
            if idx == 0:
                cluster_to_team[cluster] = 'Team A'
            elif idx == 1:
                cluster_to_team[cluster] = 'Team B'
            else:
                cluster_to_team[cluster] = 'Others'

        # Map track IDs to teams
        for tid, label in zip(track_ids, labels):
            self.player_teams[tid] = cluster_to_team[label]

        # Store team colors
        for cluster_id, team_name in cluster_to_team.items():
            self.team_colors[team_name] = tuple(cluster_centers[cluster_id])

        print(f"Team classification complete: {len(self.player_teams)} players classified")
        print(f"Teams identified: {set(self.player_teams.values())}")

        # Print team distribution
        team_counts = {}
        for team in self.player_teams.values():
            team_counts[team] = team_counts.get(team, 0) + 1
        print(f"Team distribution: {team_counts}")

    def _identify_referee_cluster(self, cluster_centers):
        """
        Identify which cluster likely represents referees based on color characteristics.

        Args:
            cluster_centers: Array of cluster center colors (BGR)

        Returns:
            Index of referee cluster, or None if can't determine
        """
        # Convert to HSV for better color analysis
        hsv_centers = []
        for bgr in cluster_centers:
            bgr_img = np.uint8([[bgr]])
            hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)[0][0]
            hsv_centers.append(hsv)

        hsv_centers = np.array(hsv_centers)

        # Referee jerseys often have:
        # 1. Low saturation (black/white stripes)
        # 2. Very low or very high value (black or white)

        saturations = hsv_centers[:, 1]
        values = hsv_centers[:, 2]

        # Find cluster with lowest saturation
        min_sat_idx = np.argmin(saturations)

        # If saturation is significantly lower than others, likely referee
        if saturations[min_sat_idx] < 50:
            return min_sat_idx

        return None  # Can't confidently identify referee

    def _identify_others_cluster(self, cluster_centers, all_colors, labels):
        """
        Identify which cluster likely represents "Others" (sideline staff, etc.)
        based on cluster size - typically the smallest cluster.

        Args:
            cluster_centers: Array of cluster center colors
            all_colors: All color samples
            labels: Cluster labels for each sample

        Returns:
            Index of "Others" cluster, or None if can't determine
        """
        # Count samples per cluster
        cluster_sizes = {}
        for label in labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

        # Find smallest cluster (likely sideline staff/others)
        if len(cluster_sizes) >= 4:
            smallest_cluster = min(cluster_sizes.keys(), key=lambda k: cluster_sizes[k])

            # If smallest cluster has < 20% of samples, it's likely "Others"
            total_samples = len(labels)
            if cluster_sizes[smallest_cluster] / total_samples < 0.2:
                return smallest_cluster

        return None

    def get_team(self, track_id):
        """
        Get the team assignment for a track ID.

        Args:
            track_id: The track ID of the player

        Returns:
            Team name string ('Team A', 'Team B', 'Referee', 'Others'), or 'Unknown'
        """
        # Check CLIP classifier first if available
        if self.use_clip and self.clip_classifier:
            clip_team = self.clip_classifier.get_team(track_id)
            if clip_team != 'Unknown':
                return clip_team

        # Fall back to color-based classification
        return self.player_teams.get(track_id, 'Unknown')

    def assign_team_with_clip(self, track_id, crop_image):
        """
        Assign team using CLIP classifier (if enabled) or fall back to color-based.

        Args:
            track_id: The track ID of the player
            crop_image: Cropped image of the player

        Returns:
            Tuple of (team_label, confidence):
                team_label: 'Team A', 'Team B', 'Referee', or 'Unknown'
                confidence: float confidence score (0.0 if using color-based)
        """
        if self.use_clip and self.clip_classifier:
            # Use CLIP classification
            team_label, team_color, confidence = self.clip_classifier.assign_team(track_id, crop_image)
            # Also store in local cache for backward compatibility
            self.player_teams[track_id] = team_label
            return team_label, confidence
        else:
            # Use color-based classification (collect samples for later clustering)
            self.collect_color_sample(track_id, crop_image)
            return 'Unknown', 0.0
