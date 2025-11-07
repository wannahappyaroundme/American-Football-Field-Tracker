import cv2
import numpy as np
from sklearn.cluster import KMeans
from config import (
    ENABLE_CLIP_TEAM_CLASSIFICATION,
    TEAM_A_HSV_RANGE,
    TEAM_B_HSV_RANGE,
    REFEREE_HSV_RANGE,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
    REFEREE_COLOR,
    UNKNOWN_COLOR
)


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

        # ⭐ NEW: Team freezing to prevent flickering (like tracker.py RobustTracker)
        self.frozen_teams = {}  # Dict mapping track_id to frozen team (immutable once set)
        self.team_confidence = {}  # Dict mapping track_id to number of consistent frames
        self.freeze_threshold = 2  # Freeze team after 2 consistent frames

        # ⭐ NEW: Pose-based team memory (선수 pose 기반 팀 정보 기억)
        # track_id가 바뀌어도 유사한 pose의 선수는 같은 팀!
        self.pose_memory = []  # List of (bbox, team_label) tuples
        self.pose_similarity_threshold = 0.5  # Pose 유사도 임계값

        if self.use_clip:
            print("TeamClassifier initialized with CLIP-based classification")
        else:
            print("TeamClassifier initialized with color-based K-means clustering")
        # print(f"Team freeze enabled: teams frozen after {self.freeze_threshold} consistent frames")  # CLIP 비활성화 시 불필요

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
        # ⭐ NEW: Check frozen teams first (highest priority - never changes!)
        if track_id in self.frozen_teams:
            return self.frozen_teams[track_id]

        # Check CLIP classifier first if available
        if self.use_clip and self.clip_classifier:
            clip_team = self.clip_classifier.get_team(track_id)
            if clip_team != 'Unknown':
                return clip_team

        # Fall back to color-based classification
        return self.player_teams.get(track_id, 'Unknown')

    def is_frozen(self, track_id):
        """
        Check if a track_id has a frozen team assignment.

        Args:
            track_id: The track ID to check

        Returns:
            bool: True if team is frozen, False otherwise
        """
        return track_id in self.frozen_teams

    def assign_team_with_clip(self, track_id, crop_image):
        """
        Assign team using CLIP classifier (if enabled) or fall back to color-based.
        ⭐ NEW: Implements team freezing logic to prevent flickering.

        Args:
            track_id: The track ID of the player
            crop_image: Cropped image of the player

        Returns:
            Tuple of (team_label, confidence):
                team_label: 'Team A', 'Team B', 'Referee', or 'Unknown'
                confidence: float confidence score (0.0 if using color-based)
        """
        # ⭐ NEW: If team is already frozen, return frozen team (never changes!)
        if track_id in self.frozen_teams:
            return self.frozen_teams[track_id], 1.0

        if self.use_clip and self.clip_classifier:
            # Use CLIP classification
            team_label, team_color, confidence = self.clip_classifier.assign_team(track_id, crop_image)

            # ⭐ NEW: Team freezing logic (like tracker.py RobustTracker)
            if team_label != 'Unknown':
                # Check if same team as previous frame
                previous_team = self.player_teams.get(track_id)

                if previous_team == team_label:
                    # Same team - increase confidence
                    self.team_confidence[track_id] = self.team_confidence.get(track_id, 0) + 1

                    # Freeze team if confidence threshold reached
                    if self.team_confidence[track_id] >= self.freeze_threshold:
                        self.frozen_teams[track_id] = team_label
                        print(f"  🔒 Team frozen for ID {track_id}: {team_label} (confidence: {self.team_confidence[track_id]} frames)")
                else:
                    # Different team - reset confidence
                    self.team_confidence[track_id] = 1
                    self.player_teams[track_id] = team_label

            # Also store in local cache for backward compatibility
            self.player_teams[track_id] = team_label
            return team_label, confidence
        else:
            # Use color-based classification (collect samples for later clustering)
            self.collect_color_sample(track_id, crop_image)
            return 'Unknown', 0.0

    def _calculate_pose_similarity(self, bbox1, bbox2):
        """
        선수 pose 유사도 계산 (위치 + 크기 + 종횡비)

        Args:
            bbox1, bbox2: [x1, y1, x2, y2]

        Returns:
            float: 유사도 (0-1, 높을수록 유사)
        """
        # 중심 거리 (정규화)
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        position_sim = max(0, 1 - distance / 300)  # 300 픽셀로 정규화

        # 크기 유사도
        size1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        size2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0

        # 종횡비 유사도
        aspect1 = (bbox1[2] - bbox1[0]) / (bbox1[3] - bbox1[1]) if (bbox1[3] - bbox1[1]) > 0 else 0
        aspect2 = (bbox2[2] - bbox2[0]) / (bbox2[3] - bbox2[1]) if (bbox2[3] - bbox2[1]) > 0 else 0
        aspect_sim = 1 - min(abs(aspect1 - aspect2) / max(aspect1, aspect2), 1) if max(aspect1, aspect2) > 0 else 0

        # 가중 평균 (위치 > 크기 > 종횡비)
        similarity = 0.5 * position_sim + 0.3 * size_ratio + 0.2 * aspect_sim

        return similarity

    def classify_team_instantly(self, track_id, crop_image, bbox):
        """
        ⭐ tracker.py 방식: 즉시 팀 분류 (HSV 범위 기반)
        ⭐ NEW: Pose 기반 팀 정보 재사용 (유사한 pose = 같은 선수 = 같은 팀)

        매 프레임마다 실행해도 빠름! K-means 없이 HSV 범위만 체크.

        Args:
            track_id: The track ID of the player
            crop_image: Cropped image of the player
            bbox: Bounding box [x1, y1, x2, y2] for pose checking

        Returns:
            Tuple of (team_label, team_color_bgr)
        """
        # ⭐ NEW: 먼저 Pose 기반으로 기존 팀 정보 찾기 (깜빡임 방지!)
        best_match_team = None
        best_similarity = self.pose_similarity_threshold  # 0.5 이상만 매칭

        for prev_bbox, prev_team in self.pose_memory:
            similarity = self._calculate_pose_similarity(bbox, prev_bbox)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_team = prev_team

        if best_match_team is not None:
            # 유사한 pose 발견! 기존 팀 정보 재사용
            self.player_teams[track_id] = best_match_team

            # Pose 정보 업데이트 (최신 bbox로)
            self.pose_memory = [(b, t) for b, t in self.pose_memory if t != best_match_team]
            self.pose_memory.append((bbox, best_match_team))

            # 팀 색상 반환
            if best_match_team == 'Team A':
                return best_match_team, TEAM_A_COLOR
            elif best_match_team == 'Team B':
                return best_match_team, TEAM_B_COLOR
            elif best_match_team == 'Referee':
                return best_match_team, REFEREE_COLOR
            else:
                return best_match_team, UNKNOWN_COLOR

        # ⭐ PHASE 2: Try CLIP first if enabled, fallback to HSV
        if self.use_clip and self.clip_classifier:
            team_label, team_color, confidence = self.clip_classifier.assign_team(track_id, crop_image)

            # ⭐ VERY STRICT: Referee needs MUCH higher confidence (only 1-3 in game!)
            # Players need 50%, but Referee needs 75% to avoid false positives
            min_confidence = 0.75 if team_label == 'Referee' else 0.5

            if team_label != 'Unknown' and confidence > min_confidence:
                # CLIP successful! Use its result
                self.player_teams[track_id] = team_label
                self.pose_memory.append((bbox, team_label))
                return team_label, team_color

        # ⭐ Fallback: HSV 기반으로 팀 분류
        # 저지 색상 추출 (tracker.py의 get_team_color 로직)
        hsv_color = self._extract_jersey_hsv(crop_image)

        if hsv_color is None:
            return 'Unknown', UNKNOWN_COLOR

        # HSV 범위로 즉시 팀 분류 (줄무늬 감지 포함)
        team_label, team_color = self._classify_by_hsv_range(hsv_color, crop_image)

        # 팀 정보 저장 (캐시)
        self.player_teams[track_id] = team_label

        # Pose 정보 저장 (다음 프레임에서 재사용)
        self.pose_memory.append((bbox, team_label))

        # 메모리 관리: 너무 많이 쌓이면 오래된 것 제거 (최근 50개만 유지)
        if len(self.pose_memory) > 50:
            self.pose_memory = self.pose_memory[-50:]

        return team_label, team_color

    def _detect_stripes(self, region):
        """
        줄무늬 패턴 감지 (심판 유니폼용)

        Args:
            region: 분석할 이미지 영역 (numpy array)

        Returns:
            bool: 줄무늬 패턴이 감지되면 True
        """
        if region.size == 0:
            return False

        # 그레이스케일 변환
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # 픽셀 값의 표준편차 계산 (줄무늬는 명암 변화가 크므로 표준편차가 높음)
        std_dev = np.std(gray)

        # ⭐ STRICTER: 표준편차가 70 이상이면 줄무늬로 판단 (심판만 1-3명!)
        # 일반 유니폼은 30-50, 약한 줄무늬 50-65, 명확한 줄무늬는 70 이상
        return std_dev > 70

    def _extract_jersey_hsv(self, crop_image):
        """
        tracker.py의 get_team_color() 함수 개선
        저지 영역에서 HSV 색상 추출 (상의만 집중 분석)
        """
        if crop_image.size == 0:
            return None

        h, w = crop_image.shape[:2]

        # ⭐ 상의 영역만 분석 (10%-40% 높이) - 하의 제외!
        # Team B (흰색 상의)와 Referee (줄무늬) 구별을 위해
        shirt_y1 = int(h * 0.10)
        shirt_y2 = int(h * 0.40)
        shirt_x1 = int(w * 0.2)
        shirt_x2 = int(w * 0.8)

        if shirt_y2 <= shirt_y1 or shirt_x2 <= shirt_x1:
            return None

        shirt_region = crop_image[shirt_y1:shirt_y2, shirt_x1:shirt_x2]

        if shirt_region.size == 0:
            return None

        # HSV 변환
        shirt_hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
        pixels = shirt_hsv.reshape(-1, 3).astype(np.float32)

        # 그림자/반사광 필터링
        brightness = pixels[:, 2]
        saturation = pixels[:, 1]
        mask = (brightness > 50) & (brightness < 210) & (saturation > 30)
        filtered_pixels = pixels[mask]

        if len(filtered_pixels) < 20:
            return None

        # K-means로 가장 채도 높은 색상 선택
        n_clusters = min(2, len(filtered_pixels))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(filtered_pixels)

        best_color = None
        best_saturation = 0

        for center in kmeans.cluster_centers_:
            if center[1] > best_saturation:
                best_saturation = center[1]
                best_color = center

        return best_color if best_color is not None else kmeans.cluster_centers_[0]

    def _classify_by_hsv_range(self, hsv_color, crop_image):
        """
        tracker.py의 classify_team_fixed() 함수 개선
        HSV 범위로 팀 분류 + 줄무늬 패턴 감지

        Args:
            hsv_color: HSV 색상 값
            crop_image: 선수 크롭 이미지 (줄무늬 감지용)

        Returns:
            Tuple of (team_label, team_color_bgr)
        """
        h, s, v = hsv_color

        # ⭐ 우선순위 1: 줄무늬 패턴 감지 (심판 전용!)
        # 상의 영역 추출
        if crop_image is not None and crop_image.size > 0:
            h_crop, w_crop = crop_image.shape[:2]
            shirt_y1 = int(h_crop * 0.10)
            shirt_y2 = int(h_crop * 0.40)
            shirt_x1 = int(w_crop * 0.2)
            shirt_x2 = int(w_crop * 0.8)

            if shirt_y2 > shirt_y1 and shirt_x2 > shirt_x1:
                shirt_region = crop_image[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
                if self._detect_stripes(shirt_region):
                    # 줄무늬 감지 → 심판!
                    return "Referee", REFEREE_COLOR

        # ⭐ 우선순위 2: Team A (노란색 - 채도 높음)
        (h_min_a, s_min_a, v_min_a), (h_max_a, s_max_a, v_max_a) = TEAM_A_HSV_RANGE
        if h_min_a <= h <= h_max_a and s_min_a <= s <= s_max_a and v_min_a <= v <= v_max_a:
            return "Team A", TEAM_A_COLOR

        # ⭐ 우선순위 3: Team B (흰색 - 매우 밝고 채도 낮음)
        (h_min_b, s_min_b, v_min_b), (h_max_b, s_max_b, v_max_b) = TEAM_B_HSV_RANGE
        if h_min_b <= h <= h_max_b and s_min_b <= s <= s_max_b and v_min_b <= v <= v_max_b:
            return "Team B", TEAM_B_COLOR

        # ⭐ 우선순위 4: Referee (HSV 범위 - 백업)
        (h_min_r, s_min_r, v_min_r), (h_max_r, s_max_r, v_max_r) = REFEREE_HSV_RANGE
        if h_min_r <= h <= h_max_r and s_min_r <= s <= s_max_r and v_min_r <= v <= v_max_r:
            return "Referee", REFEREE_COLOR

        return "Unknown", UNKNOWN_COLOR
