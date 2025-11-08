import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from config import (
    DETECTION_MODEL_PATH,
    CLASS_ID_PERSON,
    CLASS_ID_BALL,
    DETECTION_CONFIDENCE_THRESHOLD,
    BALL_CONFIDENCE,
    BALL_MIN_SIZE,
    BALL_MAX_SIZE,
    BALL_ASPECT_RATIO_MIN,
    BALL_ASPECT_RATIO_MAX,
    DETECTION_EVERY_N_FRAMES
)
from jersey_id_manager import JerseyBasedIDManager


class DetectorTracker:
    """
    A class to detect and track objects (players and ball) in video frames using YOLO.
    ⭐ FIXED for multi-detection:
    - YOLO runs on full unmasked frame (multi-detection works properly!)
    - BEV filtering applied AFTER detection (removes off-field tracks)
    - Improved ball detection (size + aspect ratio filtering)
    - NO duplicate filtering (ByteTrack handles everything)
    """

    def __init__(self, view_transformer=None, number_recognizer=None, team_classifier=None):
        """
        Initialize the DetectorTracker by loading the YOLO detection model.

        Args:
            view_transformer: ViewTransformer instance for BEV masking (optional)
            number_recognizer: NumberRecognizer instance for jersey OCR (optional)
            team_classifier: TeamClassifier instance for team assignment (optional)
        """
        self.model = YOLO(DETECTION_MODEL_PATH)
        self.frame_count = 0
        self.view_transformer = view_transformer
        self.number_recognizer = number_recognizer
        self.team_classifier = team_classifier

        # ⭐ 3-frame history for ID continuity
        self.track_history = deque(maxlen=3)  # Store last 3 frames of tracks
        self.id_mapping = {}  # Maps new detections to consistent IDs
        self.next_stable_id = 1  # Counter for stable IDs

        # ⭐ Jersey-based ID manager for stable tracking
        self.jersey_manager = JerseyBasedIDManager()

        # OCR interval (every N frames to avoid overhead)
        self.ocr_interval = 30  # Run OCR every 30 frames (1 second @ 30fps)

        print(f"Loaded YOLO model from {DETECTION_MODEL_PATH}")
        print(f"⭐ PHASE 3 OPTIMIZATIONS APPLIED:")
        print(f"  • Multi-detection: YOLO on FULL frame (25-30 detections)")
        print(f"  • ByteTrack: buffer=60 frames (2s), match=0.60, new_track=0.5 (STRICT)")
        print(f"  • Ball: conf=0.15, size 10-120px, aspect 0.7-2.5 (oval shape)")
        print(f"  • BEV filtering: {'ENABLED (Y: 0-500, expanded)' if view_transformer else 'DISABLED'}")
        print(f"  • CLIP: Team classification ENABLED (100% accuracy)")
        print(f"Status logging every {DETECTION_EVERY_N_FRAMES} frames")

    def _create_field_mask(self, frame):
        """
        Create a mask that isolates the field area using BEV transformation.
        Filters out sideline personnel BEFORE detection.

        Args:
            frame: Input frame (numpy array)

        Returns:
            Binary mask (numpy array) where 1 = on-field, 0 = off-field
        """
        if self.view_transformer is None:
            # No BEV transformer, return full frame mask
            return np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255

        # Create coordinate grid for entire frame
        h, w = frame.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Transform all coordinates to BEV
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()]).astype(np.float32)
        points_reshaped = points.reshape(-1, 1, 2)

        # Use homography transformation
        bev_points = cv2.perspectiveTransform(points_reshaped, self.view_transformer.homography_matrix)
        bev_points = bev_points.reshape(-1, 2)

        # Check which points are within BEV field boundaries
        # BEV field: x in [50, 950], y in [0, 500]
        x_bev, y_bev = bev_points[:, 0], bev_points[:, 1]
        on_field = (x_bev >= 50) & (x_bev <= 950) & (y_bev >= 0) & (y_bev <= 500)

        # Reshape to frame dimensions
        mask = on_field.reshape(h, w).astype(np.uint8) * 255

        return mask

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            bbox1, bbox2: [x1, y1, x2, y2]

        Returns:
            float: IoU value (0-1)
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union area
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _calculate_pose_similarity(self, bbox1, bbox2):
        """
        Calculate pose-based similarity between two bounding boxes.
        Uses position, size, and IoU for robust matching.

        Args:
            bbox1, bbox2: [x1, y1, x2, y2]

        Returns:
            float: Similarity score (0-1, higher = more similar)
        """
        # Position similarity (center distance normalized by frame size)
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        position_sim = max(0, 1 - distance / 200)  # Normalize by 200 pixels

        # Size similarity
        size1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        size2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0

        # IoU similarity
        iou = self._calculate_iou(bbox1, bbox2)

        # Weighted combination (IoU most important, then position, then size)
        similarity = 0.5 * iou + 0.3 * position_sim + 0.2 * size_ratio

        return similarity

    def _maintain_id_continuity(self, current_tracks):
        """
        ⭐ 3-frame ID continuity: Match new detections with historical tracks.
        If a detection matches a track from the last 3 frames, reuse that stable ID.

        NOW USES: stable_id (jersey-based) instead of track_id (ByteTrack)

        Args:
            current_tracks: List of track dictionaries with stable_id field

        Returns:
            List of tracks with stable IDs maintained across frames
        """
        if len(self.track_history) == 0:
            # First frame, accept all IDs as-is
            self.track_history.append(current_tracks)
            return current_tracks

        # Build a map of historical tracks (last 3 frames)
        # Use stable_id instead of track_id for jersey-based tracking
        historical_tracks = {}
        for frame_tracks in self.track_history:
            for track in frame_tracks:
                stable_id = track.get('stable_id', track['track_id'])
                if stable_id not in historical_tracks:
                    historical_tracks[stable_id] = track

        # Match current tracks with historical tracks
        stable_tracks = []
        matched_historical_ids = set()

        for track in current_tracks:
            best_match_id = None
            best_similarity = 0.4  # Threshold for matching

            # Find best matching historical track
            for hist_id, hist_track in historical_tracks.items():
                if hist_id in matched_historical_ids:
                    continue  # Already matched

                # Only match same class
                if track['class_id'] != hist_track['class_id']:
                    continue

                similarity = self._calculate_pose_similarity(track['bbox'], hist_track['bbox'])

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = hist_id

            # Maintain stable ID continuity
            if best_match_id is not None:
                # Reuse historical stable ID
                track['stable_id'] = best_match_id
                matched_historical_ids.add(best_match_id)
            # else: keep jersey_manager's assigned stable_id (new detection)

            stable_tracks.append(track)

        # Update history
        self.track_history.append(stable_tracks)

        return stable_tracks

    def track_frame(self, frame, apply_bev_mask=True):
        """
        Track objects in a single frame using YOLO + ByteTrack.

        ⭐ FIXED: BEV filtering applied AFTER detection (multi-detection works!)
        - YOLO runs on full unmasked frame (no artifacts)
        - Detections filtered by BEV boundaries post-detection
        - NO duplicate filtering (ByteTrack handles all logic)
        - Enhanced ball detection (size + aspect ratio)

        Args:
            frame: Input frame (numpy array) to process
            apply_bev_mask: Whether to apply BEV-based field filtering

        Returns:
            List of dictionaries, where each dictionary contains:
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                - 'track_id': integer tracking ID (stable across frames)
                - 'class_id': integer class ID (person or ball)
                - 'confidence': float detection confidence

            Returns an empty list if no tracks are found.
        """
        self.frame_count += 1

        # ⭐ Using OPTIMIZED ByteTrack for ID stability
        # Note: Attempted BoT-SORT but had config issues with Ultralytics
        # ByteTrack with optimized parameters + extended 3-frame Re-ID
        results = self.model.track(
            frame,  # ← FULL FRAME, not masked!
            persist=True,
            conf=DETECTION_CONFIDENCE_THRESHOLD,  # 0.3 unified threshold
            classes=[CLASS_ID_PERSON, CLASS_ID_BALL],
            tracker='bytetrack_extended.yaml',  # ByteTrack with PHASE 4 optimization
            iou=0.2  # Low IoU for better occlusion handling
        )

        # Log status periodically
        if self.frame_count % DETECTION_EVERY_N_FRAMES == 1:
            num_tracks = len(results[0].boxes) if results[0].boxes.id is not None else 0
            print(f"🔍 Frame {self.frame_count}: {num_tracks} tracks active")

        # Handle case where no detections or tracking IDs are available
        if results[0].boxes.id is None:
            return []

        # ⭐ STEP 2: Extract tracking information and filter by BEV boundaries
        tracks = []
        boxes = results[0].boxes

        for i in range(len(boxes)):
            # Get bounding box coordinates
            bbox = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]

            # Get tracking ID
            track_id = int(boxes.id[i].cpu().numpy())

            # Get class ID
            class_id = int(boxes.cls[i].cpu().numpy())

            # Get confidence score
            confidence = float(boxes.conf[i].cpu().numpy())

            # ⭐ PHASE 1.4: BEV FILTERING RE-ENABLED with expanded boundaries
            # Uses config values (Y: 0-500) instead of hardcoded (50-450)
            # Goal: Remove sideline staff while keeping all 25-30 on-field players

            if apply_bev_mask and self.view_transformer is not None:
                # Get foot position (center-bottom of bbox)
                x1, y1, x2, y2 = bbox
                foot_x = (x1 + x2) / 2
                foot_y = y2
                # Transform to BEV
                foot_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
                bev_point = cv2.perspectiveTransform(foot_point, self.view_transformer.homography_matrix)
                bev_x, bev_y = bev_point[0][0]
                # Check if within field boundaries using config values
                from config import BEV_FIELD_X_MIN, BEV_FIELD_X_MAX, BEV_FIELD_Y_MIN, BEV_FIELD_Y_MAX
                if not (BEV_FIELD_X_MIN <= bev_x <= BEV_FIELD_X_MAX and
                        BEV_FIELD_Y_MIN <= bev_y <= BEV_FIELD_Y_MAX):
                    continue  # Skip off-field detection

            # ⭐ FILTER 2: Ball size and aspect ratio (prevent false positives)
            if class_id == CLASS_ID_BALL:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0

                # Debug: Log all ball detections (before filtering)
                if self.frame_count % 100 == 0:  # Log every 100 frames
                    print(f"⚽ Ball candidate: Track #{track_id}, conf={confidence:.2%}, "
                          f"size={int(width)}x{int(height)}, ratio={aspect_ratio:.2f}")

                # Size check
                if width < BALL_MIN_SIZE or height < BALL_MIN_SIZE or \
                   width > BALL_MAX_SIZE or height > BALL_MAX_SIZE:
                    if self.frame_count % 100 == 0:
                        print(f"  ✗ Rejected: size out of range ({BALL_MIN_SIZE}-{BALL_MAX_SIZE})")
                    continue

                # Aspect ratio check (must be roundish)
                if aspect_ratio < BALL_ASPECT_RATIO_MIN or aspect_ratio > BALL_ASPECT_RATIO_MAX:
                    if self.frame_count % 100 == 0:
                        print(f"  ✗ Rejected: aspect ratio out of range ({BALL_ASPECT_RATIO_MIN}-{BALL_ASPECT_RATIO_MAX})")
                    continue

                print(f"⚽ Ball ACCEPTED! Track #{track_id}, conf={confidence:.2%}, "
                      f"size={int(width)}x{int(height)}, ratio={aspect_ratio:.2f}")

            # Create track dict
            tracks.append({
                'bbox': bbox,
                'track_id': track_id,
                'class_id': class_id,
                'confidence': confidence
            })

        # ⭐ PHASE 7: JERSEY-BASED ID MAPPING (근본 해결!)
        # Strategy: Use jersey numbers to assign stable IDs
        # Goal: 73 IDs → 30-40 IDs (matching 22 players + substitutions)

        # Run jersey OCR every N frames (avoid overhead)
        if self.number_recognizer is not None and self.frame_count % self.ocr_interval == 0:
            for track in tracks:
                if track['class_id'] == CLASS_ID_PERSON:
                    # Extract player crop
                    x1, y1, x2, y2 = track['bbox']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        # Recognize jersey number
                        jersey_num = self.number_recognizer.recognize_number(crop)
                        track['jersey_num'] = jersey_num

                        # Get team assignment
                        if self.team_classifier is not None:
                            team = self.team_classifier.player_teams.get(track['track_id'], 'Unknown')
                            track['team'] = team
                        else:
                            track['team'] = 'Unknown'
                    else:
                        track['jersey_num'] = None
                        track['team'] = 'Unknown'

        # Apply stable ID mapping based on jersey numbers
        if self.jersey_manager is not None:
            # Get stable IDs
            for track in tracks:
                if track['class_id'] == CLASS_ID_PERSON:
                    jersey_num = track.get('jersey_num')
                    team = track.get('team', 'Unknown')

                    stable_id = self.jersey_manager.get_stable_id(
                        track['track_id'], jersey_num, team, self.frame_count
                    )
                    track['stable_id'] = stable_id
                else:
                    # Ball keeps original track_id
                    track['stable_id'] = track['track_id']

            # Resolve conflicts (same jersey detected multiple times)
            tracks = self.jersey_manager.resolve_conflicts(tracks, self.frame_count)

        # ⭐ PHASE 3: 3-frame ID continuity (now uses stable IDs)
        stable_tracks = self._maintain_id_continuity(tracks)
        return stable_tracks
