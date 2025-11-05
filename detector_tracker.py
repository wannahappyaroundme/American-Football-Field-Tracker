from ultralytics import YOLO
from config import (
    DETECTION_MODEL_PATH,
    CLASS_ID_PERSON,
    CLASS_ID_BALL,
    DETECTION_CONFIDENCE_THRESHOLD,
    BALL_CONFIDENCE,
    ENABLE_CLIP_CLASSIFICATION,
    ENABLE_CLIP_ENTITY_FILTERING,
    CLIP_FRAME_INTERVAL,
    CLIP_EXCLUDE_ENTITIES,
    CLIP_SHOW_REFEREES,
    DETECTION_EVERY_N_FRAMES
)


class DetectorTracker:
    """
    A class to detect and track objects (players and ball) in video frames using YOLO.
    """

    def __init__(self, clip_classifier=None):
        """
        Initialize the DetectorTracker by loading the YOLO detection model.

        Args:
            clip_classifier: Optional CLIPEntityClassifier instance for semantic filtering
        """
        self.model = YOLO(DETECTION_MODEL_PATH)
        self.clip_classifier = clip_classifier
        self.frame_count = 0
        self.entity_cache = {}  # Cache CLIP results: track_id -> (entity_type, confidence)
        print(f"Loaded YOLO model from {DETECTION_MODEL_PATH}")
        print(f"Enhanced tracking: ByteTrack with extended buffer")
        print(f"Track buffer: 900 frames (30 seconds persistence)")
        print(f"Detection delay mode: conf=0.25, new_track_thresh=0.5")
        print(f"Status logging every {DETECTION_EVERY_N_FRAMES} frames")

        if self.clip_classifier:
            print(f"CLIP entity filtering enabled (interval: {CLIP_FRAME_INTERVAL} frames)")

    def track_frame(self, frame):
        """
        Track objects in a single frame with optional CLIP entity classification.

        Args:
            frame: Input frame (numpy array) to process

        Returns:
            List of dictionaries, where each dictionary contains:
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                - 'track_id': integer tracking ID
                - 'class_id': integer class ID (person or ball)
                - 'entity_type': (optional) 'player', 'referee', 'sideline', 'other'
                - 'entity_confidence': (optional) float confidence score

            Returns an empty list if no tracks are found or results[0].boxes.id is None.
        """
        self.frame_count += 1

        # Run YOLO tracking with ByteTrack
        # track_buffer=900으로 30초 동안 track 유지
        # conf를 높여서 detection 딜레이, 대신 tracking으로 안정성 유지
        results = self.model.track(
            frame,
            persist=True,
            conf=0.25,  # detection threshold 높임 (딜레이 효과)
            classes=[CLASS_ID_PERSON, CLASS_ID_BALL],
            tracker='bytetrack_extended.yaml',  # ByteTrack with extended buffer
            iou=0.2  # IoU 임계값 낮춤 (tracking 안정성)
        )

        # Log status periodically
        if self.frame_count % DETECTION_EVERY_N_FRAMES == 1:
            num_tracks = len(results[0].boxes) if results[0].boxes.id is not None else 0
            print(f"🔍 Frame {self.frame_count}: {num_tracks} tracks active")

        # Handle case where no detections or tracking IDs are available
        if results[0].boxes.id is None:
            return []

        # Extract tracking information
        tracks = []
        boxes = results[0].boxes

        # Collect person crops for batch CLIP classification
        person_crops = []
        person_track_indices = []

        # Process each detected object with CLASS-SPECIFIC confidence filtering
        for i in range(len(boxes)):
            # Get bounding box coordinates
            bbox = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]

            # Get tracking ID
            track_id = int(boxes.id[i].cpu().numpy())

            # Get class ID
            class_id = int(boxes.cls[i].cpu().numpy())

            # Get confidence score
            confidence = float(boxes.conf[i].cpu().numpy())

            # ===== 클래스별 신뢰도 필터링 (핵심!) =====
            # 사람: 낮은 임계값 (0.3) - 왼쪽 객체 포함, 초반 detection 개선
            if class_id == CLASS_ID_PERSON:
                if confidence < 0.3:  # 0.5 → 0.3 (더 관대하게)
                    continue

            # 공: 매우 낮은 임계값 (0.15) - 공 디텍션 대폭 개선!
            elif class_id == CLASS_ID_BALL:
                if confidence < 0.15:  # 0.2 → 0.15 (더 낮춤)
                    continue

                # 공 크기 필터링 (false positive 제거)
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                # 너무 크거나 작은 공 감지 제외 (미식축구공은 다양한 크기로 보일 수 있음)
                if width < 5 or height < 5 or width > 300 or height > 300:
                    continue

                # 공 감지 로그 (디버깅)
                print(f"⚽ Ball detected! Track #{track_id}, conf={confidence:.2%}, size={int(width)}x{int(height)}")

            # Create track dict with confidence
            track = {
                'bbox': bbox,
                'track_id': track_id,
                'class_id': class_id,
                'confidence': confidence
            }

            # For person detections, prepare for CLIP classification
            if class_id == CLASS_ID_PERSON:
                # Prepare for CLIP classification
                if self.clip_classifier and ENABLE_CLIP_ENTITY_FILTERING:
                    person_crops.append((i, track_id, bbox))
                    person_track_indices.append(len(tracks))

            tracks.append(track)

        # Run CLIP classification on person crops (batch processing)
        if person_crops and self.frame_count % CLIP_FRAME_INTERVAL == 0:
            crops = []
            track_ids = []

            for idx, track_id, bbox in person_crops:
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    crops.append(crop)
                    track_ids.append(track_id)

            # Batch classify entities
            if len(crops) > 0:
                results = self.clip_classifier.batch_classify_entities(crops, track_ids)

                # Update tracks with CLIP results and cache
                for crop_idx, (entity_type, confidence) in enumerate(results):
                    track_id = track_ids[crop_idx]
                    track_idx = person_track_indices[crop_idx]

                    # Store in cache
                    self.entity_cache[track_id] = (entity_type, confidence)

                    # Add to track
                    tracks[track_idx]['entity_type'] = entity_type
                    tracks[track_idx]['entity_confidence'] = confidence

        # Apply cached CLIP results for non-classification frames
        elif person_crops:
            for idx, track_id, bbox in person_crops:
                if track_id in self.entity_cache:
                    entity_type, confidence = self.entity_cache[track_id]
                    track_idx = person_track_indices[person_crops.index((idx, track_id, bbox))]
                    tracks[track_idx]['entity_type'] = entity_type
                    tracks[track_idx]['entity_confidence'] = confidence

        # Filter out excluded entities
        if ENABLE_CLIP_ENTITY_FILTERING:
            filtered_tracks = []
            for track in tracks:
                # Always keep ball
                if track['class_id'] == CLASS_ID_BALL:
                    filtered_tracks.append(track)
                    continue

                # Check entity type
                entity_type = track.get('entity_type', 'player')

                # Filter logic
                if entity_type in CLIP_EXCLUDE_ENTITIES:
                    continue  # Exclude sideline personnel, photographers, etc.

                if entity_type == 'referee' and not CLIP_SHOW_REFEREES:
                    continue  # Optionally exclude referees

                filtered_tracks.append(track)

            return filtered_tracks

        return tracks
