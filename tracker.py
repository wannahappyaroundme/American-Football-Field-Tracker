"""
Football Tactical Analysis Engine
==================================

Advanced real-time player tracking and tactical analysis system.

Features:
1. Dynamic frame processing for 1x playback speed
2. YOLOv8 detection + DeepSORT tracking for robust ID persistence
3. Integrated pose estimation for player posture analysis
4. Intelligent team classification based on jersey color and formation
5. Homography-based bird's eye view for tactical visualization

Upgrades from v1:
- SORT → DeepSORT (appearance-based tracking)
- Static FRAME_SKIP → Dynamic motion detection
- Added pose estimation (MediaPipe)
- Added team classification (K-means + formation analysis)
- Added bird's eye view (homography transformation)

Author: Computer Vision Engineer
Date: October 2025
Version: 2.0 - Tactical Analysis Engine
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Tuple, List, Optional, Dict
import tracker_config as config

# Advanced tracking imports
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    from sort import Sort
    DEEPSORT_AVAILABLE = False
    print("Warning: DeepSORT not available, falling back to SORT")

# New module imports (with error handling for optional features)
try:
    from frame_analyzer import AdaptiveFrameProcessor
    FRAME_ANALYZER_AVAILABLE = True
except ImportError:
    FRAME_ANALYZER_AVAILABLE = False
    print("Warning: frame_analyzer not available, dynamic processing disabled")

try:
    from pose_estimator import PoseEstimator
    POSE_ESTIMATOR_AVAILABLE = True
except ImportError:
    POSE_ESTIMATOR_AVAILABLE = False
    print("Warning: pose_estimator not available, pose estimation disabled")

try:
    from team_classifier import TeamClassifier
    TEAM_CLASSIFIER_AVAILABLE = True
except ImportError:
    TEAM_CLASSIFIER_AVAILABLE = False
    print("Warning: team_classifier not available, team classification disabled")

try:
    from field_homography import FieldLineDetector, HomographyCalculator, BirdsEyeView
    FIELD_HOMOGRAPHY_AVAILABLE = True
except ImportError:
    FIELD_HOMOGRAPHY_AVAILABLE = False
    print("Warning: field_homography not available, bird's eye view disabled")


# ============================================================================
# PART 1: MODEL INITIALIZATION AND SETUP
# ============================================================================

def load_yolo_model(model_path: str = 'yolov8n.pt'):
    """
    Load pre-trained YOLOv8 model.
    
    Args:
        model_path: Path to YOLO model weights
        
    Returns:
        YOLO model instance
    """
    print(f"Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully!")
    return model


def initialize_tracker(use_deepsort: bool = True):
    """
    Initialize tracker (DeepSORT or SORT fallback).
    
    Args:
        use_deepsort: Use DeepSORT if available
        
    Returns:
        Tracker instance
    """
    if use_deepsort and DEEPSORT_AVAILABLE:
        print(f"Initializing DeepSORT tracker...")
        print(f"  max_age={config.DEEPSORT_MAX_AGE}")
        print(f"  max_iou_distance={config.DEEPSORT_MAX_IOU_DISTANCE}")
        print(f"  max_cosine_distance={config.DEEPSORT_MAX_COSINE_DISTANCE}")
        
        tracker = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.SORT_MIN_HITS,
            max_iou_distance=config.DEEPSORT_MAX_IOU_DISTANCE,
            max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
            embedder="mobilenet",
            half=True,
            today=None
        )
        return tracker, 'deepsort'
    else:
        print(f"Initializing SORT tracker (fallback)...")
        print(f"  max_age={config.SORT_MAX_AGE}, min_hits={config.SORT_MIN_HITS}")
        from sort import Sort
        tracker = Sort(
            max_age=config.SORT_MAX_AGE,
            min_hits=config.SORT_MIN_HITS,
            iou_threshold=config.SORT_IOU_THRESHOLD
        )
        return tracker, 'sort'


def initialize_advanced_modules():
    """
    Initialize advanced analysis modules.
    
    Returns:
        Dictionary of initialized modules
    """
    modules = {}
    
    # Dynamic frame processor
    if config.ENABLE_DYNAMIC_PROCESSING and FRAME_ANALYZER_AVAILABLE:
        print("Initializing dynamic frame processor...")
        modules['frame_processor'] = AdaptiveFrameProcessor(
            enable_dynamic=True,
            method=config.FRAME_CHANGE_METHOD,
            threshold=config.FRAME_CHANGE_THRESHOLD,
            min_process_interval=1,
            max_skip_frames=10
        )
    else:
        if config.ENABLE_DYNAMIC_PROCESSING and not FRAME_ANALYZER_AVAILABLE:
            print("  Skipped (module not available)")
        modules['frame_processor'] = None
    
    # Pose estimator
    if config.ENABLE_POSE_ESTIMATION and POSE_ESTIMATOR_AVAILABLE:
        print("Initializing pose estimator...")
        modules['pose_estimator'] = PoseEstimator(
            model=config.POSE_MODEL,
            confidence=config.POSE_CONFIDENCE
        )
    else:
        if config.ENABLE_POSE_ESTIMATION and not POSE_ESTIMATOR_AVAILABLE:
            print("  Skipped (module not available)")
        modules['pose_estimator'] = None
    
    # Team classifier
    if config.ENABLE_TEAM_CLASSIFICATION and TEAM_CLASSIFIER_AVAILABLE:
        print("Initializing team classifier...")
        modules['team_classifier'] = TeamClassifier(
            num_teams=config.NUM_TEAMS,
            jersey_sample_top=config.JERSEY_SAMPLE_TOP,
            jersey_sample_bottom=config.JERSEY_SAMPLE_BOTTOM,
            init_frames=config.KMEANS_INIT_FRAMES,
            update_interval=config.KMEANS_UPDATE_INTERVAL,
            crouch_threshold=config.CROUCH_THRESHOLD,
            los_tolerance=config.LINE_OF_SCRIMMAGE_TOLERANCE
        )
    else:
        if config.ENABLE_TEAM_CLASSIFICATION and not TEAM_CLASSIFIER_AVAILABLE:
            print("  Skipped (module not available)")
        modules['team_classifier'] = None
    
    # Field line detector
    if config.ENABLE_BIRDS_EYE_VIEW and FIELD_HOMOGRAPHY_AVAILABLE:
        print("Initializing bird's eye view system...")
        modules['field_detector'] = FieldLineDetector()
        modules['homography_calc'] = HomographyCalculator(
            field_length=config.FIELD_LENGTH,
            field_width=config.FIELD_WIDTH,
            output_width=config.BIRDS_EYE_WIDTH,
            output_height=config.BIRDS_EYE_HEIGHT
        )
        modules['birds_eye_view'] = BirdsEyeView(
            width=config.BIRDS_EYE_WIDTH,
            height=config.BIRDS_EYE_HEIGHT,
            field_length=config.FIELD_LENGTH,
            field_width=config.FIELD_WIDTH
        )
    else:
        if config.ENABLE_BIRDS_EYE_VIEW and not FIELD_HOMOGRAPHY_AVAILABLE:
            print("  Skipped (module not available)")
        modules['field_detector'] = None
        modules['homography_calc'] = None
        modules['birds_eye_view'] = None
    
    return modules


# ============================================================================
# PART 2: ROI MASKING
# ============================================================================

def create_roi_mask(frame: np.ndarray, 
                    top_percent: float = 0.2, 
                    bottom_percent: float = 0.1) -> np.ndarray:
    """
    Create a Region of Interest (ROI) mask that excludes top and bottom portions.
    
    This removes scoreboard/UI (top) and lower crowd/ads (bottom), focusing
    on the main play area.
    
    Args:
        frame: Input frame
        top_percent: Percentage of frame height to exclude from top (0.0-1.0)
        bottom_percent: Percentage of frame height to exclude from bottom (0.0-1.0)
        
    Returns:
        Binary mask (white = ROI, black = excluded)
    """
    height, width = frame.shape[:2]
    
    # Create white mask (all included initially)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Calculate exclusion boundaries
    top_boundary = int(height * top_percent)
    bottom_boundary = int(height * (1 - bottom_percent))
    
    # Black out excluded regions
    mask[0:top_boundary, :] = 0  # Top region
    mask[bottom_boundary:, :] = 0  # Bottom region
    
    return mask


def apply_roi_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply ROI mask to frame, setting excluded areas to black.
    
    Args:
        frame: Input frame
        mask: Binary ROI mask
        
    Returns:
        Masked frame
    """
    # Convert mask to 3-channel if needed
    if len(mask.shape) == 2:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_3ch = mask
    
    # Apply mask
    masked_frame = cv2.bitwise_and(frame, mask_3ch)
    
    return masked_frame


def visualize_roi(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Visualize ROI by drawing semi-transparent overlay.
    
    Args:
        frame: Input frame
        mask: Binary ROI mask
        
    Returns:
        Frame with ROI visualization
    """
    vis = frame.copy()
    
    # Create semi-transparent red overlay for excluded regions
    overlay = vis.copy()
    overlay[mask == 0] = [0, 0, 200]  # Red for excluded
    
    # Blend with original
    cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
    
    return vis


# ============================================================================
# PART 3: YOLO DETECTION
# ============================================================================

def run_yolo_detection(model, frame: np.ndarray, 
                       conf_threshold: float = 0.5,
                       target_classes: List[int] = [0, 32]) -> np.ndarray:
    """
    Run YOLO detection on frame and filter results.
    
    Args:
        model: YOLO model instance
        frame: Input frame (can be masked)
        conf_threshold: Minimum confidence score
        target_classes: List of COCO class IDs to detect
                       0 = person, 32 = sports ball
        
    Returns:
        detections: Array of shape (N, 5) with format [x1, y1, x2, y2, score]
    """
    # Run inference
    results = model(frame, verbose=False, conf=conf_threshold)
    
    detections = []
    
    # Process results
    for result in results:
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            continue
        
        # Extract box data
        for box in boxes:
            # Get class ID
            cls_id = int(box.cls[0])
            
            # Filter by target classes
            if cls_id not in target_classes:
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get confidence score
            conf = float(box.conf[0])
            
            # Add to detections
            detections.append([x1, y1, x2, y2, conf])
    
    # Convert to numpy array
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))
    
    return detections


def separate_detections(detections: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate detections into players and balls (for visualization purposes).
    
    Note: SORT doesn't need this separation, but useful for different colored boxes.
    
    Args:
        detections: All detections
        model: YOLO model (to check classes)
        
    Returns:
        Tuple of (player_detections, ball_detections)
    """
    # For now, we'll assume all detections are players
    # In a more sophisticated version, you could track class info
    return detections, np.empty((0, 5))


# ============================================================================
# PART 4: SORT TRACKING
# ============================================================================

def update_tracker(tracker, detections: np.ndarray, tracker_type: str = 'sort', frame: np.ndarray = None) -> np.ndarray:
    """
    Update tracker with new detections (supports both SORT and DeepSORT).
    
    Args:
        tracker: Tracker instance (SORT or DeepSORT)
        detections: Array of shape (N, 5) with format [x1, y1, x2, y2, score]
        tracker_type: Type of tracker ('sort' or 'deepsort')
        frame: Original frame (required for DeepSORT)
        
    Returns:
        tracks: Array of shape (M, 5) with format [x1, y1, x2, y2, track_id]
    """
    if tracker_type == 'deepsort':
        # DeepSORT requires different format
        if len(detections) == 0:
            tracks = tracker.update_tracks([], frame=frame)
        else:
            # Convert detections to DeepSORT format: [[bbox, confidence, class], ...]
            dets_deepsort = []
            for det in detections:
                x1, y1, x2, y2, score = det
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]
                dets_deepsort.append((bbox, score, 0))  # class=0 (person)
            
            tracks = tracker.update_tracks(dets_deepsort, frame=frame)
        
        # Convert DeepSORT tracks to standard format
        tracks_array = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            bbox = track.to_ltrb()  # Get [left, top, right, bottom]
            track_id = track.track_id
            tracks_array.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
        
        return np.array(tracks_array) if tracks_array else np.empty((0, 5))
    
    else:  # SORT
        return tracker.update(detections)


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

def draw_detections(frame: np.ndarray, 
                   detections: np.ndarray,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   label: str = "Detection") -> np.ndarray:
    """
    Draw detection bounding boxes on frame.
    
    Args:
        frame: Input frame
        detections: Array of detections [x1, y1, x2, y2, score]
        color: BGR color tuple
        label: Label prefix
        
    Returns:
        Frame with drawn boxes
    """
    vis = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2, score = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence
        text = f"{label}: {score:.2f}"
        cv2.putText(vis, text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis


def draw_tracks(frame: np.ndarray, 
                tracks: np.ndarray,
                color: Tuple[int, int, int] = (255, 0, 255),
                thickness: int = 2) -> np.ndarray:
    """
    Draw tracked objects with IDs on frame.
    
    This is the main visualization function - showing persistent track IDs.
    
    Args:
        frame: Input frame
        tracks: Array of tracks [x1, y1, x2, y2, track_id]
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Frame with drawn tracks
    """
    vis = frame.copy()
    
    if len(tracks) == 0:
        return vis
    
    for track in tracks:
        try:
            x1, y1, x2, y2, track_id = track
            # Safely convert to int (handle both float and string)
            x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
            track_id = int(float(track_id))
        except (ValueError, TypeError) as e:
            # Skip invalid tracks
            continue
        
        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        
        # Draw track ID (most important!)
        # Make ID very visible with background
        text = f"ID: {track_id}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Draw background rectangle for text
        cv2.rectangle(vis, 
                     (x1, y1 - text_size[1] - 10),
                     (x1 + text_size[0] + 10, y1),
                     color, -1)
        
        # Draw text
        cv2.putText(vis, text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(vis, (center_x, center_y), 4, color, -1)
    
    return vis


def create_display_simple(tracked_frame: np.ndarray,
                         num_detections: int,
                         num_tracks: int,
                         fps: float,
                         frame_count: int) -> np.ndarray:
    """
    Create simple, fast display with just tracked frame and stats.
    
    Args:
        tracked_frame: Frame with tracks drawn
        num_detections: Number of detections
        num_tracks: Number of active tracks  
        fps: Current FPS
        frame_count: Current frame number
        
    Returns:
        Display frame with stats overlay
    """
    display = tracked_frame.copy()
    
    # Add stats overlay
    stats_y = 30
    stats = [
        f"Frame: {frame_count}",
        f"FPS: {fps:.1f}",
        f"Detections: {num_detections}",
        f"Active Tracks: {num_tracks}",
        f"Press 'q' to quit, 'p' to pause"
    ]
    
    for stat in stats:
        # Background
        (w, h), _ = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display, (5, stats_y - 20), (w + 15, stats_y + 5), (0, 0, 0), -1)
        # Text
        cv2.putText(display, stat, (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        stats_y += 30
    
    return display


def create_display(frame: np.ndarray, 
                  masked_frame: np.ndarray,
                  tracked_frame: np.ndarray,
                  roi_mask: np.ndarray,
                  num_detections: int,
                  num_tracks: int,
                  fps: float) -> np.ndarray:
    """
    Create comprehensive display showing all processing stages.
    
    Args:
        frame: Original frame
        masked_frame: Frame with ROI mask applied
        tracked_frame: Frame with tracks drawn
        roi_mask: Binary ROI mask
        num_detections: Number of detections in current frame
        num_tracks: Number of active tracks
        fps: Current processing FPS
        
    Returns:
        Combined display frame
    """
    # Visualize ROI on original frame
    roi_vis = visualize_roi(frame, roi_mask)
    
    # Add info overlay to tracked frame
    info_frame = tracked_frame.copy()
    
    # Draw semi-transparent info box
    info_text = [
        f"FPS: {fps:.1f}",
        f"Detections: {num_detections}",
        f"Active Tracks: {num_tracks}"
    ]
    
    y_offset = 30
    for text in info_text:
        cv2.rectangle(info_frame, (10, y_offset - 25), (250, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(info_frame, text, (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 35
    
    # Create 2x2 grid
    # Top: ROI visualization | Masked frame
    # Bottom: Tracked result | Info
    
    top_row = np.hstack([roi_vis, masked_frame])
    bottom_row = np.hstack([tracked_frame, info_frame])
    combined = np.vstack([top_row, bottom_row])
    
    # Add labels
    label_color = (255, 255, 255)
    label_bg = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Label each quadrant
    h, w = frame.shape[:2]
    labels = [
        ("1. ROI Mask Visualization", (10, 30)),
        ("2. Masked Frame (YOLO Input)", (w + 10, 30)),
        ("3. Tracked Objects", (10, h + 30)),
        ("4. Tracking Info", (w + 10, h + 30))
    ]
    
    for text, (x, y) in labels:
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        cv2.rectangle(combined, (x - 5, y - 25), (x + text_size[0] + 5, y + 5), label_bg, -1)
        cv2.putText(combined, text, (x, y), font, 0.7, label_color, 2)
    
    return combined


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def process_video(video_path: str, output_path: Optional[str] = None):
    """
    Main video processing loop with YOLO detection and SORT tracking.
    
    Args:
        video_path: Path to input video
        output_path: Optional path to save output video
    """
    print("="*60)
    print("Football Player and Ball Tracker")
    print("="*60)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Properties:")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {frame_count_total}")
    
    # Initialize models
    print("\n" + "="*60)
    print("Initializing Models...")
    print("="*60)
    
    model = load_yolo_model(config.YOLO_MODEL_PATH)
    tracker, tracker_type = initialize_tracker(use_deepsort=config.USE_DEEPSORT)
    
    # Initialize advanced modules
    print("\n" + "="*60)
    print("Initializing Advanced Modules...")
    print("="*60)
    modules = initialize_advanced_modules()
    
    # Initialize video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Output is 2x2 grid
        writer = cv2.VideoWriter(output_path, fourcc, fps,
                                (frame_width * 2, frame_height * 2))
        print(f"\nOutput will be saved to: {output_path}")
    
    # Processing loop
    print("\n" + "="*60)
    print("Processing Video...")
    print("="*60)
    
    frame_count = 0
    total_time = 0
    
    last_known_tracks = np.empty((0, 5))  # 마지막 추적 결과를 저장할 변수
    detections_for_display = 0            # 화면에 표시할 탐지 개수 저장 변수
    
    while True:
        start_time = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video reached.")
            break
        
        frame_count += 1
        
        # ====================================================================
        # PART 2: ROI MASKING
        # ====================================================================
        
        # Create ROI mask (exclude top and bottom)
        roi_mask = create_roi_mask(frame, 
                                   top_percent=config.ROI_TOP_PERCENT,
                                   bottom_percent=config.ROI_BOTTOM_PERCENT)
        
        # Apply mask to frame
        masked_frame = apply_roi_mask(frame, roi_mask)
        
        # ====================================================================
        # PART 3&4: YOLO DETECTION & SORT TRACKING
        # ====================================================================
        
        # 프레임 스킵 로직: FRAME_SKIP 간격으로만 분석 수행
        # tracker.py의 while 루프 안
        if frame_count % config.FRAME_SKIP == 0:
        # --- 분석을 수행하는 프레임 ---
            detections = run_yolo_detection(
                model,
                masked_frame,
                conf_threshold=config.YOLO_CONF_THRESHOLD,
                target_classes=config.YOLO_TARGET_CLASSES
            )
            tracks = update_tracker(tracker, detections, tracker_type=tracker_type, frame=frame)

            # 결과를 다음 프레임들을 위해 저장
            last_known_tracks = tracks
            detections_for_display = len(detections) # 실제 탐지된 개수를 저장
        else:
            # --- 분석을 건너뛰는 프레임 ---
            tracks = last_known_tracks
            detections_for_display = 0 # 탐지를 안했으므로 0으로 설정
            
        # ====================================================================
        # PART 5: VISUALIZATION
        # ====================================================================
        
        # Draw tracks on original frame
        tracked_frame = draw_tracks(frame, tracks, 
                                    color=config.TRACK_COLOR,
                                    thickness=config.TRACK_THICKNESS)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        total_time += elapsed
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        # Create simple, fast display (single panel instead of 2x2 grid)
        display = create_display_simple(tracked_frame, detections_for_display, 
                                       len(tracks), avg_fps, frame_count)
        
        # Resize for speed (smaller window = faster rendering)
        if display.shape[1] > 1280:
            scale = 1280 / display.shape[1]
            w = int(display.shape[1] * scale)
            h = int(display.shape[0] * scale)
            display = cv2.resize(display, (w, h))
        
        cv2.imshow('Football Tracker - Real-Time', display)
        
        # Write to output if specified
        if writer:
            writer.write(display)
        
        # Print progress (less frequently)
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}/{frame_count_total} | "
                  f"Detections: {detections_for_display} | "
                  f"Tracks: {len(tracks)} | "
                  f"FPS: {avg_fps:.1f}")
        
        # ====================================================================
        # USER CONTROLS
        # ====================================================================
        
        # Non-blocking key check for real-time playback
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nUser requested quit.")
            break
        elif key == ord('p'):
            print("\nPaused. Press any key to continue...")
            cv2.waitKey(0)
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n" + "="*60)
    print(f"Processing Complete!")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Average FPS: {frame_count / total_time:.1f}")
    print("="*60)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        process_video(config.VIDEO_INPUT_PATH, config.VIDEO_OUTPUT_PATH)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

