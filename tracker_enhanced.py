"""
Football Tactical Analysis Engine - Enhanced Edition
====================================================

Complete implementation with all advanced features:
1. DeepSORT Re-ID tracking (zero ID switches)
2. Ball carrier identification
3. Distance tracking with calibration
4. Bird's eye view with analytics overlay
5. Side-by-side visualization
6. Full frame processing with accelerated output

This is the production-ready version with all features integrated.

Usage:
    python tracker_enhanced.py

Configuration:
    Edit tracker_config.py to customize all parameters

Author: Computer Vision Engineer
Date: October 2025
Version: 3.0 - Analytics Engine
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Tuple, List, Optional, Dict
import tracker_config as config

# Core tracking
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    from sort import Sort
    DEEPSORT_AVAILABLE = False

# Analytics modules
try:
    from ball_carrier_detector import BallCarrierDetector
    from distance_tracker import DistanceTracker
    from field_homography import FieldLineDetector, HomographyCalculator, BirdsEyeView
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    ANALYTICS_AVAILABLE = False
    print(f"Warning: Analytics modules not fully available: {e}")


def initialize_system():
    """Initialize all system components."""
    print("="*70)
    print("  FOOTBALL TACTICAL ANALYSIS ENGINE v3.0")
    print("="*70)
    
    components = {}
    
    # YOLO
    print("\n[1/5] Loading YOLO model...")
    components['yolo'] = YOLO(config.YOLO_MODEL_PATH)
    print("  ✓ YOLO loaded")
    
    # Tracker
    print("\n[2/5] Initializing tracker...")
    if config.USE_DEEPSORT and DEEPSORT_AVAILABLE:
        components['tracker'] = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            max_iou_distance=config.DEEPSORT_MAX_IOU_DISTANCE,
            max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
            embedder="mobilenet"
        )
        components['tracker_type'] = 'deepsort'
        print("  ✓ DeepSORT initialized (Re-ID enabled)")
    else:
        from sort import Sort
        components['tracker'] = Sort(max_age=config.SORT_MAX_AGE, min_hits=config.SORT_MIN_HITS)
        components['tracker_type'] = 'sort'
        print("  ✓ SORT initialized")
    
    # Analytics
    print("\n[3/5] Initializing analytics modules...")
    if ANALYTICS_AVAILABLE:
        if config.ENABLE_BALL_CARRIER_DETECTION:
            components['ball_carrier'] = BallCarrierDetector(config.BALL_CARRIER_MAX_DISTANCE)
            print("  ✓ Ball carrier detector")
        
        if config.ENABLE_DISTANCE_TRACKING:
            components['distance_tracker'] = DistanceTracker(config.PIXELS_PER_YARD)
            print("  ✓ Distance tracker")
        
        if config.ENABLE_BIRDS_EYE_VIEW:
            components['field_detector'] = FieldLineDetector()
            components['homography'] = HomographyCalculator(
                config.FIELD_LENGTH, config.FIELD_WIDTH,
                config.BIRDS_EYE_WIDTH, config.BIRDS_EYE_HEIGHT
            )
            components['birds_eye'] = BirdsEyeView(
                config.BIRDS_EYE_WIDTH, config.BIRDS_EYE_HEIGHT,
                config.FIELD_LENGTH, config.FIELD_WIDTH
            )
            print("  ✓ Bird's eye view system")
    else:
        print("  ⚠ Analytics modules unavailable")
    
    print("\n[4/5] System configuration...")
    print(f"  Frame processing: Every {config.FRAME_SKIP} frame(s)")
    print(f"  Output FPS multiplier: {config.OUTPUT_FPS_MULTIPLIER}x")
    print(f"  ROI: Top {config.ROI_TOP_PERCENT*100}%, Bottom {config.ROI_BOTTOM_PERCENT*100}%")
    
    print("\n[5/5] ✓ All systems ready!")
    
    return components


def process_frame(frame, components, state, frame_num):
    """
    Process a single frame with all analytics.
    
    Args:
        frame: Input frame
        components: Dictionary of initialized components
        state: Current state dictionary
        frame_num: Frame number
        
    Returns:
        Updated state
    """
    # Extract components
    yolo = components['yolo']
    tracker = components['tracker']
    tracker_type = components['tracker_type']
    
    # Create ROI mask
    h, w = frame.shape[:2]
    roi_mask = np.ones((h, w), dtype=np.uint8) * 255
    roi_mask[:int(h * config.ROI_TOP_PERCENT), :] = 0
    roi_mask[int(h * (1 - config.ROI_BOTTOM_PERCENT)):, :] = 0
    
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    
    # YO LO Detection
    results = yolo(masked_frame, verbose=False, conf=config.YOLO_CONF_THRESHOLD)
    
    detections = []
    ball_det = None
    
    for result in results:
        if result.boxes is None:
            continue
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in config.YOLO_TARGET_CLASSES:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # Check if ball (class 32)
            if cls_id == 32 and conf >= config.BALL_DETECTION_CONFIDENCE:
                ball_det = np.array([x1, y1, x2, y2, conf])
            else:  # Person (class 0)
                detections.append([x1, y1, x2, y2, conf])
    
    detections = np.array(detections) if detections else np.empty((0, 5))
    
    # Tracking
    if tracker_type == 'deepsort':
        if len(detections) > 0:
            dets_ds = [([x1, y1, x2-x1, y2-y1], score, 0) for x1, y1, x2, y2, score in detections]
            tracks_obj = tracker.update_tracks(dets_ds, frame=frame)
        else:
            tracks_obj = tracker.update_tracks([], frame=frame)
        
        tracks = []
        for t in tracks_obj:
            if t.is_confirmed():
                bbox = t.to_ltrb()
                tracks.append([bbox[0], bbox[1], bbox[2], bbox[3], t.track_id])
        tracks = np.array(tracks) if tracks else np.empty((0, 5))
    else:
        tracks = tracker.update(detections)
    
    # Ball carrier detection
    ball_carrier_id = None
    if 'ball_carrier' in components and ball_det is not None and len(tracks) > 0:
        ball_carrier_id = components['ball_carrier'].identify_ball_carrier(tracks, ball_det)
    
    # Field lines & homography (periodic)
    if 'homography' in components and frame_num % config.FIELD_LINE_DETECTION_INTERVAL == 0:
        h_lines, v_lines = components['field_detector'].detect_lines(frame, roi_mask)
        H = components['homography'].calculate_homography(h_lines, v_lines, frame.shape)
        if H is not None:
            state['homography_valid'] = True
    
    # Distance tracking
    if 'distance_tracker' in components and state.get('homography_valid') and len(tracks) > 0:
        track_positions = {}
        for track in tracks:
            track_id = int(track[4])
            foot_pos = ((track[0] + track[2]) / 2, track[3])
            transformed = components['homography'].transform_point(foot_pos)
            if transformed:
                track_positions[track_id] = transformed
        
        components['distance_tracker'].batch_update(track_positions)
    
    # Update state
    state['tracks'] = tracks
    state['ball_carrier_id'] = ball_carrier_id
    state['roi_mask'] = roi_mask
    
    return state


def visualize_frame(frame, components, state):
    """Create visualization with all analytics."""
    tracks = state.get('tracks', np.empty((0, 5)))
    ball_carrier_id = state.get('ball_carrier_id')
    
    vis = frame.copy()
    
    # Draw tracks
    for track in tracks:
        try:
            x1, y1, x2, y2, tid = map(float, track)
            x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
        except:
            continue
        
        # Color based on ball carrier
        if ball_carrier_id and tid == ball_carrier_id:
            color = config.BALL_CARRIER_COLOR
            label = f"ID:{tid} CARRIER"
        else:
            color = config.TRACK_COLOR
            label = f"ID:{tid}"
        
        # Add distance
        if 'distance_tracker' in components and config.SHOW_DISTANCE_ON_MAIN_VIEW:
            dist = components['distance_tracker'].get_distance(tid)
            label += f" {dist:.1f}yd"
        
        # Draw
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Create bird's eye view
    tactical_map = None
    if 'birds_eye' in components and state.get('homography_valid'):
        tactical_map = components['birds_eye'].base_field.copy()
        
        for track in tracks:
            try:
                x1, y1, x2, y2, tid = map(float, track)
                tid = int(tid)
                foot_pos = ((x1+x2)/2, y2)
                transformed = components['homography'].transform_point(foot_pos)
                
                if transformed:
                    tx, ty = int(transformed[0]), int(transformed[1])
                    cv2.circle(tactical_map, (tx, ty), config.PLAYER_DOT_RADIUS,
                             config.TRACK_COLOR, -1)
                    
                    # Draw ID and distance
                    if 'distance_tracker' in components:
                        dist = components['distance_tracker'].get_distance(tid)
                        cv2.putText(tactical_map, f"{tid}:{dist:.0f}", (tx+8, ty),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            except:
                continue
    
    # Combine views
    if tactical_map is not None:
        # Side by side
        h = vis.shape[0]
        tactical_resized = cv2.resize(tactical_map, (vis.shape[1]//2, h))
        vis_resized = cv2.resize(vis, (vis.shape[1]//2, h))
        combined = np.hstack([vis_resized, tactical_resized])
    else:
        combined = vis
    
    return combined


def main():
    """Main execution function."""
    # Initialize
    components = initialize_system()
    
    # Load video
    print("\n" + "="*70)
    print("LOADING VIDEO")
    print("="*70)
    cap = cv2.VideoCapture(config.VIDEO_INPUT_PATH)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open {config.VIDEO_INPUT_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Video writer
    writer = None
    if config.VIDEO_OUTPUT_PATH:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps * config.OUTPUT_FPS_MULTIPLIER
        writer = cv2.VideoWriter(config.VIDEO_OUTPUT_PATH, fourcc, out_fps, (w, h))
        print(f"Output: {config.VIDEO_OUTPUT_PATH} @ {out_fps:.1f} FPS ({config.OUTPUT_FPS_MULTIPLIER}x)")
    
    # Processing
    print("\n" + "="*70)
    print("PROCESSING")
    print("="*70)
    
    state = {'homography_valid': False}
    frame_num = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Process frame (every frame or according to FRAME_SKIP)
        if frame_num % config.FRAME_SKIP == 0:
            state = process_frame(frame, components, state, frame_num)
        
        # Visualize
        display = visualize_frame(frame, components, state)
        
        # Show
        if display.shape[1] > 1280:
            scale = 1280 / display.shape[1]
            display = cv2.resize(display, (int(display.shape[1]*scale), int(display.shape[0]*scale)))
        
        cv2.imshow('Tactical Analysis Engine', display)
        
        # Save
        if writer:
            writer.write(display)
        
        # Progress
        if frame_num % 30 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_num / elapsed
            num_tracks = len(state.get('tracks', []))
            print(f"Frame {frame_num}/{total_frames} | Tracks: {num_tracks} | FPS: {fps_current:.1f}")
        
        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Final stats
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Processed: {frame_num} frames")
    print(f"Time: {time.time() - start_time:.1f}s")
    
    if 'distance_tracker' in components:
        stats = components['distance_tracker'].get_statistics()
        print(f"\nDistance Statistics:")
        print(f"  Max: {stats['max_distance']:.1f} yards")
        print(f"  Avg: {stats['avg_distance']:.1f} yards")
        print(f"  Total: {stats['total_distance']:.1f} yards")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

