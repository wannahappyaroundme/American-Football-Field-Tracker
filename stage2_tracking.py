"""
Stage 2: Add ByteTrack (Stable Track IDs)

Goal: Verify that ByteTrack can maintain consistent track IDs across frames
      without ID switching.

Validation Criteria:
- [ ] Track ID stability: Are track IDs consistent frame-to-frame for the same player?
- [ ] ID switches: How many ID switches occur? (Target: < 0.01 per frame)
- [ ] Track persistence: Do tracks survive brief occlusions?
- [ ] Total unique IDs: Is the number reasonable? (Expected: 22 players + refs + sideline ≈ 30-40 IDs)
"""

import cv2
from ultralytics import YOLO

# Configuration (match your project's config.py)
VIDEO_PATH = 'input/video.mp4'
OUTPUT_PATH = 'output/stage2_tracking.mp4'
MODEL_PATH = 'yolov8n.pt'

# Detection settings (from config.py)
PERSON_CONFIDENCE = 0.3  # DETECTION_CONFIDENCE_THRESHOLD
BALL_CONFIDENCE = 0.15   # BALL_CONFIDENCE
CLASS_ID_PERSON = 0
CLASS_ID_BALL = 32

# ByteTrack settings (from config.py)
TRACKER_CONFIG = 'bytetrack_extended.yaml'  # 900-frame buffer (30 seconds @ 30fps)
IOU_THRESHOLD = 0.2  # TRACKING_IOU_THRESHOLD (low for better occlusion handling)
LOG_INTERVAL = 90    # DETECTION_EVERY_N_FRAMES


def main():
    print("=" * 60)
    print("STAGE 2: ADD BYTETRACK (STABLE TRACK IDS)")
    print("=" * 60)

    # Load YOLO model
    print(f"\nLoading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {VIDEO_PATH}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"ERROR: Could not create output video: {OUTPUT_PATH}")
        return

    print(f"\nOutput video: {OUTPUT_PATH}")
    print(f"\nByteTrack Configuration:")
    print(f"  Config file: {TRACKER_CONFIG}")
    print(f"  Track buffer: 900 frames (30 seconds @ 30fps)")
    print(f"  IoU threshold: {IOU_THRESHOLD} (low = better occlusion handling)")
    print(f"  Person confidence: {PERSON_CONFIDENCE}")
    print(f"\nProcessing frames...")
    print("-" * 60)

    # Track statistics
    frame_count = 0
    seen_track_ids = set()
    track_id_switches = 0
    previous_tracks = {}  # {track_id: (cx, cy)}
    frame_with_tracks = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ⭐ Add tracking with ByteTrack (extended buffer)
        results = model.track(
            frame,
            persist=True,  # ⭐ Enable tracking
            conf=PERSON_CONFIDENCE,  # 0.3 (same as main system)
            classes=[CLASS_ID_PERSON, CLASS_ID_BALL],
            tracker=TRACKER_CONFIG,  # bytetrack_extended.yaml (900 frames)
            iou=IOU_THRESHOLD,  # 0.2 (low for better tracking through occlusion)
            verbose=False
        )

        # Check if tracking IDs exist
        boxes = results[0].boxes
        if boxes.id is None:
            # No tracks detected
            cv2.putText(frame, f"Frame: {frame_count} - No tracks", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
            continue

        frame_with_tracks += 1

        # Process tracks
        current_tracks = {}
        active_tracks = 0

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            if class_id != CLASS_ID_PERSON:
                continue

            track_id = int(boxes.id[i])
            confidence = float(boxes.conf[i])
            bbox = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)

            # Calculate bbox center
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            current_tracks[track_id] = (cx, cy)
            active_tracks += 1

            # Check for ID switching (same position, different ID)
            if track_id not in seen_track_ids:
                # New track ID - check if it's at same position as previous track
                for prev_id, (prev_cx, prev_cy) in previous_tracks.items():
                    if prev_id in current_tracks:
                        # Previous ID still exists, skip
                        continue

                    # Calculate distance to previous track position
                    dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5

                    if dist < 30:  # Same position (30 pixels)
                        track_id_switches += 1
                        if frame_count % 30 == 0:
                            print(f"⚠️  Frame {frame_count}: ID switch detected! "
                                  f"{prev_id} → {track_id} (distance: {dist:.1f}px)")

                seen_track_ids.add(track_id)

            # Draw tracking box
            color = (0, 255, 0) if track_id in previous_tracks else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {confidence:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        previous_tracks = current_tracks.copy()

        # Status overlay
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Active Tracks: {active_tracks}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Total IDs Seen: {len(seen_track_ids)}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"ID Switches: {track_id_switches}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Write frame
        out.write(frame)

        # Print progress
        if frame_count % LOG_INTERVAL == 0:
            progress = (frame_count / total_frames) * 100
            switches_per_frame = track_id_switches / frame_count if frame_count > 0 else 0
            print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%): "
                  f"{active_tracks} tracks, {len(seen_track_ids)} total IDs, "
                  f"{track_id_switches} switches ({switches_per_frame:.4f}/frame)")

    # Cleanup
    cap.release()
    out.release()

    # Calculate metrics
    id_switches_per_frame = track_id_switches / frame_count if frame_count > 0 else 0
    avg_tracks_per_frame = frame_with_tracks / frame_count if frame_count > 0 else 0

    # Print results
    print("-" * 60)
    print("\n" + "=" * 60)
    print("STAGE 2 RESULTS")
    print("=" * 60)
    print(f"\nProcessing Statistics:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames with tracks: {frame_with_tracks} ({avg_tracks_per_frame:.1%})")
    print(f"  Duration: {frame_count/fps:.1f} seconds")

    print(f"\nTracking Statistics:")
    print(f"  Total unique track IDs: {len(seen_track_ids)}")
    print(f"  ID switches detected: {track_id_switches}")
    print(f"  ID switches per frame: {id_switches_per_frame:.4f}")

    print(f"\nOutput saved to: {OUTPUT_PATH}")

    # Evaluation
    print("\n" + "=" * 60)
    print("VALIDATION CHECKLIST")
    print("=" * 60)

    # Check 1: Track ID stability
    print(f"\n[1] Track ID Stability")
    print(f"    Target: < 0.01 switches per frame")
    print(f"    Result: {id_switches_per_frame:.4f} switches per frame")
    print(f"    Target switches: < {frame_count * 0.01:.0f} total")
    print(f"    Actual switches: {track_id_switches} total")

    if id_switches_per_frame < 0.01:
        print(f"    ✓ PASS: {id_switches_per_frame:.4f} < 0.01")
    else:
        print(f"    ✗ FAIL: {id_switches_per_frame:.4f} >= 0.01")
        print(f"\n    Recommended fixes:")
        print(f"    → Lower match_thresh in {TRACKER_CONFIG} (0.8 → 0.7)")
        print(f"    → Increase track_buffer (900 → 1200 frames)")
        print(f"    → Check if detections are consistent (re-run Stage 1)")

    # Check 2: Total unique IDs
    print(f"\n[2] Total Unique IDs")
    print(f"    Expected: 30-50 IDs (22 players + refs + sideline)")
    print(f"    Result: {len(seen_track_ids)} IDs")

    if 20 <= len(seen_track_ids) <= 60:
        print(f"    ✓ PASS: Reasonable number of IDs")
    else:
        print(f"    ✗ WARNING: Unexpected number of IDs")
        if len(seen_track_ids) > 60:
            print(f"\n    Too many IDs detected! Recommended fixes:")
            print(f"    → Increase PERSON_CONFIDENCE (0.3 → 0.4)")
            print(f"    → Add field masking to exclude sideline (Stage 1 enhancement)")
            print(f"    → Add ROI filtering (exclude top/bottom regions)")
        else:
            print(f"\n    Too few IDs detected! Recommended fixes:")
            print(f"    → Lower PERSON_CONFIDENCE (0.3 → 0.25)")
            print(f"    → Check if video shows all players")

    # Check 3: Track persistence
    print(f"\n[3] Track Persistence")
    print(f"    Configuration: 900-frame buffer = 30 seconds @ 30fps")
    print(f"    Manual check required:")
    print(f"    → Review {OUTPUT_PATH}")
    print(f"    → Verify tracks survive brief occlusions")
    print(f"    → Look for green boxes (existing track) vs blue boxes (new track)")

    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)

    pass_count = 0
    total_checks = 2

    if id_switches_per_frame < 0.01:
        pass_count += 1
    if 20 <= len(seen_track_ids) <= 60:
        pass_count += 1

    print(f"Automated checks passed: {pass_count}/{total_checks}")

    if pass_count == total_checks:
        print(f"\n✓ All automated checks PASSED!")
        print(f"\nNext steps:")
        print(f"  1. Review {OUTPUT_PATH} for track persistence")
        print(f"  2. If satisfied, proceed to Stage 3 (Ball Carrier)")
    else:
        print(f"\n✗ Some checks FAILED!")
        print(f"\nNext steps:")
        print(f"  1. Apply recommended fixes above")
        print(f"  2. Adjust ByteTrack parameters in {TRACKER_CONFIG}")
        print(f"  3. Re-run Stage 2")
        print(f"\nByteTrack tuning guide:")
        print(f"  - match_thresh: Lower = more lenient matching (0.8 → 0.7 → 0.6)")
        print(f"  - track_buffer: Higher = longer persistence (900 → 1200 → 1500)")
        print(f"  - new_track_thresh: Higher = fewer new tracks (0.6 → 0.7 → 0.8)")

    print("=" * 60)


if __name__ == "__main__":
    main()
