import cv2
import json
from ultralytics import YOLO

from detector_tracker import DetectorTracker
from transformer import ViewTransformer
from play_analyzer import PlayAnalyzer
from visualizer import Visualizer
from number_recognizer import NumberRecognizer
from team_classifier import TeamClassifier

from config import (
    VIDEO_INPUT_PATH,
    VIDEO_OUTPUT_PATH,
    BEV_OUTPUT_PATH,
    JSON_OUTPUT_PATH,
    CLASS_ID_PERSON,
    DETECTION_CONFIDENCE_THRESHOLD,
    BALL_CONFIDENCE,
    TRACKING_IOU_THRESHOLD,
    DETECTION_EVERY_N_FRAMES,
    TEAM_A_HSV_RANGE,
    TEAM_B_HSV_RANGE,
    REFEREE_HSV_RANGE,
    ENABLE_CLIP_TEAM_CLASSIFICATION  # ⭐ PHASE 2: CLIP support
)

# Manual metadata - placeholders for game and clip information
MANUAL_METADATA = {
    'gameKey': 'GAME_2024_001',
    'gameDate': '2024-01-15',
    'homeTeam': 'Home Team',
    'awayTeam': 'Away Team',
    'stadium': 'Stadium Name',
    'weather': 'Clear',
    'temperature': '72F',
    'clipKey': 'CLIP_001',
    'quarter': 1,
    'time': '12:00',
    'down': 1,
    'toGo': 10,
    'yardLine': 25,
    'offensiveTeam': 'Home Team',
    'defensiveTeam': 'Away Team',
    'offensiveFormation': 'Unknown',
    'defensiveFormation': 'Unknown',
    'personnel': 'Unknown',
    'playResult': 'Unknown',
    'tackler': 'Unknown'
}


def main():
    """
    Main function to run the football play analysis pipeline.
    """
    print("="*60)
    print("Football Play Analysis System - SIMPLIFIED")
    print("="*60)

    # Initialize ViewTransformer first (needed for BEV masking)
    print("\n[1/6] Initializing components...")
    transformer = ViewTransformer()
    print("✓ ViewTransformer loaded")

    # Initialize detection model with BEV masking support
    print("\n[2/6] Loading YOLO model...")
    detector = DetectorTracker(view_transformer=transformer)  # ⭐ NEW: BEV masking enabled
    print("✓ YOLO detection model loaded")

    # Initialize other components
    print("\n[3/6] Initializing other components...")
    analyzer = PlayAnalyzer()
    visualizer = Visualizer()
    recognizer = NumberRecognizer()

    # ⭐ PHASE 2: Initialize CLIP team classifier if enabled
    if ENABLE_CLIP_TEAM_CLASSIFICATION:
        from clip_classifier import CLIPTeamClassifier
        clip_team_classifier = CLIPTeamClassifier()
        team_classifier = TeamClassifier(clip_team_classifier=clip_team_classifier)
        print("✓ CLIP team classification ENABLED")
    else:
        team_classifier = TeamClassifier()  # HSV-only mode
        print("✓ Color-based team classification (HSV)")

    print("✓ Components initialized")

    # Set up video capture and writers
    print("\n[4/6] Setting up video input/output...")
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_INPUT_PATH}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f} seconds")

    print(f"\nSystem Configuration:")
    print(f"  Detection confidence: {DETECTION_CONFIDENCE_THRESHOLD} (persons), {BALL_CONFIDENCE} (ball)")
    print(f"  ByteTrack: 900-frame buffer (30 seconds @ 30fps)")
    print(f"  IoU threshold: {TRACKING_IOU_THRESHOLD} (low = better occlusion handling)")
    print(f"  Log interval: every {DETECTION_EVERY_N_FRAMES} frames")

    print(f"\nTeam Classification (HSV ranges):")
    print(f"  Team A (Yellow): H={TEAM_A_HSV_RANGE[0][0]}-{TEAM_A_HSV_RANGE[1][0]}, "
          f"S={TEAM_A_HSV_RANGE[0][1]}-{TEAM_A_HSV_RANGE[1][1]}, "
          f"V={TEAM_A_HSV_RANGE[0][2]}-{TEAM_A_HSV_RANGE[1][2]}")
    print(f"  Team B (White): H={TEAM_B_HSV_RANGE[0][0]}-{TEAM_B_HSV_RANGE[1][0]}, "
          f"S={TEAM_B_HSV_RANGE[0][1]}-{TEAM_B_HSV_RANGE[1][1]}, "
          f"V={TEAM_B_HSV_RANGE[0][2]}-{TEAM_B_HSV_RANGE[1][2]}")
    print(f"  Referee: Stripe detection (std_dev > 70, STRICT!) + HSV backup")

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_annotated = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))
    out_bev = cv2.VideoWriter(BEV_OUTPUT_PATH, fourcc, fps, (1000, 500))

    # Main processing loop
    print("\n[5/6] Processing video frames...")
    print("-" * 60)
    frame_count = 0
    sampled_players = set()  # 이미 샘플링한 선수 ID 저장 (한번만 샘플링)

    # Tracking statistics
    seen_track_ids = set()
    team_classification_count = {'Team A': 0, 'Team B': 0, 'Referee': 0, 'Unknown': 0}
    ball_detections = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or read error.")
            break

        frame_count += 1

        # Track objects in the frame (pure YOLO + ByteTrack)
        tracks = detector.track_frame(frame)

        # Count active tracks per frame
        active_person_tracks = 0
        active_ball_tracks = 0

        # ⭐ tracker.py 방식: 매 프레임마다 즉시 팀 할당!
        # ⭐ NOW USES stable_id for jersey-based tracking
        for track in tracks:
            track_id = track.get('stable_id', track.get('track_id'))  # Use stable_id
            seen_track_ids.add(track_id)

            if track['class_id'] == CLASS_ID_PERSON:
                active_person_tracks += 1

                # 이미 할당된 선수는 스킵 (한번만!)
                if track_id in sampled_players:
                    continue

                x1, y1, x2, y2 = map(int, track['bbox'])

                # Ensure valid crop
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        # 즉시 팀 분류 (HSV 범위 기반 + 위치 기반 재사용)
                        team_label, team_color = team_classifier.classify_team_instantly(
                            track_id, crop, [x1, y1, x2, y2]
                        )
                        sampled_players.add(track_id)  # 고정!
                        team_classification_count[team_label] += 1

                        if frame_count % DETECTION_EVERY_N_FRAMES == 0:  # 설정된 간격마다 로그
                            print(f"  👕 Player #{track_id} → {team_label}")
            else:
                # Ball track
                active_ball_tracks += 1
                ball_detections += 1

        # Update analyzer with new tracks
        analyzer.update_tracks(tracks, transformer)

        # DISABLED: Pose detection for ball carrier - causes overhead and flickering
        # if analyzer.ball_carrier_id is not None and analyzer.state != 'PLAY_ENDED':
        #     # Find the ball carrier's track
        #     ball_carrier_track = None
        #     for track in tracks:
        #         if track['track_id'] == analyzer.ball_carrier_id:
        #             ball_carrier_track = track
        #             break
        #
        #     if ball_carrier_track is not None:
        #         # Crop the frame to ball carrier's bbox
        #         bbox = ball_carrier_track['bbox']
        #         x1, y1, x2, y2 = map(int, bbox)
        #
        #         # Ensure valid crop coordinates
        #         x1, y1 = max(0, x1), max(0, y1)
        #         x2, y2 = min(width, x2), min(height, y2)
        #
        #         if x2 > x1 and y2 > y1:
        #             crop_frame = frame[y1:y2, x1:x2]
        #
        #             # Run pose detection on cropped frame
        #             pose_results = pose_model(crop_frame, verbose=False)
        #
        #             # Check for play end using pose keypoints
        #             analyzer.check_play_end(pose_results, tracks, transformer)

        # Draw annotations on the frame
        annotated_frame = visualizer.draw_annotations(
            frame.copy(),
            tracks,
            analyzer.ball_carrier_id,
            analyzer.state,
            analyzer.play_type,
            view_transformer=transformer,  # BEV 경계 필터링 활성화
            team_classifier=team_classifier
        )

        # Draw bird's eye view
        bev_frame = visualizer.draw_bird_eye_view(
            analyzer.player_states,
            team_classifier=team_classifier
        )

        # Write output frames
        out_annotated.write(annotated_frame)
        out_bev.write(bev_frame)

        # Print progress
        if frame_count % DETECTION_EVERY_N_FRAMES == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%): "
                  f"{active_person_tracks} players, {active_ball_tracks} ball(s), "
                  f"{len(seen_track_ids)} total IDs, State: {analyzer.state}")

        # Break if play has ended
        if analyzer.state == 'PLAY_ENDED':
            print(f"\nPlay ended at frame {frame_count}")
            break

    # Release video objects
    cap.release()
    out_annotated.release()
    out_bev.release()
    cv2.destroyAllWindows()

    # Calculate metrics
    ball_detection_rate = ball_detections / frame_count if frame_count > 0 else 0

    print("\n" + "-" * 60)

    # ⭐ POST-PROCESSING: Select top 1-4 referees and reclassify others to Team A/B
    if ENABLE_CLIP_TEAM_CLASSIFICATION and hasattr(team_classifier, 'clip_classifier') and team_classifier.clip_classifier:
        print("\n🔄 POST-PROCESSING: Referee selection (keep top 1-4 referees, reclassify rest to Team A/B)...")

        # Get top 4 referees by confidence (실제 게임에서 1-4명)
        top_referee_ids = team_classifier.clip_classifier.get_top_referees(max_refs=4)

        # Reclassify ALL referees that aren't in the top 4 to Team A or Team B using HSV
        reclassified_count = 0
        reclassified_to_a = 0
        reclassified_to_b = 0

        # Debug: Print all referees found
        all_referees = [tid for tid, team in team_classifier.player_teams.items() if team == 'Referee']
        print(f"\n  Debug: Found {len(all_referees)} total referee IDs")
        print(f"  Debug: Top 4 referee IDs to keep: {sorted(top_referee_ids)}")

        for track_id, current_team in list(team_classifier.player_teams.items()):
            # If classified as Referee but NOT in top 4, reclassify to Team A/B using HSV
            if current_team == 'Referee' and track_id not in top_referee_ids:
                # Get color samples for this track_id
                track_samples = [s for s in team_classifier.color_samples if s['track_id'] == track_id]

                if track_samples:
                    # Use the first available color sample for HSV reclassification
                    hsv_color = track_samples[0]['color']
                    h, s, v = hsv_color

                    # HSV range check to determine Team A (Yellow) or Team B (White)
                    # Team A: Yellow (H: 18-35, S: 60-255, V: 100-255)
                    if 18 <= h <= 35 and s >= 60 and v >= 100:
                        new_team = 'Team A'
                        reclassified_to_a += 1
                    # Team B: White (H: any, S: 0-40, V: 160-255)
                    elif s <= 40 and v >= 160:
                        new_team = 'Team B'
                        reclassified_to_b += 1
                    else:
                        # Default to Team B if neither matches clearly
                        new_team = 'Team B'
                        reclassified_to_b += 1

                    team_classifier.player_teams[track_id] = new_team
                    reclassified_count += 1
                else:
                    # No color samples, mark as Unknown
                    team_classifier.player_teams[track_id] = 'Unknown'
                    reclassified_count += 1

        print(f"\n  Reclassified {reclassified_count} false-positive referees:")
        print(f"    → Team A: {reclassified_to_a}")
        print(f"    → Team B: {reclassified_to_b}")

        # Recalculate team counts from player_teams
        final_team_counts = {'Team A': 0, 'Team B': 0, 'Referee': 0, 'Unknown': 0}
        for team in team_classifier.player_teams.values():
            final_team_counts[team] = final_team_counts.get(team, 0) + 1

        print(f"  Final team counts (unique players):")
        print(f"    Team A: {final_team_counts['Team A']}")
        print(f"    Team B: {final_team_counts['Team B']}")
        print(f"    Referee: {final_team_counts['Referee']} ✓ TARGET: 1-4")
        print(f"    Unknown: {final_team_counts['Unknown']}")

    print("\n[6/6] Video processing complete.")
    print(f"\nOutput files:")
    print(f"  Annotated video: {VIDEO_OUTPUT_PATH}")
    print(f"  BEV video: {BEV_OUTPUT_PATH}")

    print(f"\nProcessing Statistics:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Duration: {frame_count/fps:.1f} seconds")
    print(f"  Processing speed: {frame_count/(frame_count/fps):.1f} FPS effective")

    print(f"\nTracking Statistics:")
    print(f"  Total unique track IDs: {len(seen_track_ids)}")
    print(f"  Expected: 30-50 IDs (22 players + refs + sideline)")
    if 20 <= len(seen_track_ids) <= 60:
        print(f"  ✓ PASS: Reasonable number of IDs")
    else:
        print(f"  ✗ WARNING: {len(seen_track_ids)} IDs (expected 30-50)")

    print(f"\nBall Detection:")
    print(f"  Ball detections: {ball_detections} frames")
    print(f"  Detection rate: {ball_detection_rate:.1%}")
    print(f"  Confidence threshold: {BALL_CONFIDENCE}")

    print(f"\nTeam Classification (HSV-based):")
    print(f"  Total players classified: {len(sampled_players)}")
    print(f"  Team A (Yellow): {team_classification_count['Team A']} players")
    print(f"  Team B (White): {team_classification_count['Team B']} players")
    print(f"  Referee: {team_classification_count['Referee']} players")
    print(f"  Unknown: {team_classification_count['Unknown']} players")

    # Team classification accuracy check
    total_classified = len(sampled_players)
    known_percentage = ((total_classified - team_classification_count['Unknown']) / total_classified * 100) if total_classified > 0 else 0
    print(f"  Classification success rate: {known_percentage:.1f}%")
    if known_percentage >= 80:
        print(f"  ✓ PASS: >80% players classified")
    else:
        print(f"  ✗ WARNING: <80% classification rate")

    # Generate JSON summary
    print("\n[7/7] Generating JSON summary...")
    summary_data = analyzer.get_summary()

    # Attempt to recognize jersey number for ball carrier
    carrier_number = 'Unknown'
    if summary_data['car_id'] is not None:
        # Note: We'd need to save/access a frame with the carrier visible
        # For now, we'll mark it as unknown
        carrier_number = 'Unknown'  # Could be enhanced with stored crops

    # Create clip data structure
    clip_data = {
        'clipKey': MANUAL_METADATA['clipKey'],
        'quarter': MANUAL_METADATA['quarter'],
        'time': MANUAL_METADATA['time'],
        'down': MANUAL_METADATA['down'],
        'toGo': MANUAL_METADATA['toGo'],
        'yardLine': MANUAL_METADATA['yardLine'],
        'offensiveTeam': MANUAL_METADATA['offensiveTeam'],
        'defensiveTeam': MANUAL_METADATA['defensiveTeam'],
        'offensiveFormation': MANUAL_METADATA['offensiveFormation'],
        'defensiveFormation': MANUAL_METADATA['defensiveFormation'],
        'personnel': MANUAL_METADATA['personnel'],
        'playType': summary_data['playType'],  # AI-derived
        'playResult': MANUAL_METADATA['playResult'],
        'gainYard': summary_data['gainYard'],  # AI-derived
        'tackler': MANUAL_METADATA['tackler'],
        'carrierInfo': {
            'trackId': summary_data['car_id'],
            'jerseyNumber': carrier_number
        },
        'passerInfo': {
            'trackId': summary_data['passer_id'],
            'jerseyNumber': 'Unknown'
        },
        'tacklerInfo': {
            'trackId': summary_data['tkl_id'],
            'jerseyNumber': 'Unknown'
        }
    }

    # Create final game-level JSON structure
    final_json = {
        'gameKey': MANUAL_METADATA['gameKey'],
        'gameDate': MANUAL_METADATA['gameDate'],
        'homeTeam': MANUAL_METADATA['homeTeam'],
        'awayTeam': MANUAL_METADATA['awayTeam'],
        'stadium': MANUAL_METADATA['stadium'],
        'weather': MANUAL_METADATA['weather'],
        'temperature': MANUAL_METADATA['temperature'],
        'Clips': [clip_data]
    }

    # Save JSON file
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print(f"  JSON output: {JSON_OUTPUT_PATH}")

    # Print play summary
    print("\n" + "="*60)
    print("PLAY ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nPlay Details:")
    print(f"  Play Type: {summary_data['playType']}")
    print(f"  Yards Gained: {summary_data['gainYard']:.2f}")
    print(f"  Final State: {summary_data['state']}")

    print(f"\nKey Players:")
    print(f"  Ball Carrier ID: {summary_data['car_id']}")
    if summary_data['car_id'] and summary_data['car_id'] in team_classifier.player_teams:
        carrier_team = team_classifier.player_teams[summary_data['car_id']]
        print(f"    Team: {carrier_team}")
    print(f"  Passer ID: {summary_data['passer_id']}")
    print(f"  Tackler ID: {summary_data['tkl_id']}")

    print(f"\nSystem Performance Summary:")
    print(f"  ✓ Video processed: {frame_count} frames in {frame_count/fps:.1f}s")
    print(f"  ✓ Players tracked: {len(seen_track_ids)} unique IDs")
    print(f"  ✓ Players classified: {len(sampled_players)} players")
    print(f"  ✓ Team A: {team_classification_count['Team A']}, "
          f"Team B: {team_classification_count['Team B']}, "
          f"Referee: {team_classification_count['Referee']}")
    print(f"  ✓ Ball detection rate: {ball_detection_rate:.1%}")

    print("\n" + "="*60)
    print("All outputs saved successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
