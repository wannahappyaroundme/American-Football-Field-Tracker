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
    POSE_MODEL_PATH
)

# Manual metadata - placeholders for game and clip information
MANUAL_METADATA = {
    # Game-level information
    'gameKey': 'GAVI20240720',
    'date': '2024-07-20(토) 13:00',
    'type': 'FriendlyMatch',
    'score': {'home': 0, 'away': 0},  # Update with actual score
    'region': 'GyeonggiGangwon',
    'location': '단국대 운동장',
    'homeTeam': 'GyeonggiGangwonAllStar',
    'awayTeam': 'SeoulVikings',

    # Clip-level information
    'clipKey': '1',
    'offensiveTeam': 'Home',  # or 'Away'
    'quarter': 1,
    'down': 1,
    'toGoYard': 10,
    'specialTeam': False,  # Set to True for KICKOFF/RETURN plays

    # Start/End positions (will be calculated by yard line logic)
    'start': {'side': 'OWN', 'yard': 25},
    'end': {'side': 'OWN', 'yard': 35},

    # Significant plays
    'significantPlays': [None, None, None, None]
}


def main():
    """
    Main function to run the football play analysis pipeline.
    """
    print("="*60)
    print("Football Play Analysis System")
    print("="*60)

    # Initialize detection and pose models
    print("\n[1/6] Loading models...")
    detector = DetectorTracker()
    pose_model = YOLO(POSE_MODEL_PATH)
    print(f"Loaded pose model from {POSE_MODEL_PATH}")

    # Initialize other components
    print("\n[2/6] Initializing components...")
    transformer = ViewTransformer()
    analyzer = PlayAnalyzer()
    visualizer = Visualizer()
    recognizer = NumberRecognizer()
    team_classifier = TeamClassifier()

    # Set up video capture and writers
    print("\n[3/6] Setting up video input/output...")
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_INPUT_PATH}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_annotated = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))
    out_bev = cv2.VideoWriter(BEV_OUTPUT_PATH, fourcc, fps, (1000, 500))

    # Main processing loop
    print("\n[4/6] Processing video frames...")
    frame_count = 0

    # Debug: Track BEV coordinates for the first few frames
    debug_bev_coords = []

    # Early team classification flag
    team_classified = False
    EARLY_CLASSIFICATION_FRAME = 100  # Classify teams after 100 frames

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or read error.")
            break

        frame_count += 1

        # Early team classification after collecting enough samples
        if not team_classified and frame_count == EARLY_CLASSIFICATION_FRAME:
            print(f"\nPerforming early team classification at frame {frame_count}...")
            team_classifier.classify_teams(n_teams=4)
            team_classified = True

        # Track objects in the frame
        tracks = detector.track_frame(frame)

        # Update analyzer with new tracks
        analyzer.update_tracks(tracks, transformer)

        # Store crop images for all person tracks (for jersey number recognition and team classification)
        for track in tracks:
            if track['class_id'] == 0:  # CLASS_ID_PERSON
                track_id = track['track_id']
                bbox = track['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                # Ensure valid crop coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                if x2 > x1 and y2 > y1:
                    crop_image = frame[y1:y2, x1:x2].copy()

                    # Check if player is on field before storing
                    foot_pos = transformer.get_foot_position(bbox)
                    bev_pos = transformer.transform_point(foot_pos)

                    # Debug: Collect BEV coordinates for first 10 detections
                    if len(debug_bev_coords) < 10:
                        debug_bev_coords.append({
                            'track_id': track_id,
                            'bev_pos': bev_pos,
                            'on_field': transformer.is_on_field(bev_pos)
                        })

                    if transformer.is_on_field(bev_pos):
                        analyzer.store_player_crop(track_id, crop_image)
                        team_classifier.collect_color_sample(track_id, crop_image)

        # Pose detection for ball carrier (if play is active)
        if analyzer.ball_carrier_id is not None and analyzer.state != 'PLAY_ENDED':
            # Find the ball carrier's track
            ball_carrier_track = None
            for track in tracks:
                if track['track_id'] == analyzer.ball_carrier_id:
                    ball_carrier_track = track
                    break

            if ball_carrier_track is not None:
                # Crop the frame to ball carrier's bbox
                bbox = ball_carrier_track['bbox']
                x1, y1, x2, y2 = map(int, bbox)

                # Ensure valid crop coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                if x2 > x1 and y2 > y1:
                    crop_frame = frame[y1:y2, x1:x2]

                    # Run pose detection on cropped frame
                    pose_results = pose_model(crop_frame, verbose=False)

                    # Check for play end using pose keypoints
                    analyzer.check_play_end(pose_results, tracks, transformer)

        # Draw annotations on the frame (with off-field filtering and team labels)
        annotated_frame = visualizer.draw_annotations(
            frame.copy(),
            tracks,
            analyzer.ball_carrier_id,
            analyzer.state,
            analyzer.play_type,
            view_transformer=transformer,  # Pass transformer for filtering
            team_classifier=team_classifier  # Pass team classifier for labels
        )

        # Draw bird's eye view
        bev_frame = visualizer.draw_bird_eye_view(analyzer.player_states)

        # Write output frames
        out_annotated.write(annotated_frame)
        out_bev.write(bev_frame)

        # Print progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames - State: {analyzer.state}")

        # Break if play has ended
        if analyzer.state == 'PLAY_ENDED':
            print(f"\nPlay ended at frame {frame_count}")
            break

    # Final team classification if not done early
    if not team_classified:
        print("\n[5/6] Classifying teams based on jersey colors...")
        team_classifier.classify_teams(n_teams=4)  # Team A, Team B, Referee, Others

    # Release video objects
    cap.release()
    out_annotated.release()
    out_bev.release()
    cv2.destroyAllWindows()

    print(f"\n[5.5/6] Video processing complete.")
    print(f"Annotated video saved to: {VIDEO_OUTPUT_PATH}")
    print(f"BEV video saved to: {BEV_OUTPUT_PATH}")

    # Debug: Print BEV coordinates
    from config import BEV_FIELD_X_MIN, BEV_FIELD_X_MAX, BEV_FIELD_Y_MIN, BEV_FIELD_Y_MAX
    print("\n=== DEBUG: BEV Coordinates Sample ===")
    for info in debug_bev_coords:
        print(f"Track ID {info['track_id']}: BEV=({info['bev_pos'][0]:.1f}, {info['bev_pos'][1]:.1f}), On-field={info['on_field']}")
    print(f"Field bounds: X=[{BEV_FIELD_X_MIN}, {BEV_FIELD_X_MAX}], Y=[{BEV_FIELD_Y_MIN}, {BEV_FIELD_Y_MAX}]")
    print("=" * 40)

    summary_data = analyzer.get_summary()

    # Calculate start and end yard lines from BEV positions
    print("Calculating yard line positions...")
    start_yard_info = {'side': 'OWN', 'yard': 0}
    end_yard_info = {'side': 'OWN', 'yard': 0}

    if summary_data['playType'] != 'UNSURE':
        if analyzer.start_ball_bev is not None:
            start_yard_info = transformer.bev_to_yard_line(analyzer.start_ball_bev)
            print(f"  Start position: {start_yard_info['side']} {start_yard_info['yard']}")

        if analyzer.end_ball_bev is not None:
            end_yard_info = transformer.bev_to_yard_line(analyzer.end_ball_bev)
            print(f"  End position: {end_yard_info['side']} {end_yard_info['yard']}")

    # Get team assignments for key players
    carrier_team = None
    passer_team = None
    tackler_team = None

    if summary_data['car_id'] is not None:
        carrier_team = team_classifier.get_team(summary_data['car_id'])
        print(f"  Ball carrier team: {carrier_team} (track_id: {summary_data['car_id']})")

    if summary_data['passer_id'] is not None:
        passer_team = team_classifier.get_team(summary_data['passer_id'])
        print(f"  Passer team: {passer_team} (track_id: {summary_data['passer_id']})")

    if summary_data['tkl_id'] is not None:
        tackler_team = team_classifier.get_team(summary_data['tkl_id'])
        print(f"  Tackler team: {tackler_team} (track_id: {summary_data['tkl_id']})")

    # Attempt to recognize jersey numbers for key players
    print("Attempting jersey number recognition...")

    carrier_number = None
    passer_number = None
    tackler_number = None

    # Recognize carrier number
    if summary_data['car_id'] is not None:
        best_crop = analyzer.get_best_crop(summary_data['car_id'])
        if best_crop is not None:
            recognized = recognizer.recognize_number(best_crop)
            if recognized and recognized.isdigit():
                carrier_number = int(recognized)
                print(f"  Ball carrier #{carrier_number} (track_id: {summary_data['car_id']})")
            else:
                print(f"  Ball carrier: Could not recognize number (track_id: {summary_data['car_id']})")

    # Recognize passer number (if different from carrier)
    if summary_data['passer_id'] is not None and summary_data['passer_id'] != summary_data['car_id']:
        best_crop = analyzer.get_best_crop(summary_data['passer_id'])
        if best_crop is not None:
            recognized = recognizer.recognize_number(best_crop)
            if recognized and recognized.isdigit():
                passer_number = int(recognized)
                print(f"  Passer #{passer_number} (track_id: {summary_data['passer_id']})")
            else:
                print(f"  Passer: Could not recognize number (track_id: {summary_data['passer_id']})")

    # Recognize tackler number
    if summary_data['tkl_id'] is not None:
        best_crop = analyzer.get_best_crop(summary_data['tkl_id'])
        if best_crop is not None:
            recognized = recognizer.recognize_number(best_crop)
            if recognized and recognized.isdigit():
                tackler_number = int(recognized)
                print(f"  Tackler #{tackler_number} (track_id: {summary_data['tkl_id']})")
            else:
                print(f"  Tackler: Could not recognize number (track_id: {summary_data['tkl_id']})")

    # Create clip data structure matching the provided schema
    clip_data = {
        'clipKey': MANUAL_METADATA['clipKey'],
        'offensiveTeam': MANUAL_METADATA['offensiveTeam'],
        'quarter': MANUAL_METADATA['quarter'],
        'down': MANUAL_METADATA['down'],
        'toGoYard': MANUAL_METADATA['toGoYard'],
        'playType': summary_data['playType'],  # AI-derived: 'RUN' or 'PASS'
        'specialTeam': MANUAL_METADATA['specialTeam'],
        'start': start_yard_info,  # AI-derived from BEV coordinates
        'end': end_yard_info,      # AI-derived from BEV coordinates
        'gainYard': summary_data['gainYard'],  # AI-derived
        'car': {
            'num': carrier_number,  # AI-derived (OCR) or None
            'pos': None,  # Position inference not implemented
            'team': carrier_team  # AI-derived (color clustering)
        },
        'car2': {
            'num': passer_number if summary_data['playType'] == 'PASS' else None,
            'pos': 'QB' if summary_data['playType'] == 'PASS' else None,
            'team': passer_team if summary_data['playType'] == 'PASS' else None
        },
        'tkl': {
            'num': tackler_number,  # AI-derived (OCR) or None
            'pos': None,  # Position inference not implemented
            'team': tackler_team  # AI-derived (color clustering)
        },
        'tkl2': {
            'num': None,  # Secondary tackler detection not implemented
            'pos': None,
            'team': None
        },
        'significantPlays': MANUAL_METADATA['significantPlays']
    }

    # Create final game-level JSON structure
    final_json = {
        'gameKey': MANUAL_METADATA['gameKey'],
        'date': MANUAL_METADATA['date'],
        'type': MANUAL_METADATA['type'],
        'score': MANUAL_METADATA['score'],
        'region': MANUAL_METADATA['region'],
        'location': MANUAL_METADATA['location'],
        'homeTeam': MANUAL_METADATA['homeTeam'],
        'awayTeam': MANUAL_METADATA['awayTeam'],
        'Clips': [clip_data]
    }

    # Save JSON file
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print(f"JSON summary saved to: {JSON_OUTPUT_PATH}")

    # Print summary
    print("\n" + "="*60)
    print("PLAY SUMMARY")
    print("="*60)
    print(f"Play Type: {summary_data['playType']}")
    print(f"Yards Gained: {summary_data['gainYard']:.2f}")
    print(f"Ball Carrier ID: {summary_data['car_id']}")
    print(f"Passer ID: {summary_data['passer_id']}")
    print(f"Tackler ID: {summary_data['tkl_id']}")
    print(f"Final State: {summary_data['state']}")
    print("="*60)


if __name__ == "__main__":
    main()
