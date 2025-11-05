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
    POSE_MODEL_PATH,
    ENABLE_CLIP_CLASSIFICATION,
    ENABLE_CLIP_ENTITY_FILTERING,
    ENABLE_CLIP_TEAM_CLASSIFICATION,
    CLASS_ID_PERSON
)

# Import CLIP classifiers if enabled
if ENABLE_CLIP_CLASSIFICATION:
    try:
        from clip_classifier import CLIPEntityClassifier, CLIPTeamClassifier
        CLIP_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: CLIP not available. Please install dependencies: pip install -r requirements.txt")
        print(f"Error: {e}")
        CLIP_AVAILABLE = False
else:
    CLIP_AVAILABLE = False

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
    print("Football Play Analysis System")
    print("="*60)

    # Initialize CLIP classifiers if available
    clip_entity_classifier = None
    clip_team_classifier = None

    if CLIP_AVAILABLE:
        print("\n[1/7] Initializing CLIP classifiers...")
        try:
            if ENABLE_CLIP_ENTITY_FILTERING:
                clip_entity_classifier = CLIPEntityClassifier()
                print("✓ CLIP entity classifier loaded")

            if ENABLE_CLIP_TEAM_CLASSIFICATION:
                clip_team_classifier = CLIPTeamClassifier()
                print("✓ CLIP team classifier loaded")
        except Exception as e:
            print(f"Warning: Failed to load CLIP classifiers: {e}")
            print("Continuing without CLIP classification...")
            clip_entity_classifier = None
            clip_team_classifier = None

    # Initialize detection and pose models
    print("\n[2/7] Loading YOLO models...")
    detector = DetectorTracker(clip_classifier=clip_entity_classifier)
    pose_model = YOLO(POSE_MODEL_PATH)
    print(f"Loaded pose model from {POSE_MODEL_PATH}")

    # Initialize other components
    print("\n[3/7] Initializing components...")
    transformer = ViewTransformer()
    analyzer = PlayAnalyzer()
    visualizer = Visualizer()
    recognizer = NumberRecognizer()
    team_classifier = TeamClassifier(clip_team_classifier=clip_team_classifier)

    # Set up video capture and writers
    print("\n[4/7] Setting up video input/output...")
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
    print("\n[5/7] Processing video frames...")
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or read error.")
            break

        frame_count += 1

        # Track objects in the frame (with CLIP entity filtering)
        tracks = detector.track_frame(frame)

        # Classify teams using CLIP if enabled (프리즈된 팀은 건너뛰기)
        if clip_team_classifier and frame_count % 10 == 0:  # Every 10 frames
            for track in tracks:
                if track.get('entity_type') == 'player':  # Only classify players
                    track_id = track['track_id']

                    # 이미 프리즈된 팀은 재분류 건너뛰기 (성능 최적화)
                    if clip_team_classifier.is_frozen(track_id):
                        continue

                    # 새 선수 또는 프리즈 안 된 선수만 분류
                    x1, y1, x2, y2 = map(int, track['bbox'])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        team_classifier.assign_team_with_clip(track_id, crop)

        # Update analyzer with new tracks
        analyzer.update_tracks(tracks, transformer)

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
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames - State: {analyzer.state}")

        # Break if play has ended
        if analyzer.state == 'PLAY_ENDED':
            print(f"\nPlay ended at frame {frame_count}")
            break

    # Release video objects
    cap.release()
    out_annotated.release()
    out_bev.release()
    cv2.destroyAllWindows()

    print(f"\n[6/7] Video processing complete.")
    print(f"Annotated video saved to: {VIDEO_OUTPUT_PATH}")
    print(f"BEV video saved to: {BEV_OUTPUT_PATH}")

    # Print team classification results
    if clip_team_classifier:
        print("\n👥 Team Classification Results:")
        team_counts = {}
        for track_id, team in clip_team_classifier.player_teams.items():
            team_counts[team] = team_counts.get(team, 0) + 1
        for team, count in team_counts.items():
            print(f"  {team}: {count} players")

        frozen_count = len(clip_team_classifier.frozen_teams)
        print(f"  Frozen teams: {frozen_count} players")

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
