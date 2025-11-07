import cv2
import numpy as np
from config import (
    PLAYER_COLOR,
    BALL_COLOR,
    BALL_CARRIER_COLOR,
    CLASS_ID_PERSON,
    CLASS_ID_BALL,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
    REFEREE_COLOR,
    UNKNOWN_COLOR
)


class Visualizer:
    """
    A class to visualize tracking results and bird's eye view.
    """

    def __init__(self):
        """
        Initialize the Visualizer.
        """
        print("Visualizer initialized")

    def draw_annotations(self, frame, tracks, ball_carrier_id, play_state, play_type, view_transformer=None, team_classifier=None):
        """
        Draw annotations on the frame including bounding boxes, play state, and play type.

        Args:
            frame: The video frame to draw on (numpy array)
            tracks: List of track dictionaries with 'bbox', 'track_id', 'class_id'
            ball_carrier_id: The track_id of the player carrying the ball (or None)
            play_state: String representing the current play state
            play_type: String representing the current play type
            view_transformer: ViewTransformer instance for filtering off-field players (optional)
            team_classifier: TeamClassifier instance for displaying team info (optional)

        Returns:
            The annotated frame
        """
        # Draw play state and play type on the top-left corner
        y_offset = 30
        cv2.putText(frame, f"Play State: {play_state}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        y_offset += 35
        cv2.putText(frame, f"Play Type: {play_type}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        # Draw bounding boxes for all tracks
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            class_id = track['class_id']

            # Filter out off-field players if transformer is provided
            if view_transformer is not None and class_id == CLASS_ID_PERSON:
                foot_pos = view_transformer.get_foot_position(bbox)
                bev_pos = view_transformer.transform_point(foot_pos)

                if not view_transformer.is_on_field(bev_pos):
                    continue  # Skip drawing this player - they're off the field

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)

            # Determine the color based on class, entity type, team, and ball carrier status
            if class_id == CLASS_ID_BALL:
                color = BALL_COLOR
                # 공 신뢰도 표시
                ball_conf = track.get('confidence', 0.0)
                label = f"Ball #{track_id} ({ball_conf:.0%})"
            elif class_id == CLASS_ID_PERSON:
                # Get entity type if available from CLIP
                entity_type = track.get('entity_type', 'player')
                entity_confidence = track.get('entity_confidence', 0.0)

                # Get team info if available
                team_name = ""
                team_color = UNKNOWN_COLOR  # 기본값: 회색 (팀 미배정)
                if team_classifier is not None:
                    team = team_classifier.get_team(track_id)
                    team_name = team if team != 'Unknown' else ""
                    # Map team to color
                    if team == 'Team A':
                        team_color = TEAM_A_COLOR  # 파란색
                    elif team == 'Team B':
                        team_color = TEAM_B_COLOR  # 빨간색
                    elif team == 'Referee':
                        team_color = REFEREE_COLOR  # 노란색
                    else:
                        team_color = UNKNOWN_COLOR  # 회색

                # Determine color and label (볼 캐리어 = 주황색 우선!)
                if track_id == ball_carrier_id:
                    color = BALL_CARRIER_COLOR  # 주황색 (0, 165, 255) BGR
                    label_prefix = "🏈 CARRIER"
                elif entity_type == 'referee':
                    color = REFEREE_COLOR
                    label_prefix = "Referee"
                else:
                    color = team_color  # 팀 색상 (미배정 시 회색)
                    label_prefix = "Player"

                # Build label with freeze indicator
                label_parts = [label_prefix, f"#{track_id}"]
                if team_name:
                    label_parts.append(f"({team_name})")

                # 프리즈 상태 표시
                if team_classifier is not None and hasattr(team_classifier, 'clip_classifier'):
                    if team_classifier.clip_classifier and team_classifier.clip_classifier.is_frozen(track_id):
                        label_parts.append("🔒")  # 프리즈된 팀

                label = " ".join(label_parts)
            else:
                color = (128, 128, 128)  # Gray for unknown
                label = f"#{track_id}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def draw_bird_eye_view(self, player_states, map_size=(1000, 500), team_classifier=None):
        """
        Draw a bird's eye view visualization of player positions and paths.

        Args:
            player_states: Dictionary mapping track_id to player state information
                          Each state should contain 'path' with a list of (x, y, status) tuples
                          where status is 'active' or 'occluded'
            map_size: Tuple (width, height) for the BEV canvas size
            team_classifier: Optional TeamClassifier instance for color-coding by team

        Returns:
            The bird's eye view image (numpy array)
        """
        # Create a blank canvas for the bird's eye view
        width, height = map_size
        bev_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw each player's path
        for track_id, state in player_states.items():
            if 'path' not in state or len(state['path']) == 0:
                continue

            path = state['path']

            # Get team color for this player
            color = PLAYER_COLOR  # Default color
            if team_classifier is not None:
                team = team_classifier.get_team(track_id)
                if team == 'Team A':
                    color = TEAM_A_COLOR
                elif team == 'Team B':
                    color = TEAM_B_COLOR
                elif team == 'Referee':
                    color = REFEREE_COLOR
                elif team == 'Unknown':
                    color = UNKNOWN_COLOR

            # Draw path segments
            for i in range(1, len(path)):
                x1, y1, status1 = path[i - 1]
                x2, y2, status2 = path[i]

                # Convert coordinates to integers
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))

                # Draw solid line for 'active' segments, dotted for 'occluded'
                if status2 == 'active':
                    cv2.line(bev_image, pt1, pt2, color, 2)
                elif status2 == 'occluded':
                    # Draw dotted line by drawing short segments with gaps
                    self._draw_dotted_line(bev_image, pt1, pt2, color, 2, gap=5)

            # Draw the current position (last point in path)
            if len(path) > 0:
                last_x, last_y, last_status = path[-1]
                cv2.circle(bev_image, (int(last_x), int(last_y)), 5, color, -1)

                # Draw track ID label
                label = f"#{track_id}"
                cv2.putText(bev_image, label, (int(last_x) + 8, int(last_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return bev_image

    def _draw_dotted_line(self, image, pt1, pt2, color, thickness, gap=5):
        """
        Draw a dotted line on the image.

        Args:
            image: The image to draw on
            pt1: Start point (x1, y1)
            pt2: End point (x2, y2)
            color: Line color
            thickness: Line thickness
            gap: Gap size between dots
        """
        import numpy as np

        x1, y1 = pt1
        x2, y2 = pt2

        # Calculate distance and number of segments
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_segments = int(dist / gap)

        if num_segments == 0:
            return

        # Draw dotted segments
        for i in range(0, num_segments, 2):
            t1 = i / num_segments
            t2 = min((i + 1) / num_segments, 1.0)

            seg_pt1 = (int(x1 + t1 * (x2 - x1)), int(y1 + t1 * (y2 - y1)))
            seg_pt2 = (int(x1 + t2 * (x2 - x1)), int(y1 + t2 * (y2 - y1)))

            cv2.line(image, seg_pt1, seg_pt2, color, thickness)
