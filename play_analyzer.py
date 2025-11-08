from scipy.spatial import distance
from config import CLASS_ID_PERSON, CLASS_ID_BALL, PIXELS_PER_YARD_BEV


class PlayAnalyzer:
    """
    The brain of the operation - analyzes play state, tracks players,
    identifies ball carriers, and calculates statistics.
    """

    def __init__(self):
        """
        Initialize the PlayAnalyzer with state machine and storage.
        """
        # State machine - possible states: 'PRE_SNAP', 'PLAY_ACTIVE', 'BALL_IN_AIR', 'PLAY_ENDED'
        self.state = 'PRE_SNAP'

        # Storage for player states and paths
        self.player_states = {}  # Dict mapping track_id to player state info

        # Key player identifiers
        self.ball_carrier_id = None
        self.passer_id = None
        self.tackler_id = None

        # Play information
        self.play_type = 'RUN'  # 기본값 RUN (던지지 않으면 모두 RUN)

        # Ball positions for yardage calculation
        self.start_ball_bev = None
        self.end_ball_bev = None
        self.gain_yard = 0

        # Storage for player crop images (for jersey number recognition)
        self.player_crops = {}  # Dict mapping track_id to list of crop images

        print("PlayAnalyzer initialized in PRE_SNAP state")

    def update_tracks(self, tracks, view_transformer):
        """
        Update player tracking and state machine logic.

        Args:
            tracks: List of track dictionaries with 'bbox', 'track_id', 'class_id'
            view_transformer: ViewTransformer instance for coordinate transformation

        Returns:
            None
        """
        # Separate ball and person tracks
        ball_track = None
        person_tracks = []

        for track in tracks:
            if track['class_id'] == CLASS_ID_BALL:
                ball_track = track
            elif track['class_id'] == CLASS_ID_PERSON:
                person_tracks.append(track)

        # Update player positions in player_states
        # ⭐ NOW USES stable_id instead of track_id for jersey-based tracking
        current_track_ids = set()
        for person_track in person_tracks:
            track_id = person_track.get('stable_id', person_track.get('track_id'))  # Use stable_id

            # Get foot position and transform to BEV
            foot_pos = view_transformer.get_foot_position(person_track['bbox'])
            bev_pos = view_transformer.transform_point(foot_pos)

            # ⭐ OFF-FIELD FILTERING DISABLED: Accept all 25-30 detections for CLIP
            # Previous filtering removed too many players (BEV boundaries too strict)
            # CLIP will distinguish on-field players vs sideline staff

            # if not view_transformer.is_on_field(bev_pos):
            #     continue  # Skip this player - they're off the field

            current_track_ids.add(track_id)

            # Initialize player state if new
            if track_id not in self.player_states:
                self.player_states[track_id] = {
                    'path': [],
                    'last_seen_frame': 0,
                    'last_known_position': None
                }

            # Add position to path with 'active' status
            self.player_states[track_id]['path'].append((bev_pos[0], bev_pos[1], 'active'))
            self.player_states[track_id]['last_known_position'] = bev_pos
            self.player_states[track_id]['last_seen_frame'] = 0  # Reset counter

        # Handle occluded players (not currently detected)
        for track_id in list(self.player_states.keys()):
            if track_id not in current_track_ids:
                state = self.player_states[track_id]
                state['last_seen_frame'] += 1

                # Predict position for occluded player (simple: keep last position)
                if state['last_known_position'] is not None and state['last_seen_frame'] <= 30:
                    # Add predicted position with 'occluded' status
                    x, y = state['last_known_position']
                    state['path'].append((x, y, 'occluded'))

        # Identify ball carrier (closest person to ball)
        if ball_track is not None:
            ball_foot_pos = view_transformer.get_foot_position(ball_track['bbox'])
            ball_bev_pos = view_transformer.transform_point(ball_foot_pos)

            # Find closest person to ball (only consider on-field players)
            min_distance = float('inf')
            closest_person_id = None

            for person_track in person_tracks:
                person_id = person_track.get('stable_id', person_track.get('track_id'))  # Use stable_id
                person_foot_pos = view_transformer.get_foot_position(person_track['bbox'])
                person_bev_pos = view_transformer.transform_point(person_foot_pos)

                # Skip off-field players
                if not view_transformer.is_on_field(person_bev_pos):
                    continue

                dist = distance.euclidean(ball_bev_pos, person_bev_pos)

                if dist < min_distance:
                    min_distance = dist
                    closest_person_id = person_id

            # Update ball carrier
            previous_carrier = self.ball_carrier_id
            self.ball_carrier_id = closest_person_id

            # State machine logic
            if self.state == 'PRE_SNAP' and closest_person_id is not None:
                # Play has started
                self.state = 'PLAY_ACTIVE'
                self.play_type = 'RUN'  # Default to RUN (던지지 않으면 모두 RUN)
                self.passer_id = closest_person_id
                self.start_ball_bev = ball_bev_pos
                print(f"Play started: State -> PLAY_ACTIVE (RUN), Passer: {self.passer_id}")

            elif self.state == 'BALL_IN_AIR' and closest_person_id is not None:
                # Ball was caught
                self.state = 'PLAY_ACTIVE'
                print(f"Ball caught by player {closest_person_id}: State -> PLAY_ACTIVE")

        else:
            # Ball not detected - do NOT set to PASS just because ball is missing!
            # This was the bug - ball detection failure != ball in air
            pass

    def check_play_end(self, ball_carrier_pose, tracks, view_transformer):
        """
        Check if the play has ended by analyzing the ball carrier's pose.

        Args:
            ball_carrier_pose: Keypoints result from the Pose model (or None)
            tracks: List of current tracks
            view_transformer: ViewTransformer instance

        Returns:
            None
        """
        if self.state != 'PLAY_ACTIVE' or self.ball_carrier_id is None:
            return

        if ball_carrier_pose is None:
            return

        # Check if carrier is down using keypoints
        # Keypoint indices (COCO format): 13=left knee, 14=right knee, 15=left ankle, 16=right ankle
        try:
            keypoints = ball_carrier_pose[0].keypoints.xy[0].cpu().numpy()

            if len(keypoints) >= 17:
                left_knee = keypoints[13]
                right_knee = keypoints[14]
                left_ankle = keypoints[15]
                right_ankle = keypoints[16]

                # Check if knee is below ankle (player is down)
                # Use the lower knee and higher ankle for robust check
                min_knee_y = min(left_knee[1], right_knee[1])
                max_ankle_y = max(left_ankle[1], right_ankle[1])

                if min_knee_y > max_ankle_y:  # Y increases downward in image coordinates
                    # Player is down - play ended
                    self.state = 'PLAY_ENDED'

                    # Find ball carrier's BEV position for end position
                    ball_carrier_track = None
                    for track in tracks:
                        track_stable_id = track.get('stable_id', track.get('track_id'))
                        if track_stable_id == self.ball_carrier_id:
                            ball_carrier_track = track
                            break

                    if ball_carrier_track is not None:
                        foot_pos = view_transformer.get_foot_position(ball_carrier_track['bbox'])
                        self.end_ball_bev = view_transformer.transform_point(foot_pos)

                    # Find closest other person (tackler)
                    min_distance = float('inf')
                    closest_tackler_id = None

                    for track in tracks:
                        track_stable_id = track.get('stable_id', track.get('track_id'))
                        if track['class_id'] == CLASS_ID_PERSON and track_stable_id != self.ball_carrier_id:
                            person_foot_pos = view_transformer.get_foot_position(track['bbox'])
                            person_bev_pos = view_transformer.transform_point(person_foot_pos)

                            if self.end_ball_bev is not None:
                                dist = distance.euclidean(self.end_ball_bev, person_bev_pos)
                                if dist < min_distance:
                                    min_distance = dist
                                    closest_tackler_id = track_stable_id

                    self.tackler_id = closest_tackler_id

                    # Calculate yards gained
                    self.calculate_yards_gained()

                    print(f"Play ended: Carrier down. Tackler: {self.tackler_id}, Yards gained: {self.gain_yard:.2f}")

        except Exception as e:
            print(f"Error checking play end: {e}")

    def calculate_yards_gained(self):
        """
        Calculate the yards gained during the play.

        Returns:
            None (updates self.gain_yard)
        """
        if self.start_ball_bev is None or self.end_ball_bev is None:
            self.gain_yard = 0
            return

        # Calculate Euclidean distance in BEV coordinates
        pixel_distance = distance.euclidean(self.start_ball_bev, self.end_ball_bev)

        # Convert pixels to yards
        self.gain_yard = pixel_distance / PIXELS_PER_YARD_BEV

        # Debug info
        print(f"  Yard calculation: {pixel_distance:.1f} pixels / {PIXELS_PER_YARD_BEV} = {self.gain_yard:.2f} yards")
        print(f"  Start BEV: ({self.start_ball_bev[0]:.1f}, {self.start_ball_bev[1]:.1f})")
        print(f"  End BEV: ({self.end_ball_bev[0]:.1f}, {self.end_ball_bev[1]:.1f})")

    def store_player_crop(self, track_id, crop_image):
        """
        Store a crop image for a player to use for jersey number recognition later.

        Args:
            track_id: The track ID of the player
            crop_image: The cropped image of the player

        Returns:
            None
        """
        if track_id not in self.player_crops:
            self.player_crops[track_id] = []

        # Store up to 5 crop images per player (to later pick the best one)
        if len(self.player_crops[track_id]) < 5:
            self.player_crops[track_id].append(crop_image)

    def get_best_crop(self, track_id):
        """
        Get the best quality crop image for a player (largest by area).

        Args:
            track_id: The track ID of the player

        Returns:
            The best crop image, or None if no crops available
        """
        if track_id not in self.player_crops or len(self.player_crops[track_id]) == 0:
            return None

        # Return the largest crop (by area) as a proxy for best quality
        crops = self.player_crops[track_id]
        best_crop = max(crops, key=lambda img: img.shape[0] * img.shape[1])
        return best_crop

    def analyze_player_posture(self, pose_result):
        """
        Analyze player posture to detect crouching/stance.

        COCO keypoints (17 total):
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        5: left_shoulder, 6: right_shoulder
        7: left_elbow, 8: right_elbow
        9: left_wrist, 10: right_wrist
        11: left_hip, 12: right_hip
        13: left_knee, 14: right_knee
        15: left_ankle, 16: right_ankle

        Args:
            pose_result: Keypoints result from YOLOv8 pose model

        Returns:
            Dict with posture info: {'is_crouching': bool, 'hip_height': float, 'knee_angle': float}
        """
        if pose_result is None:
            return {'is_crouching': False, 'hip_height': 0, 'knee_angle': 0}

        try:
            keypoints = pose_result[0].keypoints.xy[0].cpu().numpy()

            if len(keypoints) < 17:
                return {'is_crouching': False, 'hip_height': 0, 'knee_angle': 0}

            # Get key body points
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_knee = keypoints[13]
            right_knee = keypoints[14]

            # Calculate average positions
            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            avg_knee_y = (left_knee[1] + right_knee[1]) / 2

            # Hip height (distance from shoulder to hip) - lower = more crouched
            hip_height = avg_hip_y - avg_shoulder_y

            # Knee bend (distance from hip to knee) - smaller = more bent
            knee_bend = avg_knee_y - avg_hip_y

            # Crouching detection: hip-shoulder distance < 70% of hip-knee distance
            # This indicates player is bent forward/down
            is_crouching = hip_height < (knee_bend * 0.7)

            return {
                'is_crouching': is_crouching,
                'hip_height': float(hip_height),
                'knee_bend': float(knee_bend),
                'stance': 'CROUCHED' if is_crouching else 'UPRIGHT'
            }

        except Exception as e:
            print(f"Error analyzing posture: {e}")
            return {'is_crouching': False, 'hip_height': 0, 'knee_angle': 0}

    def detect_player_movement(self, track_id, current_bev_pos):
        """
        Detect player movement speed and direction.

        Args:
            track_id: The track ID of the player
            current_bev_pos: Current BEV position (x, y)

        Returns:
            Dict with movement info: {'speed': float, 'direction': str, 'is_moving': bool}
        """
        if track_id not in self.player_states:
            # Initialize tracking for this player
            self.player_states[track_id] = {
                'path': [current_bev_pos],
                'state': 'active',
                'last_seen_frame': 0
            }
            return {'speed': 0.0, 'direction': 'STATIONARY', 'is_moving': False}

        player_state = self.player_states[track_id]
        path = player_state['path']

        # Need at least 2 positions to calculate movement
        if len(path) < 2:
            path.append(current_bev_pos)
            return {'speed': 0.0, 'direction': 'STATIONARY', 'is_moving': False}

        # Get last position
        last_pos = path[-1]

        # Calculate distance moved (in pixels)
        distance_moved = distance.euclidean(last_pos, current_bev_pos)

        # Convert to yards per frame (approximate speed)
        speed_yards_per_frame = distance_moved / PIXELS_PER_YARD_BEV

        # Determine if player is moving (threshold: 0.5 yards per frame)
        is_moving = speed_yards_per_frame > 0.5

        # Calculate direction
        dx = current_bev_pos[0] - last_pos[0]
        dy = current_bev_pos[1] - last_pos[1]

        # Determine movement direction
        if abs(dx) < 5 and abs(dy) < 5:  # Small threshold in pixels
            direction = 'STATIONARY'
        elif abs(dx) > abs(dy):
            direction = 'HORIZONTAL' if dx > 0 else 'HORIZONTAL_BACK'
        else:
            direction = 'VERTICAL' if dy > 0 else 'VERTICAL_BACK'

        # Update path (keep last 30 positions for analysis)
        path.append(current_bev_pos)
        if len(path) > 30:
            path.pop(0)

        return {
            'speed': float(speed_yards_per_frame),
            'direction': direction,
            'is_moving': is_moving,
            'distance_moved': float(distance_moved)
        }

    def get_summary(self):
        """
        Get a summary of the play with all collected data.

        Returns:
            Dictionary containing play information
        """
        return {
            'playType': self.play_type,
            'gainYard': round(self.gain_yard, 2),
            'car_id': self.ball_carrier_id,
            'passer_id': self.passer_id,
            'tkl_id': self.tackler_id,
            'state': self.state
        }
