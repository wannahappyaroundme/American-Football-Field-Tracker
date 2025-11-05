# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a football play analysis system that uses computer vision and AI to automatically extract play statistics from game footage. The system employs a two-model pipeline (YOLOv8 detection + YOLOv8 pose) to track players, identify ball carriers, determine play types (RUN/PASS), and calculate yards gained.

**Technology Stack:**
- Python 3.8+
- **ultralytics** (YOLOv8 detection & pose estimation)
- **opencv-python** (cv2 for video processing, homography transformations)
- **numpy** (numerical computations)
- **scipy** (Euclidean distance calculations)
- **scikit-learn** (K-means clustering for team classification)
- **pytesseract** (OCR for jersey numbers - requires separate Tesseract installation)

**Note:** Some files contain Korean comments (e.g., config.py has "통합 설정 파일").

## Running the System

### Initial Setup (Required Once)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Note: Tesseract OCR must be installed separately on the system.

2. **Calibrate homography matrix:**
   ```bash
   python calibrate_homography.py
   ```
   This must be done before running the main pipeline. The tool displays the first video frame and requires clicking 4 field points (e.g., corners of hash marks), then pressing 's' to save. The homography matrix enables accurate BEV transformation and yardage calculations.

3. **Configure metadata:**
   Edit `MANUAL_METADATA` in `main.py` with game context (teams, date, quarter, etc.).

### Running Analysis

```bash
python main.py
```

This processes the video and generates:
- `output/result.mp4` - Annotated video with bounding boxes and play state
- `output/bev.mp4` - Bird's eye view with player paths
- `output/clip_summary.json` - Structured play data

## Dual Implementation Systems

This codebase contains **two separate implementations**:

### 1. Main Pipeline (main.py + config.py)
The primary, well-documented implementation described throughout this file.
- Uses modular component architecture (DetectorTracker, PlayAnalyzer, ViewTransformer, etc.)
- State machine with 4 states
- Team classification at frame 100
- Stores player crops for jersey number recognition

### 2. Alternative Pipeline (tracker.py + tracker_config.py)
An enhanced alternative implementation with additional features:
- **Stadium masking:** Filters detections to only include on-field players using green field detection
- **ROI filtering:** Defines region of interest to exclude sidelines/stands
- **Persistent dots with fade:** Visual trail dots that fade over time
- **Team assignment freezing:** Prevents team classification flickering
- Advanced field recognition using HSV color space for green field detection

**When to use which:**
- Use `main.py` for standard play analysis with comprehensive documentation
- Use `tracker.py` for scenarios requiring better sideline filtering or visual enhancements
- Both systems share similar architecture but have different configuration files

## Quick Reference

**Common Tasks:**

| Task | File to Modify | What to Change |
|------|---------------|----------------|
| Adjust detection sensitivity | `config.py` | `DETECTION_CONFIDENCE_THRESHOLD` (lower = more detections) |
| Change visualization colors | `config.py` | `COLOR_*` constants (BGR format) |
| Modify state machine logic | `play_analyzer.py` | `update_tracks()` method |
| Add new play types | `play_analyzer.py` | Add state transitions in `update_tracks()` |
| Change team classification timing | `main.py` | Frame number where `team_classifier.assign_teams()` is called |
| Adjust yardage calculation | `config.py` | `PIXELS_PER_YARD_BEV` (must match calibration) |
| Change yard line reference | `config.py` | `BEV_LEFT_YARD_LINE`, `BEV_DIRECTION` |
| Improve jersey number recognition | `number_recognizer.py` | OCR preprocessing in `recognize_number()` |
| Fine-tune team clustering | `team_classifier.py` | K-means parameters, HSV filtering thresholds |

## Architecture

### Data Flow Pipeline

```
Video Frame → DetectorTracker → PlayAnalyzer → Visualizer → Output Videos
                    ↓               ↓             ↓
              ViewTransformer   PoseModel    TeamClassifier
                                    ↓          (frame 100)
                              Play End Detection
```

### Main Pipeline Execution (main.py)

1. **Initialization Phase**
   - Load YOLO detection and pose models
   - Initialize all components (DetectorTracker, ViewTransformer, PlayAnalyzer, Visualizer, TeamClassifier)
   - Open video input/output streams

2. **Processing Loop**
   - Read frame-by-frame
   - Detect and track objects (players, ball)
   - Transform coordinates to BEV
   - Update play analyzer state machine
   - Collect color samples for team classification
   - **At frame 100**: Run team classification on all tracked players
   - Run pose detection on ball carrier
   - Draw annotations and BEV visualization
   - Break when state == 'PLAY_ENDED'

3. **Finalization Phase**
   - Recognize jersey numbers from stored crops
   - Generate JSON output with merged AI + manual metadata
   - Release video streams

### Critical Components

**config.py** - Central configuration hub
- All paths, model settings, detection thresholds, and visualization colors
- Key constants: `CLASS_ID_PERSON = 0`, `CLASS_ID_BALL = 32`, `PIXELS_PER_YARD_BEV = 20`
- Field configuration: `BEV_LEFT_YARD_LINE`, `BEV_DIRECTION` for yard line calculation

**play_analyzer.py** - The "brain" of the system
- Implements a 4-state state machine: `PRE_SNAP` → `PLAY_ACTIVE` → `BALL_IN_AIR` → `PLAY_ENDED`
- Maintains `player_states` dict with movement paths for each tracked player
- Tracks occluded players by predicting positions (marked as 'occluded' in path)
- Stores player crop images in `player_crops` dict for jersey number recognition
- State transitions:
  - PRE_SNAP → PLAY_ACTIVE: When ball carrier first detected
  - PLAY_ACTIVE → BALL_IN_AIR: Ball leaves carrier (auto-sets playType to "PASS")
  - BALL_IN_AIR → PLAY_ACTIVE: Ball caught by new player
  - PLAY_ACTIVE → PLAY_ENDED: Carrier's knee below ankle (pose detection)

**transformer.py** - Coordinate transformation
- Loads homography matrix from `homography_matrix.npy`
- `get_foot_position(bbox)`: Returns center-bottom point `((x1+x2)/2, y2)` - represents player's foot
- `transform_point(point_tuple)`: Converts camera view → BEV coordinates using `cv2.perspectiveTransform`
- `bev_to_yard_line(bev_pos)`: Converts BEV coordinates → field yard line info (side: OWN/OPP, yard: number)
- Critical for accurate distance/yardage measurements

**detector_tracker.py** - Object detection and tracking
- Wraps YOLOv8 with `persist=True` for consistent tracking across frames
- Returns tracks as list of dicts: `{'bbox': [x1,y1,x2,y2], 'track_id': int, 'class_id': int}`
- Returns empty list if `results[0].boxes.id is None` (no tracks)

**visualizer.py** - Two visualization methods
- `draw_annotations()`: Draws bounding boxes with color coding (blue=player, yellow=ball, red=ball carrier) and overlays play state/type
- `draw_bird_eye_view()`: Renders BEV with solid lines for 'active' paths and dotted lines for 'occluded' predictions

**number_recognizer.py** - Jersey number OCR
- Uses Tesseract OCR to recognize jersey numbers from player crop images
- Applies grayscale conversion and binary thresholding for better accuracy
- Configured to recognize digits only (0-9)
- Note: Side-view footage may limit recognition accuracy

**team_classifier.py** - Color-based team assignment
- Uses K-means clustering on jersey colors to identify teams
- Extracts dominant color from upper 40% of player crops (jersey area)
- Works in HSV color space for robustness (filters by saturation/value)
- Clusters players into 4 groups: Team A, Team B, Referee, Others
- Referee detection: identifies low-saturation clusters (black/white striped jerseys)
- "Others" detection: identifies smallest cluster (< 20% of samples, likely sideline staff)
- Called early in pipeline (frame 100) to classify all tracked players

### Key Algorithms

**Ball Carrier Identification**
- Finds person track with minimum Euclidean distance to ball in BEV coordinates
- Uses `scipy.spatial.distance.euclidean(ball_bev_pos, person_bev_pos)`

**Yards Gained Calculation**
```python
pixel_distance = euclidean_distance(start_ball_bev, end_ball_bev)
gain_yard = pixel_distance / PIXELS_PER_YARD_BEV
```

**Yard Line Calculation**
```python
yards_from_left = x_bev / PIXELS_PER_YARD_BEV
yard_line = BEV_LEFT_YARD_LINE + yards_from_left  # or adjusted based on BEV_DIRECTION
# Determine OWN vs OPP side (assuming 100-yard field)
side = 'OWN' if yard_line <= 50 else 'OPP'
yard = int(round(yard_line)) if side == 'OWN' else int(round(100 - yard_line))
```

**Play End Detection (Pose-Based)**
- Uses COCO keypoint indices: knee (13, 14), ankle (15, 16)
- Logic: `min_knee_y > max_ankle_y` → player is down
- Identifies tackler as closest person track to ball carrier at play end

**Jersey Number Recognition**
- Stores up to 5 crop images per player during tracking
- After play ends, selects best crop (largest by area) for each key player
- Runs Tesseract OCR with digit-only whitelist
- Returns recognized number or None if unreadable

**Team Classification**
- Collects color samples from player crops throughout tracking phase
- At frame 100 (or configurable point), performs K-means clustering
- Uses HSV color space with filtering: saturation > 30, value 30-220
- Averages multiple samples per player for robust color estimation
- Identifies special clusters:
  - Referee: lowest saturation cluster (< 50), indicating striped jerseys
  - Others: smallest cluster (< 20% of samples), indicating sideline personnel
- Remaining clusters assigned to Team A and Team B

## Configuration Details

**Homography Calibration**
- The 4 clicked points map to destination BEV coordinates: `[[0,500], [1000,500], [1000,0], [0,0]]`
- BEV canvas is always 1000×500 pixels
- `PIXELS_PER_YARD_BEV` must match the real-world dimensions of your calibrated rectangle
  - Formula: If rectangle = X yards wide, then `PIXELS_PER_YARD_BEV = 1000 / X`
  - Default: `PIXELS_PER_YARD_BEV = 20` → 50 yards wide (1000 / 20 = 50)

**Yard Line Configuration**
- `BEV_LEFT_YARD_LINE`: Reference yard line at x=0 in BEV (relative to offensive team)
- `BEV_DIRECTION`:
  - `'LEFT_TO_RIGHT'`: Offense moves from left to right (left = OWN, right = OPP)
  - `'RIGHT_TO_LEFT'`: Offense moves from right to left (right = OWN, left = OPP)
- These settings must match your calibration setup

**Detection Tuning**
- `DETECTION_CONFIDENCE_THRESHOLD = 0.3`: Lower = more detections (may include false positives)
- Only detects two classes: person (0) and sports ball (32) from COCO dataset

## Common Modifications

**Adjusting State Machine Logic**
- Modify `update_tracks()` in PlayAnalyzer for different state transition conditions
- Current logic assumes single-play clips; batch processing would require reset mechanism

**Improving Occlusion Handling**
- Current: Simple prediction (maintains last position for up to 30 frames)
- Enhancement: Implement Kalman filter in `update_tracks()` when `track_id not in current_track_ids`

**Enhancing Jersey Number Recognition**
- Current: Stores 5 crops per player, uses largest crop for OCR
- Enhancements:
  - Filter crops by quality metrics (sharpness, lighting)
  - Use ensemble voting across multiple crops
  - Apply image preprocessing (contrast enhancement, rotation correction)
  - Fine-tune for side-view angles

**Improving Team Classification**
- Current: K-means clustering on jersey colors with referee/others detection
- Enhancements:
  - Adjust timing of classification (currently frame 100)
  - Fine-tune HSV filtering thresholds for different lighting conditions
  - Add manual override for misclassified players
  - Consider player position/formation patterns in addition to color

**Handling Side-View Footage**
- YOLOv8 detection handles overlapping players well
- For severely occluded players, current system predicts position linearly
- PRE_SNAP stance detection: Pose model detects knee positions, but currently only used for play-end detection

## Important Constraints

- System processes one play per execution (breaks when state == 'PLAY_ENDED')
- Requires `homography_matrix.npy` to exist before running `main.py`
- Requires Tesseract OCR installed separately on the system
- Manual metadata must be set in code (no CLI arguments)
- Output paths are hardcoded in config.py
- YOLO models auto-download on first run (requires internet)
- Yard line calculation assumes 100-yard field structure
- Team classification runs at frame 100 (requires sufficient player visibility)

## JSON Output Schema

The system outputs a structured JSON file matching this schema:

```json
{
  "gameKey": "GAVI20240720",
  "date": "2024-07-20(토) 13:00",
  "type": "FriendlyMatch",
  "score": {"home": 0, "away": 0},
  "region": "GyeonggiGangwon",
  "location": "단국대 운동장",
  "homeTeam": "GyeonggiGangwonAllStar",
  "awayTeam": "SeoulVikings",
  "Clips": [
    {
      "clipKey": "1",
      "offensiveTeam": "Home",
      "quarter": 1,
      "down": 1,
      "toGoYard": 10,
      "playType": "RUN",
      "specialTeam": false,
      "start": {"side": "OWN", "yard": 25},
      "end": {"side": "OWN", "yard": 35},
      "gainYard": 10,
      "car": {"num": 23, "pos": null},
      "car2": {"num": 11, "pos": "QB"},
      "tkl": {"num": 79, "pos": null},
      "tkl2": {"num": null, "pos": null},
      "significantPlays": [null, null, null, null]
    }
  ]
}
```

**AI-Derived Fields** (automatically populated):
- `playType`: "RUN" or "PASS" based on ball movement
- `gainYard`: Calculated from BEV start/end positions
- `start` / `end`: Yard line positions derived from BEV coordinates
- `car.num`, `car2.num`, `tkl.num`: Jersey numbers from OCR (may be null if unreadable)
- `car.team`, `car2.team`, `tkl.team`: Team assignments from color clustering (Team A/B/Referee/Others)

**Manual Fields** (from MANUAL_METADATA in main.py):
- Game context: gameKey, date, teams, location, etc.
- Play context: quarter, down, toGoYard, offensiveTeam
- Formations and significant plays (currently placeholders)

## Common Pitfalls

1. **"FileNotFoundError: homography_matrix.npy"**
   - Cause: Running `main.py` before calibration
   - Solution: Run `python calibrate_homography.py` first

2. **Incorrect Yardage Calculations**
   - Cause: `PIXELS_PER_YARD_BEV` doesn't match calibration rectangle
   - Solution: Verify formula `PIXELS_PER_YARD_BEV = 1000 / rectangle_width_in_yards`
   - Example: If you calibrated a 50-yard section, use `PIXELS_PER_YARD_BEV = 20`

3. **Inverted Yard Lines (OWN shows as OPP)**
   - Cause: `BEV_DIRECTION` doesn't match camera angle
   - Solution: Toggle between `'LEFT_TO_RIGHT'` and `'RIGHT_TO_LEFT'` in config.py

4. **Missing Tesseract Error**
   - Cause: Tesseract OCR not installed on system
   - Solution:
     - macOS: `brew install tesseract`
     - Ubuntu: `sudo apt-get install tesseract-ocr`
     - Windows: Download from GitHub releases

5. **Team Classification Inaccurate**
   - Cause: Frame 100 may not have enough visible players
   - Solution: Adjust classification timing in `main.py` (e.g., move to frame 150 or 200)

6. **Play Ends Too Early/Late**
   - Cause: Pose detection threshold sensitivity
   - Solution: Review knee/ankle keypoint logic in `play_analyzer.py`

## Performance Considerations

- **Processing Speed:** Typically 10-30 FPS depending on hardware (real-time for 30 FPS video on modern GPUs)
- **Frame 100 Classification:** Team classification at frame 100 is configurable - adjust if players aren't visible yet
- **Occlusion Tracking:** Predictions maintained for up to 30 frames (1 second at 30 FPS)
- **Memory Usage:** Stores up to 5 crops per player; consider limiting for very long clips
- **YOLO Model Loading:** ~1-2 seconds on first run; models are cached afterward

## Debugging Tips

- If yardage is inaccurate: Recalibrate homography and verify `PIXELS_PER_YARD_BEV`
- If yard line calculations are wrong: Check `BEV_LEFT_YARD_LINE` and `BEV_DIRECTION` settings
- If missing detections: Lower `DETECTION_CONFIDENCE_THRESHOLD` in config
- If false positives: Raise threshold
- If jersey numbers not recognized: Check Tesseract installation, review crop quality in debugger
- Check state transitions by monitoring print statements in PlayAnalyzer
- BEV output helps visualize if transformation is correct (field should look rectangular)
- For overlapping players: Review detection confidence and consider lowering threshold
- Use `tracker.py` if sideline personnel are being detected as players (has stadium masking)

## Future Enhancements

**TODO Items:**
1. Position inference (RB, WR, QB, etc.) based on player behavior and jersey numbers
2. Secondary tackler detection (identify multiple defenders involved)
3. PRE_SNAP stance detection (verify all players in set position)
4. ~~Team classification using jersey color clustering~~ (✓ Implemented in team_classifier.py)
5. Support for KICKOFF/RETURN play types
6. Multi-clip batch processing with state machine reset
7. Enhanced occlusion tracking with Kalman filtering
8. Real-time feedback during processing (progress bar, preview window)
9. Improve team classification accuracy with manual overrides or formation analysis
