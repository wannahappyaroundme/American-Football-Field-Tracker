# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **offline American football video analysis system** that uses YOLOv8 object detection to track players, classify them by team based on jersey color, and generate a side-by-side video output showing both the annotated original footage and a tactical top-down field map.

**Core Philosophy**: Accuracy and detailed analysis are prioritized over real-time performance. This is a batch processing system designed for post-game analysis, not live streaming.

## System Architecture

### Single-Script Monolithic Design

The system is intentionally designed as a consolidated single-script architecture:

- **[tracker.py](tracker.py)** (~1150 lines): Complete analysis pipeline with all processing stages
- **[tracker_config.py](tracker_config.py)**: All tunable parameters separated into configuration file

**Key Design Decision**: While this could be split into modules, the monolithic approach provides simplicity, easier maintenance, and better understandability for a focused application. All functionality flows linearly through the pipeline.

### Processing Pipeline (Sequential Stages)

The system processes video through these stages **in order**:

1. **Stadium/Field Recognition (Stage 0)** - Creates HSV-based mask to isolate the playing field and exclude non-field areas (crowd, coaches, sideline personnel)
2. **Static Homography Calculation (Stage 1)** - Calculates transformation matrix from first frame, then caches and reuses it for all subsequent frames ("cookie value" approach)
3. **YOLOv8 Object Detection (Stage 2)** - Runs on masked frames (background removed) to detect players
4. **Team Classification (Stage 3)** - Uses K-Means clustering on jersey torso colors (HSV color space)
5. **Robust Object Tracking (Stage 3.5)** - RobustTracker maintains player IDs across frames and **permanently freezes team assignments** after confidence is established
6. **Coordinate Transformation (Stage 4)** - Maps player foot positions to top-down tactical view using cached homography
7. **Visualization (Stage 5)** - Side-by-side output with persistent dots on tactical map

### Critical Implementation Details

**Team Assignment Freezing (Most Important Feature)**:
- Problem: Per-frame color detection is noisy, causing team assignments to flicker
- Solution: RobustTracker caches team assignments and freezes them after confidence threshold
- **AGGRESSIVE MODE**: Each player needs only **2 consecutive frames** (v2.2+) classified as the same team before permanently frozen
- Once frozen, team assignment **never changes** regardless of subsequent color detection results
- Visual indicators: Green borders and 🔒 emoji show frozen players in output video
- Console feedback: Real-time notifications when teams freeze, statistics every 30 frames
- This eliminates team flickering and ensures consistent player identification throughout the video

**Cached Homography (Performance Critical)**:
- Homography matrix calculated **once** from first frame only
- Reused for entire video (cookie value approach)
- Assumption: Camera angle remains relatively stable throughout footage
- Performance impact: Eliminates ~50-100ms of line detection per frame (5-10x speedup)

**Stadium Masking (Accuracy Critical)**:
- HSV-based green field detection creates binary mask
- YOLO runs on masked frames (background zeroed out)
- Eliminates false positives from crowd, coaches, and sideline objects
- Dramatically improves team classification accuracy by only analyzing field players

**Persistent Tactical Dots**:
- Dots accumulate on a separate layer (not the field template)
- Gradual fade (alpha = 0.98) prevents overcrowding while maintaining movement history
- Field template remains fresh on each frame, only dots fade
- Shows player movement trails and formation changes over time

## Common Development Commands

### Run Analysis
```bash
python tracker.py
```
- Expects `INPUT_VIDEO` (default: `zoomed_game.mp4`) in project directory
- Outputs to `OUTPUT_VIDEO` (default: `output_analysis.mp4`)
- Processing speed: 8-15 FPS on CPU, 25-35 FPS on GPU

### Install Dependencies
```bash
pip install -r requirements.txt
```
- Only 4 dependencies: opencv-python, numpy, ultralytics, scikit-learn
- First run auto-downloads YOLOv8 model (~6MB)

### Configuration Changes
Edit [tracker_config.py](tracker_config.py) to adjust:
- Video input/output paths
- YOLO confidence thresholds
- Team color HSV ranges (most important tuning parameter)
- Stadium masking parameters
- Tracking thresholds
- Tactical map settings

**DO NOT modify [tracker.py](tracker.py) for configuration - always use [tracker_config.py](tracker_config.py)**

## Configuration Guidelines

### Team Color Tuning (Critical for Accuracy)

This is **the most important configuration step**. Team classification accuracy depends entirely on correct HSV ranges:

1. Extract a sample frame with clear jersey visibility
2. Use HSV color picker or OpenCV to analyze jersey colors
3. Update `TEAM_A_HSV_RANGE`, `TEAM_B_HSV_RANGE`, `REFEREE_HSV_RANGE` in [tracker_config.py](tracker_config.py)
4. HSV ranges should be tight enough to distinguish teams but loose enough to handle lighting variations

**Example HSV Analysis**:
```python
import cv2
import numpy as np

frame = cv2.imread('sample_frame.jpg')
jersey_region = frame[y1:y2, x1:x2]
hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
avg_color = np.mean(hsv, axis=(0,1))
print(f"H={avg_color[0]}, S={avg_color[1]}, V={avg_color[2]}")
```

### Stadium Masking Tuning

If players on field are being excluded or crowd is being detected:
- Adjust `FIELD_HSV_LOWER` and `FIELD_HSV_UPPER` for field grass color
- Natural grass vs artificial turf have different HSV values
- Check field coverage percentage in output - should be 25-50% of frame
- Adjust `ROI_TOP_PERCENT` and `ROI_BOTTOM_PERCENT` to exclude scoreboard/crowd areas

### Tracking Parameters

- `MAX_TRACKING_FRAMES = 90`: How long to maintain tracks without detection (higher = more persistent tracking)
- `TRACKING_IOU_THRESHOLD = 0.20`: IoU matching threshold (lower = more lenient matching, more aggressive)
- `TEAM_ASSIGNMENT_CONFIDENCE = 2`: Frames needed before team freeze (v2.2+: reduced from 5 to 2 for faster stabilization)
- `FREEZE_TEAM_ASSIGNMENT = True`: Always keep this enabled to prevent team flickering

### Visual Debug Settings (v2.2+)

- `SHOW_FROZEN_INDICATOR = True`: Shows green border around players with frozen teams in video output
- `PRINT_FREEZE_EVENTS = True`: Console notification when each player's team freezes (🔒)
- `PRINT_TRACKING_STATS = True`: Statistics every 30 frames showing frozen/unfrozen player counts

**Quick Tuning**: If you still see flickering, set `TEAM_ASSIGNMENT_CONFIDENCE = 1` for instant freeze (most aggressive mode)

## Code Structure and Organization

### Function Organization in tracker.py

Functions are organized by pipeline stage with clear separators:

```
PART 0: STADIUM/FIELD RECOGNITION
- create_stadium_mask()
- apply_roi_mask()
- create_combined_mask()

PART 1: STATIC HOMOGRAPHY CALCULATION
- detect_field_lines()
- find_line_intersections()
- calculate_homography()

PART 1.5: SIMPLE OBJECT TRACKING
- class RobustTracker (핵심!)
  - calculate_iou()
  - update()  # Team freezing logic here

PART 2: TEAM CLASSIFICATION
- get_team_color()
- class AdaptiveTeamClassifier
- classify_team_fixed()

PART 3: TOP-DOWN VIEW CREATION
- create_field_template()
- transform_point_to_topdown()

PART 4: MAIN PROCESSING PIPELINE
- process_video()  # Main entry point
```

### Key Classes

**RobustTracker** (lines 361-487):
- Maintains object persistence across frames using IoU-based matching
- Implements team assignment freezing to prevent flickering
- Each track has: `id`, `bbox`, `team_label`, `team_color`, `team_confidence`, `team_frozen`, `age`, `hits`
- Critical for stable team identification

**AdaptiveTeamClassifier** (lines 574-619):
- Alternative to fixed HSV ranges
- Analyzes all players in frame and clusters similar jersey colors
- Useful when team colors are unknown or when dealing with multiple teams

## Performance Characteristics

**Processing Speed** (1920×1080 video):
- CPU (Intel i7): 8-12 FPS → 1000 frames in ~90 seconds
- GPU (RTX 3060): 25-30 FPS → 1000 frames in ~35 seconds
- Apple M3 Max: 15-20 FPS → 1000 frames in ~55 seconds

**Bottlenecks**:
- YOLO detection: 50-60% of processing time
- Visualization: 15-25% of processing time
- Team classification: 5-10% of processing time
- Homography transform: <2% (cached!)

**Optimization Opportunities**:
- Process every Nth frame instead of all frames (reduces accuracy)
- Use smaller YOLO model (yolov8n → yolov8nano)
- Reduce output video resolution
- Skip visualization during processing, render afterward

## Known Limitations and Considerations

1. **Static Homography Assumption**: Accuracy degrades if camera angle changes significantly during video. Homography is calculated once from first frame and reused.

2. **Manual Color Tuning Required**: HSV ranges must be manually configured for each game/team. No automatic calibration.

3. **No Ball Tracking** (by default): System currently focuses on player detection. Ball detection (class 32) is in code but disabled by default. Enable with `ENABLE_BALL_DETECTION = True`.

4. **First Frame Dependency**: System requires clear field lines visible in first frame for accurate homography. If first frame is unsuitable, manually skip to better frame.

5. **No Multi-Camera Support**: Assumes single camera angle throughout video.

6. **Offline Processing Only**: Not optimized for real-time analysis.

## Common Issues and Debugging

**Issue: No players detected or very few detections**
- Solution: Lower `YOLO_CONFIDENCE` threshold (try 0.3)
- Check stadium mask coverage (should be 25-50% of frame)

**Issue: Wrong team classifications or frequent team switches**
- Solution v2.2+: System now freezes after just 2 frames (much faster stabilization)
- If still flickering: Set `TEAM_ASSIGNMENT_CONFIDENCE = 1` for instant freeze
- Verify team color HSV ranges are correct in [tracker_config.py](tracker_config.py)
- Ensure `FREEZE_TEAM_ASSIGNMENT = True` is enabled
- Lower `TRACKING_IOU_THRESHOLD` to 0.15-0.20 for more persistent tracking
- Check console output for 🔒 freeze notifications to verify freezing is working
- Look for green borders in output video confirming frozen teams

**Issue: Players outside field being detected**
- Solution: Adjust `FIELD_HSV_LOWER/UPPER` for better field isolation
- Increase `FIELD_BOUNDARY_EROSION` to exclude field edges
- Adjust `ROI_TOP_PERCENT` and `ROI_BOTTOM_PERCENT`

**Issue: Homography inaccurate (player positions wrong on tactical map)**
- Solution: Ensure first frame has clear yard lines and sidelines visible
- Check for at least 4 line intersection points
- Consider manually selecting a better frame for homography calculation

**Issue: Tactical map dots disappearing (blinking)**
- Solution: Ensure `PERSISTENT_DOTS = True`
- Increase `DOT_FADE_ALPHA` (closer to 1.0 = less fading)

## Model Information

**YOLOv8n (Nano) Specifications**:
- Model size: 6.2 MB (3.2M parameters)
- COCO mAP50-95: 37.2%
- Uses class 0 (person) for player detection
- Trained on 262K person instances from COCO dataset
- Inference speed: 25-40ms per frame on CPU, 5-10ms on GPU

**Detection Accuracy Expectations**:
- Clear players: 90-95% detection rate
- Partially occluded: 70-80% detection rate
- Heavy pileups: 40-60% individual separation
- With stadium masking: 95%+ precision (minimal false positives)

## Future Enhancement Directions

Based on README.md "Future Development" section:

**Short-term**:
- Ball detection and possession tracking (code exists but disabled)
- Enhanced team classification using ML instead of HSV ranges
- Dynamic homography for camera angle changes

**Medium-term**:
- Custom YOLO model trained on football-specific datasets
- Player pose estimation for action detection
- Multiple camera angle support

**Long-term**:
- Advanced analytics (speed, acceleration, heatmaps)
- Event detection (tackles, passes, touchdowns)
- Play segmentation and highlight generation

## Code Modification Guidelines

When modifying this codebase:

1. **Preserve the pipeline structure** - stages flow linearly and depend on each other
2. **Never skip the stadium masking step** - it's critical for accuracy
3. **Respect the cached homography approach** - recalculating per-frame defeats the performance optimization
4. **Maintain team freezing logic** - removing this will cause flickering team assignments
5. **Keep configuration in tracker_config.py** - don't hardcode values in tracker.py
6. **Test with the same video format** - system assumes consistent camera angle throughout

## Version History

- **v2.2 (October 2025)**: Aggressive team freezing improvements - reduced confidence from 5→2 frames, added visual indicators (green borders, 🔒 emoji), tracking improvements (IoU 0.25→0.20, max age 60→90), comprehensive statistics and console feedback
- **v2.1 (October 2025)**: Added team assignment freezing and RobustTracker, separated configuration file
- **v2.0 (October 2025)**: Complete rewrite with YOLOv8, stadium masking, static homography, persistent tactical dots
- **v1.0**: Initial implementation with basic field line detection

## Dependencies

From [requirements.txt](requirements.txt):
```
opencv-python>=4.8.0     # Video I/O, image processing, homography
numpy>=1.24.0            # Array operations
ultralytics>=8.0.0       # YOLOv8 object detection
scikit-learn>=1.3.0      # K-Means clustering for team classification
```
