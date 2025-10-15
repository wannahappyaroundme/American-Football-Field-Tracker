# Quick Start Guide - Football Tracker

## Installation (3 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# This installs:
# - opencv-python (video processing)
# - ultralytics (YOLOv8)
# - filterpy (Kalman filter)
# - scipy (Hungarian algorithm)
# - numpy (arrays)
```

## Run the Tracker (< 1 minute)

```bash
# Simply run:
python tracker.py
```

**First run**: YOLOv8 model (~6MB) will download automatically.

## What You'll See

A window with **4 panels**:

```
┌─────────────────┬─────────────────┐
│ 1. ROI Mask     │ 2. Masked Frame │
│ (what's analyzed)│ (YOLO input)   │
├─────────────────┼─────────────────┤
│ 3. Tracked      │ 4. Info         │
│ (IDs + boxes)   │ (stats)         │
└─────────────────┴─────────────────┘
```

**Panel 3** is the most important: Shows each player with their unique **Track ID**.

## Configuration (Optional)

Edit `tracker_config.py`:

### Basic Settings

```python
# Input/Output
VIDEO_INPUT_PATH = "your_video.mp4"
VIDEO_OUTPUT_PATH = "tracked_output.mp4"  # or None

# Detection sensitivity
YOLO_CONF_THRESHOLD = 0.5  # Lower = more detections

# ROI (Region of Interest)
ROI_TOP_PERCENT = 0.2      # Exclude top 20%
ROI_BOTTOM_PERCENT = 0.1   # Exclude bottom 10%
```

### Advanced Tracking

```python
# Track persistence (occlusion handling)
SORT_MAX_AGE = 1        # Increase to 5-10 for better occlusion handling

# Track confirmation
SORT_MIN_HITS = 3       # Decrease to 1-2 for faster tracking

# Matching sensitivity
SORT_IOU_THRESHOLD = 0.3  # Lower = more lenient matching
```

## Common Issues & Quick Fixes

### Issue: No Players Detected

**Fix**: Lower confidence threshold
```python
YOLO_CONF_THRESHOLD = 0.3  # In tracker_config.py
```

### Issue: IDs Keep Switching

**Fix**: Increase track survival
```python
SORT_MAX_AGE = 5
SORT_IOU_THRESHOLD = 0.2
```

### Issue: Too Slow (< 5 FPS)

**Fix**: Use smallest model
```python
YOLO_MODEL_PATH = "yolov8n.pt"  # Already default
```

Or process every other frame:
```python
FRAME_SKIP = 2
```

### Issue: ROI Not Covering Field

**Fix**: Adjust masking percentages
```python
ROI_TOP_PERCENT = 0.15    # Less exclusion
ROI_BOTTOM_PERCENT = 0.05
```

Look at **Panel 1** to see the ROI coverage.

## Understanding Track IDs

- Each player gets a **unique ID** (e.g., ID: 5)
- **Same player = same ID** across frames (ideally)
- IDs may switch if player is heavily occluded
- Increase `SORT_MAX_AGE` to reduce ID switches

## Keyboard Controls

- **`q`**: Quit
- **`p`**: Pause/resume

## Output

If `VIDEO_OUTPUT_PATH` is set, saves a video with the 2×2 grid showing all processing stages.

## Performance Expectations

| Hardware | Expected FPS | Quality |
|----------|-------------|---------|
| Laptop CPU | 5-8 FPS | Acceptable |
| Desktop CPU (i7) | 8-12 FPS | Good |
| GPU (RTX 3060) | 30+ FPS | Excellent |

## Next Steps

1. **Read full documentation**: `TRACKER_README.md`
2. **Tune parameters**: `tracker_config.py`
3. **Analyze results**: Observe Panel 3 for tracking quality

## Workflow

```
1. Run tracker.py with default settings
   ↓
2. Look at Panel 3 - are players tracked?
   ↓
3. If yes: Great! Adjust ROI if needed
   ↓
4. If no: Lower YOLO_CONF_THRESHOLD
   ↓
5. If IDs switch: Increase SORT_MAX_AGE
   ↓
6. Iterate until satisfied
```

## Example Configuration Sets

### Conservative (Fewer false positives)
```python
YOLO_CONF_THRESHOLD = 0.6
SORT_MIN_HITS = 5
SORT_MAX_AGE = 1
```

### Aggressive (Catch more players)
```python
YOLO_CONF_THRESHOLD = 0.3
SORT_MIN_HITS = 2
SORT_MAX_AGE = 10
```

### Balanced (Default)
```python
YOLO_CONF_THRESHOLD = 0.5
SORT_MIN_HITS = 3
SORT_MAX_AGE = 1
```

## Troubleshooting Commands

```bash
# Check if ultralytics installed
python -c "import ultralytics; print('OK')"

# Check if YOLO model exists
ls ~/.cache/ultralytics/  # Linux/Mac
dir %USERPROFILE%\.cache\ultralytics\  # Windows

# Manually download model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Test on single image
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')('test.jpg')"
```

## Getting Help

- **Full documentation**: `TRACKER_README.md`
- **Configuration reference**: `tracker_config.py` (has inline comments)
- **Algorithm details**: `sort.py` (SORT implementation with comments)

## What's Different from Old System?

This new system:
- ✅ Tracks **players**, not field lines
- ✅ Assigns **unique IDs** to each player
- ✅ Uses **YOLO** (state-of-the-art detection)
- ✅ Uses **SORT** (proven tracking algorithm)
- ✅ **ROI masking** instead of HSV color masking
- ✅ Focus on **player/ball tracking**, not homography

## Success Criteria

You're ready to proceed when:
- ✅ Players have bounding boxes (Panel 3)
- ✅ Each player has a visible Track ID
- ✅ IDs stay consistent for at least 30-60 frames
- ✅ Processing FPS is acceptable for your use case

---

**Need more detail?** See `TRACKER_README.md`  
**Ready to customize?** Edit `tracker_config.py`  
**Want to understand the code?** Read `tracker.py` and `sort.py`

**Have fun tracking! 🏈**

