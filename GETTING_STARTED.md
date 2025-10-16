# Getting Started - Football Video Analysis Tool

## ✨ Your Clean, Simple System is Ready!

The project has been **completely restructured** into a clean, single-script system focused on accuracy and detailed analysis.

## 📁 Project Structure (Clean!)

```
camera_tracking/
├── tracker.py           ⭐ Main analysis script (ALL functionality)
├── requirements.txt     📦 Dependencies (4 packages)
├── README.md           📚 Complete documentation
├── zoomed_game.mp4     🎬 Your input video
└── yolov8n.pt          🤖 YOLO model (auto-downloaded)
```

**That's it!** Just 3 essential files (plus your video).

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Installs only 4 packages:
- `opencv-python` - Video processing
- `numpy` - Arrays
- `ultralytics` - YOLOv8
- `scikit-learn` - K-Means clustering

### Step 2: Configure Team Colors

**⚠️ IMPORTANT**: Before running, adjust team colors in `tracker.py`:

```python
# Edit these lines (around line 30):
TEAM_A_HSV_RANGE = ((90, 50, 50), (130, 255, 255))    # Blue jerseys
TEAM_B_HSV_RANGE = ((0, 0, 180), (180, 30, 255))      # White jerseys
REFEREE_HSV_RANGE = ((0, 0, 0), (180, 255, 60))       # Black jerseys
```

**How to find HSV values**:
- Use an online HSV color picker
- Or watch the first few frames and estimate
- H (Hue): 0-180, S (Saturation): 0-255, V (Value): 0-255

### Step 3: Run Analysis

```bash
python tracker.py
```

**Processing:**
- Analyzes first frame for field geometry
- Processes every frame (slow but accurate)
- Saves output to `output_analysis.mp4`

**Expected time**: ~1-2 minutes for 1000 frames

## 📺 Output

`output_analysis.mp4` contains:

```
┌──────────────────────┬──────────────┐
│  ORIGINAL FOOTAGE    │ TACTICAL MAP │
│                      │              │
│  • Team-colored      │  • Top-down  │
│    bounding boxes    │    field view│
│  • Team labels       │  • Player    │
│  • Confidence scores │    positions │
└──────────────────────┴──────────────┘
```

## 🎯 What the System Does

### 1. Static Homography (First Frame)
- Detects yard lines and sidelines
- Calculates transformation matrix
- Uses for entire video (accurate if camera stable)

### 2. Player Detection (Every Frame)
- YOLOv8 detects all persons
- Handles overlapping players
- Returns bounding boxes

### 3. Team Classification (Every Player)
- Extracts torso region
- Finds dominant color (K-Means)
- Matches to Team A/B/Referee ranges

### 4. Tactical Visualization (Every Frame)
- Transforms player positions to top-down view
- Draws on tactical map
- Combines with original frame
- Saves to output video

## ⚙️ Configuration Options

Edit constants at top of `tracker.py`:

```python
# Video files
INPUT_VIDEO = "zoomed_game.mp4"      # Change to your video
OUTPUT_VIDEO = "output_analysis.mp4"  # Change output name

# Detection sensitivity
YOLO_CONFIDENCE = 0.5  # Lower = more detections (0.3-0.7)

# Top-down view size
FIELD_WIDTH = 400      # Pixels
FIELD_HEIGHT = 600     # Pixels
```

## 🔧 Troubleshooting

### No Players Detected

**Solution**: Lower confidence threshold
```python
YOLO_CONFIDENCE = 0.3  # In tracker.py
```

### Wrong Team Colors

**Solution**: Adjust HSV ranges
```python
# Extract a jersey sample, convert to HSV, use those values
TEAM_A_HSV_RANGE = ((your_h_min, your_s_min, your_v_min), 
                    (your_h_max, your_s_max, your_v_max))
```

### Homography Inaccurate

**Solution**: 
- Ensure first frame has clear yard lines
- Check that at least 4 line intersections are visible
- May need to manually skip to a better frame

### Slow Processing

**This is normal!** The system prioritizes accuracy over speed.
- Expected: 8-15 FPS processing
- For 1000 frames: ~60-120 seconds

## 📖 Full Documentation

See `README.md` for:
- Complete technical details
- Algorithm explanations
- Accuracy expectations
- Future improvements

## ✅ Success Checklist

Your system is working when:
- ✅ `output_analysis.mp4` is created
- ✅ Players have colored bounding boxes
- ✅ Team labels are mostly correct
- ✅ Tactical map shows player positions
- ✅ Side-by-side view works

## 🎯 Next Steps

1. **Run it**: `python tracker.py`
2. **Review output**: Open `output_analysis.mp4`
3. **Tune colors**: Adjust HSV ranges for better team classification
4. **Iterate**: Re-run until team colors are accurate

## 💡 Pro Tips

1. **Team Color Tuning**: Spend time getting HSV ranges right - this is critical
2. **First Frame Selection**: If first frame lacks field lines, modify script to use a different frame
3. **Quality Input**: Higher quality video = better detection and classification
4. **Lighting**: Consistent lighting helps team classification accuracy

---

## TL;DR - Just Do This

```bash
# 1. Install
pip install -r requirements.txt

# 2. Edit team colors in tracker.py (lines 30-38)

# 3. Run
python tracker.py

# 4. Watch output
# Open: output_analysis.mp4
```

**Your clean, consolidated football analysis system is ready!** 🏈

See `README.md` for complete documentation.

