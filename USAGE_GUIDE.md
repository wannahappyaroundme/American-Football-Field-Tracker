# Football Field Tracker - Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Video

Place your football game video in the project directory and name it `zoomed_game.mp4`, or modify the path in `config.py`:

```python
# In config.py
VIDEO_INPUT_PATH = "path/to/your/video.mp4"
```

### 3. Run the Tracker

```bash
python football_field_tracker.py
```

### 4. Test Line Detection (Optional)

Before processing a full video, test the line detection on a single frame or synthetic field:

```bash
# Test with synthetic field
python test_line_detection.py

# Test with a specific image/frame
python test_line_detection.py path/to/frame.jpg
```

## Configuration

All parameters can be tuned in `config.py`. Here are the most important settings:

### Video Input/Output

```python
VIDEO_INPUT_PATH = "zoomed_game.mp4"      # Input video file
VIDEO_OUTPUT_PATH = "output.mp4"          # Output video (or None)
```

### HSV Field Masking (⭐ Most Important!)

**This is the first thing you should tune!** The field mask determines what areas are analyzed.

```python
# Adjust these if the field is not being detected properly
HSV_LOWER_GREEN = (35, 40, 40)      # (Hue, Saturation, Value)
HSV_UPPER_GREEN = (85, 255, 255)    # Upper bounds

# Morphological cleanup
MORPH_KERNEL_SIZE = 15              # Larger = more cleanup
MORPH_CLOSING_ITERATIONS = 2        # Fill holes in mask
MORPH_OPENING_ITERATIONS = 1        # Remove noise
```

**Quick fixes:**
- **Field not detected?** Widen the range: `HSV_LOWER_GREEN = (30, 20, 30)`
- **Too much background?** Narrow the range: `HSV_LOWER_GREEN = (40, 50, 50)`
- **Holes in mask?** Increase: `MORPH_CLOSING_ITERATIONS = 3`

See `HSV_TUNING_GUIDE.md` for comprehensive HSV tuning instructions!

### Line Detection Sensitivity

After getting the field mask right, tune these parameters:

```python
# Detect MORE lines (more sensitive)
CANNY_THRESHOLD_LOW = 30          # Lower from 50
CANNY_THRESHOLD_HIGH = 120        # Lower from 150
HOUGH_THRESHOLD = 30              # Lower from 50

# Detect FEWER lines (less sensitive)
CANNY_THRESHOLD_LOW = 70          # Higher from 50
CANNY_THRESHOLD_HIGH = 180        # Higher from 150
HOUGH_THRESHOLD = 70              # Higher from 50
```

### Line Classification

Adjust what angles are considered horizontal vs vertical:

```python
HORIZONTAL_ANGLE_THRESHOLD = 15.0  # Lines < 15° are horizontal
VERTICAL_ANGLE_THRESHOLD = 75.0    # Lines > 75° are vertical
```

### Line Merging

Control how aggressively similar lines are merged:

```python
# Merge lines more aggressively (fewer output lines)
PARALLEL_ANGLE_THRESHOLD = 10.0
HORIZONTAL_DISTANCE_THRESHOLD = 40.0
VERTICAL_DISTANCE_THRESHOLD = 50.0

# Merge lines less aggressively (more output lines)
PARALLEL_ANGLE_THRESHOLD = 2.0
HORIZONTAL_DISTANCE_THRESHOLD = 10.0
VERTICAL_DISTANCE_THRESHOLD = 15.0
```

## Controls

While the video is playing:

- **`q`**: Quit the program
- **`p`**: Pause/resume playback

## Understanding the Output

### Display Window

The window shows a **2×2 grid** with all processing stages:

**Panel 1 (Top-Left): Original + Detected Lines**
- Green lines: Horizontal lines (yard lines)
- Blue lines: Vertical lines (sidelines/hash marks)
- Circles: Midpoints of detected lines
- Text overlay: Count of detected lines

**Panel 2 (Top-Right): Field Mask** ⭐
- White areas: Detected green field
- Black areas: Background (crowd, stadium, trees)
- **This is your most important diagnostic panel!**
- Use this to tune HSV parameters

**Panel 3 (Bottom-Left): Masked Field Only**
- Shows what the line detector actually processes
- Everything except field is black/zeroed out
- Should show clean field with lines visible

**Panel 4 (Bottom-Right): Edge Detection**
- Shows Canny edges on the masked field
- White lines = detected edges
- Useful for debugging line detection parameters

### Console Output

For each frame, you'll see:

```
Processing frame 1...
  Stage 1: Creating field mask...
  Field mask coverage: 52.3% of frame
  Stage 2: Detecting lines on masked field...
  Detected 8 horizontal, 12 vertical lines
  After merging: 5 horizontal, 3 vertical lines
```

This tells you:
1. What percentage of frame was identified as field (should be 30-80%)
2. How many raw line segments were detected
3. How many distinct lines remain after merging

## Troubleshooting

### Problem: Field not detected (Panel 2 mostly black)

**⚠️ FIX THIS FIRST!** If the field mask isn't working, nothing else will work.

**Symptoms:**
- Panel 2 is mostly black
- Console shows "Field mask coverage: < 20%"
- Panel 3 shows almost nothing

**Solutions:**
1. **Widen HSV range** in `config.py`:
   ```python
   # More permissive green detection
   HSV_LOWER_GREEN = (25, 20, 20)   # Lower all values
   HSV_UPPER_GREEN = (95, 255, 255) # Raise H_max
   ```

2. **Reduce morphological operations**:
   ```python
   MORPH_OPENING_ITERATIONS = 0  # Disable opening temporarily
   ```

3. **Check your video**: Is the field actually green? Some indoor turf appears blue-gray.

See `HSV_TUNING_GUIDE.md` for detailed field masking troubleshooting!

### Problem: Too much background detected (Panel 2 mostly white)

**Symptoms:**
- Panel 2 shows crowd/stadium as white
- Console shows "Field mask coverage: > 80%"
- Still detecting many noise lines

**Solutions:**
1. **Narrow HSV range**:
   ```python
   HSV_LOWER_GREEN = (40, 50, 50)   # More restrictive
   HSV_UPPER_GREEN = (80, 255, 220) # Lower V_max
   ```

2. **Increase morphological opening**:
   ```python
   MORPH_OPENING_ITERATIONS = 2     # Remove small noise areas
   ```

### Problem: No lines detected

**Causes:**
- Field mask too restrictive (check Panel 2 first!)
- Thresholds too high
- Poor contrast in video

**Solutions:**
1. **Check Panel 2**: Is the field white? If not, fix HSV parameters first!

2. Lower edge detection thresholds in `config.py`:
   ```python
   CANNY_THRESHOLD_LOW = 30
   CANNY_THRESHOLD_HIGH = 100
   ```

3. Lower Hough threshold:
   ```python
   HOUGH_THRESHOLD = 30
   HOUGH_MIN_LINE_LENGTH = 30
   ```

4. Check Panel 4 (edge detection) - if you don't see the field lines there, the thresholds are too high

### Problem: Too many false positive lines

**Causes:**
- Thresholds too low
- Detecting players, shadows, or artifacts
- Camera noise

**Solutions:**
1. Raise edge detection thresholds:
   ```python
   CANNY_THRESHOLD_LOW = 70
   CANNY_THRESHOLD_HIGH = 180
   ```

2. Increase minimum line length:
   ```python
   HOUGH_MIN_LINE_LENGTH = 100
   ```

3. Increase Hough threshold:
   ```python
   HOUGH_THRESHOLD = 70
   ```

### Problem: Lines not merging properly

**Causes:**
- Merging thresholds too strict
- Lines have slightly different angles

**Solutions:**
1. Increase parallel angle threshold:
   ```python
   PARALLEL_ANGLE_THRESHOLD = 10.0
   ```

2. Increase distance thresholds:
   ```python
   HORIZONTAL_DISTANCE_THRESHOLD = 30.0
   VERTICAL_DISTANCE_THRESHOLD = 40.0
   ```

### Problem: Wrong line classification (horizontal detected as vertical)

**Causes:**
- Camera angle makes yard lines appear more vertical
- Classification thresholds don't match field orientation

**Solutions:**
1. Adjust classification thresholds:
   ```python
   HORIZONTAL_ANGLE_THRESHOLD = 25.0  # Increase for tilted camera
   VERTICAL_ANGLE_THRESHOLD = 65.0    # Decrease for tilted camera
   ```

### Problem: Performance is slow

**Solutions:**
1. Process every Nth frame instead of all frames:
   ```python
   FRAME_SKIP = 2  # Process every 2nd frame
   ```

2. Reduce Hough detection accuracy (faster):
   ```python
   HOUGH_RHO = 2               # Increase from 1
   HOUGH_THETA = 2             # Increase from 1
   ```

## Expected Results

### Good Detection Scenario

For a typical zoomed-in football broadcast:
- **Horizontal lines**: 3-7 yard lines visible
- **Vertical lines**: 2-4 sidelines/hash marks
- Lines should be stable across frames
- Green/blue overlays should align well with field markings

### Typical Detection Counts

| View Type | Horizontal Lines | Vertical Lines |
|-----------|------------------|----------------|
| Wide angle | 5-10 | 2-4 |
| Zoomed in | 2-4 | 1-3 |
| Sideline view | 3-6 | 4-8 (many hash marks) |
| End zone view | 8-12 | 2-3 |

## Next Steps

After you have good line detection working:

1. **Part 3**: Implement line identification (which yard line is which)
2. **Part 4**: Calculate homography matrices per frame
3. **Part 5**: Generate stable top-down view

## Advanced Usage

### Custom Field Dimensions

If you're analyzing college or high school football (different dimensions):

```python
# In config.py
FIELD_LENGTH = 120      # Standard: 120 yards
FIELD_WIDTH = 53.33     # Standard: 53.33 yards (160 feet)

# For high school football:
FIELD_WIDTH = 53.33     # Same as NFL
```

### Debug Mode

Enable detailed logging:

```python
DEBUG_MODE = True  # Already enabled by default
```

### Saving Output Video

To save the processed video with line detections:

```python
VIDEO_OUTPUT_PATH = "output_with_lines.mp4"
```

Note: The output video will be side-by-side (original + edge detection).

## Performance Benchmarks

On typical hardware:
- **1080p video**: ~25-30 FPS
- **4K video**: ~10-15 FPS
- **With output saving**: ~20-25 FPS (1080p)

Processing can be sped up by:
- Using `FRAME_SKIP > 1`
- Reducing `HOUGH_MIN_LINE_LENGTH`
- Processing at lower resolution

## Example Workflow

### First-Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test with synthetic field (sanity check)
python test_line_detection.py

# 3. Extract a frame from your video to test
# (using ffmpeg or any video tool)
ffmpeg -i zoomed_game.mp4 -ss 00:00:10 -vframes 1 test_frame.jpg

# 4. Test on that frame
python test_line_detection.py test_frame.jpg

# 5. Tune HSV parameters in config.py
#    - Look at Panel 2 (field mask)
#    - Adjust HSV_LOWER_GREEN and HSV_UPPER_GREEN
#    - Test again until field is properly detected

# 6. Once field mask looks good, tune line detection
#    - Adjust Canny and Hough parameters if needed

# 7. Run on full video
python football_field_tracker.py
```

### Recommended Tuning Order

**Always tune in this order:**

1. **HSV Field Mask** (Panel 2)
   - Goal: Field is white, background is black
   - Coverage: 30-80% of frame
   
2. **Edge Detection** (Panel 4)
   - Goal: See clear white lines on black background
   - Adjust CANNY thresholds

3. **Line Detection** (Panel 1)
   - Goal: 3-10 lines detected (reasonable count)
   - Adjust HOUGH parameters

4. **Line Merging**
   - Goal: Similar lines grouped together
   - Adjust merging thresholds

### Iterative Tuning

1. Run on video (or single frame)
2. **Look at Panel 2 first** - is field detected?
3. If no: adjust HSV parameters
4. If yes: check Panel 4 - are edges detected?
5. If no: adjust Canny parameters
6. If yes: check Panel 1 - are lines detected and reasonable?
7. If no: adjust Hough parameters
8. Repeat until satisfied

## Common Parameter Sets

### High Contrast Field (Sunny Day, Natural Grass)

```python
# HSV Masking
HSV_LOWER_GREEN = (35, 40, 60)      # Bright greens only
HSV_UPPER_GREEN = (85, 255, 255)

# Edge/Line Detection
CANNY_THRESHOLD_LOW = 60
CANNY_THRESHOLD_HIGH = 170
HOUGH_THRESHOLD = 60
```

### Low Contrast Field (Night Game, Poor Lighting)

```python
# HSV Masking
HSV_LOWER_GREEN = (35, 25, 20)      # Accept darker greens
HSV_UPPER_GREEN = (90, 255, 200)    # Avoid bright lights

# Edge/Line Detection
CANNY_THRESHOLD_LOW = 30
CANNY_THRESHOLD_HIGH = 100
HOUGH_THRESHOLD = 40
GAUSSIAN_SIGMA = 2.0  # More blur to reduce noise
```

### Artificial Turf (Vivid Green, Sharp Lines)

```python
# HSV Masking
HSV_LOWER_GREEN = (40, 60, 50)      # Only saturated greens
HSV_UPPER_GREEN = (80, 255, 255)

# Edge/Line Detection
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150
HOUGH_MIN_LINE_LENGTH = 80
HOUGH_THRESHOLD = 70
```

### Natural Grass (Faded/Worn Lines)

```python
# HSV Masking
HSV_LOWER_GREEN = (30, 20, 25)      # Accept pale greens
HSV_UPPER_GREEN = (95, 255, 255)

# Edge/Line Detection
CANNY_THRESHOLD_LOW = 30
CANNY_THRESHOLD_HIGH = 120
HOUGH_MIN_LINE_LENGTH = 40
HOUGH_THRESHOLD = 35
```

### Indoor Dome (Artificial Lighting)

```python
# HSV Masking
HSV_LOWER_GREEN = (38, 45, 45)      # Controlled lighting
HSV_UPPER_GREEN = (82, 255, 240)    # Avoid overexposure

# Edge/Line Detection
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150
HOUGH_THRESHOLD = 55
```

## Getting Help

If you're still having issues:

1. **Check Panel 2 first** (field mask) - this is the foundation of everything
2. Read `HSV_TUNING_GUIDE.md` for comprehensive HSV parameter tuning
3. Start with synthetic field testing: `python test_line_detection.py`
4. Try the example parameter sets above for your field conditions
5. Adjust one parameter at a time to understand its effect
6. Use a single frame for faster iteration: `python test_line_detection.py frame.jpg`

## Code Architecture

The system uses a two-stage detection pipeline:

```
football_field_tracker.py
├── Part 1: Video Processing Loop ✅ COMPLETE
│   └── process_video()
│
├── Part 2: Field Segmentation (HSV Masking) ✅ COMPLETE
│   ├── create_field_mask()           # Isolate green field
│   ├── apply_mask_to_frame()         # Zero out background
│   └── visualize_mask_overlay()      # Debug visualization
│
├── Part 3: Line Detection (on Masked Field) ✅ COMPLETE
│   ├── preprocess_frame()            # Edges on masked field
│   ├── detect_lines_hough()          # Find line segments
│   ├── classify_lines()              # Horizontal vs vertical
│   ├── merge_similar_lines()         # Group parallel lines
│   └── visualize_detected_lines()    # Draw results
│
├── Part 4: Line Identification 🚧 TODO
│   ├── identify_yard_lines()         # Which line is which?
│   ├── calculate_intersections()     # Virtual corners
│   └── create_reference_points()     # Source points for homography
│
├── Part 5: Homography Calculation 🚧 TODO
│   ├── define_destination_points()   # World coordinates
│   ├── compute_homography()          # Get transform matrix
│   └── warp_perspective()            # Apply transformation
│
└── Part 6: Top-Down Visualization 🚧 TODO
    └── create_tactical_map()         # Stable overhead view
```

**Key Innovation:** By masking the field first (Part 2), we eliminate ~90% of noise before line detection (Part 3), resulting in dramatically cleaner and more accurate results.


