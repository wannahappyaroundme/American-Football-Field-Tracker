# Football Field Tracker - Dynamic Homography System

A real-time computer vision system for analyzing zoomed-in and panning American football footage, creating a consistent top-down tactical map through dynamic homography calculation.

## Overview

This system processes football game footage where the camera zooms and pans, making it difficult to maintain spatial awareness. By detecting yard lines and hash marks in each frame and computing a new homography matrix dynamically, we can transform the perspective to a stable top-down view.

## Current Implementation Status

### ✅ Part 1: Main Video Processing Loop
- Video loading with property detection
- Frame-by-frame processing loop
- Display and output video capabilities
- User controls (pause/quit)

### ✅ Part 2: Field Segmentation (HSV Color Masking)
- **HSV Color Space Conversion**: Convert frame to HSV for robust color detection
- **Green Field Isolation**: Use cv2.inRange to create binary mask of playing field
- **Morphological Cleanup**: Remove noise and fill gaps in mask
- **Mask Validation**: Ensure field coverage meets minimum threshold
- **Background Elimination**: Zero out crowd, stadium, and other non-field elements

### ✅ Part 3: Automated Line Detection and Filtering (on Masked Field)
- **Image Preprocessing**: Grayscale conversion, Gaussian blur, Canny edge detection
- **Line Detection**: Hough Line Transform (Probabilistic) for robust line segment detection
- **Line Classification**: Automatic classification into horizontal (yard lines) and vertical (sidelines/hash marks) based on angle
- **Line Merging**: Intelligent grouping of parallel, nearby lines to reduce noise
- **Visualization**: Real-time display of detected lines with color coding

### 🚧 Coming Next
- Part 4: Dynamic reference point identification and line-to-yard mapping
- Part 5: Per-frame homography calculation and perspective transformation
- Part 6: Side-by-side visualization of original and transformed views

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Place your football game video in the project directory as `zoomed_game.mp4`
2. Run the script:

```bash
python football_field_tracker.py
```

3. Controls:
   - Press `q` to quit
   - Press `p` to pause/resume

### Saving Output Video

To save the processed video, edit `football_field_tracker.py`:

```python
OUTPUT_PATH = "output.mp4"  # Change from None
```

## Technical Details

### Two-Stage Detection Pipeline

#### Stage 1: Field Segmentation (HSV Masking)

1. **HSV Conversion**
   - Convert BGR to HSV color space for lighting-invariant color detection
   - HSV separates color (Hue) from brightness (Value)

2. **Green Field Isolation**
   - Use cv2.inRange with tunable HSV range (default: H=35-85, S=40-255, V=40-255)
   - Creates binary mask: white = field, black = background

3. **Morphological Cleanup**
   - Closing operation: fills holes in field mask
   - Opening operation: removes small noise specks
   - Kernel size and iterations configurable

4. **Validation**
   - Check field coverage percentage
   - Warn if below minimum threshold (default: 20%)

#### Stage 2: Line Detection (on Masked Field)

1. **Preprocessing**
   - Apply mask to zero out non-field areas
   - Grayscale conversion for simplified processing
   - Gaussian blur (5×5 kernel, σ=1.5) to reduce noise
   - Canny edge detection (thresholds: 50-150) for high-contrast edges

2. **Hough Line Detection**
   - Probabilistic Hough Transform for line segment detection
   - Parameters optimized for field lines:
     - Distance resolution: 1 pixel
     - Angular resolution: 1 degree
     - Minimum line length: 50 pixels
     - Maximum gap: 10 pixels

3. **Line Classification**
   - **Horizontal lines** (yard lines): angle < 15°
   - **Vertical lines** (sidelines/hash marks): angle > 75°
   - Angle calculated using arctangent of slope

4. **Line Merging**
   - Groups parallel lines (angle difference < 5°)
   - Merges nearby lines (distance < 20-30 pixels)
   - Averages positions for robust representation

### Code Structure

```
football_field_tracker.py
├── Part 1: Video Processing
│   ├── load_video()
│   └── process_video()
│
├── Part 2: Field Segmentation
│   ├── create_field_mask()
│   ├── apply_mask_to_frame()
│   └── visualize_mask_overlay()
│
├── Part 3: Line Detection (on Masked Field)
│   ├── preprocess_frame()
│   ├── detect_lines_hough()
│   ├── calculate_line_angle()
│   ├── calculate_line_length()
│   ├── classify_lines()
│   ├── merge_similar_lines()
│   ├── filter_and_classify_lines()
│   └── visualize_detected_lines()
│
├── Part 4: Reference Point Identification (Coming)
├── Part 5: Homography Calculation (Coming)
└── Part 6: Final Visualization (Coming)
```

## Parameters You Can Tune

All parameters are centralized in `config.py` for easy tuning.

### HSV Field Masking (Most Important!)
```python
# Adjust these first if field is not detected properly
HSV_LOWER_GREEN = (35, 40, 40)      # (Hue, Saturation, Value)
HSV_UPPER_GREEN = (85, 255, 255)    # Widen range if field not detected

# Morphological cleanup
MORPH_KERNEL_SIZE = 15              # Larger = more aggressive cleanup
MORPH_CLOSING_ITERATIONS = 2        # Fill holes in mask
MORPH_OPENING_ITERATIONS = 1        # Remove noise specks
```

See `HSV_TUNING_GUIDE.md` for detailed HSV parameter tuning instructions.

### Canny Edge Detection
```python
CANNY_THRESHOLD_LOW = 50       # Lower = more edges
CANNY_THRESHOLD_HIGH = 150     # Adjust based on contrast
```

### Hough Line Detection
```python
HOUGH_THRESHOLD = 50           # Lower = more lines detected
HOUGH_MIN_LINE_LENGTH = 50     # Minimum line length in pixels
HOUGH_MAX_LINE_GAP = 10        # Maximum gap to link segments
```

### Line Classification
```python
HORIZONTAL_ANGLE_THRESHOLD = 15.0  # Max angle for horizontal
VERTICAL_ANGLE_THRESHOLD = 75.0    # Min angle for vertical
```

### Line Merging
```python
PARALLEL_ANGLE_THRESHOLD = 5.0           # Max angle difference
HORIZONTAL_DISTANCE_THRESHOLD = 20.0     # Max distance (horizontal)
VERTICAL_DISTANCE_THRESHOLD = 30.0       # Max distance (vertical)
```

## Visualization

The output window shows a 2×2 grid with all processing stages:

**Top-Left (Panel 1):** Original frame with detected lines
  - Green lines: Horizontal (yard lines)
  - Blue lines: Vertical (sidelines/hash marks)
  - Circles: Line midpoints

**Top-Right (Panel 2):** Field mask (HSV segmentation)
  - White: Detected field area
  - Black: Background (crowd, stadium, etc.)
  - Use this to tune HSV parameters

**Bottom-Left (Panel 3):** Masked field only
  - Shows what the line detector "sees"
  - Everything except field is zeroed out

**Bottom-Right (Panel 4):** Edge detection output
  - Shows Canny edges on masked field
  - Useful for debugging line detection

## Next Steps

The system is designed modularly to add:
1. **OCR/Template Matching**: Identify which yard line is which
2. **Virtual Corner Calculation**: Compute line intersections (even outside frame)
3. **Homography Matrix**: Calculate perspective transform per frame
4. **Top-Down View**: Apply transformation to create stable tactical map

## Performance Notes

- Processes ~30 FPS on modern hardware
- Can be optimized by processing every Nth frame
- GPU acceleration possible with OpenCV CUDA support

## License

MIT License

## Author

Senior Computer Vision Engineer specializing in real-time sports analytics


