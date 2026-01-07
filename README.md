# RoadRunner
Digital Image Processing Project - Line Detection, Car Detection, and Sign Detection

## Overview
RoadRunner is a computer vision project that implements three core detection capabilities using **classical computer vision methods** (non-machine learning approaches):

1. **Line Detection** - Using Hough Transform
2. **Car Detection** - Using Haar Cascade Classifiers
3. **Sign Detection** - Using Color-based Segmentation and Shape Analysis

All implementations use OpenCV and traditional image processing techniques without deep learning.

## Features

### 1. Line Detection
- **Method**: Probabilistic Hough Transform
- **Capabilities**:
  - General line detection in images
  - Road lane detection with Region of Interest (ROI)
  - Configurable parameters for different scenarios
- **Use Cases**: Lane keeping systems, road marking detection

### 2. Car Detection
- **Method**: Haar Cascade Classifier
- **Capabilities**:
  - Detect vehicles in images
  - Multi-scale detection
  - Adjustable sensitivity parameters
- **Use Cases**: Traffic monitoring, vehicle counting

### 3. Sign Detection
- **Method**: Color-based Segmentation + Shape Analysis
- **Capabilities**:
  - Detect red signs (stop signs, yield signs)
  - Detect blue signs (information signs)
  - Detect yellow signs (warning signs)
  - Shape classification (circle, triangle, rectangle)
- **Use Cases**: Traffic sign recognition, road safety systems

## Dataset

This project uses the **Road/Lane Detection Evaluation 2013** dataset, which is derived from the KITTI Vision Benchmark Suite.

- **Source**: [Kaggle - Road/Lane Detection Evaluation 2013](https://www.kaggle.com/datasets/tryingit/roadlane-detection-evaluation-2013)
- **Original Source**: [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)

The dataset contains various road scenarios suitable for testing line detection and object detection algorithms.

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/ebylmz/road-runner.git
cd road-runner

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
Run the main application with test images:
```bash
python scripts/detect.py path/to/your/image.jpg
```

### KITTI Dataset Demo
Run detection on KITTI dataset sequences:
```bash
python scripts/detect_kitti.py path/to/kitti/image.png
```

The KITTI demo displays:
- **Lines**: Shown as green lines (lane markings)
- **Cars**: Shown with blue bounding boxes
- **Signs**: Shown with colored bounding boxes (red/blue/yellow)

Results are saved to `kitti_results/` directory with separate outputs for each detection type and a combined visualization.

### Using Individual Detectors

#### Line Detection
```python
from src.detectors import LineDetector
import cv2

# Load image
image = cv2.imread('road.jpg')

# Create detector
detector = LineDetector(threshold=50, min_line_length=30)

# Detect lines
lines = detector.detect_lines(image)

# Detect lane lines with ROI
result, lane_lines = detector.detect_lane_lines(image)

# Save result
cv2.imwrite('output.jpg', result)
```

#### Car Detection
```python
from src.detectors import CarDetector
import cv2

# Load image
image = cv2.imread('traffic.jpg')

# Create detector
detector = CarDetector(scale_factor=1.1, min_neighbors=3)

# Detect cars
result, cars = detector.detect_and_draw(image)

print(f"Detected {len(cars)} cars")
cv2.imwrite('cars_detected.jpg', result)
```

#### Sign Detection
```python
from src.detectors import SignDetector
import cv2

# Load image
image = cv2.imread('road_signs.jpg')

# Create detector
detector = SignDetector(min_area=500)

# Detect all signs
signs = detector.detect_all_signs(image)

# Draw detections
result = detector.draw_detections(image, signs)

print(f"Red signs: {len(signs['red'])}")
print(f"Blue signs: {len(signs['blue'])}")
print(f"Yellow signs: {len(signs['yellow'])}")

cv2.imwrite('signs_detected.jpg', result)
```

## Project Structure
```
RoadRunner/
├── scripts/                         # Detection scripts
│   ├── detect.py                    # Main detections script
│   └── detect_kitti.py              # KITTI dataset demo script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── src/
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── line_detector.py        # Line detection implementation
│   │   ├── car_detector.py         # Car detection implementation
│   │   └── sign_detector.py        # Sign detection implementation
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py          # Image processing utilities
├── results/                         # Output directory for results
├── kitti_results/                   # KITTI demo output directory
└── models/                          # Cascade classifier models
```

## Technical Details

### Line Detection Algorithm
1. Convert image to grayscale
2. Apply Gaussian blur to reduce noise
3. Perform Canny edge detection
4. Apply Probabilistic Hough Transform
5. Filter and group detected lines

**Parameters**:
- `rho`: Distance resolution (default: 1)
- `theta`: Angle resolution (default: π/180)
- `threshold`: Minimum votes (default: 50)
- `min_line_length`: Minimum line length (default: 50)
- `max_line_gap`: Maximum gap between segments (default: 10)

### Car Detection Algorithm
1. Convert image to grayscale
2. Apply histogram equalization
3. Run Haar Cascade classifier at multiple scales
4. Apply non-maximum suppression
5. Return bounding boxes

**Parameters**:
- `scale_factor`: Scale multiplier (default: 1.1)
- `min_neighbors`: Minimum neighbors (default: 3)
- `min_size`: Minimum detection size (default: 50x50)

### Sign Detection Algorithm
1. Convert image to HSV color space
2. Apply color thresholding for red/blue/yellow
3. Perform morphological operations to clean mask
4. Find contours in binary mask
5. Filter by area and shape characteristics
6. Classify shape (circle, triangle, rectangle)

**Parameters**:
- `min_area`: Minimum sign area (default: 500)
- `max_area`: Maximum sign area (default: 50000)
- `min_circularity`: Circle threshold (default: 0.7)

## Output

The application generates detection results in the `results/` directory:
- `line_detection.png` - Detected lines
- `lane_detection.png` - Lane lines with ROI
- `car_detection.png` - Detected cars (if applicable)
- `sign_detection.png` - Detected traffic signs

## Limitations

- **Line Detection**: Works best with clear, well-defined lines; may struggle with curved roads
- **Car Detection**: Haar cascades have lower accuracy than modern deep learning methods; sensitive to viewing angle
- **Sign Detection**: Color-based detection is affected by lighting conditions; requires distinct colors