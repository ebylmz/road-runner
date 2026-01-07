"""
KITTI Dataset Detection Demo
Demonstrates Line, Car, and Sign Detection on KITTI Dataset sequences
"""

import cv2
import sys
import os
import urllib.request
import zipfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detectors import LineDetector, CarDetector, SignDetector
from utils.image_utils import save_image



def process_kitti_sequence(image_path, output_dir='kitti_results'):
    """
    Process a KITTI dataset image with all detectors

    Args:
        image_path: Path to KITTI image
        output_dir: Output directory for results
    """
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Create a combined result image
    combined_result = image.copy()

    # 1. LINE DETECTION
    try:
        line_detector = LineDetector(
            threshold=50,
            min_line_length=50,
            max_line_gap=30
        )

        # Detect lane lines with ROI
        _, lane_lines = line_detector.detect_lane_lines(image)

        # Draw lines as lines (not bounding boxes)
        if len(lane_lines) > 0:
            for line in lane_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(combined_result, (x1, y1), (x2, y2), (0, 255, 0), 3)

    except Exception as e:
        print(f"      Line detection error: {e}")

    # 2. CAR DETECTION
    try:
        car_detector = CarDetector(
            scale_factor=1.05,
            min_neighbors=2,
            min_size=(30, 30)
        )

        cars = car_detector.detect_cars(image)

        # Draw cars with bounding boxes
        if len(cars) > 0:
            for (x, y, w, h) in cars:
                cv2.rectangle(combined_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(combined_result, 'Car', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    except Exception as e:
        print(f"      Car detection error: {e}")

    # 3. SIGN DETECTION
    try:
        sign_detector = SignDetector(
            min_area=300,
            max_area=30000
        )

        signs = sign_detector.detect_all_signs(image)

        # Draw signs with bounding boxes
        for sign in signs['red']:
            x, y, w, h = sign['bbox']
            cv2.rectangle(combined_result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(combined_result, f'Red {sign["shape"]}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for sign in signs['blue']:
            x, y, w, h = sign['bbox']
            cv2.rectangle(combined_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(combined_result, f'Blue {sign["shape"]}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for sign in signs['yellow']:
            x, y, w, h = sign['bbox']
            cv2.rectangle(combined_result, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(combined_result, f'Yellow {sign["shape"]}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    except Exception as e:
        print(f"      Sign detection error: {e}")

    # Save combined result
    combined_path = os.path.join(output_dir, f"{Path(image_path).stem}_detections.png")
    save_image(combined_result, combined_path, verbose=False)


def process_kitti_directory(dataset_path, output_dir='kitti_results'):
    """
    Process all images in a KITTI dataset directory with all detectors

    Args:
        dataset_path: Path to KITTI dataset directory
        output_dir: Output directory for results
    """
    # Check if dataset path exists
    if not os.path.isdir(dataset_path):
        print(f"Error: Provided path '{dataset_path}' is not a directory or does not exist.")
        sys.exit(1)

    # Get list of image files
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Iterate over all images in the directory with a progress bar
    for image_name in tqdm(image_files, desc="Processing KITTI images"):
        image_path = os.path.join(dataset_path, image_name)

        # Process the image
        process_kitti_sequence(image_path, output_dir=output_dir)


def main():
    """Main entry point for KITTI dataset demo"""

    # Check if user provided KITTI dataset path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print(f"Using provided KITTI dataset directory: {dataset_path}")
    else:
        print("No KITTI dataset directory provided")
        print("Please provide the path to a KITTI dataset directory as a command-line argument.")
        sys.exit(1)

    # Process the dataset directory
    process_kitti_directory(dataset_path)


if __name__ == "__main__":
    main()
