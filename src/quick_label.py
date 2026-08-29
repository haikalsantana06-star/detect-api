"""
quick_label.py - ROI (Region of Interest) labeling tool for desk zones

This script allows users to manually define desk bounding boxes on reference
images using OpenCV rectangle selection. The coordinates are saved to zones.json.

Usage:
    python quick_label.py --angle angle_1
    python quick_label.py --angle angle_2
    python quick_label.py --angle angle_3

Workflow:
    1. User manually groups images into dataset/sorted/angle_1 ... angle_6
    2. User runs: python quick_label.py --angle angle_1
    3. Program opens a reference image for that angle
    4. User draws exactly 3 bounding boxes in order:
       - First rectangle = desk_a
       - Second rectangle = desk_b
       - Third rectangle = desk_c
       - Fourth rectangle = desk_d
    5. Coordinates are saved into zones.json
    6. User runs: python auto_crop.py --angle angle_1
    7. Crops are saved to dataset/crops/{desk}/to_sort/
    8. User manually moves images to empty/ or occupied/
    9. User runs train_classifier.py
    10. User runs detect.py
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np


DESK_NAMES = ['desk_a', 'desk_b', 'desk_c', 'desk_d']


class ROILabeler:
    """
    Interactive ROI labeling tool using OpenCV.

    Allows user to draw bounding boxes on a reference image to define
    desk regions. Each desk is labeled in order (desk_a, desk_b, desk_c).
    """

    def __init__(self, image_path, angle_name, zones_path='zones.json'):
        """
        Initialize the ROI labeler.

        Args:
            image_path: Path to reference image
            angle_name: Name of the angle being labeled (e.g., 'angle_1')
            zones_path: Path to zones.json configuration file
        """
        self.image_path = Path(image_path)
        self.angle_name = angle_name
        self.zones_path = Path(zones_path)

        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.clone = self.image.copy()
        self.height, self.width = self.image.shape[:2]

        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.rectangles = [] 

        self.window_name = f'ROI Labeling - {angle_name} (Press ESC to cancel)'

        self.zones = self._load_zones()

    def _load_zones(self):
        """Load existing zones from zones.json if it exists."""
        if self.zones_path.exists():
            with open(self.zones_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'desks': DESK_NAMES.copy(),
                'zones': {},
                'classes': ['empty', 'occupied'],
                'input_size': [96, 96]
            }

    def _save_zones(self):
        """Save zones to zones.json."""
        with open(self.zones_path, 'w') as f:
            json.dump(self.zones, f, indent=2)
        print(f"Zones saved to: {self.zones_path}")

    def _mouse_callback(self, event, x, y, flags, param):
        """
        OpenCV mouse callback for drawing rectangles.

        Left mouse button: Start drawing / continue drawing
        Right mouse button: Undo last rectangle
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.is_drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.image = self.clone.copy()
                self._draw_all_rectangles()
                cv2.rectangle(
                    self.image,
                    self.start_point,
                    (x, y),
                    (0, 255, 0),
                    2
                )

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.end_point = (x, y)

            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

         
            rect = (x1, y1, x2 - x1, y2 - y1)
            self.rectangles.append(rect)

            
            self.image = self.clone.copy()
            self._draw_all_rectangles()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.rectangles:
                self.rectangles.pop()
                self.image = self.clone.copy()
                self._draw_all_rectangles()
                print(f"Undid last rectangle. {len(self.rectangles)} rectangles remaining.")

    def _draw_all_rectangles(self):
        """Draw all completed rectangles and current preview."""
        colors = [
            (0, 255, 0),    # hijau untuk desk a
            (255, 0, 0),    # Biru untuk desk b
            (0, 165, 255),  # Oren untuk desk c
            (243, 232, 0),  # teuing warna naon jang deck d
        ] 

        for i, rect in enumerate(self.rectangles):
            x, y, w, h = rect
            color = colors[i % len(colors)]
            cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)

            label = DESK_NAMES[i] if i < len(DESK_NAMES) else f'rect_{i}'
            cv2.putText(
                self.image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    def _draw_instructions(self):
        """Draw instructions on the image."""
        instructions = [
            f"Draw rectangles for: {', '.join(DESK_NAMES)}",
            "Left-click & drag: Draw rectangle",
            "Right-click: Undo last rectangle",
            "Press ENTER when done",
            "Press ESC to cancel",
            f"Progress: {len(self.rectangles)}/{len(DESK_NAMES)}"
        ]

        y_offset = 30
        for i, text in enumerate(instructions):
            y = y_offset + i * 25
            cv2.putText(
                self.image,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    def run(self):
        """
        Run the interactive ROI labeling session.

        Returns:
            True if labeling completed successfully, False if cancelled
        """
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print(f"\n{'='*60}")
        print(f"ROI Labeling: {self.angle_name}")
        print(f"Image: {self.image_path.name} ({self.width}x{self.height})")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("  1. Draw exactly 3 rectangles in order:")
        print("     - First rectangle = desk_a")
        print("     - Second rectangle = desk_b")
        print("     - Third rectangle = desk_c")
        print("     - Fourth rectangle = desk_d")
        print("  2. Left-click and drag to draw")
        print("  3. Right-click to undo last rectangle")
        print("  4. Press ENTER when done")
        print("  5. Press ESC to cancel")
        print()

        while True:
            display_image = self.image.copy()
            self._draw_instructions()

            cv2.imshow(self.window_name, display_image)

            # Wait for key
            key = cv2.waitKey(1) & 0xFF

            # Check if we have enough rectangles
            if len(self.rectangles) >= len(DESK_NAMES):
                print(f"\nAll {len(DESK_NAMES)} rectangles drawn!")
                print("Press ENTER to save, ESC to cancel and restart")

            # ESC key
            if key == 27:
                print("Cancelled by user.")
                cv2.destroyAllWindows()
                return False

            # ENTER key
            elif key == 13:
                if len(self.rectangles) >= len(DESK_NAMES):
                    break
                else:
                    print(f"Please draw at least {len(DESK_NAMES)} rectangles first.")

        # Save the zones
        self._save_zones_to_json()

        cv2.destroyAllWindows()
        return True

    def _save_zones_to_json(self):
        """Save the drawn rectangles to zones.json."""
        if self.angle_name not in self.zones['zones']:
            self.zones['zones'][self.angle_name] = {}

        # Save each rectangle as a zone
        for i, rect in enumerate(self.rectangles):
            if i < len(DESK_NAMES):
                desk_name = DESK_NAMES[i]
                x, y, w, h = rect
                self.zones['zones'][self.angle_name][desk_name] = {
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                }
                print(f"  {desk_name}: x={x}, y={y}, w={w}, h={h}")

        if 'desks' not in self.zones:
            self.zones['desks'] = DESK_NAMES.copy()

        #save filess
        self._save_zones()

        print(f"\nSuccessfully saved zones for {self.angle_name}!")


def get_reference_image(angle_name, sorted_dir='dataset/sorted'):
    """
    Find a reference image for the given angle.

    Args:
        angle_name: Name of the angle (e.g., 'angle_1')
        sorted_dir: Directory containing angle subfolders

    Returns:
        Path to the first image found in the angle folder
    """
    angle_path = Path(sorted_dir) / angle_name

    if not angle_path.exists():
        raise ValueError(f"Angle folder not found: {angle_path}")

    # Case-insensitive image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    for ext in image_extensions:
        images = list(angle_path.glob(f'*{ext}'))
        if images:
            return images[0]

    raise ValueError(f"No images found in {angle_path}")


def main():
    parser = argparse.ArgumentParser(
        description='ROI labeling tool for desk zones',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_label.py --angle angle_1
  python quick_label.py --angle angle_2
  python quick_label.py --angle angle_3

Workflow:
  1. python quick_label.py --angle angle_1
  2. Draw 3 rectangles on the reference image
  3. Coordinates saved to zones.json
  4. python auto_crop.py --angle angle_1
  5. Label crops manually (move to empty/ or occupied/)
  6. python train_classifier.py
  7. python detect.py
        """
    )
    parser.add_argument(
        '--angle',
        type=str,
        required=True,
        help='Angle to label (e.g., angle_1, angle_2, ...)'
    )
    parser.add_argument(
        '--sorted-dir',
        default='dataset/sorted',
        help='Directory containing angle subfolders'
    )
    parser.add_argument(
        '--zones',
        default='zones.json',
        help='Path to zones.json configuration file'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Use specific image instead of first image in angle folder'
    )

    args = parser.parse_args()

    if not args.angle.startswith('angle_'):
        print(f"Error: Angle name must start with 'angle_' (e.g., angle_1)")
        return

    try:
        # Get reference image
        if args.image:
            image_path = Path(args.image)
        else:
            image_path = get_reference_image(args.angle, args.sorted_dir)

        print(f"Using reference image: {image_path}")

        # Run labeler
        labeler = ROILabeler(image_path, args.angle, args.zones)
        success = labeler.run()

        if success:
            print("\nNext step: python auto_crop.py --angle", args.angle)
        else:
            print("\nLabeling cancelled.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
