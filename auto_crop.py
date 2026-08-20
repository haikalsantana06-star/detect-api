"""
auto_crop.py - Crop desk regions from sorted angle images

This script crops desk regions from camera images based on zone definitions
in zones.json. Each angle folder is processed independently.

Usage:
    python auto_crop.py --angle angle_1           # Process specific angle
    python auto_crop.py --angle angle_1 --force    # Overwrite existing crops
    python auto_crop.py --all                      # Process all angles

Workflow:
    1. User manually groups images into dataset/sorted/angle_1 ... angle_6
    2. User runs: python quick_label.py --angle angle_1
    3. User draws desk bounding boxes on reference image
    4. Coordinates saved to zones.json
    5. User runs: python auto_crop.py --angle angle_1
    6. Crops are saved to dataset/crops/{desk}/to_sort/
    7. User manually moves images to empty/ or occupied/
    8. User runs train_classifier.py
    9. User runs detect.py
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def load_zones(zones_path='zones.json'):
    """
    Load zone definitions from zones.json.

    Args:
        zones_path: Path to zones.json configuration file

    Returns:
        Dictionary containing zone definitions
    """
    with open(zones_path, 'r') as f:
        return json.load(f)


def get_image_files(directory):
    """
    Get all image files from directory with case-insensitive extension matching.

    Handles .jpg, .JPG, .jpeg, .JPEG, .png, .PNG extensions.

    Args:
        directory: Path to directory to scan

    Returns:
        Sorted list of Path objects for image files
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []

    if not directory.exists():
        return []

    for item in directory.iterdir():
        if item.is_file():
            ext = item.suffix.lower()
            if ext in image_extensions:
                image_files.append(item)

    return sorted(image_files)


def crop_images(sorted_dir='dataset/sorted', output_dir='dataset/crops',
                zones_path='zones.json', force=False, angle_filter=None):
    """
    Crop desk regions from sorted angle images.

    Args:
        sorted_dir: Directory containing angle_* subfolders with raw images
        output_dir: Base directory for cropped output
        zones_path: Path to zones.json configuration file
        force: Overwrite existing crops if True
        angle_filter: Only process this angle if specified (e.g., 'angle_1')
    """
    zones = load_zones(zones_path)
    sorted_path = Path(sorted_dir)
    output_path = Path(output_dir)

    # Validate sorted_dir exists
    if not sorted_path.exists():
        print(f"Error: Sorted directory not found: {sorted_dir}")
        print(f"Please ensure images are organized in: {sorted_dir}/angle_1/, {sorted_dir}/angle_2/, etc.")
        return

    desks = zones.get('desks', ['desk_a', 'desk_b', 'desk_c'])

    for desk in desks:
        (output_path / desk / 'to_sort').mkdir(parents=True, exist_ok=True)

    #per-angle
    angle_folders = sorted([
        d for d in sorted_path.iterdir()
        if d.is_dir() and d.name.startswith('angle_')
    ])

    if not angle_folders:
        print(f"Warning: No angle folders found in {sorted_dir}")
        print(f"Expected folders named: angle_1, angle_2, angle_3, ...")
        return

    total_crops = 0
    total_images = 0
    skipped_existing = 0
    errors = 0

    for angle_folder in angle_folders:
        angle_name = angle_folder.name

        if angle_filter and angle_name != angle_filter:
            continue

        if angle_name not in zones['zones']:
            print(f"Warning: No zone definitions for {angle_name} in zones.json")
            print(f"  Run: python quick_label.py --angle {angle_name}")
            continue

        angle_zones = zones['zones'][angle_name]

        # Get images with case-insensitive extension matching
        image_files = get_image_files(angle_folder)

        if not image_files:
            print(f"Warning: No images found in {angle_name}")
            continue

        print(f"\n{'='*50}")
        print(f"Processing: {angle_name} ({len(image_files)} images)")
        print(f"{'='*50}")

        angle_crops = 0

        for img_path in tqdm(image_files, desc=f"Cropping {angle_name}"):
            total_images += 1
            try:
                img = Image.open(img_path)
                img_filename = img_path.stem

                for desk_name in desks:
                    if desk_name not in angle_zones:
                        continue

                    zone = angle_zones[desk_name]
                    # Save to to_sort directory
                    crop_path = output_path / desk_name / 'to_sort' / f"{img_filename}_{desk_name}.jpg"

                    # Skip if crop exists and not force
                    if crop_path.exists() and not force:
                        skipped_existing += 1
                        continue

                    # Crop the region
                    crop = img.crop((
                        zone['x'],
                        zone['y'],
                        zone['x'] + zone['w'],
                        zone['y'] + zone['h']
                    ))

                    # Resize if target size specified
                    target_size = zones.get('input_size')
                    if target_size:
                        crop = crop.resize((target_size[0], target_size[1]), Image.LANCZOS)

                    # Save crop
                    crop.save(crop_path, quality=95)
                    total_crops += 1
                    angle_crops += 1

            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                errors += 1

        print(f"  -> {angle_crops} crops generated for {angle_name}")

    # Summary
    print(f"\n{'='*50}")
    print("CROPPING COMPLETE")
    print(f"{'='*50}")
    print(f"Total images processed: {total_images}")
    print(f"Total crops generated: {total_crops}")
    print(f"Skipped (existing): {skipped_existing}")
    print(f"Errors: {errors}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext step: Manually label crops in:")
    for desk in desks:
        print(f"  - dataset/crops/{desk}/to_sort/")
    print(f"\nMove images to empty/ or occupied/ directories, then run:")
    print(f"  python train_classifier.py")


def main():
    parser = argparse.ArgumentParser(
        description='Crop desk regions from sorted angle images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_crop.py --angle angle_1          # Process only angle_1
  python auto_crop.py --all                    # Process all angles
  python auto_crop.py --force                   # Overwrite existing crops
  python auto_crop.py --sorted-dir dataset/sorted --output-dir dataset/crops

Workflow:
  1. python quick_label.py --angle angle_1  (define zones)
  2. python auto_crop.py --angle angle_1    (crop images)
  3. Manually label crops (move to empty/ or occupied/)
  4. python train_classifier.py
  5. python detect.py
        """
    )
    parser.add_argument('--angle', type=str,
                        help='Process only this angle (e.g., angle_1)')
    parser.add_argument('--all', action='store_true',
                        help='Process all angles (default behavior)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing crops')
    parser.add_argument('--sorted-dir', default='dataset/sorted',
                        help='Input sorted images directory')
    parser.add_argument('--output-dir', default='dataset/crops',
                        help='Output crops directory')
    parser.add_argument('--zones', default='zones.json',
                        help='Zones configuration file')

    args = parser.parse_args()

    crop_images(
        sorted_dir=args.sorted_dir,
        output_dir=args.output_dir,
        zones_path=args.zones,
        force=args.force,
        angle_filter=args.angle
    )


if __name__ == '__main__':
    main()
