"""
detect.py - Office desk occupancy detection

This script runs inference on camera images to detect whether desks are
occupied or empty using a trained MobileNetV2 classifier.

Usage:
    python detect.py --image path/to/image.jpg
    python detect.py --folder path/to/folder
    python detect.py --image path/to/image.jpg --angle angle_1
    python detect.py --random 5 --indo              # Test on 5 random images
    python detect.py --random 5 --indo --save      # Save annotated images to test_data/

Workflow:
    1. Run train_classifier.py to train the model
    2. Run this script on new camera images
    3. Results show desk occupancy status and confidence scores
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
from torchvision import transforms, models


IMG_SIZE = (96, 96)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 binary classifier for desk occupancy detection.

    Architecture must match the training model exactly.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super(MobileNetV2Classifier, self).__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights
            self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except ImportError:
            self.model = models.mobilenet_v2(pretrained=pretrained)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_model(model_path):
    """
    Load trained model from checkpoint file.

    Args:
        model_path: Path to .pth checkpoint file

    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint.get('config', {'img_size': IMG_SIZE})

    model = MobileNetV2Classifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully. Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    return model


def load_zones(zones_path='zones.json'):
    """
    Load zone definitions from zones.json.
    
    Args:
        zones_path: Path to zones.json configuration file

    Returns:
        Dictionary containing zone definitions and optional person mappings
    """
    with open(zones_path, 'r') as f:
        return json.load(f)


def get_image_files(directory):
    """
    Get all image files from directory with case-insensitive extension matching.

    Handles .jpg, .JPG, .jpeg, .JPEG, .png, .PNG extensions.
    """
    if not directory.exists():
        return []

    extensions = {'.jpg', '.jpeg', '.png'}
    files = []
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in extensions:
            files.append(item)
    return sorted(files)


def preprocess_image(image, target_size=IMG_SIZE):

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert('RGB')).unsqueeze(0)


def detect_with_best_angle(image_path, model, zones, desks,
                          confidence_threshold=0.5, person_map=None):
    
    img = Image.open(image_path)

    # Seek for the best angle
    best_angle = None
    best_avg_confidence = 0
    all_angle_results = {}

    for angle_name, angle_zones in zones['zones'].items():
        total_confidence = 0
        desk_count = 0
        angle_desks = {}

        for desk in desks:
            if desk not in angle_zones:
                continue

            zone = angle_zones[desk]

            # Crop the desk region
            crop = img.crop((
                zone['x'],
                zone['y'],
                zone['x'] + zone['w'],
                zone['y'] + zone['h']
            ))

            # Preprocess and run inference
            input_tensor = preprocess_image(crop).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                output = output.view(-1)
                probability = torch.sigmoid(output).item()
                is_occupied = probability >= confidence_threshold

            angle_desks[desk] = {
                'occupied': is_occupied,
                'confidence': probability,
                'zone': zone
            }
            total_confidence += probability
            desk_count += 1

        if desk_count > 0:
            avg_confidence = total_confidence / desk_count
            all_angle_results[angle_name] = {
                'desks': angle_desks,
                'avg_confidence': avg_confidence
            }

            if avg_confidence > best_avg_confidence:
                best_avg_confidence = avg_confidence
                best_angle = angle_name

    if best_angle is None:
        print("Warning: No valid angle found")
        return None

    print(f"Auto-detected angle: {best_angle} (avg conf: {best_avg_confidence:.2f})")

    # Build final result
    results = {
        'image': str(image_path),
        'timestamp': datetime.now().isoformat(),
        'angle': best_angle,
        'desks': {}
    }

    # Use person_map from zones.json if available
    if person_map is None:
        person_map = zones.get('person_map', {})

    best_result = all_angle_results[best_angle]
    for desk, data in best_result['desks'].items():
        results['desks'][desk] = {
            'occupied': data['occupied'],
            'confidence': data['confidence'],
            'zone': data['zone'],
            'person': person_map.get(desk, desk)
        }

    return results


def detect_single_image(image_path, model, zones, desks, angle_name=None,
                        confidence_threshold=0.5, person_map=None):
    """
    Detect occupancy in a single image.

    Args:
        image_path: Path to input image
        model: Trained classifier model
        zones: Zone definitions from zones.json
        desks: List of desk names to check
        angle_name: Specific angle to use (if None, auto-detect)
        confidence_threshold: Minimum confidence for occupied classification
        person_map: Optional dict mapping desk names to person names

    Returns:
        Dictionary with detection results
    """
    results = {
        'image': str(image_path),
        'timestamp': datetime.now().isoformat(),
        'angle': angle_name,
        'desks': {}
    }

    img = Image.open(image_path)

    # Get angle zones
    if angle_name:
        if angle_name not in zones['zones']:
            print(f"Warning: Angle '{angle_name}' not found in zones.json")
            return results
        angle_zones = zones['zones'][angle_name]
    else:
        # Auto-detect: find first matching angle
        for angle in zones['zones']:
            if all(desk in zones['zones'][angle] for desk in desks):
                angle_name = angle
                angle_zones = zones['zones'][angle]
                print(f"Auto-detected angle: {angle_name}")
                break
        else:
            print("Warning: No matching angle found for zones")
            return results

    if person_map is None:
        person_map = zones.get('person_map', {})

    for desk in desks:
        if desk not in angle_zones:
            continue

        zone = angle_zones[desk]


        crop = img.crop((
            zone['x'],
            zone['y'],
            zone['x'] + zone['w'],
            zone['y'] + zone['h']
        ))

        # Preprocess and run inference
        input_tensor = preprocess_image(crop).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            # FIXED: Use view(-1) instead of squeeze() for safe handling
            output = output.view(-1)
            probability = torch.sigmoid(output).item()
            is_occupied = probability >= confidence_threshold

        results['desks'][desk] = {
            'occupied': is_occupied,
            'confidence': probability,
            'zone': zone,
            'person': person_map.get(desk, desk)
        }

    return results


def detect_folder(folder_path, model, zones, desks, angle_name=None,
                  confidence_threshold=0.5, output_json=None, person_map=None):
    """
    Detect occupancy for all images in a folder.

    Args:
        folder_path: Path to folder containing images
        model: Trained classifier model
        zones: Zone definitions from zones.json
        desks: List of desk names to check
        angle_name: Specific angle to use (if None, auto-detect)
        confidence_threshold: Minimum confidence for occupied classification
        output_json: Optional path to save results as JSON
        person_map: Optional dict mapping desk names to person names

    Returns:
        List of detection results for all images
    """
    folder = Path(folder_path)
    image_files = get_image_files(folder)

    if not image_files:
        print(f"No images found in {folder_path}")
        return []

    print(f"Processing {len(image_files)} images from {folder_path}")

    all_results = []
    for img_path in tqdm(image_files, desc='Detecting'):
        result = detect_single_image(
            img_path, model, zones, desks, angle_name,
            confidence_threshold, person_map
        )
        all_results.append(result)

    return all_results


def print_results(results, person_map=None, show_hadir=True):
    """
    Print detection results in a formatted table.

    Args:
        results: List of detection results
        person_map: Optional dict mapping desk names to person names
        show_hadir: If True, print HADIR/TIDAK HADIR in Indonesian
    """
    if show_hadir:
        print("\n" + "=" * 60)
        print("HASIL DETEKSI KEHADIRAN")
        print("=" * 60)
    else:
        print("\n" + "=" * 50)
        print("DETECTION RESULT")
        print("=" * 50)

    for result in results:
        img_name = Path(result['image']).name
        print(f"\n{img_name}:")

        for desk, data in result['desks'].items():
            person = person_map.get(desk, desk) if person_map else desk
            conf = data['confidence']

            if show_hadir:
                status = "HADIR" if data['occupied'] else "TIDAK HADIR"
                print(f"  {desk} ({person}) -> {status} ({conf:.2f})")
            else:
                status = "OCCUPIED" if data['occupied'] else "EMPTY"
                print(f"  {desk}: {status} (confidence: {conf:.4f})")


def print_result_table(results):
    """Print results in a formatted table."""
    print("\n" + "-" * 70)
    print(f"{'Image':<25} | {'desk_a':<12} | {'desk_b':<12} | {'desk_c':<12}")
    print("-" * 70)

    for result in results:
        img_name = Path(result['image']).stem[:24]
        desk_status = []
        for desk in ['desk_a', 'desk_b', 'desk_c']:
            if desk in result['desks']:
                occ = result['desks'][desk]['occupied']
                conf = result['desks'][desk]['confidence']
                status = f"{'HADIR' if occ else 'TIDAK'}({conf:.2f})"
                desk_status.append(status)
            else:
                desk_status.append('-')
        print(f"{img_name:<25} | {desk_status[0]:<12} | {desk_status[1]:<12} | {desk_status[2]:<12}")

    print("-" * 70)


def annotate_image(image_path, result, zones, output_path, person_map=None):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return

    if person_map is None:
        person_map = {}

    for desk, data in result['desks'].items():
        zone = data['zone']
        x, y = zone['x'], zone['y']
        w, h = zone['w'], zone['h']

        color = (0, 255, 0) if data['occupied'] else (0, 0, 255)  
        status = "HADIR" if data['occupied'] else "TIDAK HADIR"
        person = person_map.get(desk, desk)
        confidence = data['confidence']

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        label = f"{desk} ({person}) -> {status} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(img, (x, y - label_h - 10), (x + label_w + 10, y), color, -1)
        cv2.putText(img, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)
    cv2.imwrite(str(output_path), img)
    print(f"Annotated image saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Office desk occupancy detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
input format:
  python detect.py --image photo.jpg
  python detect.py --folder dataset/sorted/angle_1
  python detect.py --image photo.jpg --angle angle_1
  python detect.py --image photo.jpg --output result.json
  python detect.py --folder dataset/sorted/angle_1 --threshold 0.7
  python detect.py --image photo.jpg --indo         

Output format:
  desk_a (Asep) -> HADIR (0.85)
  desk_b (Budi) -> TIDAK HADIR (0.54)
  desk_c (Siti) -> HADIR (0.85)
        """
    )
    parser.add_argument('--image', type=str, help='Single image to process')
    parser.add_argument('--folder', type=str, help='Folder of images to process')
    parser.add_argument('--angle', type=str,
                        help='Camera angle (e.g., angle_1). Auto-detected if not specified.')
    parser.add_argument('--model', default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--zones', default='zones.json',
                        help='Path to zones configuration')
    parser.add_argument('--output', type=str,
                        help='Output JSON file for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Occupancy confidence threshold (0-1)')
    parser.add_argument('--indo', action='store_true',
                        help='Show output in Indonesian (HADIR/TIDAK HADIR)')
    parser.add_argument('--random', type=int, default=0,
                        help='Pick N random images from dataset/raw (default: 0 = disabled, use 5 for 5 random images)')
    parser.add_argument('--test', action='store_true',
                        help='Pick 5 random images from dataset/raw for testing')
    parser.add_argument('--save', action='store_true',
                        help='Save annotated images to output directory')
    parser.add_argument('--output-dir', type=str, default='test_data',
                        help='Directory to save annotated images (default: test_data)')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using: python train_classifier.py")
        return

    # Load model and zones FIRST (before any processing)
    model = load_model(args.model)
    zones = load_zones(args.zones)
    desks = zones['desks']
    person_map = zones.get('person_map', {})
    print(f"Using device: {DEVICE}")

    # Handle random image selection from dataset/raw
    if args.random > 0:
        import random
        raw_dir = Path('dataset/raw')
        image_files = get_image_files(raw_dir)
        if not image_files:
            print(f"Error: No images found in {raw_dir}")
            return
        num_images = min(args.random, len(image_files))
        random_images = random.sample(image_files, num_images)
        print(f"Random {num_images} images selected from dataset/raw:")
        for img in random_images:
            print(f"  - {img.name}")

        # Setup output directory
        output_dir = None
        if args.save:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nAnnotated images will be saved to: {output_dir}/")

        # Process each random image
        all_results = []

        # Use best angle detection unless specific angle is provided
        use_best_angle = (args.angle is None)

        for img_path in random_images:
            if use_best_angle:
                # Try all angles and pick the best one
                result = detect_with_best_angle(
                    str(img_path), model, zones, desks,
                    args.threshold, person_map
                )
            else:
                # Use specified angle
                result = detect_single_image(
                    str(img_path), model, zones, desks, args.angle,
                    args.threshold, person_map
                )

            if result and result['desks']:
                print_results([result], person_map, show_hadir=args.indo)

                # Save annotated image if --save flag is set
                if output_dir:
                    annotated_path = output_dir / img_path.name
                    annotate_image(img_path, result, zones, annotated_path, person_map)

                all_results.append(result)
            print()

        # Save JSON results
        json_path = args.output if args.output else output_dir / 'results.json' if output_dir else None
        if json_path:
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {json_path}")
        return

    # Legacy --test flag (same as --random 5)
    if args.test:
        import random
        raw_dir = Path('dataset/raw')
        image_files = get_image_files(raw_dir)
        if not image_files:
            print(f"Error: No images found in {raw_dir}")
            return
        num_images = min(5, len(image_files))
        random_images = random.sample(image_files, num_images)
        print(f"Testing on {num_images} random images from dataset/raw:")
        for img in random_images:
            print(f"  - {img.name}")

        # Setup output directory
        output_dir = None
        if args.save:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nAnnotated images will be saved to: {output_dir}/")

        # Process each random image
        all_results = []
        use_best_angle = (args.angle is None)

        for img_path in random_images:
            if use_best_angle:
                result = detect_with_best_angle(
                    str(img_path), model, zones, desks,
                    args.threshold, person_map
                )
            else:
                result = detect_single_image(
                    str(img_path), model, zones, desks, args.angle,
                    args.threshold, person_map
                )

            if result and result['desks']:
                print_results([result], person_map, show_hadir=args.indo)

                # Save annotated image if --save flag is set
                if output_dir:
                    annotated_path = output_dir / img_path.name
                    annotate_image(img_path, result, zones, annotated_path, person_map)

                all_results.append(result)
            print()

        # Save JSON results
        json_path = args.output if args.output else output_dir / 'results.json' if output_dir else None
        if json_path:
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {json_path}")
        return

    if not args.image and not args.folder:
        parser.error('Either --image or --folder must be provided')

    if args.image:
        result = detect_single_image(
            args.image, model, zones, desks, args.angle,
            args.threshold, person_map
        )

        if result['desks']:
            print_results([result], person_map, show_hadir=args.indo)
        else:
            print("No detection results. Check zones.json configuration.")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {args.output}")

    elif args.folder:
         results = detect_folder(
        args.folder, model, zones, desks, args.angle,
        args.threshold, args.output, person_map
    )

    if results:
        print_results(results, person_map, show_hadir=args.indo)
        print_result_table(results)

        if args.save:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for result in results:
                image_path = result['image']
                img_name = Path(image_path).name
                annotated_path = output_dir / img_name

                annotate_image(
                    image_path,
                    result,
                    zones,
                    annotated_path,
                    person_map
                )
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
