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
import re
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
        Loaded model in evaluation mode, checkpoint dict
    """
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = MobileNetV2Classifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully. Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    return model, checkpoint


def load_all_angle_models(models_dir='models'):
    """
    Load all available angle-specific models.

    Looks for models/best_model_{angle}.pth and models/best_model.pth (fallback).
    Returns dict: {angle_name: (model, checkpoint)}

    Returns:
        {'angle_1': (model, checkpoint), 'angle_2': (model, checkpoint), 'default': ...}
    """
    models_dir = Path(models_dir)
    angle_models = {}

    # Load angle-specific models
    for model_file in models_dir.glob('best_model_angle_*.pth'):
        angle_name = model_file.stem.replace('best_model_', '')
        model, checkpoint = load_model(str(model_file))
        angle_models[angle_name] = (model, checkpoint)
        print(f"  Loaded {angle_name} model: {model_file.name}")

    # Load default fallback model
    default_path = models_dir / 'best_model.pth'
    if default_path.exists():
        model, checkpoint = load_model(str(default_path))
        angle_models['default'] = (model, checkpoint)
        print(f"  Loaded default model: best_model.pth")
    elif angle_models:
        # Use first angle model as fallback
        first = next(iter(angle_models.values()))
        angle_models['default'] = first
        print(f"  No default model found, using first available as fallback")

    return angle_models


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
    if angle_name is None:
        # Try to infer angle from image filename
        inferred = infer_angle_from_path(image_path)
        if inferred and inferred in zones['zones']:
            angle_name = inferred
            angle_zones = zones['zones'][angle_name]
            print(f"Inferred angle '{angle_name}' from image path")
        else:
            # Auto-detect angle using SSIM reference matching
            detected = detect_angle_by_ssim(image_path, zones)
            if detected:
                angle_name = detected
                angle_zones = zones['zones'][angle_name]
                print(f"Auto-detected angle: {angle_name} (by SSIM matching)")
            else:
                # Last resort: first angle in zones.json
                first_angle = next(iter(zones['zones'].keys()))
                angle_name = first_angle
                angle_zones = zones['zones'][first_angle]
                print(f"Warning: Could not auto-detect angle, using fallback: {angle_name}")
    elif angle_name not in zones['zones']:
        print(f"Warning: Angle '{angle_name}' not found in zones.json")
        return results
    else:
        angle_zones = zones['zones'][angle_name]

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


def infer_angle_from_path(folder_path):
    """Infer angle name from folder path (e.g. dataset/sorted/angle_2 -> angle_2)."""
    match = re.search(r'(angle_\d+)', str(folder_path))
    return match.group(1) if match else None


# ─── Angle Detection via SSIM Reference Matching ────────────────────────────────

try:
    from skimage.metrics import structural_similarity as ssim_sim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False


def build_ssim_references(sorted_dir='dataset/sorted'):
    """
    Build average SSIM reference crops for each angle from labeled images.

    Returns dict: {angle_name: {desk_name: avg_grayscale_crop}}
    """
    import numpy as np

    sorted_path = Path(sorted_dir)
    if not sorted_path.exists():
        return {}

    refs = {}
    for angle_folder in sorted_path.iterdir():
        if not angle_folder.is_dir():
            continue
        angle_match = re.match(r'(angle_\d+)', angle_folder.name)
        if not angle_match:
            continue
        angle_name = angle_match.group(1)

        imgs = [f for f in angle_folder.iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not imgs:
            continue

        desks = ['desk_a', 'desk_b', 'desk_c', 'desk_d']
        # Load zones for this angle
        try:
            with open('zones.json') as f:
                all_zones = json.load(f)['zones']
            angle_zones = all_zones.get(angle_name, {})
        except (FileNotFoundError, KeyError):
            continue

        desk_crops = {d: [] for d in desks}
        for img_path in imgs:
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                continue
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            for d in desks:
                if d not in angle_zones:
                    continue
                z = angle_zones[d]
                crop = gray[z['y']:z['y']+z['h'], z['x']:z['x']+z['w']]
                if crop.size > 0:
                    desk_crops[d].append(crop)

        if all(desk_crops[d] for d in desks):
            refs[angle_name] = {}
            for d in desks:
                stack = np.stack(desk_crops[d])
                refs[angle_name][d] = np.mean(stack, axis=0)

    return refs


# Cached references (built once on first use)
_ssim_refs_cache = None


def _get_ssim_refs():
    global _ssim_refs_cache
    if _ssim_refs_cache is None:
        _ssim_refs_cache = build_ssim_references()
    return _ssim_refs_cache


def detect_angle_by_ssim(image_path, zones):
    """
    Auto-detect which angle an image belongs to using SSIM reference matching.

    Strategy:
    - Build average reference crops for each angle from labeled images
    - For the test image, crop with each angle's zones
    - Compare each crop to its angle's reference via SSIM
    - Pick the angle whose total SSIM across all desks is highest

    Works for slight camera tilt/rotation and lighting variations.

    Returns:
        angle_name (str) of the best matching angle, or None if uncertain.
    """
    if not _HAS_SSIM:
        return None

    refs = _get_ssim_refs()
    if not refs:
        return None

    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        return None

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    desks = ['desk_a', 'desk_b', 'desk_c', 'desk_d']
    all_zones = zones.get('zones', zones)

    angle_scores = {}
    for angle_name, ref_desks in refs.items():
        if angle_name not in all_zones:
            continue
        score = 0.0
        valid = 0
        for d in desks:
            if d not in ref_desks or d not in all_zones[angle_name]:
                continue
            z = all_zones[angle_name][d]
            crop = gray[z['y']:z['y']+z['h'], z['x']:z['x']+z['w']]
            if crop.size == 0:
                continue
            try:
                s = ssim_sim(crop, ref_desks[d], data_range=255)
                score += s
                valid += 1
            except Exception:
                continue
        if valid > 0:
            angle_scores[angle_name] = score / valid

    if not angle_scores:
        return None

    best_angle = max(angle_scores, key=angle_scores.get)
    best_score = angle_scores[best_angle]

    # If all scores are suspiciously low, something is wrong
    if best_score < 0.1:
        return None

    return best_angle


def detect_angle_by_features(image_path, zones, desks):
    """
    Auto-detect which angle an image belongs to using ORB feature matching.

    Strategy: extract ORB keypoints from the full image, count how many fall
    inside each angle's desk zone regions. The angle whose zones contain the
    most keypoints is the correct one (those zones actually contain desks).

    Works for slight camera tilt/rotation since we compare feature density,
    not exact template matching.

    Returns:
        angle_name (str) of the best matching angle, or None if uncertain.
    """
    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        return None

    # Grayscale for ORB
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints on full image
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, edgeThreshold=31)
    kp = orb.detect(gray, None)

    angle_scores = {}

    for angle_name, angle_zones in zones['zones'].items():
        score = 0

        for desk in desks:
            if desk not in angle_zones:
                continue
            zone = angle_zones[desk]
            x, y, w, h = zone['x'], zone['y'], zone['w'], zone['h']

            # Count keypoints inside this zone
            for point in kp:
                px, py = point.pt
                if x <= px < x + w and y <= py < y + h:
                    score += 1

        angle_scores[angle_name] = score

    if not angle_scores:
        return None

    # Pick angle with highest keypoint density in its desk zones
    best_angle = max(angle_scores, key=angle_scores.get)
    best_score = angle_scores[best_angle]

    # If the best score is suspiciously low (no features at all), be conservative
    if best_score < 3:
        return None

    return best_angle


def detect_folder(folder_path, model, zones, desks, angle_name=None,
                  confidence_threshold=0.5, output_json=None, person_map=None,
                  angle_models=None):
    """
    Detect occupancy for all images in a folder.

    Args:
        folder_path: Path to folder containing images
        model: Trained classifier model (used when angle_name is explicitly set)
        zones: Zone definitions from zones.json
        desks: List of desk names to check
        angle_name: Specific angle to use (if None, auto-detect per image)
        confidence_threshold: Minimum confidence for occupied classification
        output_json: Optional path to save results as JSON
        person_map: Optional dict mapping desk names to person names
        angle_models: Optional dict {angle_name: model} for auto angle selection.
                      If provided, angle is detected per image via SSIM and the
                      correct model is used.

    Returns:
        List of detection results for all images
    """
    folder = Path(folder_path)
    image_files = get_image_files(folder)

    if not image_files:
        print(f"No images found in {folder_path}")
        return []

    # If angle is explicitly set, use one model for all images
    use_per_image_detection = (angle_name is None and angle_models is not None)

    if not use_per_image_detection:
        if angle_name is None:
            inferred = infer_angle_from_path(folder_path)
            if inferred:
                angle_name = inferred
                print(f"Inferred angle '{angle_name}' from folder path")
            else:
                print(f"Warning: Could not infer angle. Use --angle to specify.")

    print(f"Processing {len(image_files)} images from {folder_path}")

    def get_model_and_angle(img_path):
        """Returns (model, angle_name) for this image."""
        if use_per_image_detection:
            detected = detect_angle_by_ssim(str(img_path), zones)
            if detected and detected in angle_models:
                return angle_models[detected][0], detected
            # Fallback: use default model
            if 'default' in angle_models:
                return angle_models['default'][0], 'default'
            # Last resort: first available
            first_tuple = next(iter(angle_models.values()))
            first_model = first_tuple[0]
            first_angle = next(iter(angle_models.keys()))
            return first_model, first_angle
        else:
            return model, angle_name

    all_results = []
    for img_path in tqdm(image_files, desc='Detecting'):
        mdl, detected_angle = get_model_and_angle(img_path)
        result = detect_single_image(
            img_path, mdl, zones, desks, detected_angle,
            confidence_threshold, person_map
        )
        all_results.append(result)

    return all_results

def detect_folder_best_angle(folder_path, model, zones, desks,
                              confidence_threshold=0.5, output_json=None, person_map=None):
    """
    Sama seperti detect_folder, tapi pakai auto-detect angle per gambar
    (bandingin semua angle, pilih confidence tertinggi) - bukan asal
    ambil angle pertama yang ada di zones.json.
    """
    folder = Path(folder_path)
    image_files = get_image_files(folder)

    if not image_files:
        print(f"No images found in {folder_path}")
        return []

    print(f"Processing {len(image_files)} images from {folder_path} (best-angle mode)")

    all_results = []
    for img_path in tqdm(image_files, desc='Detecting'):
        result = detect_with_best_angle(
            str(img_path), model, zones, desks,
            confidence_threshold, person_map
        )
        if result:
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

Models:
  Place per-angle models in models/ directory:
    models/best_model_angle_1.pth
    models/best_model_angle_2.pth
    models/best_model.pth  (fallback if no angle match)

  Angle is auto-detected via ORB feature matching. Use --angle to force.

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
    parser.add_argument('--model', default='models',
                        help='Path to model directory (default: models/)')
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

    # Load zones
    zones = load_zones(args.zones)
    desks = zones['desks']
    person_map = zones.get('person_map', {})
    print(f"Using device: {DEVICE}")

    # Load all available angle-specific models
    angle_models = load_all_angle_models(args.model if Path(args.model).exists() else 'models')

    # Determine effective model for explicit --angle flag
    if args.angle:
        if args.angle in angle_models:
            explicit_model = angle_models[args.angle][0]
        elif 'default' in angle_models:
            explicit_model = angle_models['default'][0]
        else:
            print(f"Error: No model found for angle '{args.angle}' and no default model.")
            print("Available models: " + ", ".join(angle_models.keys()))
            return
    else:
        explicit_model = None  # will be auto-selected per image

    print(f"Loaded {len(angle_models)} model(s): {list(angle_models.keys())}")

    # Helper: get model for a given detected angle
    def get_model_for_angle(detected_angle):
        if detected_angle and detected_angle in angle_models:
            return angle_models[detected_angle][0], detected_angle
        if 'default' in angle_models:
            return angle_models['default'][0], 'default'
        # fallback: first available
        fallback = next(iter(angle_models.values()))[0]
        return fallback, list(angle_models.keys())[0]

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

        for img_path in random_images:
            if args.angle:
                # Use specified angle
                mdl, used_angle = get_model_for_angle(args.angle)
                result = detect_single_image(
                    str(img_path), mdl, zones, desks, args.angle,
                    args.threshold, person_map
                )
            else:
                # Auto-detect angle via SSIM, then use that angle's model
                detected = detect_angle_by_ssim(str(img_path), zones)
                mdl, used_angle = get_model_for_angle(detected)
                print(f"[{img_path.name}] Detected angle: {used_angle}")
                result = detect_single_image(
                    str(img_path), mdl, zones, desks, used_angle,
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

        for img_path in random_images:
            if args.angle:
                mdl, used_angle = get_model_for_angle(args.angle)
                result = detect_single_image(
                    str(img_path), mdl, zones, desks, args.angle,
                    args.threshold, person_map
                )
            else:
                detected = detect_angle_by_ssim(str(img_path), zones)
                mdl, used_angle = get_model_for_angle(detected)
                print(f"[{img_path.name}] Detected angle: {used_angle}")
                result = detect_single_image(
                    str(img_path), mdl, zones, desks, used_angle,
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
        if args.angle:
            mdl, _ = get_model_for_angle(args.angle)
            result = detect_single_image(
                args.image, mdl, zones, desks, args.angle,
                args.threshold, person_map
            )
        else:
            detected = detect_angle_by_ssim(args.image, zones)
            mdl, used_angle = get_model_for_angle(detected)
            print(f"Detected angle: {used_angle}")
            result = detect_single_image(
                args.image, mdl, zones, desks, used_angle,
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
            args.folder, None, zones, desks, args.angle,
            args.threshold, args.output, person_map,
            angle_models=angle_models
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
