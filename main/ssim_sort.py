"""
ssim_sort.py - Auto-sort desk crops using SSIM similarity matching

Uses empty reference images to automatically classify crops as empty or occupied
based on Structural Similarity Index (SSIM).

Usage:
    python ssim_sort.py --desk desk_a
    python ssim_sort.py --desk desk_a --preview
    python ssim_sort.py --desk desk_a --threshold 0.90

Logic:
    1. Load ALL empty reference images from desk/empty/
    2. Compare each image in desk/to_sort/ against references using SSIM
    3. If SSIM >= 0.90 → very similar → move to empty/
    4. If SSIM <= 0.70 → very different → move to occupied/
    5. If 0.70 < SSIM < 0.90 → uncertain → move to review/ (manual check)

Workflow:
    1. Manually label some images as empty (done)
    2. Run this script to auto-sort remaining images
    3. Check review/ folder for uncertain cases
    4. Manually sort review/ images
    5. Run train_classifier.py
"""

import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not installed. Using basic pixel comparison.")
    print("Install with: pip install scikit-image")


EMPTY_THRESHOLD = 0.90   
OCCUPIED_THRESHOLD = 0.70  

def get_image_files(directory):
    """Get all image files from directory (case-insensitive)."""
    if not directory.exists():
        return []

    extensions = {'.jpg', '.jpeg', '.png'}
    files = []
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in extensions:
            files.append(item)
    return sorted(files)


def load_image_as_array(image_path, target_size=None):
    """Load image and convert to grayscale numpy array."""
    img = Image.open(image_path).convert('L')  
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)


def compute_ssim(img1, img2):
    """Compute SSIM between two images (0 = different, 1 = identical)."""
    if HAS_SKIMAGE:
        return ssim(img1, img2)
    else:
      
        correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
        return max(0, min(1, correlation))


def compute_max_ssim(test_img, reference_images):
    """
    Compute maximum SSIM between test image and all reference images.

    Uses ALL reference images to handle variations in empty state.
    """
    max_ssim = 0
    for ref_img in reference_images:
        
        if ref_img.shape != test_img.shape:
            ref_pil = Image.fromarray(ref_img).resize((test_img.shape[1], test_img.shape[0]))
            ref_img = np.array(ref_pil)

        score = compute_ssim(test_img, ref_img)
        max_ssim = max(max_ssim, score)

    return max_ssim


def ssim_sort_desk(desk_name, crops_dir='dataset/crops', preview=False, force=False):
    """
    Auto-sort desk crops using SSIM similarity matching.

    Args:
        desk_name: Desk identifier (e.g., 'desk_a')
        crops_dir: Base crops directory
        preview: If True, only show results without moving files
        force: If True, overwrite existing classifications
    """
    desk_path = Path(crops_dir) / desk_name
    empty_dir = desk_path / 'empty'
    occupied_dir = desk_path / 'occupied'
    review_dir = desk_path / 'review'
    to_sort_dir = desk_path / 'to_sort'

    
    if not empty_dir.exists():
        print(f"Error: Empty reference folder not found: {empty_dir}")
        return None

    if not to_sort_dir.exists():
        print(f"Error: To-sort folder not found: {to_sort_dir}")
        return None


    occupied_dir.mkdir(exist_ok=True)
    review_dir.mkdir(exist_ok=True)

   
    empty_refs = get_image_files(empty_dir)
    print(f"Using ALL {len(empty_refs)} reference images from {empty_dir}")

    if len(empty_refs) < 1:
        print("Error: No reference images found!")
        return None

    
    print("Loading reference images...")
    reference_images = []
    ref_size = None

    for ref_path in tqdm(empty_refs, desc="Loading refs"):
        ref_array = load_image_as_array(ref_path)
        reference_images.append(ref_array)
        if ref_size is None:
            ref_size = ref_array.shape

    
    to_sort_images = get_image_files(to_sort_dir)
    print(f"\nFound {len(to_sort_images)} images to sort")

    if not to_sort_images:
        print("No images to sort!")
        return None

    print(f"\nClassification thresholds:")
    print(f"  SSIM >= {EMPTY_THRESHOLD} → EMPTY")
    print(f"  SSIM <= {OCCUPIED_THRESHOLD} → OCCUPIED")
    print(f"  {OCCUPIED_THRESHOLD} < SSIM < {EMPTY_THRESHOLD} → REVIEW")
    print("=" * 60)

    results = {
        'empty': [],
        'occupied': [],
        'review': [],
        'skipped': []
    }

    for img_path in tqdm(to_sort_images, desc="Sorting"):
        try:
            test_img = load_image_as_array(img_path, target_size=ref_size)

            max_ssim_score = compute_max_ssim(test_img, reference_images)

            if max_ssim_score >= EMPTY_THRESHOLD:
                predicted_label = 'empty'
            elif max_ssim_score <= OCCUPIED_THRESHOLD:
                predicted_label = 'occupied'
            else:
                predicted_label = 'review'

            results[predicted_label].append({
                'path': img_path,
                'ssim_score': max_ssim_score
            })

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            results['skipped'].append(img_path)

    print("\n" + "=" * 60)
    print("SSIM SORTING RESULTS")
    print("=" * 60)
    print(f"Total images: {len(to_sort_images)}")
    print(f"Classified as EMPTY: {len(results['empty'])}")
    print(f"Classified as OCCUPIED: {len(results['occupied'])}")
    print(f"Classified as REVIEW: {len(results['review'])}")
    print(f"Skipped (error): {len(results['skipped'])}")
    print("=" * 60)

    if results['empty']:
        scores = [r['ssim_score'] for r in results['empty']]
        print(f"\nEMPTY: SSIM range {min(scores):.3f} - {max(scores):.3f}")
    if results['occupied']:
        scores = [r['ssim_score'] for r in results['occupied']]
        print(f"OCCUPIED: SSIM range {min(scores):.3f} - {max(scores):.3f}")
    if results['review']:
        scores = [r['ssim_score'] for r in results['review']]
        print(f"REVIEW: SSIM range {min(scores):.3f} - {max(scores):.3f}")

    if preview:
        print("\n" + "=" * 60)
        print("PREVIEW MODE - No files were moved")
        print("=" * 60)

        print("\nFirst 5 images classified as EMPTY:")
        for r in results['empty'][:5]:
            print(f"  {r['path'].name} (SSIM: {r['ssim_score']:.3f})")

        print("\nFirst 5 images classified as OCCUPIED:")
        for r in results['occupied'][:5]:
            print(f"  {r['path'].name} (SSIM: {r['ssim_score']:.3f})")

        print("\nFirst 5 images classified as REVIEW:")
        for r in results['review'][:5]:
            print(f"  {r['path'].name} (SSIM: {r['ssim_score']:.3f})")

        return results

    print("\n" + "=" * 60)
    print("MOVING FILES...")
    print("=" * 60)

    dest_map = {
        'empty': empty_dir,
        'occupied': occupied_dir,
        'review': review_dir
    }

    for label, file_list in results.items():
        if label == 'skipped':
            continue
        dest_dir = dest_map[label]
        count = 0

        for item in file_list:
            src = item['path']
            dest = dest_dir / src.name

            if dest.exists() and not force:
                continue

            shutil.move(str(src), str(dest))
            count += 1

        print(f"Moved {count} images to {label}/")

    print("\n" + "=" * 60)
    print("FINAL COUNTS")
    print("=" * 60)
    print(f"{desk_name}/empty/: {len(get_image_files(empty_dir))} images")
    print(f"{desk_name}/occupied/: {len(get_image_files(occupied_dir))} images")
    print(f"{desk_name}/review/: {len(get_image_files(review_dir))} images")
    print(f"{desk_name}/to_sort/: {len(get_image_files(to_sort_dir))} images")

    if results['review']:
        print("\n" + "=" * 60)
        print("IMPORTANT: Check review/ folder!")
        print("Manually sort these uncertain images to empty/ or occupied/")
        print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Auto-sort desk crops using SSIM similarity matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ssim_sort.py --desk desk_a          # Run sorting
  python ssim_sort.py --desk desk_a --preview  # Preview only

Thresholds:
  SSIM >= 0.90 → EMPTY (very confident)
  SSIM <= 0.70 → OCCUPIED (very confident)
  0.70 < SSIM < 0.90 → REVIEW (need manual check)

Directory structure after sorting:
  dataset/crops/{desk}/
    ├── empty/     ← Confident empty
    ├── occupied/  ← Confident occupied
    ├── review/     ← Uncertain (manual check needed)
    └── to_sort/   ← Remaining to sort
        """
    )
    parser.add_argument('--desk', type=str, required=True,
                       choices=['desk_a', 'desk_b', 'desk_c'],
                       help='Desk to sort')
    parser.add_argument('--crops-dir', default='dataset/crops',
                       help='Base crops directory')
    parser.add_argument('--preview', action='store_true',
                       help='Preview results without moving files')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing files')

    args = parser.parse_args()

    print("=" * 60)
    print("SSIM AUTO-SORT")
    print("=" * 60)
    print(f"Desk: {args.desk}")
    print(f"Empty threshold: {EMPTY_THRESHOLD}")
    print(f"Occupied threshold: {OCCUPIED_THRESHOLD}")
    print(f"Mode: {'PREVIEW' if args.preview else 'LIVE'}")
    print("=" * 60)

    results = ssim_sort_desk(
        desk_name=args.desk,
        crops_dir=args.crops_dir,
        preview=args.preview,
        force=args.force
    )

    if not args.preview:
        print("\nDone!")


if __name__ == '__main__':
    main()
