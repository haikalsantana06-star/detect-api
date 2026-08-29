"""
split_crops_by_angle.py - Split existing crops into per-angle directories

Detects angle by timestamp prefix:
  2026.08.14_* -> angle_1
  2026.08.24_* -> angle_2

Usage:
    python split_crops_by_angle.py
"""

import shutil
from pathlib import Path

CROPS_DIR = Path('dataset/crops')

def split_crops():
    angle_date_map = {
        'angle_1': '2026.08.14',
        'angle_2': '2026.08.24',
    }

    counts = {a: 0 for a in angle_date_map}

    for desk_dir in CROPS_DIR.iterdir():
        if not desk_dir.is_dir() or not desk_dir.name.startswith('desk_'):
            continue

        for class_dir in desk_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name not in ('empty', 'occupied'):
                continue

            for img_path in class_dir.iterdir():
                if not img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    continue

                matched_angle = None
                for angle, date_prefix in angle_date_map.items():
                    if img_path.name.startswith(date_prefix):
                        matched_angle = angle
                        break

                if matched_angle is None:
                    print(f"WARNING: Could not identify angle for {img_path}")
                    continue

                # Target path: dataset/crops/{desk}/{class}/{angle}/
                target_dir = class_dir / matched_angle
                target_dir.mkdir(exist_ok=True)

                target_path = target_dir / img_path.name
                shutil.copy2(img_path, target_path)
                counts[matched_angle] += 1

    for angle, count in counts.items():
        print(f"  {angle}: {count} crops copied")
    print("Done!")


if __name__ == '__main__':
    split_crops()
