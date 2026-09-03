"""SSIM-based angle auto-detection."""
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import json
import logging

logger = logging.getLogger(__name__)

_refs_cache: dict[str, dict[str, np.ndarray]] = {}
_zones_cache: dict | None = None


def _load_zones() -> dict:
    global _zones_cache
    if _zones_cache is not None:
        return _zones_cache

    zones_path = Path(__file__).parent.parent / "zones.json"
    if not zones_path.exists():
        zones_path = Path("zones.json")

    if not zones_path.exists():
        raise FileNotFoundError("zones.json not found")

    with open(zones_path) as f:
        raw = json.load(f)
    _zones_cache = raw["zones"]
    return _zones_cache


def _build_ssim_references(sorted_dir: str = "dataset/sorted") -> dict[str, dict[str, np.ndarray]]:
    """
    Build average grayscale reference crops for each angle from labeled images.
    Returns: {angle_name: {desk_name: avg_grayscale_crop}}
    """
    global _refs_cache
    if _refs_cache:
        return _refs_cache

    sorted_path = Path(sorted_dir)
    if not sorted_path.exists():
        logger.warning(f"SSIM reference dir not found: {sorted_dir}")
        _refs_cache = {}
        return _refs_cache

    all_zones = _load_zones()
    desks = ["desk_a", "desk_b", "desk_c", "desk_d"]

    for angle_folder in sorted_path.iterdir():
        if not angle_folder.is_dir():
            continue
        if not angle_folder.name.startswith("angle_"):
            continue

        angle_name = angle_folder.name

        img_files = [
            f for f in angle_folder.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not img_files:
            continue

        if angle_name not in all_zones:
            continue

        desk_crops: dict[str, list[np.ndarray]] = {d: [] for d in desks}

        for img_path in img_files:
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                continue
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            for d in desks:
                if d not in all_zones[angle_name]:
                    continue
                z = all_zones[angle_name][d]
                crop = gray[z["y"]:z["y"] + z["h"], z["x"]:z["x"] + z["w"]]
                if crop.size > 0:
                    desk_crops[d].append(crop)

        if all(desk_crops[d] for d in desks):
            _refs_cache[angle_name] = {}
            for d in desks:
                stack = np.stack(desk_crops[d])
                _refs_cache[angle_name][d] = np.mean(stack, axis=0)

    logger.info(f"Built SSIM references for angles: {list(_refs_cache.keys())}")
    return _refs_cache


def detect_angle(image_path: str | Path | np.ndarray) -> str | None:
    """
    Auto-detect which angle an image belongs to using SSIM reference matching.
    Returns angle name (e.g. 'angle_1') or None if no confident match.
    """
    refs = _build_ssim_references()
    if not refs:
        logger.warning("No SSIM references available, cannot auto-detect angle")
        return None

    if isinstance(image_path, (str, Path)):
        img_cv = cv2.imread(str(image_path))
        if img_cv is None:
            return None
    else:
        img_cv = image_path

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    all_zones = _load_zones()
    desks = ["desk_a", "desk_b", "desk_c", "desk_d"]

    angle_scores: dict[str, float] = {}

    for angle_name, ref_desks in refs.items():
        if angle_name not in all_zones:
            continue
        score = 0.0
        valid = 0
        for d in desks:
            if d not in ref_desks or d not in all_zones[angle_name]:
                continue
            z = all_zones[angle_name][d]
            crop = gray[z["y"]:z["y"] + z["h"], z["x"]:z["x"] + z["w"]]
            if crop.size == 0:
                continue
            try:
                s = ssim(crop, ref_desks[d], data_range=255)
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

    if best_score < 0.1:
        return None

    logger.info(f"Detected angle: {best_angle} (score: {best_score:.3f})")
    return best_angle
