"""Inference engine: crop desk zones, preprocess, run PyTorch model in batches."""
import base64
import logging

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import Zone, get_config
from model_loader import ModelLoader

logger = logging.getLogger(__name__)

IMG_SIZE = (96, 96)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


def _decode_image(source: str | bytes) -> np.ndarray:
    """Decode base64 string or raw bytes to OpenCV image (BGR)."""
    if isinstance(source, str):
        if "," in source:
            source = source.split(",", 1)[1]
        raw = base64.b64decode(source)
    else:
        raw = source

    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _crop_and_preprocess(img: np.ndarray, zone: Zone) -> torch.Tensor:
    """Crop a desk zone from the image and preprocess for model input."""
    x, y, w, h = zone.x, zone.y, zone.w, zone.h
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError(f"Empty crop at zone ({x}, {y}, {w}, {h})")

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    tensor = _transform(pil)
    return tensor


def run_inference(
    image_source: str | bytes,
    angle: str,
    threshold: float,
    model_loader: ModelLoader,
) -> dict[str, dict]:
    """
    Run batched inference on an image for a given angle.

    Crops ALL desk zones first, stacks into one batch tensor, runs a single
    forward pass, then extracts probabilities per desk.

    Args:
        image_source: Base64 string or raw bytes of the image
        angle: Angle name (e.g. 'angle_1')
        threshold: Occupancy threshold (0-1)
        model_loader: ModelLoader instance with cached models

    Returns:
        Dict mapping desk name -> {occupied: bool, confidence: float, person: str}
    """
    img = _decode_image(image_source)
    model = model_loader.load_model(angle)
    config = get_config()
    angle_zones = config.zones.get(angle, {})
    person_map = config.person_map

    if not angle_zones:
        return {}

    # Step 1: Crop all zones and collect tensors
    tensors: list[torch.Tensor] = []
    desk_names: list[str] = []

    for desk, zone in angle_zones.items():
        try:
            tensor = _crop_and_preprocess(img, zone)
            tensors.append(tensor)
            desk_names.append(desk)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Crop failed for {desk} (angle {angle}): {e}")
            # Use a zero tensor as placeholder; will be flagged as error
            tensors.append(torch.zeros(3, *IMG_SIZE))
            desk_names.append(desk)

    # Step 2: Stack into single batch tensor [N, C, H, W]
    batch_tensor = torch.stack(tensors).to(model_loader.device)

    # Step 3: Single forward pass for all desks
    with torch.no_grad():
        outputs = model(batch_tensor)
        outputs = outputs.view(-1)
        probabilities = torch.sigmoid(outputs)

    # Step 4: Extract results per desk
    results = {}
    for i, desk in enumerate(desk_names):
        probability = probabilities[i].item()
        is_occupied = probability >= threshold

        results[desk] = {
            "occupied": is_occupied,
            "confidence": round(float(probability), 4),
            "person": person_map.get(desk, desk),
        }

    return results
