"""Inference engine: crop desk zones, preprocess, run PyTorch model."""
import base64
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
import logging

from model_loader import ModelLoader
from config import get_config, Zone

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
    Run inference on an image for a given angle.

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

    results = {}

    for desk, zone in angle_zones.items():
        try:
            tensor = _crop_and_preprocess(img, zone)
            tensor = tensor.unsqueeze(0).to(model_loader.device)

            with torch.no_grad():
                output = model(tensor)
                output = output.view(-1)
                probability = torch.sigmoid(output).item()
                is_occupied = probability >= threshold

            results[desk] = {
                "occupied": is_occupied,
                "confidence": round(float(probability), 4),
                "person": person_map.get(desk, desk),
            }
        except Exception as e:
            logger.error(f"Inference failed for {desk} (angle {angle}): {e}")
            results[desk] = {
                "occupied": False,
                "confidence": 0.0,
                "person": person_map.get(desk, desk),
            }

    return results
