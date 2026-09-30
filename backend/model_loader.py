"""Load and cache PyTorch models per angle."""
import logging
from pathlib import Path

import torch
from torch import nn
from torchvision import models

logger = logging.getLogger(__name__)


class MobileNetV2Classifier(nn.Module):
    """Must match the architecture used in train_classifier.py exactly."""

    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights
            self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except ImportError:
            self.model = models.mobilenet_v2(weights=None)

        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ModelLoader:
    """Load and cache one model per angle."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self._cache: dict[str, nn.Module] = {}

    def load_model(self, angle: str) -> nn.Module:
        """Load model for given angle, cached in memory."""
        if angle in self._cache:
            return self._cache[angle]

        model_path = self.model_dir / f"best_model_{angle}.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Available models: {list(self.model_dir.glob('best_model_*.pth'))}"
            )

        logger.info(f"Loading model from {model_path} on {self.device}")

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        model = MobileNetV2Classifier(pretrained=False)
        state_dict = checkpoint["model_state_dict"]

        # Keys in checkpoint are "model.features..." / "model.classifier..."
        # matching the self.model attribute name, so no prefix stripping needed.
        # (If keys were bare "features..." they'd already be correct as-is.)

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        logger.info(f"Model for '{angle}' loaded. Val acc: {checkpoint.get('val_acc', 'N/A')}")
        self._cache[angle] = model
        return model

    def get_loaded_angles(self) -> list[str]:
        """Return list of angles with cached models."""
        return list(self._cache.keys())

    def preload_all(self, angles: list[str]) -> None:
        """Pre-load all specified angles at startup."""
        for angle in angles:
            try:
                self.load_model(angle)
            except FileNotFoundError as e:
                logger.warning(f"Could not preload {angle}: {e}")
