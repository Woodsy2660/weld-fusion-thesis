"""Single-image inference. Returns a posterior over frame classes for BPA construction."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from weldfusion.frame import class_order
from weldfusion.vision.dataset import get_eval_transforms
from weldfusion.vision.model import load_model


class VisionClassifier:
    """Inference wrapper for a trained ResNet-50."""

    def __init__(self, model_path, config, device="cpu", temperature: float = 1.0):
        self.device = torch.device(device)
        self.model = load_model(model_path, device=device)
        self.model.to(self.device)
        self.transform = get_eval_transforms(config)
        self.classes = list(class_order())
        self.temperature = temperature

    def predict_logits(self, image_path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).cpu().numpy()[0]
        return logits

    def predict(self, image_path) -> dict[str, float]:
        from weldfusion.calibration.temperature import apply_temperature

        logits = self.predict_logits(image_path)
        probs = apply_temperature(logits[None, :], self.temperature)[0]
        return {cls: float(p) for cls, p in zip(self.classes, probs)}
