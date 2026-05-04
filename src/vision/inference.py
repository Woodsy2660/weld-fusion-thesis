"""
Single-image inference for vision modality.
Returns 4-class confidence scores for fusion pipeline.
"""
import torch
import numpy as np
from PIL import Image

from src.vision.model import load_model
from src.vision.dataset import get_eval_transforms


class VisionClassifier:
    """Inference wrapper for the trained ResNet-50 model."""

    def __init__(self, model_path, config, device="cpu"):
        self.device = torch.device(device)
        self.model = load_model(model_path, num_classes=4, device=device)
        self.model.to(self.device)
        self.transform = get_eval_transforms(config)
        self.classes = config["classes"]

    def predict(self, image_path):
        """
        Run inference on a single image.

        Args:
            image_path: path to RGB image (RealSense capture or Blender render)

        Returns:
            dict: {class_name: confidence_score} for all 4 classes
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {cls: float(prob) for cls, prob in zip(self.classes, probs)}
