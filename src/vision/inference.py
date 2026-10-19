"""
Single-image inference for vision modality.
Returns confidence scores for fusion pipeline.

Head count comes from configs/frame.yaml, not hardcoded.
"""
import torch
import numpy as np
from PIL import Image

from src.vision.model import load_model
from src.vision.dataset import get_eval_transforms
from src.utils.frame import load_frame


class VisionClassifier:
    """Inference wrapper for the trained ResNet-50 model."""

    def __init__(self, model_path, config, device="cpu", frame=None):
        self.device = torch.device(device)
        self.frame = frame or load_frame()
        # num_heads from frame, not hardcoded
        self.model = load_model(model_path, num_heads=self.frame.num_heads, device=device)
        self.model.to(self.device)
        self.transform = get_eval_transforms(config)
        self.head_names = self.frame.head_names

    def predict(self, image_path):
        """
        Run inference on a single image.

        Args:
            image_path: path to RGB image (RealSense capture or Blender render)

        Returns:
            dict: {head_name: confidence_score} for all heads (from frame)
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return {head: float(prob) for head, prob in zip(self.head_names, probs)}
