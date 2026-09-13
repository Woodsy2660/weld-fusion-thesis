"""ResNet-50 with a classifier head sized from frame.yaml vision_heads.

The vision classifier outputs logits for 4 focal elements (V1-V4), NOT 5
Theta singletons. The head count is read from configs/frame.yaml - never
hardcoded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from weldfusion.frame import vision_head_count


def num_heads() -> int:
    """Number of vision output heads from frame.yaml."""
    return vision_head_count()


def build_model(n_heads: int | None = None, pretrained: bool = True):
    """Build ResNet-50 with head sized for vision focal elements.

    Parameters
    ----------
    n_heads : int | None
        Number of output logits. If None, reads from frame.yaml.
    pretrained : bool
        Whether to use ImageNet pretrained weights.

    Returns
    -------
    model : nn.Module
        ResNet-50 with adjusted final layer.
    """
    n_heads = num_heads() if n_heads is None else n_heads
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_heads)
    return model


def freeze_backbone(model) -> None:
    """Freeze all layers except the final classifier."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def unfreeze_backbone(model) -> None:
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


def load_model(path, n_heads: int | None = None, device: str = "cpu"):
    """Load a trained model from disk."""
    model = build_model(n_heads=n_heads, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
