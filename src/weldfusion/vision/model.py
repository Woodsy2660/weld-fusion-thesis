"""ResNet-50 with a classifier head sized from the frame (len(Theta))."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from weldfusion.frame import class_order


def num_classes() -> int:
    return len(class_order())


def build_model(n_classes: int | None = None, pretrained: bool = True):
    n_classes = num_classes() if n_classes is None else n_classes
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model


def freeze_backbone(model) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def load_model(path, n_classes: int | None = None, device: str = "cpu"):
    model = build_model(n_classes=n_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
