"""
ResNet-50 model setup for 4-class weld defect classification.
Step 6: pretrained backbone + custom classifier head.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=4, pretrained=True):
    """Build ResNet-50 with modified final layer."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def load_model(path, num_classes=4, device="cpu"):
    """Load a trained model from checkpoint."""
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
