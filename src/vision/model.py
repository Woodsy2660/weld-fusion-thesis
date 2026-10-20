"""
ResNet-50 backbone for the vision modality.

The classifier emits one logit per VISION FOCAL ELEMENT, not per class in
Theta. The head count comes from configs/frame.yaml -- nothing here hardcodes
it, so adding or removing a focal element is a config change.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402


ARCHITECTURES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}

# Backbone stages in order, so "freeze_through: layer2" is resolvable without
# hardcoding which module names exist.
RESNET_STAGES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


def build_model(num_heads=None, pretrained=True, frame=None,
                architecture="resnet50", freeze_through=None):
    """
    A ResNet with the final FC resized to the frame's vision head count.

    num_heads is optional and exists only for tests; in normal use the count is
    taken from the frame so the model cannot drift out of step with the BPA
    construction that consumes its outputs.

    freeze_through freezes every stage up to and including the named one, for
    the whole run. ResNet-18 carries 11.7M parameters against ResNet-50's
    23.5M, which on roughly 5,908 independent source frames is a real argument
    rather than a formality.
    """
    if num_heads is None:
        frame = frame or load_frame()
        num_heads = frame.num_heads

    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}; "
                         f"known: {sorted(ARCHITECTURES)}")
    ctor, weight_enum = ARCHITECTURES[architecture]
    model = ctor(weights=weight_enum if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_heads)
    model.num_heads = num_heads
    model.architecture = architecture

    if freeze_through:
        set_frozen_through(model, freeze_through)
    return model


def set_frozen_through(model, stage):
    """Freeze conv1..stage inclusive; everything after stays trainable."""
    if stage not in RESNET_STAGES:
        raise ValueError(f"freeze_through={stage!r} is not a ResNet stage; "
                         f"known: {RESNET_STAGES}")
    cutoff = RESNET_STAGES.index(stage)
    frozen = set(RESNET_STAGES[:cutoff + 1])
    for name, module in model.named_children():
        if name in frozen:
            for param in module.parameters():
                param.requires_grad = False
    return frozen


def load_model(path, num_heads=None, device="cpu", frame=None,
               architecture=None):
    """Load a trained checkpoint, sizing the head from the frame.

    architecture=None detects it from the checkpoint: ResNet-50 bottleneck
    blocks have a conv3, ResNet-18 basic blocks do not. Callers that omitted
    the argument previously got resnet50 and failed on resnet18 checkpoints.
    """
    state = torch.load(path, map_location=device)
    if architecture is None:
        architecture = ("resnet50" if "layer1.0.conv3.weight" in state
                        else "resnet18")
    model = build_model(num_heads=num_heads, pretrained=False, frame=frame,
                        architecture=architecture)
    model.load_state_dict(state)
    model.eval()
    return model
