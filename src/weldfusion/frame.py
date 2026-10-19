"""
Frame of discernment interface for the weldfusion pipeline.

This module wraps src.utils.frame to provide a clean interface for the
pipeline stages. All frame-related information is read from configs/frame.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure src is on path for importing src.utils.frame
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame as _load_frame  # noqa: E402

if TYPE_CHECKING:
    from src.utils.frame import Frame


# Module-level cache for the frame (loaded once)
_frame_cache: "Frame | None" = None


def get_frame() -> "Frame":
    """Get the cached frame, loading from config if needed."""
    global _frame_cache
    if _frame_cache is None:
        _frame_cache = _load_frame()
    return _frame_cache


def reload_frame() -> "Frame":
    """Force reload the frame from config. Use sparingly."""
    global _frame_cache
    _frame_cache = _load_frame()
    return _frame_cache


# ---------------------------------------------------------------------------
# Theta and class utilities
# ---------------------------------------------------------------------------

def theta() -> frozenset[str]:
    """Return Theta as a frozenset of class symbols."""
    return frozenset(get_frame().classes)


def class_order() -> list[str]:
    """Return class symbols in sorted order."""
    return sorted(get_frame().classes)


def num_classes() -> int:
    """Number of classes in Theta."""
    return len(get_frame().classes)


# ---------------------------------------------------------------------------
# Focal element type
# ---------------------------------------------------------------------------

# A focal element is a non-empty subset of Theta represented as a frozenset.
Focal = frozenset[str]


# ---------------------------------------------------------------------------
# Per-modality focal elements
# ---------------------------------------------------------------------------

def vision_focal_elements() -> dict[str, Focal]:
    """Return vision focal elements as {head: frozenset}."""
    frame = get_frame()
    return {h: frozenset(v) for h, v in frame.heads_for("vision").items()}


def geometry_focal_elements() -> dict[str, Focal]:
    """Return geometry focal elements as {head: frozenset}."""
    frame = get_frame()
    return {h: frozenset(v) for h, v in frame.heads_for("geometry").items()}


def ut_focal_elements() -> dict[str, Focal]:
    """Return UT focal elements as {head: frozenset}."""
    frame = get_frame()
    return {h: frozenset(v) for h, v in frame.heads_for("ut").items()}


def modality_focal_elements(modality: str) -> dict[str, Focal]:
    """Return focal elements for any modality as {head: frozenset}."""
    frame = get_frame()
    return {h: frozenset(v) for h, v in frame.heads_for(modality).items()}


# ---------------------------------------------------------------------------
# Vision head utilities (for classifier interface)
# ---------------------------------------------------------------------------

def vision_head_names() -> list[str]:
    """Return vision head names in classifier output order."""
    return get_frame().head_names


def num_vision_heads() -> int:
    """Number of vision heads."""
    return get_frame().num_heads
