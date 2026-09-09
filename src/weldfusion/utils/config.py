"""YAML config loading and hashing. Frame of discernment lives in configs/frame.yaml."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Locate the repository root by walking up from this file, then cwd."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "frame.yaml").exists():
            return parent
    cwd = Path.cwd()
    if (cwd / "configs" / "frame.yaml").exists():
        return cwd
    raise FileNotFoundError("Could not find configs/frame.yaml from package or cwd")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty config: {path}")
    return data


def config_hash(cfg: Any) -> str:
    """Stable SHA-256 of a config mapping (or any JSON-serialisable object)."""
    payload = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_named(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml (name may omit .yaml)."""
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    return load_yaml(Path("configs") / filename)
