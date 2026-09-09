"""Run manifests: every result must be traceable to config hash, git SHA, seed, time."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from weldfusion.utils.config import config_hash, repo_root


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (OSError, subprocess.CalledProcessError):
        return False


def write_manifest(
    dest: str | Path,
    *,
    configs: dict[str, Any] | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a JSON manifest and return it.

    `configs` is a mapping of name -> loaded config dict (hashed individually
    and as a whole).
    """
    configs = configs or {}
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "seed": seed,
        "config_hashes": {name: config_hash(cfg) for name, cfg in configs.items()},
        "config_hash_all": config_hash(configs),
    }
    if extra:
        manifest.update(extra)

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
