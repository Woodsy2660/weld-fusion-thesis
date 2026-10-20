"""
Diagnostic: calibration of a checkpoint before and after scalar temperature
scaling. T is fitted on VAL logits only (minimum NLL, log-spaced grid), then
applied to TEST. Not the production calibrator -- none exists yet on this path.

Usage:
    python src/vision/diagnostics/temperature_check.py TAG [TAG ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402
from src.vision.loso import expected_calibration_error  # noqa: E402


def load(tag, split, heads):
    df = pd.read_parquet(REPO_ROOT / "results" / "vision" / f"logits_{split}_{tag}.parquet")
    L = df[[f"logit_{h}" for h in heads]].to_numpy().astype(np.float64).copy()
    y = np.array([heads.index(h) for h in df.head_true])
    return L, y


def nll(L, y, T):
    z = L / T
    z = z - z.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))


def main():
    heads = load_frame().head_names
    print(f"{'tag':<34}{'T':>6}  {'val ECE raw→T':>16}  {'test ECE raw→T':>16}  {'test NLL raw→T':>16}")
    for tag in sys.argv[1:]:
        Lv, yv = load(tag, "val", heads)
        Lt, yt = load(tag, "test", heads)
        grid = np.exp(np.linspace(np.log(0.25), np.log(8), 400))
        T = float(grid[np.argmin([nll(Lv, yv, t) for t in grid])])
        ev0, _ = expected_calibration_error(Lv, yv)
        ev1, _ = expected_calibration_error(Lv / T, yv)
        et0, _ = expected_calibration_error(Lt, yt)
        et1, _ = expected_calibration_error(Lt / T, yt)
        print(f"{tag:<34}{T:6.3f}  {ev0:7.4f}→{ev1:7.4f}  {et0:7.4f}→{et1:7.4f}  "
              f"{nll(Lt, yt, 1.0):7.4f}→{nll(Lt, yt, T):7.4f}")


if __name__ == "__main__":
    main()
