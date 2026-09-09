"""DEV-ONLY physics-informed A-scan feature generator.

THIS MODULE MUST NEVER BE A SOURCE OF REPORTED RESULTS.
It exists so the UT training skeleton and BPA construction can be
exercised without a USM 100 on the bench.

Physics (simplified):
    sound path S = z / cos(theta)                 (first-leg indication)
    time_of_flight = S / v_shear
    amplitude ~ reflector response * exp(-alpha S)
    SNR from that amplitude against a noise floor
    backwall echo reduced when a planar reflector shadows the backwall

U (undercut) is a surface geometry class: UT is blind to it, so U and N
share the "no subsurface indication" feature distribution.

Usage: python -m weldfusion.ut.synthetic
Output filename is prefixed DEV_ONLY_ so it cannot be mistaken for data.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from weldfusion.frame import class_order
from weldfusion.ut.features import FEATURE_NAMES
from weldfusion.utils.config import load_named, repo_root

DEV_ONLY_BANNER = (
    "DEV-ONLY synthetic A-scan features. Not for reported results, "
    "not for thesis tables, not for model selection."
)

# Reflector model (z_mm, size_mm, type)
# type: volumetric | planar | none
_REFLECTOR = {
    "P": {"z_mm": 5.0, "size_mm": 1.5, "kind": "volumetric", "n_echoes": 3},
    "L": {"z_mm": 8.0, "size_mm": 4.0, "kind": "planar", "n_echoes": 1},
    "C": {"z_mm": 6.0, "size_mm": 3.0, "kind": "notch", "n_echoes": 1},
    "N": {"z_mm": None, "size_mm": 0.0, "kind": "none", "n_echoes": 0},
    "U": {"z_mm": None, "size_mm": 0.0, "kind": "none", "n_echoes": 0},
}

# Relative linear amplitude by kind (before attenuation).
_KIND_GAIN = {
    "volumetric": 0.25,
    "planar": 0.80,
    "notch": 1.00,
    "none": 0.04,
}

_ATTENUATION_PER_MM = 0.015  # nepers / mm, order-of-magnitude steel shear
_NOISE_LINEAR = 0.03
_BACKWALL_LINEAR = 1.0


def _first_leg_path(z_mm: float, theta_rad: float) -> float:
    return z_mm / math.cos(theta_rad)


def features_for_class(
    cls: str,
    rng: np.random.Generator,
    *,
    thickness_mm: float,
    theta_deg: float,
    velocity_mm_per_us: float,
) -> dict[str, float]:
    spec = _REFLECTOR[cls]
    theta = math.radians(theta_deg)
    kind = spec["kind"]
    jitter = rng.normal(0.0, 0.08)

    if spec["z_mm"] is None:
        # No subsurface indication: ToF sits near the backwall path.
        s_back = thickness_mm / math.cos(theta)
        tof = s_back / velocity_mm_per_us + rng.normal(0.0, 0.15)
        peak_lin = _KIND_GAIN["none"] * math.exp(-_ATTENUATION_PER_MM * s_back)
        peak_lin = max(1e-6, peak_lin * math.exp(jitter))
        width = abs(rng.normal(0.25, 0.08))
        echoes = max(0.0, rng.normal(0.2, 0.3))
        backwall = _BACKWALL_LINEAR * math.exp(-_ATTENUATION_PER_MM * 2 * s_back)
        backwall *= math.exp(rng.normal(0.0, 0.05))
    else:
        z = float(spec["z_mm"]) + rng.normal(0.0, 0.3)
        z = min(max(z, 0.5), thickness_mm)
        s = _first_leg_path(z, theta)
        tof = s / velocity_mm_per_us + rng.normal(0.0, 0.05)
        size = max(0.2, spec["size_mm"] * math.exp(rng.normal(0.0, 0.1)))
        peak_lin = _KIND_GAIN[kind] * size * math.exp(-_ATTENUATION_PER_MM * s)
        peak_lin = max(1e-6, peak_lin * math.exp(jitter))
        # Volumetric reflectors ring longer; notches are sharp.
        width = (1.2 if kind == "volumetric" else 0.45) * (0.5 + 0.15 * size)
        width = abs(width + rng.normal(0.0, 0.05))
        echoes = max(0.0, spec["n_echoes"] + rng.normal(0.0, 0.35))
        shadow = 0.4 if kind in {"planar", "notch"} else 0.15
        s_back = thickness_mm / math.cos(theta)
        backwall = (
            _BACKWALL_LINEAR
            * (1.0 - shadow)
            * math.exp(-_ATTENUATION_PER_MM * 2 * s_back)
            * math.exp(rng.normal(0.0, 0.05))
        )

    peak_db = 20.0 * math.log10(peak_lin)
    noise = _NOISE_LINEAR * math.exp(rng.normal(0.0, 0.1))
    snr_db = 20.0 * math.log10(peak_lin / (noise + 1e-12))
    backwall_db = 20.0 * math.log10(max(backwall, 1e-12))

    return {
        "peak_amplitude_db": round(peak_db, 3),
        "time_of_flight_us": round(max(0.0, tof), 3),
        "signal_width_6db_us": round(width, 3),
        "snr_db": round(snr_db, 3),
        "backwall_echo_db": round(backwall_db, 3),
        "echo_count": round(max(0.0, echoes), 3),
    }


def generate(root: Path | None = None) -> Path:
    cfg = load_named("ut_train")["synthetic"]
    if not cfg.get("dev_only", True):
        raise RuntimeError("Refusing to generate: synthetic.dev_only is not true")

    n = int(cfg["n_samples_per_class"])
    seed = int(cfg["seed"])
    theta = float(cfg["probe_angle_deg"])
    velocity = float(cfg["shear_velocity_mm_per_us"])
    thickness = float(load_named("registration")["coupon"]["thickness_mm"])
    rng = np.random.default_rng(seed)

    rows = []
    for cls in class_order():
        for _ in range(n):
            row = {"defect_class": cls, "dev_only": True}
            row.update(
                features_for_class(
                    cls,
                    rng,
                    thickness_mm=thickness,
                    theta_deg=theta,
                    velocity_mm_per_us=velocity,
                )
            )
            rows.append(row)
    rng.shuffle(rows)

    root = root or repo_root()
    out_dir = root / "data" / "raw" / "ut" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "DEV_ONLY_synthetic_ascan_features.csv"
    fieldnames = ["defect_class", "dev_only", *FEATURE_NAMES]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# {DEV_ONLY_BANNER}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(DEV_ONLY_BANNER)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    return out_path


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
