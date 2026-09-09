"""STUB — Blender synthetic weld generation.

The original scripts produced Figure 7 of the preliminary report
(RGB + depth renders of single-V GMAW beads with controlled undercut,
reinforcement and misalignment). They are not in this repository and
must be recovered from the machine that rendered Figure 7, or rebuilt.

Contract (when recovered):
    - Emit paired RGB and metric-depth (mm) renders into blender/exports/
    - Coupons: 300 x 150 x 10 mm, single-V, Grade 250/350 appearance
    - Depth in millimetres, not Blender normalised units
    - Scribed datums at s = 0, 100, 200, 300 mm
    - This is a DEV / domain-shift source, not a substitute for measured
      geometry on steel

Do not run this stub. It will not produce Figure 7.
"""

from __future__ import annotations


def generate_welds(**kwargs) -> None:
    raise NotImplementedError(
        "Blender weld generation is missing. The original scripts produced "
        "Figure 7 of the preliminary report and must be recovered or rebuilt. "
        "See the module docstring in blender/scripts/generate_welds.py."
    )


if __name__ == "__main__":
    generate_welds()
