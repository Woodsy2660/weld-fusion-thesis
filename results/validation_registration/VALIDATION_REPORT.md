# Registration Pipeline Validation Report

**Date:** 2026-10-17
**Pipeline:** `src/geometry/registration_v2.py`
**Captures tested:** 36 (15 coupons × 2 faces + 6 repeats)

## Summary

| Metric | Value |
|--------|-------|
| Total captures | 36 |
| Passed (no flags) | 0 |
| Warnings (baseline error, y_crown exceeded) | 31 |
| Failures (missing holes, no depth) | 5 |

**Outcome:** Hole detection needs calibration. The adaptive thresholding parameters
are tuned for synthetic tests but pick up spurious dark regions (weld spatter, shadows)
in real images.

## What Works

1. **Pure-numpy fallback deprojection** — works without pyrealsense2
2. **Face derivation** — correctly identifies cap vs root from hole positions
3. **Point cloud extraction** — ~400k points per capture, transforms work
4. **Baseline validation logic** — correctly flags out-of-tolerance values
5. **y_crown tracking** — computes bead centreline deviation

## What Needs Work

### Hole Detection (critical)

The detector finds too many candidates:
- C11_top_r0: 333 dark blob candidates → 17 classified as "holes"
- C11_root_r0: 418 candidates → 35 classified as "holes"
- Expected: 2-3 holes per image (Ø6, Ø8, optional Ø10)

**Root cause:** Adaptive thresholding (`block_size=51, C=10`) picks up every dark
region including:
- Weld spatter
- Stereo shadows at plate edges
- Surface oxidation patterns
- Bead irregularities

**Next steps:**
1. Review diagnostic images: `results/diagnostics/hole_detection/*.png`
2. Add shape quality filter (circularity, solidity)
3. Add position constraints (datum holes should be in datum half: y > 20mm)
4. Consider edge-based detection instead of blob-based (Canny → Hough circles)
5. Use colour information (machined bore walls are bright/reflective)

### y_crown Threshold

All captures exceed the 3mm threshold (y_crown_max ~12mm). This is because:
- The bead is not centred at y=0 in the current pose estimate
- The spurious hole detection gives wrong pose

**Deferred until:** Hole detection is fixed

## Diagnostic Artefacts

```
results/diagnostics/hole_detection/
├── C11_top_r0_holes.png    # Annotated detections
├── C11_top_r0_thresh.png   # Adaptive threshold mask
├── C11_root_r0_holes.png
├── C11_root_r0_thresh.png
├── C01_top_r0_holes.png
└── C01_top_r0_thresh.png
```

Legend:
- **Yellow:** Plate mask boundary
- **Blue:** All dark blob candidates
- **Green:** Classified as origin (Ø6)
- **Red:** Classified as axis (Ø8)
- **Cyan:** Classified as check (Ø10)

## Synthetic Test Results

All 8 synthetic tests pass — the pipeline geometry is correct when given accurate
hole positions. The issue is upstream in detection, not downstream in pose solve.

```
tests/test_registration_synthetic.py::TestOpeningCentroidBias::test_d6_hole_bias_matches_theory PASSED
tests/test_registration_synthetic.py::TestOpeningCentroidBias::test_baseline_shortened_by_bias PASSED
tests/test_registration_synthetic.py::TestRimFitting::test_rim_fit_recovers_d6_centre PASSED
tests/test_registration_synthetic.py::TestRimFitting::test_rim_fit_recovers_d8_centre PASSED
tests/test_registration_synthetic.py::TestPoseRecovery::test_baseline_accurate_to_0p5mm PASSED
tests/test_registration_synthetic.py::TestPoseRecovery::test_origin_position_accurate PASSED
tests/test_registration_synthetic.py::TestEndForEndInvariance::test_end_for_end_baseline_unchanged PASSED
tests/test_registration_synthetic.py::TestRootFace::test_root_face_baseline_unchanged PASSED
```
