# CLAUDE.md — read this first, every session

Honours thesis codebase: **multi-modal sensor fusion for AI-assisted defect
classification in structural steel butt welds** (B. Wood, SID 530511030, USYD
AMME4111, supervisor Mehala Balamuri). Thesis due **early November 2026**.

This file is short on purpose. The full context lives in `docs/`. **Read
`docs/00-START-HERE.md` before doing anything non-trivial** — it tells you which
documents matter for the task in hand.

---

## What this system does

Three sensing modalities observe the same weld. Each is physically blind to some
defect classes. Their outputs are converted to Dempster–Shafer basic probability
assignments over a shared frame of discernment and fused. The thesis contribution
is the fusion framework and its analysis, **not** any individual classifier.

| Modality | Sensor | Sees |
|---|---|---|
| Vision (V) | Intel RealSense D435i, colour | Surface appearance |
| Geometry (G) | Intel RealSense D435i, depth | Bead profile, dimensional conformance |
| Ultrasonic (U) | Evident EPOCH 650, 60°/70° angle beam | Subsurface discontinuities |

---

## Hard rules — violating any of these invalidates results

1. **Group by coupon, never by cell.** Any train/test split, cross-validation fold
   or bootstrap resample over real specimen data groups on `coupon_id`. Cells within
   a coupon share plate heat, welder drift, surface finish and lighting.
2. **Group public-corpus data by `source_sequence`, never by filename.** A naive
   split put 81.59% of samples in the same split as a sample from their own source
   sequence. See `docs/04-vision-module.md`.
3. **Class weights, calibration parameters and thresholds are derived from the
   training split only.** Never corpus-wide, never from the full dataset.
4. **Depth units must be 100 µm** (`rs.option.depth_units = 0.0001`). The device
   default of 1000 µm makes the 0.5 mm undercut limit unmeasurable.
5. **No depth post-processing filters.** Especially no hole-filling — it
   interpolates across genuine undercut and fabricates conforming geometry.
6. **Geometry pipeline parameters are frozen before results are seen.** ROI width,
   datum reference and plane-fit extent are set a priori from the ISO 5817
   definition. Tuning them against classification output is leakage.
7. **Labels are never edited on the basis of a model output.** The label file is
   versioned and frozen before training.
8. **Vision may not assert `{L}` or `{C}` alone.** Lack of fusion and cracking are
   subsurface; the frame forbids it and so must the code.
9. **Missing or insufficient data maps to Θ**, not to a guessed class. This is a
   designed advantage of the framework — do not paper over it with imputation.

---

## Repository layout

```
src/
  frame/          frame of discernment, focal elements, set algebra
  vision/         dataset, training, inference, calibration
  geometry/       depth capture, extraction, uncertainty, BPA
  ut/     A-scan features, classifier, calibration
  fusion/         discounting, Murphy, Dempster, pignistic
  evaluation/     splits, metrics, ablation, significance tests
  registration/   datum detection, pose solve, cell indexing
data/
  raw/            immutable captures — never modify in place
  labels/         versioned label CSVs
  interim/        derived, reproducible
results/          figures, tables, run manifests
docs/             the documentation in this folder
```

---

## Before you write code

- Check `docs/08-current-status.md` for what is done, in flight, or blocked.
- Check `docs/DECISIONS.md` for decisions already taken. Do not relitigate them;
  if one looks wrong, say so and ask.
- Follow `docs/10-conventions.md` for style, seeds, data formats and reproducibility.

## When you finish a piece of work

- Append to `docs/DECISIONS.md` if you made or changed a design decision.
- Update `docs/08-current-status.md`.
- Both are part of the deliverable, not optional housekeeping.
