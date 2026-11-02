# Simulated ideal vision

Labelled simulation of a cleaner vision module on the same 150 weld cells, fused with the **measured** ultrasonic masses. Nothing in this folder is a measured vision result. Measured rows are rebuilt from the real evidence and are accepted only if the validation gate passes.

The script writes only under `results/thesis/sim_vision/`. Every filename starts with `sim_`.

## Run

From the repository root, with numpy, scipy, matplotlib and pandas installed:

```powershell
py -3 scripts/sim_ideal_vision/simulate_ideal_vision.py
```

Defaults point at `results/thesis/data/`. Override them if you move the inputs:

```powershell
py -3 scripts/sim_ideal_vision/simulate_ideal_vision.py `
  --fusion results/thesis/data/fusion_results.json `
  --labels results/thesis/data/labels_v2_2026-10-26.csv `
  --calibration results/thesis/data/calibration_v2.json `
  --out results/thesis/sim_vision `
  --human-baseline 0.80
```

`--human-baseline` is optional. Omit it and the curve has no human-performance line. The value is drawn as given; the script does not estimate it.

Install the four libraries with:

```powershell
py -3 -m pip install -r scripts/sim_ideal_vision/requirements.txt
```

## Read the validation gate first

The script rebuilds vision and ultrasonic masses from the stored belief and plausibility, fuses them again, and compares the result with `fusion_results.json`. It prints a pass/fail table and stops before any simulation if a check fails. A failure is a mismatch to be read, not a cue to edit the combination rule until the numbers agree.

Do not use anything under `results/thesis/sim_vision/` from a run whose gate did not print `RESULT: PASS`. The same table is copied into `sim_summary.md`.

What the gate checks:

- vision targets are `{P}` 26, `{N,U}` 43, `{N,U,L}` 81
- recovered ultrasonic ignorance mass is the constant that rounds to 0.2406
- every decision for V, U and V+U, under Dempster and Murphy, matches the stored decision
- conflict K matches within 1e-6
- macro-F1 on `{N, U, P, L}` matches
- vision alone decides P on 141 cells
- U and V+U each have 97 N/U ties and 29 decisions for L

## How to read the simulation

`q` mixes the measured vision confusion matrix with a perfect identity. `q = 0` keeps the measured error pattern. `q = 1` predicts the vision target. `s` is the fraction of porosity cells the camera is allowed to see; the rest are treated as internal porosity and given the target `{N,U,L}`. The realistic curve uses `s = 0.6`. The hidden cells are a fixed-size random subset, size rounded to the nearest integer, drawn again for every replicate.

Each replicate draws a hard outcome from that matrix, then a confidence from a Beta distribution whose mean is the expected accuracy at that `q` (concentration 20), clipped to `[0.34, 0.98]`. The replicate's own four-fold macro-F1 sets the vision discount. Fusion then uses the real ultrasonic masses.

Seed `20261101`, 50 replicates at each `q` in `{0, 0.1, ..., 1}` and each `s` in `{1.0, 0.6}`. Reported intervals are the mean and the population standard deviation across replicates (divisor `n`), the same convention as the stored vision fold SD.

**Ideal** is the smallest `q` at `s = 0.6` whose mean vision macro-F1 is at least 0.65 and whose mean `{P}` recall is at least 0.70. **Perfect** is `q = 1`, `s = 1`.

## Outputs

All of these are in `results/thesis/sim_vision/`.

| File | What it is |
| --- | --- |
| `sim_summary.md` | Gate, model, reference levels, tables, threshold, limitations |
| `sim_t_modality.tex` | Vision quality: measured, ideal, perfect |
| `sim_t_elements.tex` | Per-outcome precision, recall, F1, support |
| `sim_t_ablation.tex` | V, U and V+U under both rules |
| `sim_t_recall.tex` | Per-class recall |
| `sim_t_mcnemar.tex` | Exploratory U to V+U comparison |
| `sim_f_confusion.pdf` | Measured and ideal confusion matrices |
| `sim_f_fusion.pdf` | Macro-F1 bars and recall heatmap |
| `sim_f_curve.pdf` | V+U performance against vision quality |

PNG copies of the figures are written beside the PDFs at 300 dpi. Each `.tex` file is only a `tabular` body. Its first line is a `% Simulated: ...` comment; paste that sentence into `\caption{}`. Drop the file into an `article` document that loads `booktabs` and `amsmath`.

Figures carry the word SIMULATED because the file compares a simulated sensor with the measured one. Axis text marks which panel is the measured module.

## What is real

Ultrasonic masses, labels, folds, the discount formula, Dempster's rule, Murphy's rule, the pignistic decision and the alphabetical tie-break are the measured pipeline. Geometry is omitted because its mass on the frame is 1. Only the vision basic probability assignment changes with `q` and `s`.
