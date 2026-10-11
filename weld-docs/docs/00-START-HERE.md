# 00 — Start here

**Last updated: 2026-10-09**

You are working on an undergraduate honours thesis codebase with a hard deadline in
early November 2026. The research contribution is an evidence-theoretic sensor fusion
framework, not any individual machine learning model. Code quality matters, but
**correctness of the experimental protocol matters more** — a leaked split or a
mis-scoped calibration invalidates a result in a way no amount of refactoring fixes.

## Reading order by task

| If you are working on… | Read |
|---|---|
| Anything at all | `CLAUDE.md` (repo root), this file |
| Fusion, frame, focal elements, BPAs | `02-evidence-framework.md` |
| Datasets, labels, registration, cells | `03-data-and-labels.md` |
| The vision classifier | `04-vision-module.md`, then `03` |
| Depth capture or dimensional measurement | `05-geometry-module.md` |
| Ultrasonic features or classifier | `06-ultrasonic-module.md` |
| Splits, metrics, ablation, statistics | `07-fusion-and-evaluation.md` |
| Planning or sequencing work | `08-current-status.md`, `09-roadmap.md` |
| Style, naming, reproducibility | `10-conventions.md` |
| "Why is it done this way?" | `DECISIONS.md` |

## The one-paragraph version

A weld coupon is divided into 25 mm cells along its length. Each cell is observed by
three modalities. Each modality produces a basic probability assignment over subsets
of the frame Θ = {N, U, P, L, C} — but only over the subsets it can physically
resolve, which is the central design idea. The BPAs are discounted by measured
modality reliability, combined with Murphy averaging plus Dempster's rule, and
converted to a decision by the pignistic transform. The headline experiment is a
seven-condition ablation (V, G, U, V+G, V+U, G+U, V+G+U) compared against both
ground truth and an algebraic ceiling derived from the frame structure before any
data existed.

## What "done" looks like

Three per-cell BPA tables over the same 150 cells, same index, aligned; a fusion
engine that consumes them; an ablation table with significance tests; and a
comparison against the algebraic ceiling. Everything else is supporting work.

## Things that have already gone wrong once

Listed here so they do not happen twice. Details in the module documents.

1. **Sequence leakage in the public image corpora** — 81.59% contamination, every
   in-distribution figure invalidated.
2. **Class weights computed corpus-wide** instead of from the training split.
3. **`final_welding/Lack of Fusion` mapped to the `{N,U}` head** — L is not in
   `{N,U}`. Must map to `{N,U,L,C}`.
4. **A 200 mm camera standoff taken from the wrong datasheet** (D405, not D435i).
   The D435i cannot focus below ~280 mm. Now 300 mm.
5. **Datum hole position specified at y = 68 mm** when the plate half-width is 70 mm,
   leaving 2 mm of edge distance. Now y = 50 mm.

The pattern in all five: a number carried forward from an earlier assumption without
rechecking it against the thing it describes. If a constant in the code has no
comment saying where it came from, treat it as suspect.
