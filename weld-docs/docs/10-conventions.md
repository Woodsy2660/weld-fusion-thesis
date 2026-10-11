# 10 — Conventions

## Code

- **Python 3.11.** The `pyrealsense2` wheels lag newer releases; do not upgrade the
  acquisition environment past what has a wheel.
- Type hints on public functions. Docstrings stating units for every physical
  quantity — `depth_mm`, not `depth`.
- **Units in variable names** wherever a quantity is physical. The single most
  common bug class in this project is a millimetre/metre or pixel/millimetre mix-up.
- No notebooks in `src/`. Notebooks live in `notebooks/` and are for exploration
  only; anything that produces a thesis figure is a script.
- Prefer boring and explicit over clever. This code is read by an examiner.

## Data

- `data/weld-dataset/` and `data/welding-defect-kaggle/` are **immutable** raw
  sources. Never write to them, never modify in place.
- `data/processed/vision_patches/` holds per-instance crops with `manifest.csv`.
  Reproducible from raw sources via `src/vision/convert_and_remap.py`.
- `data/merged/` holds deduplicated, split-assigned data ready for training.
  Reproducible via `src/vision/merge_dedup_split.py`. Safe to delete.
- When real specimen captures begin, `data/raw/` will hold immutable depth frames
  and images with metadata sidecars. Label CSVs will be versioned in `data/labels/`
  (e.g., `labels_v3_2026-10-14.csv`). Never overwrite a label file; write a new
  version and record why in `DECISIONS.md`.
- Depth frames: individual `.npy`, int16, with a sidecar JSON carrying depth units,
  standoff, exposure, laser power, ASIC temperature, firmware, SDK version, USB type.
- Per-cell tables: Parquet, with a CSV mirror for inspection.

## Reproducibility

Every script that produces a result must:

1. Accept and record a random seed. Set it for `random`, `numpy` and `torch`.
2. Write a **run manifest** to `results/<run_id>/manifest.json` containing: git
   commit hash, seed, full config, input file hashes, package versions, timestamp,
   and hostname.
3. Write outputs under `results/<run_id>/` and never to a bare filename at the repo
   root.

A result without a manifest is not a result. If a number appears in the thesis, it
must be traceable to a run id.

## Naming

- `coupon_id`: `C01` … `C15`
- `zone_id`: `C07-Z2`
- `cell_id`: `C07-Z2-04` — zero-padded cell index within the coupon
- Faces: `top` and `root`, never `front`/`back` or `A`/`B`
- Focal elements in code: frozensets of single-character strings,
  `frozenset({'N','U'})`. Serialised as sorted, comma-joined: `"N,U"`.

## Figures

Follow the project's established style, matched to the thesis typesetting.

- Times New Roman throughout: matplotlib `font.family: serif`,
  `font.serif: Times New Roman`.
- Italic variables in axis labels via LaTeX math mode.
- **No figure title** — the LaTeX caption handles it.
- Clean spines: remove top and right.
- Grid alpha 0.2.
- DPI 300.
- Export both PNG and PDF.
- For safety-critical reliability plots, show **lower confidence bounds only**.

## LaTeX

- **IEEE numeric citations with a manually maintained bibliography** — not BibTeX.
- Structure uses `\include{preamble}` and `\include{titlepage}`.
- Figures use `[H]` placement with `\usepackage{float}`.
- `\FloatBarrier` from `\usepackage{placeins}` at section boundaries.
- Cross-references use `\ref{}` with labels, never hard-coded numbers. Section
  numbering has already shifted once when Chapter 3 was absorbed into Chapter 1.
- When renumbering citations, produce a **citation audit table** mapping old number
  to new, and check every in-text reference against it.

## Git

- Commit messages: imperative mood, one line, with a body if the change affects a
  documented decision.
- A commit that changes a documented decision must also update `DECISIONS.md` in the
  same commit.
- Do not commit `data/raw/`, model weights, or anything over 50 MB. Use a manifest
  with hashes instead.

## Testing

- Every mathematical identity in `02-evidence-framework.md` has a test.
- Every physical formula has a round-trip test: forward-compute, invert, assert
  recovery.
- Regression tests listed at the end of `07-fusion-and-evaluation.md` must pass
  before any result is reported.

## Working with the thesis author

- **Say when something looks wrong.** Five significant errors in this project so far
  were constants carried forward from a superseded assumption. If a number has no
  comment explaining where it came from, question it.
- **Do not resolve a disagreement by preferring the most recent number.** The box-area
  discrepancy (open question 1) is open precisely because recency was not accepted as
  evidence.
- Flag schedule risk explicitly. A refactor that is right but late is wrong.
- When a decision is made, write it to `DECISIONS.md` with the date and the reasoning.
  The reasoning matters more than the decision — it is what lets a future session tell
  whether the decision still applies.
