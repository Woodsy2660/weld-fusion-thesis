# Vision Dataset Pipeline — Rebuild Report

**Date:** 2026-09-01 / 2026-09-02
**Branch:** `vision_model_training`
**Scope:** Rebuild the vision data pipeline against the datasets actually on disk, and wire the ResNet-50 training script for the resulting class imbalance.

**Target taxonomy (unchanged):** `{porosity, lack_of_fusion, crack, no_defect}`
**Architecture (unchanged):** ResNet-50, ImageNet pretrained

---

## 1. Quarantined the prediction data

`data/Vision/LoHi-Weld/` → `models/lohi-weld-baselines/` — 1.3 GB, 60 YOLOv7-tiny checkpoint runs.

Added `models/lohi-weld-baselines/DO_NOT_USE_AS_TRAINING_DATA.md` recording that the `.txt` files under `test/*/labels/` are **model predictions, not ground truth**. `convert_and_remap.py` only walks `data/Vision/`, so the tree is now structurally unreachable from any DataLoader.

### Old-path references found

None in live code paths. Three in docs and dead code:

| Location | Issue |
|---|---|
| `data/README.md:9` | "Place in: `raw/vision/lohi_weld/`" — path never existed |
| `README.md:46` | Dataset table still lists the old three-source naming |
| `src/vision/remap_classes.py:43` | `RAW_DIRS` points at `data/raw/vision/*`; `CLASS_MAP` uses names (`pores`, `good welding`, `crazing`) present in **none** of the datasets |

`remap_classes.py` was marked **SUPERSEDED** in its docstring; its logic was left untouched. Had it been run, it would have silently produced an empty dataset — every `RAW_DIRS` entry misses, and every `CLASS_MAP` key misses.

---

## 2. `src/vision/convert_and_remap.py` (new)

Detection/segmentation annotations → single-label classification folders under `data/staged/{source}/{class}/`.

### Results

| Dataset | Scanned | Kept | Discarded | `no_defect` candidates |
|---|---|---|---|---|
| final_welding | 23,266 | 9,032 | 14,126 | 108 |
| weld_dataset | 3,022 | 1,433 | 1,512 | 77 |
| wijaya | 1,089 | 440 | 649 | 0 |
| **Total** | **27,377** | **10,905** | **16,287** | **185** |

### Kept, by target class

| Dataset | porosity | lack_of_fusion | crack | no_defect |
|---|---|---|---|---|
| final_welding | 4,135 | 3,333 | 1,564 | — |
| weld_dataset | 338 | — | 1,095 | — |
| wijaya | — | — | — | 440 |
| **Total** | **4,473** | **3,333** | **2,659** | **440** |

### Discarded, by source class

| Class | Count | | Class | Count |
|---|---|---|---|---|
| Burn Through | 3,668 | | Defect | 379 |
| Spatters | 3,337 | | deposit | 319 |
| Irregular | 2,896 | | Bad Weld | 270 |
| Excess-Reinforcement | 1,320 | | slag-inclusion | 143 |
| Arc strike | 1,234 | | root concvity | 30 |
| stain | 1,193 | | excess root penetration | 28 |
| Overlap | 767 | | mechanical-damage | 23 |
| Undercut | 680 | | | |

### Implementation notes

- **Mixed annotation formats.** Verified against the real files — field counts run 5, 11, 17, 19, 21, 23, 25, 27, 33, 43, 87 … 639, all odd (`class` + N×`x y` pairs). Only field 0 is read, so field count is irrelevant.
- **Dominant class** = highest instance count per image. Ties break to the **lowest class ID** so runs are reproducible regardless of filesystem ordering.
- **Unmapped source classes raise**, rather than defaulting to discard. A new class in a re-export fails loudly instead of vanishing.
- **Empty / missing annotations** go to `no_defect_candidates/` for manual review. They are never auto-labelled `no_defect` — an empty label file means "nothing annotated," which is not the same as "no defect present."
- Class names are read from each dataset's `data.yaml`, with `nc` cross-checked against the length of `names`.

---

## 3. `src/vision/merge_dedup_split.py` (new)

`data/staged/` → `data/merged/{class}/` → `data/processed/vision/{train,val,test}/{class}/`

### Deduplication (MD5 over raw bytes, across all datasets)

| Category | Count |
|---|---|
| Unique images kept | 10,698 |
| Duplicates removed | 207 |
| — within final_welding | 181 |
| — cross-class conflicts | 26 |

**Zero benign cross-dataset duplicates.** The three sources are essentially disjoint; the 181 are Roboflow re-exports of the same frame inside FINAL WELDING.

### Label conflicts — 11 images, two contradictory labels each

| Conflicting pair | Count |
|---|---|
| lack_of_fusion vs porosity | 4 |
| crack vs lack_of_fusion | 4 |
| lack_of_fusion vs no_defect | 1 |
| crack vs no_defect | 1 |
| (remaining) | 1 |

The last two matter: a Wijaya **"Good Weld"** image is byte-identical to a FINAL WELDING image annotated as a **defect**. One annotator is wrong. Since `no_defect` comes entirely from Wijaya, it is worth knowing whether this is an isolated slip or evidence that "Good Weld" contains defective welds.

These are dropped from **every** class rather than resolved first-wins — keeping one arbitrarily would inject a known-wrong label. The script writes `data/merged/label_conflicts.json` with full paths on both sides of each conflict for adjudication.

### Final class distribution — verified by directory listing

| Class | train | val | test | **TOTAL** |
|---|---|---|---|---|
| porosity | 3,443 | 430 | 431 | **4,304** |
| lack_of_fusion | 2,644 | 330 | 331 | **3,305** |
| crack | 2,121 | 265 | 266 | **2,652** |
| no_defect | 349 | 43 | 45 | **437** |
| **TOTAL** | **8,557** | **1,068** | **1,073** | **10,698** |

Stratified 80/10/10, seed 42, split per class independently. Source mix: final_welding 8,828 · weld_dataset 1,433 · wijaya 437.

Disk: `data/staged` 804 MB · `data/merged` 768 MB · `data/processed` 770 MB.

---

## 4. Training-side changes

### `configs/vision_train.yaml`

```yaml
model:
  freeze_layers: true
  freeze_epochs: 5
data:
  data_dir: data/processed/vision
training:
  class_weights: inverse_frequency
```

### `src/vision/train.py`

- `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`, stepped per epoch.
- `set_backbone_frozen()` — only `fc.*` trainable for the first 5 epochs, then full unfreeze. The optimizer holds every parameter from the start, so unfreezing needs no rebuild and the head's Adam moment estimates survive the transition.
- Inverse-frequency weighted `CrossEntropyLoss`, `w_c = N / (K · n_c)`, normalised to mean 1 so loss magnitude (and usable learning rate) stays comparable to an unweighted run.
- Per-epoch log now reports phase (`frozen` / `full`) and current LR.

### `src/vision/dataset.py`

- `resolve_data_dirs()` — `data_dir` is authoritative, paths resolve against repo root.
- `compute_class_weights()` — prints per-class counts and weights.
- Guard: raises if `ImageFolder` finds a different number of class folders than `model.num_classes` (an empty class directory is the usual cause).
- Logs that all sources are resized to 224×224 at load time, and that the ~150×40 subset is upscaled ~5× and aspect-distorted.

### Other

| File | Change |
|---|---|
| `requirements.txt` | `pyyaml>=6.0`; full set resolves with no conflicts |
| `.gitignore` | `data/Vision/`, `data/staged/`, `data/merged/`, `models/lohi-weld-baselines/` |
| `src/vision/evaluate.py` | Uses `resolve_data_dirs()`; prints the directory it evaluates |
| `scripts/prepare_and_train.bat` | 4-step pipeline, aborts on any step's failure |

### Bug fixed in the specified commands

`python src/vision/train.py` fails with `ModuleNotFoundError: No module named 'src'` — Python puts the **script's** directory on `sys.path`, not the cwd, so `from src.vision.model import …` cannot resolve. Both `train.py` and `evaluate.py` now bootstrap the repo root, and the `.bat` sets `PYTHONPATH`. The checkpoint directory is anchored to the repo root rather than the cwd.

---

## 5. Files created and modified

### Created

| File | Purpose |
|---|---|
| `src/vision/convert_and_remap.py` | All three datasets → `data/staged/` |
| `src/vision/merge_dedup_split.py` | staged → merged → processed |
| `scripts/prepare_and_train.bat` | Windows 4-step pipeline |
| `models/lohi-weld-baselines/DO_NOT_USE_AS_TRAINING_DATA.md` | Prediction-data warning |
| `docs/vision_pipeline_report.md` | This report |

### Modified

`configs/vision_train.yaml` · `src/vision/train.py` · `src/vision/dataset.py` · `src/vision/evaluate.py` · `requirements.txt` · `.gitignore` · `src/vision/remap_classes.py` (docstring only)

### Moved

`data/Vision/LoHi-Weld/` → `models/lohi-weld-baselines/`

### Generated data

`data/staged/` (11,090 files + `manifest.json`) · `data/merged/` (10,698) · `data/processed/vision/` (10,698 + `split_manifest.json`)

---

## 6. Not done

**Training and evaluation were not run.** The WSL environment has no torch; the GPU is on the Windows side. Steps 3–4 of the batch file are syntax-checked and logically reviewed, but not executed end-to-end.

**One hiccup worth recording:** the first merge run was killed by a tool timeout's SIGTERM at ~8,494 files — `nohup` alone did not detach it from the process group. Re-launched with `setsid` and unbuffered output; it ran to completion, and all figures above come from that clean run.

---

## 7. Before you train

**Imbalance came out at 9.8:1 — just under the 10:1 threshold, so no warning fired.** Treat that as a near-miss, not a pass: it only cleared because dedup removed 169 porosity images.

Three things, in order of likely impact:

1. **`no_defect` is a single-source monoculture.** All 437 images come from Wijaya. The model may learn to separate *that dataset's imaging conditions* from the others rather than clean welds from defects. The 185 images in `data/staged/*/no_defect_candidates/` are mostly from FINAL WELDING and would break that — review them and move genuine clean welds into `data/staged/<source>/no_defect/`, then re-run the merge.
2. **The test split has 45 `no_defect` images.** One misclassification moves recall by ~2 points. Per-class recall on this class will be noisy across runs.
3. **Adjudicate `data/merged/label_conflicts.json`**, particularly the `crack`/`no_defect` and `lack_of_fusion`/`no_defect` pairs.

Class weights (`no_defect` ≈ 6.1×) compensate in the loss, but they do not substitute for real data in the class the fusion stage most needs to be right about.

### Running it

```bat
scripts\prepare_and_train.bat
```

Steps 1–2 regenerate what is already on disk (~40 min of copying). To skip straight to training:

```bat
set PYTHONPATH=%CD%
python src\vision\train.py --config configs\vision_train.yaml
python src\vision\evaluate.py --model models\vision\best_resnet50.pth
```
