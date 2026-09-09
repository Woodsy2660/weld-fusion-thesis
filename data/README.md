# Data (not in git)

Datasets are gitignored. Nothing in this directory is a reported result.

## Vision (RGB)

Download and place as class-named image folders (or convert YOLO/COCO first):

- LoHi-WELD → `raw/vision/lohi_weld/` — https://github.com/SylvioBlock/LoHi-Weld
- Wijaya → `raw/vision/wijaya_kaggle/` — Kaggle `sukmaadhiwijaya/welding-defect-object-detection`
- FINAL WELDING → `raw/vision/final_welding/` — Roboflow `final-welding-dataset-qrz4x-cliho`

Then: `python -m weldfusion.vision.remap_classes`

Processed layout uses folder names from `configs/frame.yaml` (`no_defect`, `undercut`, `porosity`, `lack_of_fusion`, `crack`).

LoHi-WELD ships detection labels. `remap_classes.py` still assumes class-named folders; a converter is not implemented.

## Geometry

- Blender renders: `blender/scripts/generate_welds.py` is a stub. The original scripts produced Figure 7 of the preliminary report and must be recovered or rebuilt.
- abonyilab 3D scanner: https://github.com/abonyilab/3D-scanner-data → `raw/geometry/abonyilab_3d/`

## Ultrasonic

- Real USM 100 exports: `raw/ut/` — parser stubbed (`weldfusion.ut.io_usm100`). Format unconfirmed.
- `python -m weldfusion.ut.synthetic` writes `raw/ut/synthetic/DEV_ONLY_synthetic_ascan_features.csv`. **Not for reported results.**

## Ground truth

YAML records keyed by `(coupon_id, cell_id)` in `groundtruth/`. Independent U references must not be the profilometer under test.
