# weld-fusion-thesis

Multi-modal weld defect assessment fusing vision, ultrasonic and geometric
evidence under Dempster-Shafer theory.

## The frame of discernment

`configs/frame.yaml` is the **single source of truth**. No module hardcodes the
head count or the head-to-focal-element mapping — read it via
`src/utils/frame.py:load_frame()`.

```
Theta = {N, U, P, L, C}
  N  no defect / acceptable
  U  geometric non-conformance to ISO 5817 — undercut, excess reinforcement,
     or misalignment beyond limits
  P  porosity
  L  lack of fusion / lack of penetration
  C  crack
```

## Vision targets three focal elements, not classes

```
V1  {P}        surface-breaking porosity
V3  {N,U}      surface geometric anomaly present, magnitude unmeasured
V4  {N,U,L,C}  no conclusive surface evidence
```

V2 `{C}` was **removed** — it held 60 patches from 46 source frames and scored
precision 0.19. The numbering skips V2 deliberately so existing artefacts stay valid.

Two invariants, enforced by `src/bpa/construct.py:validate_modality_bpa()` and
pinned by `tests/test_frame_and_bpa.py`:

- **Vision never predicts `U` alone.** It can see that a bead deviates; it
  cannot measure whether the deviation breaches the ISO 5817 limit. Geometry
  decides that. This division of labour is the entire reason `V3` is the
  compound `{N,U}` rather than the singleton `{U}`.
- **Vision never predicts `L` alone.** Lack of fusion is subsurface and has no
  reliable surface signature. `L` appears only inside `V4`.

`{P}` and `{C}` *are* singletons, and that is fine — they are declared heads.
The invariant is "only declared focal elements", not "no singletons". Do not
tighten the validator the wrong way.

A previous iteration trained a `lack_of_fusion` head directly. That was wrong:
it asked the camera to classify a defect it cannot observe.

## Vision data pipeline

```
data/                                         raw sources, read-only
  weld-dataset/                               LoHi-WELD source, 4 classes
  welding-defect-kaggle/The Welding Defect Dataset/   Wijaya, 3 classes
        |
        |  src/vision/convert_and_remap.py
        v
data/processed/vision_patches/{V1,V3,V4}/     per-instance crops (V2 removed)
data/processed/vision_patches/manifest.csv    provenance + split column
        |
        |  src/vision/merge_dedup_split.py
        v
data/merged/                                  deduplicated, split-assigned
```

Note: The Roboflow "FINAL WELDING DATASET" (23,266 images) is referenced in the
original plan but not present in the repository. The single-source retrain decision
(see `weld-docs/docs/04-vision-module.md`) may supersede its use.

### Per-instance cropping, not whole-image dominant class

One patch per annotation **instance**, labelled by that instance's own class.
Two reasons:

- **Scale.** Inference runs on ~10 mm weld-cell crops. Training on whole weld
  images teaches features at the wrong scale.
- **Information.** An image with five pores and one crack is not "porosity" —
  a dominant-class vote throws the crack away.

Boxes are expanded by `CONTEXT_MARGIN` (1.75×) for surrounding bead context and
clipped to image bounds. Segmentation polygons reduce to their axis-aligned
bounding box. Only field 0 of a label line is trusted for the class ID, so
mixed 5-field bbox and variable-length polygon lines parse identically.

### Letterbox, never stretch

Sources are 640×640, ~780×190 and ~150×40. Squashing a 3.7:1 crop to a square
destroys bead geometry — which is precisely the signal `V3` depends on. Patches
preserve aspect ratio and pad to square by edge replication (a flat fill would
inject a synthetic edge the network learns). `pad_fraction` is recorded per
patch in the manifest so the distortion is auditable.

### Negative sampling for V4

Annotation-free regions of the same images. Negative box sizes are drawn from
the **same size distribution as the positives** — if negatives were
systematically larger or smaller, the network would learn "crop size predicts
class" and no metric in the report would reveal it.
`tests/test_vision_splits.py` asserts the medians stay within 4×.

### Perceptual deduplication, at the source-image level, before cropping

MD5 catches only byte-identical files. Roboflow exports routinely contain
augmented copies — rotated, flipped, brightness-shifted versions of one source
frame — and MD5 misses every one. If augmented siblings straddle the split,
test accuracy is inflated and nothing in the metrics shows it.

`imagehash.phash`, Hamming distance ≤ 5, with MD5 retained as an exact
pre-filter. Applied **before** cropping so no patch of a dropped image survives.

Perceptual siblings whose instance→head composition **disagrees** are
contradictory ground truth: every copy is dropped, never first-wins. Written to
`data/merged/label_conflicts.json`.

### Grouped splitting

Dozens of patches share one source image — same weld, same lighting, often
overlapping pixels. Splits are grouped on `manifest.source_image` and stratified
by head where the grouping permits. **A random patch-level split is a bug, not
a fallback**; `tests/test_vision_splits.py` fails if any source image appears in
more than one split.

## Class mapping

`configs/class_map.yaml`. Unmapped source classes **raise** — a class appearing
in a re-exported dataset must stop the pipeline, never vanish into a silent
discard.

Two mappings are **PROVISIONAL** and need human confirmation
(`python src/vision/review_samples.py` writes the contact sheets):

- `weld_dataset/discontinuity` → V3. If these are cracks they belong in V2.
- `final_welding/Crack` → V2. Surface-breaking cracks are uncommon in GMAW and
  the instance count is high enough to suspect the label covers something else.

`final_welding/cluster of porosity` → V1 was **added during implementation** —
it is one of the 15 classes in that source's `data.yaml` but was absent from the
mapping spec, and omitting it would raise. Confirm it.

## Training

`src/vision/train.py`. Head count from the frame. Cosine LR, 5-epoch backbone
freeze then unfreeze with the optimizer holding all parameters throughout (so
the head's Adam moments survive the transition). Inverse-frequency class
weights normalised to mean 1.

- **Model selection is on macro-F1, not accuracy.** With a ~40:1 head imbalance,
  accuracy would happily select a model that never predicts V2.
- **Raw validation logits are persisted** to `results/vision/logits_fold{k}.npz`.
  Temperature scaling is fitted downstream on these, and BPA construction
  consumes the calibrated probabilities. Without them the fusion stage cannot
  run — cheap to write, expensive to discover missing.
- Training **fails loudly** if any declared head has zero training patches.
  Never silently fit fewer heads than the frame declares.
- `results/vision/run_manifest_fold{k}.json` records git SHA, config hashes,
  seed, class weights, patch counts per head and dedup counts.

## Do not

- Use `models/lohi-weld-baselines/` as training data. Those `.txt` files under
  `test/*/labels/` are YOLOv7 **predictions**, not ground truth. The quarantine
  is deliberate.
- Hardcode the head mapping outside `configs/frame.yaml`.
- Use a random patch-level split.
- Resolve label conflicts by first-wins.

## Environment

WSL venv at `~/weldenv` (torch 2.6.0+cu124, RTX 2070). The default PyTorch
wheel targets CUDA 13 and will report `cuda: False` against a 12.7 driver —
install from the `cu124` index.

```bash
~/weldenv/bin/python -m pytest tests/ -q
~/weldenv/bin/python src/vision/convert_and_remap.py
~/weldenv/bin/python src/vision/merge_dedup_split.py
~/weldenv/bin/python src/vision/train.py --config configs/vision_train.yaml
~/weldenv/bin/python src/vision/evaluate.py --model models/vision/best_resnet50_fold0.pth
```
