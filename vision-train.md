# CONTEXT

The vision data pipeline was rebuilt well - dedup, label-conflict handling,
loud failure on unmapped classes, sys.path fix. Keep all of that.

But it targets the wrong taxonomy, and that has caused two expensive errors.
Fix those, re-run the conversion, then train.

DO NOT rebuild from scratch. convert_and_remap.py and merge_dedup_split.py are
structurally correct; they need a corrected mapping, per-instance cropping, and
a stronger dedup.


# THE TARGET IS FOUR FOCAL ELEMENTS, NOT FOUR CLASSES

The vision modality does not classify over Theta = {N,U,P,L,C}. Its focal
elements in the Dempster-Shafer frame are:

    V1  {P}        surface-breaking porosity
    V2  {C}        surface-breaking crack
    V3  {N,U}      surface geometric anomaly present, magnitude unmeasured
    V4  {N,U,L}    no conclusive surface evidence

Vision NEVER predicts U alone (it cannot measure whether a geometric deviation
exceeds the ISO 5817 limit - geometry decides that) and NEVER predicts L alone
(it cannot see subsurface defects). L appears only inside {N,U,L}.

The current target {porosity, lack_of_fusion, crack, no_defect} trains a head
for lack_of_fusion, which is an internal defect vision has no access to.


# PART A - FRAME AND CONFIG

## A1. configs/frame.yaml

Broaden the definition of U and add the head mapping:

    classes:
      N: "no defect / acceptable"
      U: "geometric non-conformance to ISO 5817 - undercut, excess
          reinforcement, or misalignment beyond limits"
      P: "porosity"
      L: "lack of fusion / lack of penetration"
      C: "crack"

    vision_heads:
      V1: ["P"]
      V2: ["C"]
      V3: ["N", "U"]
      V4: ["N", "U", "L"]

This is the SINGLE SOURCE OF TRUTH. No module hardcodes the head count or the
head-to-focal-element mapping.

## A2. Model and BPA

- vision/model.py: head size read from frame.yaml.
- bpa/construct.py: vision BPA assigns mass to the focal elements named in
  vision_heads, never to Theta singletons. Confirm validate_modality_bpa()
  rejects a singleton assignment; add a test if it does not.


# PART B - CORRECTED LABEL MAPPING

Move the mapping into configs/class_map.yaml. Roughly 9,000 images are
currently discarded that belong in V3 and V4.

    sources:
      final_welding:
        Porosity:                 V1
        Crack:                    V2
        "Lack of Fusion":         V3    # surface-visible LOF is a geometric
                                        # anomaly at the toe; vision cannot see
                                        # internal LOF
        Undercut:                 V3
        Irregular:                V3
        "Excess-Reinforcement":   V3
        Overlap:                  V3
        "Burn Through":           V3
        "root concvity":          V3    # sic - match the source spelling
        "excess root penetration": V3
        Spatters:                 V4
        "Arc strike":             V4
        "mechanical-damage":      V4
        "slag-inclusion":         discard
        Defect:                   discard
        "Bad Weld":               discard
      weld_dataset:
        pore:                     V1
        discontinuity:            V3    # PROVISIONAL - see Part F
        stain:                    V4
        deposit:                  V4
      wijaya:
        "Good Weld":              V4
        "Bad Weld":               discard
        Defect:                   discard

Rationale to preserve in comments:

- V3 collects every surface-visible geometric anomaly. Vision flags that a
  deviation exists; geometry measures whether it exceeds the limit. That
  division of labour is the whole point of the compound focal element.
- Spatter, arc strike and stain carry no evidence about porosity, cracking or
  geometry. They belong in V4, and including them makes the model robust to
  spatter - which the C4 welding condition will produce in quantity.
- Including stain/spatter/arc-strike in V4 also fixes the single-source
  monoculture: V4 stops being 437 Wijaya-only images and becomes ~6,500 across
  all three datasets.

Unmapped source classes must still RAISE, not default to discard. Keep that.


# PART C - PER-INSTANCE CROPS, NOT WHOLE-IMAGE DOMINANT CLASS

## The problem

Current behaviour assigns each image a single label by majority instance count.
Two failures:

1. **Scale mismatch.** Inference operates on ~10 mm weld-cell crops. Training on
   whole weld images teaches features at the wrong scale.
2. **Information loss.** An image with five pores and one crack becomes
   "porosity" and the crack is discarded.

## The fix

For every annotation instance, emit one patch:

- Crop the bounding box with a context margin of 1.5x to 2x the box dimensions,
  clipped to image bounds.
- Label the patch by its own instance class, not the image's dominant class.
- Segmentation polygons: take the axis-aligned bounding box of the polygon.

## Negative sampling for V4

- Sample boxes from annotation-free regions of the same images.
- Draw negative box sizes from the SAME size distribution as the positives.
  If negatives are systematically larger or smaller, the model learns
  "crop size predicts class" and the metrics will not reveal it.

## Aspect ratio - do not stretch

The current pipeline resizes everything to 224x224, and the report notes a
~150x40 subset is upscaled ~5x and aspect-distorted. A 3.7:1 aspect distortion
destroys bead geometry, which is exactly what V3 depends on.

Letterbox instead: preserve aspect, pad to square with edge-replicate or a
neutral fill, then resize. Record the padding fraction per patch in the
manifest so it can be audited.

## Output

    data/processed/vision_patches/{V1,V2,V3,V4}/
    data/processed/vision_patches/manifest.csv

manifest.csv columns:
    patch_path, head, source_dataset, source_image, source_label,
    box_x, box_y, box_w, box_h, is_negative, pad_fraction

source_image is REQUIRED - Part E depends on it.


# PART D - DEDUPLICATION MUST BE PERCEPTUAL, NOT MD5

MD5 over raw bytes catches only byte-identical files. It found 181 duplicates
inside final_welding and called the sources "essentially disjoint".

**Roboflow exports routinely contain augmented copies** - rotated, flipped,
brightness-shifted versions of the same source frame. These are not
byte-identical, so MD5 misses every one of them. If augmented siblings land on
opposite sides of the split, test accuracy is inflated and nothing in the
metrics will show it.

- Replace MD5 with perceptual hashing (imagehash.phash), Hamming distance
  threshold ~5, applied at the SOURCE IMAGE level before cropping.
- Report the count found, and how many are cross-dataset versus within-dataset.
- Keep the existing MD5 pass as a fast exact-match pre-filter if useful.

Keep the existing label-conflict handling unchanged - dropping conflicted
images from every class rather than resolving first-wins is correct.
data/merged/label_conflicts.json should still be written.


# PART E - GROUPED SPLIT BY SOURCE IMAGE

Per-instance cropping means many patches share a source image - same weld, same
lighting, often overlapping pixels.

- Split with GroupShuffleSplit / GroupKFold on groups = manifest.source_image.
- Stratify by head where the grouping permits.
- tests/test_vision_splits.py: assert no source_image appears in more than one
  split.
- A random patch-level split is a bug, not a fallback.


# PART F - REVIEW ARTEFACTS (produce, then continue)

Two mappings need human confirmation. Generate the evidence and keep going -
both live in class_map.yaml, so changing either is one line plus a re-run.

1. weld_dataset `discontinuity` (2,977 instances) -> provisionally V3.
   Crop 50 random samples, tile to
   results/review/discontinuity_sample.png

2. final_welding `Crack` (~2,659) -> V2.
   Surface-breaking cracks are uncommon in GMAW; this count is high enough to
   suspect mislabelling. Crop 50 random samples, tile to
   results/review/crack_sample.png

Print both paths and state that the user must confirm before the results are
written up.


# PART G - TRAIN

Keep what already works: cosine LR, 5-epoch backbone freeze then unfreeze with
the optimizer holding all parameters, inverse-frequency class weights
normalised to mean 1, the dataset-dir resolution fix, the num_classes guard.

Changes:

- Head count from frame.yaml.
- Recompute class weights after the remap - the balance changes substantially.
- Augmentation: random crop, horizontal and vertical flip, rotation, colour
  jitter. Nothing that changes defect appearance semantically.
- **Persist raw logits** for the inner validation fold to
  results/vision/logits_fold{k}.npz. Temperature scaling is fitted downstream
  on these; without them the calibration stage and therefore BPA construction
  cannot run. This is easy to omit and expensive to discover later.
- Fail loudly if any active head has zero training samples. Never silently
  train fewer heads than configured.
- Primary metric MACRO-F1. Report per-class precision/recall/F1 and the
  confusion matrix. Use utils/metrics.py - evaluate.py must import it.
- Write results/vision/metrics_fold{k}.json plus a run manifest with config
  hash, git SHA, seed, class weights, patch counts per head, and dedup counts.

Commit the metrics files.


# NON-GOALS

- Do NOT fabricate metrics or placeholder data.
- Do NOT download additional datasets.
- Do NOT use models/lohi-weld-baselines/ - those .txt files are model
  predictions, not ground truth. The quarantine is correct; leave it.
- Do NOT hardcode the head mapping outside configs/frame.yaml.
- Do NOT use a random patch-level split.
- Do NOT resolve label conflicts by first-wins.
- Do NOT decide the discontinuity or Crack mappings yourself.


# FINISH BY

1. Full test suite passing.
2. Printing: patches per head, source images per split, perceptual-duplicate
   counts, macro-F1 and per-class F1, and both contact sheet paths.
3. A before/after table of how many images the remap recovered from discard.
4. Updating CLAUDE.md: four-head target, broadened U definition, per-instance
   cropping, grouped splitting, perceptual dedup.