# Claude Code Prompt — Stratified Split, Three-Head Frame, Retrain, Leave-One-Source-Out

**Created:** 30 September 2026
**Target branch:** branch from `vision_model_training` as `vision/stratified-split-and-three-head`
**Supersedes fold 0:** all fold-0 metrics are invalidated by Task A
**Predecessor:** `claude/claude-code-pad-leakage-and-mapping-prompt.md` (diagnostics complete)

## Why this task exists

The pad-leakage diagnostic cleared the crop-geometry confound but surfaced a larger
problem: the grouped split produced train/val/test sets with very different
source-label compositions, so every fold-0 per-head F1 is a weighted average over the
wrong weights. That is fixed first. The three provisional mappings are now settled,
and the frame change to three heads follows.

### Diagnostic results this task builds on

Leakage — all thresholds cleared, no tiling required:

| matching | V3 ΔF1 | V4 ΔF1 | three-head macro |
|---|---|---|---|
| pad_fraction only | −0.0008 | −0.0065 | 0.8443 |
| box_area only | −0.0140 | −0.0285 | 0.8425 |
| joint (area decile × pad tercile) | −0.0321 | −0.0449 | 0.8324 |

Mapping decisions, all settled:

- `weld_dataset/discontinuity` → V3 **CONFIRMED.** V3 scores F1 0.829 within
  `final_welding` alone, where no `discontinuity` patches exist, so the mapping is not
  propping up the head. The UMAP island is dataset identity, not defect identity:
  crop geometry alone predicts source dataset at 0.790 accuracy against a 0.522
  baseline, and median box area is 224 px² for `weld_dataset` against 8,451 px² for
  `final_welding`.
- `final_welding/cluster of porosity` → V1 **CONFIRMED.** Its weak recall (0.659)
  tracks its padding (mean 0.479 against 0.171 for `pore`), not its label.
- `final_welding/Crack` → V2 **REJECTED.** V2 is removed from the frame; see Task B.

Composition drift that motivates Task A (corpus share → test share):

| head | label | corpus | test | shift |
|---|---|---|---|---|
| V3 | `final_welding/Overlap` | 32.6% | 1.2% | −31.4 |
| V4 | `weld_dataset/stain` | 36.8% | 8.0% | −28.8 |
| V4 | `final_welding/__negative__` | 13.6% | 40.6% | +27.0 |
| V3 | `final_welding/Burn Through` | 5.5% | 22.0% | +16.5 |
| V1 | `final_welding/Porosity` | 43.4% | 25.1% | −18.4 |

---

## Prompt (paste from here)

# Task: stratified split, three-head frame, retrain, leave-one-source-out

Branch from `vision_model_training` as `vision/stratified-split-and-three-head`.

## Context

Fold 0 of the vision focal-element classifier is invalid for two reasons, both now
diagnosed. The crop-geometry confound was investigated and cleared — joint
pad-and-area matched V3/V4 F1 deltas are −0.032 and −0.045, both under the 0.05
threshold, so the letterbox stays and no aspect-aware tiling is needed. But the
grouped split assigned whole source-label populations to single splits, so the test
set's composition bears little resemblance to the corpus. All fold-0 per-head metrics
are therefore averages over the wrong weights and must be recomputed after a
corrected split.

Separately, the V2 `{C}` head is unsupported by the data (60 patches from 46 frames,
precision 0.19) and is being removed from the frame.

## HARD CONSTRAINTS

1. **Do not regenerate the patch corpus.** `data/processed/vision_patches/` stays
   fixed. Task A re-assigns splits within the existing manifest; it does not re-crop.
2. **Do not implement aspect-aware tiling or mask-aware pooling.** The diagnostic
   cleared that confound. Letterboxing with edge replication is retained unchanged.
3. **Head identifiers V1, V3 and V4 keep their names.** V2 is deleted. Do NOT renumber
   V3 to V2 or V4 to V3 — renaming would silently invalidate every existing artefact,
   test and contact sheet. The frame goes from four heads to three, with a gap in the
   numbering, and that is intentional.
4. The model's output dimension and head ordering come from `configs/frame.yaml`. No
   module hardcodes 3 or 4. `logits_*_fold0.npz` from the old run is now incompatible
   and must not be read by any new code.
5. Preserve the leakage guard throughout: no `source_frame` and no `source_image` may
   appear in more than one split.

---

## Task A — label-stratified grouped split

Replace the greedy quota-filling assignment in `src/vision/merge_dedup_split.py`.

**Requirement.** Keep every source frame whole, and preserve the joint
`(head, source_dataset, source_label)` composition across train / val / test at the
existing 80 / 10 / 10 ratio.

**Approach, in order. Stop at the first that meets acceptance.**

1. `sklearn.model_selection.StratifiedGroupKFold` with `groups=source_frame` and
   `y =` the frame's dominant `(source_dataset, source_label)` by patch count.
   Use 10 folds; take one for test, one for val, the remaining eight for train.
2. If that fails acceptance, use a greedy deficit-minimising assignment: sort frames
   by descending patch count, and assign each frame to whichever split currently has
   the largest total shortfall against its target composition vector, measured as
   the sum over labels of |current_share − target_share|.

Report which approach was used and why.

**Acceptance for Task A.** For every `(head, source_label)` pair whose corpus share
within its head is at least 2%:

- |test share − corpus share| ≤ 5 percentage points
- |val share − corpus share| ≤ 5 percentage points

Pairs below 2% are exempt from the bound but must still appear in the reported table.
Also assert the leakage guard still holds on both `source_frame` and `source_image`.

Write `results/vision/split_composition.md` with the full corpus-vs-train-vs-val-vs-test
table per head, and the achieved maximum drift per head.

---

## Task B — three-head frame

### `configs/frame.yaml`

Θ is unchanged: `{N, U, P, L, C}`. The vision modality's focal elements become:

| head | focal element | change |
|---|---|---|
| V1 | `{P}` | unchanged |
| ~~V2~~ | ~~`{C}`~~ | **removed** |
| V3 | `{N,U}` | unchanged |
| V4 | `{N,U,L,C}` | **widened from `{N,U,L}`** |

The widening of V4 is forced and must not be skipped. V4 previously asserted "no
conclusive surface evidence, and definitely not a crack". With no crack head, vision
can no longer support the "not a crack" half, so `C` must re-enter the ignorance set.
A three-head frame that leaves V4 as `{N,U,L}` would emit false evidence.

### `configs/class_map.yaml`

`final_welding/Crack` → **V4** (was V2). Not discard. A crack patch is a legitimate
member of `{N,U,L,C}`, so mapping it to the ignorance head asserts nothing false and
retains 60 patches of training signal for what "no conclusive evidence" looks like.

### Consequences to verify and report

Head counts become V1 4,984; V3 18,926; V4 19,791 + 60 = 19,851. Imbalance drops from
329.9:1 to 3.98:1. Recompute inverse-frequency weights `w_c = N/(K·n_c)` with
N = 43,761 and K = 3, normalised to mean 1 — expected approximately 1.98 / 0.52 / 0.50.
Assert the computed weights match those to two decimal places, or report the
discrepancy.

### `src/bpa/construct.py`

`validate_modality_bpa()` must be **parameterised by modality**. Legal focal elements:

| modality | legal focal elements |
|---|---|
| vision | `{P}`, `{N,U}`, `{N,U,L,C}`, Θ |
| geometry | `{U}`, `{N,P,L,C}`, Θ |
| ut | `{P}`, `{L}`, `{C}`, `{N,U}`, Θ |

The existing test asserting that `{C}` is accepted must be split: `{C}` is now
**rejected for vision** and **accepted for ut**. Do not weaken the rule to "no
singletons" — `{P}` is still a legal vision singleton.

### New frame-algebra tests

Add `tests/test_frame_algebra.py`, asserting against `frame.yaml` rather than
hardcoded sets:

1. **Design property.** `{N,U,L,C} ∩ {N,P,L,C} ∩ {N,U} = {N}` — the three-way
   conjunction of vision's ignorance head, geometry's within-limit head and UT's
   `{N,U}` head isolates the sound weld.
2. **Two-modality route to N.** `{N,U} ∩ {N,P,L,C} = {N}` — vision saw a deviation,
   geometry measured it within ISO 5817 limits.
3. **Vision + geometry ceiling.** Over all pairwise intersections of vision and
   geometry focal elements, the reachable singletons are exactly `{N}`, `{U}` and
   `{P}`, and `{L}` and `{C}` are not separable — they remain fused as `{L,C}`.
   This is a thesis proposition; encode it as an executable test.
4. `{P} ∩ {U} = ∅`, so vision-porosity against geometry-undercut contributes to the
   conflict mass K rather than to any focal element.

---

## Task C — retrain

Two backbones, identical config otherwise, both reported: **ResNet-50** and
**ResNet-18**. ResNet-18 is a genuine candidate, not a formality — 11.7M parameters
against 23.5M, on roughly 5,908 independent frames.

Changes from the fold-0 config, all of which address the epoch-13 overfit
(val loss 0.4840 → 0.9051 while macro-F1 stayed flat):

| setting | fold 0 | new |
|---|---|---|
| max epochs | 30 | **18** |
| early-stopping patience | 8 | **5** |
| weight decay | 1e-4 | **1e-3** |
| backbone freezing | all frozen 5 epochs, then all trainable | **freeze through `layer2` permanently**; train `layer3`, `layer4`, `fc` |
| augmentation | none at runtime | **see below** |

**Augmentation.** Collapsing Roboflow siblings removed variation that was previously
in the data, so this restores signal rather than adding noise.

- horizontal flip, p=0.5
- vertical flip, p=0.5 — weld beads have no canonical up
- rotation ±15°
- brightness and contrast jitter, 0.2
- mild Gaussian blur, p=0.2

Do **not** apply hue or saturation jitter — these are near-monochrome steel surfaces
and hue shifts are unphysical. Do not apply aggressive scale jitter; crop scale
carries some real signal.

Model selection on validation macro-F1, which is now a meaningful number since all
three heads are supported. Log validation loss every epoch alongside it and include
both in the report, so the overfit onset is visible.

**Persist logits with patch identifiers.** The fold-0 npz could not be joined to the
manifest. Every logits artefact from now on carries `patch_path`, `source_frame`,
`source_dataset`, `source_label`, `split`, `pad_fraction`, `box_area` and
`aspect_ratio` alongside the logit array and labels. Write parquet, not npz.

---

## Task D — leave-one-source-out

The single most important experiment in this task. Within-dataset k-fold says nothing
about transfer, and the real RealSense coupon images will resemble none of the three
sources. Accuracy by source on fold 0 was `final_welding` 0.794, `weld_dataset` 0.938,
`wijaya` 0.974 — a gradient that predicts how this model behaves on unseen imagery.

Train on two sources, test on the third, using the best backbone from Task C:

1. Hold out `weld_dataset` — test spans V1, V3, V4. Full metrics.
2. Hold out `final_welding` — test spans V1, V3, V4. Full metrics.
3. Hold out `wijaya` — **degenerate, and must be reported as such.** All 1,049 wijaya
   patches are V4, so the held-out test set has one class. Report V4 recall only.
   Do **not** compute precision, F1 or a macro average for this fold, and say plainly
   in the report why they are undefined.

For each fold report per-head precision, recall, F1 and support on the held-out
source, beside the in-distribution figures from Task C. The gap is the transfer
estimate and it belongs in the thesis.

Also report **expected calibration error** per fold. Calibration degrades under
distribution shift, and the LOSO ECE is the empirical basis for choosing the Shafer
discount factor α in the BPA construction that follows this task.

---

## Task E — regression tests and re-verification

1. Update the `pad_fraction` Cramér's V baseline in `tests/test_no_geometry_leak.py`.
   The split has changed, so re-measure and hardcode the new value with a comment
   naming this run's git SHA. The old baseline is stale, not a target.
2. Re-run the joint geometry-matched subsample check (box-area decile × pad-fraction
   tercile) after the retrain, as an acceptance gate: V3 and V4 matched-minus-unmatched
   F1 deltas must both stay under 0.05. If either exceeds it, stop and report rather
   than proceeding to LOSO.
3. All existing tests pass, with the `{C}`-acceptance test updated per Task B and any
   test asserting four heads updated to read the count from `frame.yaml`.

## Deliverables

```
configs/frame.yaml                                 (modified: V2 removed, V4 widened)
configs/class_map.yaml                             (modified: Crack -> V4)
src/vision/merge_dedup_split.py                    (modified: stratified assignment)
src/bpa/construct.py                               (modified: per-modality validation)
tests/test_frame_algebra.py                        (new)
results/vision/split_composition.md
results/vision/metrics_resnet50.json
results/vision/metrics_resnet18.json
results/vision/logits_val_<backbone>.parquet
results/vision/logits_test_<backbone>.parquet
results/vision/loso_metrics.json
results/vision/loso_report.md
results/vision/geometry_recheck.json
results/vision/retrain_report.md
results/vision/run_manifest_<backbone>.json
```

Manifests follow the existing convention: git SHA, seed, config SHA-256 hashes
(first 16) for frame and class_map and vision_train, torch and CUDA versions,
computed class weights, per-head patch counts, achieved split composition drift.

## Report structure

`retrain_report.md` opens with a comparison table: fold 0 against ResNet-50 against
ResNet-18, per-head P/R/F1 with support, and a three-head macro. State explicitly that
fold-0 figures are not comparable on equal terms, because the split composition
changed — they are shown for continuity, not as a baseline that was beaten.

Then the overfit evidence (val loss and macro-F1 by epoch for both backbones), then
the geometry re-check, then a pointer to `loso_report.md`.

`loso_report.md` leads with the transfer gap: in-distribution macro-F1 against
held-out macro-F1 for each of the two valid folds. That gap, not the headline
accuracy, is the number that predicts performance on the October coupon images.

Write plainly. If ResNet-18 matches or beats ResNet-50, say so and recommend it — a
smaller backbone generalising as well on limited data is a finding, not a concession.
If the LOSO gap is large, state the size of it in the first paragraph.

## Acceptance

- `pytest tests/ -q` passes, including the new frame-algebra tests
- Task A acceptance bounds met, or the failure reported with the achieved drift
- No `source_frame` or `source_image` in more than one split
- Joint geometry-matched V3/V4 F1 deltas both under 0.05 after retrain
- Class weights within 0.01 of 1.98 / 0.52 / 0.50
- Every macro metric is a three-head macro; no four-head figure appears anywhere
- `git diff --stat` shows no change to `data/processed/vision_patches/` contents

## Prompt ends

---

## Notes for interpreting the results

**A worse number after Task A is the expected outcome, not a regression.** Fold-0
metrics were computed on a test set over-weighted toward easy `weld_dataset` patches
(V1 test was 66.7% `pore` against 54.2% in the corpus; V3 test was 42.3%
`discontinuity` against 35.5%). A correctly stratified split should move the headline
down. That is the number becoming true, not the model getting worse.

**Watch the ResNet-18 comparison honestly.** If it matches ResNet-50 the smaller model
should be adopted, because it halves the overfit pressure and the training cost for
every experiment that follows.

**The LOSO gap sets the BPA discount.** If held-out macro-F1 falls well below
in-distribution macro-F1, that gap — not the in-distribution accuracy — is the honest
basis for the Shafer discount α applied to the vision BPA, since the real coupon
images are themselves an unseen source.

## Downstream, after this task

1. Temperature scaling fitted on validation logits, ECE reported on test, per the
   Guo et al. protocol. Note the mild double-dip: the validation split is used for
   both model selection and temperature fitting, so state it.
2. BPA construction: `m(F_i) = (1−α)·p_i` for the three heads, `m(Θ) = α`, with α
   informed by the LOSO gap and reported with a sensitivity sweep over
   α ∈ [0.05, 0.30].
3. Grouped 5-fold on `source_frame` for per-head variance, using the stratified
   assignment from Task A.
4. V1 aspect tiling. V1 F1 falls from 0.86 in the least-padded decile to 0.34 in the
   seventh, and V1's median box is 63 px² — a 28× upscale to 224×224. A pore is
   inherently compact, so an elongated porosity box is a cluster and tiles into square
   sub-crops legitimately. This is a targeted fix for one head, not the general tiling
   that the diagnostic ruled out.
