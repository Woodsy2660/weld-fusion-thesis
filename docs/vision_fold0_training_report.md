# Vision Focal-Element Classifier — Fold 0 Training Report

**Date:** 13 September 2026
**Branch:** `vision_model_training`
**Git SHA:** `87db5fc73abcb0840318a85fc0730dc7351629c9`
**Seed:** 42 · **Fold:** 0 · **Device:** NVIDIA GeForce RTX 2070, torch 2.6.0+cu124

**Config hashes (SHA-256, first 16):**

| file | hash |
|---|---|
| `configs/frame.yaml` | `8e3c5f0576244d72` |
| `configs/class_map.yaml` | `500f40bc53e0cb75` |
| `configs/vision_train.yaml` | `6786b79083b211a7` |

---

## 0. Summary

First training run against the corrected Dempster–Shafer target. The vision modality now predicts four **focal elements** rather than four classes in Θ.

| metric | value |
|---|---|
| Test macro-F1 (4 heads) | **0.7075** |
| Test macro-F1 (3 supported heads) | **0.8392** |
| Test accuracy | 0.8652 |
| Test macro precision | 0.6753 |
| Test macro recall | 0.8404 |
| Best validation macro-F1 | 0.6614 (epoch 21) |
| Epochs run | 29 of 30 (early stop, patience 8) |

**The four-head macro-F1 of 0.7075 must not be quoted.** One of the four heads — V2 `{C}` — rests on 6 test patches drawn from 46 source frames and scores precision 0.1923. It contributes a quarter of the macro average while being statistically meaningless. The three supported heads average 0.8392.

---

## 1. Target definition

From `configs/frame.yaml`, the single source of truth. No module hardcodes the head count or mapping.

```
Θ = {N, U, P, L, C}
  N  no defect / acceptable
  U  geometric non-conformance to ISO 5817 — undercut, excess
     reinforcement, or misalignment beyond limits
  P  porosity
  L  lack of fusion
  C  crack
```

| head | focal element | meaning |
|---|---|---|
| V1 | `{P}` | surface-breaking porosity |
| V2 | `{C}` | surface-breaking crack |
| V3 | `{N,U}` | surface geometric anomaly present, magnitude unmeasured |
| V4 | `{N,U,L}` | no conclusive surface evidence |

Two invariants, enforced by `src/bpa/construct.py:validate_modality_bpa()` and pinned by `tests/test_frame_and_bpa.py`:

- **Vision never predicts `U` alone.** It can see that a bead deviates; it cannot measure whether the deviation breaches the ISO 5817 limit. That is the geometry modality's measurement. This is exactly why V3 is the compound `{N,U}` and not the singleton `{U}`.
- **Vision never predicts `L` alone.** Lack of fusion is subsurface with no reliable surface signature. `L` appears only inside V4.

`{P}` and `{C}` *are* singletons and are legal, because they are declared heads. The validator's rule is "only declared focal elements", not "no singletons".

The previous taxonomy `{porosity, lack_of_fusion, crack, no_defect}` trained a head for `lack_of_fusion` — an internal defect the camera cannot observe. That head has been removed.

---

## 2. Test results

Held-out split: 4,376 patches from 1,332 source frames. Grouped so no source frame appears in more than one split.

### 2.1 Per head

| head | focal | precision | recall | F1 | support | corpus frames |
|---|---|---|---|---|---|---|
| V1 | `{P}` | 0.7316 | 0.7756 | 0.7529 | 499 | 1,170 |
| **V2** | `{C}` | **0.1923** | 0.8333 | **0.3125** | **6** | **46** |
| V3 | `{N,U}` | 0.8955 | 0.8605 | 0.8776 | 1,892 | 4,736 |
| V4 | `{N,U,L}` | 0.8817 | 0.8924 | 0.8870 | 1,979 | 5,863 |
| **macro** | — | 0.6753 | 0.8404 | 0.7075 | 4,376 | 1,332 |

Corpus frames are unique source frames contributing to each head across the whole dataset, not test-split counts.

Macro recall (0.8404) sits far above macro precision (0.6753). That gap is the signature of the 3.93× inverse-frequency weight on V2 buying recall with false positives.

### 2.2 Confusion matrix

Rows are ground truth, columns are prediction.

|  | → V1 | → V2 | → V3 | → V4 | total |
|---|---|---|---|---|---|
| **V1** `{P}` | **387** | 1 | 46 | 65 | 499 |
| **V2** `{C}` | 0 | **5** | 1 | 0 | 6 |
| **V3** `{N,U}` | 75 | 17 | **1628** | 172 | 1,892 |
| **V4** `{N,U,L}` | 67 | 3 | 143 | **1766** | 1,979 |
| **predicted total** | 529 | 26 | 1,818 | 2,003 | 4,376 |

### 2.3 Reading the off-diagonals

**V3 ↔ V4 is the largest error mass: 315 patches** (172 + 143). This is the expected hard boundary. `{N,U}` and `{N,U,L}` differ by a single subsurface symbol with no surface signature, so the distinction rests entirely on whether a visible geometric deviation is present. 8–9% leakage is the honest cost of that boundary, and it is not a bug in the mapping.

**V2's column is the expensive failure.** Reading down it: 17 true-V3 and 3 true-V4 patches called crack, against 5 correct — 21 false positives for 5 true. Every false crack call injects mass onto `{C}`, a *singleton* focal element. A singleton commits far harder under Dempster's rule of combination than a compound set does, because it has no overlap with competing hypotheses to dilute it. V2's errors are therefore the most damaging errors this model can make in the fused decision.

**V1 leaks in both directions.** 111 porosity patches escape to V3/V4 (46 + 65), and 142 V3/V4 patches arrive as porosity (75 + 67). Small dark pores and small dark surface marks converge once a crop is letterboxed to 224×224.

---

## 3. Training dynamics

Cosine LR from 1×10⁻⁴ to 1×10⁻⁶, backbone frozen for 5 epochs then fully unfrozen, Adam, weight decay 1×10⁻⁴, batch 64. Model selection on validation macro-F1, early stopping patience 8.

### 3.1 Full epoch history

| ep | lr | train loss | train acc | val loss | val acc | val macro-F1 |
|---|---|---|---|---|---|---|
| 1 | 1.00e-04 | 1.0373 | 0.6550 | 1.0854 | 0.6181 | 0.4358 |
| 2 | 9.97e-05 | 0.8541 | 0.7092 | 1.0259 | 0.6181 | 0.4529 |
| 3 | 9.89e-05 | 0.7729 | 0.7275 | 0.9608 | 0.6117 | 0.4530 |
| 4 | 9.76e-05 | 0.7234 | 0.7384 | 1.0000 | 0.5836 | 0.4396 |
| 5 | 9.57e-05 | 0.6971 | 0.7429 | 0.9455 | 0.5996 | 0.4486 |
| 6 | 9.34e-05 | 0.4567 | 0.8331 | 0.7531 | 0.6865 | 0.5165 |
| 7 | 9.05e-05 | 0.3181 | 0.8844 | 0.6492 | 0.7598 | 0.5642 |
| 8 | 8.73e-05 | 0.2699 | 0.8987 | 0.5629 | 0.7879 | 0.5891 |
| 9 | 8.36e-05 | 0.2378 | 0.9134 | 0.5810 | 0.8197 | 0.6176 |
| 10 | 7.96e-05 | 0.2022 | 0.9226 | 0.5208 | 0.8279 | 0.6270 |
| 11 | 7.52e-05 | 0.1797 | 0.9329 | 0.7764 | 0.7617 | 0.5698 |
| 12 | 7.06e-05 | 0.2086 | 0.9222 | 0.6158 | 0.7877 | 0.5954 |
| **13** | 6.58e-05 | 0.1614 | 0.9376 | **0.4840** | 0.8350 | 0.6322 |
| 14 | 6.08e-05 | 0.1410 | 0.9452 | 0.4990 | 0.8323 | 0.6291 |
| 15 | 5.57e-05 | 0.1213 | 0.9518 | 0.4941 | 0.8451 | 0.6441 |
| 16 | 5.05e-05 | 0.1133 | 0.9551 | 0.5233 | 0.8446 | 0.6448 |
| 17 | 4.53e-05 | 0.1049 | 0.9591 | 0.5487 | 0.8457 | 0.6450 |
| 18 | 4.02e-05 | 0.0936 | 0.9623 | 0.5437 | 0.8469 | 0.6440 |
| 19 | 3.52e-05 | 0.0817 | 0.9674 | 0.6103 | 0.8510 | 0.6397 |
| 20 | 3.04e-05 | 0.0785 | 0.9680 | 0.6673 | 0.8627 | 0.6606 |
| **21** | 2.58e-05 | 0.0658 | 0.9729 | 0.6613 | 0.8691 | **0.6614** |
| 22 | 2.14e-05 | 0.0565 | 0.9783 | 0.7358 | 0.8679 | 0.6512 |
| 23 | 1.74e-05 | 0.0503 | 0.9801 | 0.8130 | 0.8707 | 0.6581 |
| 24 | 1.37e-05 | 0.0438 | 0.9827 | 0.7900 | 0.8686 | 0.6536 |
| 25 | 1.05e-05 | 0.0414 | 0.9847 | 0.8303 | 0.8748 | 0.6612 |
| 26 | 7.63e-06 | 0.0364 | 0.9859 | 0.8743 | 0.8688 | 0.6528 |
| 27 | 5.28e-06 | 0.0334 | 0.9865 | 0.8690 | 0.8695 | 0.6540 |
| 28 | 3.42e-06 | 0.0321 | 0.9877 | 0.8857 | 0.8729 | 0.6554 |
| 29 | 2.08e-06 | 0.0307 | 0.9881 | 0.9051 | 0.8711 | 0.6524 |

Epoch 6 is where the backbone unfreezes — train loss drops 0.697 → 0.457 and val macro-F1 jumps 0.449 → 0.517 immediately.

### 3.2 The model overfits from epoch 13

Validation loss bottoms at **0.4840** (epoch 13) then climbs monotonically to **0.9051** (epoch 29) — it nearly doubles. Across that same span validation macro-F1 moves only from 0.6322 to 0.6524, peaking at 0.6614.

Training accuracy reaches **0.9881** against validation **0.8711**, a gap of 11.7 points.

The run spent 16 of 29 epochs growing more confident on the training set without learning anything transferable. **Selecting on macro-F1 rather than loss is what preserved a usable checkpoint.** Had selection been on validation loss the run would have stopped at epoch 13 with a lower F1; had it been on accuracy it would have taken epoch 25, deeper into the overfit.

### 3.3 Test macro-F1 exceeds validation macro-F1

Test macro-F1 (0.7075) comes in **above** best validation macro-F1 (0.6614). A gap in that direction usually indicates an unrepresentatively easy test split.

Here it is V2 noise. The head scored differently on its 6 test patches than on its 6 validation patches, and because V2 carries a quarter of the macro average, two or three patches move the headline by several points. This is not evidence of generalisation. It is a direct demonstration of why the four-head macro figure is unusable.

### 3.4 Class weights

Inverse frequency, `w_c = N / (K · n_c)`, normalised to mean 1.

| head | train patches | weight |
|---|---|---|
| V1 | 3,987 | 0.0473 |
| V2 | 48 | **3.9283** |
| V3 | 15,141 | 0.0125 |
| V4 | 15,833 | 0.0119 |

The V2:V4 weight ratio is 330:1. That is what produced V2's recall of 0.8333 at precision 0.1923 — the loss was instructed to care about 48 examples as much as about 15,833.

---

## 4. Data provenance

This section contains the finding with the longest reach, and it changes how every count in this report should be read.

### 4.1 The corpus is 4.6× smaller than its file count

Roboflow encodes the originating frame in each filename as `<stem>.rf.<export hash>.jpg`. Scoring perceptual hashing against that identity on 8,000 `final_welding` training images:

| | |
|---|---|
| phash precision vs filename identity | **0.9699** |
| phash recall | **0.1892** |
| pairs: TP / FP / FN | 18,764 / 582 / 80,397 |

phash almost never merges two genuinely different frames — but on its own recovers only 19% of augmented siblings, because rotation and shear push a sibling well past Hamming distance 5. In that 8,000-image sample, **903 distinct source frames** account for all 8,000 files.

Filename identity, MD5 and phash were therefore combined into a single union-find over source images, applied **before** cropping.

| stage | count |
|---|---|
| source image files on disk | 27,377 |
| exact-duplicate groups (MD5) | 1,098 |
| perceptual groups (phash ≤ 5) | 2,223 |
| — within-dataset | 1,542 |
| — cross-dataset | 681 |
| images dropped as siblings | 21,469 (78.4%) |
| **unique weld frames** | **5,908** |

For comparison, the previous MD5-only pass found 181 duplicates and concluded the three datasets were "essentially disjoint". They are not — 681 perceptual groups span datasets.

`results/review/dedup_verification.png` shows the grouping visually: every member of a group shares one base filename before the `.rf.` suffix.

### 4.2 Effect on instance counts

| head | focal | before dedup | after dedup | retained |
|---|---|---|---|---|
| V1 | `{P}` | 52,845 | 6,206 | 11.7% |
| **V2** | `{C}` | **2,659** | **60** | **2.3%** |
| V3 | `{N,U}` | 65,112 | 19,352 | 29.7% |
| V4 | `{N,U,L}` | 109,675 | 14,276 | 13.0% |
| discard | — | 2,907 | 1,213 | 41.7% |
| **total** | | **233,198** | **41,107** | **17.6%** |

The suspicion that 2,659 crack instances was implausibly high for GMAW was **correct**, but the mechanism was not mislabelling across many frames — it was a small number of frames exported roughly 44 times each. There are 60 crack instances across 46 unique frames.

### 4.3 Label conflicts

867 groups of **byte-identical** images carrying differing instance→head compositions. Every copy is dropped, never first-wins. Written to `data/merged/label_conflicts.json` with full paths and per-image head compositions.

Conflict detection deliberately applies to MD5-exact duplicates only. Applied to perceptual siblings it fired on ~2,000 groups, because Roboflow's rotate and crop legitimately change which instances survive in a frame — a composition difference between augmented siblings is expected, not contradictory. Restricting to exact duplicates preserves the original MD5-based semantics.

---

## 5. Patch corpus

43,761 patches from 5,908 unique frames. One patch per annotation instance, labelled by that instance's own class.

### 5.1 Per head and split

| head | focal | train | val | test | total |
|---|---|---|---|---|---|
| V1 | `{P}` | 3,987 | 498 | 499 | 4,984 |
| V2 | `{C}` | 48 | 6 | 6 | 60 |
| V3 | `{N,U}` | 15,141 | 1,893 | 1,892 | 18,926 |
| V4 | `{N,U,L}` | 15,833 | 1,979 | 1,979 | 19,791 |
| **total** | | **35,009** | **4,376** | **4,376** | **43,761** |
| source frames | | 3,242 | 1,334 | 1,332 | 5,908 |

Head imbalance ratio 329.9:1 (V4 19,791 vs V2 60).

Note the frame counts: train holds 3,242 frames but 80% of the patches, because frames differ greatly in instance density. The greedy grouped assignment fills head quotas by patch count while keeping every frame whole, so a few dense frames carry disproportionate patch mass into train.

### 5.2 Full composition by source class

**V1 `{P}` — 4,984**

| source | class | patches |
|---|---|---|
| weld_dataset | pore | 2,699 |
| final_welding | Porosity | 2,165 |
| final_welding | cluster of porosity | 120 |

**V2 `{C}` — 60**

| source | class | patches |
|---|---|---|
| final_welding | Crack | 60 |

**V3 `{N,U}` — 18,926**

| source | class | patches |
|---|---|---|
| weld_dataset | discontinuity | 6,718 |
| final_welding | Overlap | 6,162 |
| final_welding | Lack of Fusion | 3,278 |
| final_welding | Irregular | 1,194 |
| final_welding | Burn Through | 1,049 |
| final_welding | Excess-Reinforcement | 400 |
| final_welding | Undercut | 100 |
| final_welding | root concvity | 15 |
| final_welding | excess root penetration | 10 |

**V4 `{N,U,L}` — 19,791**

| source | class | patches |
|---|---|---|
| weld_dataset | stain | 7,282 |
| final_welding | Spatters | 2,819 |
| weld_dataset | `__negative__` | 2,793 |
| final_welding | `__negative__` | 2,696 |
| weld_dataset | deposit | 2,625 |
| wijaya | Good Weld | 689 |
| final_welding | Arc strike | 496 |
| wijaya | `__negative__` | 360 |
| final_welding | mechanical-damage | 31 |

**Discarded:** wijaya `Defect` 680, wijaya `Bad Weld` 455, final_welding `slag-inclusion` 78 — 1,213 instances.

### 5.3 What the remap recovered

**22,781 instances** that the previous four-class taxonomy discarded now carry evidence: Overlap, Burn Through, Irregular, Undercut, Excess-Reinforcement, root concavity and excess root penetration into V3; Spatters, stain, deposit, Arc strike and mechanical-damage into V4.

The single-source monoculture is also fixed. The old `no_defect` class was 437 images, 100% Wijaya. V4 is now 19,791 patches drawn 64% from `weld_dataset`, 31% from `final_welding`, 5% from `wijaya`.

| | old 4-class | new focal elements |
|---|---|---|
| source images | 27,377 | 5,908 unique frames |
| training units | 10,698 whole images | 43,761 per-instance patches |
| smallest class | no_defect 437 | V2 60 |
| imbalance ratio | 9.8:1 | 329.9:1 |
| smallest class sources | 1 (Wijaya only) | V2: 1 (final_welding only) |

### 5.4 Cropping and geometry

- **Context margin 1.75×** around each annotation box, clipped to image bounds.
- **Letterboxed** to 224×224 preserving aspect, padded by edge replication. A flat fill would inject a synthetic edge the network learns; stretching a 3.7:1 crop to square destroys bead geometry, which is precisely what V3 depends on.
- **Mean padding fraction 0.305.** 9,002 patches (20.6%) are more than half padding.
- **1,982 degenerate boxes skipped** (under 8 px after expansion).
- Segmentation polygons reduce to their axis-aligned bounding box. Only field 0 of a label line is trusted for class ID, so mixed 5-field bbox and variable-length polygon lines parse identically.

### 5.5 Negative sampling

5,849 V4 patches are negatives sampled from annotation-free regions of the same images. Box sizes are drawn from the positive size distribution:

| | median box area |
|---|---|
| positives | 277 px² |
| negatives | 203 px² |

A 1.4× separation. `tests/test_vision_splits.py` fails if the medians drift more than 4× apart, because systematically larger or smaller negatives would let the network learn "crop size predicts class" with nothing in the metrics to reveal it.

---

## 6. Errors requiring attention

### 6.1 CRITICAL — V2 is not supported by the data

60 patches from 46 frames: 48 train, 6 validation, 6 test, all from `final_welding`. Precision 0.1923 (5 true, 21 false). Class weight 3.93× against 0.0119× for V4.

Any macro average over four heads inherits this head's variance. A single patch changing outcome moves the four-head macro-F1 by roughly 2 points.

**Needs a decision.** Options:
1. Fold `Crack` into V3, giving a defensible three-head model.
2. Keep V2 and report it explicitly as unsupported with n=6.
3. Source additional surface-crack data — the only route to a real V2 head.

`results/review/crack_sample.png` shows 50 sampled instances: mostly whole bead runs with boxes that appear image-level rather than defect-level, with only a handful of genuine dark linear features. **This mapping has not been confirmed and I have not decided it.**

### 6.2 HIGH — Overfitting after epoch 13

Validation loss 0.4840 → 0.9051 over the final 16 epochs while macro-F1 stayed flat. Train accuracy 0.9881 vs validation 0.8711.

With 5,908 effective frames behind 43,761 patches, model capacity exceeds what the data supports. Suggested: stop at ~15 epochs; raise weight decay above 1×10⁻⁴; strengthen augmentation; test a smaller backbone.

### 6.3 HIGH — Headline metric dominated by a 6-sample head

Test macro-F1 (0.7075) exceeds best validation macro-F1 (0.6614), the reverse of the usual direction, traceable entirely to V2.

**Report as** per-head F1 with support counts plus a three-head macro average (0.8392), never a single four-head number.

### 6.4 MEDIUM — V1 precision 0.7316

75 V3 and 67 V4 patches read as porosity; V1 loses 111 patches to V3/V4. Worth testing a larger context margin than 1.75×, or higher input resolution for V1-scale features.

### 6.5 MEDIUM — V3 ↔ V4 exchange of 315 patches

Expected by construction — the two focal elements differ by one subsurface symbol with no surface signature. Recorded rather than flagged as a defect, but it bounds how much the fusion stage can rely on V3/V4 separation alone.

### 6.6 MEDIUM — 20.6% of patches are majority padding

9,002 of 43,761 exceed 0.5 padding fraction. Highly elongated annotation boxes — long toe lines, bead-length defects — produce crops that are mostly replicated edge. `pad_fraction` is in the manifest for auditing. Consider aspect-aware tiling for boxes beyond ~3:1 instead of one padded crop.

### 6.7 OPEN — Three class mappings unconfirmed

| mapping | patches | status |
|---|---|---|
| `weld_dataset/discontinuity` → V3 | 6,718 | **PROVISIONAL.** Largest single contributor to V3. Contact sheet shows bead edges and toe profiles — consistent with a geometric reading, nothing crack-like. Not human-confirmed. |
| `final_welding/Crack` → V2 | 60 | **PROVISIONAL.** See 6.1. |
| `final_welding/cluster of porosity` → V1 | 120 | **ADDED DURING IMPLEMENTATION.** One of the 15 classes in that source's `data.yaml` but absent from the mapping specification. Unmapped classes raise rather than silently discard, so the pipeline could not run without a decision. Treated as porosity. |

All three live in `configs/class_map.yaml`. Changing any is one line plus a re-run.

### 6.8 OPEN — Calibration not fitted

Raw pre-softmax logits are persisted for validation and test, shape `(4376, 4)` with head names attached. Temperature scaling has **not** been fitted, so no calibrated probabilities exist and no BPA has been constructed. The fusion stage is blocked until this runs.

### 6.9 OPEN — Single fold

Fold 0 only. With V2 at 6 test patches and V3/V4 separated by a genuinely ambiguous boundary, one fold gives no variance estimate on any head. Grouped k-fold on `source_frame` is needed before write-up.

---

## 7. What the model is ready for

### Ready

- **Discriminating V3 from V4** — geometric anomaly against no conclusive surface evidence — at F1 0.8776 and 0.8870 across 3,871 test patches. This is the discrimination the vision modality exists to contribute.
- **Feeding V1, V3 and V4 mass into the fusion stage**, once temperature scaling is fitted. Three-head macro F1 0.8392.
- **Serving as the reproducible baseline** for architecture and augmentation experiments. Config hashes, git SHA, seed, class weights, patch counts and dedup counts are all recorded in `run_manifest_fold0.json`.
- **Demonstrating that the compound focal element works.** V3 `{N,U}` is learnable at 0.88 without the model ever being asked whether a deviation breaches ISO 5817. The division of labour with the geometry modality holds up empirically — which is a thesis result in its own right, independent of the headline number.

### Not ready

- **Any quoted four-head macro-F1.** 0.7075 averages one meaningless term.
- **Emitting mass on V2 `{C}`.** At precision 0.19 on a singleton focal element, false crack mass commits hard under Dempster's rule and propagates a confident wrong answer into the fused decision. This is the most consequential limitation in the report.
- **Any claim about crack detection**, in either direction. 60 instances from 46 frames supports no statement.
- **Write-up of results.** Two mappings unconfirmed, one fold, no calibration, a known overfit. All four are addressable; none is addressed.

---

## 8. Artefacts

| path | contents |
|---|---|
| `results/vision/metrics_fold0.json` | per-head P/R/F1, confusion matrix, 29-epoch history |
| `results/vision/run_manifest_fold0.json` | git SHA, config/frame/class_map hashes, seed, class weights, patch counts, dedup counts |
| `results/vision/logits_fold0.npz` | raw validation logits (4376×4) + labels + head names — input to temperature scaling |
| `results/vision/logits_test_fold0.npz` | raw test logits |
| `results/review/crack_sample.png` | 50 `Crack` instances — mapping decision pending |
| `results/review/discontinuity_sample.png` | 50 `discontinuity` instances — mapping decision pending |
| `results/review/dedup_verification.png` | perceptual groups shown as augmented siblings |
| `data/merged/label_conflicts.json` | 867 exact-duplicate groups with contradictory labels |
| `data/processed/vision_patches/manifest.csv` | 43,761 rows: patch path, head, source dataset, source image, source frame, source label, box geometry, is_negative, pad_fraction, split |
| `data/processed/vision_patches/split_summary.json` | per-head per-split counts |
| `data/processed/vision_patches/dedup_report.json` | dedup group and removal counts |
| `models/vision/best_resnet50_fold0.pth` | checkpoint selected at epoch 21 on validation macro-F1 |

`.gitignore` carries an exception so `results/vision/*.npz` and `*.json` are tracked — the repo's blanket `*.npz` rule would otherwise exclude the logits the calibration stage depends on.

---

## 9. Reproduction

```bash
~/weldenv/bin/python -m pytest tests/ -q                 # 32 passed
~/weldenv/bin/python src/vision/convert_and_remap.py     # 27,377 -> 5,908 frames, 43,761 patches
~/weldenv/bin/python src/vision/merge_dedup_split.py     # grouped split on source_frame
~/weldenv/bin/python src/vision/train.py --config configs/vision_train.yaml
~/weldenv/bin/python src/vision/evaluate.py --model models/vision/best_resnet50_fold0.pth
```

Environment: WSL venv at `~/weldenv`. The default PyTorch wheel targets CUDA 13 and reports `cuda: False` against this machine's 12.7 driver — install from the `cu124` index.

`--keep-augmented-siblings` on `convert_and_remap.py` retains all 233,198 instances instead of collapsing each frame to one representative. Siblings share a `source_frame`, so the grouped split still cannot leak; it inflates class weights and apparent dataset size by the export multiplier without adding information.

### Test coverage

32 tests, all passing. Notable assertions:

- no `source_frame` appears in more than one split (the leakage guard)
- no `source_image` appears in more than one split
- `validate_modality_bpa()` rejects mass on `{U}` alone and `{L}` alone
- `validate_modality_bpa()` **accepts** `{P}` and `{C}` — guarding against someone tightening the rule to "no singletons"
- every class in each source `data.yaml` has a mapping in `class_map.yaml`
- unmapped classes raise rather than defaulting to discard
- negatives are V4-only
- negative box areas stay within 4× of the positive median
- model head count comes from the frame, not a hardcoded constant

---

## 10. Recommended next actions

1. **Decide the `Crack` → V2 mapping** from `results/review/crack_sample.png`. This gates everything downstream, because it determines whether the model has three heads or four.
2. **Confirm `discontinuity` → V3** from `results/review/discontinuity_sample.png`. 6,718 patches — the largest single contributor to V3 — rest on it.
3. **Confirm `cluster of porosity` → V1.**
4. **Re-run with early stopping around epoch 15** and higher weight decay, to address the overfit.
5. **Fit temperature scaling** on `logits_fold0.npz`, then construct the BPA through `src/bpa/construct.py`.
6. **Run grouped k-fold** on `source_frame` for variance estimates per head.

Steps 1–3 are one-line changes in `configs/class_map.yaml` plus a re-run. Nothing in this report should be written up until they are settled.
