# Leave-One-Source-Out — Transfer to an Unseen Dataset

**Date:** 1 October 2026 · **Backbone:** ResNet-50 · **Seed:** 42
**Frame:** V1 `{P}`, V3 `{N,U}`, V4 `{N,U,L,C}` — three heads
**Every macro figure is a three-head macro.**

---

## The transfer gap

**In-distribution macro-F1 of roughly 0.91 falls to 0.43 and 0.22 when a whole
source dataset is held out. The gap is between 46 and 72 points.**

| held out | in-distribution¹ | held-out | **gap** | accuracy | ECE | n test |
|---|---|---|---|---|---|---|
| `weld_dataset` | 0.8893 | **0.4314** | **−0.4579** | 0.5584 | 0.3392 | 22,117 |
| `final_welding` | 0.9390 | **0.2219** | **−0.7171** | 0.2650 | 0.6846 | 20,595 |
| `wijaya` | 0.9092 | — degenerate — | — | 0.5033 | 0.4068 | 1,049 |

¹ best validation macro-F1 during that fold's own training, with validation
drawn from the two retained sources. This is the correct comparator: the same
model, measured on held-out data from the distribution it trained on, against
held-out data from a distribution it never saw.

For three balanced classes, a macro-F1 of 0.2219 is **worse than chance**.

This is the one measurement in the whole programme that sequence leakage cannot
inflate. The folds train and test on disjoint datasets by construction, so no
near-duplicate can span the boundary. Every in-distribution figure reported
elsewhere — fold 0, Task C, Task H — sits on a split with 81.59% sequence
sharing. These do not.

---

## Per-fold results

### Fold 1 — hold out `weld_dataset`

Train 19,676 / val 1,968 / test 22,117. Ran 18 epochs, best at 17.
Loses 54.2% of V1, 35.5% of V3 and 64.0% of V4 training patches.

| head | precision | recall | F1 | support |
|---|---|---|---|---|
| V1 `{P}` | 0.1649 | 0.1312 | **0.1461** | 2,699 |
| V3 `{N,U}` | 0.5304 | 0.4396 | 0.4807 | 6,718 |
| V4 `{N,U,L,C}` | 0.6280 | 0.7121 | 0.6674 | 12,700 |

### Fold 2 — hold out `final_welding`

Train 20,911 / val 2,255 / test 20,595. Early-stopped at 17, best at 12.
Loses 45.8% of V1, 64.5% of V3 and 30.7% of V4 training patches.

| head | precision | recall | F1 | support |
|---|---|---|---|---|
| V1 `{P}` | 0.2756 | 0.2105 | 0.2387 | 2,285 |
| V3 `{N,U}` | 0.2449 | **0.0166** | **0.0311** | 12,208 |
| V4 `{N,U,L,C}` | 0.2649 | 0.7824 | 0.3958 | 6,102 |

### Fold 3 — hold out `wijaya` — DEGENERATE

Train 38,562 / val 4,150 / test 1,049. Ran 18 epochs, best at 16.

**All 1,049 held-out patches are V4.** With a single class in the test set there
are no true negatives, so precision, F1 and any macro average are **undefined**,
not merely poor. Only recall is computable.

| | value |
|---|---|
| V4 recall | **0.5033** |
| precision / F1 / macro | undefined |

This fold is also barely a holdout — removing `wijaya` costs only 5.3% of V4's
training data and 0% of V1 and V3 — while being the most expensive to run,
since training keeps 97.6% of the corpus. It is reported for completeness and
should carry little weight.

---

## The V3 collapse is the most important result here

**Holding out `final_welding`, V3 scores F1 0.0311 at recall 0.0166 — the model
identified 203 of 12,208 V3 patches.**

Trained on `weld_dataset`, every V3 patch the model ever saw was
`discontinuity`. Asked to recognise `final_welding`'s `Overlap`,
`Lack of Fusion`, `Irregular`, `Burn Through`, `Undercut` and
`Excess-Reinforcement` as the same focal element, it recognised almost none of
them.

This bears directly on the `discontinuity → V3` mapping, though not in the way
the earlier evidence suggested. Two results must be held together:

| evidence | value | reading |
|---|---|---|
| V3 F1 within `final_welding` only (no `discontinuity` present) | **0.9231** on 771 patches | each source's V3 is learnable on its own |
| V3 F1 on `final_welding` when trained without it | **0.0311** on 12,208 patches | the two are nearly untransferable to each other |

The mapping is not "propping up" the head — `final_welding`'s V3 stands alone at
0.9231 when `final_welding` is in training. But the premise that
`weld_dataset/discontinuity` and `final_welding/Overlap` are *the same focal
element* is not supported by what the model learns. They share almost no
transferable features.

**This does not mean the mapping is wrong.** Two defects can be the same
category under ISO 5817 and still look nothing alike across a camera, a working
distance and a surface finish. It does mean the single-head mapping is carrying
an assumption the data does not corroborate, and that should be stated in the
thesis rather than assumed.

V1 shows the same pattern in both directions — F1 0.1461 and 0.2387. `pore` and
`Porosity` are not interchangeable to this model either.

V4 degrades least (0.6674 and 0.3958), which is expected of an ignorance head
whose job is mostly to recognise the absence of conclusive evidence.

---

## Calibration — the model is confidently wrong

Expected calibration error, 15 equal-width bins over max-softmax confidence:

| held out | ECE | patches at confidence ≥ 0.8 | mean confidence there | mean accuracy there |
|---|---|---|---|---|
| `weld_dataset` | 0.3392 | 17,457 (78.9%) | 0.965 | 0.592 |
| `final_welding` | **0.6846** | 18,546 (90.1%) | **0.984** | **0.256** |
| `wijaya` | 0.4068 | 864 (82.4%) | 0.966 | 0.542 |

On fold 2, **90% of predictions are made at a mean confidence of 0.984 while
being right 25.6% of the time.** The model does not signal that it has left its
distribution. It fails silently and confidently.

This is the specific failure mode Dempster-Shafer fusion is meant to defend
against, and it is why the discount matters.

### Implication for the Shafer discount α

The planned BPA construction is `m(F_i) = (1−α)·p_i` with `m(Θ) = α`, and the
planned sensitivity sweep was α ∈ [0.05, 0.30].

**That range is too narrow.** An ECE of 0.68 on an unseen source means the
calibrated probabilities are wrong by roughly that much in expectation, and the
October RealSense coupon images are themselves a fourth unseen source. A
discount that leaves 70–95% of the mass on the focal elements would propagate
confident error into the fused decision.

The LOSO ECE range of 0.34–0.68 is the empirical basis for α. The sweep should
extend to at least 0.70, and the operating point should be chosen from the
worst fold, not the mean — the coupon images have no reason to resemble the
easier of the two.

---

## Method

Each fold trains from ImageNet initialisation on two sources and tests on the
third, using the Task H configuration unchanged: 18 epochs, patience 5, weight
decay 1e-3, permanent freeze through `layer2`, cosine LR from 1e-4, runtime
augmentation, inverse-frequency class weights recomputed on that fold's
training split, model selection on validation macro-F1.

Validation is drawn at **frame granularity** from the two retained sources
(10%, seed 42), so frames stay whole and the in-distribution comparator is
itself leakage-guarded at the `source_frame` level.

Artefacts: `loso_metrics.json` (per-fold metrics, ECE bins, per-epoch history),
`logits_loso_resnet50.parquet` (per-patch logits with join keys).

---

## Caveats, stated plainly

**LOSO here tests the class map as much as visual transfer.** The three sources
have **completely disjoint label vocabularies** — no source label appears in more
than one dataset. Holding out `weld_dataset` removes `pore`, `discontinuity`,
`stain` and `deposit` entirely; the model has never seen any of them. So this is
not "same defects, different camera" but "different annotation vocabularies that
`class_map.yaml` asserts are the same focal elements". A large gap is partly a
statement about that assertion.

**Folds 1 and 2 are near-complementary halves.** Each loses between a third and
two-thirds of every head's training evidence, so some of the drop is reduced
training data rather than distribution shift. The two cannot be separated with
three datasets.

**Only two folds are informative.** The `wijaya` fold is single-class and
yields one recall figure.

**One supporting claim did not reproduce.** The `discontinuity → V3`
justification in `configs/class_map.yaml` cites a geometry-only probe predicting
source dataset at 0.790 against a 0.522 baseline. Measured on this corpus:
**accuracy 0.4909 against a 0.6661 majority baseline — a negative lift of
−0.1752**, macro-F1 0.3404. Median annotation box areas are also near-identical
between the two main sources (270 px² for `weld_dataset`, 267 px² for
`final_welding`), not the 224 vs 8,451 px² cited. Those two claims should not be
repeated without re-deriving them. See `retrain_report.md`.

---

## What to carry into the thesis

The headline is **not** the 0.91 in-distribution figure. It is the gap.

> Held out from an entire source dataset, three-head macro-F1 falls from ~0.91
> to 0.43 and 0.22, with expected calibration error of 0.34 and 0.68. The model
> does not currently transfer to an unseen imaging source, and does not signal
> when it has failed.

That is a reportable, defensible result about the limits of the available data,
and it sets the discount for the fusion stage honestly. A thesis that measures
this and says so is stronger than one that reports 0.91 and is asked about
transfer in the viva.

### Next steps it implies

1. **Set α from the worst LOSO fold**, with the sweep extended to 0.70.
2. **Temperature scaling will not fix this.** Temperature is fitted on
   in-distribution validation data; it cannot correct a 0.68 ECE on a
   distribution it never saw. Report in-distribution ECE after scaling and the
   LOSO ECE separately.
3. **Treat the coupon images as a fourth source from the start** — collect and
   annotate enough of them to fine-tune on, rather than expecting transfer.
4. **Reconsider whether one V3 head can span all sources**, or whether the
   thesis should report V3 per-source and fuse later.
