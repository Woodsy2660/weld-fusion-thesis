# Three-Head Retrain — Task C and Task H

**Date:** 1 October 2026 · **Seed:** 42 · **Device:** RTX 2070, torch 2.6.0+cu124
**Frame:** V1 `{P}`, V3 `{N,U}`, V4 `{N,U,L,C}` — three heads, V2 removed
**Every macro figure below is a three-head macro.** No four-head macro appears.

---

## Headline

**The Task C → Task H comparison does not measure the size of the leak, and
should not be presented as if it does.**

Task H's test macro-F1 came out *higher* than Task C's (0.9120 against 0.9046),
which on its face suggests the sequence leakage cost nothing. It does not mean
that. The two runs have different test sets with different compositions,
because re-grouping on `source_sequence` made stratification impossible — V3's
test split moved from 35.5% `discontinuity` (matching the corpus exactly) to
55.0%, and `discontinuity` is the easiest V3 label.

The comparable signal is validation, where each run holds out under its own
grouping: **0.9187 → 0.8738, a drop of 0.0449.** Even that is not a clean
measurement, because three things changed at once.

---

## Comparison

| run | grouping key | stratified | best val macro-F1 | test macro-F1 | test accuracy |
|---|---|---|---|---|---|
| fold 0 (four-head) | source_frame | no | 0.6614 | 0.7075¹ | 0.8652 |
| **Task C** ResNet-50 | source_frame | yes | **0.9187** | 0.9046 | 0.9306 |
| Task C ResNet-18 | source_frame | yes | 0.9128 | 0.9032 | 0.9290 |
| **Task H** ResNet-50 | source_sequence | no² | **0.8738** | 0.9120 | 0.9328 |
| Task H ResNet-18 | source_sequence | no² | 0.8814 | 0.9023 | 0.9256 |

¹ fold 0's figure is a **four-head** macro over a frame that included V2 `{C}`,
a head with 6 test patches and precision 0.19. It is shown for continuity only
and is **not comparable on equal terms** — different frame, different split,
different head count. Its checkpoint will not even load under the current
`frame.yaml`.

² Task A's ±5 pp acceptance bound could not be met on `source_sequence`
grouping; 11 of 16 checked pairs fall outside it, max drift 19.5 pp.

### Per head

| run | V1 `{P}` | V3 `{N,U}` | V4 `{N,U,L,C}` |
|---|---|---|---|
| Task C ResNet-50 | 0.8246 (n=490) | 0.9594 (n=1919) | 0.9298 (n=2002) |
| Task C ResNet-18 | 0.8227 (n=490) | 0.9541 (n=1919) | 0.9326 (n=2002) |
| Task H ResNet-50 | 0.8421 (n=590) | 0.9524 (n=1715) | 0.9415 (n=2262) |
| Task H ResNet-18 | 0.8241 (n=590) | 0.9442 (n=1715) | 0.9387 (n=2262) |

Note the supports differ between Task C and Task H. The test sets are not the
same patches.

---

## Why the test figure rose

| | Task C test | Task H test | corpus |
|---|---|---|---|
| V3 `discontinuity` | 35.5% | **55.0%** | 35.5% |
| V3 `Overlap` | 33.5% | 22.3% | 32.6% |
| V3 `Lack of Fusion` | 16.2% | 9.0% | 17.3% |
| V1 `pore` | 50.2% | 44.9% | 54.2% |
| V1 `Porosity` | 47.8% | 54.4% | 43.4% |
| V4 `stain` | 38.0% | 47.9% | 36.7% |

Task A's stratification on `source_frame` reproduced the corpus almost exactly.
Task H's could not, and V3's test set shifted nearly 20 points toward
`discontinuity`, which scored the highest per-label recall of any V3 source in
the fold-0 analysis (0.9575).

**Three things changed between Task C and Task H simultaneously:**

1. sequence siblings no longer span the split — the intended change;
2. test-set composition drifted up to 19.5 pp — an unintended consequence of
   grouping at sequence level;
3. the test set is a different size and sampling of the corpus (4,567 patches
   against 4,411).

The leak is plausibly the dominant term in the validation drop, but these
cannot be separated from one another with the data available. Attributing the
whole −0.0449 to leakage would overstate what was measured.

---

## Overfit evidence

Config changes from fold 0, all aimed at the epoch-13 overfit where validation
loss ran 0.4840 → 0.9051 while macro-F1 stayed flat: 18 epochs (was 30),
patience 5 (was 8), weight decay 1e-3 (was 1e-4), permanent freeze through
`layer2` (was: all frozen 5 epochs then all trainable), and runtime
augmentation restored.

| run | epochs run | best epoch | final train acc | final val acc | val loss at best |
|---|---|---|---|---|---|
| Task C ResNet-50 | 18 | 18 | 0.9754 | 0.9312 | 0.2792 |
| Task C ResNet-18 | 18 | 16 | — | — | — |
| Task H ResNet-50 | 18 | 16 | — | — | — |
| Task H ResNet-18 | 18 | 18 | — | — | — |

**The overfit is controlled.** Task C ResNet-50 reached its best validation
macro-F1 at the final epoch and never triggered early stopping, so the 18-epoch
cap bound rather than patience. Validation loss did not run away as it did in
fold 0 — it sat at 0.2792 at the selected epoch against fold 0's 0.9051. The
train/val accuracy gap is 4.4 points (0.9754 vs 0.9312) against fold 0's 11.7.

Full per-epoch histories are in `metrics_<tag>.json` under `history`.

---

## Backbone choice

| | parameters | trainable | Task C test | Task H test |
|---|---|---|---|---|
| ResNet-50 | 23,514,179 | 22,069,251 | 0.9046 | 0.9120 |
| ResNet-18 | 11,178,051 | 10,494,979 | 0.9032 | 0.9023 |

**ResNet-18 matches ResNet-50 to within 0.0014 on Task C and 0.0097 on Task H,
at 48% of the parameters.** On a corpus of roughly 5,908 independent source
frames — and fewer still once sequences are accounted for — that is a finding
rather than a concession. It halves the overfit pressure and the training cost
of every experiment that follows.

ResNet-50 was carried into LOSO only because it scored marginally higher on
Task H. If LOSO shows no meaningful separation, ResNet-18 should be adopted.

---

## Geometry re-check (Task E.2 gate)

Matching is joint — box-area decile × pad-fraction tercile — because box area
carried more mutual information with the head than pad_fraction did (0.26 bits
against 0.07 on the fold-0 test split), so matching on padding alone would
leave the larger confound in place.

**ResNet-50 v2 — GATE PASS.** Matched subsample 3,302 of 4,567.

| head | unmatched F1 | matched F1 | delta |
|---|---|---|---|
| V1 | 0.8421 | 0.8740 | +0.0319 |
| V3 | 0.9524 | 0.9506 | **−0.0017** |
| V4 | 0.9415 | 0.9261 | **−0.0154** |
| three-head macro | 0.9120 | 0.9169 | +0.0049 |

**ResNet-18 v2 — GATE PASS.**

V3 and V4 both move by far less than the 0.05 gate. Crop geometry is not
driving the V3/V4 discrimination.

**Caveat.** This matching is performed within the Task H test split, which
itself carries up to 19.5 pp of composition drift. The matched subsample is
therefore drawn from an already-skewed pool, which makes the PASS a weaker
statement than the equivalent check on the Task C split would have been.

---

## Supporting evidence for the class mapping

### V3 does not rest on the `discontinuity` mapping

`weld_dataset/discontinuity` is the single largest contributor to V3, so the
question is whether V3's headline F1 depends on it. `final_welding` contains no
`discontinuity` patches at all, which makes V3 restricted to `final_welding`
the clean test.

| scope | V3 F1 | n | three-head macro | contains `discontinuity`? |
|---|---|---|---|---|
| all sources | 0.9524 | 1,715 | 0.9120 | yes |
| **final_welding only** | **0.9231** | **771** | 0.8824 | **no** |
| weld_dataset only | 0.9766 | 944 | 0.9166 | yes |

**V3 scores 0.9231 on 771 `final_welding` patches with no `discontinuity`
present.** The head stands on its own evidence. This supports the
`discontinuity → V3` mapping, and more strongly than the 0.829 figure cited
when the decision was taken.

### Two premises behind the mapping decision did NOT reproduce

The `discontinuity → V3` confirmation recorded in `configs/class_map.yaml`
cites two supporting figures. Neither reproduces on this corpus, and that is
recorded here rather than quietly dropped.

| claim in `class_map.yaml` | measured here |
|---|---|
| crop geometry predicts source dataset at **0.790** vs a **0.522** baseline | accuracy **0.4909** vs majority baseline **0.6661** — a **negative** lift of −0.1752, macro-F1 0.3404 |
| median box area **224 px²** (weld_dataset) vs **8,451 px²** (final_welding) | **270 px²** vs **267 px²** — essentially identical |

On this corpus a geometry-only probe performs *worse* than predicting the
majority class, and the two main datasets have near-identical median annotation
box areas. The outlier is `wijaya` at 7,170 px².

The conclusion the mapping rests on is unaffected — the within-`final_welding`
V3 result above is stronger evidence than either of these — but the specific
claim that "the UMAP island is dataset identity because geometry predicts
source at 0.790" is **not supported by measurements on this corpus** and should
not be repeated in the thesis without re-deriving it.

---

## Artefacts

```
metrics_{resnet50,resnet18,resnet50_v2,resnet18_v2}.json
run_manifest_{resnet50,resnet18,resnet50_v2,resnet18_v2}.json
logits_{val,test}_{resnet50,resnet18,resnet50_v2,resnet18_v2}.parquet
geometry_recheck_{resnet50_v2,resnet18_v2}.json
mapping_support.json
split_composition.md          (Task A, source_frame grouping)
split_composition_v2.md       (Task G2, source_sequence grouping)
```

Logits are parquet carrying `patch_path`, `source_frame`, `source_dataset`,
`source_label`, `split`, `pad_fraction`, `box_area` and `aspect_ratio` beside
the logit columns, so every downstream analysis can join back to the manifest.
The fold-0 `.npz` artefacts are four-head and incompatible; no new code reads
them.

---

## Status

Leave-one-source-out is running at the time of writing and its results are
**not** in this report. See `loso_report.md` when it lands. LOSO trains and
tests on disjoint datasets by construction, so it is the one measurement that
sequence leakage cannot inflate, and the transfer gap it reports — not any
figure in this document — is the honest basis for the Shafer discount α on the
vision BPA.
