# 04 — Vision module

## What it outputs

A BPA over the vision focal elements, per cell:

```
{P}            V1 output
{N,U}          V3 output
{N,U,L}        V4 output
Θ              ignorance
```

A three-way softmax classifier (one output per element, single cross-entropy).
A fourth `{C}` output existed in an earlier design and was **removed**; C has
since left the experimental frame entirely (see `configs/frame.yaml`).

Softmax over the three outputs, temperature-scaled, then mapped to mass. Mass on Θ
comes from low confidence, occlusion, or a cell with insufficient valid pixels.

## The leakage finding — the most important thing in this module

Three public weld-image corpora were aggregated, relabelled and split. The resulting
in-distribution figures looked respectable. A sequence audit then found:

> **81.59% of samples share a source sequence with a sample in another split.**

**Every in-distribution figure from that pipeline is invalid**, including the
in-distribution arm of the leave-one-source-out comparison.

Honest transfer figures under LOSO:

| Held-out source | macro-F1 | ECE |
|---|---|---|
| A | 0.43 | 0.34 |
| B | 0.22 | 0.68 |

**Chance baseline is 0.3023**, computed on the actual (unbalanced) test class
distribution by simulating uniform guessing. An earlier report claimed chance was
0.33 by assuming balanced classes — wrong reasoning, though the conclusion held.
Source B is at or below chance *and* grossly overconfident.

### How this is written up

As a **methodological contribution**, not an embarrassment. Three public corpora,
combined and naively split, produce plausible-looking numbers; a sequence audit shows
the split was 81.59% contaminated and the honest transfer figure is at or below
chance. That is a sharper claim than a middling accuracy number.

### Related diagnostics, settled

| Check | Result |
|---|---|
| Crop-geometry confound | **Cleared.** Joint-matched deltas −0.032 and −0.045, both under the 0.05 threshold |
| Letterbox geometry | `pad_fraction = 1 − 1/AR` exactly, so the leakage channel is closed-form and provably removed |
| Composition drift | Overlap 32.6% → 1.2% after stratified split |
| Box-area discrepancy | **Unresolved.** An agent reported 270 vs 267 px²; re-checking `predictions_fold0.csv` found the original figure stable across both splits (val 189/8298, test 224/8451). Needs one more look. Do not resolve by preferring the most recent number. |

## Mandatory protocol rules

1. **Group by `source_sequence`.** Not by filename, not by directory, not by a hash
   of the image. Sequence stems are detected by the audit script in
   `src/vision/leakage_audit.py`.
2. **Class weights from the training split only.** For the current split,
   `w = 35105 / (3n)` gives **2.9677 / 0.7712 / 0.7319**, normalising to
   **1.9914 / 0.5175 / 0.4911**. The acceptance test is *that the weights were derived
   from the split in use* — not that they match any fixed triple. An earlier spec
   pinned them to corpus-wide values; that was wrong.
3. **Temperature scaling on the inner validation fold only.** Never on test.
4. **Report ECE alongside F1.** A model with ECE 0.68 is useless as an evidence source
   however good its F1 looks, because discounting will reduce it to near-vacuous.

## Open fix — do before any retrain

`final_welding/Lack of Fusion` currently maps to **V3 `{N,U}`**. This is wrong:
L is not an element of `{N,U}`. It must map to **V4 `{N,U,L,C}`**.

## Retrain plan — single source

**Decision: train on `final_welding` alone.** The three corpora were originally
unlabelled, segmented under different conventions, and carry different acquisition
geometry. Combining them bought sample count at the cost of a label space that does
not mean the same thing across sources — which is exactly what LOSO exposed.
`final_welding` is also closest in appearance to the real coupons: as-welded steel,
similar standoff, lighting and bead scale. Matching the deployment distribution
matters more than n.

Two stages (an earlier three-stage plan was cut for schedule):

1. **Baseline.** `final_welding` only, stratified split grouped on `source_sequence`,
   three-way classifier, class weights from the train split. This is the number that goes in
   the thesis.
2. **Initialisation ablation.** ImageNet-init vs public-weld-pretrained-init, both
   fine-tuned on the **real coupon images**. This is a result either way: if the
   leaky corpora still help as a feature initialiser, that is a useful finding about
   weld-image transfer; if they do not, it sharpens the leakage story.

## Fine-tuning on real coupons

Roughly 150 cells. **Do not fine-tune all of ResNet-50 on 150 samples.** Freeze the
backbone and train the heads (linear probe), or unfreeze only the last block.

Multiple photographs per cell inflate the image count but **not the cell count** —
they are augmentation. Grouping stays by `coupon_id` regardless of how many images a
cell has.

## Architecture note

The objective is written as "a convolutional surface-appearance classifier", not
"ResNet-50", deliberately. Do not bind the implementation to an architecture; if the
initialisation ablation suggests a change, make it.

A DINOv2 few-shot comparison was in the original plan and has been **cut** — one
paragraph of future work.
