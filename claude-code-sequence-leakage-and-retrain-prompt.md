# Claude Code Prompt — Sequence-Leakage Audit and Conditional Retrain

**Created:** 30 September 2026
**Target branch:** branch from `vision/stratified-split-and-three-head` as `vision/sequence-leakage-audit`
**Predecessor:** `claude/claude-code-split-frame-retrain-prompt.md` (Tasks A and B complete, C running)
**Runs after:** Task C finishes. Do not interrupt the running job.

## Why this task exists

The first execution of the retrain reached validation macro-F1 **0.8491 at epoch 1**
and peaked at **0.8791 by epoch 3**, exceeding fold 0's final three-head test macro
(0.8392) within three epochs, then declining.

Part of that jump is legitimate. Fold 0 froze the entire backbone for five epochs, so
its comparable point is epoch 6 at 0.5165, not epoch 1 at 0.4358. The three-head frame
removed a head scoring near zero from the macro, and the 330:1 imbalance with its
3.93× weight on a 48-patch head is gone. A better-posed problem should train faster.

But 0.5165 → 0.8491 is a large step, and **saturating by epoch 3 is a symptom worth
chasing**. The most likely benign explanation is that the problem genuinely got easier.
The most likely non-benign explanation is near-duplicate leakage that the existing
guard cannot see.

### The gap in the current leakage guard

The guard asserts that no `source_frame` and no `source_image` appears in more than one
split. That closes exact and filename-linked duplication. It does **not** close:

- **Sequence structure.** If any source dataset is video-derived, consecutive frames of
  the same weld are near-identical, carry different filenames, and are different
  `source_frame` values. They will often survive perceptual hashing, because small
  camera motion pushes them past Hamming distance 5.
- **The known phash recall ceiling.** The dedup audit measured phash recall at
  **0.1892** against filename identity. Filename identity did the heavy lifting for
  Roboflow siblings, which only works within a single export. Cross-export and
  cross-dataset near-duplicates are recovered at under a fifth.

The stratified assignment may make this worse rather than better: frames from one
sequence share a dominant label, so stratifying by dominant label actively scatters
them across splits, where the old greedy assignment might have kept them together by
accident.

---

## Prompt (paste from here)

# Task: sequence-leakage audit and conditional retrain

Branch from `vision/stratified-split-and-three-head` as `vision/sequence-leakage-audit`.
Run this only after Task C (both backbones) has finished. Do not interrupt it.

## Context

The three-head retrain on the stratified split reached val macro-F1 0.8491 at epoch 1
and 0.8791 at epoch 3, against fold 0's final three-head test macro of 0.8392. Before
that number is carried into calibration, BPA construction or the thesis, it must be
established that it is not inflated by near-duplicate frames spanning the train and
test splits.

The existing leakage guard covers `source_frame` and `source_image` only. The dedup
audit measured perceptual-hash recall at 0.1892, so a large majority of perceptual
siblings were recovered by filename identity rather than by image content — a
mechanism that does not work across exports or across datasets.

## HARD CONSTRAINTS

1. **Do not change `configs/frame.yaml` or `configs/class_map.yaml`.** The three-head
   frame is settled.
2. **Do not retrain anything in Tasks F or G.** Those are audits using the existing
   checkpoint. Retraining happens only in Task H, and only if Task G triggers.
3. **Do not delete patches or images.** If re-deduplication is required it changes the
   grouping key and the split assignment, not the corpus on disk.
4. Report a clean result as readily as a dirty one. A negative finding here is the
   good outcome and must not be talked up into a problem, nor a positive finding
   softened.

---

## Task F — audit

### F1. Sequence detection

For each source dataset, take `source_image` basenames and strip the Roboflow suffix
(`.rf.<hash>` and the extension). Derive a **sequence stem** by replacing the final run
of digits in the remaining basename with a placeholder — `weld_0042` and `weld_0043`
share the stem `weld_####`.

Report per dataset:

- number of distinct stems, and the distribution of members per stem
- number of stems with ≥3 members whose integer suffixes form a contiguous or
  near-contiguous run (gaps ≤ 2), which is the signature of video-derived frames
- the 20 largest stems with example filenames, so the pattern is human-checkable

Handle each dataset's own convention. `weld_dataset`, `final_welding` and `wijaya` may
number differently or not at all; a dataset with no digit runs simply reports zero
sequences and that is a valid result.

### F2. Cross-split sequence audit

Using the stems from F1 and the current split assignment:

- fraction of stems (with ≥2 members) whose frames occupy more than one split
- **fraction of test patches whose sequence stem also appears in train** — this is the
  headline number
- the same fraction for val
- a per-dataset breakdown of both

### F3. Embedding nearest-neighbour audit

Using the best Task C checkpoint, extract penultimate-layer features for the whole test
split and for the train split (subsample train to 10,000 patches if needed for
tractability; state the subsample size and seed).

For every test patch, find its nearest train-split neighbour by cosine distance.
Report:

- the full distribution of test→train nearest-neighbour cosine distance
- **the fraction of test patches with nearest-train cosine distance below 0.05**
- as a control, the distribution of test→test nearest-neighbour distance excluding
  self, so the two can be compared on the same axes
- the same two distributions plotted together in
  `results/diagnostics/nn_distance_test_vs_train.png`

Then generate `results/review/nn_pairs_test_train.png`: the **50 closest test↔train
pairs**, shown side by side, each annotated with cosine distance, both source datasets,
both source labels, both split assignments, and both heads.

**This contact sheet is the decisive artefact of the whole task.** If those pairs are
visibly the same weld photographed a frame apart, the split is leaking and no
statistic is needed to establish it. If they are merely similar welds, it is not.

### F4. Grouping-headroom check

Train nothing. Instead, quantify how much the grouping is buying, using only the
existing checkpoint and the manifest: report the fraction of test patches whose
`source_frame` has at least one perceptual-group sibling (from the existing
`dedup_report.json` union-find) assigned to train. The guard should make this zero by
construction — if it is not, the union-find and the split assignment disagree and that
is a bug to report before anything else.

### F5. Verdict

Apply these thresholds and state each verdict explicitly.

| test | threshold | verdict if exceeded |
|---|---|---|
| F2: test patches sharing a sequence stem with train | ≥ 5% | sequence leakage confirmed |
| F3: test patches with nearest-train cosine distance < 0.05 | ≥ 2% | near-duplicate leakage confirmed |
| F4: test frames with a perceptual sibling in train | > 0% | guard/union-find inconsistency — **bug, report immediately** |

Write `results/diagnostics/sequence_leakage_report.md` opening with a verdict box
carrying the measured value beside each threshold.

**Stop here and report if all three are clean.** In that case the 0.879 is real, the
speed of convergence is explained by the better-posed problem, and the work proceeds to
Tasks D and E of the predecessor prompt unchanged. Say so plainly in the report.

---

## Task G — conditional re-grouping (only if F5 confirms leakage)

Do not run this unless F2 or F3 exceeded its threshold.

### G1. Introduce a stronger grouping key

Add a `source_sequence` column to `data/processed/vision_patches/manifest.csv`,
computed as the union-find over three relations, in this order:

1. the existing union-find from `dedup_report.json` (filename identity, MD5, phash ≤ 5)
2. **sequence stem**, where F1 found sequence structure
3. **embedding adjacency** — connect any two frames whose mean patch-feature cosine
   distance is below 0.05, using the Task C checkpoint

Relation 3 is the one that addresses the phash recall ceiling directly: features from a
trained network are invariant to the rotation and shear that defeated phash at Hamming
5. Report how many groups each relation contributes and how many frames each merges, so
the marginal value of each is visible.

### G2. Re-split on the new key

Re-run the Task A stratified assignment with `groups=source_sequence` instead of
`groups=source_frame`. The Task A acceptance bounds still apply: ≤5 percentage points
drift on test and val for every `(head, source_label)` pair at ≥2% share within its
head.

Extend the leakage guard test to assert no `source_sequence` spans splits, alongside
the existing `source_frame` and `source_image` assertions.

Expect the effective corpus to shrink. Report the new group count against the 5,908
frames, and the new per-head per-split counts. If the drift bounds can no longer be met
because groups have become too coarse, report that rather than relaxing the bound —
a corpus that cannot be both grouped and stratified is itself a finding about the data.

### G3. Recompute class weights

On the new train split only, `w_c = N_train / (K · n_c^train)` with K = 3, normalised to
mean 1. Report the weights with the train-split head counts behind them. Do not assert
against a hardcoded target.

---

## Task H — retrain (only if Task G ran)

Re-run Task C unchanged — same config, same two backbones, same augmentation — on the
re-split corpus.

Then produce a three-way comparison in `results/vision/retrain_report.md`:

| run | grouping key | val macro-F1 | test macro-F1 (3-head) |
|---|---|---|---|
| fold 0 | source_frame, unstratified | 0.6614 | 0.8392 |
| Task C | source_frame, stratified | — | — |
| Task H | source_sequence, stratified | — | — |

**The Task C → Task H drop is the size of the leak.** State it as a number in the first
paragraph. If it is large, that is the most important single result in this task, and
every downstream number must be taken from Task H rather than Task C.

---

## Task I — continue

Whichever path was taken, proceed to Tasks D and E of
`claude-code-split-frame-retrain-prompt.md` — leave-one-source-out, the geometry
re-check gate, and the regression tests — using the final checkpoint. Do not run LOSO
against a checkpoint that Task H superseded.

Add to the LOSO report the two figures that currently exist only outside the repo:

- V3 F1 restricted to `final_welding` patches only, which is the evidence that the
  `discontinuity` → V3 mapping is not propping up the head (previously 0.829)
- a geometry-only logistic probe predicting `source_dataset`, reported against the
  majority baseline (previously 0.790 against 0.522)

Both are cheap, both belong in the thesis, and neither is currently reproducible from
repo artefacts.

## Deliverables

```
results/diagnostics/sequence_stems.json
results/diagnostics/sequence_leakage_report.md
results/diagnostics/nn_distance_test_vs_train.png
results/diagnostics/nn_audit.json
results/review/nn_pairs_test_train.png
results/diagnostics/run_manifest_sequence_audit.json
```

If Task G runs, additionally:

```
data/processed/vision_patches/manifest.csv         (modified: + source_sequence)
data/processed/vision_patches/sequence_groups.json
results/vision/split_composition_v2.md
results/vision/metrics_resnet50_v2.json
results/vision/metrics_resnet18_v2.json
results/vision/logits_{val,test}_<backbone>_v2.parquet
```

## Acceptance

- `pytest tests/ -q` passes, with the sequence guard added if Task G ran
- Every threshold in F5 has a measured value and an explicit verdict
- `nn_pairs_test_train.png` exists and shows 50 pairs with distances annotated
- If Task G ran: no `source_sequence` spans splits, and the Task A drift bounds are met
  or the failure is reported with achieved drift
- No change to image files on disk — `git diff --stat` shows manifest changes only

## Report tone

The verdict box goes first, with numbers. If the split is clean, say so in one sentence
and do not hedge it into ambiguity — a clean audit is a result, and the fast convergence
then has a mundane explanation that should be stated. If it is dirty, give the size of
the effect in the first paragraph and do not lead with the remediation.

## Prompt ends

---

## Notes for interpreting the results

**The contact sheet decides it, not the statistics.** Cosine distance below 0.05 in a
trained network's feature space is suggestive but not proof — two genuinely different
welds photographed under identical lighting can land close together. Fifty side-by-side
pairs settle in one look what a threshold argues about.

**A clean result is the likelier outcome and is worth stating confidently.** Three
things changed at once between fold 0 and this run: the frame went from four heads to
three (removing a head that scored near zero from the macro), the class imbalance fell
from 329.9:1 to 3.98:1, and the backbone now trains `layer3` and `layer4` from epoch 1
instead of being fully frozen for five epochs. Any one of those would speed convergence;
together they plausibly account for the whole jump.

**If it is dirty, the fix improves the thesis rather than damaging it.** Embedding-based
grouping is a stronger deduplication method than perceptual hashing and the reason is
explainable in one sentence — learned features are invariant to the rotation and shear
that defeat phash. Reporting that you found the ceiling of phash dedup empirically
(recall 0.1892), diagnosed its consequence, and replaced it with a method that works is
a better methods chapter than never having looked.

**Either way, LOSO remains the arbiter of transfer.** Sequence leakage inflates the
in-distribution number. It does not touch the leave-one-source-out folds, which train
and test on disjoint datasets by construction. If in-distribution sits at 0.88 and
held-out lands near 0.6, that gap is the honest basis for the Shafer discount α on the
vision BPA — because the October coupon images are themselves an unseen source.
