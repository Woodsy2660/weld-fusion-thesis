# Claude Code Prompt — pad_fraction Leakage Diagnostic + Class Mapping Evidence Pack

**Created:** 30 September 2026
**Target repo branch:** branch from `vision_model_training` as `diagnostics/pad-leakage-and-mapping`
**Upstream artefacts:** Fold 0 Training Report, git SHA `87db5fc73abcb0840318a85fc0730dc7351629c9`
**Scope:** evidence only — no retraining, no config changes, no corpus regeneration

This prompt addresses items 0 and 1 of the post-fold-0 action plan: quantify whether
crop geometry leaks the target label, and assemble the evidence needed to settle the
three provisional class mappings. Both are blocking — nothing downstream (frame
change, retrain, calibration, BPA construction) should proceed until they resolve.

---

## Prompt (paste from here)

# Task: pad_fraction leakage diagnostic + class mapping evidence pack

Branch from `vision_model_training` as `diagnostics/pad-leakage-and-mapping`.

## Context

Fold 0 of the vision focal-element classifier reports V3 `{N,U}` F1 = 0.8776 and
V4 `{N,U,L}` F1 = 0.8870. Two unresolved risks make those numbers unsafe to write up.

**Risk A — crop-geometry confound.** Patches are letterboxed to 224x224 with edge
replication. Mean pad_fraction is 0.305 and 9,002 of 43,761 patches (20.6%) are
more than half padding. The V3 source classes (Overlap, Lack of Fusion, Irregular,
Burn Through, Undercut) are elongated toe-line boxes; the V4 source classes (stain,
spatter, deposit, negatives) are compact blobs. If pad_fraction correlates with head,
the network can read the head off crop geometry without looking at the weld, and
V3-vs-V4 is precisely the discrimination the vision modality exists to provide.

**Risk B — unconfirmed mappings.** Three entries in `configs/class_map.yaml` are
provisional: `weld_dataset/discontinuity` -> V3 (6,718 patches, the largest single
contributor to V3), `final_welding/cluster of porosity` -> V1 (120 patches), and
`final_welding/Crack` -> V2 (60 patches).

## HARD CONSTRAINTS

1. **Do not modify `configs/frame.yaml`, `configs/class_map.yaml`, or
   `configs/vision_train.yaml`.** This task produces evidence only. The human makes
   the mapping decisions. If you believe a config is wrong, write it in the report.
2. **Do not retrain.** Use `models/vision/best_resnet50_fold0.pth` as-is. Inference
   and feature extraction only.
3. **Do not regenerate the patch corpus.** `data/processed/vision_patches/` is fixed
   for this task.
4. The frame remains four heads for this diagnostic (V1 `{P}`, V2 `{C}`,
   V3 `{N,U}`, V4 `{N,U,L}`), because we are diagnosing the existing fold-0 run.
   Mirror the existing report convention: report every macro metric twice, once over
   four heads and once over the three supported heads (V1, V3, V4), and never quote a
   four-head macro alone.

## Step 0 — fix the prediction/manifest join

`results/vision/logits_test_fold0.npz` has shape (4376, 4) with labels and head
names but no patch identifier, so it cannot be joined to
`data/processed/vision_patches/manifest.csv`.

Re-run inference over the val and test splits with a deterministic, unshuffled
dataloader and persist, per patch: `patch_path`, `source_frame`, `source_dataset`,
`source_label`, `head_true`, `head_pred`, the four raw logits, `pad_fraction`,
`box_w`, `box_h`, `box_area`, `aspect_ratio`, `is_negative`, `split`.

Write to `results/diagnostics/predictions_fold0.parquet` (and `.csv` if under 50 MB).
Assert the recovered logits match `logits_test_fold0.npz` to within 1e-5 — if they
do not, stop and report, because something is non-deterministic in the eval path.

## Step 1 — descriptive association

Compute over the test split, and separately over the full corpus:

- pad_fraction mean, median, sd, IQR, and fraction above 0.5, **per head**
- the same, broken down per (source_dataset, source_label)
- aspect_ratio and box_area distributions per head
- Cramer's V between head and pad_fraction binned into deciles
- mutual information I(head; pad_fraction) in bits, and I(head; aspect_ratio)

Plot overlaid pad_fraction densities per head (`pad_fraction_by_head.png`) and a
box_area-vs-aspect_ratio scatter coloured by head (`box_geometry_by_head.png`).

## Step 2 — geometry-only probe

Can crop geometry alone predict the head, with no image content?

Fit multinomial logistic regression on the training split and evaluate on the test
split, using **the same grouped splits as the real model** so results are
comparable. Three feature sets, reported separately:

- A: `pad_fraction` alone
- B: `aspect_ratio` alone
- C: `pad_fraction`, `aspect_ratio`, `box_w`, `box_h`, `box_area` (log-scaled)

Report accuracy, per-head P/R/F1, four-head macro-F1 and three-head macro-F1 for each.

Establish the null properly rather than assuming 1/K: permute `head_true` on the test
split 1,000 times (seed 42), recompute macro-F1 each time, and report the mean and
95th percentile. Write the permutation distribution to JSON.

## Step 3 — does the real model's accuracy track pad_fraction?

Using `predictions_fold0.parquet`, stratify the test split into pad_fraction deciles
and compute per-head F1 within each decile. Plot F1-vs-decile per head
(`f1_by_pad_decile.png`) with support annotated on each point.

The specific pattern to look for and state explicitly in the report: V3 accurate
mainly in high-pad deciles and V4 accurate mainly in low-pad deciles. That would be
the direct signature of the model keying on crop shape.

## Step 4 — pad-matched subsample (the decisive number)

Construct a test subsample in which V3 and V4 have matched pad_fraction
distributions, by stratified resampling within pad_fraction deciles down to the
per-decile minimum count across the two heads. Keep V1 and V2 unmatched but report
them. Recompute per-head P/R/F1 on the matched subsample.

Report the delta against the unmatched test figures, with bootstrap 95% CIs
(1,000 resamples, seed 42) on both the matched and unmatched F1 so the delta can be
judged against sampling noise. Record the matched subsample size.

**This is the headline output of the whole task.** It answers, in one number, how
much of V3/V4 performance survives when crop geometry carries no information.

## Step 5 — apply these decision thresholds

State each verdict explicitly in the report. Do not soften them.

| test | threshold | verdict if exceeded |
|---|---|---|
| Cramer's V (head vs pad decile) | >= 0.30 | material association |
| Probe C three-head macro-F1 | >= permutation p95 + 0.10 | geometry alone is predictive |
| Matched-subsample V3 or V4 F1 drop | >= 0.05 | performance partly attributable to padding |

If any threshold is exceeded, add a section costing the two candidate remedies
against this corpus: (a) aspect-aware tiling for boxes beyond 3:1 — state how many
patches and which heads are affected, and how many tiles would result;
(b) mask-aware pooling that excludes padded regions from the feature map. Recommend
one. **Do not implement either in this task.**

## Step 6 — mapping evidence pack

For each of `weld_dataset/discontinuity`, `final_welding/cluster of porosity`,
`final_welding/Crack`, and as visual controls `final_welding/Overlap`,
`final_welding/Undercut`, `weld_dataset/stain` and `weld_dataset/deposit`:

Generate a contact sheet of 64 instances to
`results/review/mapping/contact_<dataset>_<label>.png`, sampled **stratified by
pad_fraction quartile and box aspect quartile** (not the first 50, and not uniformly
random — we need to see the elongated and heavily padded cases specifically).

Each cell must show two images side by side: the original crop with the annotation
box overlaid, and **the letterboxed 224x224 tensor exactly as the network receives
it**. The second is the important one — it reveals whether the feature survives
padding. Annotate each cell with pad_fraction, aspect ratio, box area in px, and
whether fold 0 classified it correctly.

Then produce these quantitative cuts:

1. **Per-source-label recall.** For every source_label inside each head, the fold-0
   recall and support. If `discontinuity` behaves like the other V3 labels it
   supports the mapping; if it is a visible outlier, flag it.
2. **Embedding.** Extract penultimate-layer (2048-d) features for all V3 test
   patches, reduce to 2D by PCA then UMAP (seed 42), colour by source_label.
   Save as `v3_embedding_by_source_label.png`. Report whether `discontinuity`
   forms a cluster separable from Overlap / Undercut / Excess-Reinforcement, and
   quantify it with silhouette score of `discontinuity` against the rest of V3.
3. **Label inventory.** Enumerate every class in each source `data.yaml` with its
   instance count post-dedup. State whether `discontinuity` appears to be that
   dataset's catch-all class, since a catch-all would make a single-head mapping
   unsafe regardless of how the contact sheet looks.

Write `results/diagnostics/mapping_evidence.md` with, for each of the three
provisional mappings: the evidence, a recommendation, and the confidence in it.
Recommendations only — the config stays untouched.

## Step 7 — regression tests

Add to `tests/test_vision_splits.py` (or a new `tests/test_no_geometry_leak.py`):

- Cramer's V between head and binned pad_fraction stays below the value measured in
  this run plus 0.05, so a future pipeline change cannot silently worsen it. Hardcode
  the measured baseline with a comment giving this run's git SHA.
- pad_fraction is present and non-null for every manifest row.
- A geometry-only probe on feature set C does not exceed permutation p95 + 0.10.
  If it **already** does, mark the test `xfail(strict=True)` with the measured value
  in the reason string, so the failure is recorded rather than hidden.

All existing 32 tests must still pass.

## Deliverables

```
results/diagnostics/predictions_fold0.parquet
results/diagnostics/pad_fraction_stats.json
results/diagnostics/geometry_probe_metrics.json
results/diagnostics/permutation_null.json
results/diagnostics/matched_subsample_metrics.json
results/diagnostics/pad_fraction_by_head.png
results/diagnostics/box_geometry_by_head.png
results/diagnostics/f1_by_pad_decile.png
results/diagnostics/v3_embedding_by_source_label.png
results/diagnostics/pad_leakage_report.md
results/diagnostics/mapping_evidence.md
results/diagnostics/run_manifest_diagnostics.json
results/review/mapping/contact_<dataset>_<label>.png   (7 sheets)
```

`run_manifest_diagnostics.json` follows the existing convention: git SHA, seed,
config SHA-256 hashes (first 16) for frame/class_map/vision_train, torch and CUDA
versions, checkpoint path and its hash.

Extend the `.gitignore` exception so `results/diagnostics/*.json` and `*.png` are
tracked. Keep the parquet untracked if it exceeds 50 MB and say so in the report.

## Report structure

`pad_leakage_report.md` must open with a verdict box: for each of the three
thresholds in Step 5, PASS or FAIL with the measured value. Then the matched-subsample
table as the primary result, then the supporting analysis.

Write plainly and do not round in a flattering direction. If the conclusion is that
a material part of V3/V4 performance is crop geometry, say so in the first paragraph.
A negative result here is worth more than a reassuring one, because it is discoverable
by an examiner and cheaper to fix now than after the fusion stage is built on it.

## Acceptance

- `pytest tests/ -q` passes (32 existing + new)
- Every deliverable exists
- Recovered logits match the stored npz to 1e-5
- No file under `configs/` is modified — verify with `git diff --stat configs/`
- Every macro metric in the report appears as both four-head and three-head

## Prompt ends

---

## Notes for interpreting the results

**If Step 4 shows a large F1 drop, that is a finding, not a failure.** The remedy —
aspect-aware tiling for boxes beyond 3:1 — also *increases* the number of training
units for the elongated defect classes, which are exactly the classes V3 depends on.
The corpus gets better, not smaller.

**If Step 6's silhouette score shows `discontinuity` clustering apart from the other
V3 labels, do not remap on that basis alone.** Separation could reflect
`weld_dataset`'s camera, lighting and surface finish rather than a different defect
population. The leave-one-source-out ablation is the decisive test for that question
and belongs in a later task.

**Sequencing.** Steps 0–5 are the leakage question and Step 6 is the mapping question;
they are independent and can be reported separately if the agent needs to split the
work. Step 6 requires human judgement on the contact sheets before any config change,
so generate the artefacts and stop.

## Downstream tasks this unblocks

1. Frame change to three heads — remove V2 `{C}`, map `final_welding/Crack` to the
   ignorance head, widen V4 to `{N,U,L,C}`. Drops head imbalance from 329.9:1 to
   3.98:1 and normalised class weights from a 330:1 spread to roughly 4:1.
2. Retrain with the overfit remedies: cap 18 epochs, patience 5, weight decay 1e-3,
   freeze through `layer2`, runtime augmentation restored, ResNet-18 ablation.
3. Temperature scaling on validation logits, ECE reported on test, then BPA
   construction with a Shafer discount and an alpha sensitivity sweep.
4. Grouped 5-fold on `source_frame` for per-head variance.
5. Leave-one-source-out ablation, three ways — the closest available estimate of
   transfer to the real RealSense coupon images.

Note that the invariant test currently asserting `validate_modality_bpa()` accepts
`{C}` must be parameterised by modality before task 1: `{C}` ceases to be a legal
vision focal element while remaining legal for UT.
