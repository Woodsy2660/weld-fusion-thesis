# Sequence-Leakage Audit and Conditional Re-grouping

**Date:** 1 October 2026 · **Branch:** `vision/sequence-leakage-audit` · **Seed:** 42
**Frame:** three heads — V1 `{P}`, V3 `{N,U}`, V4 `{N,U,L,C}`
**Checkpoint used for embedding work:** `models/vision/best_resnet50.pth` (Task C)

---

## Verdict

| test | threshold | measured | verdict |
|---|---|---|---|
| **F2** test patches sharing a sequence stem with train | ≥ 5% | **81.59%** | **CONFIRMED** |
| **F3** test patches with nearest-train cosine < 0.05 | ≥ 2% | **32.83%** | **CONFIRMED** |
| **F4** test frames with a perceptual sibling in train | > 0% | **0.00%** | CONSISTENT |

**The split leaked.** 81.59% of test patches came from a weld sequence that also
appears in training — sixteen times the threshold. The existing guard could not
see it, because consecutive frames of one weld are different files, with
different names, and perceptual-hash distances above the threshold that caught
the Roboflow siblings.

The guard itself is not broken. F4 is clean: the perceptual union-find and the
split assignment agree exactly. `source_frame` closed what it was built to
close. It was simply the wrong grouping key for a corpus containing video.

---

## F1 — sequence detection

A **sequence stem** is the source-image basename with the Roboflow
`.rf.<hash>` suffix stripped and the final run of digits replaced by a
placeholder, so `weld_0042` and `weld_0043` share `weld_####`.

| dataset | frames | distinct stems | stems with >1 member | sequence-like stems¹ | frames inside them |
|---|---|---|---|---|---|
| final_welding | 2,748 | 734 | 223 | 76 | 1,406 |
| weld_dataset | 2,800 | 1,009 | 900 | 9 | 904 |
| wijaya | 360 | 16 | 14 | 14 | 358 |

¹ ≥3 members whose integer suffixes form a run with gaps ≤2 — the signature of
video-derived frames.

The largest stems make the mechanism unambiguous:

| dataset | stem | members | longest run | example |
|---|---|---|---|---|
| wijaya | `SampleV2_2_mp4-##` | 85 | **80** | `SampleV2_2_mp4-10_jpg.rf.…` |
| wijaya | `SampleV2_1_mp4-##` | 79 | 54 | `SampleV2_1_mp4-11_jpg.rf.…` |
| weld_dataset | `26_###` | **291** | 10 | `26_100.jpg` |
| weld_dataset | `25_###` | 251 | 11 | `25_100.jpg` |
| final_welding | `good_welding_nul_ttttt###_png` | 114 | 4 | `good_welding_nul_ttttt103_png_jpg.rf.…` |

The wijaya filenames contain `mp4`. `final_welding` carries a `bad_weld_vid###`
series. These are video frames, and `weld_dataset` numbers frames of one weld
consecutively.

---

## F2 — cross-split sequence audit

- stems with ≥2 frames: **1,132**
- of those, spanning more than one split: **464 (41.0%)**
- **test patches sharing a stem with train: 3,599 / 4,411 = 81.59%**
- val patches sharing a stem with train: 2,914 / 4,245 = 68.65%

| dataset | test patches | share with train | val patches | share with train |
|---|---|---|---|---|
| final_welding | 2,059 | 75.18% | 1,987 | 50.98% |
| weld_dataset | 2,244 | 86.59% | 2,158 | 83.46% |
| wijaya | 108 | **100.00%** | 100 | **100.00%** |

### Sensitivity of the stem heuristic

Collapsing the final digit run can over-merge: `IMG##############` is a 14-digit
timestamp and `##` merges every two-digit name. It changes nothing.

| stem restriction | test share with train |
|---|---|
| all stems (as reported) | 81.59% |
| digit run ≤ 4 (excludes timestamps) | 81.44% |
| stem has ≥3 non-digit characters | 81.52% |
| both restrictions | 81.38% |

A 0.21-point spread across every variant. The finding is not a stemming
artefact.

Restricting to stems with an unambiguous sequence marker:

| dataset | marker | frames | test patches | share with train |
|---|---|---|---|---|
| wijaya | filename contains `mp4` | 358 | 108 | 100.0% |
| weld_dataset | `weld-id_frame` pattern | 915 | 945 | 100.0% |
| final_welding | `good_welding_*` series | 751 | 186 | 98.9% |

---

## F3 — embedding nearest-neighbour audit

ResNet-50 penultimate features, L2-normalised. Test (4,411) against a train
subsample of 10,000 of 35,105, seed 42.

| | mean | median | p01 | p05 | min |
|---|---|---|---|---|---|
| test → nearest train | 0.0797 | 0.0661 | 0.0135 | 0.0243 | 0.0000 |
| test → nearest test (control) | 0.0791 | 0.0661 | −0.0000 | 0.0000 | −0.0000 |

- **below 0.05: 32.83%** (threshold 2%) → **CONFIRMED**
- control, test→test below 0.05: **32.17%**
- excluding `__negative__`, annotated defects only: **29.43%** (n = 3,830)
- `__negative__` patches only: **55.25%** (n = 581)

### This number needs its control to be read correctly

The test→test control sits at 32.17%, statistically indistinguishable from the
32.83% headline. Test patches are **no closer to train than to each other** —
which is what you would expect in a corpus where sequence structure pervades
every split, and which means the raw fraction alone establishes little.

Two refinements make it meaningful. First, 43 of the 50 closest pairs are
`__negative__` — blank background crops that are near-identical whether or not
they share a weld. Excluding them, **29.43%** of annotated-defect test patches
still lie within 0.05 of a train patch, fifteen times the threshold. Second,
the seven non-negative pairs among the 50 closest are visibly the same image
with **different `source_frame` values** — genuine cross-split duplicates that
the guard could not see:

| cosine | source | label | head | same source_frame? |
|---|---|---|---|---|
| 0.0080 | final_welding | cluster of porosity | V1 | no |
| 0.0080 | wijaya | Good Weld | V4 | no |
| 0.0098 | final_welding | Spatters | V4 | no |
| 0.0113 | final_welding | Porosity | V1 | no |
| 0.0128 | final_welding | Porosity | V1 | no |
| 0.0132 | wijaya | Good Weld | V4 | no |
| 0.0132 | final_welding | Spatters | V4 | no |

Of the 50 closest pairs, 50 share a source label, 50 share a head, 46 share a
source dataset.

Artefacts: `nn_distance_test_vs_train.png`, `../review/nn_pairs_test_train.png`,
`nn_audit.json`.

### Caveat on the feature space

These features come from a model trained on the leaking split, so it has
learned representations that pull near-duplicate pairs together. That biases
the distances toward looking closer than they truly are — inflating F3 rather
than deflating it. F3 is corroboration for F2 and input to G1, not the
deciding vote.

---

## F4 — grouping-headroom check

`dedup_report.json` records only counts, not group membership, so the method
named in the task specification was not available. An equivalent check was used
instead, and it is exact rather than approximate.

`source_frame` **is** the perceptual union-find's representative, and
`convert_and_remap.py` dropped every non-representative from the corpus. So if
each surviving image is its own representative, no perceptual group can have
two members in the corpus, and cross-split siblings are zero by construction.

- patches whose `source_image` ≠ `source_frame`: **0**
- `source_frame` values shared by more than one image: **0**

Both zero across all 43,761 patches. The union-find and the split assignment
agree.

---

## G — conditional re-grouping

### G1. `source_sequence`

Union-find over three relations.

| relation | merges | groups after |
|---|---|---|
| 1. perceptual union-find (already in `source_frame`) | 0 | 5,908 |
| 2. sequence stem | 4,149 | 1,759 |
| 3. embedding adjacency, cosine < 0.05 | 1,214 | **545** |

Relation 1 contributes nothing further, as F4 predicts — it is already encoded
in `source_frame`.

### The result is unusable, and that is the finding

| | 3 relations | stem only (1+2) |
|---|---|---|
| groups | 545 | 1,759 |
| largest group | 2,189 frames (**37.1%**) | 291 frames |
| top-3 groups | **83.9% of all patches** | — |
| train groups after split | **3** | 1,408 |
| max test drift | **53.2 pp** | 19.5 pp |
| pairs outside ±5 pp | 26 of 16 checked | 11 of 16 checked |

Relation 3 — single-linkage at cosine 0.05 under transitive closure — creates a
giant connected component. 41,830 pairs fell below the threshold and chained
37% of the corpus into one group.

This was tested for a plausible cause and rejected: `__negative__` patches are
blank and might have dragged frame-level means together, but excluding them
made the chaining **worse** (469 groups, largest 1,719 frames). The chaining is
intrinsic to single-linkage at this threshold, not an artefact of the negatives.

**Neither grouping meets the Task A acceptance bound.** This is reported, not
relaxed. A corpus whose sequences are large enough that entire source-label
populations live inside one of them cannot be both grouped and stratified —
`weld_dataset/discontinuity` is concentrated in `26_###`, `25_###` and `27_###`,
so once those are atomic it cannot be distributed evenly.

### G2. Re-split

Task H used the **stem-only** grouping: it is the relation that directly
addresses the confirmed F2 finding, and the only one that leaves a trainable
corpus.

| head | focal | train | val | test | total |
|---|---|---|---|---|---|
| V1 | `{P}` | 3,931 | 463 | 590 | 4,984 |
| V3 | `{N,U}` | 15,002 | 2,209 | 1,715 | 18,926 |
| V4 | `{N,U,L,C}` | 16,135 | 1,454 | 2,262 | 19,851 |
| **total** | | **35,068** | **4,126** | **4,567** | **43,761** |
| groups | | 1,408 | 176 | 175 | 1,759 |

Leakage guard extended: none of `source_frame`, `source_image`,
`source_sequence` spans splits. Max test drift V1 11.0 pp, V3 19.5 pp,
V4 11.2 pp — all above the ±5 pp bound.

A defect was found and fixed in the splitter while doing this: it fell back to
the greedy fill whenever `StratifiedGroupKFold` missed acceptance, even when
`StratifiedGroupKFold` was strictly better. It now keeps whichever is better,
which alone moved max drift from 30.3 pp to 19.5 pp.

### G3. Class weights

Recomputed on the new train split, `w_c = N_train / (K · n_c)`, K = 3,
normalised to mean 1:

| head | train patches | weight |
|---|---|---|
| V1 | 3,931 | 1.992 |
| V3 | 15,002 | 0.522 |
| V4 | 16,135 | 0.486 |

---

## What this does and does not establish

**Established.** The split leaked, at 81.59% by sequence stem and 29.43% by
embedding proximity among annotated defects. The mechanism is video-derived and
consecutively-numbered frames. The existing guard was correct for what it
covered and blind to this. The corpus cannot be simultaneously grouped at
sequence level and stratified by label.

**Not established.** The *size* of the leak's effect on performance. Task H
changed the grouping and broke the stratification at the same time, so the
Task C → Task H difference confounds the two. See `../vision/retrain_report.md`.

**Unaffected by any of this.** The leave-one-source-out folds, which train and
test on disjoint datasets by construction and cannot be inflated by sequence
leakage. That remains the trustworthy transfer estimate.
