# Status — Stratified Split, Three-Head Frame, Retrain, LOSO

**Branch:** `vision/stratified-split-and-three-head`
**Prompt:** `claude-code-split-frame-retrain-prompt.md`
**Date:** 30 September 2026
**State:** Tasks A and B complete; Task C running; Tasks D and E outstanding

---

## Done

### Task A — label-stratified grouped split

Approach 1 (`StratifiedGroupKFold`, `groups=source_frame`, `y` = the frame's
dominant `(source_dataset, source_label)`, 10 folds — one test, one val, eight
train) met acceptance on the first attempt, so the greedy fallback was not
needed.

| | pairs checked | outside ±5 pp bound |
|---|---|---|
| result | 16 | **0** |

Maximum test drift among pairs at ≥2% share:

| head | max test drift |
|---|---|
| V1 | 4.3 pp |
| V3 | 1.2 pp |
| V4 | 2.1 pp |

Leakage guard holds on both `source_frame` and `source_image` — no value spans
splits. Full corpus-vs-train-vs-val-vs-test table written to
`results/vision/split_composition.md`.

Resulting split:

| head | focal element | train | val | test | total |
|---|---|---|---|---|---|
| V1 | `{P}` | 3,943 | 551 | 490 | 4,984 |
| V3 | `{N,U}` | 15,174 | 1,833 | 1,919 | 18,926 |
| V4 | `{N,U,L,C}` | 15,988 | 1,861 | 2,002 | 19,851 |
| **total** | | **35,105** | **4,245** | **4,411** | **43,761** |
| frames | | 4,728 | 590 | 590 | 5,908 |

**Bug found and fixed in the greedy fallback.** Scoring composition distance and
size shortfall additively let the composition term permanently starve val and
test — the fallback put **every frame in train** (shares 1.000 / 0.000 / 0.000).
The size quota now gates the candidate set and composition only chooses among
splits still under quota. Result: 0.800 / 0.100 / 0.100 exactly, with all three
heads present in every split. The fallback is not used on the real data, but it
is exercised by the test suite and had to be correct.

### Task B — three-head frame

- `configs/frame.yaml`: V2 `{C}` removed. V3 and V4 keep their names — no
  renumbering, so existing artefacts, contact sheets and manifests stay valid.
  V4 widened `{N,U,L}` → `{N,U,L,C}`.
- `configs/class_map.yaml`: `final_welding/Crack` → V4 (not discard). The two
  confirmed mappings annotated with their evidence.
- Heads are now **recomputed from the class map at split time** rather than
  trusted from the manifest, so the frame change propagated to the 60 Crack
  patches without re-cropping anything. `data/processed/vision_patches/` is
  untouched.
- `src/utils/frame.py`: any `<modality>_heads` block in the frame is picked up
  automatically, so adding a modality is a config change, not a code change.
  `geometry_heads` and `ut_heads` declared.
- `src/bpa/construct.py`: `validate_modality_bpa()` parameterised by modality.
  `{C}` is **rejected for vision** and **accepted for ut**. The rule remains
  "only this modality's declared focal elements", never "no singletons" — `{P}`
  is still a legal vision singleton and `{U}` a legal geometry singleton.

Consequences verified:

| quantity | expected | measured |
|---|---|---|
| V1 patches | 4,984 | 4,984 |
| V3 patches | 18,926 | 18,926 |
| V4 patches | 19,791 + 60 | 19,851 |
| imbalance | 3.98:1 | **3.98:1** |

**Discrepancy to flag on the class-weight acceptance.** Two different weight
vectors exist and only one meets the ±0.01 bound:

| basis | V1 | V3 | V4 | within 0.01 of 1.98/0.52/0.50? |
|---|---|---|---|---|
| whole corpus (N=43,761, K=3) | 1.9810 | 0.5217 | 0.4974 | yes, all three |
| **train split** (what the loss uses) | **1.991** | 0.517 | 0.491 | V1 off by **0.011** |

The weights actually applied are computed on the train split, as they must be —
computing them over the whole corpus would leak val/test composition into the
loss. To two decimal places the train-split V1 weight is 1.99 against the
expected 1.98. Reported rather than silently accepted.

### `tests/test_frame_algebra.py`

All four propositions encoded against `frame.yaml`, not hardcoded sets:

1. `{N,U,L,C} ∩ {N,P,L,C} ∩ {N,U} = {N}` — three-way conjunction isolates the
   sound weld.
2. `{N,U} ∩ {N,P,L,C} = {N}` — vision saw a deviation, geometry measured it
   within ISO 5817 limits.
3. Vision + geometry reach exactly `{N}`, `{U}`, `{P}`; L and C are not
   separable.
4. `{P} ∩ {U} = ∅` — feeds the conflict mass K rather than any focal element.

**My first version of proposition 3 was wrong.** I asserted the intersection
equals `{L,C}`, but V4 ∩ G2 = `{N,L,C}` — N comes along. The correct claim is
that neither L nor C is ever reached without the other, which is what the test
now checks.

**59 tests pass**, including the updated four-head → three-head assertions in
`test_frame_and_bpa.py` and `test_vision_splits.py`.

---

## Running

### Task C — retrain

ResNet-50 in progress, ResNet-18 chained behind it on the same GPU.

| epoch | train loss / acc | val loss / acc | val macro-F1 (3-head) |
|---|---|---|---|
| 1 | 0.4252 / 0.8344 | 0.2904 / 0.8780 | 0.8491 |
| 2 | — | — | — |
| 3 | 0.2358 / 0.9093 | 0.2567 / 0.9015 | **0.8791** |
| 4 | 0.2143 / 0.9187 | 0.2435 / 0.8989 | 0.8729 |
| 5 | 0.1991 / 0.9233 | 0.3289 / 0.8949 | 0.8548 |

For contrast, fold 0's epoch 1 val macro-F1 was 0.4358. The three-head frame
plus the corrected split makes this a substantially better-posed problem from
the first epoch.

Config changes applied: 18 epochs, patience 5, weight decay 1e-3, permanent
freeze through `layer2` (ResNet-50 22.1M/23.5M trainable; ResNet-18
10.5M/11.2M), and runtime augmentation — h/v flip, ±15° rotation,
brightness/contrast 0.2, Gaussian blur p=0.2. No hue or saturation jitter, no
aggressive scale jitter.

Logits are written as **parquet carrying the join keys** (`patch_path`,
`source_frame`, `source_dataset`, `source_label`, `split`, `pad_fraction`,
`box_area`, `aspect_ratio`), fixing the fold-0 defect where the npz could not be
joined to the manifest.

---

## Outstanding

| task | state |
|---|---|
| Task C — ResNet-18 | queued behind ResNet-50 |
| Task E.2 — geometry re-check gate | script written, awaits test logits |
| Task D — LOSO ×3 | script written, awaits best backbone |
| Task E.1 — `tests/test_no_geometry_leak.py` | not started; baseline must be re-measured on the new split |
| `retrain_report.md`, `loso_report.md` | not started |

---

## Notes on prompt premises

**The prompt's leakage figures came from a fuller diagnostic run than I
completed.** It cites joint (area decile × pad tercile) matching deltas of
−0.0321 / −0.0449, a within-`final_welding` V3 F1 of 0.829, and a source-identity
probe at 0.790 accuracy. My run produced only the **pad-only** matching, with
V3 −0.0029 and V4 −0.0070. The mapping decisions are the user's and have been
implemented as given; the joint matching is re-run as the Task E.2 gate, which
will produce a directly comparable number on the new split.

**The composition-drift diagnosis does check out against my data.** V1 test was
66.7% `pore` against 54.2% in the corpus, and V3 test was 42.3% `discontinuity`
against 35.5% — both matching the prompt's figures. That is the defect Task A
fixes.

**`tests/test_no_geometry_leak.py` did not exist.** Task E.1 says to *update*
its Cramér's V baseline; the previous task was superseded before the file was
written, so it will be created with a freshly measured baseline rather than
updated.
