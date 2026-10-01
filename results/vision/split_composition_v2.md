# Split composition

Assignment: **StratifiedGroupKFold(n_splits=10, groups=source_frame)**

Acceptance: every (head, source_label) pair at or above 2% of its head must sit within ±5 percentage points of its corpus share in both val and test. Pairs below that share are exempt from the bound but are still listed.

- pairs checked: **16**
- pairs outside the bound: **11**

## Pairs outside the bound

| head | label | split | corpus | split | drift (pp) |
|---|---|---|---|---|---|
| V1 | `weld_dataset/pore` | test | 54.2% | 44.9% | -9.2 |
| V1 | `final_welding/Porosity` | test | 43.4% | 54.4% | +11.0 |
| V3 | `weld_dataset/discontinuity` | val | 35.5% | 19.7% | -15.8 |
| V3 | `weld_dataset/discontinuity` | test | 35.5% | 55.0% | +19.5 |
| V3 | `final_welding/Overlap` | test | 32.6% | 22.3% | -10.3 |
| V3 | `final_welding/Lack of Fusion` | val | 17.3% | 32.4% | +15.0 |
| V3 | `final_welding/Lack of Fusion` | test | 17.3% | 9.0% | -8.3 |
| V4 | `weld_dataset/stain` | val | 36.7% | 26.5% | -10.1 |
| V4 | `weld_dataset/stain` | test | 36.7% | 47.9% | +11.2 |
| V4 | `final_welding/Spatters` | test | 14.2% | 7.0% | -7.2 |
| V4 | `weld_dataset/deposit` | val | 13.2% | 7.6% | -5.6 |

## Head totals

| head | focal element | patches | class weight |
|---|---|---|---|
| V1 | `{P}` | 4,984 | 1.9810 |
| V3 | `{N,U}` | 18,926 | 0.5217 |
| V4 | `{N,U,L,C}` | 19,851 | 0.4974 |

## V1 `{P}` — 4,984 patches

Maximum test drift among pairs at ≥2% share: **11.0 pp**

| source label | corpus | train | val | test | test drift (pp) |
|---|---|---|---|---|---|
| `weld_dataset/pore` | 54.2% (2,699) | 55.6% | 54.0% | 44.9% | -9.2 |
| `final_welding/Porosity` | 43.4% (2,165) | 41.7% | 43.8% | 54.4% | +11.0 |
| `final_welding/cluster of porosity` | 2.4% (120) | 2.7% | 2.2% | 0.7% | -1.7 |

`·` = below the 2% threshold, exempt from the bound.

## V3 `{N,U}` — 18,926 patches

Maximum test drift among pairs at ≥2% share: **19.5 pp**

| source label | corpus | train | val | test | test drift (pp) |
|---|---|---|---|---|---|
| `weld_dataset/discontinuity` | 35.5% (6,718) | 35.6% | 19.7% | 55.0% | +19.5 |
| `final_welding/Overlap` | 32.6% (6,162) | 33.3% | 35.5% | 22.3% | -10.3 |
| `final_welding/Lack of Fusion` | 17.3% (3,278) | 16.1% | 32.4% | 9.0% | -8.3 |
| `final_welding/Irregular` | 6.3% (1,194) | 6.8% | 3.7% | 5.2% | -1.1 |
| `final_welding/Burn Through` | 5.5% (1,049) | 5.8% | 5.1% | 4.0% | -1.5 |
| `final_welding/Excess-Reinforcement` | 2.1% (400) | 1.8% | 3.3% | 3.5% | +1.4 |
| `final_welding/Undercut` · | 0.5% (100) | 0.5% | 0.2% | 0.9% | +0.4 |
| `final_welding/root concvity` · | 0.1% (15) | 0.1% | 0.0% | 0.0% | -0.1 |
| `final_welding/excess root penetration` · | 0.1% (10) | 0.1% | 0.1% | 0.0% | -0.1 |

`·` = below the 2% threshold, exempt from the bound.

## V4 `{N,U,L,C}` — 19,851 patches

Maximum test drift among pairs at ≥2% share: **11.2 pp**

| source label | corpus | train | val | test | test drift (pp) |
|---|---|---|---|---|---|
| `weld_dataset/stain` | 36.7% (7,282) | 36.0% | 26.5% | 47.9% | +11.2 |
| `final_welding/Spatters` | 14.2% (2,819) | 14.8% | 18.4% | 7.0% | -7.2 |
| `weld_dataset/__negative__` | 14.1% (2,793) | 13.8% | 13.3% | 16.4% | +2.4 |
| `final_welding/__negative__` | 13.6% (2,696) | 13.8% | 17.6% | 9.2% | -4.3 |
| `weld_dataset/deposit` | 13.2% (2,625) | 13.2% | 7.6% | 16.7% | +3.4 |
| `wijaya/Good Weld` | 3.5% (689) | 3.6% | 6.4% | 0.4% | -3.0 |
| `final_welding/Arc strike` | 2.5% (496) | 2.6% | 3.5% | 1.5% | -1.0 |
| `wijaya/__negative__` · | 1.8% (360) | 1.7% | 6.0% | 0.2% | -1.6 |
| `final_welding/Crack` · | 0.3% (60) | 0.3% | 0.6% | 0.5% | +0.2 |
| `final_welding/mechanical-damage` · | 0.2% (31) | 0.2% | 0.1% | 0.1% | -0.1 |

`·` = below the 2% threshold, exempt from the bound.
