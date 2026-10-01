# Split composition

Assignment: **StratifiedGroupKFold(n_splits=10, groups=source_frame)**

Acceptance: every (head, source_label) pair at or above 2% of its head must sit within ±5 percentage points of its corpus share in both val and test. Pairs below that share are exempt from the bound but are still listed.

- pairs checked: **16**
- pairs outside the bound: **0**

All checked pairs are within the bound.

## Head totals

| head | focal element | patches | class weight |
|---|---|---|---|
| V1 | `{P}` | 4,984 | 1.9810 |
| V3 | `{N,U}` | 18,926 | 0.5217 |
| V4 | `{N,U,L,C}` | 19,851 | 0.4974 |

## V1 `{P}` — 4,984 patches

Maximum test drift among pairs at ≥2% share: **4.3 pp**

| source label | corpus | train | val | test | test drift (pp) |
|---|---|---|---|---|---|
| `weld_dataset/pore` | 54.2% (2,699) | 55.3% | 49.4% | 50.2% | -3.9 |
| `final_welding/Porosity` | 43.4% (2,165) | 42.3% | 48.1% | 47.8% | +4.3 |
| `final_welding/cluster of porosity` | 2.4% (120) | 2.4% | 2.5% | 2.0% | -0.4 |

`·` = below the 2% threshold, exempt from the bound.

## V3 `{N,U}` — 18,926 patches

Maximum test drift among pairs at ≥2% share: **1.2 pp**

| source label | corpus | train | val | test | test drift (pp) |
|---|---|---|---|---|---|
| `weld_dataset/discontinuity` | 35.5% (6,718) | 35.4% | 36.4% | 35.5% | +0.0 |
| `final_welding/Overlap` | 32.6% (6,162) | 33.0% | 28.1% | 33.5% | +0.9 |
| `final_welding/Lack of Fusion` | 17.3% (3,278) | 17.0% | 21.2% | 16.2% | -1.2 |
| `final_welding/Irregular` | 6.3% (1,194) | 6.3% | 6.6% | 6.5% | +0.2 |
| `final_welding/Burn Through` | 5.5% (1,049) | 5.6% | 5.6% | 5.2% | -0.4 |
| `final_welding/Excess-Reinforcement` | 2.1% (400) | 2.1% | 1.3% | 2.7% | +0.5 |
| `final_welding/Undercut` · | 0.5% (100) | 0.5% | 0.7% | 0.3% | -0.2 |
| `final_welding/root concvity` · | 0.1% (15) | 0.1% | 0.1% | 0.1% | -0.0 |
| `final_welding/excess root penetration` · | 0.1% (10) | 0.0% | 0.0% | 0.2% | +0.1 |

`·` = below the 2% threshold, exempt from the bound.

## V4 `{N,U,L,C}` — 19,851 patches

Maximum test drift among pairs at ≥2% share: **2.1 pp**

| source label | corpus | train | val | test | test drift (pp) |
|---|---|---|---|---|---|
| `weld_dataset/stain` | 36.7% (7,282) | 36.5% | 37.2% | 38.0% | +1.3 |
| `final_welding/Spatters` | 14.2% (2,819) | 14.8% | 11.5% | 12.1% | -2.1 |
| `weld_dataset/__negative__` | 14.1% (2,793) | 14.0% | 15.0% | 13.9% | -0.1 |
| `final_welding/__negative__` | 13.6% (2,696) | 13.5% | 14.6% | 13.3% | -0.3 |
| `weld_dataset/deposit` | 13.2% (2,625) | 13.1% | 13.2% | 13.8% | +0.6 |
| `wijaya/Good Weld` | 3.5% (689) | 3.5% | 3.4% | 3.6% | +0.1 |
| `final_welding/Arc strike` | 2.5% (496) | 2.5% | 2.7% | 2.7% | +0.2 |
| `wijaya/__negative__` · | 1.8% (360) | 1.8% | 1.9% | 1.8% | -0.0 |
| `final_welding/Crack` · | 0.3% (60) | 0.3% | 0.2% | 0.7% | +0.4 |
| `final_welding/mechanical-damage` · | 0.2% (31) | 0.2% | 0.2% | 0.1% | -0.1 |

`·` = below the 2% threshold, exempt from the bound.
