# Split composition

Assignment: **StratifiedGroupKFold(n_splits=10, groups=source_sequence)**

Acceptance: every (head, source_label) pair at or above 2% of its head must sit within ±5 percentage points of its corpus share in both val and test. Pairs below that share are exempt from the bound but are still listed.

- pairs checked: **10**
- pairs outside the bound: **10**

## Pairs outside the bound

| head | label | split | corpus | split | drift (pp) |
|---|---|---|---|---|---|
| V3 | `final_welding/Overlap` | val | 69.0% | 42.6% | -26.4 |
| V3 | `final_welding/Overlap` | test | 69.0% | 34.0% | -35.0 |
| V3 | `final_welding/Irregular` | val | 13.4% | 36.7% | +23.3 |
| V3 | `final_welding/Irregular` | test | 13.4% | 38.5% | +25.1 |
| V3 | `final_welding/Burn Through` | test | 11.7% | 23.0% | +11.3 |
| V4 | `final_welding/Lack of Fusion` | val | 34.9% | 48.8% | +13.8 |
| V4 | `final_welding/Lack of Fusion` | test | 34.9% | 41.1% | +6.2 |
| V4 | `final_welding/Spatters` | val | 30.1% | 5.0% | -25.0 |
| V4 | `final_welding/Spatters` | test | 30.1% | 23.8% | -6.2 |
| V4 | `final_welding/__negative__` | val | 28.7% | 39.6% | +10.9 |

## Head totals

| head | focal element | patches | class weight |
|---|---|---|---|
| V1 | `{P}` | 2,285 | 2.0007 |
| V3 | `{N,U}` | 8,930 | 0.5119 |
| V4 | `{N,U,L}` | 9,380 | 0.4874 |

## V1 `{P}` — 2,285 patches

Maximum test drift among pairs at ≥2% share: **0.5 pp**

| source label | corpus | train | val | test | test drift (pp) |
|---|---|---|---|---|---|
| `final_welding/Porosity` | 94.7% (2,165) | 94.6% | 95.5% | 95.3% | +0.5 |
| `final_welding/cluster of porosity` | 5.3% (120) | 5.4% | 4.5% | 4.7% | -0.5 |

`·` = below the 2% threshold, exempt from the bound.

## V3 `{N,U}` — 8,930 patches

Maximum test drift among pairs at ≥2% share: **35.0 pp**

| source label | corpus | train | val | test | test drift (pp) |
|---|---|---|---|---|---|
| `final_welding/Overlap` | 69.0% (6,162) | 70.6% | 42.6% | 34.0% | -35.0 |
| `final_welding/Irregular` | 13.4% (1,194) | 12.1% | 36.7% | 38.5% | +25.1 |
| `final_welding/Burn Through` | 11.7% (1,049) | 11.4% | 14.3% | 23.0% | +11.3 |
| `final_welding/Excess-Reinforcement` | 4.5% (400) | 4.7% | 0.4% | 1.5% | -3.0 |
| `final_welding/Undercut` · | 1.1% (100) | 1.0% | 4.4% | 3.0% | +1.9 |
| `final_welding/root concvity` · | 0.2% (15) | 0.1% | 1.6% | 0.0% | -0.2 |
| `final_welding/excess root penetration` · | 0.1% (10) | 0.1% | 0.0% | 0.0% | -0.1 |

`·` = below the 2% threshold, exempt from the bound.

## V4 `{N,U,L}` — 9,380 patches

Maximum test drift among pairs at ≥2% share: **6.2 pp**

| source label | corpus | train | val | test | test drift (pp) |
|---|---|---|---|---|---|
| `final_welding/Lack of Fusion` | 34.9% (3,278) | 33.6% | 48.8% | 41.1% | +6.2 |
| `final_welding/Spatters` | 30.1% (2,819) | 32.2% | 5.0% | 23.8% | -6.2 |
| `final_welding/__negative__` | 28.7% (2,696) | 28.1% | 39.6% | 27.3% | -1.4 |
| `final_welding/Arc strike` | 5.3% (496) | 5.1% | 6.5% | 7.1% | +1.9 |
| `final_welding/Crack` · | 0.6% (60) | 0.7% | 0.0% | 0.3% | -0.3 |
| `final_welding/mechanical-damage` · | 0.3% (31) | 0.4% | 0.0% | 0.3% | -0.0 |

`·` = below the 2% threshold, exempt from the bound.
