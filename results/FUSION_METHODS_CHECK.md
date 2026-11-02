# Fusion Module Methodology Verification

## Run Identity

**Entry script**: `scripts/chapter4_execute.py`
**Config**: Hardcoded in script (lines 35–46)
**BPA input files**:
- Vision: `results/vision/coupon_eval/vision_bpas_v2.json`
- Geometry: `results/geometry/bpa_geometry.csv`
- Ultrasonic: `results/ut/bpa_ut.csv`

**Fusion output files**:
- `results/fusion/fusion_results.json`
- `results/fusion/statistics.json`

**Combination methods**: Dempster and Murphy (both computed in parallel)

---

## Verification Table

### Item 0: Run Identity

| # | Item | Status | Actual Value | Evidence | Quote |
|---|------|--------|--------------|----------|-------|
| 0.1 | Entry script | **MATCH** | `scripts/chapter4_execute.py` | `scripts/chapter4_execute.py:1-7` | `\"Chapter 4 Execution Script\"` |
| 0.2 | Vision BPA source | **MATCH** | `results/vision/coupon_eval/vision_bpas_v2.json` | `chapter4_execute.py:510` | `\"with open(PROJECT_ROOT / 'results' / 'vision' / 'coupon_eval' / 'vision_bpas_v2.json')\"` |
| 0.3 | Geometry BPA source | **MATCH** | `results/geometry/bpa_geometry.csv` | `chapter4_execute.py:516` | `\"with open(PROJECT_ROOT / 'results' / 'geometry' / 'bpa_geometry.csv')\"` |
| 0.4 | UT BPA source | **MATCH** | `results/ut/bpa_ut.csv` | `chapter4_execute.py:523` | `\"with open(PROJECT_ROOT / 'results' / 'ut' / 'bpa_ut.csv')\"` |
| 0.5 | Fusion output | **MATCH** | `results/fusion/fusion_results.json` | `chapter4_execute.py:646` | `\"with open(PROJECT_ROOT / 'results' / 'fusion' / 'fusion_results.json', 'w')\"` |

---

### Item 1: Discount Factors

| # | Item | Status | Actual Value | Evidence | Quote |
|---|------|--------|--------------|----------|-------|
| 1.1 | Shafer discount formula | **MATCH** | m^α(A) = (1−α)·m(A), m^α(Θ) = α | `chapter4_execute.py:420-424` | `\"m_P = (1 - alpha_V) * p[0]; m_Theta = alpha_V\"` |
| 1.2 | α formula | **MATCH** | α = 1 − (F1 − T)/(1 − T) | `chapter4_execute.py:176-178` | `\"skill = (f1_pooled - TRIVIAL_BASELINES['geometry']) / (1 - ...); alpha_G = 1 - skill\"` |
| 1.3a | Trivial baseline: Vision | **MATCH** | 0.2568 | `chapter4_execute.py:40` | `\"'vision': 0.2568\"` |
| 1.3b | Trivial baseline: Geometry | **MATCH** | 0.4265 | `chapter4_execute.py:41` | `\"'geometry': 0.4265\"` |
| 1.3c | Trivial baseline: UT | **MATCH** | 0.2551 | `chapter4_execute.py:42` | `\"'ultrasonic': 0.2551\"` |
| 1.4a | F1 estimator: Vision | **MATCH** | mean-of-folds | `calibration_v2.json:46` | `\"estimator\": \"mean_of_folds\"` |
| 1.4b | F1 estimator: Geometry | **MATCH** | pooled (over 39 decided cells) | `alpha_derivation.json:13` | `\"estimator\": \"pooled\"` |
| 1.4c | F1 estimator: UT | **MATCH** | mean-of-folds | `metrics_ut.json:46` | `\"f1_macro_source\": \"mean_of_folds\"` |
| 1.5 | α clamp to [0, 1] | **MATCH** | Clamped | `chapter4_execute.py:178` | `\"alpha_G = max(0.0, min(1.0, alpha_G))\"` |
| 1.6a | α_V used | **MATCH** | 0.909 | `calibration_v2.json:23` | `\"alpha_V\": 0.909296351710439` |
| 1.6b | α_G used | **MATCH** | 1.0 (clamped) | `alpha_derivation.json:5-6` | `\"alpha_G\": 1.0, \"clamped\": true` |
| 1.6c | α_U used | **MATCH** | 0.241 | `alpha_ut.json:2` | `\"alpha\": 0.24055830183192806` |

---

### Item 2: Combination

| # | Item | Status | Actual Value | Evidence | Quote |
|---|------|--------|--------------|----------|-------|
| 2.1 | Dempster implementation | **MATCH** | Returns DempsterResult with K | `src/weldfusion/fusion/dempster.py:32-87` | `\"Returns the combined BPA and the conflict mass K\"` |
| 2.2 | Murphy implementation | **MATCH** | Average then combine (n−1) times | `src/weldfusion/fusion/murphy.py:65-76` | `\"avg_bpa = average_bpas(valid_bpas); for _ in range(n - 1): result = dempster_combine(...)\"` |
| 2.3 | Excluded modality handling | **MATCH** | Vacuous {Θ: 1.0} = identity | `dempster.py:50-54` | `\"if bpa1 is None or _is_vacuous(bpa1): return DempsterResult(bpa=dict(bpa2) if bpa2 else {}, K=0.0)\"` |
| 2.4 | K stored per cell | **MATCH** | Stored in results | `fusion_results.json:7` | `\"K\": 0.0` (per cell) |
| 2.5 | BPA sum = 1 verified | **MATCH** | Asserted in geometry BPA | `chapter4_execute.py:479-480` | `\"assert abs(total - 1.0) < 1e-9\"` |
| 2.6 | 7 subsets computed | **MATCH** | V, G, U, VG, VU, GU, VGU | `chapter4_execute.py:568-576` | `\"subsets = [frozenset({'V'}), ..., frozenset({'V', 'G', 'U'})]\"` |

---

### Item 3: Decision and Ties

| # | Item | Status | Actual Value | Evidence | Quote |
|---|------|--------|--------------|----------|-------|
| 3.1 | Pignistic transform | **MATCH** | BetP(x) = Σ m(A)/\|A\| | `decision.py:35-43` | `\"share = mass / cardinality; for x in focal: betp[x] += share\"` |
| 3.2 | Tie-breaking | **MATCH** | Alphabetical (min) | `decision.py:87-89` | `\"candidates = [x for x, p in betp.items() if p == max_prob]; return min(candidates)\"` |
| 3.3 | Bel ≤ Pl invariant | **MATCH** | Tested and passes | `test_fusion_module.py:141-154` | `\"assert bel <= pl + 1e-12\"` |
| 3.4 | [Bel, Pl] stored | **MATCH** | Stored per element | `fusion_results.json:14-30` | `\"bel_pl\": {\"P\": [0.027, 0.937], ...}` |

---

### Item 4: Tests (pytest)

| Test | Status | Evidence |
|------|--------|----------|
| TestVacuousIdentity::test_vacuous_left | **PASS** | `pytest tests/test_fusion_module.py -v` |
| TestVacuousIdentity::test_vacuous_right | **PASS** | `pytest tests/test_fusion_module.py -v` |
| TestConflict::test_conflict_K | **PASS** | Hand-computed K=0.72 verified |
| TestConflict::test_conflict_masses | **PASS** | m({P})=0.286, m({L})=0.643, m(Θ)=0.071 |
| TestPignisticCardinality::test_pignistic_values | **PASS** | BetP(N)=BetP(U)=BetP(L)=0.30, BetP(P)=0.10 |
| TestCommutativity::test_commutativity | **PASS** | m1⊕m2 = m2⊕m1 |
| TestAssociativity::test_associativity | **PASS** | (m1⊕m2)⊕m3 = m1⊕(m2⊕m3) |
| TestBelPl::test_bel_le_pl | **PASS** | Bel ≤ Pl for all singletons |
| TestBelPl::test_bel_le_pl_compound | **PASS** | Bel ≤ Pl for all pairs |
| TestMassSum::test_mass_sum | **PASS** | Combined BPA sums to 1.0 |

**Test summary**: 10 passed, 0 failed

---

## Critical Findings

### 1. Geometry Fully Discounted (α_G = 1.0)

**Status**: Expected given the data
**Reason**: F1_pooled (0.204) < trivial baseline (0.4265), so α = 1.0 and geometry contributes no evidence.

**Evidence**:
```json
// results/geometry/alpha_derivation.json
{
  "f1_pooled": 0.2040816326530612,
  "trivial_baseline": 0.4265,
  "alpha_G": 1.0,
  "clamped": true
}
```

**Impact**: Under all conditions including G, geometry adds no information. The GV condition equals V-only; the GU condition equals U-only.

---

### 2. Vision Temperature Hit Boundary

**Status**: Documented limitation
**Reason**: Temperature scaling optimisation hit upper bound (100.0)

**Evidence**:
```json
// results/vision/coupon_eval/calibration_v2.json
{
  "old_temperature": 9.999994,
  "new_temperature": 99.999994,
  "hit_boundary": true,
  "ece_pre": 0.397,
  "ece_post": 0.082
}
```

**Impact**: Calibration improved ECE from 0.40 to 0.08, but the optimizer did not converge naturally. Results should note this.

---

### 3. {L} Unreachable Under V+G (Proposition 3.2)

**Status**: Verified correct

**Reasoning**:
- Vision focal elements: {P}, {N,U}, {N,U,L}, Θ
- Geometry focal elements: {U}, {N,P,L}, Θ
- Neither modality assigns mass to {L} alone
- Under Dempster's rule, {L} can only receive mass if both sources agree on a set containing L
- {N,U,L} ∩ {N,P,L} = {N,L} — still not {L} alone
- Therefore, no singleton {L} mass can emerge from V+G combination

**Evidence**: `fusion_results.json` shows GV condition has `l_count: 0`

---

### 4. UT Dominates Fusion Results

**Observation**: All conditions including U produce identical F1 (0.482)

**Evidence**:
```json
// results/fusion/statistics.json
"U_dempster": {"f1": 0.4822916666666667},
"UV_dempster": {"f1": 0.4822916666666667},
"GU_dempster": {"f1": 0.4822916666666667},
"GUV_dempster": {"f1": 0.4822916666666667}
```

**Root cause**:
- α_U = 0.241 (76% of UT evidence retained)
- α_V = 0.909 (9% of vision evidence retained)
- α_G = 1.0 (0% of geometry evidence retained)
- UT dominates because it retains far more evidence

---

### 5. No Uncertainty Floor in Current Pipeline

**Status**: Correct (matches thesis)
**Evidence**: Searched `chapter4_execute.py` — no `uncertainty_floor` parameter.

**Note**: The deprecated `src/fusion/dempster_shafer.py` had `uncertainty_floor=0.05`, but this module is NOT used by the current fusion pipeline.

---

## Confirmed Matches

1. **Shafer discount formula**: m^α(A) = (1−α)·m(A), m^α(Θ) = α ✓
2. **Alpha formula**: α = 1 − (F1 − T)/(1 − T), clamped to [0,1] ✓
3. **Trivial baselines**: V=0.2568, G=0.4265, U=0.2551 ✓
4. **F1 estimators**: V=mean-of-folds, G=pooled, U=mean-of-folds ✓
5. **Dempster's rule**: K computed and stored ✓
6. **Murphy's rule**: Average then combine (n−1) times ✓
7. **Excluded modality**: Vacuous {Θ: 1.0}, identity under Dempster ✓
8. **Pignistic transform**: BetP(x) = Σ m(A)/|A| ✓
9. **Tie-breaking**: Alphabetical (min) ✓
10. **Belief ≤ Plausibility**: Tested and verified ✓

---

## Text Changes Required

None identified. The fusion methodology text matches the implementation.

---

## Ablation Blocking Items

**None.** All fusion components verified:
- BPA inputs exist and are correctly formatted
- Shafer discounting applied per-modality
- Alpha values computed and stored
- Dempster/Murphy combinations implemented correctly
- K, BetP, [Bel,Pl] all stored
- Tests pass

---

## Hyperparameter Summary

### Discount Factors

| Modality | F1 | Trivial | α | Evidence Retained | Estimator |
|----------|----|---------|----|-------------------|-----------|
| Vision | 0.324 | 0.2568 | 0.909 | 9.1% | mean-of-folds |
| Geometry | 0.204 | 0.4265 | 1.0 (clamped) | 0% | pooled |
| Ultrasonic | 0.821 | 0.2551 | 0.241 | 75.9% | mean-of-folds |

### Combination Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Frame | {N, U, P, L} | `chapter4_execute.py:35` |
| Combination methods | Dempster, Murphy | `chapter4_execute.py:617-624` |
| Ablation subsets | 7 (all combinations of {V, G, U}) | `chapter4_execute.py:568-576` |
| Bootstrap resamples | 2000 | `chapter4_execute.py:45` |

### Decision Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Decision rule | argmax BetP | `decision.py:87` |
| Tie-breaking | Alphabetical (min) | `decision.py:89` |
| Confidence interval | 95% coupon-block bootstrap | `statistics.json:2000 resamples` |

### Statistical Tests

| Test | Comparison | b | c | χ² | p_Holm | Significant |
|------|------------|---|---|-----|--------|-------------|
| McNemar | G→GU | 63 | 5 | 47.8 | 1.9e-11 | Yes |
| McNemar | V→UV | 77 | 15 | 40.4 | 6.1e-10 | Yes |
| McNemar | GV→GUV | 77 | 15 | 40.4 | 4.0e-10 | Yes |
| McNemar | V→GV | 0 | 0 | 0.0 | 1.0 | No |

---

## Provenance

- **Verification date**: 2026-10-29
- **Entry script**: `scripts/chapter4_execute.py`
- **Core modules**: `src/weldfusion/fusion/dempster.py`, `murphy.py`, `decision.py`
- **Test file**: `tests/test_fusion_module.py` (10/10 passed)
- **Results files**: `results/fusion/fusion_results.json`, `results/fusion/statistics.json`
- **Alpha sources**: `results/vision/coupon_eval/calibration_v2.json`, `results/geometry/alpha_derivation.json`, `results/ut/alpha_ut.json`
