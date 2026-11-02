# Phase A Diagnostic — Chapter 4 Results Readiness

**Date:** 25 October 2026
**Auditor:** Claude Opus 4.5
**Mode:** Read-only diagnostic; no writes to `results/` except this report.

---

## A1. Per-Cell Evidence Inventory

The fusion step requires three mass-function files keyed on the same 150 cell identifiers.

### Vision BPA

| Item | Value |
|------|-------|
| Path | `results/vision/coupon_eval/vision_bpas.json` |
| Status | **EXISTS** |
| Row count | 150 (exact match) |
| Key column | `cell_id` |
| Key match to labels | **EXACT** (verified against `data/labels_v1_2026-10-17.csv`) |
| Mass columns | `{P}`, `{N,U}`, `{N,U,L}`, `Theta` |
| Declared focal elements match | **YES** |
| Mass sum per row | 1.0 (checked: all rows use α = 0.8820008465037826, masses = (1-α) × calibrated probs) |
| Negative masses | None |
| Undeclared focal elements | None |

**No issues.**

### Ultrasonic BPA

| Item | Value |
|------|-------|
| Path | `results/ut/bpa_ut.csv` |
| Status | **EXISTS** |
| Row count | 150 (exact match) |
| Key column | `cell_id` |
| Key match to labels | **EXACT** |
| Mass columns | `m_{L}`, `m_{N,U}`, `m_{P}`, `m_Theta` |
| Declared focal elements match | **YES** |
| Mass sum per row | 1.0 (all rows: α = 0.24055830183192806 constant) |
| Negative masses | None |
| Undeclared focal elements | None |

**No issues.**

### Geometry BPA

| Item | Value |
|------|-------|
| Path | `results/geometry/summary.csv` (also `summary.json`) |
| Status | **EXISTS** |
| Row count | 150 (exact match) |
| Key column | `cell_id` |
| Key match to labels | **EXACT** |
| Mass columns | `m_U`, `m_NPLC` |
| Declared focal elements | `{U}`, `{N,P,L}`, `Θ` |
| Declared focal elements match | **NO — FAULT** |
| Mass sum per row | Does NOT sum to 1.0 for non-vacuous rows |
| Vacuous rows | 111 of 150 (`is_vacuous=True`); these have `m_U=0`, `m_NPLC=0` |

**FAULTS:**

1. **Five-class naming residue.** Column is `m_NPLC` not `m_{N,P,L}`. This is a 5-class frame relic (the original had C for crack). The frame now declares `{N,P,L}` for geometry.

2. **No explicit `m_Theta` column.** Vacuous cells have `m_U=0, m_NPLC=0` implying all mass on Theta, but the column is absent. Non-vacuous cells have `m_U` + `m_NPLC` = 1.0 with no Theta residual.

3. **No α_G discounting applied.** The stored masses are raw exceedance probabilities, not Shafer-discounted. α_G has never been derived (see A4.1).

---

### Can a three-way join on cell_id proceed today?

**YES, structurally.** All three files contain exactly 150 rows with matching `cell_id` values. A join is mechanically possible.

**NO, semantically.** The geometry BPA is not in canonical form:
- Column naming does not match the 4-class frame.
- No α_G discount has been applied.
- Vacuous cells lack explicit Θ mass.

**Blockers for fusion execution:**
1. α_G must be derived first (B1.1).
2. Geometry BPA must be reformatted with proper column names and explicit Θ mass.

---

## A2. Fusion Implementation Readiness

### Dempster's Rule

| Item | Status |
|------|--------|
| Implementation exists | **YES** |
| Path | `src/fusion/dempster_shafer.py:30` |
| Entry point | `dempster_combine(bpa1, bpa2)` |

The implementation handles "theta" as a special key representing total ignorance. Conflict K is computed internally but **not returned** — only the normalised result.

### Murphy Averaging

| Item | Status |
|------|--------|
| Implementation exists | **YES** |
| Path | `src/fusion/dempster_shafer.py:69` |
| Entry point | `murphy_average(bpa_list)` |

Averages all BPAs, then combines the average with itself (n-1) times.

### Conflict Mass K

| Item | Status |
|------|--------|
| Computed | YES (internal to `dempster_combine`) |
| Stored per cell | **NO** |
| Returned | NO — only the normalised BPA is returned |

**FAULT:** K must be recorded per cell for §4.4 reporting. The `src/weldfusion/pipeline/phase2_percell.py` implementation DOES return K, but the simpler `src/fusion/dempster_shafer.py` does not.

### Ablation Driver

| Item | Status |
|------|--------|
| Implementation exists | **YES** |
| Path | `src/weldfusion/pipeline/phase2_percell.py:238` |
| Entry point | `run_all_ablations(table, modality_alphas)` |

The ablation is implemented as column selection: only BPAs from modalities in the subset are combined. However:

**FAULT:** The vacuous-BPA assertion is implicit. An excluded modality contributes `None`, which is filtered out before combination. This is **semantically correct** (Dempster's rule treats a missing argument as identity), but there is no explicit assertion that verifies this.

### Pignistic Transform

| Item | Status |
|------|--------|
| Referenced | YES — `src/weldfusion/pipeline/phase2_percell.py:29` imports `pignistic, decide` |
| Implementation exists | **NO** |
| Missing files | `src/weldfusion/fusion/decision.py`, `src/weldfusion/fusion/ablation.py`, `src/weldfusion/fusion/dempster.py`, `src/weldfusion/fusion/murphy.py` |

**FAULT:** The `src/weldfusion/fusion/` directory does not exist. The imports in `phase2_percell.py` will fail at runtime.

**CRITICAL:** Cannot verify whether the pignistic transform divides by |A| correctly for compound elements because the implementation is missing.

### End-to-End Fusion Execution

| Item | Status |
|------|--------|
| Has fusion ever run | **NO** — no fusion result files found |
| Expected outputs | `fused_bpa`, `conflict_K`, `betp`, `predicted_label` per cell per ablation condition |
| Found | None |

**BLOCKED:** Fusion has never been executed because:
1. The `src/weldfusion/fusion/` module is incomplete.
2. α_G has never been derived.
3. Geometry BPA is not in canonical form.

---

## A3. Section-by-Section Availability Matrix

| § | Deliverable | Status | Source / Blocker |
|---|---|---|---|
| 4.1 | Table: data completeness per modality per fold | **PARTIAL** | Missing: patches-per-cell for vision. Available: UT readings (150), geometry vacuous count (111), fold assignments. |
| 4.2 | Table: predicted vs measured uncertainty per measurand | **PARTIAL** | Geometry u_total_mm available; UT amplitude→class mapping available; vision calibration ECE available. |
| 4.2 | Figure: UT amplitude, {L} vs {P} | **AVAILABLE** | `results/ut/features.csv` has amplitude_pct with true_class |
| 4.2 | Vision verification: transfer gap | **AVAILABLE** | `results/vision/loso_report.md`, `loso_metrics.json` |
| 4.2 | Vision verification: nearest-neighbour audit | **AVAILABLE** | `results/diagnostics/sequence_leakage_report.md`, `nn_audit.json` |
| 4.2 | Vision verification: calibration before/after | **AVAILABLE** | `results/vision/coupon_eval/calibration_results.json` |
| 4.2 | Geometry verification: repeatability vs u | **AVAILABLE** | `results/geometry/_diagnostics/repeatability_*.json` |
| 4.2 | Geometry verification: undercut exclusion signal-to-u | **PARTIAL** | Undercut u_total = 0.275mm stored; signal analysis requires recomputation |
| 4.2 | UT verification: calibration, mapping, depth-folding | **AVAILABLE** | `results/ut/mapping_audit.md`, `metrics_ut.json` |
| 4.3 | Table: per-modality results (F1, α, ECE) | **PARTIAL** | Vision: complete. UT: complete. Geometry: **α_G MISSING**, no F1/ECE computed. |
| 4.3 | Figure: three confusion matrices | **PARTIAL** | Vision: derivable from `kfold_results.json`. UT: in `metrics_ut.json`. Geometry: **ABSENT** — deterministic, but no stored predictions. |
| 4.4 | Table: seven-condition fusion results | **ABSENT** | Fusion never executed. Blocked by missing fusion module and α_G. |
| 4.4 | Figure: per-class F1 as modalities added | **ABSENT** | Blocked by missing fusion results. |
| 4.5 | Human baseline | **ABSENT** | No inspector assessment data exists anywhere in the repo. |

---

## A4. Known Open Items — Confirm/Refute

### A4.1. α_G has never been derived

**CONFIRMED.**

- No `alpha_derivation.json` or equivalent exists for geometry.
- No script computes geometry F1 against the 150 cells.
- Geometry output is deterministic (exceedance probability), but it must still be scored against ground truth across folds to derive α.
- Per-cell geometry assignments exist in `results/geometry/summary.csv` (`m_U`, `is_vacuous`), so α_G **can be computed** without rerunning the pipeline.

**Required for Phase B:** Score geometry's `m_U > 0.5` → {U} decision (or vacuous → Θ) against `labels_v1:true_class`, compute macro-F1 per fold, derive α_G = 1 - (F1 - 0.4163)/(1 - 0.4163), clamped to [0,1].

### A4.2. Vision temperature sat on its search boundary

**CONFIRMED.**

| Value | Source |
|-------|--------|
| Optimal temperature | 9.999994023633219 |
| Search bounds (UT code, same library) | [0.1, 10.0] |
| Upper bound | **10.0** |

The temperature is 6 × 10⁻⁶ below the upper bound. The calibration is **unconverged**. ECE = 0.136 is an upper bound, not a minimum.

**Required for Phase B:** Re-run temperature scaling with bounds [0.1, 50.0] or [0.1, 100.0] to find the true optimum. Report both values.

### A4.3. Vision pooled macro-F1 was never computed

**CONFIRMED.**

| Metric | Value | Source |
|--------|-------|--------|
| Mean-of-folds macro-F1 | 0.3242 | `kfold_results.json:mean_macro_f1` |
| Pooled macro-F1 | **NOT STORED** | — |

The per-fold `y_true` and `y_pred` arrays are stored in `kfold_results.json`. Pooling them and computing macro-F1 gives approximately 0.353 (needs verification).

**Impact on α:**
- Mean-of-folds α_V: 1 - (0.3242 - 0.2338)/(1 - 0.2338) = 0.882
- Pooled α_V: 1 - (0.353 - 0.2338)/(1 - 0.2338) ≈ 0.844

**Required for Phase B:** Verify pooled F1 from stored predictions. Report both estimators.

### A4.4. Which partition produced the LOSO transfer figures?

**CONFIRMED: `source_frame` grouping (old split).**

From `results/vision/loso_report.md` line 163:
> "itself leakage-guarded at the `source_frame` level."

However, this is **not a problem** for LOSO. The LOSO folds train and test on **disjoint datasets by construction** (hold out entire source dataset), so sequence leakage within a dataset cannot inflate the transfer gap. The `source_frame` vs `source_sequence` distinction applies to the in-distribution validation split within each LOSO fold, not to the holdout test set.

**The LOSO figures may be cited in §4.2 without qualification.**

### A4.5. Vision per-class F1 and confusion matrix

**CONFIRMED: Not in canonical JSON, but DERIVABLE.**

`kfold_results.json` contains per-fold arrays:
- `y_true`: ground truth indices (0=V1, 1=V3, 2=V4)
- `y_pred`: predicted indices
- `confusion_matrix`: stored per fold

Per-class F1 can be computed from pooled predictions using `sklearn.metrics.f1_score(average=None)`.

---

## Closing Assessment

### Buildable Today with Zero Blockers

| Section | Deliverable |
|---------|-------------|
| 4.2 | UT amplitude figure |
| 4.2 | Vision transfer gap (LOSO report) |
| 4.2 | Vision nearest-neighbour audit |
| 4.2 | Geometry repeatability diagnostics |
| 4.2 | UT mapping and calibration checks |

### Buildable After One Decision

| Decision | Owner | Affected Sections |
|----------|-------|-------------------|
| Label convention (Overfill/Misaligned → U or N) | Ben | 4.3 (baselines, α values), 4.4 (all fusion metrics) |

**Impact of decision:**
- Current: Overfill(10) + Misaligned(3) → U. Trivial baselines: V 0.2338, G 0.4163.
- Design-doc: Overfill + Misaligned → N. Trivial baselines: V 0.2568, G 0.4444.
- α values shift by ~3 percentage points.

### Blocked on Computation

Execute in this order:

1. **Derive α_G.** Score geometry assignments (`m_U > 0.5` → {U}, else Θ) against `labels_v1:true_class` over 4 folds. Apply α formula with baseline 0.4163 (or 0.4444 under design-doc). Clamp to [0,1].

2. **Re-run vision temperature scaling.** Widen bounds to [0.1, 50.0]. Report new temperature and ECE.

3. **Compute vision pooled macro-F1.** Concatenate `y_true`, `y_pred` from all folds, compute macro-F1. Report alongside mean-of-folds.

4. **Reformat geometry BPA.** Rename `m_NPLC` → `m_{N,P,L}`, add explicit `m_Theta` column, apply α_G discount.

5. **Complete fusion module.** Implement or locate `src/weldfusion/fusion/decision.py` (pignistic transform), `dempster.py`, `murphy.py`, `ablation.py`.

6. **Execute fusion for all 7 ablation subsets.** Store `fused_bpa`, `conflict_K`, `betp`, `predicted_label` per cell.

7. **Compute §4.4 metrics.** Pignistic macro-F1 with clustered-bootstrap CI, set-valued coverage, cardinality, K statistics.

### Not Recoverable

| Item | Reason |
|------|--------|
| §4.5 Human baseline | No inspector assessment data exists anywhere in the repository. The section must be cut or marked as future work. |
| Fusion results | Cannot produce from existing artefacts — requires code completion and execution. |

---

## Summary of Faults Found

| # | Severity | Description |
|---|----------|-------------|
| 1 | **HIGH** | Fusion module incomplete (`src/weldfusion/fusion/` missing) |
| 2 | **HIGH** | α_G never derived |
| 3 | **HIGH** | Vision temperature calibration unconverged (hit boundary 10.0) |
| 4 | **MEDIUM** | Geometry BPA uses 5-class column naming (`m_NPLC`) |
| 5 | **MEDIUM** | Geometry BPA lacks explicit Θ mass column |
| 6 | **MEDIUM** | Geometry BPA not α-discounted |
| 7 | **LOW** | Vision pooled F1 not computed (derivable from stored predictions) |
| 8 | **LOW** | Conflict K not returned by basic Dempster implementation |
| 9 | **INFO** | No inspector data for §4.5 (not a fault — data was never collected) |

---

**END OF PHASE A. Awaiting Ben's instruction to proceed to Phase B.**
