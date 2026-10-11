# 08 — Current status

**Last updated: 2026-10-09. Update this file whenever you finish a piece of work.**

## Overall

Specimens are welded and collected. The ultrasonic instrument arrives in the week of
13 October. The vision pipeline has produced a significant negative result and needs
a clean retrain. The geometry module has hardware configured but no captures yet.
Fusion code is specified but cannot be validated end to end until three aligned BPA
tables exist.

Roughly four weeks remain, of which the last week is writing only.

## Module status

| Module | State | Notes |
|---|---|---|
| Frame / focal elements | **Done** | Three-head vision frame settled; ceiling proposition proved |
| Vision | **Needs retrain** | Leakage finding complete; L→V4 remap outstanding; single-source retrain not yet run |
| Geometry | **Hardware ready, no data** | D435i configured at 300 mm; step block not yet made |
| Ultrasonic | **Blocked on instrument** | Prep work can and should proceed now |
| Fusion | **Specified, untested end to end** | Phase 2 implementable; needs real BPAs to validate |
| Evaluation | **Partially built** | Splits and metrics exist; clustered bootstrap and ceiling calculator outstanding |
| Registration | **Not built** | Datum holes being drilled; pose solve and cell indexer needed |

## Done

- Frame of discernment and per-modality focal elements declared and justified.
- Two-modality ceiling proposition proved and written up.
- Sequence-leakage audit: 81.59% contamination quantified.
- LOSO transfer evaluation: macro-F1 0.43 / 0.22, ECE 0.34 / 0.68.
- Correct chance baseline computed (0.3023) on the actual class distribution.
- Crop-geometry confound cleared (joint-matched deltas −0.032 / −0.045).
- Letterbox geometry solved in closed form: `pad_fraction = 1 − 1/AR`.
- Stratified split reducing composition overlap 32.6% → 1.2%.
- Class-weight derivation corrected to train-split-only.
- 45 weld zones produced with logged parameters; coupons collected.
- D435i configuration settled, including the 300 mm standoff correction.
- Datum hole geometry corrected to y = 50 mm after the plate width was confirmed at
  70 mm per half.
- Documentation repository layout updated to match actual codebase structure
  (`weld-docs/CLAUDE.md`, `10-conventions.md`, main `CLAUDE.md`). V4 focal element
  corrected to `{N,U,L,C}`. Loose prompt files consolidated into `claude prompts/`.
  Credentials file added to `.gitignore`.

## In flight

- Drilling datum holes; stamping coupon IDs; punch-marking zone boundaries.
- Clean vision and depth capture of all coupons, **before** any couplant touches them.
- Cell-level label file construction from the welding parameter log.
- UT feature-extractor dry run against synthetic A-scans.

## Blocked

| Item | Blocked on |
|---|---|
| UT feature extraction on real data | Instrument arrival, week of 13 Oct |
| Fusion end-to-end validation | All three BPA tables existing |
| Measured uncertainty budget | Step block fabrication |

## Open questions requiring a decision

| # | Question | Status |
|---|---|---|
| 1 | Box-area 270 vs 267 px² discrepancy | Unresolved. Re-check `predictions_fold0.csv`. Do not settle by recency. |
| 2 | McNemar comparison family size | Proposed m = 4; needs confirming and writing into code as a constant |
| 3 | Radiography — booked or cut? | Objective 6 currently promises it. Confirm or remove the clause. |
| 4 | ISO 5817 Level B only, or B and C? | Objective 3 says B; undercut at B is marginal at 6.4:1 |
| 5 | Third datum hole | Optional; decide before drilling is finished |

## Known stale text in the thesis document

These will read badly against the actual results if not fixed:

- Objective 4 and the Scope section both state that ultrasonic instrumentation "was
  not secured". It has been hired. Reword to cover a reduced-scan-density outcome.
- Objective 6 promises radiographic examination — see open question 3.
- `system-configuration-v2.md` still records the datum hole position as y = +68 mm.
  Correct to y = 50 mm.

## Hard constraints to respect when planning

- The ultrasonic hire window cannot be moved or repeated.
- Couplant contaminates the surface and promotes flash rust, so **all optical capture
  must precede all ultrasonic testing**. This ordering is irreversible.
- Flash rust is already appearing on the as-welded surfaces. Capture early, and treat
  every coupon identically — selectively cleaning some would confound surface
  condition with capture order, which correlates with coupon, which is the fold group.
- **Experimental freeze 1 November.** After that, writing only.
