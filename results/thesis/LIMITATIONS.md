# Chapter 4 Limitations

## 1. Geometry Coverage

39 of 150 cells decided (26%). Zone C (cells 7-9) entirely vacuous across all 15 coupons.
Coverage by cell index:
- Index 0: 0%
- Index 1: 40%
- Index 2: 80%
- Index 3: 67%
- Index 4-6: 13-40%
- Index 7-9: 0%

Cause: insufficient valid depth pixels near coupon ends. The asymmetric pattern (Zone A partial,
Zone C complete dropout) is not fully explained by arc-start/arc-stop transients alone.

## 2. Two Coupons Entirely Vacuous

C04 (Sound) and C13 (Misaligned) have zero geometry coverage. C13 is the only misalignment
coupon, so misalignment has no geometry evidence.

## 3. Geometry Reliability

Alpha derived from 10 undercut cells pooled across folds. Per-fold estimation was not possible
due to fold counts of 2/4/2/2. The pooled estimator is biased toward folds with more samples.

Alpha_G = 1.0000 (clamped to 1.0). Geometry performs worse than trivial baseline because
all 39 decided cells predict {U} due to misalignment exceedance, but only 10 are true {U}.

## 4. Vacuity Independence

Chi-squared test: chi2 = 4.17, df = 3, p = 0.24. Cannot reject independence, but power is low
at n = 39. The vacuity pattern may or may not be independent of true class.

## 5. Vision Calibration

Old temperature: 9.999994 (hit upper bound 10.0)
New temperature: 100.0000
NEW TEMPERATURE HIT BOUNDARY 100.0 - calibration unconverged

ECE improved from 0.3972 to 0.0819.

## 6. Uncertainty Budget

u_align = 0.020 mm is a placeholder, not measured. The total predicted uncertainty (0.083 mm)
underestimates the measured repeatability (0.275 mm) by a factor of 3.3.

## 7. No Human Inspector Baseline

Section 4.5 was cut. No inspector assessment data exists. Human comparison moves to Chapter 5
as a limitation.

## 8. Effective Sample Size

Design effect: 5.78
n_eff: 26.0

Confidence intervals are wide. Per-class results for small categories are indicative only.

## 9. {L} Decisions Under V+G

Theory predicts zero {L} decisions under V+G (neither modality can assign mass to {L} alone).
Observed: 0 (as expected)



## 10. Bootstrap Resamples

2000 coupon-level resamples used for all confidence intervals.
