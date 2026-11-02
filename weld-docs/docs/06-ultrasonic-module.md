# 06 — Ultrasonic module

The only modality that can separate L from C (see the ceiling proposition in
`02-evidence-framework.md`). Trained on real specimen data only — there is no public
corpus of angle-beam A-scans from this geometry.

## Instrument

**Evident EPOCH 650**, hired for a short window (~AU$170/day). The hire window is the
only immovable deadline in the project — everything else is software and can be rerun.

| Setting | Value |
|---|---|
| Probes | **60° primary**, 70° secondary, 4 MHz |
| Index | **25 mm** along s (~180 positions total) |
| Calibration | IIW V1/V2 block, 0° probe for thickness, couplant |

### Why 60° and not 70°

For sidewall lack of fusion on a single-V bevel of half-angle β, normal incidence on
the fusion face requires

```
θ = 90° − β
```

With β = 30°, θ = 60°. A 70° beam strikes the sidewall 10° off normal and loses
signal. Sanity check: β = 0 (a straight sidewall) gives θ = 90°, a surface-skimming
beam, which is correct.

70° is retained as a secondary angle for toe-region and near-surface coverage.

**An earlier recommendation of 70° as primary was wrong.** Do not reintroduce it.

### Depth folding

With a half-skip path, the apparent depth from surface distance u is

```
z = u mod 2t
```

The feature extractor must unfold this. **Leg number is itself a feature** — see below.

For t = 10 mm at 60°: half-skip surface distance = t·tan θ = 17.3 mm; full skip =
34.6 mm. Working band is roughly 15–50 mm from the weld centreline.

## Feature vector — seven features (revised 2026-10-23)

**The original spec assumed waveform access. The EPOCH 650 CSV export contains
only gate readings — no waveform array.** The following features are therefore
**unavailable**:

| Original feature | Status | Reason |
|---|---|---|
| `signal_width_6db_us` | unavailable | requires waveform |
| `snr_db` | unavailable | requires waveform or noise measurement |
| `echo_count` | unavailable | one reading per cell, so always 0 or 1 |
| `geometry_echo_db` | unavailable | no separate geometry echo logged |
| `backwall_echo_db` | **invalid** | angle-beam has no backwall — delete any reference |

**Available features from EPOCH 650 gate readings:**

| Feature | Source column | Null value | Notes |
|---|---|---|---|
| `located` | presence of `Gate1 SoundPath` | 0 | 1 if indication present |
| `amplitude_pct` | `Gate1 Current Amplitude` | 0 | % FSH, not dB |
| `soundpath_mm` | `Gate1 SoundPath` | 0 | equivalent to ToF × velocity/2 |
| `depth_mm` | `Gate1 Depth to Reflector` | 0 | after instrument unfolding |
| `surface_distance_mm` | `Gate1 Surface Distance` | 0 | from beam index point |
| `saturated` | amplitude >= 100 | 0 | indicates clipping |
| `dist_from_root_mm` | `abs(soundpath - 20.0)` | 0 | root-band ambiguity feature |

**Why the root-band feature:** At 60° in 10 mm plate the half-skip soundpath is
`t/cos(60°) = 20.0 mm`. This is simultaneously the leg 1 / leg 2 boundary, the
root-face depth, and where root protrusion sits. Indications near 20 mm are
ambiguous by construction, so the distance from that band is given explicitly.

## Classifier (revised 2026-10-23)

**Multinomial logistic regression** with standardisation and `class_weight="balanced"`.
At n=150 with 7 features and effective sample size ~27 (after coupon clustering),
a deep model would overfit badly. k=4 coupon-grouped folds from the label file.

**XGBoost and Random Forest have been cut** from scope for the UT modality — cite
as future work. The small sample size makes logistic regression the appropriate
choice.

**Do not use isotonic regression for calibration** — with 150 points across three
classes it will overfit. Use temperature scaling if ECE > 0.15, fitted within
training folds only.

Actual class counts (4-class experimental frame, C removed):

| Focal element | Count |
|---|---|
| `{N,U}` | 93 |
| `{L}` | 31 |
| `{P}` | 26 |

Total: 150 cells from 15 coupons.

## Export risk and the capture fallback

**The waveform export path is not confirmed.** The supplier had no sample CSV, and
Appendix B of the EPOCH 650 manual is a setup-parameter list, **not** a waveform data
dictionary. It is not established that the instrument exports the 1024-point A-scan
array the feature extractor was written against.

**Capture three ways at every position:**

1. CSV / instrument data file export, if available.
2. **BMP screen capture** of the A-scan at every position (the EPOCH 650 does this
   reliably).
3. **Continuous video** of the screen across the scan, as a backstop.

If (1) fails, features are read from (2): amplitude in dB, time-of-flight, 6 dB width,
echo count and SNR are all readable from a gated screen grab, by hand or via a small
image pipeline. That degrades precision, not feasibility.

**Record which path was actually used.** It is a methods-section fact and a declared
limitation.

## Preparation to complete before the instrument arrives

Every hour of debugging done beforehand is an hour not lost with the instrument on
the clock.

- [ ] Dry-run the feature extractor closed-loop against synthetic A-scans: feed a
      known time-of-flight and amplitude in, confirm the same values come out.
- [ ] Depth-unfolding unit test: known z, forward-fold to u, unfold, assert recovery.
- [ ] Pre-print the index sheet: ~180 rows, coupon ID / s / filename / amplitude /
      ToF / notes.
- [ ] Verify the screen capture actually writes where expected — on position 1, not
      position 180.

## Synthetic A-scans

A physically grounded synthetic generator exists (`synthetic-data-spec.md`). Its role
is **unit-testing the feature extractor**, not training the classifier. Training on
synthetic and testing on real would be a weak result. If instrument access fails
entirely, the contingency in Objective 4 applies and the substitution must be
declared explicitly in the results.

## Datum hole interference

A through-hole is a reflector. The main datum holes sit at s = 15 and s = 285, which
are in the excluded end cells, so interference there is harmless. The optional third
hole at s = 150 is in a live cell — check for a spurious indication on the first scan
and note it if present.
