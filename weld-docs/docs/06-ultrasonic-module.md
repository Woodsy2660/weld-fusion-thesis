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

## Feature vector — nine features

```
peak_amplitude_db
time_of_flight_us
signal_width_6db_us
snr_db
geometry_echo_db        replaces "backwall" — no backwall exists in angle-beam
echo_count
flaw_depth_mm           after unfolding
flaw_offset_mm
leg_number
```

`geometry_echo_db` is a common error source: a normal-beam pipeline would use a
backwall echo, but angle-beam inspection has no backwall. Use the geometry echo.

## Classifier

Random Forest. Isotonic regression for calibration, fitted on inner validation folds
only. Nested GroupKFold grouped by `coupon_id`.

**XGBoost has been cut** from scope — cite as future work.

Expected class counts at 25 mm cells:

| Focal element | Approx. count |
|---|---|
| `{N,U}` | 112 |
| `{L}` | 32 |
| `{P}` | 28 |
| `{C}` | 16 |

Thin but trainable for a Random Forest on nine features. The `{C}` arm is the weak
one — report it as attemptable, not conclusive.

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
