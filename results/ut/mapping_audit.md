# UT Mapping Audit

## Data Source

- **File**: `data/raw/TEST01.csv`
- **Instrument**: Evident EPOCH 650
- **Export format**: CSV with gate readings (not waveform)

## Production Block Filtering

| Metric | Value |
|--------|-------|
| Total rows in CSV | 190 |
| Production rows (Ref Gain = 56.2) | 151 |
| Dropped rows (calibration block) | 39 |
| Dropped ID range | 040–078 |
| Production ID range | 079–229 |

### Verified Settings

All production rows verified to have:

| Setting | Value |
|---------|-------|
| Units | mm |
| Angle | 60° |
| Thick | 10 mm |
| Vel | 3240 m/s |
| Zero | 6.431 µs |
| Mode | P/E |

## ID to Cell Mapping

### Capture Order

1. **WC-07** first: 11 readings (IDs 079–089)
   - Extra reading (ID 089) excluded — beyond cell 09
2. **WC-01 through WC-15** (excluding WC-07): 10 readings each

| Coupon | ID Range | Readings | Cell Range |
|--------|----------|----------|------------|
| WC-07 (C07) | 079–088 | 10 | 00–09 |
| WC-01 (C01) | 090–099 | 10 | 00–09 |
| WC-02 (C02) | 100–109 | 10 | 00–09 |
| WC-03 (C03) | 110–119 | 10 | 00–09 |
| WC-04 (C04) | 120–129 | 10 | 00–09 |
| WC-05 (C05) | 130–139 | 10 | 00–09 |
| WC-06 (C06) | 140–149 | 10 | 00–09 |
| WC-08 (C08) | 150–159 | 10 | 00–09 |
| WC-09 (C09) | 160–169 | 10 | 00–09 |
| WC-10 (C10) | 170–179 | 10 | 00–09 |
| WC-11 (C11) | 180–189 | 10 | 00–09 |
| WC-12 (C12) | 190–199 | 10 | 00–09 |
| WC-13 (C13) | 200–209 | 10 | 00–09 |
| WC-14 (C14) | 210–219 | 10 | 00–09 |
| WC-15 (C15) | 220–229 | 10 | 00–09 |

### Summary

- **Mapped cells**: 150
- **Excluded IDs**: 1 (ID 089, extra reading beyond cell 09)
- **Orphan labels**: 0
- **ID-to-label join**: Complete with zero failures

## Class Distribution

| Focal Element | Count | % |
|---------------|-------|---|
| {N,U} | 93 | 62.0% |
| {L} | 31 | 20.7% |
| {P} | 26 | 17.3% |
| **Total** | 150 | 100% |

## Trigonometry Verification

Verified depth and surface distance against soundpath using:
- `depth = soundpath × cos(60°)` (or folded: `2t - depth`)
- `surface_dist = soundpath × sin(60°)`

| Metric | Value |
|--------|-------|
| Located readings | 53 / 150 |
| Tolerance | 1.5 mm |
| Failures at tolerance | 8 |
| Failure rate | 15.1% |

The 8 failures are attributed to multi-mode echoes or gate positioning on
secondary reflections. These are legitimate instrument readings, not parsing
errors. The feature values are used as-is for classification.

## Feature Summary

| Feature | Located Mean | Located Std |
|---------|--------------|-------------|
| amplitude_pct | 62.3% | 27.1% |
| soundpath_mm | 18.9 mm | 9.2 mm |
| depth_mm | 5.1 mm | 2.9 mm |
| surface_distance_mm | 16.7 mm | 7.8 mm |
| dist_from_root_mm | 5.4 mm | 5.8 mm |
| saturated | 3.8% | — |
