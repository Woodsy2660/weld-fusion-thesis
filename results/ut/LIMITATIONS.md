# UT Modality Limitations

## Unavailable Features

The original feature specification assumed waveform access. The EPOCH 650 CSV
export contains only gate readings — no waveform array. The following features
are therefore **unavailable**:

| Feature | Status | Reason |
|---------|--------|--------|
| `signal_width_6db_us` | Unavailable | Requires waveform |
| `snr_db` | Unavailable | Requires waveform or noise measurement |
| `echo_count` | Unavailable | One reading per cell, so always 0 or 1 |
| `geometry_echo_db` | Unavailable | No separate geometry echo logged |
| `backwall_echo_db` | **Invalid** | Angle-beam has no backwall |

## Available Features (7 total)

| Feature | Source Column | Description |
|---------|---------------|-------------|
| `located` | Presence of Gate1 SoundPath | 1 if indication present |
| `amplitude_pct` | Gate1 Current Amplitude | % FSH, not dB |
| `soundpath_mm` | Gate1 SoundPath | Equivalent to ToF × velocity/2 |
| `depth_mm` | Gate1 Depth to Reflector | After instrument unfolding |
| `surface_distance_mm` | Gate1 Surface Distance | From beam index point |
| `saturated` | amplitude >= 100 | Indicates clipping |
| `dist_from_root_mm` | abs(soundpath - 20.0) | Root-band ambiguity feature |

## Methodological Impact

### Reduced Feature Space

With 7 features instead of the original 9+, the classifier operates on a more
constrained representation. However, the 7 available features capture the
primary discriminators:

- **Amplitude**: Primary strength indicator
- **Depth location**: Position in weld thickness
- **Surface distance**: Lateral position from beam index
- **Root-band proximity**: Ambiguity near the half-skip boundary

### Model Selection

XGBoost and Random Forest were cut from scope for this modality. With n=150
samples and effective sample size ~27 (after coupon clustering), a deep model
would overfit. Multinomial logistic regression is appropriate.

### Calibration

Isotonic regression was avoided — with 150 points across 3 classes it would
overfit. Temperature scaling was available but not required (ECE = 0.108 < 0.15).

## Data Capture Path

The waveform export path was not available. Features were extracted from the
CSV gate readings export. This is a declared limitation that should be noted
in the thesis methods section.

The capture order was:
1. CSV export of gate readings (primary path, used)
2. BMP screen captures (contingency, not required)
3. Continuous video (backstop, not required)

## Trigonometry Discrepancies

8 of 53 located readings (15.1%) showed trigonometric inconsistencies beyond
1.5mm tolerance. These are attributed to:

- Multi-mode echoes
- Gate positioning on secondary reflections
- Instrument calibration variations

The readings were retained for classification as they represent legitimate
instrument outputs, not parsing errors.
