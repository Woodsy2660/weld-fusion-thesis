# Ultrasonic Module Methodology Verification

## Run Identity

**Entry script**: `src/ut/features.py` (feature extraction), `src/ut/train.py` (training), `src/ut/calibrate.py` (calibration), `src/ut/bpa.py` (BPA construction)
**EPOCH export file**: `data/raw/TEST01.csv`
**BPA output used by fusion**: `results/ut/bpa_ut.csv`
**Metrics file**: `results/ut/metrics_ut.json`
**Git commit**: Not explicitly recorded in results files

**Label file**: `data/labels_v1_2026-10-17.csv`
**Label convention**: v1 (original)
**Mapping from true class to UT target**:
- P → {P}
- L → {L}
- N, U → {N,U}

---

## Verification Table

| Item | Status | Actual Value | file:line | Quote |
|------|--------|--------------|-----------|-------|
| 0.1 | OK | | See Run Identity above | — |
| 0.2 | OK | v1 labels, P→{P}, L→{L}, N/U→{N,U} | `src/ut/features.py:338` | `labels_path = REPO_ROOT / 'data' / 'labels_v1_2026-10-17.csv'` |
| 1.1 | OK | Evident EPOCH 650 | `results/ut/mapping_audit.md:7` | `"**Instrument**: Evident EPOCH 650"` |
| 1.2 | NOT FOUND | — | — | No probe part number, frequency, element size, or wedge recorded in export headers, logs, or notes |
| 1.3 | NOT FOUND | — | — | No couplant recorded in any file |
| 2.1 | **DIFFERS** | Mode=P/E, Angle=60, Vel=3240, Thick=10, Zero=6.431, Gain=6, RefGain=56.2, Range=50, Delay=0, G1Level=20 | `data/raw/TEST01.csv:75-190` | `"115,mm,...6,56.2,50,0,3240,6.431,60,10,Off,635,0,40,20,..."` |
| 2.2 | OK | No DAC/TCG | `data/raw/TEST01.csv` header | No DAC or TCG columns with values |
| 2.3 | OK | 39 dropped (IDs 040–078) | `results/ut/mapping_audit.md:14-16` | `"Dropped rows (calibration block): 39"` |
| 3.1 | NOT FOUND | — | — | No record of cell position marking method or spacing |
| 3.2 | OK | IDs 079–229, WC-07 first, 11th reading excluded | `src/ut/features.py:90-123` | `"cur = 79  # First production ID"` ... `"assert cur - 1 == 229"` |
| 3.3 | OK | 150 cells, 1 reading per cell | `results/ut/mapping_audit.md:60-61` | `"**Mapped cells**: 150"` ... `"**Excluded IDs**: 1 (ID 089, extra reading beyond cell 09)"` |
| 4.1 | NOT FOUND | — | — | No corner-calibration readings stored |
| 4.2 | NOT FOUND | — | — | No recovered refracted angle from back-wall spacing |
| 4.3 | NOT FOUND | — | — | Velocity and zero source not documented |
| 5.1 | OK | Uses instrument depth | `src/ut/features.py:160-161,173` | `"elif typ == 'Gate1DepthToReference': depth = val_float"` ... `"depth_mm = depth if depth is not None else 0.0"` |
| 5.2 | OK | verify_trigonometry exists, 8 failures at 1.5mm | `src/ut/features.py:193-241`, `results/ut/mapping_audit.md:80-85` | `"Failures at tolerance: 8"` |
| 6.1 | OK | 7 features in order | `src/ut/train.py:21-29` | `"FEATURE_COLS = ['located', 'amplitude_pct', 'soundpath_mm', 'depth_mm', 'surface_distance_mm', 'saturated', 'dist_from_root_mm']"` |
| 6.2 | OK | Fill 0.0 for all when not located | `src/ut/features.py:170-176` | `"amplitude_pct = amplitude if amplitude is not None else 0.0"` ... |
| 6.3 | OK | Saturation ≥100%, root ref = \|SA−20\| | `src/ut/features.py:175-176` | `"saturated = 1 if amplitude is not None and amplitude >= 100 else 0"` ... `"dist_from_root_mm = abs(soundpath_mm - HALF_SKIP_SOUNDPATH_MM)"` |
| 7.1 | OK | StandardScaler fit on train only | `src/ut/train.py:120-122` | `"scaler = StandardScaler()"` ... `"X_train_scaled = scaler.fit_transform(X_train)"` |
| 7.2 | OK | lbfgs, balanced, max_iter=1000, random_state=42 | `src/ut/train.py:126-131` | `"LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)"` |
| 7.3 | OK | 4 folds from label file, no calibration holdout | `src/ut/train.py:95,109-112` | `"folds = sorted(df['fold'].unique())"` ... `"train_mask = df['fold'] != fold"` |
| 8.1 | OK | 0.15 threshold coded | `src/ut/calibrate.py:127` | `"ECE_THRESHOLD = 0.15"` |
| 8.2 | OK | 10 bins equal-width, ECE=0.108 | `src/ut/calibrate.py:18`, `results/ut/calibration.json:2` | `"n_bins: int = 10"` ... `"ece_before": 0.10804785828401538` |
| 8.3 | OK | No calibration applied (ECE < 0.15) | `results/ut/calibration.json:4-5` | `"calibration_applied": false, "temperature": 1.0` |
| 9.1 | OK | m_focal = (1-α)·p_i, m_Θ = α | `src/ut/bpa.py:42-71` | `"bpa[focal] = (1 - alpha) * proba[i]"` ... `"bpa['Theta'] = alpha"` |
| 9.2 | OK | α=0.241, trivial=0.255 (most_frequent), F1 pooled | `results/ut/alpha_ut.json:2-4` | `"alpha": 0.24055830183192806, "f1_trivial": 0.2551440329218107"` |
| 9.3 | OK | No fixed floor | `src/ut/bpa.py:42-71` | No floor added in construct_bpa() |
| 10.1 | OK | Settings assertion before parsing | `src/ut/features.py:71-86` | `"assertions = {'Units': 'mm', 'Angle': '60', 'Thick': '10', 'Vel': '3240', 'Zero': '6.431', 'Mode': 'P/E'}"` |
| 10.2 | NOT FOUND | — | — | No r=0.977 correlation script or output exists |

---

## Text Changes Required

1. **Item 1.2**: Text says "[part number, frequency, element size]"; code has NOT FOUND. Probe details must be added from acquisition records or marked as not recorded.

2. **Item 1.3**: Text says "[couplant]"; code has NOT FOUND. Couplant type must be added from acquisition records or marked as not recorded.

3. **Item 2.1**: Text says "gain 56.2 dB, gate 1 at 20% screen height"; actual header shows Gain=6 dB, Ref Gain=56.2 dB, G1 Level=20%. The "gain 56.2 dB" in the text appears to conflate Gain and Ref Gain columns. Clarify: operational Gain=6 dB, Reference Gain=56.2 dB.

4. **Item 3.1**: Text says "each marked cell position"; no record of marking method. Either document the marking method or note as not recorded.

5. **Item 4.1–4.3**: Text says "zero offset set from the coupon end corner at the half-skip surface distance (17.3 mm → 20 mm sound path), velocity at full skip (34.6 mm → 40 mm)"; no stored record of these calibration readings exists. Either recover and store the readings or note as not recorded.

6. **Item 10.2**: Text says "per coupon, cells with an indication against the number expected from the weld log" with r=0.977; no correlation script or output exists. Either implement or remove this claim.

---

## Appendix: Parameter Table

### Instrument

| Parameter | Value | Source |
|-----------|-------|--------|
| Instrument | Evident EPOCH 650 | `results/ut/mapping_audit.md:7` |
| Probe part number | NOT FOUND | — |
| Probe frequency | NOT FOUND | — |
| Element size | NOT FOUND | — |
| Wedge | NOT FOUND | — |
| Couplant | NOT FOUND | — |

### Settings (from production block header)

| Parameter | Value | Source |
|-----------|-------|--------|
| Mode | P/E | `data/raw/TEST01.csv` header |
| Angle | 60° | `data/raw/TEST01.csv` header |
| Velocity | 3240 m/s | `data/raw/TEST01.csv` header |
| Thickness | 10 mm | `data/raw/TEST01.csv` header |
| Zero offset | 6.431 µs | `data/raw/TEST01.csv` header |
| Gain | 6 dB | `data/raw/TEST01.csv` header |
| Ref Gain | 56.2 dB | `data/raw/TEST01.csv` header |
| Range | 50 mm | `data/raw/TEST01.csv` header |
| Delay | 0 µs | `data/raw/TEST01.csv` header |
| G1 Start | 0 mm | `data/raw/TEST01.csv` header |
| G1 Width | 40 mm | `data/raw/TEST01.csv` header |
| G1 Level | 20% | `data/raw/TEST01.csv` header |
| Units | mm | `data/raw/TEST01.csv` header |
| DAC/TCG | None | `data/raw/TEST01.csv` header |

### Features (ordered)

| # | Feature | Source Column | Null Value |
|---|---------|---------------|------------|
| 1 | located | presence of Gate1 SoundPath | 0 |
| 2 | amplitude_pct | Gate1 Current Amplitude | 0.0 |
| 3 | soundpath_mm | Gate1 SoundPath | 0.0 |
| 4 | depth_mm | Gate1 Depth to Reflector | 0.0 |
| 5 | surface_distance_mm | Gate1 Surface Distance | 0.0 |
| 6 | saturated | amplitude ≥ 100 | 0 |
| 7 | dist_from_root_mm | \|soundpath - 20.0\| | 0.0 |

### Scaler

| Parameter | Value | Source |
|-----------|-------|--------|
| Type | StandardScaler | `src/ut/train.py:120` |
| Fit on | Training fold only | `src/ut/train.py:121` |

### Classifier

| Parameter | Value | Source |
|-----------|-------|--------|
| Type | LogisticRegression | `src/ut/train.py:126` |
| Solver | lbfgs | `src/ut/train.py:127` |
| max_iter | 1000 | `src/ut/train.py:128` |
| class_weight | balanced | `src/ut/train.py:129` |
| random_state | 42 | `src/ut/train.py:130` |
| C | 1.0 (default) | sklearn default, not overridden |
| multi_class | auto (multinomial for lbfgs) | sklearn default, not overridden |

### Cross-Validation

| Parameter | Value | Source |
|-----------|-------|--------|
| Folds | 4 | `data/labels_v1_2026-10-17.csv` fold column |
| Grouping | By coupon | labels fold column |
| Calibration holdout | None | `src/ut/train.py:109-117` |

### Calibration

| Parameter | Value | Source |
|-----------|-------|--------|
| ECE threshold | 0.15 | `src/ut/calibrate.py:127` |
| ECE actual | 0.108 | `results/ut/calibration.json:2` |
| Calibration applied | No | `results/ut/calibration.json:4` |
| Temperature | 1.0 | `results/ut/calibration.json:5` |
| Binning | 10 bins, equal-width | `src/ut/calibrate.py:18,32` |

### Alpha and Baseline

| Parameter | Value | Source |
|-----------|-------|--------|
| F1 macro (pooled) | 0.821 | `results/ut/metrics_ut.json:19` |
| F1 macro (mean-of-folds) | 0.820 ± 0.036 | `results/ut/metrics_ut.json:20-21` |
| Trivial baseline | 0.255 (most_frequent) | `results/ut/metrics_ut.json:45` |
| α_U | 0.241 | `results/ut/alpha_ut.json:2` |
| Formula | α = 1 − (F1 − T)/(1 − T) | `src/ut/bpa.py:25-39` |

---

## Provenance

- **Verification date**: 2026-10-29
- **EPOCH export**: `data/raw/TEST01.csv`
- **Feature extraction**: `src/ut/features.py`
- **Training**: `src/ut/train.py`
- **Calibration**: `src/ut/calibrate.py`
- **BPA construction**: `src/ut/bpa.py`
- **Results**: `results/ut/metrics_ut.json`, `results/ut/alpha_ut.json`, `results/ut/bpa_ut.csv`
