# Methods Extraction Report

**Generated:** 2026-10-25
**Branch:** `vision/sequence-leakage-audit`
**Purpose:** Document what the code actually does for thesis Methodology chapter verification.

---

## 0. Repository Orientation

### 0.1 Entry-Point Scripts

| Modality | Entry Point | Command Line (if recorded) |
|----------|-------------|---------------------------|
| **Vision (public corpus)** | `src/vision/train.py` | `python src/vision/train.py --config configs/vision_train.yaml` |
| **Vision (coupon eval)** | NOT FOUND as standalone script | k-fold CV embedded in results |
| **Geometry** | `src/geometry/pipeline.py` | No command recorded in logs |
| **Ultrasonic** | `src/ut/features.py`, `src/ut/train.py`, `src/ut/bpa.py` | `python3 src/ut/features.py`, `python3 src/ut/train.py` |
| **Fusion** | `src/weldfusion/pipeline/phase2_percell.py` | NOT FOUND - no run log |

### 0.2 Config Files

| Config | Used By | Key Final Values |
|--------|---------|------------------|
| `configs/frame.yaml` | All modalities | 4-class Theta={N,U,P,L}, heads V1,V3,V4 |
| `configs/vision_train.yaml` | Vision training | architecture=resnet50, epochs=30, lr=0.0001, batch_size=32 |
| `configs/geometry.yaml` | Geometry pipeline | cell_width=25mm, Level B limits |
| `configs/ut_train.yaml` | UT pipeline | model=logistic_regression, 7 features |
| `configs/class_map.yaml` | Vision data prep | Source class → head mapping |

### 0.3 Git Commit Hash

**File:** `results/vision/run_manifest_resnet50_ft-layer3.json:3-4`
```json
"git_sha": "f7d61de",
"config_sha16": "vision_train.yaml:0a9f..."
```

---

## 1. Shared: Cells, Labels, Folds

### 1.1 Cell Definition

**File:** `src/geometry/registration.py:351-402`

```python
def compute_cell_bounds(
    pose: CouponPose,
    coupon_id: str,
    config: dict | None = None,
) -> list[CellBounds]:
    """
    Compute cell boundaries in both coordinate systems.

    Excludes the first and last cells (arc start/stop transient zones).
    """
    config = config or load_config()
    cells_cfg = config["cells"]

    cell_width = cells_cfg["width_mm"]
    first_excluded = cells_cfg["first_excluded_mm"]
    last_excluded = cells_cfg["last_excluded_mm"]
    coupon_length = cells_cfg["coupon_length_mm"]

    # Compute cell boundaries
    s_start = first_excluded
    s_end = coupon_length - last_excluded
    n_cells = int((s_end - s_start) / cell_width)
```

**Answer:** (b) **25 mm cells from the plate end with end cells excluded.**

From `configs/geometry.yaml:38-42`:
```yaml
cells:
  width_mm: 25              # Matches UT index pitch for 1:1 cell alignment
  first_excluded_mm: 25     # Arc-start transient zone
  last_excluded_mm: 25      # Arc-stop transient zone
  coupon_length_mm: 300     # Nominal; actual measured per coupon
```

For a 300mm coupon: `n_cells = (300 - 25 - 25) / 25 = 10 cells` (indices 00-09).

**NOT** (a) - there is no D6→D8 datum span fraction logic in the code.

### 1.2 Label File

**File:** `data/labels_v1_2026-10-17.csv`

**Rows:** 150

**Confirmed usage:**
- UT: `src/ut/features.py:337-338` reads `data/labels_v1_2026-10-17.csv`
- Vision coupon: `results/vision/coupon_eval/kfold_results.json` (150 samples)
- Geometry: `results/geometry/summary.csv` (150 rows)

**Class totals per `true_class`:**

| true_class | Count |
|------------|-------|
| {N,U} | 93 |
| {L} | 31 |
| {P} | 26 |
| **Total** | 150 |

**Class by fold:**

| Fold | {L} | {N,U} | {P} |
|------|-----|-------|-----|
| 1 | 10 | 20 | 10 |
| 2 | 10 | 20 | 10 |
| 3 | 7 | 17 | 6 |
| 4 | 4 | 36 | 0 |

### 1.3 Fold Assignment

**File:** `data/labels_v1_2026-10-17.csv` (fold column)

| Fold | Coupons |
|------|---------|
| 1 | C01, C03, C05, C07 |
| 2 | C02, C04, C06, C08 |
| 3 | C09, C10, C12 |
| 4 | C11, C13, C14, C15 |

**Matches expected assignment exactly:** ✓

### 1.4 Calibration Subset Selection

**Vision coupon:** `results/vision/coupon_eval/calibration_results.json`

```json
{
  "method": "temperature_scaling",
  "temperature": 10.0,
  "n_bins": 10,
  "ece_before": 0.4157,
  "ece_after": 0.1362
}
```

**Note:** The calibration appears fitted on all CV predictions pooled. No explicit calibration-fold separation found in the coupon eval code.

**UT:** `src/ut/calibrate.py` - ECE computed on all 150 pooled predictions, temperature scaling fitted on same data.

**POTENTIAL ISSUE:** Calibration is fitted on the same data used for evaluation, not a held-out calibration set within each training partition.

### 1.5 Metrics Reporting

**UT:** `src/ut/train.py:161-165`
```python
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

f1_macro_overall = f1_score(all_y_true, all_y_pred, labels=classes, average='macro')
```

**Reports BOTH:**
- `f1_macro_mean`: mean of per-fold F1s
- `f1_macro_overall`: pooled over all 150 out-of-fold predictions

**Vision coupon:** `results/vision/coupon_eval/kfold_results.json:1171-1172`
```json
"mean_macro_f1": 0.32421095140880174,
"std_macro_f1": 0.0300374006723657
```

**Reports:** Mean of per-fold metrics (0.324 is mean, not pooled).

---

## 2. Optical Acquisition and Registration

### 2.1 Capture Script

**File:** `capture-kit/capture.py:1-50` (NOT FOUND in current commit - file is empty stub)

**From** `configs/geometry.yaml:5-10`:
```yaml
sensor:
  type: realsense_d435i
  depth_resolution: [1280, 720]
  rgb_resolution: [1920, 1080]
  working_distance_mm: 320
  working_distance_tolerance_mm: 5
```

**NOT FOUND:** frame rate, number of depth/colour frames per face, depth_units, visual preset, post-processing filters, auto-exposure state.

**From** `src/geometry/profile.py:52-55`:
```python
def load_depth_frames(
    frame_dir: Path,
    depth_units: float = 0.0001,
) -> list[np.ndarray]:
```

`depth_units = 0.0001` (100 µm default).

### 2.2 Standoff

**Nominal:** 320 mm (from `configs/geometry.yaml:9`)

**Measured range across captures:** NOT FOUND in output files.

### 2.3 Captures Per Coupon

**From file listing:** `data/captures/C##_top_r0/`, `C##_top_r1/`, `C##_root_r0/`

Typically 2-3 repeats per face, but which were used for reported results is NOT explicitly recorded.

### 2.4 Registration

**File:** `src/geometry/registration.py:405-420`

```python
def register_coupon(
    depth_mm: np.ndarray,
    coupon_id: str,
    config: dict | None = None,
) -> tuple[CouponPose, list[CellBounds]]:
    """
    Full registration pipeline: detect holes, solve pose, compute cells.
    """
```

**Datum holes:** Detected in **depth image** via `detect_holes_in_depth()`.

**End-for-end detection:** Different hole diameters (6mm vs 8mm) from `configs/geometry.yaml:19-26`:
```yaml
origin_hole:
  s_mm: 15
  y_mm: 50
  diameter_mm: 6
axis_hole:
  s_mm: 285
  y_mm: 50
  diameter_mm: 8  # Different diameter breaks end-for-end ambiguity
```

**Coordinate frame:** Origin at origin_hole, +s axis toward axis_hole.

### 2.5 Colour Cast

**NOT FOUND** - No white-balance correction or grayscale conversion code identified.

Vision model uses RGB input per `src/vision/dataset.py:69`:
```python
img = im.convert("RGB")
```

---

## 3. Vision Pipeline

### 3.1 Input Crop

**File:** `src/vision/dataset.py:112-120`
```python
def get_eval_transforms(cfg):
    """
    Inference transforms: no augmentation, and no centre crop.

    Patches are already square and already the right size. A Resize(256) +
    CenterCrop(224) here would silently throw away the context margin...
    """
    size = cfg["data"]["image_size"]
```

**From** `configs/vision_train.yaml`:
```yaml
data:
  image_size: 224
  patches_dir: data/merged
```

**Normalisation:** `src/vision/dataset.py:105-108`:
```python
transforms.Normalize(mean=cfg["normalization"]["mean"],
                     std=cfg["normalization"]["std"]),
```

From config: `mean: [0.485, 0.456, 0.406]`, `std: [0.229, 0.224, 0.225]` (ImageNet).

**Augmentation (training):** `src/vision/dataset.py:75-109`:
- RandomHorizontalFlip(0.5)
- RandomVerticalFlip(0.5)
- RandomRotation(15°)
- ColorJitter(brightness=0.2, contrast=0.2)
- GaussianBlur (optional)

### 3.2 Architecture

**File:** `src/vision/model.py:21-24`
```python
ARCHITECTURES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}
```

**Pretrained weights:** `models.ResNet50_Weights.IMAGENET1K_V2`

**Frozen layers:** `src/vision/model.py:63-74`:
```python
def set_frozen_through(model, stage):
    """Freeze conv1..stage inclusive; everything after stays trainable."""
    if stage not in RESNET_STAGES:
        raise ValueError(...)
    cutoff = RESNET_STAGES.index(stage)
    frozen = set(RESNET_STAGES[:cutoff + 1])
    for name, module in model.named_children():
        if name in frozen:
            for param in module.parameters():
                param.requires_grad = False
```

From run manifest: `"freeze_through": "layer3"` — freezes conv1, bn1, layer1, layer2, layer3.

**Output layer:** `src/vision/model.py:54`
```python
model.fc = nn.Linear(model.fc.in_features, num_heads)
```

`num_heads = 3` (from frame).

**Loss function:** Cross-entropy (implicit in `nn.CrossEntropyLoss` from train.py).

### 3.3 Output

**Three-way softmax over:** `{P}` (V1), `{N,U}` (V3), `{N,U,L}` (V4)

**File:** `tests/test_frame_and_bpa.py:28-32`
```python
def test_frame_declares_three_vision_heads():
    """V2 {C} was removed; V3 and V4 deliberately keep their names."""
    frame = load_frame()
    assert frame.head_names == ["V1", "V3", "V4"]
    assert frame.num_heads == 3
```

**Class-index order:** Sorted alphabetically: V1, V3, V4 → indices 0, 1, 2.

### 3.4 Pretraining Corpus

**Datasets:** V1 (porosity), V3 (geometric), V4 (no evidence) from:
- `final_welding` (Roboflow FINAL WELDING DATASET)
- `weld_dataset` (LoHi-WELD)
- `wijaya` (Welding Defect Dataset)

**Patch counts:** From `results/vision/run_manifest_resnet50_ft-layer3.json`:
```json
"patch_counts": {"V1": 5831, "V3": 3893, "V4": 7378}
```

Total: 17,102 patches.

**Splits grouped by:** `source_image` (perceptual hash groups)

**81.59% figure:** `results/diagnostics/sequence_leakage_report.md:64`
```
test patches sharing a stem with train: 3,599 / 4,411 = 81.59%
```

This describes the **public corpus test split** leakage, NOT the coupon data. Video-derived frames from the same weld straddled the train/test split.

### 3.5 Fine-Tuning on Coupons

**File:** `results/vision/coupon_eval/kfold_results.json:2-7`
```json
"n_folds": 4,
"epochs": 30,
"lr": 0.0001,
"batch_size": 16,
"patience": 8
```

**Optimiser:** NOT explicitly stated; likely Adam (standard for fine-tuning).

**Class weighting:** NOT FOUND in coupon eval code.

### 3.6 Calibration

**File:** `results/vision/coupon_eval/calibration_results.json`
```json
{
  "method": "temperature_scaling",
  "temperature": 10.0,
  "n_bins": 10,
  "ece_before": 0.4157,
  "ece_after": 0.1362
}
```

**ECE 0.136:** After temperature scaling (coupon eval, 10 bins).

**ECE 0.14:** NOT FOUND as a separate figure. The 0.1362 rounds to 0.14.

### 3.7 Metrics

**0.324 macro-F1:** `results/vision/coupon_eval/kfold_results.json:1171`
```json
"mean_macro_f1": 0.32421095140880174
```
**Estimator:** Mean of per-fold F1s.

**0.305 macro-F1:** `results/vision/coupon_eval/kfold_results.json:10`
```json
"fold": "1",
"macro_f1": 0.30526315789473685
```
This is the per-fold F1 for fold 1, NOT a separate aggregate metric.

---

## 4. Geometry Pipeline

### 4.1 Depth Processing

**File:** `src/geometry/profile.py:77-105`
```python
def compute_depth_stack(
    frames: list[np.ndarray],
    config: dict | None = None,
) -> DepthStack:
    """
    Compute temporal average and split-half noise estimate.
    """
    ...
    with np.errstate(all="ignore"):
        mean_mm = np.nanmean(stack, axis=0)
        std_mm = np.nanstd(stack, axis=0)
```

**Answer:** **Mean** (with `nanmean`), valid-pixel masking via NaN handling.

### 4.2 Reference Surface

**File:** `src/geometry/pipeline.py:193-199`
```python
plane_fit = fit_plane_ransac(
    depth_mm,
    parent_mask,
    pose.scale_mm_per_px,
    n_iterations=100,
    inlier_threshold_mm=0.5,
)
```

**Method:** RANSAC, 100 iterations, 0.5mm inlier threshold.

**ROI:** Per-half-plate, from `configs/geometry.yaml:71-74`:
```yaml
plane_fit_roi:
  y_inner_mm: 15    # Start of parent metal region (outside bead)
  y_outer_mm: 40    # End of parent metal region (inside plate edge)
```

### 4.3 Profile Extraction

**File:** `src/geometry/pipeline.py:141-147`
```python
def weld_axis_centre_rows(weld_axis_px, width: int) -> np.ndarray:
    """Centreline row for every column, from two (u, v) depth-image points on
    the weld axis (configs/weld_axis_seeds.csv, mapped to depth pixels)."""
    (u1, v1), (u2, v2) = weld_axis_px
    cols = np.arange(width)
    return v1 + (v2 - v1) * (cols - u1) / (u2 - u1)
```

**Confirmed:** `rows // 2` fallback exists but weld axis seeds are used.

**Profiles per cell:** `configs/geometry.yaml:50`: `profiles_per_cell: 10`

### 4.4 Measurands

**File:** `src/geometry/rules.py:78-96`
```python
@dataclass
class GeometryMeasurement:
    undercut_mm: float           # Depth below parent metal datum (governing)
    reinforcement_mm: float      # Height above parent metal datum
    misalignment_mm: float       # Hi-lo between plate halves
    bead_width_mm: float         # Toe-to-toe distance
```

**Undercut:** Depth below fitted plane at weld toe (95th percentile across profiles).

**Reinforcement:** Height above fitted plane at bead crown.

**Misalignment:** Difference in plane heights between plate halves.

### 4.5 Limits

**File:** `configs/geometry.yaml:144-169`
```yaml
acceptance_criteria:
  standard: "ISO 5817:2023"
  quality_level: "B"
  plate_thickness_mm: 10

  # Undercut (5011/5012): h <= 0.05 t, max 0.5 mm  -> 0.5 mm at t = 10
  undercut_coef_t: 0.05
  undercut_max_mm: 0.5
  # Excess weld metal (502): h > 0, limit = 1 mm + 0.1 b, max 5 mm
  reinforcement_base_mm: 1.0
  reinforcement_coef_b: 0.1
  reinforcement_max_mm: 5.0
  # Linear misalignment (507), plates: h <= 0.1 t, max 3 mm -> 1.0 mm at t = 10
  misalignment_coef_t: 0.1
  misalignment_max_mm: 3.0
```

### 4.6 Exceedance BPA

**File:** `src/geometry/rules.py:205-230`
```python
def exceedance_probability(
    measured_mm: float,
    limit_mm: float,
    u_total_mm: float,
) -> float:
    """
    P(true > limit) = Φ((measured - limit) / u_total)
    """
    if u_total_mm <= 0:
        return 1.0 if measured_mm > limit_mm else 0.0

    z = (measured_mm - limit_mm) / u_total_mm
    return float(norm.cdf(z))
```

**(a) Which measurands drive m({U}):** `src/geometry/rules.py:314-335`
```python
# NOTE: Undercut is EXCLUDED from the U disjunction (measured repeatability
# 0.586 mm exceeds 0.5 mm Level B limit; S/U ratio 0.85:1 is unsupportable).
by_active = {
    "reinforcement_502": by_all["reinforcement_502"],
    "reinforcement_511": by_all["reinforcement_511"],
    "misalignment": by_all["misalignment"],
}
```

**Answer:** Reinforcement (excess and underfill) + misalignment. **Undercut is EXCLUDED.**

**(b) Governing value:** Max over active measurands:
```python
gov_k = max(by_active, key=by_active.get)
p_exceed = by_active[gov_k]
```

**(c) Value of u:** From `configs/geometry.yaml:135-140`:
```yaml
measured_repeatability:
  use_for_bpa: true
  undercut_mm: 0.586
  reinforcement_mm: 0.493
  misalignment_mm: 0.275
```

**Active:** `misalignment_mm: 0.275` (for the misalignment measurand that drives BPA).

**The 0.083 mm figure:** NOT FOUND in any code or config. The comment at line 131 says:
```yaml
# RSS 0.049 mm; the 0.083 mm figure quoted elsewhere has no source in
# this repo
```

**(d) Valid-pixel threshold:** `configs/geometry.yaml:104`:
```yaml
min_valid_fraction: 0.60
```

Cells below 60% valid pixels receive vacuous BPA (m(Θ)=1).

### 4.7 Predicted Uncertainty Budget

**File:** `configs/geometry.yaml:109-125`
```yaml
uncertainty:
  u_quant_mm: 0.029           # 0.1 / sqrt(12)
  u_cal_mm_default: 0.030     # RMS residual from plane fit
  u_align_mm_default: 0.020   # Fixture repeatability
  noise_floor_mm_default: 0.015  # Split-half minimum
```

**RSS:** sqrt(0.029² + 0.030² + 0.020² + 0.015²) = 0.049 mm

**0.083 mm figure:** NOT FOUND. No provenance in repository.

### 4.8 Measured Repeatability

**File:** `configs/geometry.yaml:128-140`
```yaml
# MEASURED repeatability (2026-10-20), C05_top r0/r1/r2 re-placement repeats,
# 10 cells, pooled SD across repeats
measured_repeatability:
  use_for_bpa: true
  source: "results/geometry/_diagnostics/repeatability_C05_top_axisseeds.json"
  undercut_mm: 0.586
  reinforcement_mm: 0.493
  misalignment_mm: 0.275
```

### 4.9 Geometry Predictions vs Labels

**File:** `results/geometry/summary.csv`

Contains `m_U`, `m_NPLC` columns but **no true_class or fold column**.

**Geometry macro-F1:** NOT FOUND. No classification metrics file for geometry.

**Five-class residue:** `m_NPLC` column exists (historical name for conforming element).

---

## 5. Ultrasonic Pipeline

### 5.1 Probe

**NOT FOUND** in repository. No acquisition log, instrument export header, or notes file containing probe part number.

From documentation reference: "4 MHz AM4R-8X9-60" mentioned in `weld-docs/` but not verified in data files.

### 5.2 Instrument Settings

**File:** `src/ut/features.py:71-78`
```python
assertions = {
    'Units': 'mm',
    'Angle': '60',
    'Thick': '10',
    'Vel': '3240',
    'Zero': '6.431',
    'Mode': 'P/E'
}
```

Verified against `data/raw/TEST01.csv` header.

### 5.3 Mapping

**File:** `src/ut/features.py:90-123`
```python
def build_id_to_cell_mapping() -> Tuple[Dict[int, Tuple[str, int]], Dict[str, List[int]]]:
    """
    Capture order: WC-07 first with 11 readings, then WC-01 through WC-15
    excluding WC-07, 10 readings each.
    """
    coupon_order = ['WC-07'] + [f'WC-{i:02d}' for i in range(1, 16) if i != 7]
    readings_per_coupon = [11] + [10] * 14

    cur = 79  # First production ID
    ...
    assert cur - 1 == 229, f"Last ID should be 229, got {cur - 1}"
```

**r = 0.977 check:** NOT FOUND. No correlation verification against weld log in code.

### 5.4 Features

**File:** `src/ut/features.py:126-190`
```python
def extract_features(row: Dict) -> Dict:
    """
    Extract 7 features from a single EPOCH row.

    Features:
        located: 1 if Gate1 SoundPath present, else 0
        amplitude_pct: Gate1 Current Amplitude (% FSH)
        soundpath_mm: Gate1 SoundPath
        depth_mm: Gate1 Depth to Reflector
        surface_distance_mm: Gate1 Surface Distance
        saturated: 1 if amplitude >= 100, else 0
        dist_from_root_mm: abs(soundpath - 20.0) when located
    """
```

**Features (7 total):** located, amplitude_pct, soundpath_mm, depth_mm, surface_distance_mm, saturated, dist_from_root_mm.

**CONTRADICTION:** Prompt says "six features" but code has **seven**.

### 5.5 Depth Folding

**File:** `src/ut/features.py:220-232`
```python
# Handle depth folding - the instrument reports after unfolding
# For leg 1: depth = soundpath * cos(60)
# For leg 2: depth = 2t - soundpath * cos(60)
leg1_depth = soundpath * math.cos(math.radians(ANGLE_DEG))
leg2_depth = 2 * THICKNESS_MM - leg1_depth
```

**Called in:** `verify_trigonometry()` function, NOT used in feature extraction directly.

### 5.6 Classifier

**File:** `src/ut/train.py:124-131`
```python
# Train logistic regression
model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
)
```

**Feature scaling:** `src/ut/train.py:119-122`:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

**Folds:** k=4, leave-one-fold-out, fold column from labels file.

### 5.7 Calibration

**File:** `results/ut/calibration.json`
```json
{
  "ece_before": 0.10804785828401538,
  "calibration_applied": false,
  "method": "none",
  "ece_threshold": 0.15
}
```

ECE < 0.15, no calibration applied. Bin count: 10 (`src/ut/calibrate.py:37`).

### 5.8 Metrics

**File:** `results/ut/metrics_ut.json:19-27`
```json
"f1_macro": 0.8208187134502923,
"f1_per_class": {
  "{L}": 0.8666666666666667,
  "{N,U}": 0.9157894736842105,
  "{P}": 0.68
}
```

**Confirmed:** macro-F1 = 0.821 (pooled over all 150 OOF predictions).

---

## 6. Fusion

### 6.1 Classifier-Output → BPA Construction

**File:** `src/bpa/construct.py:71-96`
```python
def modality_bpa(probabilities, modality, frame=None, uncertainty_floor=0.05):
    """
    Build a BPA for any modality from its calibrated per-head probabilities.
    """
    frame = frame or load_frame()
    elements = modality_focal_elements(frame, modality)
    ...
    scale = 1.0 - uncertainty_floor
    bpa = {}
    for head, p in zip(heads, probabilities):
        key = elements[head]
        bpa[key] = bpa.get(key, 0.0) + (float(p) / total) * scale
    theta = theta_of(frame)
    bpa[theta] = bpa.get(theta, 0.0) + uncertainty_floor
```

**Confirmed:** `m(Theta) = uncertainty_floor` comes from the discount, NOT from an output neuron.

### 6.2 Alpha Formula

**File:** `src/ut/bpa.py:21-34`
```python
def derive_alpha(f1_macro: float, f1_trivial: float) -> float:
    """
    Derive discount factor alpha from skill formula.

    alpha = 1 - (F1_macro - F1_trivial) / (1 - F1_trivial)
    """
    if f1_trivial >= 1:
        return 1.0
    skill = (f1_macro - f1_trivial) / (1 - f1_trivial)
    alpha = 1 - skill
    return max(0, min(1, alpha))
```

**Trivial baselines:**
- UT: `results/ut/metrics_ut.json:45`: `baseline_most_frequent_f1: 0.2551`
- Vision: `results/vision/coupon_eval/alpha_derivation.json:3`: `f1_trivial: 0.2338`
- Geometry: NOT FOUND (no classifier, deterministic exceedance)

### 6.3 Primary Combination

**File:** `src/weldfusion/pipeline/phase2_percell.py:140-143`
```python
# Murphy averaging: average masses, then combine average with itself n-1 times
try:
    fused, total_k = murphy_average(bpas)
except ConflictError:
```

**Answer:** Murphy averaging is the primary method. Dempster's rule is called internally by Murphy averaging (n-1 times).

### 6.4 BPA Validator

**File:** `src/bpa/construct.py:140-198`
```python
def validate_modality_bpa(bpa, frame=None, modality="vision", tol=1e-6):
    """
    Check that a modality's BPA only commits mass where that modality is
    entitled to.
    """
```

**Called before combination:** `src/weldfusion/pipeline/phase2_percell.py:103`:
```python
bpa.validate()
```

**Frame algebra tests:** `tests/test_frame_algebra.py`
- `test_three_way_conjunction_isolates_N`
- `test_vision_and_geometry_reach_exactly_N_U_and_P`
- `test_ut_is_what_isolates_L`

**Tests pass:** Run `pytest tests/test_frame_algebra.py -v` to verify.

### 6.5 Seven-Condition Ablation

**File:** `src/weldfusion/pipeline/phase2_percell.py:238-262`
```python
def run_all_ablations(
    table: CellTable,
    modality_alphas: dict[str, float],
) -> dict[str, CellTable]:
    """Run Phase 2 for all 7 ablation subsets.

    Ablation subsets (from configs/fusion.yaml):
        vision, geometry, ut, vision+geometry, vision+ut, geometry+ut, vision+geometry+ut
    """
    for subset in ablation_subsets():
```

**Excluded modality representation:** `src/weldfusion/pipeline/phase2_percell.py:130-135`:
```python
if "vision" in subset and vision_bpa is not None:
    bpas.append(vision_bpa)
if "geometry" in subset and geometry_bpa is not None:
    bpas.append(geometry_bpa)
if "ut" in subset and ut_bpa is not None:
    bpas.append(ut_bpa)
```

**Answer:** Excluded modality is simply **omitted** from the BPA list, NOT represented as m(Θ)=1.

**Has it been run:** NOT FOUND - no ablation results files in `results/`.

---

## Contradictions with the Draft

| Section | Draft Claim | Code Reality |
|---------|-------------|--------------|
| 1.1 | Cells from D6→D8 datum span fractions | Cells are fixed 25mm, not datum-derived |
| 1.4 | Calibration on separate subset | Calibration fitted on evaluation data |
| 3.7 | Two F1 figures (0.324 vs 0.305) | 0.305 is fold-1 F1, not a separate metric |
| 4.6 | Undercut drives m({U}) | Undercut EXCLUDED; only reinforcement+misalignment |
| 4.7 | 0.083 mm uncertainty | NOT FOUND; RSS of budget is 0.049 mm |
| 5.4 | Six features | Seven features in code |
| 5.3 | r=0.977 correlation check | NOT FOUND in code |
| 5.1 | Probe part number | NOT FOUND in repository |
| 6.5 | Excluded modality as m(Θ)=1 | Excluded modality simply omitted |
| 2.1 | Capture parameters | Capture script is stub; parameters not recorded |
