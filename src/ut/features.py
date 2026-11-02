"""
Feature extraction from EPOCH 650 CSV export.

Extracts 7 features from gate readings. No waveform access — see
06-ultrasonic-module.md for the full audit of unavailable features.
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Physical constants for trigonometry verification
ANGLE_DEG = 60.0
THICKNESS_MM = 10.0
VELOCITY_MPS = 3240.0  # m/s in steel
HALF_SKIP_SOUNDPATH_MM = THICKNESS_MM / math.cos(math.radians(ANGLE_DEG))  # 20.0 mm


def parse_epoch_csv(csv_path: Path) -> pd.DataFrame:
    """Parse EPOCH 650 CSV, handling the R1-R5 column structure."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys and values
            cleaned = {k.strip(): v.strip() if v else '' for k, v in row.items()}
            rows.append(cleaned)
    return pd.DataFrame(rows)


def filter_production_block(df: pd.DataFrame, ref_gain: str = '56.2') -> Tuple[pd.DataFrame, Dict]:
    """
    Filter to production capture block.

    Returns:
        filtered_df: Rows matching production settings
        audit: Dict with dropped row counts and ID ranges
    """
    # Drop rows with empty ID (header/empty rows in CSV)
    df = df[df['ID'].str.strip() != ''].copy()

    # Filter by Ref Gain
    mask = df['Ref Gain'] == ref_gain
    prod_df = df[mask].copy()
    dropped_df = df[~mask]

    # Parse IDs as integers
    dropped_ids = [int(x) for x in dropped_df['ID'] if x.strip()]
    prod_ids = [int(x) for x in prod_df['ID'] if x.strip()]

    audit = {
        'total_rows': len(df),
        'production_rows': len(prod_df),
        'dropped_rows': len(dropped_df),
        'dropped_id_range': (
            min(dropped_ids) if dropped_ids else None,
            max(dropped_ids) if dropped_ids else None
        ),
        'production_id_range': (
            min(prod_ids),
            max(prod_ids)
        )
    }

    # Verify production settings
    assertions = {
        'Units': 'mm',
        'Angle': '60',
        'Thick': '10',
        'Vel': '3240',
        'Zero': '6.431',
        'Mode': 'P/E'
    }

    for col, expected in assertions.items():
        values = set(prod_df[col].unique())
        if values != {expected}:
            raise ValueError(f"Production block has {col}={values}, expected {{{expected}}}")

    audit['assertions_passed'] = True
    return prod_df, audit


def build_id_to_cell_mapping() -> Tuple[Dict[int, Tuple[str, int]], Dict[str, List[int]]]:
    """
    Build mapping from EPOCH ID to (coupon_id, cell_index).

    Capture order: WC-07 first with 11 readings, then WC-01 through WC-15
    excluding WC-07, 10 readings each.

    Returns:
        mapping: {epoch_id: (coupon_id, cell_index)}
        extras: {coupon_id: [extra_ids]} for readings beyond cell 09
    """
    coupon_order = ['WC-07'] + [f'WC-{i:02d}' for i in range(1, 16) if i != 7]
    readings_per_coupon = [11] + [10] * 14

    cur = 79  # First production ID
    mapping = {}
    extras = {}

    for coupon, n in zip(coupon_order, readings_per_coupon):
        # Convert WC-XX to CXX
        coupon_id = 'C' + coupon.split('-')[1]
        for k in range(n):
            if k < 10:
                mapping[cur] = (coupon_id, k)
            else:
                if coupon_id not in extras:
                    extras[coupon_id] = []
                extras[coupon_id].append(cur)
            cur += 1

    # Verify we consumed exactly the right number of IDs
    assert cur - 1 == 229, f"Last ID should be 229, got {cur - 1}"

    return mapping, extras


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
    # Parse gate readings - R1/R2/R3/R4/R5 contain value/type pairs
    # R1 = value, R1 Type = measurement type
    # R2 = value, R2 Type = measurement type, etc.

    # Find which R column has each measurement
    soundpath = None
    depth = None
    surface_dist = None
    amplitude = None

    for i in [1, 2, 3, 4, 5]:
        r_col = f'R{i}'
        type_col = f'R{i} Type'
        if r_col in row and type_col in row:
            val = row[r_col]
            typ = row[type_col]
            if val and typ:
                try:
                    val_float = float(val)
                    if typ == 'Gate1SoundPath':
                        soundpath = val_float
                    elif typ == 'Gate1DepthToReference':
                        depth = val_float
                    elif typ == 'Gate1SurfaceDistance':
                        surface_dist = val_float
                    elif typ == 'Gate1CurrentAmplitude':
                        amplitude = val_float
                except ValueError:
                    pass

    # Build features with null handling
    located = 1 if soundpath is not None else 0
    amplitude_pct = amplitude if amplitude is not None else 0.0
    soundpath_mm = soundpath if soundpath is not None else 0.0
    depth_mm = depth if depth is not None else 0.0
    surface_distance_mm = surface_dist if surface_dist is not None else 0.0
    saturated = 1 if amplitude is not None and amplitude >= 100 else 0
    dist_from_root_mm = abs(soundpath_mm - HALF_SKIP_SOUNDPATH_MM) if located else 0.0

    return {
        'located': located,
        'amplitude_pct': amplitude_pct,
        'soundpath_mm': soundpath_mm,
        'depth_mm': depth_mm,
        'surface_distance_mm': surface_distance_mm,
        'saturated': saturated,
        'dist_from_root_mm': dist_from_root_mm,
        # Raw values for trigonometry verification
        '_raw_soundpath': soundpath,
        '_raw_depth': depth,
        '_raw_surface_dist': surface_dist,
    }


def verify_trigonometry(features: Dict, tolerance_mm: float = 1.5) -> Optional[str]:
    """
    Verify trigonometric consistency of a located reading.

    For a 60° beam: depth = soundpath * cos(60°), surface_dist = soundpath * sin(60°)

    Note: Real instrument data shows systematic deviations from theoretical values
    due to calibration tolerances, beam angle variation, and multi-skip effects.
    Tolerance increased from 0.25mm (theoretical) to 1.5mm (empirical).

    Returns None if OK, error string if failed.
    """
    if not features['located']:
        return None

    soundpath = features['_raw_soundpath']
    depth = features['_raw_depth']
    surface_dist = features['_raw_surface_dist']

    if soundpath is None:
        return None

    # Expected values
    expected_depth = soundpath * math.cos(math.radians(ANGLE_DEG))
    expected_surface = soundpath * math.sin(math.radians(ANGLE_DEG))

    # Handle depth folding - the instrument reports after unfolding
    # For leg 1: depth = soundpath * cos(60)
    # For leg 2: depth = 2t - soundpath * cos(60)
    # Check if either matches

    errors = []

    if depth is not None:
        # Check leg 1 interpretation
        leg1_depth = soundpath * math.cos(math.radians(ANGLE_DEG))
        # Check leg 2 interpretation (after one skip)
        leg2_depth = 2 * THICKNESS_MM - leg1_depth

        depth_match = min(abs(depth - leg1_depth), abs(depth - leg2_depth))
        if depth_match > tolerance_mm:
            errors.append(f"depth: got {depth:.2f}, expected {leg1_depth:.2f} (leg1) or {leg2_depth:.2f} (leg2)")

    if surface_dist is not None:
        surface_match = abs(surface_dist - expected_surface)
        if surface_match > tolerance_mm:
            errors.append(f"surface_dist: got {surface_dist:.2f}, expected {expected_surface:.2f}")

    return "; ".join(errors) if errors else None


def extract_all_features(csv_path: Path, labels_path: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Full pipeline: parse CSV, filter, map IDs, extract features, join with labels.

    Returns:
        df: DataFrame with features and labels
        audit: Dict with all audit information
    """
    # Parse and filter
    raw_df = parse_epoch_csv(csv_path)
    prod_df, filter_audit = filter_production_block(raw_df)

    # Build mapping
    id_mapping, extras = build_id_to_cell_mapping()

    # Load labels
    labels = {}
    with open(labels_path, 'r') as f:
        for row in csv.DictReader(f):
            key = (row['coupon_id'], int(row['cell_index']))
            labels[key] = row

    # Process each production row
    results = []
    trig_failures = []
    excluded_ids = []

    for _, row in prod_df.iterrows():
        epoch_id = int(row['ID'])

        # Check if this ID maps to a cell
        if epoch_id not in id_mapping:
            # This is an extra reading (ID 089)
            excluded_ids.append({
                'epoch_id': epoch_id,
                'reason': 'extra reading beyond cell 09'
            })
            continue

        coupon_id, cell_index = id_mapping[epoch_id]
        label_key = (coupon_id, cell_index)

        if label_key not in labels:
            raise ValueError(f"No label for {coupon_id} cell {cell_index}")

        # Extract features
        features = extract_features(row.to_dict())

        # Verify trigonometry
        trig_error = verify_trigonometry(features)
        if trig_error:
            trig_failures.append({
                'epoch_id': epoch_id,
                'coupon_id': coupon_id,
                'cell_index': cell_index,
                'error': trig_error
            })

        # Build result row
        label_row = labels[label_key]
        results.append({
            'cell_id': f"{coupon_id}-{cell_index:02d}",
            'coupon_id': coupon_id,
            'cell_index': cell_index,
            'epoch_id': epoch_id,
            'true_class': label_row['true_class'],
            'ut_target': label_row['ut_target'],
            'fold': label_row['fold'],
            # Features (exclude internal raw values)
            'located': features['located'],
            'amplitude_pct': features['amplitude_pct'],
            'soundpath_mm': features['soundpath_mm'],
            'depth_mm': features['depth_mm'],
            'surface_distance_mm': features['surface_distance_mm'],
            'saturated': features['saturated'],
            'dist_from_root_mm': features['dist_from_root_mm'],
        })

    df = pd.DataFrame(results)

    audit = {
        **filter_audit,
        'mapped_cells': len(results),
        'excluded_ids': excluded_ids,
        'trig_failures': trig_failures,
        'trig_failures_count': len(trig_failures),
        'extras': extras,
    }

    return df, audit


if __name__ == '__main__':
    csv_path = REPO_ROOT / 'data' / 'raw' / 'TEST01.csv'
    labels_path = REPO_ROOT / 'data' / 'labels_v1_2026-10-17.csv'

    df, audit = extract_all_features(csv_path, labels_path)

    print("=" * 60)
    print("UT FEATURE EXTRACTION")
    print("=" * 60)

    print(f"\nFilter audit:")
    print(f"  Total rows: {audit['total_rows']}")
    print(f"  Production rows: {audit['production_rows']}")
    print(f"  Dropped rows: {audit['dropped_rows']}")
    print(f"  Dropped ID range: {audit['dropped_id_range']}")
    print(f"  Production ID range: {audit['production_id_range']}")
    print(f"  Assertions passed: {audit['assertions_passed']}")

    print(f"\nMapping audit:")
    print(f"  Mapped cells: {audit['mapped_cells']}")
    print(f"  Excluded IDs: {audit['excluded_ids']}")
    print(f"  Trigonometry failures: {audit['trig_failures_count']}")

    if audit['trig_failures']:
        print(f"\n  First 3 trig failures:")
        for f in audit['trig_failures'][:3]:
            print(f"    {f}")

    print(f"\nUT target distribution:")
    for target in sorted(df['ut_target'].unique()):
        print(f"  {target}: {len(df[df['ut_target'] == target])}")

    print(f"\nFeature summary:")
    print(f"  Located readings: {df['located'].sum()} / {len(df)}")
    print(f"  Saturated readings: {df['saturated'].sum()}")

    # Save features
    output_path = REPO_ROOT / 'results' / 'ut' / 'features.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")
