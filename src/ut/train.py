"""
UT modality classifier training.

Multinomial logistic regression with k=4 coupon-grouped cross-validation.
Uses existing fold column from labels. See 06-ultrasonic-module.md for design.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]

# Features from EPOCH 650 gate readings
FEATURE_COLS = [
    'located',
    'amplitude_pct',
    'soundpath_mm',
    'depth_mm',
    'surface_distance_mm',
    'saturated',
    'dist_from_root_mm',
]

# Focal elements for 3-class UT target
TARGET_COL = 'ut_target'
CLASSES = ['{L}', '{N,U}', '{P}']  # Alphabetical order


def load_features(features_path: Path) -> pd.DataFrame:
    """Load extracted features."""
    df = pd.read_csv(features_path)
    # Verify columns
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df


def trivial_baseline_f1(y: np.ndarray, classes: List[str]) -> float:
    """
    Compute macro-F1 for trivial random classifier.

    A random classifier predicts proportionally to class prevalence.
    Expected F1 per class = 2 * p_i * p_i / (p_i + p_i) = p_i
    Macro-F1 = mean(p_i) = 1/n_classes

    For 3 classes: trivial F1 = 1/3 = 0.333
    But with imbalanced classes, need to compute properly.
    """
    # For a stratified random classifier, expected macro-F1 is approximately
    # the harmonic mean approach. Simpler: 1/n_classes for balanced = 0.333
    # For our imbalanced case ({N,U}=93, {L}=31, {P}=26), compute empirically.

    from sklearn.dummy import DummyClassifier

    # Most frequent baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(y, y)
    y_pred_mf = dummy.predict(y)
    f1_mf = f1_score(y, y_pred_mf, labels=classes, average='macro', zero_division=0)

    # Stratified random baseline
    dummy_strat = DummyClassifier(strategy='stratified', random_state=42)
    dummy_strat.fit(y, y)
    y_pred_strat = dummy_strat.predict(y)
    f1_strat = f1_score(y, y_pred_strat, labels=classes, average='macro', zero_division=0)

    return {
        'most_frequent': f1_mf,
        'stratified': f1_strat,
    }


def train_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    classes: List[str],
) -> Tuple[Dict, List[LogisticRegression], List[StandardScaler]]:
    """
    Train with k-fold cross-validation using existing fold column.

    Returns:
        metrics: Dict with per-fold and aggregate metrics
        models: List of fitted LogisticRegression models
        scalers: List of fitted StandardScaler transforms
    """
    folds = sorted(df['fold'].unique())
    n_folds = len(folds)

    print(f"Training with {n_folds}-fold CV")
    print(f"Classes: {classes}")
    print(f"Features: {feature_cols}")

    fold_metrics = []
    models = []
    scalers = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold in folds:
        # Split
        train_mask = df['fold'] != fold
        val_mask = df['fold'] == fold

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_val = df.loc[val_mask, feature_cols].values
        y_val = df.loc[val_mask, target_col].values

        # Standardise
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train logistic regression
        # Note: multi_class='multinomial' is deprecated; lbfgs uses multinomial by default
        model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)

        # Metrics
        f1_macro = f1_score(y_val, y_pred, labels=classes, average='macro')
        f1_per_class = f1_score(y_val, y_pred, labels=classes, average=None)
        accuracy = accuracy_score(y_val, y_pred)

        fold_metrics.append({
            'fold': fold,
            'n_train': len(y_train),
            'n_val': len(y_val),
            'f1_macro': f1_macro,
            'accuracy': accuracy,
            'f1_per_class': dict(zip(classes, f1_per_class.tolist())),
        })

        models.append(model)
        scalers.append(scaler)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba.tolist())

        print(f"  Fold {fold}: F1-macro={f1_macro:.3f}, Acc={accuracy:.3f}, n_val={len(y_val)}")

    # Aggregate metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    f1_macro_overall = f1_score(all_y_true, all_y_pred, labels=classes, average='macro')
    f1_per_class_overall = f1_score(all_y_true, all_y_pred, labels=classes, average=None)
    accuracy_overall = accuracy_score(all_y_true, all_y_pred)
    conf_matrix = confusion_matrix(all_y_true, all_y_pred, labels=classes)

    # Trivial baseline
    baseline = trivial_baseline_f1(df[target_col].values, classes)

    metrics = {
        'n_folds': n_folds,
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'classes': classes,
        'fold_metrics': fold_metrics,
        'f1_macro_mean': np.mean([f['f1_macro'] for f in fold_metrics]),
        'f1_macro_std': np.std([f['f1_macro'] for f in fold_metrics]),
        'f1_macro_overall': f1_macro_overall,
        'f1_per_class_overall': dict(zip(classes, f1_per_class_overall.tolist())),
        'accuracy_overall': accuracy_overall,
        'confusion_matrix': conf_matrix.tolist(),
        'baseline_most_frequent_f1': baseline['most_frequent'],
        'baseline_stratified_f1': baseline['stratified'],
        'all_y_true': all_y_true.tolist(),
        'all_y_pred': all_y_pred.tolist(),
        'all_y_proba': all_y_proba,
    }

    return metrics, models, scalers


def main():
    features_path = REPO_ROOT / 'results' / 'ut' / 'features.csv'
    output_dir = REPO_ROOT / 'results' / 'ut'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UT CLASSIFIER TRAINING")
    print("=" * 60)

    # Load features
    df = load_features(features_path)
    print(f"\nLoaded {len(df)} samples")

    # Class distribution
    print("\nClass distribution:")
    for cls in CLASSES:
        count = (df[TARGET_COL] == cls).sum()
        print(f"  {cls}: {count}")

    # Train
    metrics, models, scalers = train_cv(
        df, FEATURE_COLS, TARGET_COL, CLASSES
    )

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nCross-validation F1-macro:")
    print(f"  Mean: {metrics['f1_macro_mean']:.3f} ± {metrics['f1_macro_std']:.3f}")
    print(f"  Overall (pooled): {metrics['f1_macro_overall']:.3f}")

    print(f"\nPer-class F1 (overall):")
    for cls, f1 in metrics['f1_per_class_overall'].items():
        print(f"  {cls}: {f1:.3f}")

    print(f"\nBaseline comparison:")
    print(f"  Most-frequent baseline: {metrics['baseline_most_frequent_f1']:.3f}")
    print(f"  Stratified random baseline: {metrics['baseline_stratified_f1']:.3f}")

    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"  Classes: {CLASSES}")
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"  {CLASSES[i]}: {row}")

    # Save metrics
    metrics_path = output_dir / 'metrics_cv.json'

    def convert_numpy(obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    metrics_save = {k: convert_numpy(v) for k, v in metrics.items()
                    if k not in ['all_y_true', 'all_y_pred', 'all_y_proba']}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_save, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save predictions for calibration
    predictions_path = output_dir / 'cv_predictions.json'
    predictions = {
        'y_true': metrics['all_y_true'],
        'y_pred': metrics['all_y_pred'],
        'y_proba': metrics['all_y_proba'],
        'classes': CLASSES,
    }
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Predictions saved to: {predictions_path}")

    return metrics, models, scalers


if __name__ == '__main__':
    main()
