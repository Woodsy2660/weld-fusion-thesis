"""
UT modality calibration.

Computes ECE and applies temperature scaling if ECE > 0.15.
Do NOT use isotonic regression — with 150 points across 3 classes it overfits.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Tuple[float, dict]:
    """
    Compute Expected Calibration Error.

    ECE = sum_b (n_b / N) * |acc_b - conf_b|

    where acc_b is accuracy in bin b and conf_b is mean confidence in bin b.
    """
    # Get predicted class and confidence
    y_pred = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])

    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        mask = bin_indices == i
        n_b = mask.sum()
        if n_b > 0:
            acc_b = accuracies[mask].mean()
            conf_b = confidences[mask].mean()
            ece += (n_b / len(y_true)) * abs(acc_b - conf_b)
            bin_data.append({
                'bin': i,
                'n': int(n_b),
                'accuracy': float(acc_b),
                'confidence': float(conf_b),
                'gap': float(abs(acc_b - conf_b)),
            })
        else:
            bin_data.append({
                'bin': i,
                'n': 0,
                'accuracy': None,
                'confidence': None,
                'gap': None,
            })

    return ece, {'n_bins': n_bins, 'bins': bin_data}


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return calibrated probabilities."""
    scaled_logits = logits / temperature
    # Softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def find_temperature(y_true: np.ndarray, logits: np.ndarray) -> float:
    """
    Find optimal temperature using NLL minimization.

    Temperature scaling: p_calibrated = softmax(z / T)
    """
    def nll_loss(T):
        if T <= 0:
            return 1e10
        proba = apply_temperature_scaling(logits, T)
        # Clip for numerical stability
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        # NLL
        nll = -np.log(proba[np.arange(len(y_true)), y_true]).mean()
        return nll

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
    return result.x


def proba_to_logits(proba: np.ndarray) -> np.ndarray:
    """Convert probabilities back to logits (log probabilities)."""
    # Clip to avoid log(0)
    proba = np.clip(proba, 1e-10, 1 - 1e-10)
    return np.log(proba)


def main():
    predictions_path = REPO_ROOT / 'results' / 'ut' / 'cv_predictions.json'
    output_dir = REPO_ROOT / 'results' / 'ut'

    print("=" * 60)
    print("UT CALIBRATION")
    print("=" * 60)

    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    y_true = np.array(predictions['y_true'])
    y_proba = np.array(predictions['y_proba'])
    classes = predictions['classes']

    # Convert string labels to indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([class_to_idx[c] for c in y_true])

    print(f"\nSamples: {len(y_true)}")
    print(f"Classes: {classes}")

    # Compute ECE before calibration
    ece_before, ece_details_before = compute_ece(y_true_idx, y_proba)
    print(f"\nECE (before calibration): {ece_before:.4f}")

    # Decide on calibration
    ECE_THRESHOLD = 0.15
    calibration_applied = False
    temperature = 1.0
    y_proba_calibrated = y_proba

    if ece_before > ECE_THRESHOLD:
        print(f"  ECE > {ECE_THRESHOLD}, applying temperature scaling...")

        # Convert proba to logits for temperature scaling
        logits = proba_to_logits(y_proba)

        # Find optimal temperature
        temperature = find_temperature(y_true_idx, logits)
        print(f"  Optimal temperature: {temperature:.4f}")

        # Apply calibration
        y_proba_calibrated = apply_temperature_scaling(logits, temperature)
        calibration_applied = True

        # Compute ECE after calibration
        ece_after, ece_details_after = compute_ece(y_true_idx, y_proba_calibrated)
        print(f"  ECE (after calibration): {ece_after:.4f}")
    else:
        print(f"  ECE <= {ECE_THRESHOLD}, no calibration needed")
        ece_after = ece_before
        ece_details_after = ece_details_before

    # Report
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"ECE before: {ece_before:.4f}")
    print(f"ECE after:  {ece_after:.4f}")
    print(f"Calibration applied: {calibration_applied}")
    if calibration_applied:
        print(f"Temperature: {temperature:.4f}")

    # Save calibration results
    calibration_results = {
        'ece_before': float(ece_before),
        'ece_after': float(ece_after),
        'calibration_applied': calibration_applied,
        'temperature': float(temperature),
        'ece_threshold': ECE_THRESHOLD,
        'n_samples': len(y_true),
        'classes': classes,
        'method': 'temperature_scaling' if calibration_applied else 'none',
        'note': 'Isotonic regression avoided due to small sample size (n=150)',
    }

    results_path = output_dir / 'calibration.json'
    with open(results_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    print(f"\nCalibration results saved to: {results_path}")

    # Save calibrated predictions
    calibrated_predictions = {
        'y_true': predictions['y_true'],
        'y_pred': predictions['y_pred'],
        'y_proba': y_proba_calibrated.tolist(),
        'classes': classes,
        'temperature': float(temperature),
        'calibrated': calibration_applied,
    }

    calibrated_path = output_dir / 'cv_predictions_calibrated.json'
    with open(calibrated_path, 'w') as f:
        json.dump(calibrated_predictions, f)
    print(f"Calibrated predictions saved to: {calibrated_path}")

    return calibration_results


if __name__ == '__main__':
    main()
