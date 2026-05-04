"""
Dempster-Shafer evidence theory implementation with Murphy averaging.
"""
import numpy as np
from itertools import combinations


def confidence_to_bpa(confidence_scores, uncertainty_floor=0.05):
    """
    Convert classifier confidence scores to Basic Probability Assignments.

    Args:
        confidence_scores: dict {class: probability} summing to ~1.0
        uncertainty_floor: minimum mass assigned to uncertainty (theta)

    Returns:
        dict: BPA including uncertainty mass m(theta)
    """
    total = sum(confidence_scores.values())
    scale = 1.0 - uncertainty_floor

    bpa = {}
    for cls, conf in confidence_scores.items():
        bpa[cls] = (conf / total) * scale

    bpa["theta"] = uncertainty_floor
    return bpa


def dempster_combine(bpa1, bpa2):
    """
    Combine two BPAs using Dempster's rule of combination.

    Args:
        bpa1, bpa2: dicts of mass assignments

    Returns:
        dict: combined BPA
    """
    classes = [k for k in set(list(bpa1.keys()) + list(bpa2.keys())) if k != "theta"]

    # Compute conflict (K) and combined masses
    combined = {}
    conflict = 0.0

    for h1, m1 in bpa1.items():
        for h2, m2 in bpa2.items():
            product = m1 * m2

            if h1 == "theta" and h2 == "theta":
                combined["theta"] = combined.get("theta", 0) + product
            elif h1 == "theta":
                combined[h2] = combined.get(h2, 0) + product
            elif h2 == "theta":
                combined[h1] = combined.get(h1, 0) + product
            elif h1 == h2:
                combined[h1] = combined.get(h1, 0) + product
            else:
                conflict += product

    # Normalise by (1 - conflict)
    if conflict >= 1.0:
        raise ValueError(f"Total conflict = {conflict:.4f}, cannot combine")

    norm = 1.0 / (1.0 - conflict)
    return {k: v * norm for k, v in combined.items()}


def murphy_average(bpa_list):
    """
    Murphy's (2000) averaging: average all BPAs first, then combine
    the average with itself (n-1) times.

    Args:
        bpa_list: list of BPA dicts

    Returns:
        dict: fused BPA
    """
    n = len(bpa_list)
    if n == 1:
        return bpa_list[0]

    # Step 1: Average all BPAs
    all_keys = set()
    for bpa in bpa_list:
        all_keys.update(bpa.keys())

    avg_bpa = {}
    for key in all_keys:
        avg_bpa[key] = sum(bpa.get(key, 0) for bpa in bpa_list) / n

    # Step 2: Combine the average with itself (n-1) times
    result = avg_bpa.copy()
    for _ in range(n - 1):
        result = dempster_combine(result, avg_bpa)

    return result


def fuse_modalities(vision_scores, geometry_scores, ut_scores,
                     uncertainty_floor=0.05):
    """
    Full fusion pipeline: convert scores to BPAs, apply Murphy averaging.

    Args:
        vision_scores: dict {class: confidence} from ResNet-50
        geometry_scores: dict {class: confidence} from rules-based
        ut_scores: dict {class: confidence} from Random Forest

    Returns:
        dict: fused classification with confidence per class
    """
    bpas = [
        confidence_to_bpa(vision_scores, uncertainty_floor),
        confidence_to_bpa(geometry_scores, uncertainty_floor),
        confidence_to_bpa(ut_scores, uncertainty_floor),
    ]

    fused = murphy_average(bpas)

    # Remove theta for final classification
    theta = fused.pop("theta", 0)
    predicted_class = max(fused, key=fused.get)

    return {
        "classification": predicted_class,
        "confidence": fused[predicted_class],
        "all_scores": fused,
        "uncertainty": theta,
    }
