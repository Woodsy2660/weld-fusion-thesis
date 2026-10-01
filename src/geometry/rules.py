"""
ISO 5817 / AS/NZS 1554.1 rules-based dimensional assessment.
Steps 5-6 of geometry pipeline.
"""
import yaml


def load_limits(config_path="configs/geometry.yaml"):
    """Load acceptance criteria from config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["acceptance_criteria"]


def assess_conformance(measurements, limits):
    """
    Compare measured dimensions against acceptance limits.

    Args:
        measurements: dict from measure_geometry()
        limits: acceptance criteria dict

    Returns:
        dict: confidence scores per defect class {porosity, lack_of_fusion, crack, no_defect}
    """
    t = limits["plate_thickness_mm"]
    scores = {"porosity": 0.0, "lack_of_fusion": 0.0, "crack": 0.0, "no_defect": 0.0}

    # Undercut check -> can indicate crack or lack of fusion
    uc = measurements["undercut_mm"]
    uc_limit = min(limits["limits"]["undercut_max_mm"], limits["limits"]["undercut_ratio"] * t)
    if uc > uc_limit:
        severity = min(1.0, uc / (uc_limit * 2))
        scores["crack"] += severity * 0.5
        scores["lack_of_fusion"] += severity * 0.5

    # Reinforcement check -> excess indicates process issues
    rh = measurements["reinforcement_mm"]
    w = measurements["width_mm"]
    rh_limit = min(limits["limits"]["reinforcement_max_mm"], 1.0 + 0.1 * w)
    if rh > rh_limit:
        severity = min(1.0, rh / (rh_limit * 2))
        scores["porosity"] += severity * 0.3

    # Misalignment check -> indicates lack of fusion risk
    ma = measurements["misalignment_mm"]
    ma_limit = min(limits["limits"]["misalignment_max_mm"],
                   limits["limits"]["misalignment_ratio"] * t + 0.2)
    if ma > ma_limit:
        severity = min(1.0, ma / (ma_limit * 2))
        scores["lack_of_fusion"] += severity * 0.6

    # No-defect score = 1 - max defect score
    max_defect = max(scores["porosity"], scores["lack_of_fusion"], scores["crack"])
    scores["no_defect"] = max(0.0, 1.0 - max_defect)

    # Normalise to sum to 1
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    return scores
