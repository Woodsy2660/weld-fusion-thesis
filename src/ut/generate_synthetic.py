"""
Generate synthetic A-scan feature vectors from published distributions.
Based on Sun et al. (2023) and Boris et al. (2025).
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES_PER_CLASS = 100

# Feature distributions per defect class (mean, std)
# Derived from published ranges in Sun et al. 2023 and Boris et al. 2025
DISTRIBUTIONS = {
    "porosity": {
        "peak_amplitude_db": (35, 8),
        "time_of_flight_us": (4.5, 1.5),
        "signal_width_6db_us": (1.8, 0.5),
        "snr_db": (15, 4),
        "backwall_echo_db": (28, 5),
        "echo_count": (3, 1.5),
    },
    "lack_of_fusion": {
        "peak_amplitude_db": (65, 10),
        "time_of_flight_us": (3.2, 0.8),
        "signal_width_6db_us": (0.8, 0.3),
        "snr_db": (32, 6),
        "backwall_echo_db": (15, 5),
        "echo_count": (1.5, 0.8),
    },
    "crack": {
        "peak_amplitude_db": (72, 8),
        "time_of_flight_us": (3.0, 0.7),
        "signal_width_6db_us": (0.5, 0.2),
        "snr_db": (38, 5),
        "backwall_echo_db": (10, 4),
        "echo_count": (1.2, 0.5),
    },
    "no_defect": {
        "peak_amplitude_db": (10, 4),
        "time_of_flight_us": (6.5, 0.3),
        "signal_width_6db_us": (0.3, 0.1),
        "snr_db": (3, 2),
        "backwall_echo_db": (45, 5),
        "echo_count": (0.2, 0.4),
    },
}


def generate():
    np.random.seed(SEED)
    rows = []

    for cls, features in DISTRIBUTIONS.items():
        for _ in range(N_SAMPLES_PER_CLASS):
            row = {"defect_class": cls}
            for feat_name, (mean, std) in features.items():
                value = np.random.normal(mean, std)
                value = max(0, value)  # clamp non-negative
                row[feat_name] = round(value, 3)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    out_dir = Path("data/raw/ut/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_ascan_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic A-scan features -> {out_path}")
    print(df["defect_class"].value_counts())


if __name__ == "__main__":
    generate()
