"""A-scan feature extraction. Feature names are locked; do not rename."""

from __future__ import annotations

import numpy as np

from weldfusion.utils.config import load_named

FEATURE_NAMES = tuple(load_named("ut_train")["features"])


def extract_features(
    ascan: np.ndarray,
    *,
    dt_us: float,
    backwall_window: tuple[int, int] | None = None,
    noise_fraction: float = 0.1,
    echo_threshold_db: float = 6.0,
) -> dict[str, float]:
    """Extract the six locked features from a 1D A-scan amplitude trace.

    `ascan` is linear amplitude (not dB). `dt_us` is the sample period in
    microseconds. Backwall window defaults to the last 20% of the trace.
    """
    x = np.asarray(ascan, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("ascan must be a non-empty 1D array")
    mag = np.abs(x)
    peak = float(np.max(mag))
    peak_amp_db = 20.0 * np.log10(peak + 1e-12)

    peak_idx = int(np.argmax(mag))
    tof_us = peak_idx * float(dt_us)

    half_power = peak / 2.0  # -6 dB
    above = mag >= half_power
    if np.any(above):
        # Width of the contiguous region containing the peak.
        left = peak_idx
        right = peak_idx
        while left > 0 and mag[left] >= half_power:
            left -= 1
        while right < len(mag) - 1 and mag[right] >= half_power:
            right += 1
        width_us = (right - left) * float(dt_us)
    else:
        width_us = 0.0

    n_noise = max(1, int(len(mag) * noise_fraction))
    noise_rms = float(np.sqrt(np.mean(mag[:n_noise] ** 2)))
    snr_db = 20.0 * np.log10((peak + 1e-12) / (noise_rms + 1e-12))

    if backwall_window is None:
        start = int(0.8 * len(mag))
        backwall_window = (start, len(mag))
    bw = mag[backwall_window[0] : backwall_window[1]]
    backwall_db = 20.0 * np.log10(float(np.max(bw)) + 1e-12) if bw.size else 0.0

    thresh = peak * (10.0 ** (-echo_threshold_db / 20.0))
    echoes = 0
    inside = False
    for v in mag:
        if v >= thresh and not inside:
            echoes += 1
            inside = True
        elif v < thresh:
            inside = False

    return {
        "peak_amplitude_db": peak_amp_db,
        "time_of_flight_us": tof_us,
        "signal_width_6db_us": width_us,
        "snr_db": snr_db,
        "backwall_echo_db": backwall_db,
        "echo_count": float(echoes),
    }
