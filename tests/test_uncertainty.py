"""D435 depth noise model and averaging reduction."""

from __future__ import annotations

import pytest

from weldfusion.geometry.uncertainty import (
    averaged_sigma,
    combine_uncertainty,
    realsense_d435_delta_z,
)


def test_d435_rms_at_200mm():
    dz = realsense_d435_delta_z(200.0)
    assert dz == pytest.approx(0.143, abs=0.005)


def test_25_row_average():
    dz = realsense_d435_delta_z(200.0)
    assert averaged_sigma(dz, 25) == pytest.approx(dz / 5.0)
    assert averaged_sigma(dz, 25) == pytest.approx(0.0286, abs=0.001)


def test_combine_rss():
    assert combine_uncertainty(3.0, 4.0) == pytest.approx(5.0)
