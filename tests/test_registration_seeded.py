"""
Tests for the seeded registration pipeline.

Task 1b: Tests the PRIMARY registration path using manual seeds.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import csv

from src.geometry.registration_seeded import (
    HoleSeed,
    load_seeds,
    rim_fit_to_detected_hole,
    load_config,
)

from src.geometry.rim_detector import (
    RimFitResult,
    compute_gradient_magnitude,
    score_circle_at_centre,
    find_best_centre_hough,
)

from src.geometry.plate_isolation import (
    create_plate_mask_from_height,
    fit_board_depth_ransac,
    PlateRegion,
)


class TestSeedLoading:
    """Test seed CSV loading."""

    def test_load_seeds_parses_csv(self, tmp_path):
        """Seeds CSV is parsed correctly."""
        csv_path = tmp_path / "seeds.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["capture_id", "hole", "px_u", "px_v"])
            writer.writeheader()
            writer.writerow({"capture_id": "C01_top_r0", "hole": "d6", "px_u": "100", "px_v": "200"})
            writer.writerow({"capture_id": "C01_top_r0", "hole": "d8", "px_u": "500", "px_v": "200"})
            writer.writerow({"capture_id": "C02_root_r0", "hole": "d6", "px_u": "110", "px_v": "210"})
            writer.writerow({"capture_id": "C02_root_r0", "hole": "d8", "px_u": "510", "px_v": "210"})

        seeds = load_seeds(csv_path)

        assert len(seeds) == 2
        assert "C01_top_r0" in seeds
        assert "C02_root_r0" in seeds
        assert len(seeds["C01_top_r0"]) == 2
        assert seeds["C01_top_r0"][0].hole_type == "d6"
        assert seeds["C01_top_r0"][0].px_u == 100
        assert seeds["C01_top_r0"][0].px_v == 200

    def test_load_seeds_groups_by_capture(self, tmp_path):
        """Seeds are grouped by capture_id."""
        csv_path = tmp_path / "seeds.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["capture_id", "hole", "px_u", "px_v"])
            writer.writeheader()
            # Multiple captures
            for i in range(5):
                writer.writerow({"capture_id": f"C0{i+1}_top_r0", "hole": "d6", "px_u": "100", "px_v": "200"})
                writer.writerow({"capture_id": f"C0{i+1}_top_r0", "hole": "d8", "px_u": "500", "px_v": "200"})

        seeds = load_seeds(csv_path)
        assert len(seeds) == 5
        for capture_id in seeds:
            assert len(seeds[capture_id]) == 2


class TestRimFitConversion:
    """Test conversion from RimFitResult to DetectedHole."""

    def test_rim_fit_to_detected_hole_coordinates(self):
        """Coordinate convention is correct: RimFitResult (u,v) -> DetectedHole (row,col)."""
        rim_result = RimFitResult(
            centre_px=(100.5, 200.5),  # (u, v)
            hough_score=50.0,
            contrast_ratio=2.5,
            free_fit_centre=(101.0, 201.0),
            free_fit_axes=(20.0, 18.0),
            free_fit_angle=5.0,
            free_fit_diameter_mm=6.2,
            fixed_fit_centre=(100.8, 200.8),
            fixed_fit_residual=0.5,
            n_edge_points=60,
            accepted=True,
        )

        detected = rim_fit_to_detected_hole(rim_result, 6.0, "origin")

        # DetectedHole uses (row, col) = (v, u)
        assert detected.centre_px == (201.0, 101.0)  # (v, u)
        assert detected.fixed_fit_centre_px == (200.8, 100.8)  # (v, u)
        assert detected.hole_type == "origin"
        assert detected.estimated_diameter_mm == rim_result.free_fit_diameter_mm

    def test_confidence_calculation(self):
        """Confidence combines contrast ratio, edge points, and residual."""
        # High quality result
        good_rim = RimFitResult(
            centre_px=(100, 200),
            hough_score=100.0,
            contrast_ratio=3.0,  # High contrast
            free_fit_centre=(100, 200),
            free_fit_axes=(20, 20),
            free_fit_angle=0,
            free_fit_diameter_mm=6.0,
            fixed_fit_centre=(100, 200),
            fixed_fit_residual=0.5,  # Low residual
            n_edge_points=72,  # Many points
            accepted=True,
        )

        # Low quality result
        bad_rim = RimFitResult(
            centre_px=(100, 200),
            hough_score=30.0,
            contrast_ratio=1.2,  # Low contrast
            free_fit_centre=(100, 200),
            free_fit_axes=(20, 20),
            free_fit_angle=0,
            free_fit_diameter_mm=6.0,
            fixed_fit_centre=(100, 200),
            fixed_fit_residual=3.0,  # High residual
            n_edge_points=25,  # Few points
            accepted=True,
        )

        good_hole = rim_fit_to_detected_hole(good_rim, 6.0, "origin")
        bad_hole = rim_fit_to_detected_hole(bad_rim, 6.0, "origin")

        assert good_hole.confidence > bad_hole.confidence


class TestGradientDetector:
    """Test gradient-based rim detection components."""

    def test_gradient_magnitude_on_edge(self):
        """Gradient magnitude peaks at edges."""
        # Create simple image with a vertical edge
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[:, 50:] = 255

        mag = compute_gradient_magnitude(gray)

        # Gradient should peak at the edge (column 50)
        col_max = np.argmax(np.mean(mag, axis=0))
        assert 48 <= col_max <= 52  # Near the edge

    def test_circle_scoring_on_synthetic_circle(self):
        """Circle scorer finds correct centre on synthetic circle."""
        # Create image with a circle edge
        img = np.zeros((100, 100), dtype=np.uint8)
        centre = (50, 50)
        radius = 20

        # Draw a circle
        import cv2
        cv2.circle(img, centre, radius, 255, 2)

        mag = compute_gradient_magnitude(img)

        # Score at true centre should be higher than off-centre
        score_at_centre = score_circle_at_centre(mag, centre[0], centre[1], radius)
        score_off_centre = score_circle_at_centre(mag, centre[0] + 10, centre[1], radius)

        assert score_at_centre > score_off_centre

    def test_hough_finds_circle_centre(self):
        """Hough search finds approximate circle centre."""
        # Create synthetic circle
        img = np.zeros((100, 100), dtype=np.uint8)
        true_centre = (50, 50)
        radius = 15

        import cv2
        cv2.circle(img, true_centre, radius, 255, 2)

        mag = compute_gradient_magnitude(img)

        best_centre, best_score, _ = find_best_centre_hough(
            mag, radius, true_centre, window_radius_px=10
        )

        # Should find centre within a few pixels
        error = np.sqrt((best_centre[0] - true_centre[0])**2 +
                        (best_centre[1] - true_centre[1])**2)
        assert error < 3.0


class TestPlateIsolation:
    """Test plate isolation from depth."""

    def test_height_threshold_separates_plate_from_board(self):
        """Height threshold correctly separates plate from support board."""
        # Create synthetic depth map
        # Board at 320mm, plate at 310mm (10mm above)
        depth_mm = np.full((480, 640), 320.0)

        # Plate region in centre
        depth_mm[100:380, 100:540] = 310.0

        region = create_plate_mask_from_height(depth_mm, plate_thickness_mm=10.0)

        # Plate should be detected
        assert region.mask[200, 300]  # Centre of plate
        assert not region.mask[50, 50]  # Corner (board)
        assert region.method == "height_threshold"

    def test_board_depth_ransac(self):
        """RANSAC correctly fits board depth from border pixels."""
        depth_mm = np.full((480, 640), 320.0)

        # Add some noise
        rng = np.random.default_rng(42)
        depth_mm += rng.normal(0, 1, depth_mm.shape)

        z_board, n_valid = fit_board_depth_ransac(depth_mm, border_fraction=0.1)

        assert abs(z_board - 320.0) < 2.0  # Within 2mm
        assert n_valid > 1000  # Plenty of border pixels

    def test_fallback_to_largest_component(self):
        """Falls back to largest connected component when border is invalid."""
        # Create depth with no valid border (all NaN in border)
        depth_mm = np.full((480, 640), np.nan)

        # Only centre has valid depth
        depth_mm[100:380, 100:540] = 310.0

        region = create_plate_mask_from_height(depth_mm)

        # Should use largest_connected method
        assert region.method == "largest_connected"
        assert region.mask[200, 300]  # Centre should be in mask


class TestConfigIntegration:
    """Test config loading and parameter access."""

    def test_config_has_required_sections(self):
        """Config has all required sections for seeded registration."""
        config = load_config()

        # Required sections
        assert "hole_detection" in config
        assert "validation" in config
        assert "datum_holes" in config
        assert "deprojection" in config
        assert "plate_isolation" in config

    def test_frozen_parameters_present(self):
        """Frozen parameters are set in config."""
        config = load_config()

        # B4: Gradient-based rim detection
        assert config["hole_detection"]["window_radius_mm"] == 12.0
        assert config["hole_detection"]["contrast_threshold"] == 1.5

        # Loose gate for pair selection
        assert config["hole_detection"]["pair_gate_mm"] == 10.0

        # Tight tolerance for final validation
        assert config["validation"]["baseline_tolerance_mm"] == 0.5

    def test_plate_isolation_parameters(self):
        """Plate isolation parameters are set."""
        config = load_config()

        iso = config["plate_isolation"]
        assert iso["height_threshold_mm"] == 5.0  # t/2
        assert iso["border_fraction"] == 0.10
        assert iso["min_border_pixels"] == 5000
