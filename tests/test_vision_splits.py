"""Tests for vision dataset splitting.

Critical requirement: no source_image may appear in more than one fold.
Splitting at patch level puts near-identical patches on both sides of the
train/test boundary and inflates accuracy. This is the same failure mode
as splitting cells within a coupon.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from weldfusion.vision.splits import (
    load_manifest,
    grouped_kfold_split,
    validate_no_image_leakage,
)


class TestGroupedSplitting:
    """Verify grouped splitting prevents source image leakage."""

    def test_no_image_in_multiple_folds(self, tmp_path: Path):
        """Each source_image must appear in exactly one fold."""
        # Create test manifest
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "patch_path,head,source_dataset,source_image,source_label,box_x,box_y,box_w,box_h,is_negative\n"
            "V1/000001.jpg,V1,test,img_001,pore,10,10,50,50,False\n"
            "V1/000002.jpg,V1,test,img_001,pore,60,10,50,50,False\n"
            "V1/000003.jpg,V1,test,img_002,pore,10,10,50,50,False\n"
            "V3/000004.jpg,V3,test,img_003,discontinuity,10,10,50,50,False\n"
            "V4/000005.jpg,V4,test,img_004,stain,10,10,50,50,False\n"
            "V4/000006.jpg,V4,test,img_004,deposit,60,10,50,50,False\n"
        )

        df = load_manifest(manifest_path)
        splits = grouped_kfold_split(df, n_splits=2, seed=42)

        # Check no image appears in multiple folds
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_images = set(df.iloc[train_idx]["source_image"])
            val_images = set(df.iloc[val_idx]["source_image"])
            overlap = train_images & val_images
            assert not overlap, (
                f"Fold {fold_idx}: images {overlap} appear in both train and val"
            )

    def test_validate_no_image_leakage_passes_clean_split(self, tmp_path: Path):
        """validate_no_image_leakage should pass for properly grouped splits."""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "patch_path,head,source_dataset,source_image,source_label,box_x,box_y,box_w,box_h,is_negative\n"
            "V1/000001.jpg,V1,test,img_001,pore,10,10,50,50,False\n"
            "V1/000002.jpg,V1,test,img_002,pore,10,10,50,50,False\n"
            "V3/000003.jpg,V3,test,img_003,discontinuity,10,10,50,50,False\n"
            "V4/000004.jpg,V4,test,img_004,stain,10,10,50,50,False\n"
        )

        df = load_manifest(manifest_path)
        splits = grouped_kfold_split(df, n_splits=2, seed=42)

        # Should not raise
        validate_no_image_leakage(df, splits)

    def test_validate_no_image_leakage_fails_bad_split(self, tmp_path: Path):
        """validate_no_image_leakage should fail if same image in train and val."""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "patch_path,head,source_dataset,source_image,source_label,box_x,box_y,box_w,box_h,is_negative\n"
            "V1/000001.jpg,V1,test,img_001,pore,10,10,50,50,False\n"
            "V1/000002.jpg,V1,test,img_001,pore,60,10,50,50,False\n"
        )

        df = load_manifest(manifest_path)

        # Manually create a bad split (same image in both)
        bad_splits = [([0], [1])]  # Both patches from img_001

        with pytest.raises(ValueError, match="leakage"):
            validate_no_image_leakage(df, bad_splits)


class TestAllPatchesCovered:
    """All patches must be assigned to exactly one fold."""

    def test_all_patches_in_exactly_one_fold(self, tmp_path: Path):
        """Every patch index must appear in exactly one train or val set."""
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "patch_path,head,source_dataset,source_image,source_label,box_x,box_y,box_w,box_h,is_negative\n"
            "V1/000001.jpg,V1,test,img_001,pore,10,10,50,50,False\n"
            "V1/000002.jpg,V1,test,img_002,pore,10,10,50,50,False\n"
            "V3/000003.jpg,V3,test,img_003,discontinuity,10,10,50,50,False\n"
            "V4/000004.jpg,V4,test,img_004,stain,10,10,50,50,False\n"
        )

        df = load_manifest(manifest_path)
        splits = grouped_kfold_split(df, n_splits=2, seed=42)

        all_indices = set(range(len(df)))
        covered = set()

        for train_idx, val_idx in splits:
            covered.update(val_idx)

        assert covered == all_indices, "Not all patches covered by validation folds"
