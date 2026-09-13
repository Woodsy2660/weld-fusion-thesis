"""Grouped splitting for vision patches.

Patches from the same source image must never appear on both sides of a
train/test boundary. This prevents data leakage from overlapping crops
of the same weld segment.

Uses GroupKFold with groups = source_image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def load_manifest(manifest_path: Path | str) -> pd.DataFrame:
    """Load patch manifest CSV."""
    return pd.read_csv(manifest_path)


def grouped_kfold_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Generate grouped K-fold splits based on source_image.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest dataframe with 'source_image' column.
    n_splits : int
        Number of folds.
    seed : int
        Random seed (used for shuffling groups before split).

    Returns
    -------
    list of (train_indices, val_indices) tuples
    """
    if "source_image" not in df.columns:
        raise ValueError("Manifest must have 'source_image' column for grouped splitting")

    groups = df["source_image"].values
    y = df["head"].values  # Use head as stratification target

    # GroupKFold ensures no group spans folds
    gkf = GroupKFold(n_splits=n_splits)

    splits = []
    for train_idx, val_idx in gkf.split(df, y, groups):
        splits.append((list(train_idx), list(val_idx)))

    return splits


def grouped_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Single train/test split grouped by source_image.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest dataframe with 'source_image' column.
    test_size : float
        Fraction of data for test set.
    seed : int
        Random seed.

    Returns
    -------
    (train_indices, test_indices)
    """
    if "source_image" not in df.columns:
        raise ValueError("Manifest must have 'source_image' column for grouped splitting")

    groups = df["source_image"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    return list(train_idx), list(test_idx)


def validate_no_image_leakage(
    df: pd.DataFrame,
    splits: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> None:
    """Verify no source_image appears in both train and val for any fold.

    Raises ValueError if leakage is detected.
    """
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_images = set(df.iloc[list(train_idx)]["source_image"])
        val_images = set(df.iloc[list(val_idx)]["source_image"])
        overlap = train_images & val_images

        if overlap:
            raise ValueError(
                f"Image leakage in fold {fold_idx}: {len(overlap)} images appear "
                f"in both train and val. First few: {list(overlap)[:5]}"
            )


def get_source_image_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count patches per source image."""
    return df.groupby("source_image").size().to_dict()


def get_fold_stats(
    df: pd.DataFrame,
    splits: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> list[dict]:
    """Compute statistics for each fold."""
    stats = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[list(train_idx)]
        val_df = df.iloc[list(val_idx)]

        stats.append({
            "fold": fold_idx,
            "n_train_patches": len(train_idx),
            "n_val_patches": len(val_idx),
            "n_train_images": train_df["source_image"].nunique(),
            "n_val_images": val_df["source_image"].nunique(),
            "train_head_counts": train_df["head"].value_counts().to_dict(),
            "val_head_counts": val_df["head"].value_counts().to_dict(),
        })

    return stats
