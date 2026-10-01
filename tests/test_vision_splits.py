"""
The split must be GROUPED by source image.

Per-instance cropping emits many patches per source image -- same weld, same
lighting, frequently overlapping pixels. If one source image's crops land in
both train and test, test accuracy is inflated and no metric in the report will
reveal it. That is the failure these tests exist to catch.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402
from src.vision.merge_dedup_split import (  # noqa: E402
    greedy_composition_split, stratified_group_split,
)

MANIFEST = REPO_ROOT / "data" / "processed" / "vision_patches" / "manifest.csv"


def _load():
    if not MANIFEST.exists():
        pytest.skip("patch manifest not built yet")
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Synthetic -- always run, no data required
# ---------------------------------------------------------------------------
def _heads():
    """Head names come from the frame, never hardcoded."""
    return load_frame().head_names


def _synthetic(n_images=400, per_image=5):
    heads = _heads()
    k = len(heads)
    rows = []
    for i in range(n_images):
        for j in range(per_image):
            rows.append({
                "patch_path": f"{heads[i % k]}/img{i}_{j}.jpg",
                "head": heads[i % k],
                "source_dataset": "synthetic",
                "source_label": f"label{i % k}",
                "source_image": f"/src/img{i}.jpg",
                "source_frame": f"/src/img{i}.jpg",
                "is_negative": "0",
                "pad_fraction": "0.0",
            })
    return rows


def test_grouped_split_never_shares_a_source_image():
    rows, _, _ = greedy_composition_split(_synthetic(), seed=42)
    splits_per_image = defaultdict(set)
    for r in rows:
        splits_per_image[r["source_image"]].add(r["split"])
    offenders = {k: v for k, v in splits_per_image.items() if len(v) > 1}
    assert not offenders, f"source images in multiple splits: {list(offenders)[:5]}"


def test_grouped_split_assigns_every_patch():
    rows, _, _ = greedy_composition_split(_synthetic(), seed=42)
    assert all(r.get("split") in {"train", "val", "test"} for r in rows)


def test_grouped_split_is_deterministic_for_a_seed():
    a, _, _ = greedy_composition_split(_synthetic(), seed=7)
    b, _, _ = greedy_composition_split(_synthetic(), seed=7)
    assert [r["split"] for r in a] == [r["split"] for r in b]


def test_grouped_split_roughly_honours_the_ratios():
    rows, _, _ = greedy_composition_split(_synthetic(n_images=1000), seed=42)
    counts = Counter(r["split"] for r in rows)
    total = sum(counts.values())
    assert 0.70 <= counts["train"] / total <= 0.90
    assert 0.03 <= counts["val"] / total <= 0.20
    assert 0.03 <= counts["test"] / total <= 0.20


def test_every_head_present_in_every_split_when_data_allows():
    rows, _, _ = greedy_composition_split(_synthetic(n_images=1000), seed=42)
    per = defaultdict(set)
    for r in rows:
        per[r["split"]].add(r["head"])
    for split in ("train", "val", "test"):
        assert per[split] == set(_heads()), (
            f"split {split} is missing heads: {per[split]}"
        )


# ---------------------------------------------------------------------------
# Against the real manifest, once it exists
# ---------------------------------------------------------------------------
def test_real_manifest_has_no_source_image_in_two_splits():
    rows = _load()
    if "split" not in rows[0]:
        pytest.skip("split column not written yet; run merge_dedup_split.py")
    splits_per_image = defaultdict(set)
    for r in rows:
        splits_per_image[r["source_image"]].add(r["split"])
    offenders = {k: v for k, v in splits_per_image.items() if len(v) > 1}
    assert not offenders, (
        f"{len(offenders)} source images leak across splits, e.g. "
        f"{list(offenders.items())[:3]}"
    )


def test_real_manifest_has_no_source_FRAME_in_two_splits():
    """
    The stronger constraint: Roboflow exports many augmented copies of one
    frame under different filenames. Grouping only on source_image would let
    siblings straddle the split and silently inflate test accuracy.
    """
    rows = _load()
    if "split" not in rows[0] or "source_frame" not in rows[0]:
        pytest.skip("source_frame/split not present; re-run the pipeline")
    splits_per_frame = defaultdict(set)
    for r in rows:
        splits_per_frame[r["source_frame"]].add(r["split"])
    offenders = {k: v for k, v in splits_per_frame.items() if len(v) > 1}
    assert not offenders, (
        f"{len(offenders)} source FRAMES leak across splits, e.g. "
        f"{list(offenders.items())[:3]}"
    )


def test_real_manifest_has_required_columns():
    rows = _load()
    required = {"patch_path", "head", "source_dataset", "source_image",
                "source_frame", "source_label", "box_x", "box_y", "box_w",
                "box_h", "is_negative", "pad_fraction"}
    assert required <= set(rows[0]), f"missing: {required - set(rows[0])}"


def test_real_manifest_heads_match_the_frame():
    rows = _load()
    frame = load_frame()
    assert {r["head"] for r in rows} <= set(frame.head_names)


def test_real_manifest_source_images_exist():
    """source_image is what the grouping keys on; a broken path breaks Part E."""
    rows = _load()
    seen = {r["source_image"] for r in rows}
    sample = list(seen)[:50]
    missing = [p for p in sample if not Path(p).exists()]
    assert not missing, f"source images not found on disk: {missing[:3]}"


def test_negatives_land_in_the_ignorance_head():
    """
    Negatives are annotation-free regions, so they belong to whichever head is
    the modality's ignorance set -- the one containing every symbol the others
    do not resolve. Derived from the frame rather than hardcoded to V4.
    """
    rows = _load()
    frame = load_frame()
    ignorance = max(frame.head_names, key=lambda h: len(frame.focal_element(h)))
    bad = {r["head"] for r in rows if r["is_negative"] == "1"} - {ignorance}
    assert not bad, (
        f"negative patches assigned outside the ignorance head "
        f"{ignorance}: {bad}"
    )


def test_negative_box_sizes_match_the_positive_distribution():
    """
    If negatives were systematically larger or smaller than positives, the
    network could learn "crop size predicts class" and the metrics would look
    fine. Compare medians.
    """
    rows = _load()
    pos = sorted(float(r["box_w"]) * float(r["box_h"])
                 for r in rows if r["is_negative"] == "0")
    neg = sorted(float(r["box_w"]) * float(r["box_h"])
                 for r in rows if r["is_negative"] == "1")
    if len(neg) < 30:
        pytest.skip("too few negatives to compare distributions")
    med_p, med_n = pos[len(pos) // 2], neg[len(neg) // 2]
    ratio = max(med_p, med_n) / max(1e-9, min(med_p, med_n))
    assert ratio < 4.0, (
        f"negative median box area {med_n:.0f} vs positive {med_p:.0f} "
        f"({ratio:.1f}x apart) -- crop size may predict class"
    )
