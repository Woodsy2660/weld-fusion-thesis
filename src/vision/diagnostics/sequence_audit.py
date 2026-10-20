"""
Task F1, F2, F4 -- sequence-structure and grouping-consistency audit.

The existing leakage guard asserts that no source_frame and no source_image
spans splits. That closes exact and filename-linked duplication. It does not
close SEQUENCE structure: consecutive frames of one weld from a video-derived
dataset carry different filenames, are different source_frame values, and
survive perceptual hashing because small camera motion pushes them past Hamming
distance 5. The dedup audit measured phash recall at 0.1892 against filename
identity, so most siblings were caught by filename -- a mechanism that works
only within a single Roboflow export.

F1  derive a sequence stem per source image and describe the structure
F2  measure how much of test shares a stem with train  <- headline
F4  verify the perceptual union-find and the split assignment agree

No GPU, no checkpoint. F3 (embedding nearest neighbours) lives in nn_audit.py
because it needs the trained model.

Usage:
    python src/vision/diagnostics/sequence_audit.py
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Roboflow mangles the original name to "<orig>_<ext>.rf.<hash>.<ext>", so the
# original extension survives as a trailing "_jpg"/"_png" token.
RF_SPLIT = re.compile(r"\.rf\.", re.IGNORECASE)
EXT_TOKEN = re.compile(r"_(jpg|jpeg|png|bmp|JPG|JPEG|PNG)$")
FINAL_DIGITS = re.compile(r"(\d+)(?!.*\d)")
# Roboflow tiling: IMG_3688_block_0_0_png is one tile of IMG_3688. Without this
# the final-digit rule keys on the tile column, so tiles of one photograph get
# different stems and can straddle the split.
TILE_TOKEN = re.compile(r"_block_\d+_\d+(_(png|jpe?g|bmp))?$", re.IGNORECASE)

SEQ_MIN_MEMBERS = 3
SEQ_MAX_GAP = 2

# F5 thresholds
T_SEQUENCE_SHARE = 0.05      # test patches sharing a stem with train
T_GUARD_INCONSISTENCY = 0.0  # any non-zero is a bug


def sequence_stem(source_image):
    """
    'weld_0042.jpg' and 'weld_0043.jpg' -> 'weld_####'.

    Returns (stem, index) where index is the integer that was replaced, or None
    when the basename carries no digit run -- a dataset that does not number
    its files simply has no sequences, which is a valid result.
    """
    name = Path(source_image).name
    name = RF_SPLIT.split(name)[0]          # drop .rf.<hash>.<ext>
    name = Path(name).stem                  # drop a plain extension
    name = EXT_TOKEN.sub("", name)          # drop the Roboflow "_jpg" token
    name = TILE_TOKEN.sub("", name)         # tiles share their photo's stem

    m = None
    for m in FINAL_DIGITS.finditer(name):
        pass
    if m is None:
        return name, None
    stem = name[:m.start()] + "#" * len(m.group(1)) + name[m.end():]
    return stem, int(m.group(1))


def contiguous_runs(indices, max_gap=SEQ_MAX_GAP):
    """Longest near-contiguous run length among sorted integer indices."""
    if not indices:
        return 0
    xs = sorted(set(indices))
    best = run = 1
    for a, b in zip(xs, xs[1:]):
        if b - a <= max_gap:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


# ---------------------------------------------------------------------------
def f1_sequence_detection(frames_df):
    print("\n" + "=" * 78)
    print("F1 -- sequence detection")
    print("=" * 78)

    out = {}
    for ds, sub in frames_df.groupby("source_dataset"):
        stems = defaultdict(list)
        for r in sub.itertuples():
            stems[r.stem].append(r.seq_index)

        sizes = Counter(len(v) for v in stems.values())
        multi = {k: v for k, v in stems.items() if len(v) >= 2}

        seq_like = {}
        for k, idxs in stems.items():
            clean = [i for i in idxs if i is not None]
            if len(clean) >= SEQ_MIN_MEMBERS:
                run = contiguous_runs(clean)
                if run >= SEQ_MIN_MEMBERS:
                    seq_like[k] = {"members": len(clean),
                                   "longest_run": run,
                                   "min": min(clean), "max": max(clean)}

        n_unnumbered = sum(1 for r in sub.itertuples() if r.seq_index is None)
        biggest = sorted(stems.items(), key=lambda kv: -len(kv[1]))[:20]
        examples = []
        for k, idxs in biggest:
            names = sub[sub.stem == k].source_image.head(3).tolist()
            examples.append({
                "stem": k, "members": len(idxs),
                "longest_near_contiguous_run": contiguous_runs(
                    [i for i in idxs if i is not None]),
                "examples": [Path(n).name for n in names],
            })

        out[ds] = {
            "n_frames": int(len(sub)),
            "n_distinct_stems": len(stems),
            "n_stems_with_multiple_members": len(multi),
            "frames_without_a_digit_run": n_unnumbered,
            "members_per_stem_distribution": {str(k): v
                                              for k, v in sorted(sizes.items())},
            "n_sequence_like_stems": len(seq_like),
            "frames_in_sequence_like_stems": int(
                sum(v["members"] for v in seq_like.values())),
            "largest_stems": examples,
        }

        print(f"\n  {ds}: {len(sub)} frames -> {len(stems)} distinct stems")
        print(f"    stems with >1 member          : {len(multi)}")
        print(f"    frames with no digit run      : {n_unnumbered}")
        print(f"    sequence-like stems (>={SEQ_MIN_MEMBERS} members, "
              f"run>={SEQ_MIN_MEMBERS}, gap<={SEQ_MAX_GAP}): {len(seq_like)}")
        print(f"    frames inside them            : "
              f"{sum(v['members'] for v in seq_like.values())}")
        print(f"    members-per-stem: "
              f"{dict(sorted(sizes.items())[:8])}"
              f"{' ...' if len(sizes) > 8 else ''}")
        print(f"    largest stems:")
        for e in examples[:5]:
            print(f"      {e['stem']:<42} n={e['members']:<5} "
                  f"run={e['longest_near_contiguous_run']:<4} "
                  f"e.g. {e['examples'][0][:44]}")
    return out


def f2_cross_split(frames_df, patches_df):
    print("\n" + "=" * 78)
    print("F2 -- cross-split sequence audit")
    print("=" * 78)

    stem_splits = defaultdict(set)
    for r in frames_df.itertuples():
        stem_splits[r.stem].add(r.split)

    multi = {k: v for k, v in stem_splits.items()
             if len(frames_df[frames_df.stem == k]) >= 2}
    spanning = {k for k, v in stem_splits.items() if len(v) > 1}
    multi_spanning = {k for k in multi if k in spanning}

    train_stems = set(frames_df[frames_df.split == "train"].stem)

    def share_with_train(split):
        sub = patches_df[patches_df.split == split]
        if not len(sub):
            return 0.0, 0, 0
        hit = sub.stem.isin(train_stems).sum()
        return float(hit / len(sub)), int(hit), int(len(sub))

    test_share, test_hit, test_n = share_with_train("test")
    val_share, val_hit, val_n = share_with_train("val")

    print(f"  stems with >=2 frames            : {len(multi)}")
    print(f"  of those, spanning >1 split      : {len(multi_spanning)} "
          f"({100*len(multi_spanning)/max(1,len(multi)):.1f}%)")
    print(f"\n  TEST patches sharing a stem with train: "
          f"{test_hit}/{test_n} = {100*test_share:.2f}%   <- headline")
    print(f"  VAL  patches sharing a stem with train: "
          f"{val_hit}/{val_n} = {100*val_share:.2f}%")

    per_dataset = {}
    print(f"\n  {'dataset':<16}{'test n':>9}{'test share':>13}"
          f"{'val n':>9}{'val share':>12}")
    for ds in sorted(patches_df.source_dataset.unique()):
        d = patches_df[patches_df.source_dataset == ds]
        dt, dv = d[d.split == "test"], d[d.split == "val"]
        ts = float(dt.stem.isin(train_stems).mean()) if len(dt) else 0.0
        vs = float(dv.stem.isin(train_stems).mean()) if len(dv) else 0.0
        per_dataset[ds] = {
            "test_patches": int(len(dt)), "test_share_with_train": ts,
            "val_patches": int(len(dv)), "val_share_with_train": vs,
        }
        print(f"  {ds:<16}{len(dt):>9}{100*ts:>12.2f}%"
              f"{len(dv):>9}{100*vs:>11.2f}%")

    return {
        "stems_with_multiple_frames": len(multi),
        "stems_spanning_multiple_splits": len(multi_spanning),
        "fraction_of_multi_stems_spanning": (
            len(multi_spanning) / len(multi) if multi else 0.0),
        "test_patches_sharing_stem_with_train": {
            "fraction": test_share, "n": test_hit, "of": test_n},
        "val_patches_sharing_stem_with_train": {
            "fraction": val_share, "n": val_hit, "of": val_n},
        "per_dataset": per_dataset,
    }


def f4_grouping_headroom(patches_df, frames_df):
    """
    Verify the perceptual union-find and the split assignment agree.

    dedup_report.json records only counts, not group membership, so the check
    is done through source_frame instead -- which is exactly the union-find's
    output. convert_and_remap.py set source_frame to each group's REPRESENTATIVE
    and dropped the other members from the corpus entirely. So if every
    surviving image is its own representative, no two surviving images can
    belong to one perceptual group, and the cross-split sibling count is zero
    by construction.

    A row where source_image != source_frame means a non-representative
    survived, which would be a genuine union-find / manifest inconsistency.
    """
    print("\n" + "=" * 78)
    print("F4 -- grouping-headroom / guard consistency")
    print("=" * 78)

    non_rep = patches_df[patches_df.source_image != patches_df.source_frame]
    reps = frames_df.source_frame.value_counts()
    collisions = reps[reps > 1]

    print(f"  patches whose source_image != source_frame : {len(non_rep)}")
    print(f"  source_frame values shared by >1 image     : {len(collisions)}")

    if len(non_rep) == 0 and len(collisions) == 0:
        frac = 0.0
        print("  => every surviving image is its own group representative, so "
              "no\n     perceptual group has two members in the corpus. "
              "Cross-split siblings: 0.")
    else:
        frac = float(len(non_rep) / len(patches_df))
        print(f"  => INCONSISTENCY: {len(non_rep)} patches "
              f"({100*frac:.2f}%) descend from a non-representative image.")

    return {
        "method": "source_frame is the union-find representative; "
                  "dedup_report.json stores counts only, not membership",
        "patches_from_non_representative_images": int(len(non_rep)),
        "source_frames_shared_by_multiple_images": int(len(collisions)),
        "fraction_test_frames_with_perceptual_sibling_in_train": frac,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/diagnostics")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "vision_patches" / "manifest.csv")
    if "split" not in manifest.columns:
        sys.exit("manifest has no split column; run merge_dedup_split.py first")

    stems = manifest.source_image.map(lambda p: sequence_stem(p)[0])
    idxs = manifest.source_image.map(lambda p: sequence_stem(p)[1])
    manifest["stem"] = stems
    manifest["seq_index"] = idxs

    frames_df = manifest.drop_duplicates("source_frame")[
        ["source_frame", "source_image", "source_dataset", "split",
         "stem", "seq_index"]].reset_index(drop=True)

    print("=" * 78)
    print("SEQUENCE-LEAKAGE AUDIT (F1, F2, F4)")
    print("=" * 78)
    print(f"patches {len(manifest)}, source frames {len(frames_df)}")
    print(f"splits: {dict(manifest.split.value_counts())}")

    f1 = f1_sequence_detection(frames_df)
    f2 = f2_cross_split(frames_df, manifest)
    f4 = f4_grouping_headroom(manifest, frames_df)

    # ---- verdicts for the two thresholds this script can decide ----
    seq_share = f2["test_patches_sharing_stem_with_train"]["fraction"]
    guard_frac = f4["fraction_test_frames_with_perceptual_sibling_in_train"]
    verdicts = {
        "F2_sequence_leakage": {
            "measured": seq_share, "threshold": T_SEQUENCE_SHARE,
            "verdict": "CONFIRMED" if seq_share >= T_SEQUENCE_SHARE else "CLEAN",
        },
        "F4_guard_consistency": {
            "measured": guard_frac, "threshold": T_GUARD_INCONSISTENCY,
            "verdict": ("BUG" if guard_frac > T_GUARD_INCONSISTENCY
                        else "CONSISTENT"),
        },
    }

    print("\n" + "=" * 78)
    print("VERDICTS (F2, F4)")
    print("=" * 78)
    print(f"  F2 test patches sharing a sequence stem with train: "
          f"{100*seq_share:.2f}%  (threshold {100*T_SEQUENCE_SHARE:.0f}%)  "
          f"-> {verdicts['F2_sequence_leakage']['verdict']}")
    print(f"  F4 test frames with a perceptual sibling in train : "
          f"{100*guard_frac:.2f}%  (threshold 0%)  "
          f"-> {verdicts['F4_guard_consistency']['verdict']}")
    print("\n  F3 (embedding nearest neighbours) requires the Task C "
          "checkpoint; run nn_audit.py.")

    payload = {"f1_sequence_detection": f1, "f2_cross_split": f2,
               "f4_grouping_headroom": f4, "verdicts": verdicts,
               "thresholds": {"F2": T_SEQUENCE_SHARE,
                              "F4": T_GUARD_INCONSISTENCY}}
    with open(out_dir / "sequence_stems.json", "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\n-> {(out_dir/'sequence_stems.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
