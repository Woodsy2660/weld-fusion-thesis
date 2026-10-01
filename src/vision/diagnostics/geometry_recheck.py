"""
Task E.2 -- joint geometry-matched re-check after the retrain.

The pad-leakage diagnostic cleared the crop-geometry confound on the fold-0
split. The split has since changed, so the check is repeated as an acceptance
gate: V3 and V4 matched-minus-unmatched F1 deltas must both stay under 0.05.

Matching is JOINT -- box-area decile crossed with pad-fraction tercile -- not
pad alone. Box area carried more mutual information with the head than
pad_fraction did (0.26 bits against 0.07 on the fold-0 test split), so matching
on padding alone would leave the larger confound in place.

Exit status is non-zero if the gate fails, so a caller can stop before LOSO.

Usage:
    python src/vision/diagnostics/geometry_recheck.py --arch resnet50
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402

SEED = 42
N_AREA_DECILES = 10
N_PAD_TERCILES = 3
N_BOOTSTRAP = 1000
GATE = 0.05


def per_head(df, heads):
    p, r, f, s = precision_recall_fscore_support(
        df.head_true, df.head_pred, labels=heads, zero_division=0)
    return {h: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])}
            for i, h in enumerate(heads)}


def bootstrap_ci(df, head, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    yt, yp = df.head_true.to_numpy(), df.head_pred.to_numpy()
    vals = [f1_score(yt[i], yp[i], labels=[head], average="macro",
                     zero_division=0)
            for i in (rng.integers(0, len(df), len(df)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def joint_match(test, a="V3", b="V4", seed=SEED):
    """
    Equalise the joint (box-area decile x pad tercile) distribution of two
    heads by downsampling each cell to the smaller of the two.
    """
    rng = np.random.default_rng(seed)
    d = test.copy()
    d["area_bin"] = pd.qcut(d.box_area, N_AREA_DECILES, labels=False,
                            duplicates="drop")
    d["pad_bin"] = pd.qcut(d.pad_fraction, N_PAD_TERCILES, labels=False,
                           duplicates="drop")

    keep, cells = [], []
    for (ab, pb), sub in d.groupby(["area_bin", "pad_bin"]):
        xa, xb = sub[sub.head_true == a], sub[sub.head_true == b]
        n = min(len(xa), len(xb))
        cells.append({"area_bin": int(ab), "pad_bin": int(pb),
                      f"{a}_available": len(xa), f"{b}_available": len(xb),
                      "kept_each": int(n)})
        if n:
            keep.append(xa.iloc[rng.choice(len(xa), n, replace=False)])
            keep.append(xb.iloc[rng.choice(len(xb), n, replace=False)])
        keep.append(sub[~sub.head_true.isin([a, b])])
    return pd.concat(keep, ignore_index=True), cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--out", default="results/vision/geometry_recheck.json")
    args = ap.parse_args()

    frame = load_frame()
    heads = frame.head_names
    path = REPO_ROOT / "results" / "vision" / f"logits_test_{args.arch}.parquet"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Train {args.arch} first.")
    test = pd.read_parquet(path)

    print("=" * 78)
    print(f"GEOMETRY RE-CHECK -- {args.arch}")
    print("=" * 78)
    print(f"Frame: {frame}")
    print(f"test patches: {len(test)}")
    print(f"matching: box-area decile x pad tercile "
          f"({N_AREA_DECILES}x{N_PAD_TERCILES} cells), V3 vs V4")

    matched, cells = joint_match(test, "V3", "V4")
    unm, mat = per_head(test, heads), per_head(matched, heads)

    u3 = f1_score(test.head_true, test.head_pred, labels=heads,
                  average="macro", zero_division=0)
    m3 = f1_score(matched.head_true, matched.head_pred, labels=heads,
                  average="macro", zero_division=0)

    print(f"\nmatched subsample: {len(matched)} of {len(test)} patches")
    for h in heads:
        print(f"  {h}: {(matched.head_true == h).sum():>5} "
              f"(was {(test.head_true == h).sum():>5})")
    for h in ("V3", "V4"):
        mb, ub = matched[matched.head_true == h], test[test.head_true == h]
        print(f"  {h} mean pad {ub.pad_fraction.mean():.4f} -> "
              f"{mb.pad_fraction.mean():.4f}   "
              f"median area {ub.box_area.median():.0f} -> "
              f"{mb.box_area.median():.0f}")

    print(f"\n{'head':<6}{'unmatched':>11}{'95% CI':>20}{'matched':>10}"
          f"{'95% CI':>20}{'delta':>9}")
    out_heads, gate_fail = {}, []
    for h in heads:
        uci, mci = bootstrap_ci(test, h), bootstrap_ci(matched, h)
        uf, mf = unm[h]["f1"], mat[h]["f1"]
        delta = mf - uf
        out_heads[h] = {
            "unmatched": {**unm[h], "f1_ci95": list(uci)},
            "matched": {**mat[h], "f1_ci95": list(mci)},
            "f1_delta": float(delta),
        }
        if h in ("V3", "V4") and -delta >= GATE:
            gate_fail.append(h)
        print(f"{h:<6}{uf:>11.4f}{f'[{uci[0]:.4f}, {uci[1]:.4f}]':>20}"
              f"{mf:>10.4f}{f'[{mci[0]:.4f}, {mci[1]:.4f}]':>20}{delta:>+9.4f}")

    print(f"\nthree-head macro-F1: unmatched {u3:.4f} -> matched {m3:.4f} "
          f"({m3-u3:+.4f})")

    passed = not gate_fail
    print(f"\nGATE: V3 and V4 F1 drop < {GATE}  ->  "
          f"{'PASS' if passed else 'FAIL ' + ','.join(gate_fail)}")

    payload = {
        "architecture": args.arch, "seed": SEED, "gate": GATE,
        "matching": f"box-area decile x pad tercile "
                    f"({N_AREA_DECILES}x{N_PAD_TERCILES})",
        "matched_size": int(len(matched)), "unmatched_size": int(len(test)),
        "cells": cells,
        "per_head": out_heads,
        "macro_f1_three_head": {"unmatched": float(u3), "matched": float(m3),
                                "delta": float(m3 - u3)},
        "gate_passed": passed,
        "gate_failures": gate_fail,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"-> {out.relative_to(REPO_ROOT)}")

    if not passed:
        sys.exit(f"geometry gate failed for {gate_fail}; not proceeding to LOSO")


if __name__ == "__main__":
    main()
