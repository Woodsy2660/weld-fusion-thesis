"""
Model-free search over grouped train/val/test partitions of the public corpus.

The 8/1/1 assignment from merge_dedup_split.py is one draw. On final_welding,
grouped on source_sequence, it gave 90.6/5.0/4.4 because the dominant session
(the IMG_36xx tiles, ~39% of patches) landed in train with too few groups left
to fill val and test. This script draws many grouped partitions, scores each
WITHOUT any model, and picks one.

This is still a single grouped HOLDOUT, not cross-validation. The public corpus
is initialisation for the real-coupon model; cross-validation in this project
means only the real-data k=4 coupon-grouped folds.

Each candidate:
  * groups on source_sequence -- a group is never split;
  * pins the dominant group (largest by patches) to train;
  * fills test and val from the remaining groups in random order to the target
    share of patches, everything else to train.

Scored on:
  * val and test patch share, target TARGET +- SHARE_TOL      (hard)
  * distinct sequences per head per split; V1 in test >= V1_TEST_FLOOR (hard)
  * Jensen-Shannon divergence over the defect-type mix MIX_LABELS, train vs
    test AND train vs val; the WORSE of the two is minimised      (minimax)

The defect-type residual is recorded, not driven to zero: a test set whose mix
differs from train is a reasonable proxy for the shift the model will face on
real coupons. Selection is minimax over train-test and train-val because
minimising train-test alone (as first specified) pushed the skew into val --
the best train-test candidate had a val set 47% Irregular, JSD 0.204 -- and
val drives checkpoint selection. The full candidate distribution is written so
the choice can be seen in context.

Usage:
    python src/vision/partition_search.py --sources final_welding          # score only
    python src/vision/partition_search.py --sources final_welding --apply  # write split
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402

GROUP_KEY = "source_sequence"
TARGET = 0.10
SHARE_TOL = 0.015
V1_TEST_FLOOR = 25
MIX_LABELS = ["Overlap", "Irregular", "Burn Through", "Spatters"]
DRIFT_SHARE_MIN = 0.02     # same threshold merge_dedup_split.py applies


def jsd(p, q):
    """Jensen-Shannon divergence, base 2, in [0, 1]."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def scorecard(df, assign, heads):
    """Score one {group: split} assignment over the rows of df."""
    split = df[GROUP_KEY].map(assign)
    n = len(df)
    card = {}
    for s in ("train", "val", "test"):
        card[f"{s}_share"] = float((split == s).sum() / n)
    seqs = df.assign(_s=split).groupby(["head", "_s"])[GROUP_KEY].nunique()
    card["sequences"] = {h: {s: int(seqs.get((h, s), 0))
                             for s in ("train", "val", "test")} for h in heads}
    card["min_sequences_per_head_per_split"] = int(min(
        v for h in heads for v in card["sequences"][h].values()))

    def mix(s):
        c = df.loc[split == s, "source_label"].value_counts()
        return [int(c.get(l, 0)) for l in MIX_LABELS]
    card["mix_counts"] = {s: dict(zip(MIX_LABELS, mix(s)))
                          for s in ("train", "val", "test")}
    card["jsd_train_test"] = jsd(mix("train"), mix("test"))
    card["jsd_train_val"] = jsd(mix("train"), mix("val"))

    # Per-head max drift (pp) of test label share against corpus, the same
    # statistic merge_dedup_split.py reports, for comparability.
    drift = {}
    for h in heads:
        sub = df[df["head"] == h]
        corpus = sub.source_label.value_counts(normalize=True)
        test = sub.loc[split[sub.index] == "test", "source_label"].value_counts(
            normalize=True)
        d = [abs(test.get(l, 0.0) - corpus[l]) * 100
             for l in corpus.index if corpus[l] >= DRIFT_SHARE_MIN]
        drift[h] = round(max(d), 1) if d else 0.0
    card["max_test_drift_pp"] = drift

    card["feasible"] = bool(
        abs(card["val_share"] - TARGET) <= SHARE_TOL
        and abs(card["test_share"] - TARGET) <= SHARE_TOL
        and card["sequences"]["V1"]["test"] >= V1_TEST_FLOOR)
    return card


def draw(sizes, pinned, rng):
    """One random grouped partition filling test then val to TARGET share."""
    total = sizes.sum()
    order = [g for g in rng.permutation(sizes.index) if g != pinned]
    assign = {pinned: "train"}
    filled = {"test": 0, "val": 0}
    hi = (TARGET + SHARE_TOL) * total
    for g in order:
        n = sizes[g]
        for s in ("test", "val"):
            if filled[s] < TARGET * total and filled[s] + n <= hi:
                assign[g] = s
                filled[s] += n
                break
        else:
            assign[g] = "train"
    return assign


def check_leakage(m):
    """Set intersection on actual split assignments, per provenance key."""
    out = {}
    s = m[m.split.isin(["train", "val", "test"])]
    for key in ("source_sequence", "source_frame", "source_image"):
        sets = {k: set(g[key]) for k, g in s.groupby("split")}
        out[key] = {f"{a}&{b}": len(sets.get(a, set()) & sets.get(b, set()))
                    for a, b in (("train", "val"), ("train", "test"),
                                 ("val", "test"))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches-dir", default="data/processed/vision_patches")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--candidates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true",
                    help="write the selected partition into manifest.csv")
    ap.add_argument("--out", default="results/vision/partition_search.json")
    args = ap.parse_args()

    frame = load_frame()
    heads = frame.head_names
    patches = REPO_ROOT / args.patches_dir
    manifest_path = patches / "manifest.csv"
    m = pd.read_csv(manifest_path)
    if GROUP_KEY not in m.columns:
        sys.exit(f"ERROR: manifest has no {GROUP_KEY}; run "
                 f"src/vision/diagnostics/build_sequence_groups.py first")

    keep = m.source_dataset.isin(args.sources)
    df = m[keep]
    sizes = df.groupby(GROUP_KEY).size()
    pinned = sizes.idxmax()
    print(f"{len(df)} patches, {len(sizes)} {GROUP_KEY} groups from {args.sources}")
    print(f"pinned to train: {Path(str(pinned)).name[:60]}  "
          f"({sizes[pinned]} patches, {sizes[pinned] / len(df):.1%})")

    current = None
    if "split" in m.columns and df.split.isin(["train", "val", "test"]).all():
        cur_assign = df.groupby(GROUP_KEY).split.first().to_dict()
        current = scorecard(df, cur_assign, heads)

    rng = np.random.default_rng(args.seed)
    cards, assigns = [], []
    for i in range(args.candidates):
        a = draw(sizes, pinned, rng)
        c = scorecard(df, a, heads)
        c["candidate"] = i
        cards.append(c)
        assigns.append(a)

    feasible = [c for c in cards if c["feasible"]]
    print(f"\n{len(cards)} candidates, {len(feasible)} hard-feasible "
          f"(val/test share {TARGET:.0%}+-{SHARE_TOL * 100:.1f}pp, "
          f"V1 test sequences >= {V1_TEST_FLOOR})")
    if not feasible:
        sys.exit("No feasible partition. Reported, not relaxed.")
    def worst(c):
        return max(c["jsd_train_test"], c["jsd_train_val"])
    best = min(feasible, key=worst)
    j = np.array([worst(c) for c in feasible])
    print(f"max(train-test, train-val) JSD over feasible: min {j.min():.4f}  "
          f"median {np.median(j):.4f}  max {j.max():.4f}")

    def show(name, c):
        print(f"\n{name}")
        print(f"  shares train/val/test : {c['train_share']:.1%} / "
              f"{c['val_share']:.1%} / {c['test_share']:.1%}")
        for h in heads:
            print(f"  {h} sequences t/v/t   : {c['sequences'][h]}")
        print(f"  JSD train-test {c['jsd_train_test']:.4f}   "
              f"train-val {c['jsd_train_val']:.4f}")
        print(f"  max test drift (pp)   : {c['max_test_drift_pp']}")
        for s in ("train", "val", "test"):
            tot = sum(c["mix_counts"][s].values()) or 1
            print(f"  mix {s:<5}: " + "  ".join(
                f"{l} {100 * v / tot:.1f}%" for l, v in c["mix_counts"][s].items()))
    if current:
        show("CURRENT partition (merge_dedup_split.py, seed 42)", current)
    show(f"SELECTED partition (candidate {best['candidate']})", best)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "group_key": GROUP_KEY, "sources": args.sources, "seed": args.seed,
        "n_candidates": len(cards), "n_feasible": len(feasible),
        "pinned_group": str(pinned), "pinned_patches": int(sizes[pinned]),
        "target_share": TARGET, "share_tol": SHARE_TOL,
        "v1_test_floor": V1_TEST_FLOOR, "mix_labels": MIX_LABELS,
        "selection": "minimax over jsd_train_test, jsd_train_val",
        "feasible_worst_jsd_quantiles": {q: float(np.quantile(j, q))
                                   for q in (0, 0.25, 0.5, 0.75, 1)},
        "current": current, "selected": best,
    }

    if args.apply:
        a = assigns[best["candidate"]]
        backup = patches / "manifest.pre_partition_search.csv"
        shutil.copy(manifest_path, backup)
        m.loc[keep, "split"] = m.loc[keep, GROUP_KEY].map(a)
        m.loc[~keep, "split"] = "excluded"
        leaks = check_leakage(m)
        report["leakage_intersections"] = leaks
        print(f"\nLeakage (set intersection on assignments): {leaks}")
        if any(v for d in leaks.values() for v in d.values()):
            sys.exit("FATAL: a provenance key spans splits; manifest NOT written")
        m.to_csv(manifest_path, index=False)

        # Keep split_summary.json truthful: train.py copies it into the run
        # manifest.
        summ_path = patches / "split_summary.json"
        summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
        s = m[m.split.isin(["train", "val", "test"])]
        per = s.groupby(["split", "head"]).size()
        tr = {h: int(per.get(("train", h), 0)) for h in heads}
        w = {h: len(s[s.split == "train"]) / (len(heads) * tr[h]) for h in heads}
        mean_w = sum(w.values()) / len(w)
        summ.update({
            "approach": f"partition_search(candidate {best['candidate']} of "
                        f"{len(cards)}, groups={GROUP_KEY}, dominant group "
                        f"pinned to train)",
            "group_key": GROUP_KEY,
            "n_groups": int(len(sizes)),
            "sources": args.sources,
            "patches_per_head": {h: int(s[s["head"] == h].shape[0]) for h in heads},
            "patches_per_split": {k: int(v) for k, v in s.split.value_counts().items()},
            "frames_per_split": {k: int(v) for k, v in
                                 s.groupby("split").source_frame.nunique().items()},
            "class_weights": {h: round(w[h] / mean_w, 4) for h in heads},
            "imbalance_ratio": round(max(tr.values()) / min(tr.values()), 2),
            "acceptance": {"max_test_drift_pp": best["max_test_drift_pp"],
                           "note": "defect-type residual recorded, not "
                                   "optimised to zero"},
            "partition_search": args.out,
        })
        summ_path.write_text(json.dumps(summ, indent=1))
        print(f"manifest written (backup: {backup.relative_to(REPO_ROOT)})")

    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"report : {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
