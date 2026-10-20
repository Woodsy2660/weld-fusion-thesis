"""
Post-training report for one public-corpus holdout run.

Reads results/vision/logits_test_{tag}.parquet and joins source_sequence from
the manifest, then reports:

  * per-head precision / recall / F1 / support, with 95% CIs from a CLUSTER
    bootstrap over source_sequence -- sequences are resampled, never patches,
    because patches within a sequence are not independent;
  * confusion matrix over the three focal elements;
  * expected calibration error and a reliability diagram on the RAW softmax.
    No temperature scaling exists on this path yet, so these are uncalibrated
    figures.

Usage:
    python src/vision/diagnostics/holdout_report.py --tag resnet50_ft-layer3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402
from src.vision.loso import ECE_BINS, expected_calibration_error  # noqa: E402

N_BOOT = 2000


def prf(true, pred, heads):
    out = {}
    for h in heads:
        tp = np.sum((pred == h) & (true == h))
        fp = np.sum((pred == h) & (true != h))
        fn = np.sum((pred != h) & (true == h))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[h] = (p, r, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frame = load_frame()
    heads = frame.head_names
    res = REPO_ROOT / "results" / "vision"
    df = pd.read_parquet(res / f"logits_test_{args.tag}.parquet")
    man = pd.read_csv(REPO_ROOT / "data/processed/vision_patches/manifest.csv",
                      usecols=["patch_path", "source_sequence"])
    df = df.merge(man, on="patch_path", how="left", validate="one_to_one")
    if df.source_sequence.isna().any():
        sys.exit("ERROR: test patches missing source_sequence in manifest")

    true = df.head_true.to_numpy()
    pred = df.head_pred.to_numpy()
    point = prf(true, pred, heads)

    # Cluster bootstrap over sequences.
    rng = np.random.default_rng(args.seed)
    groups = df.groupby("source_sequence").indices
    keys = list(groups)
    boot = {h: [] for h in heads}
    boot_macro = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        s = prf(true[idx], pred[idx], heads)
        for h in heads:
            boot[h].append(s[h][2])
        boot_macro.append(np.mean([s[h][2] for h in heads]))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    per_head = {}
    for h in heads:
        per_head[h] = {
            "precision": float(point[h][0]), "recall": float(point[h][1]),
            "f1": float(point[h][2]), "f1_ci95_sequence_bootstrap": ci(boot[h]),
            "support_patches": int((true == h).sum()),
            "support_sequences": int(df.loc[true == h, "source_sequence"].nunique()),
        }
    macro = float(np.mean([point[h][2] for h in heads]))

    cm = pd.crosstab(pd.Categorical(true, heads), pd.Categorical(pred, heads),
                     rownames=["true"], colnames=["pred"], dropna=False)

    logits = df[[f"logit_{h}" for h in heads]].to_numpy()
    labels = np.array([heads.index(h) for h in true])
    ece, bins = expected_calibration_error(logits, labels)

    print(f"TEST -- {args.tag}   ({len(df)} patches, {len(keys)} sequences)")
    print(f"{'head':<6}{'P':>8}{'R':>8}{'F1':>8}   {'F1 95% CI (seq boot)':<22}"
          f"{'n patch':>8}{'n seq':>7}")
    for h, r in per_head.items():
        lo, hi = r["f1_ci95_sequence_bootstrap"]
        print(f"{h:<6}{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}   "
              f"[{lo:.4f}, {hi:.4f}]      {r['support_patches']:>8}"
              f"{r['support_sequences']:>7}")
    lo, hi = ci(boot_macro)
    print(f"macro-F1 {macro:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print("\nconfusion (rows true, cols pred):")
    print(cm.to_string())
    print(f"\nECE (uncalibrated, {ECE_BINS} bins): {ece:.4f}")

    # Reliability diagram.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    conf, ok = p.max(1), p.argmax(1) == labels
    edges = np.linspace(0, 1, ECE_BINS + 1)
    mids, accs, counts = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            mids.append(conf[m].mean()); accs.append(ok[m].mean()); counts.append(m.sum())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.plot(mids, accs, "o-")
    ax.set_xlabel("mean max-softmax confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"{args.tag} test, uncalibrated\nECE {ece:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    fig.tight_layout()
    png = res / f"reliability_test_{args.tag}.png"
    fig.savefig(png, dpi=150)

    out = res / f"holdout_report_{args.tag}.json"
    out.write_text(json.dumps({
        "tag": args.tag, "n_patches": len(df), "n_sequences": len(keys),
        "per_head": per_head, "macro_f1": macro,
        "macro_f1_ci95_sequence_bootstrap": ci(boot_macro),
        "confusion_matrix": cm.to_numpy().tolist(), "heads": heads,
        "ece_uncalibrated": float(ece), "ece_bins": ECE_BINS,
        "reliability": {"conf": [float(x) for x in mids],
                        "acc": [float(x) for x in accs],
                        "count": [int(x) for x in counts]},
        "n_boot": N_BOOT, "temperature_scaling": "not implemented on this path",
    }, indent=1))
    print(f"\n-> {out.relative_to(REPO_ROOT)}\n-> {png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
