"""
Task I extras -- two figures that currently exist only outside the repo.

1. V3 F1 restricted to final_welding patches only.

   weld_dataset/discontinuity is the single largest contributor to V3. If V3's
   headline F1 were resting on it, the discontinuity -> V3 mapping would be
   propping up the head rather than belonging in it. final_welding contains no
   discontinuity patches at all, so V3's F1 computed over final_welding alone
   is the clean test: a healthy number there means the head stands on its own.

2. A geometry-only logistic probe predicting source_dataset.

   If crop geometry alone identifies which dataset a patch came from, then any
   clustering by source in feature space is dataset identity -- camera,
   lighting, working distance, surface finish -- rather than defect identity.
   Reported against the majority-class baseline, because a probe that beats
   1/K but not the majority class has learned nothing.

Both are cheap, both belong in the thesis, and neither was reproducible from
repo artefacts before now.

Usage:
    python src/vision/diagnostics/mapping_support.py --tag resnet50_v2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402

SEED = 42
GEOM_FEATURES = ["pad_fraction", "log_aspect_ratio", "log_box_w", "log_box_h",
                 "log_box_area"]


def add_log_features(df):
    df = df.copy()
    for col in ("aspect_ratio", "box_w", "box_h", "box_area"):
        df[f"log_{col}"] = np.log10(np.maximum(df[col].astype(float), 1e-6))
    return df


def v3_within_source(test, frame, source="final_welding"):
    """
    Per-head metrics computed on one source dataset only.

    Restricting BOTH the true and predicted sides to patches whose ground truth
    comes from `source`; predictions may still fall on any head, so precision
    stays meaningful.
    """
    sub = test[test.source_dataset == source]
    if sub.empty:
        return None

    heads = frame.head_names
    p, r, f, s = precision_recall_fscore_support(
        sub.head_true, sub.head_pred, labels=heads, zero_division=0)
    out = {
        "source": source,
        "n_patches": int(len(sub)),
        "per_head": {h: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f[i]), "support": int(s[i])}
                     for i, h in enumerate(heads)},
        "macro_f1_three_head": float(f1_score(
            sub.head_true, sub.head_pred, labels=heads, average="macro",
            zero_division=0)),
        "accuracy": float(accuracy_score(sub.head_true, sub.head_pred)),
        "source_labels_present": sorted(sub.source_label.unique().tolist()),
    }
    return out


def source_dataset_probe(corpus):
    """
    Multinomial logistic regression: crop geometry -> source_dataset.

    Trained and evaluated on the same grouped splits the model uses, so the
    numbers sit on the same footing.
    """
    df = add_log_features(corpus)
    train = df[df.split == "train"]
    test = df[df.split == "test"]

    scaler = StandardScaler().fit(train[GEOM_FEATURES])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=SEED
                             ).fit(scaler.transform(train[GEOM_FEATURES]),
                                   train.source_dataset)
    pred = clf.predict(scaler.transform(test[GEOM_FEATURES]))

    acc = float(accuracy_score(test.source_dataset, pred))
    majority = test.source_dataset.value_counts(normalize=True)
    baseline = float(majority.iloc[0])
    labels = sorted(test.source_dataset.unique())
    p, r, f, s = precision_recall_fscore_support(
        test.source_dataset, pred, labels=labels, zero_division=0)

    return {
        "features": GEOM_FEATURES,
        "accuracy": acc,
        "majority_class_baseline": baseline,
        "majority_class": str(majority.index[0]),
        "lift_over_baseline": acc - baseline,
        "macro_f1": float(f1_score(test.source_dataset, pred,
                                   average="macro", zero_division=0)),
        "per_source": {labels[i]: {"precision": float(p[i]),
                                   "recall": float(r[i]), "f1": float(f[i]),
                                   "support": int(s[i])}
                       for i in range(len(labels))},
        "median_box_area_px2": {
            str(k): float(v) for k, v in
            corpus.groupby("source_dataset").box_area.median().items()},
        "n_train": int(len(train)), "n_test": int(len(test)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="resnet50_v2")
    ap.add_argument("--out", default="results/vision/mapping_support.json")
    args = ap.parse_args()

    frame = load_frame()
    logits = REPO_ROOT / "results" / "vision" / f"logits_test_{args.tag}.parquet"
    if not logits.exists():
        sys.exit(f"ERROR: {logits} not found")
    test = pd.read_parquet(logits)

    manifest = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "vision_patches" / "manifest.csv")
    manifest["box_area"] = manifest.box_w * manifest.box_h
    manifest["aspect_ratio"] = (
        np.maximum(manifest.box_w, manifest.box_h)
        / np.maximum(np.minimum(manifest.box_w, manifest.box_h), 1e-9))

    print("=" * 78)
    print(f"TASK I EXTRAS -- {args.tag}")
    print("=" * 78)

    # ---- 1. V3 within final_welding ----
    print("\n[1] V3 restricted to final_welding (contains NO discontinuity)")
    overall = {h: None for h in frame.head_names}
    p, r, f, s = precision_recall_fscore_support(
        test.head_true, test.head_pred, labels=frame.head_names,
        zero_division=0)
    for i, h in enumerate(frame.head_names):
        overall[h] = {"precision": float(p[i]), "recall": float(r[i]),
                      "f1": float(f[i]), "support": int(s[i])}

    per_source = {}
    for src in sorted(test.source_dataset.unique()):
        res = v3_within_source(test, frame, src)
        if res:
            per_source[src] = res

    print(f"  {'scope':<18}{'V1 F1':>9}{'V3 F1':>9}{'V4 F1':>9}"
          f"{'macro':>9}{'n':>8}")
    print(f"  {'all sources':<18}"
          + "".join(f"{overall[h]['f1']:>9.4f}" for h in frame.head_names)
          + f"{float(np.mean([overall[h]['f1'] for h in frame.head_names])):>9.4f}"
          + f"{len(test):>8}")
    for src, res in per_source.items():
        print(f"  {src:<18}"
              + "".join(f"{res['per_head'][h]['f1']:>9.4f}"
                        for h in frame.head_names)
              + f"{res['macro_f1_three_head']:>9.4f}{res['n_patches']:>8}")

    fw = per_source.get("final_welding")
    if fw:
        has_disc = "discontinuity" in fw["source_labels_present"]
        print(f"\n  final_welding V3 F1 = {fw['per_head']['V3']['f1']:.4f} "
              f"on {fw['per_head']['V3']['support']} patches")
        print(f"  discontinuity present in final_welding? {has_disc}  "
              f"(expected False)")
        print(f"  => V3 is {'NOT ' if not has_disc else ''}resting on the "
              f"discontinuity mapping.")

    # ---- 2. geometry -> source_dataset probe ----
    print("\n[2] geometry-only probe predicting source_dataset")
    probe = source_dataset_probe(manifest)
    print(f"  accuracy {probe['accuracy']:.4f} against majority baseline "
          f"{probe['majority_class_baseline']:.4f} "
          f"({probe['majority_class']})  lift {probe['lift_over_baseline']:+.4f}")
    print(f"  macro-F1 {probe['macro_f1']:.4f}")
    for src, m in probe["per_source"].items():
        print(f"    {src:<16} P {m['precision']:.4f}  R {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  n={m['support']}")
    print(f"  median box area by source (px^2): "
          f"{ {k: round(v) for k, v in probe['median_box_area_px2'].items()} }")
    print(f"  => crop geometry alone identifies the source dataset at "
          f"{100*probe['accuracy']:.1f}%.")
    print(f"     Clustering by source in feature space is dataset identity "
          f"(camera, lighting,\n     working distance), not defect identity.")

    payload = {
        "tag": args.tag, "seed": SEED,
        "per_head_all_sources": overall,
        "per_head_by_source": per_source,
        "geometry_source_probe": probe,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\n-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
