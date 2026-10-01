"""
Steps 1-4 of the pad_fraction leakage diagnostic.

Question: can the network read the focal element off crop geometry instead of
off the weld? V3 source classes (Overlap, Lack of Fusion, Irregular, Burn
Through, Undercut) are elongated toe-line boxes; V4 source classes (stain,
spatter, deposit, negatives) are compact blobs. Letterboxing turns an elongated
box into a heavily padded square, so pad_fraction may encode the label.

  Step 1  descriptive association    -> pad_fraction_stats.json + 2 plots
  Step 2  geometry-only probe        -> geometry_probe_metrics.json,
                                        permutation_null.json
  Step 3  accuracy vs pad decile     -> f1_by_pad_decile.png
  Step 4  pad-matched subsample      -> matched_subsample_metrics.json

Every macro metric is reported twice: over four heads, and over the three
supported heads (V1, V3, V4). V2 has 6 test patches and its F1 is noise, so a
four-head macro alone is never quoted.

Usage:
    python src/vision/diagnostics/pad_leakage.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, f1_score, precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402

SEED = 42
N_DECILES = 10
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
SUPPORTED = ["V1", "V3", "V4"]        # V2 has 6 test patches; excluded from the
                                       # three-head macro by the report convention

# Colour per head, consistent across all three figures.
HEAD_COLOUR = {"V1": "#2C5F8A", "V2": "#A32E28", "V3": "#B8860B", "V4": "#2E6B4F"}


# ---------------------------------------------------------------------------
# Association measures
# ---------------------------------------------------------------------------
def cramers_v(a, b):
    """
    Cramer's V between two categoricals, bias-corrected (Bergsma 2013).

    Uncorrected V is upward-biased for sparse tables, and a 4x10 table with a
    6-member class is sparse, so the correction matters here.
    """
    from scipy.stats import chi2_contingency
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    v_raw = np.sqrt(phi2 / max(1, min(r - 1, k - 1)))

    phi2_c = max(0.0, phi2 - (r - 1) * (k - 1) / max(1, n - 1))
    r_c = r - (r - 1) ** 2 / max(1, n - 1)
    k_c = k - (k - 1) ** 2 / max(1, n - 1)
    denom = max(1e-12, min(r_c - 1, k_c - 1))
    return float(v_raw), float(np.sqrt(max(0.0, phi2_c) / denom))


def mutual_information_bits(labels, values, n_bins=N_DECILES):
    """
    I(head; continuous feature) in bits, the feature discretised into
    equal-frequency bins. Binning is required for a plug-in estimate and the
    bin count is reported alongside, because MI is not binning-invariant.
    """
    from sklearn.metrics import mutual_info_score
    binned = pd.qcut(values, n_bins, labels=False, duplicates="drop")
    return float(mutual_info_score(labels, binned) / np.log(2))


def entropy_bits(labels):
    p = pd.Series(labels).value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


def describe(series):
    s = pd.Series(series).astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "sd": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(s.min()), "max": float(s.max()),
        "frac_above_0.5": float((s > 0.5).mean()),
    }


def macro_pair(y_true, y_pred, heads):
    """(four-head macro-F1, three-head macro-F1). Never report the first alone."""
    four = f1_score(y_true, y_pred, labels=heads, average="macro",
                    zero_division=0)
    three = f1_score(y_true, y_pred, labels=SUPPORTED, average="macro",
                     zero_division=0)
    return float(four), float(three)


def per_head_table(y_true, y_pred, heads):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=heads, zero_division=0)
    return {h: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])}
            for i, h in enumerate(heads)}


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------
def step1(test, corpus, heads, out_dir):
    print("\n" + "=" * 78)
    print("STEP 1 -- descriptive association")
    print("=" * 78)

    stats = {"seed": SEED, "n_deciles": N_DECILES, "scopes": {}}

    for name, df in (("test_split", test), ("full_corpus", corpus)):
        scope = {"n_patches": int(len(df)), "per_head": {}, "per_source_label": {},
                 "association": {}}

        print(f"\n--- {name} (n={len(df)}) ---")
        print(f"{'head':<6}{'n':>8}{'mean':>9}{'median':>9}{'sd':>9}"
              f"{'IQR':>9}{'>0.5':>9}")
        for h in heads:
            sub = df[df.head_true == h]
            if sub.empty:
                continue
            d = describe(sub.pad_fraction)
            scope["per_head"][h] = {
                "pad_fraction": d,
                "aspect_ratio": describe(sub.aspect_ratio),
                "box_area_px2": describe(sub.box_area),
            }
            print(f"{h:<6}{d['n']:>8}{d['mean']:>9.4f}{d['median']:>9.4f}"
                  f"{d['sd']:>9.4f}{d['iqr']:>9.4f}{d['frac_above_0.5']:>9.4f}")

        for (ds, lab), sub in df.groupby(["source_dataset", "source_label"]):
            scope["per_source_label"][f"{ds}/{lab}"] = {
                "head": sub.head_true.mode().iat[0],
                "pad_fraction": describe(sub.pad_fraction),
                "aspect_ratio": describe(sub.aspect_ratio),
                "box_area_px2": describe(sub.box_area),
            }

        deciles = pd.qcut(df.pad_fraction, N_DECILES, labels=False,
                          duplicates="drop")
        v_raw, v_corr = cramers_v(df.head_true, deciles)
        mi_pad = mutual_information_bits(df.head_true, df.pad_fraction)
        mi_asp = mutual_information_bits(df.head_true, df.aspect_ratio)
        mi_area = mutual_information_bits(df.head_true, df.box_area)
        h_head = entropy_bits(df.head_true)

        scope["association"] = {
            "cramers_v_head_vs_pad_decile": v_raw,
            "cramers_v_bias_corrected": v_corr,
            "mutual_information_bits": {
                "head_vs_pad_fraction": mi_pad,
                "head_vs_aspect_ratio": mi_asp,
                "head_vs_box_area": mi_area,
            },
            "head_entropy_bits": h_head,
            "mi_as_fraction_of_head_entropy": {
                "pad_fraction": mi_pad / h_head if h_head else 0.0,
                "aspect_ratio": mi_asp / h_head if h_head else 0.0,
                "box_area": mi_area / h_head if h_head else 0.0,
            },
        }
        print(f"  Cramer's V (head vs pad decile) : {v_raw:.4f} "
              f"(bias-corrected {v_corr:.4f})")
        print(f"  I(head; pad_fraction)           : {mi_pad:.4f} bits "
              f"({100*mi_pad/h_head:.1f}% of H(head)={h_head:.4f})")
        print(f"  I(head; aspect_ratio)           : {mi_asp:.4f} bits "
              f"({100*mi_asp/h_head:.1f}%)")
        print(f"  I(head; box_area)               : {mi_area:.4f} bits "
              f"({100*mi_area/h_head:.1f}%)")

        stats["scopes"][name] = scope

    with open(out_dir / "pad_fraction_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"\n-> {(out_dir/'pad_fraction_stats.json').relative_to(REPO_ROOT)}")

    _plot_pad_densities(corpus, heads, out_dir)
    _plot_box_geometry(corpus, heads, out_dir)
    return stats


def _plot_pad_densities(df, heads, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    bins = np.linspace(0, 1, 41)

    for h in heads:
        sub = df[df.head_true == h]
        if sub.empty:
            continue
        axes[0].hist(sub.pad_fraction, bins=bins, density=True, histtype="step",
                     linewidth=2, label=f"{h} (n={len(sub)})", color=HEAD_COLOUR[h])
        axes[1].plot(np.sort(sub.pad_fraction),
                     np.linspace(0, 1, len(sub)), linewidth=2,
                     label=h, color=HEAD_COLOUR[h])

    axes[0].set_xlabel("pad_fraction")
    axes[0].set_ylabel("density")
    axes[0].set_title("Letterbox padding by focal element")
    axes[0].axvline(0.5, color="#666", linestyle=":", linewidth=1)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("pad_fraction")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title("Empirical CDF")
    axes[1].axvline(0.5, color="#666", linestyle=":", linewidth=1)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "pad_fraction_by_head.png", dpi=150)
    plt.close(fig)
    print(f"-> {(out_dir/'pad_fraction_by_head.png').relative_to(REPO_ROOT)}")


def _plot_box_geometry(df, heads, out_dir):
    fig, ax = plt.subplots(figsize=(8.2, 6))
    rng = np.random.default_rng(SEED)
    for h in heads:
        sub = df[df.head_true == h]
        if sub.empty:
            continue
        # Subsample the big heads so V1/V2 are not buried under V3/V4.
        if len(sub) > 4000:
            sub = sub.iloc[rng.choice(len(sub), 4000, replace=False)]
        ax.scatter(sub.aspect_ratio, sub.box_area, s=6, alpha=0.25,
                   label=f"{h} (n={len(sub)})", color=HEAD_COLOUR[h],
                   edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("aspect ratio (long side / short side)")
    ax.set_ylabel("box area (px$^2$)")
    ax.set_title("Annotation box geometry by focal element")
    ax.axvline(3.0, color="#A32E28", linestyle="--", linewidth=1.2)
    ax.text(3.15, ax.get_ylim()[1] * 0.5, "3:1 tiling threshold",
            color="#A32E28", fontsize=9, rotation=90, va="top")
    ax.grid(alpha=0.2, which="both")
    leg = ax.legend(fontsize=9, markerscale=3)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    fig.tight_layout()
    fig.savefig(out_dir / "box_geometry_by_head.png", dpi=150)
    plt.close(fig)
    print(f"-> {(out_dir/'box_geometry_by_head.png').relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "A_pad_only": ["pad_fraction"],
    "B_aspect_only": ["log_aspect_ratio"],
    "C_full_geometry": ["pad_fraction", "log_aspect_ratio", "log_box_w",
                        "log_box_h", "log_box_area"],
}


def _add_log_features(df):
    df = df.copy()
    for col in ("aspect_ratio", "box_w", "box_h", "box_area"):
        df[f"log_{col}"] = np.log10(np.maximum(df[col].astype(float), 1e-6))
    return df


def step2(corpus, heads, out_dir):
    """
    Geometry-only probe. Uses the SAME grouped splits as the real model, so the
    numbers sit on the same footing as fold 0's -- a probe trained on a random
    split would be measuring something else.
    """
    print("\n" + "=" * 78)
    print("STEP 2 -- geometry-only probe (no image content)")
    print("=" * 78)

    df = _add_log_features(corpus)
    train = df[df.split == "train"]
    test = df[df.split == "test"]
    print(f"train {len(train)} patches, test {len(test)} patches "
          f"(grouped split, identical to fold 0)")

    results = {"seed": SEED, "n_train": int(len(train)), "n_test": int(len(test)),
               "feature_sets": {}}

    for name, cols in FEATURE_SETS.items():
        scaler = StandardScaler().fit(train[cols])
        # sklearn >=1.9 dropped multi_class; multinomial is the default for
        # multiclass with lbfgs. class_weight balanced so the probe is not
        # graded on its ability to ignore V2.
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED,
        ).fit(scaler.transform(train[cols]), train.head_true)

        pred = clf.predict(scaler.transform(test[cols]))
        four, three = macro_pair(test.head_true, pred, heads)
        results["feature_sets"][name] = {
            "features": cols,
            "accuracy": float(accuracy_score(test.head_true, pred)),
            "macro_f1_four_head": four,
            "macro_f1_three_head": three,
            "per_head": per_head_table(test.head_true, pred, heads),
            "predicted_head_distribution": {
                k: int(v) for k, v in pd.Series(pred).value_counts().items()},
        }
        print(f"\n  {name}  features={cols}")
        print(f"    accuracy {results['feature_sets'][name]['accuracy']:.4f}"
              f"   macro-F1 4-head {four:.4f}   3-head {three:.4f}")
        for h in heads:
            m = results["feature_sets"][name]["per_head"][h]
            print(f"      {h}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
                  f"F1 {m['f1']:.4f}  n={m['support']}")

    # ---- permutation null, not an assumed 1/K ----
    print(f"\n  Permutation null: {N_PERMUTATIONS} label shuffles on the test "
          f"split (seed {SEED})")
    best = max(results["feature_sets"],
               key=lambda k: results["feature_sets"][k]["macro_f1_three_head"])
    cols = FEATURE_SETS[best]
    scaler = StandardScaler().fit(train[cols])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=SEED
                             ).fit(scaler.transform(train[cols]), train.head_true)
    pred = clf.predict(scaler.transform(test[cols]))

    rng = np.random.default_rng(SEED)
    truth = test.head_true.to_numpy()
    null_four, null_three = [], []
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(truth)
        a, b = macro_pair(shuffled, pred, heads)
        null_four.append(a)
        null_three.append(b)

    null = {
        "n_permutations": N_PERMUTATIONS, "seed": SEED,
        "probe_used_for_null": best,
        "four_head": {
            "mean": float(np.mean(null_four)), "sd": float(np.std(null_four)),
            "p50": float(np.percentile(null_four, 50)),
            "p95": float(np.percentile(null_four, 95)),
            "p99": float(np.percentile(null_four, 99)),
            "max": float(np.max(null_four)),
        },
        "three_head": {
            "mean": float(np.mean(null_three)), "sd": float(np.std(null_three)),
            "p50": float(np.percentile(null_three, 50)),
            "p95": float(np.percentile(null_three, 95)),
            "p99": float(np.percentile(null_three, 99)),
            "max": float(np.max(null_three)),
        },
        "distribution_four_head": [float(x) for x in null_four],
        "distribution_three_head": [float(x) for x in null_three],
    }
    print(f"    4-head null: mean {null['four_head']['mean']:.4f}, "
          f"p95 {null['four_head']['p95']:.4f}")
    print(f"    3-head null: mean {null['three_head']['mean']:.4f}, "
          f"p95 {null['three_head']['p95']:.4f}")

    results["permutation_null_summary"] = {
        "four_head_p95": null["four_head"]["p95"],
        "three_head_p95": null["three_head"]["p95"],
    }

    with open(out_dir / "geometry_probe_metrics.json", "w") as f:
        json.dump(results, f, indent=1)
    with open(out_dir / "permutation_null.json", "w") as f:
        json.dump(null, f, indent=1)
    print(f"\n-> {(out_dir/'geometry_probe_metrics.json').relative_to(REPO_ROOT)}")
    print(f"-> {(out_dir/'permutation_null.json').relative_to(REPO_ROOT)}")
    return results, null


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
def step3(test, heads, out_dir):
    print("\n" + "=" * 78)
    print("STEP 3 -- does fold 0's accuracy track pad_fraction?")
    print("=" * 78)

    test = test.copy()
    test["pad_decile"] = pd.qcut(test.pad_fraction, N_DECILES, labels=False,
                                 duplicates="drop")

    rows = []
    for d, sub in test.groupby("pad_decile"):
        entry = {"decile": int(d),
                 "pad_range": [float(sub.pad_fraction.min()),
                               float(sub.pad_fraction.max())],
                 "n": int(len(sub)),
                 "accuracy": float(sub.correct.mean())}
        for h in heads:
            n = int((sub.head_true == h).sum())
            entry[f"{h}_support"] = n
            entry[f"{h}_f1"] = float(f1_score(
                sub.head_true, sub.head_pred, labels=[h], average="macro",
                zero_division=0)) if n else float("nan")
        rows.append(entry)

    table = pd.DataFrame(rows)
    print(f"{'dec':>4}{'pad range':>18}{'n':>7}{'acc':>8}" +
          "".join(f"{h+' F1':>9}{h+' n':>7}" for h in heads))
    for _, r in table.iterrows():
        line = (f"{int(r.decile):>4}"
                f"{f'{r.pad_range[0]:.3f}-{r.pad_range[1]:.3f}':>18}"
                f"{int(r.n):>7}{r.accuracy:>8.4f}")
        for h in heads:
            f1v = r[f"{h}_f1"]
            line += (f"{'--' if np.isnan(f1v) else f'{f1v:.4f}':>9}"
                     f"{int(r[f'{h}_support']):>7}")
        print(line)

    _plot_f1_by_decile(table, heads, out_dir)
    return table


def _plot_f1_by_decile(table, heads, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for h in heads:
        y = table[f"{h}_f1"].to_numpy()
        n = table[f"{h}_support"].to_numpy()
        x = table["decile"].to_numpy()
        ok = ~np.isnan(y)
        ax.plot(x[ok], y[ok], "-o", linewidth=2, markersize=6, label=h,
                color=HEAD_COLOUR[h])
        for xi, yi, ni in zip(x[ok], y[ok], n[ok]):
            ax.annotate(f"{ni}", (xi, yi), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5,
                        color=HEAD_COLOUR[h])
    ax.set_xlabel("pad_fraction decile  (0 = least padded, 9 = most padded)")
    ax.set_ylabel("F1 within decile")
    ax.set_title("Fold-0 per-head F1 against letterbox padding\n"
                 "point labels are support", fontsize=11)
    ax.set_xticks(range(N_DECILES))
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "f1_by_pad_decile.png", dpi=150)
    plt.close(fig)
    print(f"-> {(out_dir/'f1_by_pad_decile.png').relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------
def bootstrap_f1_ci(df, head, n_boot=N_BOOTSTRAP, seed=SEED):
    """Percentile bootstrap CI on one head's F1, resampling patches."""
    rng = np.random.default_rng(seed)
    y_true = df.head_true.to_numpy()
    y_pred = df.head_pred.to_numpy()
    n = len(df)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(f1_score(y_true[idx], y_pred[idx], labels=[head],
                             average="macro", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def step4(test, heads, out_dir):
    """
    Match V3 and V4 pad_fraction distributions by decile, then re-measure.

    If V3/V4 F1 survives matching, the model is reading the weld. If it falls,
    part of fold 0's V3/V4 separation was crop shape.
    """
    print("\n" + "=" * 78)
    print("STEP 4 -- pad-matched subsample (headline result)")
    print("=" * 78)

    test = test.copy()
    test["pad_decile"] = pd.qcut(test.pad_fraction, N_DECILES, labels=False,
                                 duplicates="drop")
    rng = np.random.default_rng(SEED)

    keep = []
    matching = []
    for d, sub in test.groupby("pad_decile"):
        v3 = sub[sub.head_true == "V3"]
        v4 = sub[sub.head_true == "V4"]
        n = min(len(v3), len(v4))
        matching.append({"decile": int(d), "v3_available": len(v3),
                         "v4_available": len(v4), "kept_each": int(n)})
        if n:
            keep.append(v3.iloc[rng.choice(len(v3), n, replace=False)])
            keep.append(v4.iloc[rng.choice(len(v4), n, replace=False)])
        # V1 and V2 pass through unmatched, and are reported as such.
        keep.append(sub[sub.head_true.isin(["V1", "V2"])])

    matched = pd.concat(keep, ignore_index=True)

    print(f"{'decile':>7}{'V3 avail':>10}{'V4 avail':>10}{'kept each':>11}")
    for m in matching:
        print(f"{m['decile']:>7}{m['v3_available']:>10}{m['v4_available']:>10}"
              f"{m['kept_each']:>11}")

    m_v3 = matched[matched.head_true == "V3"]
    m_v4 = matched[matched.head_true == "V4"]
    print(f"\nmatched subsample: {len(matched)} patches "
          f"(V3 {len(m_v3)}, V4 {len(m_v4)}, "
          f"V1 {(matched.head_true=='V1').sum()}, "
          f"V2 {(matched.head_true=='V2').sum()})")
    print(f"pad_fraction mean after matching: "
          f"V3 {m_v3.pad_fraction.mean():.4f}  V4 {m_v4.pad_fraction.mean():.4f}  "
          f"(before: V3 {test[test.head_true=='V3'].pad_fraction.mean():.4f}  "
          f"V4 {test[test.head_true=='V4'].pad_fraction.mean():.4f})")

    unm_heads = per_head_table(test.head_true, test.head_pred, heads)
    mat_heads = per_head_table(matched.head_true, matched.head_pred, heads)
    u4, u3 = macro_pair(test.head_true, test.head_pred, heads)
    m4, m3 = macro_pair(matched.head_true, matched.head_pred, heads)

    print(f"\n{'head':<6}{'unmatched F1':>14}{'95% CI':>20}"
          f"{'matched F1':>13}{'95% CI':>20}{'delta':>9}")
    out_heads = {}
    for h in heads:
        u_ci = bootstrap_f1_ci(test, h)
        m_ci = bootstrap_f1_ci(matched, h)
        uf, mf = unm_heads[h]["f1"], mat_heads[h]["f1"]
        out_heads[h] = {
            "unmatched": {**unm_heads[h], "f1_ci95": list(u_ci)},
            "matched": {**mat_heads[h], "f1_ci95": list(m_ci)},
            "f1_delta": float(mf - uf),
            "ci_overlap": not (m_ci[1] < u_ci[0] or u_ci[1] < m_ci[0]),
        }
        print(f"{h:<6}{uf:>14.4f}{f'[{u_ci[0]:.4f}, {u_ci[1]:.4f}]':>20}"
              f"{mf:>13.4f}{f'[{m_ci[0]:.4f}, {m_ci[1]:.4f}]':>20}"
              f"{mf-uf:>+9.4f}")

    print(f"\nmacro-F1  4-head: unmatched {u4:.4f} -> matched {m4:.4f} "
          f"({m4-u4:+.4f})")
    print(f"macro-F1  3-head: unmatched {u3:.4f} -> matched {m3:.4f} "
          f"({m3-u3:+.4f})")

    payload = {
        "seed": SEED, "n_bootstrap": N_BOOTSTRAP, "n_deciles": N_DECILES,
        "matched_subsample_size": int(len(matched)),
        "matched_head_counts": {h: int((matched.head_true == h).sum())
                                for h in heads},
        "unmatched_head_counts": {h: int((test.head_true == h).sum())
                                  for h in heads},
        "matching_by_decile": matching,
        "pad_fraction_after_matching": {
            "V3_mean": float(m_v3.pad_fraction.mean()),
            "V4_mean": float(m_v4.pad_fraction.mean()),
            "V3_mean_before": float(test[test.head_true == "V3"].pad_fraction.mean()),
            "V4_mean_before": float(test[test.head_true == "V4"].pad_fraction.mean()),
        },
        "per_head": out_heads,
        "macro_f1": {
            "four_head": {"unmatched": u4, "matched": m4, "delta": float(m4 - u4)},
            "three_head": {"unmatched": u3, "matched": m3, "delta": float(m3 - u3)},
        },
        "accuracy": {"unmatched": float(test.correct.mean()),
                     "matched": float(matched.correct.mean())},
    }
    with open(out_dir / "matched_subsample_metrics.json", "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\n-> {(out_dir/'matched_subsample_metrics.json').relative_to(REPO_ROOT)}")
    return payload


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/diagnostics")
    args = ap.parse_args()

    frame = load_frame()
    heads = frame.head_names
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(out_dir / "predictions_fold0.parquet")
    test = preds[preds.split == "test"].reset_index(drop=True)

    # Full corpus: every patch, from the manifest. Only val/test have
    # predictions, so corpus-level work is geometry only.
    manifest = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "vision_patches" / "manifest.csv")
    manifest = manifest.rename(columns={"head": "head_true"})
    manifest["box_area"] = manifest.box_w * manifest.box_h
    manifest["aspect_ratio"] = (
        np.maximum(manifest.box_w, manifest.box_h)
        / np.maximum(np.minimum(manifest.box_w, manifest.box_h), 1e-9))

    print("=" * 78)
    print("PAD_FRACTION LEAKAGE DIAGNOSTIC")
    print("=" * 78)
    print(f"Frame: {frame}")
    print(f"test split: {len(test)} patches | full corpus: {len(manifest)} patches")

    step1(test, manifest, heads, out_dir)
    step2(manifest, heads, out_dir)
    step3(test, heads, out_dir)
    step4(test, heads, out_dir)

    print("\nDone. See results/diagnostics/.")


if __name__ == "__main__":
    main()
