"""
Stage 2 of the vision pipeline: label-stratified grouped train/val/test split.

Source-image deduplication (exact MD5 + perceptual phash + Roboflow filename
identity) and label-conflict handling happen in convert_and_remap.py, because
they must run at the SOURCE IMAGE level before any cropping.

Two things happen here.

1. HEAD RE-RESOLUTION. Every patch's head is recomputed from its source_label
   through configs/class_map.yaml and configs/frame.yaml, rather than trusted
   from whatever the manifest was written with. A frame change (V2 removed, V4
   widened, Crack remapped) therefore propagates without re-cropping anything.

2. LABEL-STRATIFIED GROUPED SPLIT. Frames stay whole -- that is the hard
   leakage constraint -- while the joint (head, source_dataset, source_label)
   composition is held close to the corpus across train/val/test.

   The previous greedy quota fill balanced only head counts, and consequently
   assigned whole source-label populations to single splits: Overlap was 32.6%
   of V3 in the corpus but 1.2% of V3 in test, while Burn Through went the
   other way. Every per-head metric computed on that split was an average over
   the wrong weights.

Usage:
    python src/vision/merge_dedup_split.py [--seed 42] [--dry-run]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import DISCARD, load_class_map, load_frame  # noqa: E402

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
N_FOLDS = 10                       # 8 train / 1 val / 1 test
MIN_PATCHES_PER_HEAD = 200
MAX_IMBALANCE_RATIO = 10.0

# Task A acceptance: any (head, source_label) pair at or above this share of
# its head must land within DRIFT_BOUND percentage points in val and test.
SHARE_THRESHOLD = 0.02
DRIFT_BOUND = 5.0

NEGATIVE_LABEL = "__negative__"


def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Head re-resolution
# ---------------------------------------------------------------------------
def resolve_heads(rows, frame, class_map):
    """
    Recompute every patch's head from its source_label via the class map.

    Negatives keep the head recorded at crop time: they are annotation-free
    regions with no source class to map, and they belong to whichever head
    represents "no conclusive evidence". That head is not hardcoded here -- the
    value already in the manifest is validated against the frame instead.

    Raises if any row ends up outside the frame's declared heads, so a config
    change that orphans patches stops the pipeline rather than silently
    dropping them.
    """
    changed = Counter()
    dropped = []
    out = []

    for r in rows:
        label = r["source_label"]
        if label == NEGATIVE_LABEL:
            head = r["head"]
        else:
            mapping = class_map.get(r["source_dataset"], {})
            if label not in mapping:
                raise KeyError(
                    f"configs/class_map.yaml: {r['source_dataset']}/{label!r} "
                    f"is unmapped, but patches carrying it exist in the "
                    f"manifest. Refusing to guess."
                )
            head = mapping[label]

        if head == DISCARD:
            dropped.append(r)
            continue
        if head not in frame.head_names:
            raise ValueError(
                f"patch {r['patch_path']} resolves to head {head!r}, which is "
                f"not declared in configs/frame.yaml ({frame.head_names}). "
                f"A head was removed without remapping its patches."
            )
        if head != r["head"]:
            changed[(r["head"], head, label)] += 1
        r["head"] = head
        out.append(r)

    return out, changed, dropped


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------
GROUP_KEY = "source_frame"      # overridden by --group-key


def _group_of(row):
    """The grouping key for one patch, with a fallback chain."""
    return row.get(GROUP_KEY) or row.get("source_frame") or row["source_image"]


def _frame_key(rows_for_frame):
    """
    Stratification label for one source frame: its dominant
    (source_dataset, source_label) by patch count. Ties break on the name so
    the assignment is reproducible.
    """
    counts = Counter((r["source_dataset"], r["source_label"])
                     for r in rows_for_frame)
    top = max(counts.values())
    return min(k for k, v in counts.items() if v == top)


def stratified_group_split(rows, seed):
    """
    Approach 1: sklearn StratifiedGroupKFold over source frames.

    groups = source_frame (the hard constraint: a frame never splits)
    y      = the frame's dominant (source_dataset, source_label)

    10 folds; one becomes test, one val, the remaining eight train.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    by_frame = defaultdict(list)
    for r in rows:
        by_frame[_group_of(r)].append(r)

    frames = sorted(by_frame)
    y_raw = [_frame_key(by_frame[f]) for f in frames]

    # StratifiedGroupKFold needs at least n_splits members per class; fold
    # rare labels together so it does not refuse outright. They are exempt from
    # the acceptance bound anyway.
    counts = Counter(y_raw)
    y = [k if counts[k] >= N_FOLDS else ("__rare__", "__rare__") for k in y_raw]
    classes = sorted(set(y))
    y_idx = np.array([classes.index(k) for k in y])
    groups = np.arange(len(frames))
    X = np.zeros((len(frames), 1))

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=seed)
    folds = [test_idx for _, test_idx in sgkf.split(X, y_idx, groups)]
    # Largest folds to train keeps the 80/10/10 ratio closest.
    order = sorted(range(N_FOLDS), key=lambda i: len(folds[i]))
    test_fold, val_fold = order[0], order[1]

    assignment = {}
    for i, fold in enumerate(folds):
        split = ("test" if i == test_fold
                 else "val" if i == val_fold else "train")
        for j in fold:
            assignment[frames[j]] = split

    for r in rows:
        r["split"] = assignment[_group_of(r)]
    return rows, assignment, "StratifiedGroupKFold(n_splits=10, groups=source_frame)"


def greedy_composition_split(rows, seed):
    """
    Approach 2 (fallback): greedy deficit-minimising assignment.

    Frames sorted by descending patch count; each goes to whichever split has
    the largest total shortfall against its target composition vector, measured
    as sum over labels of |current_share - target_share|.
    """
    import random
    rng = random.Random(seed)

    by_frame = defaultdict(list)
    for r in rows:
        by_frame[_group_of(r)].append(r)

    labels = sorted({(r["source_dataset"], r["source_label"]) for r in rows})
    target = Counter((r["source_dataset"], r["source_label"]) for r in rows)
    total = sum(target.values())
    target_share = {k: target[k] / total for k in labels}

    filled = {s: Counter() for s in SPLIT_RATIOS}
    totals = {s: 0 for s in SPLIT_RATIOS}

    frames = list(by_frame)
    rng.shuffle(frames)
    frames.sort(key=lambda f: -len(by_frame[f]))

    assignment = {}
    for f in frames:
        group = Counter((r["source_dataset"], r["source_label"])
                        for r in by_frame[f])
        size = sum(group.values())

        # The size quota GATES the choice; composition only decides between
        # splits that are still under quota. Scoring both additively lets the
        # composition term -- which is always smaller for a large split --
        # starve val and test permanently.
        under = [s for s in SPLIT_RATIOS
                 if totals[s] < SPLIT_RATIOS[s] * total]
        candidates = under or list(SPLIT_RATIOS)

        def composition_distance(s):
            n = totals[s] + size
            return sum(abs((filled[s][k] + group[k]) / n - target_share[k])
                       for k in labels)

        # Tie-break on relative shortfall so the split furthest behind its
        # quota wins when compositions are equally good.
        best = min(candidates,
                   key=lambda s: (composition_distance(s),
                                  totals[s] / (SPLIT_RATIOS[s] * max(1, total))))
        assignment[f] = best
        filled[best].update(group)
        totals[best] += size

    for r in rows:
        r["split"] = assignment[_group_of(r)]
    return rows, assignment, "greedy deficit-minimising composition fill"


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------
def composition_table(rows, frame):
    """Per head: corpus/train/val/test share of each (dataset, label)."""
    out = {}
    for h in frame.head_names:
        sub = [r for r in rows if r["head"] == h]
        corpus = Counter(f"{r['source_dataset']}/{r['source_label']}" for r in sub)
        n = sum(corpus.values())
        per_split = {}
        for s in SPLIT_RATIOS:
            ss = [r for r in sub if r["split"] == s]
            c = Counter(f"{r['source_dataset']}/{r['source_label']}" for r in ss)
            per_split[s] = (c, sum(c.values()))
        rowsout = []
        for label, cnt in corpus.most_common():
            entry = {"label": label, "corpus_n": cnt,
                     "corpus_share": cnt / n if n else 0.0}
            for s in SPLIT_RATIOS:
                c, tot = per_split[s]
                entry[f"{s}_n"] = c.get(label, 0)
                entry[f"{s}_share"] = c.get(label, 0) / tot if tot else 0.0
                entry[f"{s}_drift_pp"] = 100 * (entry[f"{s}_share"]
                                                - entry["corpus_share"])
            rowsout.append(entry)
        out[h] = {"n": n, "labels": rowsout}
    return out


def check_acceptance(table):
    """Pairs at >= SHARE_THRESHOLD must be within DRIFT_BOUND pp in val/test."""
    failures, checked = [], 0
    for head, d in table.items():
        for e in d["labels"]:
            if e["corpus_share"] < SHARE_THRESHOLD:
                continue
            checked += 1
            for s in ("val", "test"):
                if abs(e[f"{s}_drift_pp"]) > DRIFT_BOUND:
                    failures.append({
                        "head": head, "label": e["label"], "split": s,
                        "corpus_share": e["corpus_share"],
                        "split_share": e[f"{s}_share"],
                        "drift_pp": e[f"{s}_drift_pp"],
                    })
    return failures, checked


def check_leakage(rows, keys=("source_frame", "source_image")):
    """None of the given keys may span splits."""
    bad = {}
    for key in keys:
        if key not in rows[0]:
            continue
        seen = defaultdict(set)
        for r in rows:
            seen[r.get(key) or r["source_image"]].add(r["split"])
        offenders = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
        if offenders:
            bad[key] = offenders
    return bad


# ---------------------------------------------------------------------------
def write_composition_report(table, frame, approach, failures, checked, path,
                             head_counts, weights):
    lines = [
        "# Split composition",
        "",
        f"Assignment: **{approach}**",
        "",
        f"Acceptance: every (head, source_label) pair at or above "
        f"{100*SHARE_THRESHOLD:.0f}% of its head must sit within "
        f"±{DRIFT_BOUND:.0f} percentage points of its corpus share in both "
        f"val and test. Pairs below that share are exempt from the bound but "
        f"are still listed.",
        "",
        f"- pairs checked: **{checked}**",
        f"- pairs outside the bound: **{len(failures)}**",
        "",
    ]
    if failures:
        lines += ["## Pairs outside the bound", "",
                  "| head | label | split | corpus | split | drift (pp) |",
                  "|---|---|---|---|---|---|"]
        for f in failures:
            lines.append(
                f"| {f['head']} | `{f['label']}` | {f['split']} | "
                f"{100*f['corpus_share']:.1f}% | {100*f['split_share']:.1f}% | "
                f"{f['drift_pp']:+.1f} |")
        lines.append("")
    else:
        lines += ["All checked pairs are within the bound.", ""]

    lines += ["## Head totals", "",
              "| head | focal element | patches | class weight |",
              "|---|---|---|---|"]
    for h in frame.head_names:
        els = "{" + ",".join(frame.focal_element(h)) + "}"
        lines.append(f"| {h} | `{els}` | {head_counts[h]:,} | "
                     f"{weights[h]:.4f} |")
    lines.append("")

    for h in frame.head_names:
        d = table[h]
        els = "{" + ",".join(frame.focal_element(h)) + "}"
        drift = max((abs(e["test_drift_pp"]) for e in d["labels"]
                     if e["corpus_share"] >= SHARE_THRESHOLD), default=0.0)
        lines += [
            f"## {h} `{els}` — {d['n']:,} patches",
            "",
            f"Maximum test drift among pairs at ≥{100*SHARE_THRESHOLD:.0f}% "
            f"share: **{drift:.1f} pp**",
            "",
            "| source label | corpus | train | val | test | test drift (pp) |",
            "|---|---|---|---|---|---|",
        ]
        for e in d["labels"]:
            mark = "" if e["corpus_share"] >= SHARE_THRESHOLD else " ·"
            lines.append(
                f"| `{e['label']}`{mark} | {100*e['corpus_share']:.1f}% "
                f"({e['corpus_n']:,}) | {100*e['train_share']:.1f}% | "
                f"{100*e['val_share']:.1f}% | {100*e['test_share']:.1f}% | "
                f"{e['test_drift_pp']:+.1f} |")
        lines += ["", "`·` = below the 2% threshold, exempt from the bound.", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--patches-dir", default="data/processed/vision_patches")
    ap.add_argument("--force-greedy", action="store_true",
                    help="skip StratifiedGroupKFold and use the fallback")
    ap.add_argument("--group-key", default="source_frame",
                    help="manifest column to group on; source_sequence closes "
                         "sequence leakage that source_frame cannot see")
    ap.add_argument("--composition-out",
                    default="results/vision/split_composition.md")
    args = ap.parse_args()

    global GROUP_KEY
    GROUP_KEY = args.group_key

    frame = load_frame()
    class_map = load_class_map(frame=frame)
    patches = REPO_ROOT / args.patches_dir
    manifest_path = patches / "manifest.csv"
    if not manifest_path.exists():
        sys.exit(f"ERROR: {manifest_path} not found. "
                 f"Run src/vision/convert_and_remap.py first.")

    rows = load_manifest(manifest_path)

    print("=" * 78)
    print("LABEL-STRATIFIED GROUPED SPLIT")
    print("=" * 78)
    print(f"Frame    : {frame}")
    print(f"Heads    : {frame.head_names} (n={frame.num_heads})")
    print(f"Manifest : {manifest_path.relative_to(REPO_ROOT)}  ({len(rows)} patches)")
    print(f"Seed     : {args.seed}   ratios {SPLIT_RATIOS}")

    # ---- head re-resolution ----
    rows, changed, dropped = resolve_heads(rows, frame, class_map)
    if changed:
        print("\nHead re-resolution from configs/class_map.yaml:")
        for (old, new, label), n in changed.most_common():
            print(f"  {old} -> {new:<4} {label:<28} {n:>6} patches")
    if dropped:
        print(f"  {len(dropped)} patches now map to discard and were removed")
    if not changed and not dropped:
        print("\nHead re-resolution: no changes (manifest already matches config)")

    n_frames = len({_group_of(r) for r in rows})
    print(f"\nGrouping key: {GROUP_KEY}")
    print(f"Groups: {n_frames}")
    if GROUP_KEY != "source_frame":
        print(f"  (against {len({r['source_frame'] for r in rows})} "
              f"source frames)")

    # ---- split ----
    # Try both, keep whichever is better. The earlier logic fell back to the
    # greedy fill whenever approach 1 missed acceptance, even when approach 1
    # was strictly the better of the two -- which is the common case when the
    # grouping is too coarse for ANY assignment to satisfy the bounds.
    candidates = []
    if not args.force_greedy:
        try:
            r, a, name = stratified_group_split([dict(x) for x in rows],
                                                args.seed)
            tbl = composition_table(r, frame)
            fails, checked = check_acceptance(tbl)
            candidates.append((len(fails), r, a, name, tbl, fails, checked))
            print(f"\n[1] {name}")
            print(f"    pairs checked {checked}, outside bound {len(fails)}")
        except Exception as exc:
            print(f"\n[1] StratifiedGroupKFold unavailable/failed: {exc}")

    r, a, name = greedy_composition_split([dict(x) for x in rows], args.seed)
    tbl = composition_table(r, frame)
    fails, checked = check_acceptance(tbl)
    candidates.append((len(fails), r, a, name, tbl, fails, checked))
    print(f"\n[2] {name}")
    print(f"    pairs checked {checked}, outside bound {len(fails)}")

    candidates.sort(key=lambda c: c[0])
    _, rows, assignment, approach, table, failures, checked = candidates[0]
    print(f"\n--> using: {approach}  ({len(failures)} pairs outside bound)")
    if failures:
        print("    NOTE: acceptance NOT met. Reported, not relaxed -- a corpus "
              "that cannot be\n          both grouped and stratified is itself "
              "a finding about the data.")

    # ---- leakage guard ----
    guard_keys = ["source_frame", "source_image"]
    if GROUP_KEY not in guard_keys:
        guard_keys.append(GROUP_KEY)
    leaks = check_leakage(rows, guard_keys)
    if leaks:
        for key, offenders in leaks.items():
            print(f"\nFATAL: {len(offenders)} {key} values span multiple splits")
            for k, v in list(offenders.items())[:3]:
                print(f"    {k} -> {v}")
        sys.exit("leakage guard failed")
    print(f"\nLeakage guard: none of {guard_keys} spans splits  OK")

    # ---- report ----
    per = {s: Counter() for s in SPLIT_RATIOS}
    imgs = Counter()
    for r in rows:
        per[r["split"]][r["head"]] += 1
    for f, s in assignment.items():
        imgs[s] += 1

    head_counts = {h: sum(per[s][h] for s in SPLIT_RATIOS)
                   for h in frame.head_names}
    n_total = sum(head_counts.values())
    k = frame.num_heads
    raw = {h: n_total / (k * head_counts[h]) if head_counts[h] else 0.0
           for h in frame.head_names}
    mean_w = sum(raw.values()) / k
    weights = {h: raw[h] / mean_w for h in frame.head_names}

    print("\nPATCHES PER HEAD PER SPLIT")
    print("=" * 78)
    hdr = (f"{'head':<7}{'focal':<14}"
           + "".join(f"{s:>9}" for s in SPLIT_RATIOS)
           + f"{'TOTAL':>10}{'weight':>9}")
    print(hdr)
    print("-" * len(hdr))
    for h in frame.head_names:
        els = "{" + ",".join(frame.focal_element(h)) + "}"
        print(f"{h:<7}{els:<14}"
              + "".join(f"{per[s][h]:>9}" for s in SPLIT_RATIOS)
              + f"{head_counts[h]:>10}{weights[h]:>9.4f}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<21}"
          + "".join(f"{sum(per[s].values()):>9}" for s in SPLIT_RATIOS)
          + f"{n_total:>10}")
    print(f"{'frames':<21}"
          + "".join(f"{imgs[s]:>9}" for s in SPLIT_RATIOS)
          + f"{n_frames:>10}")

    nz = [head_counts[h] for h in frame.head_names if head_counts[h]]
    ratio = max(nz) / min(nz)
    print(f"\nHead imbalance ratio: {ratio:.2f}:1")

    print("\nMaximum test drift per head (pairs at >=2% share):")
    for h in frame.head_names:
        d = max((abs(e["test_drift_pp"]) for e in table[h]["labels"]
                 if e["corpus_share"] >= SHARE_THRESHOLD), default=0.0)
        print(f"  {h}: {d:.1f} pp")

    if failures:
        print(f"\nWARNING: {len(failures)} pairs outside the "
              f"{DRIFT_BOUND:.0f} pp bound:")
        for f in failures[:10]:
            print(f"  {f['head']} {f['label']:<34} {f['split']:<5} "
                  f"corpus {100*f['corpus_share']:.1f}% -> "
                  f"{100*f['split_share']:.1f}%  ({f['drift_pp']:+.1f} pp)")

    # ---- health ----
    print("\nDATASET HEALTH")
    print("-" * 78)
    warned = False
    for h in frame.head_names:
        if head_counts[h] == 0:
            warned = True
            print(f"  ERROR: head '{h}' has NO patches.")
        elif head_counts[h] < MIN_PATCHES_PER_HEAD:
            warned = True
            print(f"  WARNING: head '{h}' has {head_counts[h]} patches.")
    for s in SPLIT_RATIOS:
        for h in frame.head_names:
            if head_counts[h] and per[s][h] == 0:
                warned = True
                print(f"  WARNING: split '{s}' has no '{h}' patches.")
    if ratio > MAX_IMBALANCE_RATIO:
        warned = True
        print(f"  WARNING: imbalance {ratio:.2f}:1 exceeds "
              f"{MAX_IMBALANCE_RATIO:.0f}:1")
    if not warned:
        print("  All checks passed.")

    # ---- write ----
    if not args.dry_run:
        cols = list(rows[0].keys())
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nManifest rewritten: {manifest_path.relative_to(REPO_ROOT)}")

        report = REPO_ROOT / args.composition_out
        write_composition_report(table, frame, approach, failures, checked,
                                 report, head_counts, weights)
        print(f"Composition   : {report.relative_to(REPO_ROOT)}")

        summary = {
            "seed": args.seed, "ratios": SPLIT_RATIOS, "approach": approach,
            "group_key": GROUP_KEY, "n_groups": n_frames,
            "heads": frame.head_names,
            "focal_elements": {h: frame.focal_element(h)
                               for h in frame.head_names},
            "patches_per_head": head_counts,
            "patches_per_split": {s: dict(per[s]) for s in SPLIT_RATIOS},
            "frames_per_split": dict(imgs),
            "class_weights": weights,
            "imbalance_ratio": ratio,
            "acceptance": {
                "share_threshold": SHARE_THRESHOLD,
                "drift_bound_pp": DRIFT_BOUND,
                "pairs_checked": checked,
                "pairs_outside_bound": len(failures),
                "failures": failures,
                "max_test_drift_pp": {
                    h: max((abs(e["test_drift_pp"]) for e in table[h]["labels"]
                            if e["corpus_share"] >= SHARE_THRESHOLD), default=0.0)
                    for h in frame.head_names},
            },
            "head_reresolution": {f"{o}->{n}:{lab}": c
                                  for (o, n, lab), c in changed.items()},
        }
        with open(patches / "split_summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        print(f"Summary       : "
              f"{(patches/'split_summary.json').relative_to(REPO_ROOT)}")
        print("\nNext: python src/vision/train.py --config configs/vision_train.yaml")


if __name__ == "__main__":
    main()
