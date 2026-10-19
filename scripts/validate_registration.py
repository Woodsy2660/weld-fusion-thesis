#!/usr/bin/env python3
"""
Validate registration across all captures using seeded pipeline.

Part D: Generate 30-face validation table with baseline errors and flags.

Usage:
    python scripts/validate_registration.py

Outputs:
    results/geometry/validation_table.csv
    results/geometry/validation_report.md
    results/geometry/baseline_histogram.png
"""

import sys
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.geometry.registration_seeded import (
    load_seeds,
    register_capture_seeded,
    load_config,
    create_diagnostic_overlay,
    detect_hole_from_seed,
    compute_median_colour,
)


def validate_all_captures(
    data_dir: Path,
    seeds_path: Path,
    output_dir: Path,
    save_overlays: bool = True,
) -> list[dict]:
    """Run registration on all captures with seeds.

    Returns list of result dicts for the validation table.
    """
    config = load_config()
    all_seeds = load_seeds(seeds_path)

    results = []

    # Find all captures
    captures = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("C") and d.name[1:3].isdigit()
    ])

    print(f"Found {len(captures)} captures")
    print(f"Seeds available for {len(all_seeds)} captures")
    print("=" * 70)

    for capture_dir in captures:
        capture_id = capture_dir.name
        seeds = all_seeds.get(capture_id, [])

        result = {
            "capture_id": capture_id,
            "n_seeds": len(seeds),
            "baseline_mm": None,
            "baseline_error_mm": None,
            "baseline_pass": None,
            "face_derived": None,
            "face_label": None,
            "face_match": None,
            "y_crown_max_mm": None,
            "n_flags": 0,
            "flags": "",
            "status": "skipped",
        }

        if not seeds:
            print(f"{capture_id}: SKIPPED (no seeds)")
            results.append(result)
            continue

        try:
            pose, cloud, meta = register_capture_seeded(capture_dir, seeds, config)

            baseline = meta["measured_baseline_mm"]
            baseline_error = baseline - 270.0

            result.update({
                "baseline_mm": round(baseline, 2),
                "baseline_error_mm": round(baseline_error, 2),
                "baseline_pass": abs(baseline_error) <= 0.5,
                "face_derived": pose.derived_face,
                "face_label": pose.face_label,
                "face_match": pose.face_match,
                "y_crown_max_mm": round(pose.y_crown_max_mm, 2) if not np.isnan(pose.y_crown_max_mm) else None,
                "n_flags": len(pose.flags),
                "flags": "; ".join(pose.flags),
                "status": "ok" if not pose.flags else "flagged",
            })

            # Save individual result
            result_path = output_dir / f"{capture_id}_registration.json"
            with open(result_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Save overlay
            if save_overlays:
                try:
                    colour_img = compute_median_colour(capture_dir / "color")
                    d6_seeds = [s for s in seeds if s.hole_type == "d6"]
                    d8_seeds = [s for s in seeds if s.hole_type == "d8"]

                    # Get fx from meta for proper rim detection
                    with open(capture_dir / "meta.json") as f:
                        capture_meta = json.load(f)
                    fx = capture_meta["color_intrinsics"]["fx"]
                    standoff = capture_meta.get("standoff_mm_from_depth", 320.0)

                    _, origin_rim = detect_hole_from_seed(
                        colour_img, d6_seeds[0], standoff, fx, config
                    )
                    _, axis_rim = detect_hole_from_seed(
                        colour_img, d8_seeds[0], standoff, fx, config
                    )

                    overlay = create_diagnostic_overlay(
                        colour_img, seeds, origin_rim, axis_rim, config
                    )

                    import cv2
                    overlay_path = output_dir / "overlays" / f"{capture_id}_overlay.png"
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(overlay_path), overlay)
                except Exception as e:
                    print(f"  (overlay error: {e})")

            status_str = f"baseline={baseline:.2f}mm ({baseline_error:+.2f})"
            if pose.flags:
                status_str += f" FLAGS: {len(pose.flags)}"
            print(f"{capture_id}: {status_str}")

        except Exception as e:
            result["status"] = "error"
            result["flags"] = str(e)
            print(f"{capture_id}: ERROR - {e}")

        results.append(result)

    return results


def generate_validation_table(results: list[dict], output_path: Path):
    """Generate CSV validation table."""
    fieldnames = [
        "capture_id", "n_seeds", "status",
        "baseline_mm", "baseline_error_mm", "baseline_pass",
        "face_derived", "face_label", "face_match",
        "y_crown_max_mm", "n_flags", "flags",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def generate_validation_report(results: list[dict], output_path: Path):
    """Generate markdown validation report."""
    n_total = len(results)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_flagged = sum(1 for r in results if r["status"] == "flagged")
    n_error = sum(1 for r in results if r["status"] == "error")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")

    baselines = [r["baseline_mm"] for r in results if r["baseline_mm"] is not None]
    baseline_errors = [r["baseline_error_mm"] for r in results if r["baseline_error_mm"] is not None]
    n_pass = sum(1 for r in results if r["baseline_pass"])

    report = f"""# Registration Validation Report

Generated: {datetime.now().isoformat()}

## Summary

| Metric | Count |
|--------|-------|
| Total captures | {n_total} |
| OK (no flags) | {n_ok} |
| Flagged | {n_flagged} |
| Error | {n_error} |
| Skipped (no seeds) | {n_skipped} |

## Baseline Accuracy

| Metric | Value |
|--------|-------|
| Registered faces | {len(baselines)} |
| Baseline pass (±0.5mm) | {n_pass} / {len(baselines)} |
| Mean baseline | {np.mean(baselines):.2f} mm | (expected 270.0)
| Std baseline | {np.std(baselines):.2f} mm |
| Max error | {max(baseline_errors):.2f} mm |
| Min error | {min(baseline_errors):.2f} mm |

## Per-Capture Results

| Capture | Status | Baseline (mm) | Error (mm) | Pass | Flags |
|---------|--------|---------------|------------|------|-------|
"""

    for r in results:
        baseline = f"{r['baseline_mm']:.2f}" if r["baseline_mm"] else "-"
        error = f"{r['baseline_error_mm']:+.2f}" if r["baseline_error_mm"] is not None else "-"
        pass_str = "✓" if r["baseline_pass"] else "✗" if r["baseline_pass"] is False else "-"
        flags = r["flags"][:50] + "..." if len(r["flags"]) > 50 else r["flags"]
        report += f"| {r['capture_id']} | {r['status']} | {baseline} | {error} | {pass_str} | {flags} |\n"

    report += """

## Flag Summary

"""
    # Count flag types
    flag_counts = {}
    for r in results:
        if r["flags"]:
            for flag in r["flags"].split("; "):
                flag_type = flag.split("_")[0] if "_" in flag else flag
                flag_counts[flag_type] = flag_counts.get(flag_type, 0) + 1

    if flag_counts:
        report += "| Flag Type | Count |\n|-----------|-------|\n"
        for flag_type, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            report += f"| {flag_type} | {count} |\n"
    else:
        report += "No flags raised.\n"

    with open(output_path, "w") as f:
        f.write(report)


def generate_baseline_histogram(results: list[dict], output_path: Path):
    """Generate baseline error histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping histogram")
        return

    baseline_errors = [r["baseline_error_mm"] for r in results if r["baseline_error_mm"] is not None]

    if not baseline_errors:
        print("No baseline data for histogram")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(baseline_errors, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="green", linestyle="--", linewidth=2, label="Expected (270mm)")
    ax.axvline(-0.5, color="red", linestyle=":", linewidth=1, label="±0.5mm tolerance")
    ax.axvline(0.5, color="red", linestyle=":", linewidth=1)

    ax.set_xlabel("Baseline Error (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Registration Baseline Errors (n={len(baseline_errors)})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Histogram saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate registration across all captures")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw" / "S1"))
    parser.add_argument("--seeds", default=str(REPO_ROOT / "configs" / "hole_seeds.csv"))
    parser.add_argument("--output", default=str(REPO_ROOT / "results" / "geometry"))
    parser.add_argument("--no-overlays", action="store_true", help="Skip saving overlays")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    seeds_path = Path(args.seeds)
    output_dir = Path(args.output)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    if not seeds_path.exists():
        print(f"ERROR: Seeds file not found: {seeds_path}")
        print("Run scripts/seed_holes.py first to create seeds.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run validation
    results = validate_all_captures(
        data_dir, seeds_path, output_dir,
        save_overlays=not args.no_overlays
    )

    # Generate outputs
    table_path = output_dir / "validation_table.csv"
    generate_validation_table(results, table_path)
    print(f"\nValidation table saved to: {table_path}")

    report_path = output_dir / "validation_report.md"
    generate_validation_report(results, report_path)
    print(f"Validation report saved to: {report_path}")

    histogram_path = output_dir / "baseline_histogram.png"
    generate_baseline_histogram(results, histogram_path)

    # Print summary
    n_pass = sum(1 for r in results if r["baseline_pass"])
    n_total = sum(1 for r in results if r["baseline_mm"] is not None)
    print(f"\n{'=' * 60}")
    print(f"VALIDATION COMPLETE: {n_pass}/{n_total} faces pass ±0.5mm baseline tolerance")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
