#!/usr/bin/env python3
"""Visualize the weld bead 'trench' artifact documented in LIMITATIONS.md L2.

The Intel D435 depth sensor reads the convex MIG weld bead as a dip/trench
(farther from camera) instead of a crown (closer to camera). This corrupts
all geometry measurements and explains why all decided cells show misalignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path
import json
from PIL import Image


def load_depth_data(coupon_dir: Path) -> tuple:
    """Load and average depth frames for a coupon."""
    depth_files = sorted(coupon_dir.glob("depth/*.npy"))
    if not depth_files:
        return None, None

    meta_path = coupon_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    depth_frames = []
    for df in depth_files[:10]:
        frame = np.load(df)
        depth_frames.append(frame)

    depth_avg = np.nanmean(depth_frames, axis=0)
    depth_scale = meta.get('depth_scale_m', 0.0001)
    depth_mm = depth_avg * depth_scale * 1000

    return depth_mm, meta


def load_color_image(coupon_dir: Path) -> np.ndarray:
    """Load first color image."""
    color_files = sorted(coupon_dir.glob("color/*.png"))
    if not color_files:
        return None
    return np.array(Image.open(color_files[0]))


def main():
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw" / "S1"
    output_dir = base_dir / "results" / "thesis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use C05 as documented in LIMITATIONS.md
    coupon_id = "C05"
    coupon_dir = raw_dir / f"{coupon_id}_top_r0"

    depth_mm, meta = load_depth_data(coupon_dir)
    color_img = load_color_image(coupon_dir)

    if depth_mm is None:
        print(f"Error: Could not load {coupon_dir}")
        return

    # === Figure: Side-by-side color vs depth showing trench artifact ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get image dimensions
    h, w = depth_mm.shape
    weld_y = h // 2  # Approximate weld centerline

    # Top left: Color image with cross-section line
    ax = axes[0, 0]
    if color_img is not None:
        # Resize color to match depth dimensions
        color_resized = np.array(Image.fromarray(color_img).resize((w, h)))
        ax.imshow(color_resized)
    ax.axhline(weld_y, color='red', linestyle='--', linewidth=2, label='Cross-section line')
    ax.set_title(f'{coupon_id}: Color Image\n(Convex MIG weld cap visible)', fontsize=11)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend(loc='upper right')

    # Top right: Depth image with cross-section line
    ax = axes[0, 1]
    depth_display = np.where(depth_mm > 0, depth_mm, np.nan)
    im = ax.imshow(depth_display, cmap='viridis', aspect='auto',
                   vmin=np.nanpercentile(depth_display, 5),
                   vmax=np.nanpercentile(depth_display, 95))
    ax.axhline(weld_y, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{coupon_id}: Depth Image\n(Weld appears as DARKER = farther)', fontsize=11)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Depth (mm)')

    # Bottom left: Cross-section profile showing trench
    ax = axes[1, 0]

    # Extract multiple cross-sections
    y_offsets = [weld_y - 50, weld_y, weld_y + 50]
    colors = ['green', 'red', 'blue']
    labels = ['Top plate (y-50)', 'Weld centerline', 'Bottom plate (y+50)']

    for y_off, color, label in zip(y_offsets, colors, labels):
        profile = depth_mm[y_off, :]
        x_mm = np.arange(len(profile)) * 0.445
        # Normalize to plate reference
        plate_ref = np.nanpercentile(profile[profile > 0], 50)
        ax.plot(x_mm, profile - plate_ref, color=color, linewidth=1.5, label=label, alpha=0.8)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('X position along weld (mm)')
    ax.set_ylabel('Depth relative to plate (mm)')
    ax.set_title('Depth Profiles: Weld Reads as TRENCH (positive = farther)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)

    # Annotate the trench artifact
    ax.annotate('Weld bead appears\nas trench (farther)\nnot crown (closer)',
                xy=(250, 2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Bottom right: Schematic explanation
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Draw expected vs actual
    # Expected (correct): weld crown above plates
    ax.fill_between([1, 4], [6, 6], [6.5, 6.5], color='lightblue', label='Plate (expected)')
    ax.fill_between([4, 6], [6, 6], [7.5, 7.5], color='lightgreen', label='Weld crown (expected)')
    ax.fill_between([6, 9], [6, 6], [6.5, 6.5], color='lightblue')

    ax.text(5, 8.5, 'EXPECTED: Convex weld cap', ha='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(5, 7.5), xytext=(5, 8.2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Actual (sensor error): weld appears as trench
    ax.fill_between([1, 4], [2, 2], [2.5, 2.5], color='lightblue')
    ax.fill_between([4, 6], [2, 2], [1, 1], color='red', alpha=0.5, label='Weld trench (measured)')
    ax.fill_between([6, 9], [2, 2], [2.5, 2.5], color='lightblue')

    ax.text(5, 0.3, 'MEASURED: Trench artifact', ha='center', fontsize=11, fontweight='bold', color='red')
    ax.annotate('', xy=(5, 1), xytext=(5, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.text(5, 4.5, 'Cause: Specular reflection from\nshiny weld surface causes\nstereo matching failure',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.axis('off')
    ax.set_title('L2: Depth Sensor Artifact Explanation', fontsize=11)

    plt.suptitle(f'Weld Bead Trench Artifact ({coupon_id}_top)\nDocumented in results/geometry/LIMITATIONS.md L2',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "f4_14_bead_trench_artifact.png", bbox_inches='tight', dpi=150)
    fig.savefig(output_dir / "f4_14_bead_trench_artifact.pdf", bbox_inches='tight', dpi=150)
    print(f"Saved: {output_dir / 'f4_14_bead_trench_artifact.pdf'}")

    # === Figure 2: How this causes false misalignment readings ===
    fig2, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)

    # Left side: True geometry (plates aligned, weld cap)
    ax.text(3.5, 9.5, 'TRUE GEOMETRY', ha='center', fontsize=12, fontweight='bold')
    # Left plate
    ax.fill_between([0.5, 3], [6, 6], [6.5, 6.5], color='steelblue', edgecolor='black')
    ax.text(1.75, 6.25, 'Plate L', ha='center', va='center', fontsize=9, color='white')
    # Weld cap
    ax.fill_between([3, 4], [6, 6], [7, 7], color='orange', edgecolor='black')
    ax.text(3.5, 6.5, 'Weld', ha='center', va='center', fontsize=9)
    # Right plate
    ax.fill_between([4, 6.5], [6, 6], [6.5, 6.5], color='steelblue', edgecolor='black')
    ax.text(5.25, 6.25, 'Plate R', ha='center', va='center', fontsize=9, color='white')

    # Datum lines - both at same height
    ax.plot([0.5, 3], [6.5, 6.5], 'g--', linewidth=2)
    ax.plot([4, 6.5], [6.5, 6.5], 'g--', linewidth=2)
    ax.text(3.5, 7.5, 'Datums aligned\n→ Misalignment = 0', ha='center', fontsize=10, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Right side: What sensor sees (trench artifact)
    ax.text(10.5, 9.5, 'SENSOR MEASUREMENT', ha='center', fontsize=12, fontweight='bold', color='red')
    # Left plate
    ax.fill_between([7.5, 10], [6, 6], [6.5, 6.5], color='steelblue', edgecolor='black')
    ax.text(8.75, 6.25, 'Plate L', ha='center', va='center', fontsize=9, color='white')
    # Weld appears as trench - one side higher
    ax.fill_between([10, 10.5], [4.5, 4.5], [6, 6], color='red', alpha=0.5, edgecolor='black')
    ax.fill_between([10.5, 11], [5.5, 5.5], [6, 6], color='red', alpha=0.5, edgecolor='black')
    ax.text(10.5, 5, 'Trench', ha='center', va='center', fontsize=8)
    # Right plate - appears higher due to artifact
    ax.fill_between([11, 13.5], [6.5, 6.5], [7, 7], color='steelblue', edgecolor='black')
    ax.text(12.25, 6.75, 'Plate R', ha='center', va='center', fontsize=9, color='white')

    # Datum lines - different heights!
    ax.plot([7.5, 10], [6.5, 6.5], 'r--', linewidth=2)
    ax.plot([11, 13.5], [7, 7], 'r--', linewidth=2)

    # Misalignment arrow
    ax.annotate('', xy=(10.25, 7), xytext=(10.25, 6.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(10.25, 7.8, 'False\nmisalignment!', ha='center', fontsize=10, color='red', fontweight='bold')

    ax.text(10.5, 4, 'Datums differ by 0.5-10mm\n→ All cells predict {U}', ha='center', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.axis('off')
    ax.set_title('How Trench Artifact Creates False Misalignment Readings', fontsize=13)

    plt.tight_layout()
    fig2.savefig(output_dir / "f4_15_false_misalignment.png", bbox_inches='tight', dpi=150)
    fig2.savefig(output_dir / "f4_15_false_misalignment.pdf", bbox_inches='tight', dpi=150)
    print(f"Saved: {output_dir / 'f4_15_false_misalignment.pdf'}")

    plt.close('all')
    print("\nDiagnosis complete.")
    print("\nSUMMARY:")
    print("The depth sensor reads the convex weld bead as a TRENCH (farther from camera)")
    print("due to specular reflections causing stereo matching failures.")
    print("This artifact shifts the plate datum measurements differently on each side,")
    print("creating false 'misalignment' readings of 0.7-10mm in all decided cells.")
    print("\nThis is a SENSOR ARTIFACT, not an actual fabrication defect.")
    print("The geometry α = 1.0 (total discounting) is appropriate given this limitation.")


if __name__ == "__main__":
    main()
