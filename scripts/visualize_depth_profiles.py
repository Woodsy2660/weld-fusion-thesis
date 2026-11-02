#!/usr/bin/env python3
"""Visualize depth data along the weld to diagnose misalignment readings.

Creates:
1. Side view (cross-section) profiles across the weld
2. Comparison of decided vs vacuous cells
3. Check if underfill is being read as misalignment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import json


def load_depth_data(coupon_dir: Path) -> tuple:
    """Load and average depth frames for a coupon."""
    depth_files = sorted(coupon_dir.glob("depth/*.npy"))
    if not depth_files:
        return None, None

    # Load meta
    meta_path = coupon_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # Average multiple depth frames for noise reduction
    depth_frames = []
    for df in depth_files[:10]:  # Use first 10 frames
        frame = np.load(df)
        depth_frames.append(frame)

    depth_avg = np.nanmean(depth_frames, axis=0)

    # Convert to mm
    depth_scale = meta.get('depth_scale_m', 0.0001)
    depth_mm = depth_avg * depth_scale * 1000

    return depth_mm, meta


def extract_weld_profile(depth_mm: np.ndarray, y_center: int = None,
                         profile_width: int = 10) -> np.ndarray:
    """Extract a horizontal profile along the weld centerline."""
    if y_center is None:
        y_center = depth_mm.shape[0] // 2

    # Average over profile_width rows for noise reduction
    y_start = max(0, y_center - profile_width // 2)
    y_end = min(depth_mm.shape[0], y_center + profile_width // 2)

    profile = np.nanmean(depth_mm[y_start:y_end, :], axis=0)
    return profile


def extract_cross_section(depth_mm: np.ndarray, x_position: int,
                          section_width: int = 5) -> np.ndarray:
    """Extract a vertical cross-section across the weld."""
    x_start = max(0, x_position - section_width // 2)
    x_end = min(depth_mm.shape[1], x_position + section_width // 2)

    section = np.nanmean(depth_mm[:, x_start:x_end], axis=1)
    return section


def main():
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw" / "S1"
    output_dir = base_dir / "results" / "thesis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Coupons to analyze
    # Decided: C01 (cells 2,3,6), C05 (cells 2,3), C15 (cells 1-4)
    # Vacuous: C04 (all), C13 (all)

    coupons_to_analyze = {
        'C01': {'label': 'Porosity (P)', 'decided_cells': [2, 3, 6], 'type': 'decided'},
        'C04': {'label': 'Sound (N)', 'decided_cells': [], 'type': 'vacuous'},
        'C05': {'label': 'Undercut (U)', 'decided_cells': [2, 3], 'type': 'decided'},
        'C15': {'label': 'Sound (N)', 'decided_cells': [1, 2, 3, 4], 'type': 'decided'},
    }

    # === Figure 1: Depth images comparison ===
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (coupon_id, info) in enumerate(coupons_to_analyze.items()):
        coupon_dir = raw_dir / f"{coupon_id}_top_r0"
        if not coupon_dir.exists():
            print(f"Warning: {coupon_dir} not found")
            continue

        depth_mm, meta = load_depth_data(coupon_dir)
        if depth_mm is None:
            continue

        ax = axes[idx]

        # Show depth image
        # Mask invalid values
        depth_display = np.where(depth_mm > 0, depth_mm, np.nan)

        im = ax.imshow(depth_display, cmap='viridis', aspect='auto',
                       vmin=np.nanpercentile(depth_display, 5),
                       vmax=np.nanpercentile(depth_display, 95))

        # Mark cell boundaries (approximate - 10 cells across image width)
        cell_width = depth_mm.shape[1] // 12  # Approximate
        for i in range(1, 12):
            x = i * cell_width
            color = 'red' if i-1 in info['decided_cells'] else 'gray'
            ax.axvline(x, color=color, linestyle='--', alpha=0.5, linewidth=1)

        ax.set_title(f"{coupon_id}: {info['label']}\n({info['type']}, decided cells: {info['decided_cells'] or 'none'})")
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        plt.colorbar(im, ax=ax, label='Depth (mm)')

    plt.suptitle('Depth Images: Decided vs Vacuous Coupons', fontsize=14)
    plt.tight_layout()
    fig1.savefig(output_dir / "f4_10_depth_images.png", bbox_inches='tight', dpi=150)
    print(f"Saved: {output_dir / 'f4_10_depth_images.png'}")

    # === Figure 2: Cross-section profiles at weld centerline ===
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes2 = axes2.flatten()

    for idx, (coupon_id, info) in enumerate(coupons_to_analyze.items()):
        coupon_dir = raw_dir / f"{coupon_id}_top_r0"
        if not coupon_dir.exists():
            continue

        depth_mm, meta = load_depth_data(coupon_dir)
        if depth_mm is None:
            continue

        ax = axes2[idx]

        # Find approximate weld centerline (should be around row 360 for 720 height)
        weld_y = depth_mm.shape[0] // 2

        # Extract multiple cross-sections at different x positions
        n_sections = 5
        cell_width = depth_mm.shape[1] // 12

        colors = plt.cm.viridis(np.linspace(0, 1, n_sections))

        for i, x_cell in enumerate([2, 4, 6, 8, 10]):
            x_pos = x_cell * cell_width
            section = extract_cross_section(depth_mm, x_pos)

            # Normalize to plate level (use edges as reference)
            plate_ref = np.nanmean([section[:50], section[-50:]])
            section_norm = section - plate_ref

            y_mm = np.arange(len(section)) * 0.445  # Approximate mm/pixel

            decided = (x_cell - 1) in info['decided_cells']
            linestyle = '-' if decided else '--'
            label = f"Cell {x_cell-1}" + (" (decided)" if decided else "")

            ax.plot(y_mm, section_norm, color=colors[i], linestyle=linestyle,
                    linewidth=2, label=label)

        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(1.0, color='red', linestyle=':', linewidth=1, label='ISO limit (1mm)')
        ax.axhline(-1.0, color='red', linestyle=':', linewidth=1)

        ax.set_xlabel('Y position (mm from top)')
        ax.set_ylabel('Depth relative to plate (mm)')
        ax.set_title(f"{coupon_id}: {info['label']} - Cross-sections")
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(-15, 15)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Weld Cross-Section Profiles: Checking for Underfill vs Misalignment', fontsize=14)
    plt.tight_layout()
    fig2.savefig(output_dir / "f4_11_cross_sections.png", bbox_inches='tight', dpi=150)
    print(f"Saved: {output_dir / 'f4_11_cross_sections.png'}")

    # === Figure 3: Side view along weld ===
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8))
    axes3 = axes3.flatten()

    for idx, (coupon_id, info) in enumerate(coupons_to_analyze.items()):
        coupon_dir = raw_dir / f"{coupon_id}_top_r0"
        if not coupon_dir.exists():
            continue

        depth_mm, meta = load_depth_data(coupon_dir)
        if depth_mm is None:
            continue

        ax = axes3[idx]

        # Extract profile along weld centerline
        weld_y = depth_mm.shape[0] // 2
        profile = extract_weld_profile(depth_mm, weld_y, profile_width=20)

        # Also get profiles slightly off-center (on the plates)
        profile_top = extract_weld_profile(depth_mm, weld_y - 100, profile_width=20)
        profile_bottom = extract_weld_profile(depth_mm, weld_y + 100, profile_width=20)

        x_mm = np.arange(len(profile)) * 0.445  # Approximate mm/pixel

        # Normalize to mean plate level
        plate_ref = np.nanmean([np.nanmean(profile_top), np.nanmean(profile_bottom)])

        ax.plot(x_mm, profile - plate_ref, 'b-', linewidth=1.5, label='Weld centerline', alpha=0.8)
        ax.plot(x_mm, profile_top - plate_ref, 'g--', linewidth=1, label='Top plate', alpha=0.6)
        ax.plot(x_mm, profile_bottom - plate_ref, 'r--', linewidth=1, label='Bottom plate', alpha=0.6)

        # Mark decided cells
        cell_width_mm = x_mm[-1] / 12
        for cell in info['decided_cells']:
            x_start = (cell + 1) * cell_width_mm
            x_end = (cell + 2) * cell_width_mm
            ax.axvspan(x_start, x_end, alpha=0.2, color='orange', label='Decided' if cell == info['decided_cells'][0] else '')

        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('X position along weld (mm)')
        ax.set_ylabel('Depth relative to plate (mm)')
        ax.set_title(f"{coupon_id}: {info['label']}")
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Side View Along Weld: Centerline vs Plate Surfaces', fontsize=14)
    plt.tight_layout()
    fig3.savefig(output_dir / "f4_12_side_view.png", bbox_inches='tight', dpi=150)
    print(f"Saved: {output_dir / 'f4_12_side_view.png'}")

    # === Figure 4: Detailed comparison of good vs problematic cell ===
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

    # Compare C04 (all vacuous, Sound) vs C15 (decided, also Sound)
    for idx, (coupon_id, title) in enumerate([('C04', 'Vacuous - All cells abstain'),
                                               ('C15', 'Decided - All cells predict U')]):
        coupon_dir = raw_dir / f"{coupon_id}_top_r0"
        if not coupon_dir.exists():
            continue

        depth_mm, meta = load_depth_data(coupon_dir)
        if depth_mm is None:
            continue

        ax = axes4[idx]

        # Show 2D slice around weld
        weld_y = depth_mm.shape[0] // 2
        y_range = 150  # pixels above/below weld center

        weld_region = depth_mm[weld_y-y_range:weld_y+y_range, :]

        # Normalize
        plate_ref = np.nanpercentile(weld_region, 95)
        weld_norm = weld_region - plate_ref

        im = ax.imshow(weld_norm, cmap='RdBu_r', aspect='auto',
                       vmin=-5, vmax=5,
                       extent=[0, weld_norm.shape[1]*0.445, y_range*0.445, -y_range*0.445])

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('X position (mm)')
        ax.set_ylabel('Y from weld center (mm)')
        ax.set_title(f"{coupon_id} (Sound): {title}")

        plt.colorbar(im, ax=ax, label='Depth relative to plate (mm)')

    plt.suptitle('Comparison: Why C04 is Vacuous but C15 Shows Misalignment', fontsize=14)
    plt.tight_layout()
    fig4.savefig(output_dir / "f4_13_vacuous_vs_decided.png", bbox_inches='tight', dpi=150)
    print(f"Saved: {output_dir / 'f4_13_vacuous_vs_decided.png'}")

    plt.close('all')
    print("\nDone. Check the figures for underfill vs misalignment diagnosis.")


if __name__ == "__main__":
    main()
