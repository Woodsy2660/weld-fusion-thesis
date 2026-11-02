#!/usr/bin/env python3
"""Extract depth data from a band around the weld line.

Uses the saved weld line selections and extracts a wider band
to capture the weld toes and fusion zones where LOF defects occur.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import json
from PIL import Image


class WeldBandExtractor:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.raw_dir = self.base_dir / "data" / "raw" / "S1"
        self.output_dir = self.base_dir / "results" / "thesis" / "figures" / "weld_bands"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load saved selections
        self.selections_file = self.base_dir / "configs" / "weld_selections.json"
        with open(self.selections_file) as f:
            self.selections = json.load(f)

        # Load labels for ground truth
        self.labels = self.load_labels()

    def load_labels(self):
        """Load ground truth labels."""
        labels_file = self.base_dir / "results" / "thesis" / "data" / "labels_v2_2026-10-26.csv"
        coupon_conditions = {}
        if labels_file.exists():
            import csv
            with open(labels_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row['coupon_id']
                    if cid not in coupon_conditions:
                        coupon_conditions[cid] = row['condition']
        return coupon_conditions

    def load_data(self, coupon_id):
        """Load color and depth data."""
        coupon_dir = self.raw_dir / f"{coupon_id}_top_r0"

        # Load color
        color_files = sorted(coupon_dir.glob("color/*.png"))
        color_img = np.array(Image.open(color_files[0])) if color_files else None

        # Load depth
        depth_files = sorted(coupon_dir.glob("depth/*.npy"))
        if depth_files:
            frames = [np.load(f) for f in depth_files[:10]]
            depth_avg = np.nanmean(frames, axis=0)

            meta_path = coupon_dir / "meta.json"
            with open(meta_path) as f:
                meta = json.load(f)

            depth_scale = meta.get('depth_scale_m', 0.0001)
            depth_mm = depth_avg * depth_scale * 1000
        else:
            depth_mm = None

        return color_img, depth_mm

    def extract_band(self, depth_mm, color_shape, weld_line, band_half_width_mm=15):
        """
        Extract a band of depth data around the weld line.

        Returns a 2D array where:
        - X axis = position along weld
        - Y axis = distance from weld centerline (negative = left, positive = right)
        """
        (lx1, ly1), (lx2, ly2) = weld_line

        dx = lx2 - lx1
        dy = ly2 - ly1
        length = np.sqrt(dx**2 + dy**2)

        # Unit vectors
        along = np.array([dx, dy]) / length  # Along weld
        perp = np.array([-dy, dx]) / length  # Perpendicular to weld

        # Scale factors
        scale_x = depth_mm.shape[1] / color_shape[1]
        scale_y = depth_mm.shape[0] / color_shape[0]

        # Pixel size in mm (approximate)
        px_to_mm = 0.25

        # Band dimensions
        band_half_width_px = int(band_half_width_mm / px_to_mm)
        n_along = int(length)  # One sample per pixel along weld

        # Create output array
        band_depth = np.full((2 * band_half_width_px + 1, n_along), np.nan)
        y_coords = np.linspace(-band_half_width_mm, band_half_width_mm, 2 * band_half_width_px + 1)
        x_coords = np.linspace(0, length * px_to_mm, n_along)

        # Extract depth values
        for i_along in range(n_along):
            t = i_along / n_along
            cx = lx1 + t * dx
            cy = ly1 + t * dy

            for i_perp, offset_px in enumerate(range(-band_half_width_px, band_half_width_px + 1)):
                px = int(cx + offset_px * perp[0])
                py = int(cy + offset_px * perp[1])

                dpx = int(px * scale_x)
                dpy = int(py * scale_y)

                if 0 <= dpx < depth_mm.shape[1] and 0 <= dpy < depth_mm.shape[0]:
                    d = depth_mm[dpy, dpx]
                    if 0 < d < 1000:  # Filter invalid
                        band_depth[i_perp, i_along] = d

        return band_depth, x_coords, y_coords

    def analyze_band(self, band_depth, y_coords, x_coords):
        """Analyze the band for depth variations, especially at toes."""
        # Define regions
        center_mask = np.abs(y_coords) < 3  # Center ±3mm
        left_toe_mask = (y_coords > -12) & (y_coords < -5)  # Left toe region
        right_toe_mask = (y_coords > 5) & (y_coords < 12)  # Right toe region
        plate_left_mask = y_coords < -12  # Left plate
        plate_right_mask = y_coords > 12  # Right plate

        stats = {}

        # Get plate reference (median of both plate regions)
        plate_left = band_depth[plate_left_mask, :]
        plate_right = band_depth[plate_right_mask, :]
        plate_ref = np.nanmedian([np.nanmedian(plate_left), np.nanmedian(plate_right)])

        # Normalize to plate
        band_norm = band_depth - plate_ref

        # Center (crown) statistics
        center_data = band_norm[center_mask, :]
        stats['center_median'] = float(np.nanmedian(center_data))
        stats['center_min'] = float(np.nanmin(center_data))
        stats['center_max'] = float(np.nanmax(center_data))

        # Left toe statistics
        left_toe_data = band_norm[left_toe_mask, :]
        stats['left_toe_median'] = float(np.nanmedian(left_toe_data))
        stats['left_toe_min'] = float(np.nanmin(left_toe_data))

        # Right toe statistics
        right_toe_data = band_norm[right_toe_mask, :]
        stats['right_toe_median'] = float(np.nanmedian(right_toe_data))
        stats['right_toe_min'] = float(np.nanmin(right_toe_data))

        # Asymmetry (difference between left and right plate)
        left_plate_med = np.nanmedian(band_depth[plate_left_mask, :])
        right_plate_med = np.nanmedian(band_depth[plate_right_mask, :])
        stats['plate_asymmetry'] = float(abs(left_plate_med - right_plate_med))

        # Worst undercut at toes
        stats['worst_toe_depth'] = min(stats['left_toe_min'], stats['right_toe_min'])

        return stats, band_norm, plate_ref

    def generate_visualization(self, coupon_id, band_depth, band_norm, x_coords, y_coords, stats):
        """Generate comprehensive band visualization."""
        condition = self.labels.get(coupon_id, 'Unknown')

        fig = plt.figure(figsize=(16, 10))

        # 1. Top left: 2D heatmap of normalized depth
        ax1 = fig.add_subplot(2, 2, 1)
        extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]
        im = ax1.imshow(band_norm, aspect='auto', cmap='RdBu_r',
                       vmin=-5, vmax=3, extent=extent)
        ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axhline(-5, color='yellow', linestyle=':', linewidth=1, alpha=0.7)
        ax1.axhline(5, color='yellow', linestyle=':', linewidth=1, alpha=0.7)
        ax1.axhline(-12, color='green', linestyle=':', linewidth=1, alpha=0.7)
        ax1.axhline(12, color='green', linestyle=':', linewidth=1, alpha=0.7)

        ax1.set_xlabel('Position along weld (mm)')
        ax1.set_ylabel('Distance from centerline (mm)')
        ax1.set_title(f'{coupon_id} ({condition}): Depth Band\n(Blue=below plate, Red=above plate)')
        plt.colorbar(im, ax=ax1, label='Depth rel. to plate (mm)')

        # Add region labels
        ax1.text(x_coords[-1]*0.02, 0, 'CENTER', fontsize=8, color='black', va='center')
        ax1.text(x_coords[-1]*0.02, -8, 'L TOE', fontsize=8, color='black', va='center')
        ax1.text(x_coords[-1]*0.02, 8, 'R TOE', fontsize=8, color='black', va='center')

        # 2. Top right: Cross-section profiles at multiple positions
        ax2 = fig.add_subplot(2, 2, 2)
        n_profiles = 8
        colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))

        for i, idx in enumerate(np.linspace(0, len(x_coords)-1, n_profiles, dtype=int)):
            profile = band_norm[:, idx]
            ax2.plot(y_coords, profile, color=colors[i], linewidth=1.5, alpha=0.7,
                    label=f's={x_coords[idx]:.0f}mm')

        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.axvline(-5, color='yellow', linestyle=':', alpha=0.5)
        ax2.axvline(5, color='yellow', linestyle=':', alpha=0.5)
        ax2.axvline(-12, color='green', linestyle=':', alpha=0.5)
        ax2.axvline(12, color='green', linestyle=':', alpha=0.5)

        ax2.fill_between([-12, -5], -8, 4, alpha=0.1, color='red', label='Left toe zone')
        ax2.fill_between([5, 12], -8, 4, alpha=0.1, color='red', label='Right toe zone')

        ax2.set_xlabel('Distance from weld center (mm)')
        ax2.set_ylabel('Depth relative to plate (mm)')
        ax2.set_title('Cross-Section Profiles')
        ax2.set_xlim(-15, 15)
        ax2.set_ylim(-6, 4)
        ax2.legend(fontsize=7, loc='upper right', ncol=2)
        ax2.grid(True, alpha=0.3)

        # 3. Bottom left: Longitudinal profiles (along weld at different Y positions)
        ax3 = fig.add_subplot(2, 2, 3)

        # Center profile
        center_idx = len(y_coords) // 2
        ax3.plot(x_coords, band_norm[center_idx, :], 'b-', linewidth=1.5, label='Center (y=0)')

        # Left toe profile (y ≈ -8mm)
        left_toe_idx = np.argmin(np.abs(y_coords - (-8)))
        ax3.plot(x_coords, band_norm[left_toe_idx, :], 'r-', linewidth=1.5, label='Left toe (y=-8mm)')

        # Right toe profile (y ≈ +8mm)
        right_toe_idx = np.argmin(np.abs(y_coords - 8))
        ax3.plot(x_coords, band_norm[right_toe_idx, :], 'g-', linewidth=1.5, label='Right toe (y=+8mm)')

        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.axhline(-1, color='orange', linestyle='--', alpha=0.5, label='-1mm (undercut limit)')

        ax3.set_xlabel('Position along weld (mm)')
        ax3.set_ylabel('Depth relative to plate (mm)')
        ax3.set_title('Longitudinal Profiles: Center vs Toes')
        ax3.legend(fontsize=8)
        ax3.set_ylim(-5, 3)
        ax3.grid(True, alpha=0.3)

        # 4. Bottom right: Statistics summary
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        text = f"""
{coupon_id} Weld Band Analysis
{'='*40}
Condition: {condition}

CROWN (Center ±3mm):
  Median:  {stats['center_median']:+.2f} mm
  Min:     {stats['center_min']:+.2f} mm
  Max:     {stats['center_max']:+.2f} mm

LEFT TOE (-12 to -5mm):
  Median:  {stats['left_toe_median']:+.2f} mm
  Min:     {stats['left_toe_min']:+.2f} mm

RIGHT TOE (+5 to +12mm):
  Median:  {stats['right_toe_median']:+.2f} mm
  Min:     {stats['right_toe_min']:+.2f} mm

PLATE ASYMMETRY: {stats['plate_asymmetry']:.2f} mm
WORST TOE DEPTH: {stats['worst_toe_depth']:+.2f} mm

{'─'*40}
INTERPRETATION:
"""
        # Add interpretation
        if stats['worst_toe_depth'] < -1.5:
            text += "⚠️  SIGNIFICANT UNDERCUT at toe\n"
            text += f"   Depth {abs(stats['worst_toe_depth']):.1f}mm exceeds 1mm limit"
        elif stats['worst_toe_depth'] < -0.5:
            text += "⚡ Moderate undercut at toe\n"
            text += f"   Depth {abs(stats['worst_toe_depth']):.1f}mm"
        elif stats['plate_asymmetry'] > 1.0:
            text += "📐 Plate misalignment detected\n"
            text += f"   Asymmetry {stats['plate_asymmetry']:.1f}mm"
        elif abs(stats['center_median']) > 2:
            if stats['center_median'] < 0:
                text += "📉 Underfill at crown\n"
            else:
                text += "📈 Excess reinforcement\n"
        else:
            text += "✓ Geometry within normal range"

        ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        fig.savefig(self.output_dir / f"{coupon_id}_band.png", dpi=150)
        plt.close()
        print(f"  Saved: {self.output_dir / f'{coupon_id}_band.png'}")

        return stats

    def process_coupon(self, coupon_id):
        """Process a single coupon."""
        if coupon_id not in self.selections:
            print(f"  No selection for {coupon_id}")
            return None

        sel = self.selections[coupon_id]
        weld_line = sel['weld_line']

        color_img, depth_mm = self.load_data(coupon_id)
        if color_img is None or depth_mm is None:
            print(f"  Could not load data for {coupon_id}")
            return None

        # Extract band (15mm on each side of centerline)
        band_depth, x_coords, y_coords = self.extract_band(
            depth_mm, color_img.shape, weld_line, band_half_width_mm=15
        )

        # Analyze
        stats, band_norm, plate_ref = self.analyze_band(band_depth, y_coords, x_coords)

        # Visualize
        self.generate_visualization(coupon_id, band_depth, band_norm, x_coords, y_coords, stats)

        return stats

    def run(self):
        """Process all coupons."""
        print("\n" + "="*60)
        print("Weld Band Extraction - Including Toe Regions")
        print("="*60)

        all_stats = {}

        for coupon_id in sorted(self.selections.keys()):
            print(f"\nProcessing {coupon_id}...")
            stats = self.process_coupon(coupon_id)
            if stats:
                all_stats[coupon_id] = stats

        # Generate summary
        self.generate_summary(all_stats)

        return all_stats

    def generate_summary(self, all_stats):
        """Generate summary comparing all coupons."""
        print("\n" + "="*60)
        print("SUMMARY: Weld Band Analysis - All Coupons")
        print("="*60)

        print(f"\n{'Coupon':<8} {'Condition':<12} {'Crown':<10} {'L Toe':<10} {'R Toe':<10} {'Asymm':<8} {'Status'}")
        print("-"*75)

        results = []
        for coupon_id, stats in sorted(all_stats.items()):
            condition = self.labels.get(coupon_id, '?')
            crown = stats['center_median']
            l_toe = stats['left_toe_min']
            r_toe = stats['right_toe_min']
            asymm = stats['plate_asymmetry']
            worst_toe = stats['worst_toe_depth']

            if worst_toe < -1.5:
                status = "UNDERCUT"
            elif asymm > 1.0:
                status = "MISALIGN"
            elif crown < -1:
                status = "Underfill"
            elif crown > 2:
                status = "Overfill"
            else:
                status = "OK"

            print(f"{coupon_id:<8} {condition:<12} {crown:>+7.2f}mm {l_toe:>+7.2f}mm {r_toe:>+7.2f}mm {asymm:>6.2f}mm  {status}")

            results.append({
                'coupon': coupon_id,
                'condition': condition,
                'crown': crown,
                'left_toe': l_toe,
                'right_toe': r_toe,
                'worst_toe': worst_toe,
                'asymmetry': asymm
            })

        # Summary figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        coupons = [r['coupon'] for r in results]
        x = np.arange(len(coupons))

        # Left plot: Crown vs Toe depths
        ax = axes[0]
        width = 0.25
        ax.bar(x - width, [r['crown'] for r in results], width, label='Crown', color='blue', alpha=0.7)
        ax.bar(x, [r['left_toe'] for r in results], width, label='Left Toe', color='red', alpha=0.7)
        ax.bar(x + width, [r['right_toe'] for r in results], width, label='Right Toe', color='green', alpha=0.7)

        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(-1, color='orange', linestyle='--', label='Undercut limit')

        ax.set_xticks(x)
        ax.set_xticklabels(coupons, rotation=45)
        ax.set_ylabel('Depth relative to plate (mm)')
        ax.set_title('Crown vs Toe Depths by Coupon')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Right plot: Plate asymmetry
        ax = axes[1]
        colors = {'Sound': 'green', 'Porosity': 'blue', 'Undercut': 'red',
                 'LOF': 'purple', 'Overfill': 'orange', 'Misaligned': 'brown'}
        bar_colors = [colors.get(r['condition'], 'gray') for r in results]

        ax.bar(x, [r['asymmetry'] for r in results], color=bar_colors, alpha=0.7)
        ax.axhline(1.0, color='red', linestyle='--', label='1mm limit')

        ax.set_xticks(x)
        ax.set_xticklabels(coupons, rotation=45)
        ax.set_ylabel('Plate Asymmetry (mm)')
        ax.set_title('Plate Height Asymmetry by Coupon\n(Potential misalignment indicator)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(self.output_dir / "summary_band_analysis.png", dpi=150)
        print(f"\nSaved: {self.output_dir / 'summary_band_analysis.png'}")
        plt.close()


def main():
    extractor = WeldBandExtractor()
    extractor.run()


if __name__ == "__main__":
    main()
