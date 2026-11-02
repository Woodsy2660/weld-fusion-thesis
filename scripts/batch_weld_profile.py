#!/usr/bin/env python3
"""Batch weld profile extraction - manually draw crop box and weld line for each coupon.

Usage:
    python scripts/batch_weld_profile.py

For each coupon:
1. Draw rectangle around weld plate (click and drag)
2. Close window
3. Click TWO points along weld centerline
4. Close window
5. Repeat for next coupon

Selections are saved to configs/weld_selections.json so you can resume.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from pathlib import Path
import json
from PIL import Image


class BatchWeldExtractor:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.raw_dir = self.base_dir / "data" / "raw" / "S1"
        self.output_dir = self.base_dir / "results" / "thesis" / "figures" / "weld_profiles"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.selections_file = self.base_dir / "configs" / "weld_selections.json"
        self.selections = self.load_selections()

        # Find all coupons
        self.coupons = self.find_coupons()
        print(f"Found {len(self.coupons)} coupons: {self.coupons}")

    def find_coupons(self):
        """Find all available coupon directories."""
        coupons = []
        for d in sorted(self.raw_dir.iterdir()):
            if d.is_dir() and d.name.startswith('C') and '_top_r0' in d.name:
                coupon_id = d.name.split('_')[0]
                coupons.append(coupon_id)
        return coupons

    def load_selections(self):
        """Load previous selections if they exist."""
        if self.selections_file.exists():
            with open(self.selections_file) as f:
                return json.load(f)
        return {}

    def save_selections(self):
        """Save selections to file."""
        self.selections_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.selections_file, 'w') as f:
            json.dump(self.selections, f, indent=2)
        print(f"Saved selections to {self.selections_file}")

    def load_data(self, coupon_id):
        """Load color and depth data for a coupon."""
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

    def get_crop_region(self, coupon_id, color_img):
        """Interactive crop region selection."""
        print(f"\n=== {coupon_id}: Draw rectangle around weld plate ===")

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(color_img)
        ax.set_title(f"{coupon_id}: Draw rectangle around weld plate\n(Click and drag, then CLOSE window)")

        crop_rect = [None]

        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            crop_rect[0] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            print(f"  Selected: {crop_rect[0]}")

        selector = RectangleSelector(ax, on_select, useblit=True,
                                    button=[1], minspanx=5, minspany=5,
                                    spancoords='pixels', interactive=True)
        plt.show()

        return crop_rect[0]

    def get_weld_line(self, coupon_id, color_img, crop_rect):
        """Interactive weld line selection."""
        print(f"=== {coupon_id}: Click TWO points along weld centerline ===")

        x1, y1, x2, y2 = crop_rect
        cropped = color_img[y1:y2, x1:x2]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(cropped)
        ax.set_title(f"{coupon_id}: Click TWO points along weld centerline\n(Start, then End, then CLOSE window)")

        line_points = []

        def on_click(event):
            if event.inaxes and len(line_points) < 2:
                px, py = int(event.xdata), int(event.ydata)
                line_points.append([px, py])
                ax.plot(px, py, 'ro', markersize=12)
                ax.annotate(f'P{len(line_points)}', (px+5, py-5), fontsize=12,
                           color='yellow', fontweight='bold')

                if len(line_points) == 2:
                    ax.plot([line_points[0][0], line_points[1][0]],
                           [line_points[0][1], line_points[1][1]],
                           'r-', linewidth=3)

                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        if len(line_points) == 2:
            # Convert to full image coordinates
            return [
                [line_points[0][0] + x1, line_points[0][1] + y1],
                [line_points[1][0] + x1, line_points[1][1] + y1]
            ]
        return None

    def extract_profiles(self, depth_mm, color_shape, weld_line, n_profiles=50, half_width=40):
        """Extract depth profiles along the weld line."""
        (lx1, ly1), (lx2, ly2) = weld_line

        dx = lx2 - lx1
        dy = ly2 - ly1
        length = np.sqrt(dx**2 + dy**2)

        along = np.array([dx, dy]) / length
        perp = np.array([-dy, dx]) / length

        t_values = np.linspace(0.05, 0.95, n_profiles)  # Avoid edges

        profiles = []
        px_to_mm = 0.25

        # Scale factors from color to depth
        scale_x = depth_mm.shape[1] / color_shape[1]
        scale_y = depth_mm.shape[0] / color_shape[0]

        for t in t_values:
            cx = lx1 + t * dx
            cy = ly1 + t * dy

            profile_depths = []
            profile_y = []

            for offset in range(-half_width, half_width + 1):
                px = int(cx + offset * perp[0])
                py = int(cy + offset * perp[1])

                dpx = int(px * scale_x)
                dpy = int(py * scale_y)

                if 0 <= dpx < depth_mm.shape[1] and 0 <= dpy < depth_mm.shape[0]:
                    d = depth_mm[dpy, dpx]
                    if d > 0 and d < 1000:  # Filter invalid
                        profile_depths.append(d)
                        profile_y.append(offset * px_to_mm)

            if len(profile_depths) > 20:
                profiles.append({
                    't': t,
                    'position_mm': t * length * px_to_mm,
                    'y_mm': np.array(profile_y),
                    'depth_mm': np.array(profile_depths)
                })

        return profiles

    def compute_groove_stats(self, profiles):
        """Compute groove depth statistics."""
        centerline_depths = []

        for p in profiles:
            y = p['y_mm']
            d = p['depth_mm']

            # Get center region
            center_mask = np.abs(y) < 3  # Within 3mm of center
            if center_mask.sum() > 5:
                # Plate reference from edges
                edge_mask = np.abs(y) > 5
                if edge_mask.sum() > 5:
                    plate_ref = np.nanmedian(d[edge_mask])
                    center_depth = np.nanmedian(d[center_mask]) - plate_ref
                    if np.abs(center_depth) < 15:  # Filter outliers
                        centerline_depths.append(center_depth)

        if len(centerline_depths) > 5:
            depths = np.array(centerline_depths)
            return {
                'mean': float(np.mean(depths)),
                'median': float(np.median(depths)),
                'std': float(np.std(depths)),
                'min': float(np.min(depths)),
                'max': float(np.max(depths)),
                'n_samples': len(depths)
            }
        return None

    def generate_visualization(self, coupon_id, profiles, stats):
        """Generate visualization for one coupon."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Cross-sections
        ax = axes[0]
        n_show = min(8, len(profiles))
        colors = plt.cm.viridis(np.linspace(0, 1, n_show))

        for i, idx in enumerate(np.linspace(0, len(profiles)-1, n_show, dtype=int)):
            p = profiles[idx]
            y = p['y_mm']
            d = p['depth_mm']

            edge_mask = np.abs(y) > 5
            if edge_mask.sum() > 5:
                plate_ref = np.nanmedian(d[edge_mask])
                d_rel = d - plate_ref
                ax.plot(y, d_rel, color=colors[i], linewidth=1.5, alpha=0.7)

        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axhline(-5, color='red', linestyle='--', alpha=0.5, label='-5mm')
        ax.set_xlabel('Distance from weld center (mm)')
        ax.set_ylabel('Depth relative to plate (mm)')
        ax.set_title(f'{coupon_id}: Cross-Sections')
        ax.set_ylim(-8, 4)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Right: Stats summary
        ax = axes[1]
        ax.axis('off')

        if stats:
            text = f"""
{coupon_id} Weld Groove Statistics
{'='*35}

Mean depth:   {stats['mean']:+.2f} mm
Median depth: {stats['median']:+.2f} mm
Std dev:      {stats['std']:.2f} mm

Min (deepest):    {stats['min']:+.2f} mm
Max (shallowest): {stats['max']:+.2f} mm

Samples: {stats['n_samples']}

Interpretation:
{'─'*35}
"""
            if stats['median'] < -3:
                text += "SIGNIFICANT UNDERFILL\n"
                text += f"Groove depth ~{abs(stats['median']):.1f}mm"
            elif stats['median'] < -1:
                text += "MODERATE UNDERFILL\n"
                text += f"Groove depth ~{abs(stats['median']):.1f}mm"
            elif stats['median'] > 1:
                text += "OVERFILL/REINFORCEMENT\n"
                text += f"Cap height ~{stats['median']:.1f}mm"
            else:
                text += "NEAR FLUSH WITH PLATE\n"
                text += "Well-filled weld"

            ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        fig.savefig(self.output_dir / f"{coupon_id}_profile.png", dpi=150)
        plt.close()
        print(f"  Saved: {self.output_dir / f'{coupon_id}_profile.png'}")

    def process_coupon(self, coupon_id):
        """Process a single coupon."""
        print(f"\n{'='*50}")
        print(f"Processing {coupon_id}")
        print(f"{'='*50}")

        color_img, depth_mm = self.load_data(coupon_id)
        if color_img is None or depth_mm is None:
            print(f"  Error: Could not load data for {coupon_id}")
            return None

        # Check if we have saved selection
        if coupon_id in self.selections:
            print(f"  Using saved selection for {coupon_id}")
            sel = self.selections[coupon_id]
            crop_rect = sel['crop_rect']
            weld_line = sel['weld_line']
        else:
            # Get user input
            crop_rect = self.get_crop_region(coupon_id, color_img)
            if crop_rect is None:
                print(f"  Skipped {coupon_id} - no crop region")
                return None

            weld_line = self.get_weld_line(coupon_id, color_img, crop_rect)
            if weld_line is None:
                print(f"  Skipped {coupon_id} - no weld line")
                return None

            # Save selection
            self.selections[coupon_id] = {
                'crop_rect': crop_rect,
                'weld_line': weld_line
            }
            self.save_selections()

        # Extract profiles
        profiles = self.extract_profiles(depth_mm, color_img.shape, weld_line)
        print(f"  Extracted {len(profiles)} profiles")

        # Compute stats
        stats = self.compute_groove_stats(profiles)
        if stats:
            print(f"  Groove depth: {stats['median']:.2f} mm (median)")
            self.selections[coupon_id]['stats'] = stats
            self.save_selections()

        # Generate visualization
        self.generate_visualization(coupon_id, profiles, stats)

        return stats

    def generate_summary(self):
        """Generate summary comparison of all coupons."""
        print("\n" + "="*60)
        print("SUMMARY: All Coupon Weld Groove Depths")
        print("="*60)

        # Load labels for ground truth
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

        results = []
        for coupon_id in self.coupons:
            if coupon_id in self.selections and 'stats' in self.selections[coupon_id]:
                stats = self.selections[coupon_id]['stats']
                condition = coupon_conditions.get(coupon_id, 'Unknown')
                results.append({
                    'coupon': coupon_id,
                    'condition': condition,
                    'median_depth': stats['median'],
                    'std': stats['std']
                })

        if results:
            # Sort by median depth
            results.sort(key=lambda x: x['median_depth'])

            print(f"\n{'Coupon':<8} {'Condition':<12} {'Median Depth':<14} {'Std Dev':<10} {'Fill Status'}")
            print("-"*60)
            for r in results:
                if r['median_depth'] < -3:
                    status = "UNDERFILL"
                elif r['median_depth'] < -1:
                    status = "Slight underfill"
                elif r['median_depth'] > 1:
                    status = "Overfill"
                else:
                    status = "Well-filled"

                print(f"{r['coupon']:<8} {r['condition']:<12} {r['median_depth']:>+8.2f} mm    {r['std']:>6.2f} mm   {status}")

            # Create summary figure
            fig, ax = plt.subplots(figsize=(12, 6))

            coupons = [r['coupon'] for r in results]
            depths = [r['median_depth'] for r in results]
            stds = [r['std'] for r in results]
            conditions = [r['condition'] for r in results]

            colors = {'Sound': 'green', 'Porosity': 'blue', 'Undercut': 'red',
                     'LOF': 'purple', 'Overfill': 'orange', 'Misaligned': 'brown'}
            bar_colors = [colors.get(c, 'gray') for c in conditions]

            bars = ax.bar(coupons, depths, yerr=stds, capsize=3, color=bar_colors, alpha=0.7)
            ax.axhline(0, color='black', linewidth=1)
            ax.axhline(-5, color='red', linestyle='--', alpha=0.5, label='5mm underfill')

            ax.set_xlabel('Coupon')
            ax.set_ylabel('Weld Groove Depth (mm)')
            ax.set_title('Weld Fill Level by Coupon\n(Negative = underfill, Positive = overfill)')
            ax.legend()

            # Color legend
            for cond, color in colors.items():
                ax.bar([], [], color=color, label=cond, alpha=0.7)
            ax.legend(loc='upper right', ncol=2)

            plt.tight_layout()
            fig.savefig(self.output_dir / "summary_all_coupons.png", dpi=150)
            print(f"\nSaved summary: {self.output_dir / 'summary_all_coupons.png'}")
            plt.close()

    def run(self):
        """Run batch processing."""
        print("\n" + "="*60)
        print("Batch Weld Profile Extraction")
        print("="*60)
        print(f"\nCoupons to process: {self.coupons}")
        print(f"Already processed: {list(self.selections.keys())}")

        remaining = [c for c in self.coupons if c not in self.selections]
        print(f"Remaining: {remaining}")

        if remaining:
            print("\nFor each coupon:")
            print("  1. Draw rectangle around weld plate")
            print("  2. Close window")
            print("  3. Click TWO points on weld centerline")
            print("  4. Close window")
            print("\nPress Ctrl+C to stop and save progress.\n")

        try:
            for coupon_id in self.coupons:
                self.process_coupon(coupon_id)
        except KeyboardInterrupt:
            print("\n\nInterrupted - progress saved.")
            self.save_selections()

        self.generate_summary()


def main():
    extractor = BatchWeldExtractor()
    extractor.run()


if __name__ == "__main__":
    main()
