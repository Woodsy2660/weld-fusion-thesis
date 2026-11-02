#!/usr/bin/env python3
"""Interactive weld profile extraction tool.

Usage:
1. Run the script - it will show the color image of C04
2. Draw a rectangle to crop the weld plate region (click and drag)
3. Close the crop window
4. Draw a line along the weld centerline (click two points)
5. Close the line window
6. The script will extract depth profiles and generate visualizations

Run with: python3 scripts/interactive_weld_profile.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.patches import Rectangle
from pathlib import Path
import json
from PIL import Image
import pickle


class WeldProfileExtractor:
    def __init__(self, coupon_id="C04"):
        self.base_dir = Path(__file__).parent.parent
        self.raw_dir = self.base_dir / "data" / "raw" / "S1"
        self.output_dir = self.base_dir / "results" / "thesis" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.coupon_id = coupon_id
        self.coupon_dir = self.raw_dir / f"{coupon_id}_top_r0"

        # Data
        self.color_img = None
        self.depth_mm = None
        self.meta = None

        # Selection results
        self.crop_rect = None  # (x1, y1, x2, y2)
        self.weld_line = None  # ((x1, y1), (x2, y2))

        self.load_data()

    def load_data(self):
        """Load color and depth data."""
        # Load color
        color_files = sorted(self.coupon_dir.glob("color/*.png"))
        if color_files:
            self.color_img = np.array(Image.open(color_files[0]))
            print(f"Loaded color image: {self.color_img.shape}")

        # Load depth
        depth_files = sorted(self.coupon_dir.glob("depth/*.npy"))
        if depth_files:
            frames = [np.load(f) for f in depth_files[:10]]
            depth_avg = np.nanmean(frames, axis=0)

            # Load meta for scale
            meta_path = self.coupon_dir / "meta.json"
            with open(meta_path) as f:
                self.meta = json.load(f)

            depth_scale = self.meta.get('depth_scale_m', 0.0001)
            self.depth_mm = depth_avg * depth_scale * 1000
            print(f"Loaded depth data: {self.depth_mm.shape}")

    def step1_crop_region(self):
        """Show image and let user draw crop rectangle."""
        print("\n=== STEP 1: Draw a rectangle around the weld plate region ===")
        print("Click and drag to draw rectangle, then close the window.")

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(self.color_img)
        ax.set_title(f"{self.coupon_id}: Draw rectangle around weld plate region\n(Click and drag, then close window)")

        self.crop_rect = None

        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            self.crop_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            print(f"Selected region: {self.crop_rect}")

        selector = RectangleSelector(ax, on_select, useblit=True,
                                     button=[1], minspanx=5, minspany=5,
                                     spancoords='pixels', interactive=True)
        plt.show()

        if self.crop_rect is None:
            print("No region selected, using full image")
            h, w = self.color_img.shape[:2]
            self.crop_rect = (0, 0, w, h)

        return self.crop_rect

    def step2_draw_weld_line(self):
        """Show cropped image and let user draw weld line."""
        print("\n=== STEP 2: Click TWO points to draw the weld centerline ===")
        print("Click start point, then end point, then close the window.")

        x1, y1, x2, y2 = self.crop_rect
        cropped = self.color_img[y1:y2, x1:x2]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(cropped)
        ax.set_title(f"{self.coupon_id}: Click TWO points along the weld centerline\n(Start point, then end point)")

        self.line_points = []

        def on_click(event):
            if event.inaxes and len(self.line_points) < 2:
                px, py = int(event.xdata), int(event.ydata)
                self.line_points.append((px, py))
                ax.plot(px, py, 'ro', markersize=10)
                ax.annotate(f'P{len(self.line_points)}', (px, py), fontsize=12,
                           color='red', fontweight='bold')

                if len(self.line_points) == 2:
                    ax.plot([self.line_points[0][0], self.line_points[1][0]],
                           [self.line_points[0][1], self.line_points[1][1]],
                           'r-', linewidth=2)
                    print(f"Weld line: {self.line_points[0]} -> {self.line_points[1]}")

                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        if len(self.line_points) == 2:
            # Convert to full image coordinates
            self.weld_line = (
                (self.line_points[0][0] + x1, self.line_points[0][1] + y1),
                (self.line_points[1][0] + x1, self.line_points[1][1] + y1)
            )
        else:
            print("Warning: Need exactly 2 points. Using default horizontal line.")
            cy = (y1 + y2) // 2
            self.weld_line = ((x1, cy), (x2, cy))

        return self.weld_line

    def extract_profiles_along_line(self, n_profiles=50, half_width=30):
        """Extract depth profiles perpendicular to the weld line."""
        (lx1, ly1), (lx2, ly2) = self.weld_line

        # Direction along weld
        dx = lx2 - lx1
        dy = ly2 - ly1
        length = np.sqrt(dx**2 + dy**2)

        # Unit vectors
        along = np.array([dx, dy]) / length  # Along weld
        perp = np.array([-dy, dx]) / length  # Perpendicular to weld

        # Sample points along the weld line
        t_values = np.linspace(0, 1, n_profiles)

        profiles = []
        positions_mm = []

        # Pixel to mm conversion (approximate)
        px_to_mm = 0.25  # Adjust based on your camera setup

        for t in t_values:
            # Center point on weld line
            cx = lx1 + t * dx
            cy = ly1 + t * dy

            # Extract perpendicular profile
            profile_depths = []
            profile_y = []

            for offset in range(-half_width, half_width + 1):
                px = int(cx + offset * perp[0])
                py = int(cy + offset * perp[1])

                # Check bounds for depth image (need to scale from color to depth)
                # Color is 1920x1080, depth is 1280x720
                scale_x = self.depth_mm.shape[1] / self.color_img.shape[1]
                scale_y = self.depth_mm.shape[0] / self.color_img.shape[0]

                dpx = int(px * scale_x)
                dpy = int(py * scale_y)

                if 0 <= dpx < self.depth_mm.shape[1] and 0 <= dpy < self.depth_mm.shape[0]:
                    d = self.depth_mm[dpy, dpx]
                    if d > 0:
                        profile_depths.append(d)
                        profile_y.append(offset * px_to_mm)

            if len(profile_depths) > 10:
                profiles.append({
                    't': t,
                    'position_mm': t * length * px_to_mm,
                    'y_mm': np.array(profile_y),
                    'depth_mm': np.array(profile_depths)
                })
                positions_mm.append(t * length * px_to_mm)

        return profiles, positions_mm

    def generate_visualizations(self, profiles, positions_mm):
        """Generate depth profile visualizations."""
        print("\n=== Generating visualizations ===")

        # === Figure 1: Multiple cross-sections ===
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        n_show = min(10, len(profiles))
        colors = plt.cm.viridis(np.linspace(0, 1, n_show))

        for i, idx in enumerate(np.linspace(0, len(profiles)-1, n_show, dtype=int)):
            p = profiles[idx]
            y = p['y_mm']
            d = p['depth_mm']

            # Normalize to plate level (edges)
            plate_ref = np.nanmean([np.nanmean(d[:5]), np.nanmean(d[-5:])])
            d_rel = d - plate_ref

            ax1.plot(y, d_rel, color=colors[i], linewidth=1.5,
                    label=f"s={p['position_mm']:.0f}mm")

        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.axhline(-5, color='red', linestyle='--', linewidth=1, label='5mm underfill')
        ax1.axhline(-6, color='orange', linestyle='--', linewidth=1, label='6mm underfill')

        ax1.set_xlabel('Distance from weld center (mm)', fontsize=11)
        ax1.set_ylabel('Depth relative to plate surface (mm)', fontsize=11)
        ax1.set_title(f'{self.coupon_id}: Weld Cross-Section Profiles\n(Negative = below plate surface = groove/underfill)', fontsize=12)
        ax1.legend(fontsize=8, loc='upper right', ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-10, 5)

        plt.tight_layout()
        fig1.savefig(self.output_dir / f"weld_profiles_{self.coupon_id}.png", dpi=150)
        print(f"Saved: {self.output_dir / f'weld_profiles_{self.coupon_id}.png'}")

        # === Figure 2: 2D heatmap along weld ===
        fig2, ax2 = plt.subplots(figsize=(14, 6))

        # Build 2D array
        all_y = profiles[0]['y_mm']
        heatmap = np.full((len(profiles), len(all_y)), np.nan)

        for i, p in enumerate(profiles):
            plate_ref = np.nanmean([np.nanmean(p['depth_mm'][:5]), np.nanmean(p['depth_mm'][-5:])])
            d_rel = p['depth_mm'] - plate_ref
            # Interpolate to common y grid
            heatmap[i, :len(d_rel)] = d_rel[:len(all_y)]

        extent = [all_y[0], all_y[-1], positions_mm[-1], positions_mm[0]]
        im = ax2.imshow(heatmap, aspect='auto', cmap='RdBu_r',
                       vmin=-8, vmax=4, extent=extent)

        ax2.set_xlabel('Distance from weld center (mm)', fontsize=11)
        ax2.set_ylabel('Position along weld (mm)', fontsize=11)
        ax2.set_title(f'{self.coupon_id}: Weld Groove Depth Along Weld Line\n(Blue = groove/underfill, Red = above plate)', fontsize=12)
        plt.colorbar(im, ax=ax2, label='Depth relative to plate (mm)')

        plt.tight_layout()
        fig2.savefig(self.output_dir / f"weld_heatmap_{self.coupon_id}.png", dpi=150)
        print(f"Saved: {self.output_dir / f'weld_heatmap_{self.coupon_id}.png'}")

        # === Figure 3: Centerline depth profile ===
        fig3, ax3 = plt.subplots(figsize=(12, 5))

        centerline_depths = []
        for p in profiles:
            # Get depth at center (y=0)
            center_idx = len(p['y_mm']) // 2
            if center_idx < len(p['depth_mm']):
                plate_ref = np.nanmean([np.nanmean(p['depth_mm'][:5]), np.nanmean(p['depth_mm'][-5:])])
                centerline_depths.append(p['depth_mm'][center_idx] - plate_ref)
            else:
                centerline_depths.append(np.nan)

        ax3.plot(positions_mm, centerline_depths, 'b-', linewidth=2)
        ax3.fill_between(positions_mm, centerline_depths, 0,
                        where=np.array(centerline_depths) < 0, alpha=0.3, color='blue',
                        label='Underfill region')

        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.axhline(-5, color='red', linestyle='--', linewidth=1, label='5mm depth')
        ax3.axhline(-6, color='orange', linestyle='--', linewidth=1, label='6mm depth')

        ax3.set_xlabel('Position along weld (mm)', fontsize=11)
        ax3.set_ylabel('Weld groove depth (mm)', fontsize=11)
        ax3.set_title(f'{self.coupon_id}: Weld Centerline Depth Profile\n(Negative = below plate = underfill depth)', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-10, 2)

        plt.tight_layout()
        fig3.savefig(self.output_dir / f"weld_centerline_{self.coupon_id}.png", dpi=150)
        print(f"Saved: {self.output_dir / f'weld_centerline_{self.coupon_id}.png'}")

        # === Print statistics ===
        valid_depths = [d for d in centerline_depths if np.isfinite(d)]
        if valid_depths:
            print(f"\n=== Weld Groove Statistics ({self.coupon_id}) ===")
            print(f"Mean groove depth: {np.mean(valid_depths):.2f} mm")
            print(f"Min (deepest): {np.min(valid_depths):.2f} mm")
            print(f"Max (shallowest): {np.max(valid_depths):.2f} mm")
            print(f"Std dev: {np.std(valid_depths):.2f} mm")

            # Estimate fill level
            plate_thickness = 10.0  # mm
            mean_depth = abs(np.mean(valid_depths))
            fill_level = plate_thickness - mean_depth
            print(f"\nEstimated fill level: {fill_level:.1f} mm of {plate_thickness} mm plate")
            print(f"Fill percentage: {100*fill_level/plate_thickness:.0f}%")

        plt.close('all')
        return True

    def run(self):
        """Run the full extraction pipeline."""
        print(f"\n{'='*60}")
        print(f"Interactive Weld Profile Extractor - {self.coupon_id}")
        print(f"{'='*60}")

        if self.color_img is None or self.depth_mm is None:
            print("Error: Could not load data")
            return

        # Step 1: Crop region
        self.step1_crop_region()

        # Step 2: Draw weld line
        self.step2_draw_weld_line()

        # Step 3: Extract profiles
        print("\n=== STEP 3: Extracting depth profiles ===")
        profiles, positions_mm = self.extract_profiles_along_line()
        print(f"Extracted {len(profiles)} profiles")

        # Step 4: Generate visualizations
        self.generate_visualizations(profiles, positions_mm)

        print("\n=== Done! ===")


def main():
    import sys
    coupon_id = sys.argv[1] if len(sys.argv) > 1 else "C04"

    extractor = WeldProfileExtractor(coupon_id=coupon_id)
    extractor.run()


if __name__ == "__main__":
    main()
