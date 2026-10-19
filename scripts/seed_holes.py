#!/usr/bin/env python3
"""
Manual hole seeding tool.

Task 1b Part B6: Click-based hole seeding as the PRIMARY registration path.

Usage:
    python scripts/seed_holes.py
    python scripts/seed_holes.py --start C07_top_r0   # Resume from specific capture

Two-step process:
1. View full image, draw rectangle around the weld plate (drag corners)
2. Zoomed view of selected region - click Ø6 then Ø8 holes

Controls:
  ENTER - confirm
  R     - reset/redraw
  B     - go back (from hole clicking to ROI selection)
  S     - skip this capture
  Q     - quit and save progress

The clicks are stored in FULL-FRAME pixel coordinates.
"""

import sys
import json
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.geometry.registration_v2 import compute_median_colour, compute_averaged_depth


# Global state for ROI selection
roi_state = {
    "drawing": False,
    "start": None,
    "end": None,
    "done": False,
}

# Global state for hole clicking
click_state = {
    "clicks": [],
}


def roi_mouse_handler(event, x, y, flags, param):
    """Handle mouse events for ROI selection."""
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_state["drawing"] = True
        roi_state["start"] = (x, y)
        roi_state["end"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and roi_state["drawing"]:
        roi_state["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_state["drawing"] = False
        roi_state["end"] = (x, y)


def click_handler(event, x, y, flags, param):
    """Handle mouse clicks for hole selection."""
    if event == cv2.EVENT_LBUTTONDOWN:
        click_state["clicks"].append((x, y))
        print(f"  Click {len(click_state['clicks'])}: ({x}, {y})")


def select_roi(
    colour_img: np.ndarray,
    capture_name: str,
    max_width: int = 1200,
    max_height: int = 800,
) -> tuple[int, int, int, int] | str:
    """Display full image and let user draw ROI rectangle.

    Returns:
        (x0, y0, x1, y1) in original image coordinates, or "skip"/"quit" string
    """
    h, w = colour_img.shape[:2]

    # Scale to fit screen
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    display_img = cv2.resize(colour_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Reset state
    roi_state["drawing"] = False
    roi_state["start"] = None
    roi_state["end"] = None
    roi_state["done"] = False

    window_name = f"{capture_name} - Draw rectangle (s=skip, q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, roi_mouse_handler)

    print(f"\n{capture_name}:")
    print("  Draw a rectangle around the weld plate")
    print("  Press ENTER to confirm, 'r' to reset, 's' to skip, 'q' to quit")

    while True:
        img_show = display_img.copy()

        # Draw current rectangle
        if roi_state["start"] and roi_state["end"]:
            cv2.rectangle(
                img_show,
                roi_state["start"],
                roi_state["end"],
                (0, 255, 0), 2
            )

        # Instructions
        if roi_state["start"] is None:
            text = "Click and drag to draw rectangle"
        else:
            text = "Press ENTER to confirm, 'r' to redraw"
        cv2.putText(img_show, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_show, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.imshow(window_name, img_show)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            return "quit"

        if key == ord('s'):
            cv2.destroyAllWindows()
            return "skip"

        if key == ord('r'):
            roi_state["start"] = None
            roi_state["end"] = None
            print("  Reset ROI")

        if key == 13 and roi_state["start"] and roi_state["end"]:  # Enter
            break

    cv2.destroyAllWindows()

    # Convert display coords to original image coords
    x0_d, y0_d = roi_state["start"]
    x1_d, y1_d = roi_state["end"]

    # Ensure x0 < x1 and y0 < y1
    x0_d, x1_d = min(x0_d, x1_d), max(x0_d, x1_d)
    y0_d, y1_d = min(y0_d, y1_d), max(y0_d, y1_d)

    # Scale back to original
    x0 = int(x0_d / scale)
    y0 = int(y0_d / scale)
    x1 = int(x1_d / scale)
    y1 = int(y1_d / scale)

    # Clamp to image bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)

    return (x0, y0, x1, y1)


def collect_hole_clicks(
    colour_img: np.ndarray,
    crop_offset: tuple[int, int],
    capture_name: str,
    max_width: int = 1400,
    max_height: int = 900,
) -> list[tuple[str, int, int]] | str:
    """Display cropped region and collect hole clicks.

    Returns:
        List of (hole_type, full_frame_u, full_frame_v) tuples,
        or "skip"/"quit"/"back" string
    """
    x0, y0 = crop_offset
    h, w = colour_img.shape[:2]

    # Scale to fit screen (can upscale for small regions)
    scale = min(max_width / w, max_height / h, 3.0)
    new_w, new_h = int(w * scale), int(h * scale)
    display_img = cv2.resize(colour_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Reset state
    click_state["clicks"] = []

    window_name = f"{capture_name} - Click holes (b=back, s=skip, q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, click_handler)

    print("  Now click the holes: Ø6 (smaller) then Ø8 (larger)")
    print("  Press 'b' to go back and redraw rectangle")

    while True:
        img_show = display_img.copy()

        # Draw existing clicks
        for i, (cx, cy) in enumerate(click_state["clicks"]):
            colour = (0, 255, 0) if i == 0 else (0, 0, 255)
            label = "D6" if i == 0 else "D8"
            cv2.circle(img_show, (cx, cy), 8, colour, 2)
            cv2.circle(img_show, (cx, cy), 2, colour, -1)
            cv2.putText(img_show, label, (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        # Instructions
        if len(click_state["clicks"]) == 0:
            text = "Click D6 (smaller hole) | 'b'=back"
        elif len(click_state["clicks"]) == 1:
            text = "Click D8 (larger hole) | 'b'=back"
        else:
            text = "ENTER=confirm, 'r'=redo, 'b'=back"
        cv2.putText(img_show, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_show, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imshow(window_name, img_show)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            return "quit"

        if key == ord('s'):
            cv2.destroyAllWindows()
            return "skip"

        if key == ord('b'):
            cv2.destroyAllWindows()
            print("  Going back to ROI selection...")
            return "back"

        if key == ord('r'):
            click_state["clicks"] = []
            print("  Reset clicks")

        if key == 13 and len(click_state["clicks"]) >= 2:  # Enter
            break

    cv2.destroyAllWindows()

    # Convert to full-frame coordinates
    holes = []
    for i, (cx, cy) in enumerate(click_state["clicks"][:2]):
        # Remove display scale
        crop_x = int(cx / scale)
        crop_y = int(cy / scale)

        # Add crop offset to get full-frame coords
        full_x = crop_x + x0
        full_y = crop_y + y0

        hole_type = "d6" if i == 0 else "d8"
        holes.append((hole_type, full_x, full_y))
        print(f"  {hole_type}: full-frame ({full_x}, {full_y})")

    return holes


def sort_captures_top_first(captures: list[Path]) -> list[Path]:
    """Sort captures: top faces first, then root faces, both in weld number order."""
    def sort_key(p: Path) -> tuple[int, int]:
        name = p.name
        weld_num = int(name[1:3])
        face_priority = 0 if "top" in name else 1
        return (face_priority, weld_num)
    return sorted(captures, key=sort_key)


def load_existing_seeds(seeds_path: Path) -> tuple[list[dict], set[str]]:
    """Load existing seeds and return (results_list, set_of_capture_ids)."""
    results = []
    done_captures = set()

    if seeds_path.exists():
        with open(seeds_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    "capture_id": row["capture_id"],
                    "hole": row["hole"],
                    "px_u": int(row["px_u"]),
                    "px_v": int(row["px_v"]),
                })
                done_captures.add(row["capture_id"])

    return results, done_captures


def seed_all_captures(data_dir: Path, output_path: Path, start_from: str = None) -> dict:
    """Process all captures and collect hole seeds."""
    # Find all captures
    captures = [
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("C") and d.name[1:3].isdigit()
    ]
    captures = sort_captures_top_first(captures)

    # Load existing seeds (to append to)
    results, done_captures = load_existing_seeds(output_path)

    if done_captures:
        print(f"Loaded {len(done_captures)} already-seeded captures")

    # Find starting index
    start_idx = 0
    if start_from:
        for i, cap in enumerate(captures):
            if cap.name == start_from:
                start_idx = i
                break
        print(f"Starting from {start_from} (index {start_idx + 1}/{len(captures)})")

    print(f"Found {len(captures)} captures total")
    print("Order: top faces first (C01_top, C02_top, ...), then root faces")
    print("=" * 60)

    skipped = []
    failed = []

    idx = start_idx
    while idx < len(captures):
        capture_dir = captures[idx]
        capture_name = capture_dir.name
        progress = f"[{idx + 1}/{len(captures)}]"

        # Skip if already done
        if capture_name in done_captures:
            print(f"{progress} {capture_name}: already seeded, skipping")
            idx += 1
            continue

        try:
            # Load metadata
            meta_path = capture_dir / "meta.json"
            with open(meta_path) as f:
                meta = json.load(f)

            # Load colour frames only
            colour_img = compute_median_colour(capture_dir / "color")

            # Loop for back functionality
            while True:
                # Step 1: Select ROI
                roi_result = select_roi(
                    colour_img,
                    capture_name=f"{progress} {capture_name}",
                )

                if roi_result == "quit":
                    print("\nQuitting and saving...")
                    # Save before quitting
                    save_seeds(results, output_path)
                    return make_summary(captures, results, skipped, failed)

                if roi_result == "skip":
                    print("  Skipped")
                    skipped.append(capture_name)
                    break  # Exit inner loop, continue to next capture

                x0, y0, x1, y1 = roi_result

                # Crop to selected region
                cropped = colour_img[y0:y1, x0:x1]

                # Step 2: Click holes in zoomed view
                holes_result = collect_hole_clicks(
                    cropped,
                    crop_offset=(x0, y0),
                    capture_name=f"{progress} {capture_name}",
                )

                if holes_result == "quit":
                    print("\nQuitting and saving...")
                    save_seeds(results, output_path)
                    return make_summary(captures, results, skipped, failed)

                if holes_result == "skip":
                    print("  Skipped")
                    skipped.append(capture_name)
                    break  # Exit inner loop, continue to next capture

                if holes_result == "back":
                    continue  # Go back to ROI selection (restart inner loop)

                # Success - save results
                for hole_type, px_u, px_v in holes_result:
                    results.append({
                        "capture_id": capture_name,
                        "hole": hole_type,
                        "px_u": px_u,
                        "px_v": px_v,
                    })
                done_captures.add(capture_name)

                print(f"  Done - saved 2 seeds")

                # Save after each successful capture
                save_seeds(results, output_path)
                break  # Exit inner loop, continue to next capture

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append((capture_name, str(e)))

        idx += 1

    # Final save
    save_seeds(results, output_path)
    return make_summary(captures, results, skipped, failed)


def save_seeds(results: list[dict], output_path: Path):
    """Save seeds to CSV."""
    if results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["capture_id", "hole", "px_u", "px_v"])
            writer.writeheader()
            writer.writerows(results)
        print(f"  [Saved {len(results)} seeds to {output_path.name}]")


def make_summary(captures, results, skipped, failed) -> dict:
    """Create summary dict."""
    return {
        "n_captures": len(captures),
        "n_seeded": len(results) // 2,
        "n_skipped": len(skipped),
        "n_failed": len(failed),
        "skipped": skipped,
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser(description="Manual hole seeding tool")
    parser.add_argument("--start", type=str, help="Start from specific capture (e.g., C07_top_r0)")
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data" / "raw" / "S1"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = REPO_ROOT / "configs" / "hole_seeds.csv"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    print("Manual Hole Seeding Tool")
    print("=" * 60)
    print("Two-step process for each capture:")
    print("  1. Draw rectangle around the weld plate")
    print("  2. Click D6 hole (smaller, left), then D8 hole (larger, right)")
    print("")
    print("Controls:")
    print("  ENTER  - confirm selection")
    print("  R      - reset/redraw")
    print("  B      - go back to ROI selection (from hole clicking)")
    print("  S      - skip this capture")
    print("  Q      - quit and save progress")
    print("=" * 60)

    results = seed_all_captures(data_dir, output_path, start_from=args.start)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total captures: {results['n_captures']}")
    print(f"Seeded: {results['n_seeded']}")
    print(f"Skipped: {results['n_skipped']}")
    print(f"Failed: {results['n_failed']}")

    if results["failed"]:
        print("\nFailed captures:")
        for name, err in results["failed"]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
