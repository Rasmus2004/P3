#!/usr/bin/env python3
"""Per-pixel median of several .ply frames + distance to the middle pixel. Press Run in VS Code.

Needs .ply files from 100Frames.py (they store which sensor pixel each point came from).

1. Loads the first NUM_FILES .ply files in PLY_FOLDER.
2. For every pixel, takes the median x, y, z over those frames.
3. Prints the distance to the middle pixel (and each frame's own value, to see the spread).
4. Optionally shows the median point cloud in Open3D with the middle pixel marked in red.
"""

import warnings
from pathlib import Path

import numpy as np

# =========================================================================== SETTINGS
PLY_FOLDER = Path(__file__).parent / "captures/20261126_140350"  # folder with the .ply files
NUM_FILES = 160          # how many files to use (the first N, sorted by name)
CENTER_RADIUS = 2       # 0 = only the exact middle pixel; 2 = median over a 5x5 patch, etc.
SHOW_WINDOW = True      # show the median point cloud in Open3D

# Linear correction from Calibrate.py:  corrected = (measured - CAL_OFFSET) / CAL_SLOPE
# Set CAL_SLOPE = 1.0 and CAL_OFFSET = 0.0 to see the raw, uncorrected values.
CAL_SLOPE = 1.00142
CAL_OFFSET = 0.85
# ====================================================================================

PLY_TYPES = {"float": "<f4", "uchar": "u1", "uint": "<u4", "ushort": "<u2", "double": "<f8"}


def read_ply(path):
    """Read a binary PLY written by 100Frames.py -> (vertex array, width, height, temperature or None)."""
    with open(path, "rb") as f:
        width = height = temperature = None
        count, fields = 0, []
        while True:
            line = f.readline().decode("ascii").strip()
            parts = line.split()
            if line == "end_header":
                break
            if parts[:2] == ["comment", "width"]:
                width = int(parts[2])
            elif parts[:2] == ["comment", "height"]:
                height = int(parts[2])
            elif parts[:2] == ["comment", "temperature"]:
                temperature = float(parts[2])
            elif parts[:2] == ["element", "vertex"]:
                count = int(parts[2])
            elif parts[0] == "property":
                fields.append((parts[2], PLY_TYPES[parts[1]]))
        data = np.fromfile(f, dtype=np.dtype(fields), count=count)
    if width is None or "pixel" not in data.dtype.names:
        raise SystemExit(f"{path.name} has no pixel info - recapture it with the updated 100Frames.py")
    return data, width, height, temperature


def main():
    files = sorted(PLY_FOLDER.glob("*.ply"))[:NUM_FILES]
    if not files:
        raise SystemExit(f"No .ply files found in {PLY_FOLDER}")
    if len(files) < NUM_FILES:
        print(f"Warning: only {len(files)} files found, using all of them")

    # Fill one preallocated array with every frame as a full image grid (H*W pixels);
    # missing pixels = NaN. Filled in place so memory stays low with many files.
    _, w, h, _ = read_ply(files[0])
    stack_xyz = np.full((len(files), h * w, 3), np.nan, dtype=np.float32)  # (frames, pixels, 3)
    stack_gray = np.full((len(files), h * w), np.nan, dtype=np.float16)    # 0..255 fits exactly
    temps = []
    for i, path in enumerate(files):
        data, _, _, temp = read_ply(path)
        if temp is not None:
            temps.append(temp)
        stack_xyz[i, data["pixel"], 0] = data["x"]
        stack_xyz[i, data["pixel"], 1] = data["y"]
        stack_xyz[i, data["pixel"], 2] = data["z"]
        stack_gray[i, data["pixel"]] = data["red"]
        print(f"\rLoaded {i + 1}/{len(files)}", end="", flush=True)
    print()

    # Median in chunks of pixels, so numpy's temporary copies stay small.
    median_xyz = np.empty((h * w, 3), dtype=np.float32)
    median_gray = np.empty(h * w, dtype=np.float32)
    chunk = 20000
    with warnings.catch_warnings():  # pixels with no valid value in any frame stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in range(0, h * w, chunk):
            median_xyz[s:s + chunk] = np.nanmedian(stack_xyz[:, s:s + chunk], axis=0)
            median_gray[s:s + chunk] = np.nanmedian(stack_gray[:, s:s + chunk].astype(np.float32), axis=0)

    # Middle pixel (or a small patch around it).
    cy, cx = h // 2, w // 2
    rows = np.arange(cy - CENTER_RADIUS, cy + CENTER_RADIUS + 1)
    cols = np.arange(cx - CENTER_RADIUS, cx + CENTER_RADIUS + 1)
    patch = (rows[:, None] * w + cols[None, :]).ravel()

    print(f"Used {len(files)} files from {PLY_FOLDER}")
    print(f"Image {w}x{h}, middle pixel (col {cx}, row {cy}), patch {2 * CENTER_RADIUS + 1}x{2 * CENTER_RADIUS + 1}\n")

    if temps:
        print(f"Camera temperature: {min(temps):.1f}..{max(temps):.1f} C (mean {np.mean(temps):.1f} C)\n")

    print("Per frame (before median):")
    per_frame = []
    for path, frame in zip(files, stack_xyz):
        p = frame[patch]
        p = p[~np.isnan(p[:, 2])]
        if len(p):
            d = np.linalg.norm(np.median(p, axis=0))
            per_frame.append(d)
            print(f"  {path.name}: {d:8.1f} mm")
        else:
            print(f"  {path.name}:   no valid value")

    if len(per_frame) > 1:
        pf = np.array(per_frame)
        q1, q3 = np.percentile(pf, (25, 75))
        print(f"\nFrame spread ({len(pf)}/{len(files)} valid): std {pf.std(ddof=1):.2f} mm, "
              f"IQR {q3 - q1:.2f} mm, min {pf.min():.1f}, max {pf.max():.1f} mm")
        # Drift between start and end of the run hints at warm-up / temperature effects.
        n = max(len(pf) // 10, 1)
        print(f"  first {n} frames {np.median(pf[:n]):.1f} mm, last {n} frames {np.median(pf[-n:]):.1f} mm "
              f"(drift {np.median(pf[-n:]) - np.median(pf[:n]):+.2f} mm)")

    center = median_xyz[patch]
    center = center[~np.isnan(center[:, 2])]
    if not len(center):
        raise SystemExit("\nMiddle pixel has no valid value in any frame (no return there).")
    x, y, z = np.median(center, axis=0)
    distance = float(np.sqrt(x * x + y * y + z * z))

    print("\nMEDIAN RESULT")
    print(f"  x, y, z         : {x:.1f}, {y:.1f}, {z:.1f} mm")
    print(f"  depth (Z)       : {z:.1f} mm  ({z / 1000:.3f} m)   <- compare to a tape held perpendicular")
    print(f"  distance (3D)   : {distance:.1f} mm  ({distance / 1000:.3f} m)   <- straight line from camera origin")
    if (CAL_SLOPE, CAL_OFFSET) != (1.0, 0.0):
        print(f"\nCORRECTED (CAL_SLOPE {CAL_SLOPE}, CAL_OFFSET {CAL_OFFSET})")
        print(f"  depth (Z)       : {(z - CAL_OFFSET) / CAL_SLOPE:.1f} mm")
        print(f"  distance (3D)   : {(distance - CAL_OFFSET) / CAL_SLOPE:.1f} mm")

    if SHOW_WINDOW:
        import open3d as o3d

        valid = ~np.isnan(median_xyz[:, 2])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(median_xyz[valid].astype(np.float64))
        g = (median_gray[valid] / 255.0).astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack([g, g, g]))

        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(distance * 0.01, 5.0))
        marker.translate([x, y, z])
        marker.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries(
            [pcd, marker],
            window_name=f"Median of {len(files)} frames - middle pixel: {distance:.1f} mm",
            width=1280, height=800,
            front=[0, 0, -1], up=[0, -1, 0], lookat=[x, y, z], zoom=0.5,
        )


if __name__ == "__main__":
    main()
