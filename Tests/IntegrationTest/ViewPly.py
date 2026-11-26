#!/usr/bin/env python3
"""Dead-simple .ply viewer. Press Run in VS Code.

    SPACE  next file
    B      previous file
    Q/ESC  quit
"""

from pathlib import Path

import open3d as o3d

# =========================================================================== SETTINGS
# Folder with .ply files. If it has none directly inside it, the newest subfolder is used
# (so pointing it at "captures" opens the latest run from 100Frames.py).
PLY_FOLDER = Path(__file__).parent / "captures/20261124_100809"
# ====================================================================================

folder = PLY_FOLDER
files = sorted(folder.glob("*.ply"))
if not files:
    subfolders = sorted(p for p in folder.iterdir() if p.is_dir())
    if subfolders:
        folder = subfolders[-1]  # timestamped names, so last = newest
        files = sorted(folder.glob("*.ply"))
if not files:
    raise SystemExit(f"No .ply files found in {PLY_FOLDER}")

print(f"{len(files)} files in {folder}")
print("SPACE = next, B = previous, Q/ESC = quit")

index = 0
pcd = o3d.io.read_point_cloud(str(files[0]))
print(f"[1/{len(files)}] {files[0].name}")


def show(vis, step):
    global index
    index = min(max(index + step, 0), len(files) - 1)
    new = o3d.io.read_point_cloud(str(files[index]))
    pcd.points = new.points
    pcd.colors = new.colors
    vis.update_geometry(pcd)
    print(f"[{index + 1}/{len(files)}] {files[index].name}")
    return True


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="PLY viewer", width=1280, height=800)
vis.add_geometry(pcd)
vis.register_key_callback(32, lambda v: show(v, +1))        # SPACE
vis.register_key_callback(ord("B"), lambda v: show(v, -1))  # B

# Start looking from the camera's point of view (camera looks along +Z, Y points down).
ctr = vis.get_view_control()
ctr.set_front([0, 0, -1])
ctr.set_up([0, -1, 0])

vis.run()
vis.destroy_window()
