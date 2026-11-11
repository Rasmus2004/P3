"""
Simple Helios (Lucid Vision Labs ToF camera) object scanner.

Captures a few frames, averages them, turns them into a 3D point cloud and:
  * saves it as  scans/scan_<timestamp>.ply  (open in MeshLab / CloudCompare / Open3D)
  * saves a depth heat-map PNG
  * shows the point cloud (Open3D, if installed) or the heat-map (matplotlib)

Damage detection/grading will be built on top of this later.

Requirements:
  - Lucid Arena SDK + the arena_api Python wheel (from thinklucid.com)
  - pip install numpy matplotlib        (open3d optional: pip install open3d)

Usage:
  python scan.py                 # 10 averaged frames
  python scan.py --frames 30     # more frames = less noise
  python scan.py --no-view       # just save files
"""

import argparse
import ctypes
import time
import warnings
from pathlib import Path

import numpy as np
from arena_api.system import system

OUT_DIR = Path(__file__).parent / "scans"


# --------------------------------------------------------------------------- camera
def connect():
    devices = system.create_device()
    if not devices:
        raise SystemExit("No Lucid camera found. Check cable/power/IP settings (ArenaView).")
    device = devices[0]
    model = device.nodemap["DeviceModelName"].value
    print(f"Connected to {model}")
    return device


def configure(device):
    nm = device.nodemap
    stream = device.tl_stream_nodemap
    stream["StreamBufferHandlingMode"].value = "NewestOnly"
    stream["StreamAutoNegotiatePacketSize"].value = True
    stream["StreamPacketResendEnable"].value = True

    # Helios1 uses signed ABCY16s, Helios2 uses unsigned ABCY16 - take whichever exists.
    available = nm["PixelFormat"].enumentry_names
    for fmt in ("Coord3D_ABCY16s", "Coord3D_ABCY16"):
        if fmt in available:
            nm["PixelFormat"].value = fmt
            break
    else:
        raise SystemExit(f"No 3D pixel format available. Options: {available}")
    pixel_format = nm["PixelFormat"].value
    print(f"Pixel format: {pixel_format}")

    if "Scan3dOperatingMode" in nm.feature_names:
        print(f"Operating mode: {nm['Scan3dOperatingMode'].value} "
              f"(available: {list(nm['Scan3dOperatingMode'].enumentry_names)})")

    # Scale + per-axis offsets convert raw values to millimetres.
    scale = nm["Scan3dCoordinateScale"].value
    offsets = []
    for axis in ("CoordinateA", "CoordinateB", "CoordinateC"):
        nm["Scan3dCoordinateSelector"].value = axis
        offsets.append(nm["Scan3dCoordinateOffset"].value)

    dtype = np.int16 if pixel_format.endswith("s") else np.uint16
    return dtype, scale, np.array(offsets, dtype=np.float32)


def grab_frames(device, n_frames, dtype):
    """Returns a list of (H, W, 4) raw arrays: X, Y, Z, intensity."""
    frames = []
    with device.start_stream(1):
        for i in range(n_frames):
            buf = device.get_buffer()
            h, w = buf.height, buf.width
            n_bytes = h * w * 8  # 4 channels x 16 bit
            raw = np.ctypeslib.as_array(
                ctypes.cast(buf.pdata, ctypes.POINTER(ctypes.c_uint8)), shape=(n_bytes,)
            )
            frames.append(raw.view(dtype).reshape(h, w, 4).copy())
            device.requeue_buffer(buf)
            print(f"\rCaptured frame {i + 1}/{n_frames}", end="")
    print()
    return frames


# --------------------------------------------------------------------------- processing
def flying_pixels(z, mask, max_jump):
    """Pixels on depth edges (ToF 'flying pixels'): a neighbour differs by more than max_jump mm."""
    zp = np.pad(np.where(mask, z, np.nan), 1, constant_values=np.nan)
    h, w = z.shape
    worst = np.zeros_like(z)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nb = zp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                worst = np.fmax(worst, np.abs(z - nb))      # fmax ignores missing neighbours
    return worst > max_jump


def to_point_cloud(frames, scale, offsets, args):
    """Average frames and remove noise. Returns xyz (H,W,3 mm), intensity (H,W), mask (H,W)."""
    stack = np.stack(frames).astype(np.float32)          # (N, H, W, 4)
    xyz = stack[..., :3] * scale + offsets               # mm
    intensity = stack[..., 3]

    # Per frame: drop weak-signal / zero-depth readings.
    bad = (intensity < args.min_intensity) | (stack[..., 2] == 0)
    xyz[bad] = np.nan
    seen = (~bad).sum(axis=0)

    with warnings.catch_warnings():                      # all-NaN pixels warn; they're masked anyway
        warnings.simplefilter("ignore", RuntimeWarning)
        xyz_med = np.nanmedian(xyz, axis=0)              # median: robust to single-frame spikes
        # robust jitter (MAD scaled to std): one bad frame doesn't condemn a pixel
        z_std = 1.4826 * np.nanmedian(np.abs(xyz[..., 2] - xyz_med[..., 2]), axis=0)
    int_mean = np.where(bad, 0, intensity).sum(axis=0) / np.maximum(seen, 1)
    z = xyz_med[..., 2]

    stats = {}
    mask = seen >= max(1, len(frames) // 2);            stats["signal"] = mask.sum()
    mask &= z_std <= args.max_std;                      stats["stable"] = mask.sum()
    if args.zmin is not None: mask &= z >= args.zmin
    if args.zmax is not None: mask &= z <= args.zmax
    if args.roi:
        x, y, w, h = args.roi
        roi = np.zeros_like(mask); roi[y:y + h, x:x + w] = True
        mask &= roi
    stats["roi/depth"] = mask.sum()
    if args.edge > 0:
        mask &= ~flying_pixels(z, mask, args.edge);     stats["edges"] = mask.sum()

    print("Points kept after each filter: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    return xyz_med, int_mean, mask


def outlier_removal(xyz, mask, neighbours, std_ratio):
    """Open3D statistical outlier removal: drops isolated floating points. Updates mask in place."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed - skipping statistical outlier removal.")
        return mask
    idx = np.flatnonzero(mask)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.reshape(-1, 3)[idx].astype(np.float64)))
    _, keep = pcd.remove_statistical_outlier(nb_neighbors=neighbours, std_ratio=std_ratio)
    new = np.zeros(mask.size, bool)
    new[idx[np.asarray(keep, dtype=int)]] = True
    print(f"Outlier removal: {len(idx)} -> {len(keep)} points")
    return new.reshape(mask.shape)


def save_ply(path, points, intensity):
    gray = (255 * intensity / max(intensity.max(), 1)).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), g in zip(points, gray):
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {g} {g} {g}\n")


def save_depth_png(path, xyz, mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = np.where(mask, xyz[..., 2], np.nan)
    plt.figure(figsize=(8, 6))
    plt.imshow(depth, cmap="turbo")
    plt.colorbar(label="distance [mm]")
    plt.title("Helios depth map")
    plt.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def view(ply_path, png_path):
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        # Start from the camera's point of view (Helios: camera at origin, +Z forward, +Y down).
        o3d.visualization.draw_geometries(
            [pcd], window_name="Helios scan",
            lookat=pcd.get_center(), front=[0, 0, -1], up=[0, -1, 0], zoom=0.7,
        )
    except ImportError:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        print("open3d not installed - showing depth map instead (pip install open3d for 3D).")
        plt.imshow(mpimg.imread(png_path))
        plt.axis("off")
        plt.show()


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Simple Helios object scan")
    ap.add_argument("--frames", type=int, default=30, help="frames to average (default 30)")
    ap.add_argument("--min-intensity", type=float, default=30,
                    help="drop pixels with weaker return signal (default 30)")
    ap.add_argument("--no-view", action="store_true", help="don't open a viewer")
    # --- noise filters ---
    ap.add_argument("--zmin", type=float, help="keep only points at least this far away [mm]")
    ap.add_argument("--zmax", type=float, help="keep only points at most this far away [mm] "
                                               "(cuts the background, keeps the whole object)")
    ap.add_argument("--roi", type=lambda s: tuple(int(v) for v in s.split(",")),
                    metavar="X,Y,W,H", help="optional pixel box, e.g. 200,100,240,280")
    ap.add_argument("--max-std", type=float, default=5.0,
                    help="drop pixels whose depth jitters more than this between frames [mm] (default 5)")
    ap.add_argument("--edge", type=float, default=15.0,
                    help="drop edge 'flying pixels' with a neighbour jump > this [mm], 0=off (default 15)")
    ap.add_argument("--sor-neighbours", type=int, default=20,
                    help="statistical outlier removal neighbours, 0=off (default 20)")
    ap.add_argument("--sor-std", type=float, default=2.0,
                    help="statistical outlier removal strictness, lower = stricter (default 2.0)")
    args = ap.parse_args()

    device = connect()
    try:
        dtype, scale, offsets = configure(device)
        frames = grab_frames(device, args.frames, dtype)
    finally:
        system.destroy_device()

    xyz, intensity, mask = to_point_cloud(frames, scale, offsets, args)
    if args.sor_neighbours > 0 and mask.any():
        mask = outlier_removal(xyz, mask, args.sor_neighbours, args.sor_std)
    points = xyz[mask]
    print(f"Valid points: {len(points)} / {mask.size}")
    if len(points):
        print(f"Depth range: {points[:, 2].min():.0f} - {points[:, 2].max():.0f} mm")

    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ply_path = OUT_DIR / f"scan_{stamp}.ply"
    png_path = OUT_DIR / f"scan_{stamp}_depth.png"
    np.save(OUT_DIR / f"scan_{stamp}_xyz.npy", np.where(mask[..., None], xyz, np.nan))  # for later analysis
    save_ply(ply_path, points, intensity[mask])
    save_depth_png(png_path, xyz, mask)
    print(f"Saved: {ply_path}\n       {png_path}")

    if not args.no_view:
        view(ply_path, png_path)


if __name__ == "__main__":
    main()
