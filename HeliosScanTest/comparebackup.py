"""
Compare a scan against a reference scan and show where the damage is.

Usage:
  python Compare.py PinHole              # compares scans/PinHole_xyz.npy to scans/Refference_xyz.npy
  python Compare.py MaxDmg --threshold 4
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

SCANS = Path(__file__).parent / "scans"          # works no matter which folder you run from

# Pixel box around the object (x0, x1, y0, y1). Measured from Refference_xyz.npy - update with your real ROI.
ROI = (195, 429, 192, 289)

# Used when you press Run in VS Code (no command-line arguments). Change these to pick scans.
TEST_SCAN = "Ref2"
REF_SCAN = "Refference"
THRESHOLD_MM = 3.0


def load_z(name):
    return np.load(SCANS / f"{name}_xyz.npy")[..., 2]   # depth in mm, NaN = no point


def remove_placement(diff, cutoff):
    """The object never sits exactly the same twice: fit a smooth (quadratic) surface to the
    difference and subtract it, so only local damage remains. Points differing more than
    `cutoff` mm are left out of the fit so the damage doesn't drag it along."""
    h, w = diff.shape
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    terms = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], axis=-1)
    ok = ~np.isnan(diff)
    use = ok.copy()
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(terms[use], diff[use], rcond=None)
        residual = diff - terms @ coef
        use = ok & (np.abs(residual) < cutoff)
    noise = 1.4826 * np.median(np.abs(residual[use])) if use.any() else 0.0
    return residual, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", nargs="?", default=TEST_SCAN,
                    help=f"scan name without _xyz.npy (default {TEST_SCAN})")
    ap.add_argument("--ref", default=REF_SCAN, help=f"reference scan name (default {REF_SCAN})")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_MM,
                    help=f"damage threshold in mm (default {THRESHOLD_MM})")
    args = ap.parse_args()

    x0, x1, y0, y1 = ROI
    ref = load_z(args.ref)[y0:y1, x0:x1]
    test = load_z(args.test)[y0:y1, x0:x1]

    diff, noise = remove_placement(test - ref, args.threshold)   # + = deeper than reference (dent)
    diff = median_filter(np.nan_to_num(diff), size=3)            # 3x3 smoothing removes speckle
    missing = ~np.isnan(ref) & np.isnan(test)                    # point disappeared (through-hole?)
    damage = (np.abs(diff) > args.threshold) | missing

    print(f"{args.test} vs {args.ref}")
    print(f"  noise level:     {noise:.1f} mm")
    print(f"  damaged pixels:  {damage.sum()} of {np.sum(~np.isnan(ref))} ({100 * damage.mean():.1f} %)")
    print(f"  deepest dent:    {np.nanmax(diff):.1f} mm")
    print(f"  missing points:  {missing.sum()}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    im = ax[0].imshow(diff, cmap="coolwarm", vmin=-10, vmax=10)
    ax[0].set_title("Difference to reference [mm]  (red = dent, blue = bump)")
    fig.colorbar(im, ax=ax[0])
    ax[1].imshow(damage, cmap="gray")
    ax[1].set_title(f"Damage (> {args.threshold} mm or missing)")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{args.test} vs {args.ref}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
