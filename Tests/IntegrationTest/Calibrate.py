#!/usr/bin/env python3
"""Fit a linear distance correction from real vs. measured distances. Press Run in VS Code.

1. Measure a flat target at several known distances (tape measure, always to the same
   reference point on the camera) and run MedianDistance.py for each one.
2. Put each (real_mm, python_mm) pair in MEASUREMENTS below.
3. Run this script and copy CAL_SLOPE / CAL_OFFSET into MedianDistance.py.

Model:  python = CAL_SLOPE * real + CAL_OFFSET   ->   corrected = (python - CAL_OFFSET) / CAL_SLOPE
"""

import numpy as np

# =========================================================================== SETTINGS
# (real distance in mm, distance reported by MedianDistance.py in mm)
MEASUREMENTS = [
    # 300 mm
    (300.0, 297.8),
    (300.0, 297.5),
    (300.0, 297.5),
    (300.0, 297.9),
    (300.0, 297.9),
    (300.0, 298.1),
    (300.0, 298.3),
    (300.0, 298.2),
    (300.0, 298.0),
    (300.0, 298.2),
    # 400 mm
    (400.0, 399.0),
    (400.0, 399.7),
    (400.0, 399.8),
    (400.0, 399.8),
    (400.0, 400.5),
    (400.0, 400.1),
    (400.0, 400.0),
    (400.0, 400.0),
    (400.0, 400.1),
    (400.0, 399.8),
    # 500 mm
    (500.0, 501.8),
    (500.0, 501.8),
    (500.0, 501.6),
    (500.0, 501.6),
    (500.0, 501.8),
    (500.0, 501.8),
    (500.0, 502.1),
    (500.0, 501.8),
    (500.0, 502.0),
    (500.0, 502.2),
    # 600 mm
    (600.0, 602.2),
    (600.0, 602.4),
    (600.0, 602.5),
    (600.0, 602.2),
    (600.0, 602.8),
    (600.0, 602.5),
    (600.0, 602.2),
    (600.0, 602.1),
    (600.0, 602.5),
    (600.0, 603.0),
    # 700 mm
    (700.0, 701.9),
    (700.0, 702.5),
    (700.0, 702.0),
    (700.0, 702.5),
    (700.0, 702.5),
    (700.0, 702.4),
    (700.0, 702.5),
    (700.0, 702.2),
    (700.0, 702.9),
    (700.0, 703.8),
    # 800 mm
    (800.0, 804.5),
    (800.0, 805.0),
    (800.0, 805.0),
    (800.0, 804.6),
    (800.0, 804.8),
    (800.0, 805.0),
    (800.0, 804.6),
    (800.0, 804.9),
    (800.0, 805.0),
    (800.0, 804.6),
    # 900 mm
    (900.0, 905.8),
    (900.0, 905.2),
    (900.0, 905.2),
    (900.0, 905.6),
    (900.0, 905.6),
    (900.0, 906.0),
    (900.0, 905.5),
    (900.0, 905.6),
    (900.0, 905.5),
    (900.0, 905.5),
    # 1000 mm
    (1000.0, 1001.8),
    (1000.0, 1002.1),
    (1000.0, 1001.9),
    (1000.0, 1002.0),
    (1000.0, 1002.0),
    (1000.0, 1001.9),
    (1000.0, 1002.2),
    (1000.0, 1002.6),
    (1000.0, 1002.0),
    (1000.0, 1002.1),
    # 1100 mm
    (1100.0, 1100.5),
    (1100.0, 1100.4),
    (1100.0, 1100.0),
    (1100.0, 1100.4),
    (1100.0, 1100.6),
    (1100.0, 1100.6),
    (1100.0, 1100.8),
    (1100.0, 1100.8),
    (1100.0, 1100.4),
    (1100.0, 1100.8),
    # 1200 mm
    (1200.0, 1203.4),
    (1200.0, 1204.0),
    (1200.0, 1203.8),
    (1200.0, 1203.2),
    (1200.0, 1204.6),
    (1200.0, 1204.1),
    (1200.0, 1203.5),
    (1200.0, 1203.5),
    (1200.0, 1204.4),
    (1200.0, 1203.5),
    # 1300 mm
    (1300.0, 1305.8),
    (1300.0, 1305.4),
    (1300.0, 1306.0),
    (1300.0, 1307.4),
    (1300.0, 1306.6),
    (1300.0, 1306.5),
    (1300.0, 1305.6),
    (1300.0, 1304.9),
    (1300.0, 1304.9),
    (1300.0, 1304.9),
    # 1400 mm
    (1400.0, 1404.8),
    (1400.0, 1400.6),
    (1400.0, 1406.0),
    (1400.0, 1405.8),
    (1400.0, 1406.2),
    (1400.0, 1405.5),
    (1400.0, 1405.6),
    (1400.0, 1405.6),
    (1400.0, 1405.4),
    # 1480 mm
    (1480.0, 1476.6),
    (1480.0, 1477.8),
    (1480.0, 1476.0),
    (1480.0, 1470.8),
    (1480.0, 1476.2),
    (1480.0, 1476.6),
    (1480.0, 1475.8),
    (1480.0, 1473.5),
    (1480.0, 1476.2),
    (1480.0, 1476.9),
]
TOLERANCE_MM = 5.0
# ====================================================================================


def fit(real, meas):
    slope, offset = np.polyfit(real, meas, 1)
    return slope, offset


def correct(meas, slope, offset):
    return (meas - offset) / slope


def main():
    data = np.array(MEASUREMENTS, dtype=float)
    real, meas = data[:, 0], data[:, 1]
    if len(real) < 3 or np.ptp(real) == 0:
        raise SystemExit("Need at least 3 measurements at 2+ different distances.")

    slope, offset = fit(real, meas)
    corrected = correct(meas, slope, offset)
    resid = corrected - real

    print(f"Fitted from {len(real)} measurements ({real.min():.0f}..{real.max():.0f} mm)\n")
    print(f"{'real':>8} {'python':>8} {'raw err':>8} {'corrected':>10} {'corr err':>9} {'LOO err':>8}")

    # Leave-one-out: fit without point i, predict point i -> honest estimate for new points.
    loo = np.full(len(real), np.nan)
    for i in range(len(real)):
        keep = np.arange(len(real)) != i
        if np.ptp(real[keep]) > 0:
            s, o = fit(real[keep], meas[keep])
            loo[i] = correct(meas[i], s, o) - real[i]

    for r, m, c, e, l in zip(real, meas, corrected, resid, loo):
        print(f"{r:8.1f} {m:8.1f} {m - r:+8.1f} {c:10.1f} {e:+9.1f} {l:+8.1f}")

    raw = meas - real
    print(f"\nRaw error       : mean {raw.mean():+.2f}, max |err| {np.abs(raw).max():.2f} mm")
    print(f"Corrected error : std {resid.std(ddof=2):.2f}, max |err| {np.abs(resid).max():.2f} mm")
    print(f"Leave-one-out   : RMS {np.sqrt(np.nanmean(loo ** 2)):.2f}, max |err| {np.nanmax(np.abs(loo)):.2f} mm")
    print(f"Within ±{TOLERANCE_MM:g} mm: raw {np.sum(np.abs(raw) <= TOLERANCE_MM)}/{len(raw)}, "
          f"corrected (LOO) {np.sum(np.abs(loo) <= TOLERANCE_MM)}/{len(loo)}")

    print("\nPaste into MedianDistance.py:")
    print(f"CAL_SLOPE = {slope:.5f}")
    print(f"CAL_OFFSET = {offset:.2f}")
    if abs(slope - 1) > 0.02:
        print(f"\nNote: scale error {100 * (slope - 1):+.1f}% is large for a ToF camera - check the tape is "
              "perpendicular, the reference point is consistent, and z vs 3D distance (see MedianDistance.py).")


if __name__ == "__main__":
    main()
