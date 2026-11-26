#!/usr/bin/env python3
"""Capture frames from a Lucid Helios ToF camera and save each one as a .ply point cloud.

Just press Run in VS Code. Change the settings below to suit.

Each frame is written to disk as soon as it is captured, and the camera buffer is only
released once the file is written. If the program crashes partway, every frame saved
before the crash is kept.

Output (one timestamped folder per run, next to this script by default):
    captures/20261124_153012/frame_0001.ply
    captures/20261124_153012/frame_0002.ply
    ...

Each .ply is binary little-endian with x, y, z (float32, millimetres) and a grayscale
colour from the Helios intensity channel. Opens in MeshLab, CloudCompare, Open3D, Blender.
"""

import ctypes
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from arena_api.system import system

# =========================================================================== SETTINGS
NUM_FRAMES = 20                                       # how many frames to capture
OUTPUT_DIR = Path(__file__).parent / "captures"        # parent folder for all runs
USE_TIMESTAMP_SUBFOLDER = True                         # True: new subfolder per run
BUFFER_TIMEOUT_MS = 2000                               # wait time per frame before giving up
WARMUP_FRAMES = 25                                     # frames grabbed and thrown away before saving (0 = off)
WARMUP_SECONDS = 0                                     # extra thermal warm-up: stream and discard for this long.
                                                       # ToF depth drifts while the sensor heats up; use 600-1200
                                                       # (10-20 min) for calibration / accuracy runs.

# Camera settings (defaults = what the camera was set to before, for variable control)
OPERATING_MODE = "Distance1500mm"     # "Distance1500mm" or "Distance6000mm"
EXPOSURE = "Exp250Us"                # "Exp1000Us" or "Exp250Us"
IMAGE_ACCUMULATION = 1               # 1..32 frames averaged inside the camera; fps ~= 30 / this
SPATIAL_FILTER = True                 # camera's built-in smoothing filter
CONFIDENCE_MIN = 50                  # drop pixels below this confidence (0..65535), None = off.
                                      # Removes weak / multipath pixels; lower it if valid points drop a lot
# ====================================================================================

PIXEL_FORMAT = "Coord3D_ABCY16"
INVALID_RAW = 0xFFFF  # Helios marks pixels with no/low-confidence return with raw C == 65535

# --------------------------------------------------------------------------- camera

def connect():
    devices = system.create_device()
    if not devices:
        raise SystemExit("No Lucid camera found. Check cable/power/IP settings (ArenaView).")
    device = devices[0]
    nm = device.nodemap
    print(f"Connected to {nm['DeviceModelName'].value} (S/N {nm['DeviceSerialNumber'].value})")
    return device


def configure(device, operating_mode, exposure, accumulation, spatial_filter, confidence_min):
    nm = device.nodemap
    stream = device.tl_stream_nodemap
    stream["StreamBufferHandlingMode"].value = "NewestOnly"
    stream["StreamAutoNegotiatePacketSize"].value = True
    stream["StreamPacketResendEnable"].value = True

    nm["TriggerMode"].value = "Off"  # free-running; other scripts may leave it on
    nm["PixelFormat"].value = PIXEL_FORMAT
    nm["Scan3dOperatingMode"].value = operating_mode  # set before exposure: options depend on it
    nm["ExposureTimeSelector"].value = exposure
    nm["Scan3dImageAccumulation"].value = accumulation
    nm["Scan3dSpatialFilterEnable"].value = spatial_filter
    nm["Scan3dConfidenceThresholdEnable"].value = confidence_min is not None
    if confidence_min is not None:
        nm["Scan3dConfidenceThresholdMin"].value = confidence_min

    print(f"Mode {operating_mode}, exposure {exposure} ({nm['ExposureTime'].value:.0f} us), "
          f"accumulation {accumulation}, spatial filter {spatial_filter}, confidence min {confidence_min}")


def read_scale_offset(device):
    """Per-axis scale/offset that convert raw uint16 A, B, C into millimetres."""
    nm = device.nodemap
    scale, offset = [], []
    for coord in ("CoordinateA", "CoordinateB", "CoordinateC"):
        nm["Scan3dCoordinateSelector"].value = coord
        scale.append(float(nm["Scan3dCoordinateScale"].value))
        offset.append(float(nm["Scan3dCoordinateOffset"].value))
    return np.array(scale, dtype=np.float32), np.array(offset, dtype=np.float32)


def read_temperature(device):
    """Camera temperature in deg C, or None if this camera/firmware doesn't expose it."""
    nm = device.nodemap
    try:
        return float(nm["DeviceTemperature"].value)
    except Exception:
        return None


# --------------------------------------------------------------------------- processing

def to_point_cloud(frame, scale, offset):
    """Raw ABCY16 frame -> (xyz float32 in mm, gray uint8 0..255, pixel index), invalid pixels removed.

    pixel index = row * width + col of the sensor pixel each point came from.
    """
    raw = frame.reshape(-1, 4)
    pixel = np.flatnonzero(raw[:, 2] != INVALID_RAW).astype(np.uint32)
    raw = raw[pixel]

    xyz = raw[:, :3].astype(np.float32) * scale + offset

    # Intensity (Y channel), contrast-stretched to the 1st..99th percentile.
    intensity = raw[:, 3].astype(np.float32)
    if intensity.size:
        lo, hi = np.percentile(intensity, (1, 99))
        intensity = np.clip((intensity - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    gray = (intensity * 255.0 + 0.5).astype(np.uint8)

    return xyz, gray, pixel


# --------------------------------------------------------------------------- saving

PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("pixel", "<u4"),
])


def write_ply(path, xyz, gray, pixel, width, height, temperature=None):
    """Write a binary little-endian PLY point cloud (x, y, z + grayscale RGB + pixel index).

    The sensor width/height go in header comments so the image grid can be rebuilt later.

    Writes to a .tmp file first and renames it at the end, so a crash mid-write
    never leaves a half-written .ply behind.
    """
    vertices = np.empty(len(xyz), dtype=PLY_DTYPE)
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["red"] = vertices["green"] = vertices["blue"] = gray
    vertices["pixel"] = pixel

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Lucid Helios ToF capture, units: millimetres\n"
        f"comment width {width}\n"
        f"comment height {height}\n"
        + (f"comment temperature {temperature:.2f}\n" if temperature is not None else "")
        + f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uint pixel\n"
        "end_header\n"
    )
    tmp = path.with_suffix(".ply.tmp")
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())
        f.flush()
        os.fsync(f.fileno())  # make sure it is really on disk before moving on
    tmp.replace(path)


# --------------------------------------------------------------------------- capture + save

def capture_and_save(device, frame_count, scale, offset, out_dir):
    """Grab a frame, write it to disk, then release the buffer back to the camera."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with device.start_stream(1):
        # Warm-up: grab and immediately release frames so the camera settles before saving.
        for j in range(1, WARMUP_FRAMES + 1):
            try:
                device.requeue_buffer(device.get_buffer(timeout=BUFFER_TIMEOUT_MS))
            except Exception as e:  # a missed warm-up frame doesn't matter
                print(f"\nWarm-up frame {j}: no buffer ({e})")
            print(f"\rWarm-up {j}/{WARMUP_FRAMES} (discarded)", end="", flush=True)
        if WARMUP_FRAMES:
            print()

        # Thermal warm-up: keep the sensor running (and heating) for WARMUP_SECONDS.
        if WARMUP_SECONDS > 0:
            t_end = time.monotonic() + WARMUP_SECONDS
            while (left := t_end - time.monotonic()) > 0:
                try:
                    device.requeue_buffer(device.get_buffer(timeout=BUFFER_TIMEOUT_MS))
                except Exception:
                    pass
                temp = read_temperature(device)
                temp_txt = f", {temp:.1f} C" if temp is not None else ""
                print(f"\rThermal warm-up: {left:5.0f} s left{temp_txt}   ", end="", flush=True)
            print()

        temp = read_temperature(device)
        if temp is not None:
            print(f"Camera temperature at start: {temp:.1f} C")

        for i in range(1, frame_count + 1):
            try:
                buf = device.get_buffer(timeout=BUFFER_TIMEOUT_MS)
            except Exception as e:  # timeout / lost packet: skip this frame, keep going
                print(f"\nFrame {i}: no buffer ({e}), skipped")
                continue
            try:
                if buf.is_incomplete:
                    print(f"\nFrame {i}: incomplete, skipped")
                    continue
                h, w = buf.height, buf.width
                n_bytes = h * w * 8  # 4 channels x 16 bit
                raw = np.ctypeslib.as_array(
                    ctypes.cast(buf.pdata, ctypes.POINTER(ctypes.c_uint8)), shape=(n_bytes,)
                )
                frame = raw.view(np.uint16).reshape(h, w, 4)  # view into the buffer, no copy

                xyz, gray, pixel = to_point_cloud(frame, scale, offset)
                path = out_dir / f"frame_{i:04d}.ply"
                write_ply(path, xyz, gray, pixel, w, h, read_temperature(device))
                saved += 1
                print(f"\rSaved frame {i}/{frame_count}: {path.name} ({len(xyz)} valid points)   ",
                      end="", flush=True)
            finally:
                device.requeue_buffer(buf)  # release only after the file is written
    print()
    temp = read_temperature(device)
    if temp is not None:
        print(f"Camera temperature at end: {temp:.1f} C")
    return saved


# --------------------------------------------------------------------------- main

def main():
    if NUM_FRAMES < 1:
        raise SystemExit("NUM_FRAMES must be >= 1")

    out_dir = OUTPUT_DIR
    if USE_TIMESTAMP_SUBFOLDER:
        out_dir = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    saved = 0
    try:
        device = connect()
        configure(device, OPERATING_MODE, EXPOSURE, IMAGE_ACCUMULATION, SPATIAL_FILTER, CONFIDENCE_MIN)
        scale, offset = read_scale_offset(device)
        saved = capture_and_save(device, NUM_FRAMES, scale, offset, out_dir)
    finally:
        system.destroy_device()  # always release the camera, even on error / Ctrl+C
        print(f"{saved} frame(s) saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
