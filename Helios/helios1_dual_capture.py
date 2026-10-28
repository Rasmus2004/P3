#!/usr/bin/env python3
"""
Dual LUCID Helios 1 (HLS003S-001) Time-of-Flight capture.

Runs two first-generation Helios cameras with sequential software triggering so
their 850 nm VCSEL illuminators never fire at the same time, unpacks
Coord3D_ABCY16 frames into millimetre depth maps, intensity images and Open3D
point clouds, and can merge the two clouds with a 4x4 extrinsic.

Only GenICam nodes from the original HLS003S firmware (v1.18+) are used. No
Helios2 nodes (Scan3dCommunicationChannel, Scan3dFlyingPixelsFilterEnable,
Distance1250mm/Distance3000mm, *SingleFreq/*MultiFreq) are referenced.

Requirements: Python 3.11, arena_api (LUCID Arena SDK wheel), numpy,
opencv-python, open3d.

Usage:
    python helios1_dual_capture.py --serial1 223300001 --serial2 223300002
    python helios1_dual_capture.py --mode Distance6000mm --exposure Exp250Us \
        --extrinsic T_cam2_to_cam1.npy

Keys in the live window:
    p   save the merged point cloud as .ply and open it in Open3D
    q   quit (Esc also works)
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Sequence

import cv2
import numpy as np
import open3d as o3d

log = logging.getLogger("helios1_dual")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HELIOS1_MODEL_PREFIX = "HLS003S"
PIXEL_FORMAT = "Coord3D_ABCY16"
CHANNELS = 4  # A (X), B (Y), C (Z), Y (intensity)
INVALID_RAW_Z = 0xFFFF

# Enum strings valid on Helios 1 firmware.
HELIOS1_OPERATING_MODES = ("Distance1500mm", "Distance6000mm")
HELIOS1_EXPOSURE_TIMES = ("Exp1000Us", "Exp250Us", "Exp62_5Us")

# Useful maximum range per operating mode, used for depth colorization only.
MODE_DISPLAY_RANGE_MM = {"Distance1500mm": 1500.0, "Distance6000mm": 6000.0}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HeliosError(RuntimeError):
    """Base error for this module."""


class HeliosDiscoveryError(HeliosError):
    """The expected Helios 1 cameras were not found."""


class HeliosConfigError(HeliosError):
    """A node is missing, not writable, or rejected a value."""


class HeliosCaptureError(HeliosError):
    """Triggering or buffer retrieval failed."""


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------


@dataclass
class HeliosConfig:
    """Acquisition settings applied identically to both cameras."""

    operating_mode: str = "Distance1500mm"
    exposure_time: str = "Exp1000Us"
    confidence_threshold_enable: bool = True
    confidence_threshold_min: int = 500
    spatial_filter_enable: bool = True
    image_accumulation: int = 1
    buffer_timeout_ms: int = 2000
    trigger_armed_timeout_s: float = 2.0
    invalid_depth_value: float = float("nan")  # np.nan or 0.0

    def validate(self) -> None:
        if self.operating_mode not in HELIOS1_OPERATING_MODES:
            raise ValueError(
                f"operating_mode must be one of {HELIOS1_OPERATING_MODES} on Helios 1, "
                f"got {self.operating_mode!r}"
            )
        if self.exposure_time not in HELIOS1_EXPOSURE_TIMES:
            raise ValueError(
                f"exposure_time must be one of {HELIOS1_EXPOSURE_TIMES} on Helios 1, "
                f"got {self.exposure_time!r}"
            )
        if self.image_accumulation < 1:
            raise ValueError("image_accumulation must be >= 1")
        if self.confidence_threshold_min < 0:
            raise ValueError("confidence_threshold_min must be >= 0")

    @property
    def display_range_mm(self) -> float:
        return MODE_DISPLAY_RANGE_MM.get(self.operating_mode, 1500.0)


@dataclass(frozen=True)
class CoordinateTransform:
    """Per-axis scale and offset that turn raw ABC counts into millimetres."""

    scale: np.ndarray  # (3,) float64: s_x, s_y, s_z
    offset: np.ndarray  # (3,) float64: o_x, o_y, o_z

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", np.asarray(self.scale, dtype=np.float64).reshape(3))
        object.__setattr__(self, "offset", np.asarray(self.offset, dtype=np.float64).reshape(3))


@dataclass
class CameraFrame:
    """One unpacked Helios frame."""

    serial: str
    depth_mm: np.ndarray  # (H, W) float32, invalid = config.invalid_depth_value
    intensity: np.ndarray  # (H, W) uint16
    intensity_8bit: np.ndarray  # (H, W) uint8, min-max normalized for display
    valid_mask: np.ndarray  # (H, W) bool
    point_cloud: o3d.geometry.PointCloud
    timestamp_ns: int = 0
    host_time_s: float = field(default_factory=time.time)

    @property
    def num_valid(self) -> int:
        return int(self.valid_mask.sum())


# ---------------------------------------------------------------------------
# GenICam node helpers
# ---------------------------------------------------------------------------


def _find_node(nodemap: Any, name: str) -> Any | None:
    """Return the node or None if the firmware does not expose it."""
    try:
        node = nodemap.get_node(name)
    except Exception:  # arena_api raises on unknown names in some versions
        return None
    return node


def set_node(nodemap: Any, name: str, value: Any, *, required: bool = True) -> bool:
    """Write a node value with a clear error if the camera rejects it.

    Returns True if written. With required=False a missing/unwritable node is
    logged as a warning and False is returned instead of raising.
    """
    node = _find_node(nodemap, name)
    if node is None:
        msg = f"node {name!r} not found on this firmware"
        if required:
            raise HeliosConfigError(msg)
        log.warning(msg)
        return False

    if not getattr(node, "is_writable", True):
        msg = f"node {name!r} is not writable (is the stream already running?)"
        if required:
            raise HeliosConfigError(msg)
        log.warning(msg)
        return False

    entries = getattr(node, "enumentry_names", None)
    if isinstance(value, str) and entries is not None and value not in entries:
        raise HeliosConfigError(
            f"node {name!r} has no entry {value!r}; available: {list(entries)}"
        )

    try:
        node.value = value
    except Exception as exc:
        raise HeliosConfigError(f"failed to set {name!r} = {value!r}: {exc}") from exc
    log.debug("set %s = %r", name, value)
    return True


def get_node_value(nodemap: Any, name: str) -> Any:
    node = _find_node(nodemap, name)
    if node is None:
        raise HeliosConfigError(f"node {name!r} not found on this firmware")
    return node.value


# ---------------------------------------------------------------------------
# Vectorized unpacking (pure functions, no SDK dependency)
# ---------------------------------------------------------------------------


def buffer_to_abcy16(pdata: Any, height: int, width: int) -> np.ndarray:
    """View a Coord3D_ABCY16 buffer as (H, W, 4) uint16 and return a copy.

    The copy matters: the memory belongs to the SDK and is reused after
    requeue_buffer().
    """
    ptr = ctypes.cast(pdata, ctypes.POINTER(ctypes.c_uint16))
    view = np.ctypeslib.as_array(ptr, shape=(height, width, CHANNELS))
    return view.copy()


def normalize_intensity(intensity: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Min-max normalize a uint16 intensity image to uint8.

    When a mask is given, the range is taken from valid pixels only so that
    saturated invalid pixels do not compress the display range.
    """
    src = intensity.astype(np.float32)
    sample = src[valid_mask] if valid_mask is not None and valid_mask.any() else src
    lo, hi = float(sample.min(initial=0.0)), float(sample.max(initial=0.0))
    if hi <= lo:
        return np.zeros(intensity.shape, dtype=np.uint8)
    out = (src - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def unpack_abcy16(
    raw: np.ndarray,
    transform: CoordinateTransform,
    *,
    serial: str = "",
    invalid_depth_value: float = float("nan"),
    timestamp_ns: int = 0,
) -> CameraFrame:
    """Convert a raw (H, W, 4) uint16 ABCY16 frame into a CameraFrame.

    X_mm = A * s_x + o_x,  Y_mm = B * s_y + o_y,  Z_mm = C * s_z + o_z.
    Pixels whose raw C equals 0xFFFF are invalid (low confidence / no return).
    """
    if raw.ndim != 3 or raw.shape[2] != CHANNELS or raw.dtype != np.uint16:
        raise ValueError(f"expected (H, W, 4) uint16, got {raw.shape} {raw.dtype}")

    valid = raw[..., 2] != INVALID_RAW_Z

    # (H, W, 3) * (3,) + (3,) broadcasts per axis. float64 keeps sub-mm
    # precision for large offsets (e.g. -32768 * 0.25).
    xyz = raw[..., :3].astype(np.float64) * transform.scale + transform.offset

    depth_mm = xyz[..., 2].astype(np.float32)
    depth_mm[~valid] = invalid_depth_value

    intensity = np.ascontiguousarray(raw[..., 3])
    intensity_8bit = normalize_intensity(intensity, valid)

    points = xyz[valid]  # (N, 3) float64
    gray = intensity_8bit[valid].astype(np.float64) / 255.0
    colors = np.repeat(gray[:, None], 3, axis=1)  # (N, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return CameraFrame(
        serial=serial,
        depth_mm=depth_mm,
        intensity=intensity,
        intensity_8bit=intensity_8bit,
        valid_mask=valid,
        point_cloud=pcd,
        timestamp_ns=timestamp_ns,
    )


def merge_point_clouds(
    pcd_cam1: o3d.geometry.PointCloud,
    pcd_cam2: o3d.geometry.PointCloud,
    T_cam2_to_cam1: np.ndarray = np.eye(4),
) -> o3d.geometry.PointCloud:
    """Transform camera 2's cloud into camera 1's frame and concatenate.

    Neither input is modified. T_cam2_to_cam1 maps homogeneous points in the
    camera 2 frame (mm) to the camera 1 frame (mm).
    """
    T = np.asarray(T_cam2_to_cam1, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam2_to_cam1 must be 4x4, got {T.shape}")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("T_cam2_to_cam1 bottom row must be [0, 0, 0, 1]")

    cam2_in_cam1 = o3d.geometry.PointCloud(pcd_cam2)  # copy
    cam2_in_cam1.transform(T)
    merged = o3d.geometry.PointCloud(pcd_cam1)
    merged += cam2_in_cam1
    return merged


def load_extrinsic(path: str | Path | None) -> np.ndarray:
    """Load a 4x4 matrix from .npy or whitespace/comma separated text."""
    if path is None:
        return np.eye(4)
    p = Path(path)
    if p.suffix == ".npy":
        T = np.load(p)
    else:
        T = np.loadtxt(p, delimiter="," if p.suffix == ".csv" else None)
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{p} does not contain a 4x4 matrix (got {T.shape})")
    return T


# ---------------------------------------------------------------------------
# Single camera wrapper
# ---------------------------------------------------------------------------


class Helios1Camera:
    """One HLS003S device: configuration, trigger and buffer handling."""

    def __init__(self, device: Any, label: str, config: HeliosConfig) -> None:
        self.device = device
        self.label = label
        self.config = config
        self.serial = str(get_node_value(device.nodemap, "DeviceSerialNumber"))
        self.model = str(get_node_value(device.nodemap, "DeviceModelName"))
        self.transform: CoordinateTransform | None = None
        self.streaming = False
        self._has_trigger_armed = _find_node(device.nodemap, "TriggerArmed") is not None
        if not self._has_trigger_armed:
            log.warning(
                "%s (%s): TriggerArmed node not present; triggering without the armed check",
                label, self.serial,
            )

    def __repr__(self) -> str:
        return f"Helios1Camera({self.label}, serial={self.serial}, model={self.model})"

    # -- configuration -----------------------------------------------------

    def configure(self) -> None:
        """Apply stream, 3D and trigger settings. Must run before start_stream()."""
        cfg = self.config
        stream = self.device.tl_stream_nodemap
        set_node(stream, "StreamAutoNegotiatePacketSize", True)
        set_node(stream, "StreamPacketResendEnable", True)
        set_node(stream, "StreamBufferHandlingMode", "NewestOnly")

        nm = self.device.nodemap
        # Turn triggering off while changing format/mode; some nodes are
        # locked while a trigger is pending.
        set_node(nm, "TriggerSelector", "FrameStart")
        set_node(nm, "TriggerMode", "Off")

        set_node(nm, "PixelFormat", PIXEL_FORMAT)
        set_node(nm, "Scan3dOperatingMode", cfg.operating_mode)
        set_node(nm, "ExposureTimeSelector", cfg.exposure_time)

        set_node(nm, "Scan3dConfidenceThresholdEnable", cfg.confidence_threshold_enable)
        if cfg.confidence_threshold_enable:
            set_node(nm, "Scan3dConfidenceThresholdMin", int(cfg.confidence_threshold_min))
        set_node(nm, "Scan3dSpatialFilterEnable", cfg.spatial_filter_enable)
        set_node(nm, "Scan3dImageAccumulation", int(cfg.image_accumulation))

        set_node(nm, "TriggerSelector", "FrameStart")
        set_node(nm, "TriggerMode", "On")
        set_node(nm, "TriggerSource", "Software")

        self.transform = self.read_coordinate_transform()
        log.info(
            "%s (%s) configured: mode=%s exposure=%s scale=%s offset=%s",
            self.label, self.serial, cfg.operating_mode, cfg.exposure_time,
            np.round(self.transform.scale, 6).tolist(),
            np.round(self.transform.offset, 3).tolist(),
        )

    def read_coordinate_transform(self) -> CoordinateTransform:
        """Read Scan3dCoordinateScale/Offset for CoordinateA, B and C.

        Must run after PixelFormat and Scan3dOperatingMode are set, because
        the scale depends on both.
        """
        nm = self.device.nodemap
        scale, offset = [], []
        for coord in ("CoordinateA", "CoordinateB", "CoordinateC"):
            set_node(nm, "Scan3dCoordinateSelector", coord)
            scale.append(float(get_node_value(nm, "Scan3dCoordinateScale")))
            offset.append(float(get_node_value(nm, "Scan3dCoordinateOffset")))
        if any(s == 0.0 for s in scale):
            raise HeliosConfigError(f"{self.label}: zero coordinate scale read: {scale}")
        return CoordinateTransform(scale=np.array(scale), offset=np.array(offset))

    # -- streaming ---------------------------------------------------------

    def start_stream(self) -> None:
        self.device.start_stream()
        self.streaming = True
        log.info("%s (%s) stream started", self.label, self.serial)

    def stop_stream(self) -> None:
        if not self.streaming:
            return
        try:
            self.device.stop_stream()
            log.info("%s (%s) stream stopped", self.label, self.serial)
        finally:
            self.streaming = False

    # -- acquisition -------------------------------------------------------

    def wait_trigger_armed(self) -> None:
        if not self._has_trigger_armed:
            return
        node = self.device.nodemap.get_node("TriggerArmed")
        deadline = time.monotonic() + self.config.trigger_armed_timeout_s
        while not node.value:
            if time.monotonic() > deadline:
                raise HeliosCaptureError(
                    f"{self.label} ({self.serial}): TriggerArmed not true after "
                    f"{self.config.trigger_armed_timeout_s:.1f} s"
                )
            time.sleep(0.0005)

    def trigger_and_grab(self) -> CameraFrame:
        """Fire one software trigger, wait for its buffer and unpack it."""
        if not self.streaming:
            raise HeliosCaptureError(f"{self.label}: stream is not running")
        assert self.transform is not None, "configure() must run before capture"

        self.wait_trigger_armed()
        self.device.nodemap.get_node("TriggerSoftware").execute()

        try:
            buffer = self.device.get_buffer(timeout=self.config.buffer_timeout_ms)
        except Exception as exc:
            raise HeliosCaptureError(f"{self.label} ({self.serial}): get_buffer failed: {exc}") from exc

        try:
            if getattr(buffer, "is_incomplete", False):
                raise HeliosCaptureError(f"{self.label} ({self.serial}): incomplete buffer")
            raw = buffer_to_abcy16(buffer.pdata, buffer.height, buffer.width)
            timestamp_ns = int(getattr(buffer, "timestamp_ns", 0) or 0)
        finally:
            self.device.requeue_buffer(buffer)

        return unpack_abcy16(
            raw,
            self.transform,
            serial=self.serial,
            invalid_depth_value=self.config.invalid_depth_value,
            timestamp_ns=timestamp_ns,
        )


# ---------------------------------------------------------------------------
# Dual camera system
# ---------------------------------------------------------------------------


class DualHelios1System:
    """Owns both cameras and the arena_api device lifecycle.

    Use as a context manager so streams are stopped and devices destroyed on
    any exit path, including exceptions:

        with DualHelios1System(config, serial1="...", serial2="...") as rig:
            f1, f2 = rig.capture_pair()
    """

    def __init__(
        self,
        config: HeliosConfig | None = None,
        serial1: str | None = None,
        serial2: str | None = None,
        discovery_retries: int = 6,
        discovery_wait_s: float = 5.0,
    ) -> None:
        self.config = config or HeliosConfig()
        self.config.validate()
        self.serial1 = serial1
        self.serial2 = serial2
        self.discovery_retries = discovery_retries
        self.discovery_wait_s = discovery_wait_s
        self.cam1: Helios1Camera | None = None
        self.cam2: Helios1Camera | None = None
        self._system: Any = None
        self._devices: list[Any] = []

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "DualHelios1System":
        try:
            self.open()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        from arena_api.system import system  # imported lazily so the module loads without the SDK

        self._system = system
        self._devices = self._create_devices()
        dev1, dev2 = self._select_devices(self._devices)
        self.cam1 = Helios1Camera(dev1, "Camera 1", self.config)
        self.cam2 = Helios1Camera(dev2, "Camera 2", self.config)
        for cam in (self.cam1, self.cam2):
            cam.configure()
        for cam in (self.cam1, self.cam2):
            cam.start_stream()

    def close(self) -> None:
        """Stop streams and destroy devices. Safe to call more than once."""
        for cam in (self.cam1, self.cam2):
            if cam is None:
                continue
            try:
                cam.stop_stream()
            except Exception:
                log.exception("error stopping stream on %s", cam.label)
        if self._system is not None and self._devices:
            try:
                self._system.destroy_device()  # destroys every device this process created
                log.info("devices destroyed")
            except Exception:
                log.exception("error destroying devices")
        self._devices = []
        self.cam1 = self.cam2 = None

    def _create_devices(self) -> list[Any]:
        for attempt in range(1, self.discovery_retries + 1):
            try:
                devices = list(self._system.create_device())
            except Exception as exc:  # some SDK versions raise when nothing is found
                log.debug("create_device raised: %s", exc)
                devices = []
            if devices:
                log.info("found %d device(s)", len(devices))
                return devices
            log.info(
                "no devices found (attempt %d/%d), retrying in %.0f s",
                attempt, self.discovery_retries, self.discovery_wait_s,
            )
            time.sleep(self.discovery_wait_s)
        raise HeliosDiscoveryError("no LUCID devices found; check power, cabling and NIC IP settings")

    def _select_devices(self, devices: Sequence[Any]) -> tuple[Any, Any]:
        """Pick two HLS003S devices and order them deterministically."""
        helios: dict[str, Any] = {}
        for dev in devices:
            model = str(get_node_value(dev.nodemap, "DeviceModelName"))
            serial = str(get_node_value(dev.nodemap, "DeviceSerialNumber"))
            if model.startswith(HELIOS1_MODEL_PREFIX):
                helios[serial] = dev
                log.info("  %s  serial=%s", model, serial)
            else:
                log.info("  ignoring %s serial=%s (not a Helios 1)", model, serial)

        if len(helios) < 2:
            raise HeliosDiscoveryError(
                f"need 2 {HELIOS1_MODEL_PREFIX} cameras, found {len(helios)}: {sorted(helios)}"
            )

        if self.serial1 or self.serial2:
            if not (self.serial1 and self.serial2):
                raise HeliosDiscoveryError("give both serial1 and serial2, or neither")
            if self.serial1 == self.serial2:
                raise HeliosDiscoveryError("serial1 and serial2 must differ")
            missing = [s for s in (self.serial1, self.serial2) if s not in helios]
            if missing:
                raise HeliosDiscoveryError(
                    f"serial(s) {missing} not found; connected Helios 1 serials: {sorted(helios)}"
                )
            return helios[self.serial1], helios[self.serial2]

        if len(helios) > 2:
            raise HeliosDiscoveryError(
                f"{len(helios)} Helios 1 cameras connected {sorted(helios)}; "
                "pass serial1/serial2 to choose two"
            )
        s1, s2 = sorted(helios)
        log.warning(
            "no serials given; using ascending serial order: Camera 1=%s, Camera 2=%s. "
            "Pass --serial1/--serial2 to pin the mapping.", s1, s2,
        )
        return helios[s1], helios[s2]

    # -- acquisition -------------------------------------------------------

    def capture_pair(self) -> tuple[CameraFrame, CameraFrame]:
        """Capture one frame from each camera, strictly one after the other.

        Camera 2 is only triggered after Camera 1's buffer has arrived, so its
        exposure is finished and the two illuminators never overlap.
        """
        if self.cam1 is None or self.cam2 is None:
            raise HeliosCaptureError("system is not open")
        frame1 = self.cam1.trigger_and_grab()
        frame2 = self.cam2.trigger_and_grab()
        return frame1, frame2

    @staticmethod
    def merge_point_clouds(
        pcd_cam1: o3d.geometry.PointCloud,
        pcd_cam2: o3d.geometry.PointCloud,
        T_cam2_to_cam1: np.ndarray = np.eye(4),
    ) -> o3d.geometry.PointCloud:
        return merge_point_clouds(pcd_cam1, pcd_cam2, T_cam2_to_cam1)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def colorize_depth(depth_mm: np.ndarray, max_mm: float, min_mm: float = 0.0) -> np.ndarray:
    """Map depth to a BGR colormap; invalid (nan or <= 0) pixels are black."""
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    d = np.where(valid, depth_mm, 0.0)
    norm = np.clip((d - min_mm) / max(max_mm - min_mm, 1e-6), 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def compose_view(f1: CameraFrame, f2: CameraFrame, max_mm: float, fps: float) -> np.ndarray:
    """2x2 mosaic: depth on top, intensity below, camera 1 left, camera 2 right."""
    tiles = []
    for name, f in (("Cam1", f1), ("Cam2", f2)):
        depth = _label(colorize_depth(f.depth_mm, max_mm), f"{name} {f.serial} depth  valid={f.num_valid}")
        inten = _label(cv2.cvtColor(f.intensity_8bit, cv2.COLOR_GRAY2BGR), f"{name} {f.serial} intensity")
        tiles.append((depth, inten))
    top = np.hstack([tiles[0][0], tiles[1][0]])
    bottom = np.hstack([tiles[0][1], tiles[1][1]])
    view = np.vstack([top, bottom])
    return _footer(view, fps)


def _footer(view: np.ndarray, fps: float) -> np.ndarray:
    text = f"{fps:5.1f} pairs/s   [p] save + view merged cloud   [q] quit"
    y = view.shape[0] - 12
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return view


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual Helios 1 (HLS003S) sequential-trigger viewer")
    p.add_argument("--serial1", help="DeviceSerialNumber of Camera 1 (reference frame)")
    p.add_argument("--serial2", help="DeviceSerialNumber of Camera 2")
    p.add_argument("--mode", default="Distance1500mm", choices=HELIOS1_OPERATING_MODES)
    p.add_argument("--exposure", default="Exp1000Us", choices=HELIOS1_EXPOSURE_TIMES)
    p.add_argument("--confidence-min", type=int, default=500, help="Scan3dConfidenceThresholdMin")
    p.add_argument("--no-confidence", action="store_true", help="disable the confidence threshold")
    p.add_argument("--no-spatial-filter", action="store_true")
    p.add_argument("--accumulation", type=int, default=1, help="Scan3dImageAccumulation")
    p.add_argument("--extrinsic", help="4x4 T_cam2_to_cam1 in mm (.npy, .txt or .csv)")
    p.add_argument("--zero-invalid", action="store_true", help="invalid depth = 0.0 instead of NaN")
    p.add_argument("--output-dir", default="captures", help="where .ply files are written")
    p.add_argument("--timeout-ms", type=int, default=2000, help="get_buffer timeout per camera")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    config = HeliosConfig(
        operating_mode=args.mode,
        exposure_time=args.exposure,
        confidence_threshold_enable=not args.no_confidence,
        confidence_threshold_min=args.confidence_min,
        spatial_filter_enable=not args.no_spatial_filter,
        image_accumulation=args.accumulation,
        buffer_timeout_ms=args.timeout_ms,
        invalid_depth_value=0.0 if args.zero_invalid else float("nan"),
    )
    T_cam2_to_cam1 = load_extrinsic(args.extrinsic)
    out_dir = Path(args.output_dir)
    window = "Helios 1 dual (sequential trigger)"
    max_consecutive_failures = 10

    with DualHelios1System(config, serial1=args.serial1, serial2=args.serial2) as rig:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        failures = 0
        fps = 0.0
        last = time.perf_counter()
        while True:
            try:
                f1, f2 = rig.capture_pair()
                failures = 0
            except HeliosCaptureError as exc:
                failures += 1
                log.warning("capture failed (%d/%d): %s", failures, max_consecutive_failures, exc)
                if failures >= max_consecutive_failures:
                    raise
                continue

            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            last = now

            cv2.imshow(window, compose_view(f1, f2, config.display_range_mm, fps))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                merged = rig.merge_point_clouds(f1.point_cloud, f2.point_cloud, T_cam2_to_cam1)
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"merged_{datetime.now():%Y%m%d_%H%M%S}.ply"
                o3d.io.write_point_cloud(str(path), merged)
                log.info("saved %d points to %s; close the Open3D window to resume", len(merged.points), path)
                o3d.visualization.draw_geometries(
                    [merged, o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)],
                    window_name=f"Merged cloud: {path.name}",
                )
                last = time.perf_counter()  # do not count viewer time in fps
            # Window closed with the X button.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
