import ctypes
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from arena_api.system import system


@dataclass
class HeliosScaleOffset:
    """Stores the GenICam 3D coordinate transformation parameters for a single camera."""
    scale_x: float
    offset_x: float
    scale_y: float
    offset_y: float
    scale_z: float
    offset_z: float


@dataclass
class HeliosFrameData:
    """Container for unpacked 2D and 3D data from a single Helios 1 capture."""
    serial_number: str
    depth_mm: np.ndarray          # (480, 640) float32, invalid pixels = np.nan
    depth_vis: np.ndarray         # (480, 640, 3) uint8 BGR Turbo/Jet colormap
    intensity_u16: np.ndarray     # (480, 640) uint16 raw IR amplitude
    intensity_vis: np.ndarray     # (480, 640) uint8 normalized for OpenCV display
    point_cloud: o3d.geometry.PointCloud


class DualHelios1Manager:
    """
    Manages two first-generation LUCID Helios 1 (HLS003S-001) ToF cameras.
    Uses sequential software triggering to prevent 850nm VCSEL interference.
    """

    def __init__(
        self,
        cam1_serial: Optional[str] = None,
        cam2_serial: Optional[str] = None,
        operating_mode: str = "Distance1500mm",
        exposure_selector: str = "Exp1000Us",
        confidence_min: int = 500,
        spatial_filter: bool = True,
        image_accumulation: int = 1,
    ):
        """
        Args:
            cam1_serial: Optional exact serial number for Camera 1.
            cam2_serial: Optional exact serial number for Camera 2.
            operating_mode: Helios 1 mode: 'Distance1500mm' or 'Distance6000mm'.
            exposure_selector: Helios 1 exposure: 'Exp1000Us', 'Exp250Us', or 'Exp62_5Us'.
            confidence_min: Minimum amplitude threshold to accept a pixel.
            spatial_filter: Enable on-camera spatial smoothing filter.
            image_accumulation: Number of frames accumulated on-camera per trigger (1 to 4).
        """
        self.cam1_serial = cam1_serial
        self.cam2_serial = cam2_serial
        self.operating_mode = operating_mode
        self.exposure_selector = exposure_selector
        self.confidence_min = confidence_min
        self.spatial_filter = spatial_filter
        self.image_accumulation = image_accumulation

        self.devices: List = []
        self.scales_offsets: Dict[str, HeliosScaleOffset] = {}
        self._is_streaming = False

        # Set max visualization range based on operating mode
        self.max_depth_mm = 1500.0 if "1500" in operating_mode else 6000.0

    def __enter__(self) -> "DualHelios1Manager":
        self.initialize_and_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def initialize_and_start(self) -> None:
        """Discovers, sorts, configures, and starts streaming on two HLS003S cameras."""
        all_devices = system.create_device()
        if not all_devices:
            raise RuntimeError("No LUCID Arena devices detected on the network.")

        # Filter strictly for first-generation Helios 1 (HLS003S) cameras
        helios1_devices = []
        for dev in all_devices:
            model_name = str(dev.nodemap.get_node("DeviceModelName").value)
            if model_name.startswith("HLS003S"):
                helios1_devices.append(dev)
            else:
                print(f"[Warning] Ignoring non-HLS003S device: {model_name}")

        if len(helios1_devices) < 2:
            system.destroy_device()
            raise RuntimeError(
                f"Expected at least 2 HLS003S cameras, found {len(helios1_devices)}."
            )

        # Map cameras deterministically by Serial Number
        self.devices = self._order_devices(helios1_devices)

        for idx, dev in enumerate(self.devices):
            sn = str(dev.nodemap.get_node("DeviceSerialNumber").value)
            print(f"[Init] Configuring Camera {idx + 1} (S/N: {sn})...")
            self._configure_helios1_device(dev)
            self.scales_offsets[sn] = self._read_scale_and_offset(dev)

        # Start acquisition streams on both cameras (waiting for Software Trigger)
        for dev in self.devices:
            dev.start_stream()
        self._is_streaming = True
        print("[Init] Both HLS003S cameras started and armed for sequential triggering.")

    def _order_devices(self, discovered: List) -> List:
        """Orders the two cameras by explicit user serial numbers or ascending serial order."""
        sn_map = {
            str(d.nodemap.get_node("DeviceSerialNumber").value): d for d in discovered
        }

        if self.cam1_serial and self.cam2_serial:
            if self.cam1_serial not in sn_map or self.cam2_serial not in sn_map:
                system.destroy_device()
                raise ValueError(
                    f"Requested serials ({self.cam1_serial}, {self.cam2_serial}) "
                    f"not found among connected HLS003S serials: {list(sn_map.keys())}"
                )
            return [sn_map[self.cam1_serial], sn_map[self.cam2_serial]]

        # Default deterministic sorting by serial number string
        sorted_serials = sorted(sn_map.keys())
        return [sn_map[sorted_serials[0]], sn_map[sorted_serials[1]]]

    def _configure_helios1_device(self, device) -> None:
        """Applies transport layer and GenICam 3D settings specific to HLS003S-001."""
        tl_map = device.tl_stream_nodemap
        tl_map["StreamAutoNegotiatePacketSize"].value = True
        tl_map["StreamPacketResendEnable"].value = True
        tl_map["StreamBufferHandlingMode"].value = "NewestOnly"

        nm = device.nodemap

        # Ensure TriggerMode is Off while changing core operating parameters
        nm["TriggerMode"].value = "Off"

        # 4-channel 16-bit unsigned: X (A), Y (B), Z (C), Intensity (Y)
        nm["PixelFormat"].value = "Coord3D_ABCY16"

        # Helios 1 specific Operating Mode ('Distance1500mm' or 'Distance6000mm')
        nm["Scan3dOperatingMode"].value = self.operating_mode

        # Helios 1 Exposure Time Selector ('Exp1000Us', 'Exp250Us', 'Exp62_5Us')
        nm["ExposureTimeSelector"].value = self.exposure_selector

        # Helios 1 On-Camera Filtering Nodes
        nm["Scan3dConfidenceThresholdEnable"].value = True
        nm["Scan3dConfidenceThresholdMin"].value = int(self.confidence_min)
        nm["Scan3dSpatialFilterEnable"].value = bool(self.spatial_filter)
        nm["Scan3dImageAccumulation"].value = int(self.image_accumulation)

        # Configure Software Triggering to eliminate dual-camera VCSEL crosstalk
        nm["TriggerSelector"].value = "FrameStart"
        nm["TriggerSource"].value = "Software"
        nm["TriggerMode"].value = "On"

    @staticmethod
    def _read_scale_and_offset(device) -> HeliosScaleOffset:
        """Reads CoordinateA, CoordinateB, and CoordinateC scale and offset factors."""
        nm = device.nodemap

        nm["Scan3dCoordinateSelector"].value = "CoordinateA"
        sx = float(nm["Scan3dCoordinateScale"].value)
        ox = float(nm["Scan3dCoordinateOffset"].value)

        nm["Scan3dCoordinateSelector"].value = "CoordinateB"
        sy = float(nm["Scan3dCoordinateScale"].value)
        oy = float(nm["Scan3dCoordinateOffset"].value)

        nm["Scan3dCoordinateSelector"].value = "CoordinateC"
        sz = float(nm["Scan3dCoordinateScale"].value)
        oz = float(nm["Scan3dCoordinateOffset"].value)

        return HeliosScaleOffset(sx, ox, sy, oy, sz, oz)

    def _trigger_and_grab_raw(self, device, timeout_ms: int = 2000) -> Tuple[np.ndarray, str]:
        """
        Waits for TriggerArmed, fires TriggerSoftware, copies the 4-channel uint16
        buffer via ctypes, and immediately requeues the Arena buffer.
        """
        nm = device.nodemap
        sn = str(nm["DeviceSerialNumber"].value)

        # Poll TriggerArmed so we don't drop a software trigger command
        t0 = time.perf_counter()
        while not bool(nm["TriggerArmed"].value):
            if (time.perf_counter() - t0) * 1000.0 > timeout_ms:
                raise TimeoutError(f"Camera {sn} timed out waiting for TriggerArmed.")
            time.sleep(0.0005)

        # Fire laser and capture exclusively on this camera
        nm["TriggerSoftware"].execute()

        buffer = device.get_buffer(timeout=timeout_ms)
        try:
            if buffer.is_incomplete:
                raise RuntimeError(f"Incomplete GigE buffer received from Camera {sn}.")

            height = buffer.height
            width = buffer.width

            # Cast raw C pointer to uint16 array of shape (H, W, 4) -> [A, B, C, Y]
            ptr_u16 = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))
            raw_view = np.ctypeslib.as_array(ptr_u16, shape=(height, width, 4))

            # Copy into Python-managed memory before requeuing the SDK buffer
            raw_abcy = raw_view.copy()
        finally:
            device.requeue_buffer(buffer)

        return raw_abcy, sn

    def _unpack_abcy_buffer(self, raw_abcy: np.ndarray, sn: str) -> HeliosFrameData:
        """
        Vectorized conversion of raw (480, 640, 4) uint16 ABCY data into:
        - 2D Depth Map (mm) & Colormap
        - 2D Intensity Map
        - 3D Open3D PointCloud
        """
        cal = self.scales_offsets[sn]

        raw_a = raw_abcy[:, :, 0]
        raw_b = raw_abcy[:, :, 1]
        raw_c = raw_abcy[:, :, 2]
        raw_y = raw_abcy[:, :, 3]

        # On HLS003S, invalid / low-confidence pixels have raw C (Z) == 0xFFFF (65535)
        valid_mask = raw_c < 65535

        # Vectorized coordinate conversion to millimeters: Coord_mm = Raw * Scale + Offset
        x_mm = raw_a.astype(np.float32) * cal.scale_x + cal.offset_x
        y_mm = raw_b.astype(np.float32) * cal.scale_y + cal.offset_y
        z_mm = raw_c.astype(np.float32) * cal.scale_z + cal.offset_z

        # 1. 2D Depth Map (invalid pixels set to NaN)
        depth_mm = np.where(valid_mask, z_mm, np.nan)

        # Create 8-bit colorized depth visualization for OpenCV
        depth_norm = np.clip(z_mm / self.max_depth_mm, 0.0, 1.0)
        depth_u8 = (depth_norm * 255.0).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        depth_vis[~valid_mask] = (0, 0, 0)  # Black out invalid pixels

        # 2. 2D Intensity Image (auto-scaled 99th percentile for clear display)
        p99 = float(np.percentile(raw_y, 99)) if raw_y.size > 0 else 1.0
        p99 = max(p99, 1.0)
        intensity_vis = np.clip((raw_y.astype(np.float32) / p99) * 255.0, 0, 255).astype(
            np.uint8
        )

        # 3. 3D Point Cloud (filtered to valid points only)
        xyz_Stack = np.stack((x_mm, y_mm, z_mm), axis=-1)
        valid_points = xyz_Stack[valid_mask].astype(np.float64)

        # Map normalized grayscale intensity to RGB [0.0, 1.0] for Open3D point colors
        valid_intensity = np.clip(raw_y[valid_mask].astype(np.float64) / p99, 0.0, 1.0)
        valid_colors = np.column_stack((valid_intensity, valid_intensity, valid_intensity))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd.colors = o3d.utility.Vector3dVector(valid_colors)

        return HeliosFrameData(
            serial_number=sn,
            depth_mm=depth_mm,
            depth_vis=depth_vis,
            intensity_u16=raw_y,
            intensity_vis=intensity_vis,
            point_cloud=pcd,
        )

    def capture_pair(self) -> Tuple[HeliosFrameData, HeliosFrameData]:
        """
        Sequentially triggers Camera 1, grabs its frame, then triggers Camera 2
        and grabs its frame, unpacking both in parallel-ready NumPy arrays.
        """
        if not self._is_streaming or len(self.devices) < 2:
            raise RuntimeError("Cameras are not initialized and streaming.")

        # Step 1: Fire Camera 1 laser & read out
        raw1, sn1 = self._trigger_and_grab_raw(self.devices[0])
        # Step 2: Immediately fire Camera 2 laser & read out
        raw2, sn2 = self._trigger_and_grab_raw(self.devices[1])

        # Step 3: Vectorized CPU unpacking
        frame1 = self._unpack_abcy_buffer(raw1, sn1)
        frame2 = self._unpack_abcy_buffer(raw2, sn2)

        return frame1, frame2

    @staticmethod
    def merge_point_clouds(
        pcd_cam1: o3d.geometry.PointCloud,
        pcd_cam2: o3d.geometry.PointCloud,
        T_cam2_to_cam1: np.ndarray = np.eye(4, dtype=np.float64),
    ) -> o3d.geometry.PointCloud:
        """
        Transforms Camera 2's point cloud into Camera 1's coordinate frame using
        a 4x4 homogeneous extrinsic matrix and returns the unified PointCloud.
        """
        pcd2_transformed = o3d.geometry.PointCloud(pcd_cam2)
        pcd2_transformed.transform(T_cam2_to_cam1)
        return pcd_cam1 + pcd2_transformed

    def close(self) -> None:
        """Stops streams, restores continuous trigger mode, and destroys Arena devices."""
        for dev in self.devices:
            try:
                if self._is_streaming:
                    dev.stop_stream()
                dev.nodemap["TriggerMode"].value = "Off"
            except Exception as exc:
                print(f"[Cleanup Warning] Error stopping device: {exc}")

        self._is_streaming = False
        self.devices.clear()
        system.destroy_device()
        print("[Cleanup] All Arena devices stopped and destroyed.")


def main() -> None:
    # Optional: Hardcode your two HLS003S serial numbers so Cam 1 and Cam 2 never swap
    CAM1_SERIAL = None  # e.g., "20120001"
    CAM2_SERIAL = None  # e.g., "20120002"

    # Extrinsic calibration matrix (4x4 homogeneous transform in millimeters)
    # Replace the translation/rotation below with your calibrated Cam2 -> Cam1 extrinsics
    T_cam2_to_cam1 = np.array(
        [
            [1.0, 0.0, 0.0, 250.0],  # Example: Cam 2 shifted +250 mm along X
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    with DualHelios1Manager(
        cam1_serial=CAM1_SERIAL,
        cam2_serial=CAM2_SERIAL,
        operating_mode="Distance1500mm",  # 'Distance1500mm' or 'Distance6000mm'
        exposure_selector="Exp1000Us",    # 'Exp1000Us', 'Exp250Us', or 'Exp62_5Us'
        confidence_min=500,
        spatial_filter=True,
        image_accumulation=1,
    ) as manager:
        print("\n--- Live Stream Controls ---")
        print("Press 'p' to view and save the combined 3D point cloud (.ply)")
        print("Press 'q' or ESC to exit\n")

        while True:
            t_start = time.perf_counter()
            frame1, frame2 = manager.capture_pair()
            fps = 1.0 / max(time.perf_counter() - t_start, 1e-6)

            # Convert single-channel 8-bit intensity maps to 3-channel BGR for stacking
            ir1_bgr = cv2.cvtColor(frame1.intensity_vis, cv2.COLOR_GRAY2BGR)
            ir2_bgr = cv2.cvtColor(frame2.intensity_vis, cv2.COLOR_GRAY2BGR)

            # Top row: Camera 1 & Camera 2 Depth | Bottom row: Camera 1 & Camera 2 Intensity
            top_row = np.hstack((frame1.depth_vis, frame2.depth_vis))
            bottom_row = np.hstack((ir1_bgr, ir2_bgr))
            dashboard = np.vstack((top_row, bottom_row))

            # Overlay status text
            cv2.putText(
                dashboard,
                f"Cam 1 (S/N: {frame1.serial_number}) | Pair FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                dashboard,
                f"Cam 2 (S/N: {frame2.serial_number})",
                (640 + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Resize dashboard by 0.75x if screen space is tight (1280x960 -> 960x720)
            dashboard_display = cv2.resize(dashboard, (960, 720))
            cv2.imshow("Dual LUCID Helios 1 (HLS003S-001) - Depth & IR", dashboard_display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                merged_pcd = manager.merge_point_clouds(
                    frame1.point_cloud, frame2.point_cloud, T_cam2_to_cam1
                )
                ply_filename = f"dual_helios1_{int(time.time())}.ply"
                o3d.io.write_point_cloud(ply_filename, merged_pcd)
                print(
                    f"[Saved] {ply_filename} with {len(merged_pcd.points)} valid 3D points."
                )

                # Add a 100 mm coordinate frame at Camera 1's origin for orientation
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
                o3d.visualization.draw_geometries(
                    [merged_pcd, coord_frame],
                    window_name="Merged Dual Helios 1 Point Cloud (mm)",
                )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()