import sys, math, logging, numpy as np, open3d as o3d
import pathlib; sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1]), sys.argv[1]]
logging.basicConfig(level=logging.INFO)
from arena_api import system as msys
import helios1_dual_capture as h

H, W = 480, 640
def synth(seed, scale):
    rng = np.random.default_rng(seed)
    x_mm = rng.uniform(-400, 400, (H, W)); y_mm = rng.uniform(-300, 300, (H, W)); z_mm = rng.uniform(200, 1400, (H, W))
    raw = np.empty((H, W, 4), np.uint16)
    raw[..., 0] = np.round(x_mm / scale + 32768); raw[..., 1] = np.round(y_mm / scale + 32768)
    raw[..., 2] = np.round(z_mm / scale); raw[..., 3] = rng.integers(0, 4000, (H, W))
    invalid = rng.random((H, W)) < 0.2
    raw[invalid, 2] = 0xFFFF
    return raw, invalid

# 1) pure unpack math vs per-pixel reference
raw, inv = synth(0, 0.25)
T = h.CoordinateTransform([0.25]*3, [-8192, -8192, 0])
f = h.unpack_abcy16(raw, T)
assert f.depth_mm.shape == (H, W) and f.depth_mm.dtype == np.float32
assert f.intensity.dtype == np.uint16 and f.intensity_8bit.dtype == np.uint8
assert np.isnan(f.depth_mm[inv]).all() and not np.isnan(f.depth_mm[~inv]).any()
assert len(f.point_cloud.points) == (~inv).sum()
for (r, c) in [(0, 0), (100, 200), (479, 639), (240, 320)]:
    if inv[r, c]: continue
    a, b, cc, y = (int(v) for v in raw[r, c])
    exp = (a*0.25-8192, b*0.25-8192, cc*0.25)
    assert math.isclose(f.depth_mm[r, c], exp[2], abs_tol=1e-3)
    idx = np.flatnonzero(~inv.ravel()).tolist().index(r*W+c)
    assert np.allclose(np.asarray(f.point_cloud.points)[idx], exp)
pts = np.asarray(f.point_cloud.points)
print("unpack ok: N=%d x[%.1f,%.1f] z[%.1f,%.1f]" % (len(pts), pts[:,0].min(), pts[:,0].max(), pts[:,2].min(), pts[:,2].max()))
fz = h.unpack_abcy16(raw, T, invalid_depth_value=0.0); assert (fz.depth_mm[inv] == 0).all()

# 2) merge
Tm = np.eye(4); Tm[:3, 3] = [100, 0, 0]
m = h.merge_point_clouds(f.point_cloud, f.point_cloud, Tm)
assert len(m.points) == 2*len(pts)
assert np.allclose(np.asarray(m.points)[len(pts):], pts + [100, 0, 0])
assert np.allclose(np.asarray(f.point_cloud.points), pts)  # input untouched
print("merge ok")

# 3) full rig against mock SDK, serials reversed vs sort order
d1 = msys.Device("222", "HLS003S-001"); d2 = msys.Device("111", "HLS003S-001"); d3 = msys.Device("999", "PHX050S-MC")
d1.raw, _ = synth(1, 0.25); d2.raw, _ = synth(2, 0.25)
msys.system.devices = [d3, d2, d1]
cfg = h.HeliosConfig(confidence_threshold_min=300, image_accumulation=2)
with h.DualHelios1System(cfg, serial1="222", serial2="111", discovery_wait_s=0) as rig:
    assert rig.cam1.serial == "222" and rig.cam2.serial == "111"
    for dev in (d1, d2):
        g = lambda k: dev.nodemap.get_node(k).value
        assert g("PixelFormat") == "Coord3D_ABCY16" and g("TriggerMode") == "On" and g("TriggerSource") == "Software"
        assert g("Scan3dConfidenceThresholdMin") == 300 and g("Scan3dImageAccumulation") == 2 and g("Scan3dSpatialFilterEnable") is True
        assert dev.tl_stream_nodemap.get_node("StreamBufferHandlingMode").value == "NewestOnly"
    assert np.allclose(rig.cam1.transform.offset, [-8192, -8192, 0])
    msys.EVENTS.clear()
    a, b = rig.capture_pair()
    ev = [e for e in msys.EVENTS]
    assert ev == [("222","trigger"),("222","get_buffer"),("222","requeue"),("111","trigger"),("111","get_buffer"),("111","requeue")], ev
    assert a.serial == "222" and b.serial == "111"
    view = h.compose_view(a, b, cfg.display_range_mm, 10.0)
    assert view.shape == (960, 1280, 3)
print("rig ok; teardown events:", msys.EVENTS[-3:])
assert msys.EVENTS[-3:] == [("222","stop"),("111","stop"),("system","destroy")]

# 4) exception inside the with-block still tears down
msys.EVENTS.clear()
try:
    with h.DualHelios1System(h.HeliosConfig(), discovery_wait_s=0) as rig:
        assert rig.cam1.serial == "111"  # ascending-serial default
        raise KeyError("user code crashed")
except KeyError: pass
assert msys.EVENTS[-3:] == [("111","stop"),("222","stop"),("system","destroy")], msys.EVENTS
# 5) failure during open (cam2 start_stream) -> cam1 stopped, devices destroyed
msys.EVENTS.clear(); d1.fail_on_start = True
try:
    with h.DualHelios1System(h.HeliosConfig(), discovery_wait_s=0): pass
except RuntimeError as e: print("open failure propagated:", e)
assert msys.EVENTS == [("111","start"),("111","stop"),("system","destroy")], msys.EVENTS
d1.fail_on_start = False
# 6) missing TriggerArmed and wrong serials / Helios2 enum rejected
d1b = msys.Device("333", "HLS003S-001", has_armed=False); d1b.raw = d1.raw
msys.system.devices = [d1b, d2]
with h.DualHelios1System(h.HeliosConfig(), discovery_wait_s=0) as rig:
    rig.capture_pair()
for kw in [dict(serial1="111", serial2="nope")]:
    try:
        with h.DualHelios1System(h.HeliosConfig(), discovery_wait_s=0, **kw): pass
        raise AssertionError
    except h.HeliosDiscoveryError as e: print("discovery error ok:", e)
try: h.HeliosConfig(operating_mode="Distance3000mm").validate(); raise AssertionError
except ValueError as e: print("config reject ok")
# 7) Distance6000mm scale picked up
msys.system.devices = [d1, d2]
with h.DualHelios1System(h.HeliosConfig(operating_mode="Distance6000mm"), discovery_wait_s=0) as rig:
    assert np.allclose(rig.cam1.transform.scale, 1.0)
print("ALL TESTS PASSED")
