import cv2
import os
import struct
import threading
import time
import platform
import glob
import logging
from config.settings_manager import settings

logger = logging.getLogger("CameraManager")

_MAX_RECONNECT_DELAY = 30.0
_EMPTY_DISCOVERY_TTL = 5.0
_VIDIOC_QUERYCAP = 0x80685600  # _IOR('V', 0, struct v4l2_capability), a 104-byte struct
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
_V4L2_CAP_VIDEO_M2M_MPLANE = 0x00004000
_V4L2_CAP_VIDEO_M2M = 0x00008000
_V4L2_CAP_IO_MC = 0x20000000
_V4L2_CAP_DEVICE_CAPS = 0x80000000
# Qualcomm camera server (CamX) control nodes: single-open devices that the server holds. Even a failed open
# contends with it, and an open while it restarts would take its slot, so discovery never opens them.
_CAMERA_SERVER_NODES = {"cam-req-mgr", "cam_sync"}
# Brightest sampled pixel value still counted as black. While another app streams from the same camera,
# DirectShow delivers all-zero frames (about one per second) instead of failing, and keeps doing so after
# that app lets go; only reopening the camera brings the picture back.
_BLACK_LEVEL = 4
_BLACK_REOPEN_SECONDS = 10.0
# Only that slow black stream is reopened: a dark room still streams at the camera's normal rate.
_BUSY_MAX_FPS = 3.0


def is_black(frame) -> bool:
    return frame.size > 0 and int(frame[::16, ::16].max()) <= _BLACK_LEVEL


def v4l2_can_capture(dev):
    """Whether a V4L2 node captures video; None when it cannot be opened or queried.

    Linux lists codecs, UVC metadata nodes and camera-server nodes (Qualcomm CamX) as /dev/video* too.
    """
    try:
        import fcntl

        fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    except (ImportError, OSError) as e:
        logger.info(f"Cannot query {dev}: {e}")
        return None
    try:
        caps, device_caps = struct.unpack_from("II", fcntl.ioctl(fd, _VIDIOC_QUERYCAP, bytes(104)), 84)
    except OSError as e:
        logger.info(f"Cannot query {dev}: {e}")
        return None
    finally:
        os.close(fd)
    if caps & _V4L2_CAP_DEVICE_CAPS:
        caps = device_caps
    if caps & (_V4L2_CAP_VIDEO_M2M | _V4L2_CAP_VIDEO_M2M_MPLANE):
        return False
    if caps & _V4L2_CAP_IO_MC:
        return None  # media-controller node: streams only after media-ctl/libcamera configure the pipeline
    return bool(caps & (_V4L2_CAP_VIDEO_CAPTURE | _V4L2_CAP_VIDEO_CAPTURE_MPLANE))


class CameraManager:
    """Owns camera discovery and a single capture thread that publishes the latest frame."""

    def __init__(self, camera_index=None):
        self.os_type = platform.system()
        self.arch_type = platform.machine().lower()

        self.active_camera_info = None
        self._streaming_index = None

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._discovery_lock = threading.Lock()
        self._thread = None
        self._stop_event = None

        self.metrics = {
            "state": "stopped",
            "fps_capture": 0.0,
            "drops": 0,
            "active_backend": "None",
            "resolution": "Unknown",
            "reconnects": 0
        }
        
        self.available_cameras = []
        self._discovered_at = float("-inf")
        self._frames_captured = 0
        self._last_fps_time = time.monotonic()
        
    def discover_cameras(self, force=False):
        with self._discovery_lock:
            # An empty result is cached briefly too: probing opens devices and takes seconds.
            fresh = time.monotonic() - self._discovered_at < _EMPTY_DISCOVERY_TTL
            if not force and (self.available_cameras or fresh):
                return self.available_cameras
            self.available_cameras = self._probe_cameras()
            self._discovered_at = time.monotonic()
            return self.available_cameras

    def _probe_cameras(self):
        logger.info("Starting camera discovery...")
        cameras = []
        
        if self.os_type == "Linux":
            capture, unknown = [], []
            devices = glob.glob("/dev/video*")
            for dev in devices:
                try:
                    idx = int(dev.replace("/dev/video", ""))
                    name = dev
                    try:
                        with open(f"/sys/class/video4linux/video{idx}/name", "r") as f:
                            name = f.read().strip()
                    except Exception:
                        pass

                    can_capture = None if name in _CAMERA_SERVER_NODES else v4l2_can_capture(dev)
                    if can_capture is False:
                        logger.info(f"Skipping {dev} ({name}): not a video capture device")
                        continue
                    (capture if can_capture else unknown).append({
                        "path": dev,
                        "index": idx,
                        "name": name,
                        "type": "CSI" if "qcom" in name.lower() or "isp" in name.lower() else "USB/Virtual",
                        "status": "Available" if can_capture else "Unverified"
                    })
                except Exception as e:
                    logger.debug(f"Skipped {dev}: {e}")
            # The first entry is the default camera: a real capture node, never whatever glob listed first.
            cameras = sorted(capture, key=lambda c: c["index"]) + sorted(unknown, key=lambda c: c["index"])
                    
        else:
            # Windows/Mac basic probing. OpenCV's backends log an error for every index without a camera.
            log_level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
            try:
                for idx in range(5):
                    # macOS AVFoundation segfaults when a camera that is streaming on another thread is opened again.
                    if self.active_camera_info and self._streaming_index == idx:
                        cameras.append(self.active_camera_info)
                        continue

                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        cameras.append({
                            "path": str(idx),
                            "index": idx,
                            "name": f"Camera {idx}",
                            "type": "Integrated/USB",
                            "status": "Available"
                        })
                    cap.release()
            finally:
                cv2.utils.logging.setLogLevel(log_level)
                    
        logger.info(f"Discovery complete. Found {len(cameras)} cameras.")
        return cameras

    def _benchmark_pipeline(self, init_arg, backend, desc, stop_event=None):
        logger.info(f"Benchmarking {desc}...")
        cap = cv2.VideoCapture(init_arg, backend)
        
        if not cap.isOpened():
            logger.info(f"{desc} failed to open.")
            cap.release()
            return -1, None
            
        # Optimize OpenCV buffers
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frames = 0
        bench_start = time.monotonic()
        try:
            while time.monotonic() - bench_start < 2.0 and not (stop_event and stop_event.is_set()):
                ret, _ = cap.read()
                if ret:
                    frames += 1
                else:
                    break
        finally:
            cap.release()
                
        fps = frames / max(time.monotonic() - bench_start, 1e-3)
        logger.info(f"{desc} achieved {fps:.1f} FPS.")
        return fps, {"init_arg": init_arg, "backend": backend, "desc": desc}

    def _candidate_pipelines(self, camera_index):
        pipelines_to_try = []
        
        # Rubik Pi specific
        if self.os_type == "Linux" and "aarch64" in self.arch_type:
            pipelines_to_try.extend([
                (f"qtiqmmfsrc camera={camera_index} ! video/x-raw,format=NV12 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1", cv2.CAP_GSTREAMER, "QTI QMMF HW"),
                (f"v4l2src device=/dev/video{camera_index} ! qtimididec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1", cv2.CAP_GSTREAMER, "V4L2 QTI MIDI"),
                (f"libcamerasrc camera-name=/dev/video{camera_index} ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1", cv2.CAP_GSTREAMER, "LibCamera")
            ])
            
        # Standard Fallbacks
        if self.os_type == "Windows":
            pipelines_to_try.append((camera_index, cv2.CAP_MSMF, "MSMF"))
            pipelines_to_try.append((camera_index, cv2.CAP_DSHOW, "DirectShow"))
        elif self.os_type == "Darwin":
            pipelines_to_try.append((camera_index, cv2.CAP_AVFOUNDATION, "AVFoundation"))
        else:
            pipelines_to_try.append((camera_index, cv2.CAP_V4L2, "V4L2"))
        return pipelines_to_try

    def _trusted_pipeline(self, camera_index, cached):
        """The candidate matching a cached entry, built from our own values, or None."""
        # settings.json is user-writable: never hand its pipeline strings or URLs to VideoCapture.
        for init_arg, backend, desc in self._candidate_pipelines(camera_index):
            if cached.get("init_arg") == init_arg and cached.get("backend") == backend:
                return {"init_arg": init_arg, "backend": backend, "desc": desc}
        return None

    def _find_best_pipeline(self, camera_index, stop_event=None):
        best_fps = -1
        best_pipeline = None
        
        for init_arg, backend, desc in self._candidate_pipelines(camera_index):
            if stop_event and stop_event.is_set():
                break
            fps, pipeline = self._benchmark_pipeline(init_arg, backend, desc, stop_event)
            if fps > best_fps:
                best_fps = fps
                best_pipeline = pipeline
                
        return best_pipeline

    def _apply_camera_props(self, cap, target_cam):
        cam_config = settings.get("camera", {})
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(cam_config.get("buffer_size", 1)))
        res = cam_config.get("resolution", [1280, 720])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, int(cam_config.get("fps", 30)))
        
        if target_cam.get("type") != "CSI" and self.os_type != "Darwin":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.metrics["resolution"] = f"{w}x{h}"

    def _select_camera(self, cameras):
        preferred = settings.get("camera", {}).get("preferred_camera", "")
        if str(preferred) != "":
            match = next((c for c in cameras if str(c["index"]) == str(preferred)), None)
            if match:
                return match
        return cameras[0] if cameras else None

    def _open_pipeline(self, pipeline):
        try:
            cap = cv2.VideoCapture(pipeline["init_arg"], pipeline["backend"])
        except (cv2.error, KeyError, TypeError) as e:
            logger.warning(f"Ignoring unusable capture pipeline {pipeline!r}: {e}")
            return None
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _owns(self, stop_event):
        """False for a capture thread that a restart has already replaced or stopped."""
        return self._stop_event is stop_event and not stop_event.is_set()

    def _open_camera(self, rediscover, stop_event):
        """Opens the preferred camera (cached pipeline, then benchmark, then default backend)."""
        target = self._select_camera(self.discover_cameras(force=rediscover))
        if not target:
            logger.error("No cameras detected on system!")
            return None

        idx = target["index"]
        logger.info(f"Selected Camera: {target['name']} (Index: {idx})")
        cache_key = f"cached_pipeline_{idx}"
        cached = settings.get("camera", {}).get(cache_key)
        pipeline = self._trusted_pipeline(idx, cached) if isinstance(cached, dict) else None
        if cached is not None and pipeline is None:
            logger.warning(f"Ignoring cached capture pipeline for camera {idx}: not one DeskBot would choose.")
        cap = self._open_pipeline(pipeline) if pipeline else None
        if cap is None:
            pipeline = self._find_best_pipeline(idx, stop_event)
            if pipeline and self._owns(stop_event):
                settings.update({"camera": {cache_key: pipeline}})
                cap = self._open_pipeline(pipeline)
        if cap is None:
            logger.warning("No benchmarked pipeline opened. Forcing default backend.")
            pipeline = {"desc": "Default"}
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                logger.error("Failed to open camera even after profiling.")
                return None

        if not self._owns(stop_event):
            cap.release()
            return None
        self.metrics["active_backend"] = pipeline.get("desc", "Unknown")
        self.active_camera_info = target
        self._streaming_index = idx
        try:
            self._apply_camera_props(cap, target)
        except Exception:
            cap.release()
            raise
        return cap

    def start(self):
        with self._lifecycle_lock:
            self._start_locked()

    def stop(self, timeout=5.0):
        with self._lifecycle_lock:
            self._stop_locked(timeout)

    def restart(self):
        with self._lifecycle_lock:
            self._stop_locked(5.0)
            self._start_locked()

    def restart_async(self):
        """Camera open and benchmarking take seconds, so API callers must not wait for them."""
        threading.Thread(target=self.restart, daemon=True, name="camera-restart").start()

    def _start_locked(self):
        if self._thread and self._thread.is_alive():
            return
        # Each thread gets its own stop event, so a slow old thread can never be revived by a restart.
        self._stop_event = threading.Event()
        self.metrics["state"] = "starting"
        self._thread = threading.Thread(
            target=self._capture_loop, args=(self._stop_event,), daemon=True, name="camera-capture"
        )
        self._thread.start()

    def _stop_locked(self, timeout):
        if self._stop_event:
            self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("Camera capture thread did not exit in time; abandoning it.")
        self._thread = None
        self.metrics["state"] = "stopped"
        with self._frame_lock:
            self._latest_frame = None

    def _capture_loop(self, stop_event):
        cap = None
        delay = 1.0
        retrying = False
        black_since = None
        black_frames = 0
        try:
            while not stop_event.is_set():
                if cap is None:
                    try:
                        cap = self._open_camera(rediscover=retrying, stop_event=stop_event)
                    except Exception:
                        logger.exception("Opening the camera failed; will retry.")
                        cap = None
                    if cap is None:
                        retrying = True
                        if self._owns(stop_event):
                            self.metrics["state"] = "no_camera"
                        stop_event.wait(delay)
                        delay = min(_MAX_RECONNECT_DELAY, delay * 2)
                        continue
                    retrying = False
                    black_since = None
                    if self._owns(stop_event):
                        self.metrics["state"] = "streaming"

                ret, frame = cap.read()
                if not self._owns(stop_event):
                    break
                reopen = not ret
                if not ret:
                    logger.warning("Camera read failed, attempting reconnect...")
                    retrying = True
                elif not is_black(frame):
                    black_since = None
                    delay = 1.0
                elif black_since is None:
                    black_since, black_frames = time.monotonic(), 1
                else:
                    black_frames += 1
                    black_for = time.monotonic() - black_since
                    if black_for >= _BLACK_REOPEN_SECONDS:
                        if black_frames <= _BUSY_MAX_FPS * black_for:
                            # The device is still there, so reopen the cached pipeline without probing every camera.
                            log = logger.warning if delay == 1.0 else logger.debug
                            log("Camera has sent only black frames for %.0fs; reopening it.", black_for)
                            reopen = True
                        else:
                            # Black at the normal rate (a dark room): judge the next window on its own.
                            black_since, black_frames = time.monotonic(), 1
                if reopen:
                    self.metrics["reconnects"] += 1
                    self.metrics["state"] = "reconnecting"
                    cap.release()
                    cap = None
                    self._streaming_index = None
                    stop_event.wait(delay)
                    delay = min(_MAX_RECONNECT_DELAY, delay * 2)
                    continue
                self._publish(frame)
        finally:
            # The capture thread owns its handle: releasing it from another thread mid-read can crash backends.
            if cap is not None:
                cap.release()
            if self._stop_event is stop_event:
                self._streaming_index = None

    def _publish(self, frame):
        with self._frame_lock:
            if self._latest_frame is not None:
                self.metrics["drops"] += 1
            self._latest_frame = frame

        self._frames_captured += 1
        now = time.monotonic()
        if now - self._last_fps_time >= 1.0:
            self.metrics["fps_capture"] = self._frames_captured / (now - self._last_fps_time)
            self._frames_captured = 0
            self._last_fps_time = now

    def get_frame(self):
        """Hands the newest frame to the caller (at most once); None if nothing new arrived."""
        with self._frame_lock:
            frame, self._latest_frame = self._latest_frame, None
        return frame

