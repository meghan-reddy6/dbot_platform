import cv2
import threading
import time
import platform
import glob
import os
import logging
from config.settings_manager import settings

logger = logging.getLogger("CameraManager")

class CameraManager:
    def __init__(self, camera_index=None):
        self.os_type = platform.system()
        self.arch_type = platform.machine().lower()
        
        self.running = False
        self.cap = None
        self.active_camera_info = None
        
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        
        self.metrics = {
            "fps_capture": 0.0,
            "drops": 0,
            "latency": 0.0,
            "active_backend": "None",
            "resolution": "Unknown",
            "reconnects": 0
        }
        
        self.available_cameras = []
        self._frames_captured = 0
        self._last_fps_time = time.time()
        
    def discover_cameras(self, force=False):
        if self.available_cameras and not force:
            return self.available_cameras
            
        logger.info("Starting camera discovery...")
        self.available_cameras = []
        
        if self.os_type == "Linux":
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
                    
                    self.available_cameras.append({
                        "path": dev,
                        "index": idx,
                        "name": name,
                        "type": "CSI" if "qcom" in name.lower() or "isp" in name.lower() else "USB/Virtual",
                        "status": "Available"
                    })
                except Exception as e:
                    logger.debug(f"Skipped {dev}: {e}")
                    
        else:
            # Windows/Mac basic probing
            for idx in range(5):
                # Critical Fix: macOS AVFoundation will SegFault if we cv2.VideoCapture() a camera that is currently streaming on another thread.
                if self.active_camera_info and self.active_camera_info["index"] == idx and self.cap and self.cap.isOpened():
                    self.available_cameras.append(self.active_camera_info)
                    continue
                    
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    backend = cap.getBackendName()
                    self.available_cameras.append({
                        "path": str(idx),
                        "index": idx,
                        "name": f"Camera {idx}",
                        "type": "Integrated/USB",
                        "status": "Available"
                    })
                    cap.release()
                    
        logger.info(f"Discovery complete. Found {len(self.available_cameras)} cameras.")
        return self.available_cameras

    def _benchmark_pipeline(self, init_arg, backend, desc):
        logger.info(f"Benchmarking {desc}...")
        cap = cv2.VideoCapture(init_arg, backend)
        
        if not cap.isOpened():
            logger.info(f"{desc} failed to open.")
            return -1, None
            
        # Optimize OpenCV buffers
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frames = 0
        bench_start = time.time()
        while time.time() - bench_start < 2.0:
            ret, _ = cap.read()
            if ret:
                frames += 1
            else:
                break
                
        fps = frames / (time.time() - bench_start) if frames > 0 else 0
        logger.info(f"{desc} achieved {fps:.1f} FPS.")
        
        cap.release()
        return fps, {"init_arg": init_arg, "backend": backend, "desc": desc}

    def _find_best_pipeline(self, camera_index):
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
            
        best_fps = -1
        best_pipeline = None
        
        for init_arg, backend, desc in pipelines_to_try:
            fps, pipeline = self._benchmark_pipeline(init_arg, backend, desc)
            if fps > best_fps:
                best_fps = fps
                best_pipeline = pipeline
                
        return best_pipeline

    def _apply_camera_props(self, target_cam):
        if not self.cap or not self.cap.isOpened():
            return
        cam_config = settings.get("camera", {})
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        res = cam_config.get("resolution", [1280, 720])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        
        if target_cam.get("type") != "CSI" and self.os_type != "Darwin":
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.metrics["resolution"] = f"{w}x{h}"

    def start(self):
        self.running = True
        
        if not self.available_cameras:
            self.discover_cameras()
            
        cam_config = settings.get("camera", {})
        preferred = cam_config.get("preferred_camera", "")
        
        target_cam = None
        if str(preferred) != "":
            target_cam = next((c for c in self.available_cameras if str(c["index"]) == str(preferred)), None)
            
        if not target_cam and self.available_cameras:
            target_cam = self.available_cameras[0]
            
        if not target_cam:
            logger.error("No cameras detected on system!")
            return
            
        idx = target_cam["index"]
        logger.info(f"Selected Camera: {target_cam['name']} (Index: {idx})")
        self.active_camera_info = target_cam
        
        # Check cache or benchmark
        cached_pipe = cam_config.get(f"cached_pipeline_{idx}")
        if not cached_pipe:
            best = self._find_best_pipeline(idx)
            if best:
                cam_config[f"cached_pipeline_{idx}"] = best
                settings.update({"camera": cam_config})
                cached_pipe = best
                
        if cached_pipe:
            logger.info(f"Opening pipeline: {cached_pipe['desc']}")
            self.cap = cv2.VideoCapture(cached_pipe["init_arg"], cached_pipe["backend"])
            self.metrics["active_backend"] = cached_pipe["desc"]
        else:
            logger.warning("No valid pipelines found during benchmark. Forcing default.")
            self.cap = cv2.VideoCapture(idx)
            self.metrics["active_backend"] = "Default"
            
        if self.cap and self.cap.isOpened():
            self._apply_camera_props(target_cam)
            threading.Thread(target=self._capture_loop, daemon=True).start()
        else:
            logger.error("Failed to open camera even after profiling.")
            
    def _capture_loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self._handle_reconnect()
                time.sleep(1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Camera read failed, attempting reconnect...")
                self._handle_reconnect()
                time.sleep(1)
                continue
                
            # Atomic update
            with self._frame_lock:
                if self._latest_frame is not None:
                    self.metrics["drops"] += 1
                self._latest_frame = frame
                
            self._frames_captured += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self.metrics["fps_capture"] = self._frames_captured / (now - self._last_fps_time)
                self._frames_captured = 0
                self._last_fps_time = now

    def _handle_reconnect(self):
        self.metrics["reconnects"] += 1
        if self.cap:
            self.cap.release()
            
        # 1. Try reopening same camera with cached backend
        if self.active_camera_info:
            idx = self.active_camera_info["index"]
            cam_config = settings.get("camera", {})
            cached_pipe = cam_config.get(f"cached_pipeline_{idx}")
            if cached_pipe:
                self.cap = cv2.VideoCapture(cached_pipe["init_arg"], cached_pipe["backend"])
                if self.cap.isOpened(): 
                    self._apply_camera_props(self.active_camera_info)
                    return
        
        # 2. Try alternate backend (trigger benchmark)
        if self.active_camera_info:
            idx = self.active_camera_info["index"]
            best = self._find_best_pipeline(idx)
            if best:
                self.cap = cv2.VideoCapture(best["init_arg"], best["backend"])
                if self.cap.isOpened(): 
                    self._apply_camera_props(self.active_camera_info)
                    return
                
        # 3. Try alternate camera
        self.discover_cameras()
        if self.available_cameras:
            fallback = self.available_cameras[0]
            best = self._find_best_pipeline(fallback["index"])
            if best:
                self.cap = cv2.VideoCapture(best["init_arg"], best["backend"])
                if self.cap.isOpened(): 
                    self.active_camera_info = fallback
                    self._apply_camera_props(fallback)
                    return

    def get_frame(self):
        with self._frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame.copy()
                self._latest_frame = None 
                return frame
        return None

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

