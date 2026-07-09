import cv2
import queue
import threading
import time
import platform


class CameraManager:
    def __init__(self, camera_index=0):
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.cap = None
        self.camera_index = camera_index
        self.os_type = platform.system()
        self.arch_type = platform.machine().lower()

    def start(self):
        # 1. Attempt Qualcomm Hardware-Accelerated GStreamer Pipeline (Linux aarch64)
        if self.os_type == "Linux" and "aarch64" in self.arch_type:
            print(
                "[*] Target: Qualcomm Linux ARM64. Initializing Hardware IM SDK Pipeline..."
            )
            gst_pipeline = (
                f"v4l2src device=/dev/video{self.camera_index} ! "
                "image/jpeg,width=1920,height=1080,framerate=30/1 ! "
                "qtimididec ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink drop=true max-buffers=1"
            )
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            if self.cap is not None and self.cap.isOpened():
                print("[*] Qualcomm GStreamer Pipeline Active.")
                self._start_worker()
                return
            else:
                print(
                    "[!] GStreamer initialization failed. Falling back to native V4L2..."
                )

        # 2. Native Desktop Fallbacks
        print("[*] Initializing Native OS VideoCapture Backend...")
        if self.os_type == "Windows":
            backend = cv2.CAP_DSHOW
        elif self.os_type == "Darwin":
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = cv2.CAP_V4L2

        self.cap = cv2.VideoCapture(self.camera_index, backend)

        # Safe resolution fallback
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self._start_worker()
        else:
            print("[!] FATAL: Could not open any camera backend.")

    def _start_worker(self):
        def worker():
            while self.running:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    continue
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass

        threading.Thread(target=worker, daemon=True).start()

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
