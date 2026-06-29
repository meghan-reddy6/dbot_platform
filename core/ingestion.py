import cv2
import queue
import subprocess
import threading
import os
import time

class DynamicCameraIngestion:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.cap = None
        self.gst_process = None
        self.pipe_path = '/home/ubuntu/deskbot/camera_pipe'

    def start(self):
        if os.path.exists('/usr/lib/libQnnHtp.so'):
            print("[*] Qualcomm Hardware Node Detected. Spawning isolated GStreamer pipeline...")
            self._start_gstreamer()
        else:
            print("[*] Native OS Developer Fork Detected. Starting cv2.VideoCapture(0)...")
            self._start_native()

    def _start_gstreamer(self):
        os.makedirs(os.path.dirname(self.pipe_path), exist_ok=True)
        if os.path.exists(self.pipe_path):
            try:
                os.remove(self.pipe_path)
            except Exception:
                pass
        os.mkfifo(self.pipe_path)
            
        cmd = [
            "gst-launch-1.0", 
            "v4l2src", "!",
            "video/x-raw,format=NV12,width=640,height=480,framerate=30/1", "!",
            "jpegenc", "!",
            "filesink", f"location={self.pipe_path}"
        ]
        self.gst_process = subprocess.Popen(cmd)
        
        self.cap = cv2.VideoCapture(self.pipe_path, cv2.CAP_GSTREAMER)
        self._start_worker()

    def _start_native(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._start_worker()
        
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
            self.cap.release()
        if self.gst_process:
            self.gst_process.terminate()
            self.gst_process.wait()
        if os.path.exists(self.pipe_path):
            try:
                os.remove(self.pipe_path)
            except Exception:
                pass
