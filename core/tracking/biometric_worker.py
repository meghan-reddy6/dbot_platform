"""Face-recognition worker process and its supervision."""
import logging
import multiprocessing
import queue as python_queue
import time

logger = logging.getLogger(__name__)

_CHECK_INTERVAL = 1.0
_MAX_BACKOFF = 60.0
_HEALTHY_SECONDS = 60.0
_STALL_SECONDS = 30.0
_REBUILD_AFTER_ERRORS = 5
# Spawn everywhere: forking after ONNX Runtime and OpenCV have started threads can hang the child on Linux.
_mp = multiprocessing.get_context("spawn")


def bg_biometric_process_worker(job_queue, result_queue):
    """Face-recognition worker process: receives head crops, returns embeddings."""
    logging.basicConfig(level=logging.INFO)
    from recognition.face_recognizer import FaceRecognizer

    face_recognizer = FaceRecognizer()
    while True:
        try:
            job = job_queue.get()
            if job is None:
                continue
            embedding = face_recognizer.extract_embedding(job["crop"])
            if embedding is not None:
                result_queue.put_nowait({"track_id": job["track_id"], "embedding": embedding})
            if face_recognizer.consecutive_errors >= _REBUILD_AFTER_ERRORS:
                # Repeated runtime errors (e.g. an NPU subsystem restart) need fresh ORT sessions.
                logger.error("[BIOMETRIC WORKER] Rebuilding face recognizer after repeated errors.")
                face_recognizer = FaceRecognizer()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"[BIOMETRIC WORKER ERROR] {e}")


class BiometricWorker:
    """Runs face recognition in a child process and restarts it when it dies or stops taking jobs."""

    def __init__(self):
        self.restarts = 0
        self._failures = 0
        self._restart_at = 0.0
        self._next_check = 0.0
        self._shutting_down = False
        self.start()

    def start(self):
        self.job_queue = _mp.Queue(maxsize=1)
        self.result_queue = _mp.Queue()
        self.process = _mp.Process(
            target=bg_biometric_process_worker,
            args=(self.job_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()
        self._started_at = time.monotonic()
        self._job_waiting_since = None

    def _abandon_queues(self):
        for q in (self.job_queue, self.result_queue):
            cancel = getattr(q, "cancel_join_thread", None)
            if cancel:
                # With the reader gone, a full pipe would otherwise block interpreter exit.
                cancel()

    def supervise(self, now) -> bool:
        """Restarts a dead worker with backoff; True when it did, because in-flight jobs are then lost."""
        if self._shutting_down or now < self._next_check:
            return False
        self._next_check = now + _CHECK_INTERVAL
        if self.process.is_alive():
            self._check_progress(now)
            return False
        if now < self._restart_at:
            return False
        self.restarts += 1
        self._failures += 1
        self._restart_at = now + min(_MAX_BACKOFF, 2.0 ** self._failures)
        logger.error(f"[BIOMETRIC WORKER] Process is not running; restart #{self.restarts}.")
        self._abandon_queues()
        self.start()
        return True

    def _check_progress(self, now):
        if now - self._started_at > _HEALTHY_SECONDS:
            self._failures = 0
        # A job left in the queue means the worker stopped taking work (hung inference, wedged NPU).
        if self.job_queue.empty():
            self._job_waiting_since = None
        elif self._job_waiting_since is None:
            self._job_waiting_since = now
        elif now - self._job_waiting_since > _STALL_SECONDS:
            logger.error("[BIOMETRIC WORKER] No job taken for 30 s; terminating the worker for a restart.")
            self._job_waiting_since = None
            self.process.terminate()

    def is_idle(self) -> bool:
        return self.job_queue.empty()

    def submit(self, track_id, crop) -> bool:
        try:
            self.job_queue.put_nowait({"track_id": track_id, "crop": crop})
            return True
        except python_queue.Full:
            return False

    def results(self):
        """Yields the results the worker has produced so far."""
        while not self.result_queue.empty():
            try:
                yield self.result_queue.get_nowait()
            except python_queue.Empty:
                return

    def status(self) -> dict:
        return {"alive": bool(self.process.is_alive()), "restarts": self.restarts}

    def shutdown(self):
        self._shutting_down = True
        self._abandon_queues()
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)
