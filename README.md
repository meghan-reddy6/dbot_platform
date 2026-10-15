# DeskBot V3 - AI Posture & Ergonomic Health Monitor

DeskBot V3 is an advanced, hardware-accelerated ergonomic monitoring system. It leverages a dynamic multi-model computer vision pipeline to track human posture, eye gaze, and workspace session lengths in real time, automatically deploying local audio alerts for posture deficits.

## Overview
The primary goal of DeskBot V3 is to promote healthy ergonomic habits for desk workers without requiring wearable hardware. Using standard webcams and edge-AI accelerators, the system constructs a 3D skeletal map of the user, identifies them via facial biometrics, learns their natural sitting baseline, and issues audio corrections when it detects severe slouching, neck deviation, or excessive screen time.

## Features
- **Real-Time Skeletal Posture Analysis:** Tracks 17-point keypoints using YOLOv8 to calculate spine alignment, shoulder roll, and torso depth.
- **Biometric Identity Persistence:** Uses MobileFaceNet (ArcFace) embeddings to identify users and persist their session data even if they temporarily leave the frame.
- **Dynamic 3D Head Pose Estimation:** Calculates pitch, yaw, and roll using Perspective-n-Point (PnP) geometry for neck strain and gaze tracking.
- **Automatic Audio Alerts:** Queued, thread-safe text-to-speech (TTS) engine that issues corrective audio prompts (e.g., "Lift your head", "Avoid leaning forward").
- **Live Web Dashboard & Configuration:** Local Flask-based dashboard offering real-time telemetry, session metrics, a live-annotated camera feed, and a hot-reloading settings interface.
- **Persistent JSON Configuration:** Allows users to override system defaults via a clean UI, saving preferences locally to `data/settings.json` without requiring restarts.
- **Universal Hardware Acceleration:** Dynamically scales across Windows GPUs (DirectML), NVIDIA GPUs (CUDA), Edge NPUs (Qualcomm QNN on Rubik Pi), and Apple Silicon (Accelerate/CPU) using a universal ONNX fallback pipeline.

## Tech Stack
- **Language:** Python 3.11+
- **Machine Learning / AI:** ONNX Runtime (YOLOv8n-pose, MediaPipe face detector + landmarks, MobileFaceNet ArcFace)
- **Computer Vision:** OpenCV (`opencv-python`; `opencv-python-headless` on Linux)
- **Web Framework:** Flask, served by Waitress
- **Database:** SQLite3 (Local telemetry logging)
- **Package Management:** `uv`
- **Math/Geometry:** NumPy

## Project Structure
```text
dbot/
├── deskbot_v3.py           # Application Entry Point (wiring + Waitress server)
├── core/
│   ├── tracking/           # Tracker (orchestrator), identity matching, health monitor, face worker, states
│   ├── pipeline.py         # Camera -> tracker inference loop and its health status
│   └── model_manager.py    # Validates the ONNX models against their checksums
├── alerts/                 # Thread-safe TTS Audio Alert Manager
├── analytics/              # Dashboard Session Exporter & Metrics
├── api/                    # Flask API Endpoints and input validation
├── camera/                 # VideoCapture and Frame Processing
├── config/                 # Settings Manager, defaults and data paths
├── data/                   # Runtime data: settings.json and wellness_logs.db (not in git)
├── database/               # SQLite Database schema and wrappers
├── detection/              # YOLOv8-pose person detector (ONNX Runtime)
├── models/                 # AI Models (ONNX)
├── posture/                # Posture Evaluation algorithms & Correction Engine
├── recognition/            # ArcFace Biometric Embeddings 
├── static/vendor/          # Chart.js 4.4.1 (served locally, no CDN)
├── templates/              # HTML Web Dashboard and Configuration UI
├── tests/                  # pytest suite
├── utils/                  # Universal Hardware Detector & Helpers
├── requirements.txt        # PIP dependencies
└── pyproject.toml          # UV dependencies
```

## Architecture
DeskBot uses a strictly gated, two-stage cascading inference pipeline to drastically reduce unnecessary computational overhead.

1. **Stage 1: Spatial Tracking (Always On)**
   YOLOv8 extracts bounding boxes and keypoints frame-by-frame. It provides 100% of the data required for persistent tracking and posture evaluation.
2. **Stage 2: Biometric Vectorizer (Idle by Default)**
   The heavy MobileFaceNet identity model is strictly gated. It only executes when the tracker encounters a new, unregistered bounding box or during manual recalibration, locking the identity to a persistent `UserSession`.
3. **Threading Model:**
   The camera capture thread publishes only the newest frame. The inference loop (`core/pipeline.py`) runs pose detection and posture evaluation on its own thread, capped at the `detection_fps` setting, and survives per-frame errors (they are reported by `/api/system/health`). Face recognition runs in a separate, supervised worker process that is restarted with backoff if it dies. Waitress serves the dashboard, and the `AlertManager` speaks alerts from a FIFO `Queue` on its own thread.

## Prerequisites
- Python 3.11 or higher. On Windows on ARM, use the x64 Python: PyPI has no OpenCV wheel for ARM64 Windows (not tested).
- A connected USB Webcam or integrated laptop camera.
- (Optional) `uv` for lightning-fast package installation.
- (Optional) A dedicated GPU (NVIDIA, AMD) or NPU (like the Rubik Pi's Qualcomm chip) for hardware acceleration. The system gracefully falls back to the CPU if no accelerator is found.

### Linux / Ubuntu / Rubik Pi Specific Dependencies
Spoken alerts on Linux use `espeak-ng` (or `espeak`):
```bash
sudo apt-get update
sudo apt-get install espeak-ng
```
Without it, alerts are only logged.

On Qualcomm boards, use a USB (UVC) webcam. The MIPI camera is only reachable through GStreamer (`qtiqmmfsrc`), and the OpenCV wheels from PyPI are built without GStreamer.

## Installation

### Option 1: Using `uv` (Recommended)
1. Clone the repository and navigate to the root directory.
2. Install dependencies:
   ```bash
   uv sync
   ```
   *(Alternatively, just run `uv run deskbot_v3.py` to auto-bootstrap).*

### Option 2: Using standard `pip` (Mac, Ubuntu, Windows, Rubik Pi)
Our dependency files use intelligent environment markers, meaning you can run this exact command on any OS (Windows, Mac, Linux) and `pip` will automatically ignore incompatible packages:
1. Clone the repository and navigate to the root directory.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Ubuntu/Linux/Rubik Pi:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

### Optional Hardware Acceleration (NVIDIA / Edge NPUs)
Because DeskBot uses a **Universal Fallback Architecture**, you can manually install the specific ONNX Runtime library for your hardware to unlock extreme performance:
- **NVIDIA GPU (Windows/Ubuntu):** `onnxruntime-gpu` provides the same `onnxruntime` module, so uninstall the default one first (`pip uninstall onnxruntime onnxruntime-directml`), then run `pip install onnxruntime-gpu`.
- **Rubik Pi (Qualcomm NPU):** Run `pip install onnxruntime-qnn`. Version 2.x is a plugin that installs next to `onnxruntime`, so keep `onnxruntime` installed. DeskBot loads the plugin at startup and runs its models on the NPU; ONNX Runtime's CPU provider runs any operator the NPU does not support. The NPU only appears when the board's FastRPC driver provides `/dev/fastrpc-cdsp*`. Images with the older driver (`/dev/adsprpc-smd`, for example the Qualcomm RB5 Gen 2 image with kernel 5.15) do not show an NPU to the plugin, and DeskBot then runs on the CPU. `/api/system/health` shows `has_npu` and the providers in use. Set `DESKBOT_DISABLE_QNN=1` to skip the plugin, for example if it crashes or hangs on a board. With `uv`, install it with `uv pip install onnxruntime-qnn`; a plain `uv sync` removes packages that are not in the lockfile, so use `uv sync --inexact` afterwards.
- **AMD/Intel GPU (Windows):** *No action required! `onnxruntime-directml` installs automatically on Windows.*


## Environment Configuration
The application separates hardcoded defaults from user preferences.
- **Settings:** Defaults are stored in `config/defaults.py`. User overrides are managed by `SettingsManager` and saved atomically to `data/settings.json`. You can modify settings live via the web Configuration Page without restarting the server.
- **Data directory:** Settings and the database live in `data/` (created on first run, not tracked by git). Set `DESKBOT_DATA_DIR` to store them elsewhere.
- **Models:** The ONNX models ship in `models/`. At boot `core/model_manager.py` checks them against the SHA-256 values in `config/models_manifest.json` and refuses to start if a required model is missing or altered.
- **Database:** `data/wellness_logs.db` (SQLite, WAL mode), created automatically.
- **Hardware:** Features a Universal Fallback Pipeline. It automatically attempts to hook `QNNExecutionProvider` (NPU), `CUDAExecutionProvider` (NVIDIA), or `DmlExecutionProvider` (DirectML/AMD), gracefully cascading down to `CPUExecutionProvider` across all operating systems.

## Running the Project
To start the full pipeline (Inference Engine + Flask Server):
```bash
uv run deskbot_v3.py
# OR if using standard pip:
# python deskbot_v3.py
```
Once initialized, the terminal will display: `Server pipeline active at: http://localhost:5050`. Navigate to this URL in your browser to view the live dashboard and register your biometric profile. Set `DESKBOT_PORT` to use another port.

## Security and Privacy
- The server listens on `127.0.0.1` only and has no login. Do not expose it through port forwarding or a reverse proxy: it serves the live camera feed and biometric profiles.
- Requests with a `Host` other than `localhost`, `127.0.0.1` or `[::1]` are refused (DNS-rebinding protection).
- Every state-changing request must carry the header `X-DeskBot-Request: 1`; browsers cannot add it cross-origin, which blocks CSRF from other sites.
- Face embeddings, settings and telemetry stay in `data/` on this machine. Telemetry older than `analytics_retention_days` (default 90) is purged, and deleting a profile deletes its telemetry.
- Headless board (e.g. Rubik Pi): forward the port over SSH instead of exposing it, `ssh -L 5050:127.0.0.1:5050 <user>@<board>`, then open `http://localhost:5050` on your PC.

### Upgrading from builds before the data/ directory
- `data/settings.json` is no longer tracked by git. If `git pull` refuses because that file has local changes, copy it elsewhere, run `git checkout -- data/settings.json`, pull, then copy it back into `data/`.
- An existing `wellness_logs.db` in the repository root is moved into `data/` on first start, and telemetry older than 90 days is then purged. To keep more history, set `analytics_retention_days` in `data/settings.json` before starting.
- `profiles_cache.json` is imported into the database once and then removed; an unreadable copy is left in place with a warning.
- OpenCV: Windows and macOS now use `opencv-python` (the headless wheel has no Media Foundation camera backend on Windows); Linux uses `opencv-python-headless`. pip does not remove the old packages, and all of them share one `cv2` folder, so in an existing environment run `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless` and then `pip install -r requirements.txt` again.

## Testing
```bash
uv run --group dev pytest tests
# OR with pip: pip install pytest && python -m pytest tests
```
`tests/test_person_detector.py` also compares the ONNX Runtime decoder with Ultralytics when `ultralytics` is installed (`pip install ultralytics==8.4.82`); without it that test is skipped. `tests/test_face_golden.py` runs the face pipeline on a real photo with the session settings of the machine running the tests; CI runs it on x86 and on 64-bit ARM.

## API Documentation
The local Flask server exposes several endpoints for dashboard interaction (POST bodies are JSON unless noted):
- `GET /` - Renders the main dashboard.
- `GET /config` - Renders the Configuration settings page.
- `GET /video_feed` - MJPEG stream of the annotated camera feed.
- `GET /api/metrics_slice` - Returns live telemetry JSON for all active tracking sessions, including calibration state.
- `GET /api/system/health` - Hardware, camera, inference-loop and face-worker status; `status` is `degraded` with `reasons` when something is not working.
- `GET /api/profiles` - Lists registered profiles and their limits.
- `POST /api/profile/create` - `{"name"}`: registers the currently tracked unknown person under that name (409 if it exists).
- `POST /api/profile/update` - `{"name", "slouch_sensitivity", "session_limit", "stand_requirement", "screen_gaze_limit", "ocular_break_duration"}`.
- `POST /api/profile/delete` - `{"name"}`: deletes the profile and its telemetry history.
- `POST /api/profile/recalibrate` - Drops the current posture baseline and forces the active identity back into the Calibration state.
- `POST /api/profile/restart_session` - Resets the active user's session timers.
- `GET /api/history/<name>`, `GET /api/export/json/<name>`, `GET /api/export/csv/<name>` - Stored telemetry for a profile.
- `GET /api/cameras`, `POST /api/cameras/select` - `{"index"}`: lists cameras / switches camera in the background (202).
- `GET /api/settings` - Returns the current resolved configuration payload.
- `POST /api/settings` - Accepts partial JSON config; unknown keys are ignored and invalid values are rejected with 400.
- `POST /api/settings/reset` - `{"keys": [...]}` resets those settings to defaults.
- `GET /api/settings/export`, `POST /api/settings/import` (multipart `file`, max 1 MB).

## Database
- **Technology:** SQLite3 (`sqlite3` native python module), WAL mode; telemetry is written by a background writer thread.
- **Schema:** `users` (profiles and face embeddings), `metrics` (session events) and `user_analytics_logs` (periodic posture telemetry).
- **Setup:** Created and migrated on first boot by `database/crud.py`.

## Troubleshooting
- **No Targets in Frame:** Ensure your webcam is well-lit and not covered. Ensure you have registered your face in the dashboard.
- **"camera image has been black" in health:** another app (Teams, the Camera app) is using the webcam, or the lens is covered or in the dark. Tracking pauses while the image is black, so a video call does not end your posture session or count as sitting time; it resumes when the picture returns. While another app holds the camera, DirectShow sends black frames at about one per second and keeps doing so after that app lets go, so after 10 s of that DeskBot reopens the camera, retrying with a growing delay (up to 30 s), and recovers on its own once the camera is free.
- **Alerts overlapping or not playing:** DeskBot uses PowerShell `System.Speech.Synthesis` under the hood. If alerts fail to play, ensure your Windows sound settings are active and the script isn't heavily CPU bottlenecked.
- **Tracking swaps rapidly:** Ensure there are no background reflections or portraits behind you causing YOLO ghosting.

## Contributing
When contributing, ensure all heavy ML execution is restricted to the `TrackerEngine` thread. Do not introduce blocking calls to the main tracking loop.

## License
Proprietary / Sapphire Stream Technology Private Limited Internal.

Third-party components keep their own licenses: `models/yolov8n-pose.onnx` declares AGPL-3.0 in its embedded metadata (Ultralytics), and `static/vendor/chart.umd.js` is Chart.js (MIT).

