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
- **Language:** Python 3.12+
- **Machine Learning / AI:** Ultralytics (YOLOv8-Pose), ONNX Runtime, PyTorch (DirectML)
- **Computer Vision:** OpenCV (`opencv-python`)
- **Web Framework:** Flask (Backend API & Frontend serving)
- **Database:** SQLite3 (Local telemetry logging)
- **Package Management:** `uv`
- **Math/Geometry:** NumPy, SciPy

## Project Structure
```text
dbot/
├── deskbot_v3.py           # Application Entry Point (Orchestrator & Flask App)
├── core/
│   ├── tracking/           # MOT, Session State (UserSession, Person), and Geometry
│   └── model_manager.py    # Downloads and validates ONNX/YOLO models
├── alerts/                 # Thread-safe TTS Audio Alert Manager
├── analytics/              # Dashboard Session Exporter & Metrics
├── api/                    # Flask API Endpoints
├── camera/                 # VideoCapture and Frame Processing
├── config/                 # Settings Manager and Hardcoded Defaults
├── data/                   # Persistent storage for user settings and states
├── database/               # SQLite Database schema and wrappers
├── detection/              # YOLO Person Detector and Face Detectors
├── models/                 # AI Models (populated manually)
├── posture/                # Posture Evaluation algorithms & Correction Engine
├── recognition/            # ArcFace Biometric Embeddings 
├── templates/              # HTML Web Dashboard and Configuration UI
├── tests/                  # Unit tests and script sandboxes
├── utils/                  # Universal Hardware Detector & Helpers
├── profiles_cache.json     # Saved biometric embeddings
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
   The `TrackerEngine` runs the heavy ML inference loop on a dedicated daemon thread. The Flask server runs on the main thread to serve the dashboard. The `AlertManager` runs on a third isolated thread using a FIFO `Queue` to prevent blocking the ML loop during audio playback.

## Prerequisites
- Python 3.10 or higher.
- A connected USB Webcam or integrated laptop camera.
- (Optional) `uv` for lightning-fast package installation.
- (Optional) A dedicated GPU (NVIDIA, AMD) or NPU (like the Rubik Pi's Qualcomm chip) for hardware acceleration. The system gracefully falls back to the CPU if no accelerator is found.

### Linux / Ubuntu / Rubik Pi Specific Dependencies
If you are on Linux, you must install the following system-level libraries for OpenCV (camera) and PyTTSx3 (text-to-speech) to function:
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx espeak ffmpeg
```

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
- **NVIDIA GPU (Windows/Ubuntu):** Run `pip install onnxruntime-gpu`
- **Rubik Pi (Qualcomm NPU):** Run `pip install onnxruntime-qnn`
- **AMD/Intel GPU (Windows):** *No action required! `onnxruntime-directml` installs automatically on Windows.*

### Step 3: Required AI Models
Because AI models are large, they are not tracked in Git. You **must** populate the `models/` directory before starting the application, otherwise it will exit with an error.

1. **YOLOv8-Pose (ONNX):** You can auto-download and generate this model by running the following Python one-liner in your environment:
   ```bash
   python -c "from ultralytics import YOLO; YOLO('models/yolov8n-pose.pt').export(format='onnx')"
   ```
2. **Proprietary Facial Models:** You must manually place the following internal models into the `models/` directory:
   - `mobilefacenet.onnx`
   - `face_detector.onnx`
   - `face_landmark_detector.onnx`

## Environment Configuration
The application separates hardcoded defaults from user preferences.
- **Settings:** Defaults are stored in `config/defaults.py`. User overrides are managed by `SettingsManager` and saved persistently to `data/settings.json`. You can modify settings live via the web Configuration Page without restarting the server.
- **Models:** Models are automatically downloaded by `core/model_manager.py` to the `models/` directory on first boot based on `models_manifest.json`.
- **Database:** Creates `analytics/telemetry.db` automatically in the local path.
- **Hardware:** Features a Universal Fallback Pipeline. It automatically attempts to hook `QNNExecutionProvider` (NPU), `CUDAExecutionProvider` (NVIDIA), or `DmlExecutionProvider` (DirectML/AMD), gracefully cascading down to `CPUExecutionProvider` across all operating systems.

## Running the Project
To start the full pipeline (Inference Engine + Flask Server):
```bash
uv run deskbot_v3.py
# OR if using standard pip:
# python deskbot_v3.py
```
Once initialized, the terminal will display: `Server pipeline active at: http://localhost:5000`. Navigate to this URL in your browser to view the live dashboard and register your biometric profile.

## Testing
Run unit tests for the posture analysis and correction engines using `unittest`:
```bash
python -m unittest discover tests
```

## API Documentation
The local Flask server exposes several endpoints for dashboard interaction:
- `GET /` - Renders the main dashboard.
- `GET /config` - Renders the Configuration settings page.
- `GET /api/metrics_slice` - Returns live telemetry JSON for all active tracking sessions.
- `GET /api/settings` - Returns the current resolved configuration payload.
- `POST /api/settings` - Accepts partial JSON config, saves to disk, and hot-reloads the backend.
- `POST /api/profile/register` - Registers the currently tracked user to a named biometric profile.
- `POST /api/profile/recalibrate` - Drops the current posture baseline and forces the active identity back into the Calibration state.
- `POST /api/profile/delete` - Deletes a user's biometric profile and clears their telemetry history.

## Database
- **Technology:** SQLite3 (`sqlite3` native python module)
- **Schema:** Contains `session_logs` and `posture_telemetry`.
- **Setup:** Auto-migrates and instantiates upon first boot via `analytics/db.py`.

## Troubleshooting
- **No Targets in Frame:** Ensure your webcam is well-lit and not covered. Ensure you have registered your face in the dashboard.
- **Alerts overlapping or not playing:** DeskBot uses PowerShell `System.Speech.Synthesis` under the hood. If alerts fail to play, ensure your Windows sound settings are active and the script isn't heavily CPU bottlenecked.
- **Tracking swaps rapidly:** Ensure there are no background reflections or portraits behind you causing YOLO ghosting.

## Contributing
When contributing, ensure all heavy ML execution is restricted to the `TrackerEngine` thread. Do not introduce blocking calls to the main tracking loop.

## License
Proprietary / Thundersoft Internal.
