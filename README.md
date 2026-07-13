# DeskBot V3 - AI Posture & Ergonomic Health Monitor

DeskBot V3 is an advanced, hardware-accelerated ergonomic monitoring system. It leverages a dynamic multi-model computer vision pipeline to track human posture, eye gaze, and workspace session lengths in real time, automatically deploying local audio alerts for posture deficits.

## Overview
The primary goal of DeskBot V3 is to promote healthy ergonomic habits for desk workers without requiring wearable hardware. Using standard webcams and edge-AI accelerators, the system constructs a 3D skeletal map of the user, identifies them via facial biometrics, learns their natural sitting baseline, and issues audio corrections when it detects severe slouching, neck deviation, or excessive screen time.

## Features
- **Real-Time Skeletal Posture Analysis:** Tracks 17-point keypoints using YOLOv8 to calculate spine alignment, shoulder roll, and torso depth.
- **Biometric Identity Persistence:** Uses MobileFaceNet (ArcFace) embeddings to identify users and persist their session data even if they temporarily leave the frame.
- **Dynamic 3D Head Pose Estimation:** Calculates pitch, yaw, and roll using Perspective-n-Point (PnP) geometry for neck strain and gaze tracking.
- **Automatic Audio Alerts:** Queued, thread-safe text-to-speech (TTS) engine that issues corrective audio prompts (e.g., "Lift your head", "Avoid leaning forward").
- **Live Web Dashboard:** Local Flask-based dashboard offering real-time telemetry, session metrics, and a live-annotated camera feed.
- **Hardware Acceleration:** Native Windows DirectML and ONNX Runtime support for highly efficient local Edge AI inference.

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
├── analytics/              # SQLite Database and Dashboard Session Exporter
├── config/                 # Pydantic Settings and Model Manifests
├── detection/              # YOLO Person Detector and Face Detectors
├── posture/                # Posture Evaluation algorithms & Correction Engine
├── recognition/            # ArcFace Biometric Embeddings 
├── templates/              # HTML Web Dashboard
├── tests/                  # Unit tests and script sandboxes
├── profiles_cache.json     # Saved biometric embeddings
└── pyproject.toml          # Python dependencies
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
- Python 3.12 or higher.
- `uv` (Fast Python package installer).
- A connected USB Webcam or integrated laptop camera.
- (Optional) Windows machine with DirectX 12 compatible GPU for DirectML acceleration.

## Installation
1. Clone the repository and navigate to the root directory.
2. Install dependencies using `uv`:
   ```bash
   uv venv
   uv pip install -e .
   ```
   *(Alternatively, run `uv run deskbot_v3.py` to auto-bootstrap).*

## Environment Configuration
The application relies heavily on defaults specified in `config/settings.py`. There are no hard `.env` requirements, but the following configurations are dynamically evaluated:
- **Models:** Models are automatically downloaded by `core/model_manager.py` to the `models/` directory on first boot based on `models_manifest.json`.
- **Database:** Creates `analytics/telemetry.db` automatically in the local path.
- **Hardware:** Automatically attempts to hook `DmlExecutionProvider` (DirectML) for ONNX and PyTorch if available, gracefully falling back to CPU.

## Running the Project
To start the full pipeline (Inference Engine + Flask Server):
```bash
uv run deskbot_v3.py
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
- `GET /api/metrics_slice` - Returns live telemetry JSON for all active tracking sessions.
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
