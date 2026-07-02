# DeskBot V3 - Smart Ergonomic Health Pipeline

DeskBot V3 is an advanced, hardware-accelerated ergonomic monitoring system. It leverages a dynamic **Cross-Platform Multi-Model Cascading Pipeline** to track human posture, eye gaze, and workspace sessions in real time, automatically deploying local alerts for posture deficits.

## Multi-Model Pipeline Data Flow

The system employs a strictly gated, two-stage cascading inference pipeline to drastically reduce unnecessary computational overhead and protect system responsiveness.

### Stage 1: Human Detection & Spatial Tracking (Always On)
- **Model:** YOLOv8 Nano Pose (INT8)
- **Frequency:** Frame-by-frame
- **Function:** Extracts bounding boxes, 17-point skeletal keypoints, and spatial scale. Calculates real-time 3D head pose (pitch, yaw, roll) using Perspective-n-Point (PnP) geometry and extracts pupil vectors for gaze analysis. This stage provides 100% of the data required for persistent tracking (MOT) and health evaluations without invoking heavy ID models.

### Stage 2: Biometric Embeddings Vectorizer (Gated / Idle)
- **Model:** Simulated 128-dimensional Facial Vectorizer (FP16/INT8)
- **Frequency:** Strictly Gated (Idle by default)
- **Function:** Generates a unique facial signature for the user. It is **strictly forbidden** from running frame-by-frame. It is only permitted to execute when the system detects an unassigned anchor (`self.primary_user_track_id is None`) or when the user intentionally triggers a manual recalibration request.

---

## Cross-Platform Execution Matrix

The `CrossPlatformInferenceManager` dynamically interrogates the host operating system upon instantiation and maps model execution to the optimal hardware accelerator using native SDK fallbacks. If a local proprietary backend (e.g., QNN or CoreML) is missing, it gracefully intercepts the `ImportError` and falls back to a mocked state or CPU execution.

| Host Operating System | Architecture | Hardware Accelerator | Native Inference SDK | Provider / Target |
| :--- | :--- | :--- | :--- | :--- |
| **Linux (Qualcomm Rubik Pi)** | `aarch64` | Hexagon NPU | Qualcomm AI Engine Direct | `QNNExecutionProvider` (FP16 Burst) |
| **Microsoft Windows** | `amd64` / `x86_64` | NVIDIA / AMD GPU | ONNX Runtime | `CUDAExecutionProvider` or `DmlExecutionProvider` |
| **Apple macOS** | `arm64` | Apple Neural Engine (ANE) | CoreML / MPS | Native PyTorch MPS Hook |

---

## Operational State Rules

The tracker's state machine governs all posture and biometric logic to ensure a completely seamless and fail-safe user experience:

1. **Single-Target Biometric Anchor Lock:**
   Once a user satisfies the 15-frame biometric consensus threshold (cosine similarity > cutoff limit), they are "locked" as the primary anchor (`self.primary_user_track_id`).
   
2. **Absolute Bystander Blindness:**
   All secondary skeletal detections in the frame are categorized as `Secondary Bystander` and are entirely ignored. Their posture is never evaluated, and the heavy Biometric ID model will never scan them as long as the primary anchor is locked.

3. **Manual Recalibration Override:**
   A manual recalibration event temporarily unlocks the biometric gate, flags `self.manual_recalibration_requested = True`, and bypasses historical spatial memory to force a fresh anchor lock and posture baseline.

4. **15-Frame Consensus Gateway & Unregistered Short-Circuit:**
   New users ("Unknown") are subjected to a 15-frame rolling biometric similarity consensus. If they fail to match an established profile, their state is locked to `Unregistered Guest` or `Secondary Bystander`. In these states, **all posture tracking, voice alerts, and gaze checks are permanently short-circuited**. The system will not process them until they are officially registered via the Dashboard API.

---

## Getting Started

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Server:**
   ```bash
   python deskbot_v3.py
   ```
3. **Access Dashboard:**
   Navigate to `http://localhost:5000` to view the live analytics layer and register a user profile.
