# DeskBot V3 - Single-Target Biometric Anchor System

Welcome to the **DeskBot V3** ecosystem. This project implements a high-performance, single-target biometric posture tracking pipeline designed exclusively to secure and monitor a single primary user while aggressively blinding out background bystanders.

---

## 🏗️ Architecture Overview

The DeskBot V3 tracking pipeline operates via a strict chronological data flow, moving raw frame pixels through layers of spatial filtering and biometric verification before generating posture telemetry.

1. **Object Detection**: The frame is processed to detect human bounding boxes.
2. **IoU Spatial Filter**: If two bounding boxes overlap by > 40%, the smaller box is aggressively discarded to eliminate ghost tracking loops.
3. **Biometric Consensus Lock**:
   - The system scans the remaining bounding boxes and extracts facial embeddings.
   - Using a strict L2-Normalized Cosine Similarity, the system requires a `normalized_cosine_similarity >= 0.86` for **15 consecutive frames**. 
   - Upon successful verification, the engine locks onto the target as the **Primary Anchor** and permanently bypasses facial recognition to save compute.
4. **Single-Target Posture Evaluation and Hysteresis**:
   - The primary anchor undergoes rigorous geometric depth ratio mapping and neck pitch evaluations.
   - Temporal buffers (Hysteresis) absorb pixel noise and micro-movements to ensure state transition alerts are perfectly stable.

---

## 🌟 System Core Features

### 1. Bystander Blindness
If a coworker or family member walks into the camera frame behind you, the system forcefully clamps their track identity to `"Secondary Bystander"`. All posture physics calculations, facial crop routines, and temporal timers are strictly disabled for bystanders, guaranteeing zero CPU waste and zero data bleed into your health metrics.

### 2. Temporal Track Persistence
If the primary user momentarily leaves the camera frame, the system buffers the track loss. The memory pipeline is only flushed via an **Anti-Bleed Eviction** if the user is missing for `15` continuous frames, preventing minor occlusions (like covering your face or leaning out of view) from instantly breaking the session.

### 3. Manual Recalibration Overrides
Posture drift happens when you adjust your desk chair height. A `manual_recalibration_requested` override instantly breaks the biometric lock, flushes previous accumulator buffers, and forces a clean 3-second recalibration sweep without needing to restart the server.

---

## 🚀 Quick-Start Deployment Guide

To deploy the DeskBot V3 tracker, utilize the `uv` package manager in PowerShell:

```powershell
# 1. Clone the repository and navigate to the project root
cd path/to/dbot

# 2. Run the application utilizing UV (automatically resolves dependencies)
uv run python deskbot_v3.py
```
*Note: The frontend dashboard is hosted via Flask at `http://localhost:5000`.*

---

## 📖 Variable Glossary & State Matrix

### State Matrix
The `tracked_person.state` variable defines the exact lifecycle tier of the user:

| State | Trigger Condition |
|-------|-------------------|
| `Unregistered Target` | The user is actively tracked but matches no profiles. Health telemetry is halted. |
| `Calibrating` | User is locked. Accumulating 3.0s of baseline ratio/pitch averages. |
| `Tracking Active` | User is anchored and posture is within healthy `calibrated_baseline_neck_pitch` limits. |
| `Posture Deficit Alert` | User has sustained a `>0.70` pitch drop or `<0.90` depth drop for `2.5s`. |
| `Secondary Bystander` | Target is explicitly excluded from the Biometric Anchor loop. |

### Glossary of Key Internal Variables
* `biometric_consensus_frame_counter`: Frame accumulator required (15) to achieve anchor lock.
* `sustained_slouch_debounce_timer`: Floating-point clock required to trigger (2.5s) or clear (1.5s) alerts.
* `current_torso_depth_ratio`: The live depth approximation calculation of the tracked skeleton.
* `calibrated_baseline_neck_pitch`: The 3-second temporal average established during `Calibrating`.
