import os
import signal
import sys
import threading
import time
import cv2
from flask import Flask, jsonify, request, render_template

from database.crud import DatabaseManager
from core.ingestion import DynamicCameraIngestion
from core.inference import AIInferenceEngine
from core.tracking import TrackerEngine

db_manager = DatabaseManager()
ingestion_pipeline = DynamicCameraIngestion()
inference_engine = AIInferenceEngine()
tracker_engine = TrackerEngine(inference_engine, db_manager)

app = Flask(__name__)
state_mutex = threading.Lock()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/metrics_slice', methods=['GET'])
def metrics_slice():
    snapshot = []
    with state_mutex:
        with tracker_engine.mutex:
            for tid, target in tracker_engine.targets.items():
                snapshot.append({
                    "id": str(target.track_id),
                    "name": str(target.name),
                    "state": str(target.state),
                    "sitting_time": float(target.sitting_duration_clock),
                    "standing_time": float(target.standing_duration_clock),
                    "pitch": float(target.pitch),
                    "session_limit": float(target.session_limit),
                    "stand_requirement": float(target.stand_requirement),
                    "gaze_away_time": float(target.gaze_away_clock)
                })
    return jsonify(snapshot)

@app.route('/api/profile/create', methods=['POST'])
def create_profile_api():
    data = request.json
    name = data.get('name')
    if not name: return jsonify({"error": "Name required"}), 400
    
    embedding = None
    with tracker_engine.mutex:
        for tid, target in tracker_engine.targets.items():
            if target.name == "Unknown":
                embedding = target.embedding
                break
                
    if embedding is None:
        return jsonify({"error": "No unregistered target found."}), 400
        
    try:
        db_manager.create_profile(name, embedding)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    tracker_engine.sync_profiles()
    
    with tracker_engine.mutex:
        for tid, target in tracker_engine.targets.items():
            if target.name == "Unknown":
                target.name = name
                target.state = "Calibrating"
                target.calibration_start = time.time()
                break
                
    return jsonify({"success": True})

@app.route('/api/profiles', methods=['GET'])
def get_profiles_api():
    profiles = db_manager.read_profiles()
    resp = []
    for name, p in profiles.items():
        resp.append({
            "user_name": name,
            "slouch_sensitivity": p["slouch_sensitivity"],
            "session_limit": p["session_limit"],
            "stand_requirement": p["stand_requirement"],
            "gaze_away_limit": p["gaze_away_limit"]
        })
    return jsonify(resp)

@app.route('/api/profile/update', methods=['POST'])
def update_profile_api():
    data = request.json
    name = data.get('name')
    slouch = data.get('slouch_sensitivity')
    limit = data.get('session_limit')
    stand_req = data.get('stand_requirement')
    gaze_limit = data.get('gaze_away_limit')
    db_manager.update_profile(name, slouch, limit, stand_req, gaze_limit)
    tracker_engine.sync_profiles()
    return jsonify({"success": True})

@app.route('/api/profile/delete', methods=['POST'])
def delete_profile_api():
    data = request.json
    name = data.get('name')
    db_manager.delete_profile(name)
    tracker_engine.sync_profiles()
    
    with tracker_engine.mutex:
        for tid, target in tracker_engine.targets.items():
            if target.name == name:
                target.name = "Unknown"
                target.state = "Unregistered Guest - Monitoring Suspended"
                
    return jsonify({"success": True})

latest_frame = None

def generate_frames():
    while True:
        with state_mutex:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
            else:
                frame_bytes = None
                
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    from flask import Response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def tracking_worker():
    global latest_frame
    while ingestion_pipeline.running:
        frame = ingestion_pipeline.get_frame(timeout=1.0)
        if frame is None:
            continue
        try:
            detections = inference_engine.run_inference(frame)
            tracker_engine.update(frame, detections, frame.shape)
            
            annotated_frame = frame.copy()
            with tracker_engine.mutex:
                for tid, target in tracker_engine.targets.items():
                    box = target.box
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    color = (0, 255, 0) if target.name != "Unknown" else (0, 0, 255)
                    if target.state == "Calibrating": color = (255, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{target.name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            with state_mutex:
                latest_frame = annotated_frame
                
        except Exception as e:
            print(f"Tracking error: {e}")
            time.sleep(0.1)

def signal_handler(sig, frame):
    print("\\n[!] SIGINT received. Commencing clean resource teardown...")
    ingestion_pipeline.stop()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    
    print("[*] Starting DeskBot V3 Pro Pipeline...")
    ingestion_pipeline.start()
    
    track_thread = threading.Thread(target=tracking_worker, daemon=True)
    track_thread.start()
    
    print("[*] Starting Flask Daemon on http://localhost:5000")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    finally:
        ingestion_pipeline.stop()
        print("[*] Clean Teardown Complete.")
