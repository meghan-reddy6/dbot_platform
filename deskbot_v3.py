# d:/Thundersoft/dbot/deskbot_v3.py
import time
import cv2
import logging
import threading
from flask import Flask, render_template, Response, request, jsonify

# SURGICAL LOG SILENCER: Disables high-frequency HTTP request polling print streams completely
logging.getLogger('werkzeug').setLevel(logging.ERROR)
import flask.cli
flask.cli.show_server_banner = lambda *args: None

from database.crud import DatabaseManager
from core.ingestion import DynamicCameraIngestion
from core.inference import AIInferenceEngine
from core.tracking import TrackerEngine, state_mutex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeskBotV3.MainOrchestrator")

app = Flask(__name__)
db_conn = DatabaseManager()
inference_engine = AIInferenceEngine()
health_evaluator = TrackerEngine(inference_engine, db_conn)
camera_bridge = DynamicCameraIngestion()

latest_frame_buffer = None

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/metrics_slice')
def get_metrics_slice():
    with state_mutex:
        snapshot = []
        for t in list(health_evaluator.targets.values()):
            snapshot.append({
                "id": t.track_id, 
                "name": t.name, 
                "state": t.state,
                "sitting_time": t.sitting_duration_clock, 
                "standing_time": t.standing_duration_clock, 
                "pitch": t.pitch,
                "session_limit": t.session_limit,
                "stand_requirement": t.stand_requirement,
                "gaze_away_time": t.gaze_away_clock
            })
        return jsonify(snapshot)

@app.route('/api/profiles')
def get_profiles():
    with state_mutex:
        profiles_list = []
        raw_profiles = db_conn.load_all_profiles()
        for name, p in raw_profiles.items():
            profiles_list.append({
                "user_name": name,
                "slouch_sensitivity": p["slouch_sensitivity"],
                "session_limit": p["session_limit"],
                "stand_requirement": p["stand_requirement"],
                "gaze_away_limit": p["gaze_away_limit"]
            })
        return jsonify(profiles_list)

@app.route('/api/profile/create', methods=['POST'])
def create_profile_endpoint():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Profile name is required."}), 400
        
    with state_mutex:
        target_to_register = None
        for t in health_evaluator.targets.values():
            if t.name == "Unknown" or t.state == "Unregistered Guest":
                target_to_register = t
                break
                
        if target_to_register is None:
            return jsonify({"error": "No unregistered skeleton targets currently visible in camera frame."}), 400
            
        try:
            db_conn.create_profile(name, target_to_register.embedding)
            target_to_register.name = name
            target_to_register.state = "Calibrating"
            target_to_register.calibration_start = time.time()
            target_to_register.calibration_accumulator = []
            target_to_register.calibration_announced = False
            health_evaluator.sync_profiles()
            return jsonify({"message": f"Successfully registered user profile for {name}."})
        except Exception as e:
            return jsonify({"error": f"Database insertion failed: {str(e)}"}), 500

@app.route('/api/profile/update', methods=['POST'])
def update_profile():
    data = request.get_json()
    db_conn.update_profile(
        data["name"], data["slouch_sensitivity"], data["session_limit"], data["stand_requirement"], data["gaze_away_limit"]
    )
    health_evaluator.sync_profiles()
    return jsonify({"message": "Synchronized profile configurations successfully."})

@app.route('/api/profile/delete', methods=['POST'])
def delete_profile_endpoint():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Profile name is required."}), 400
    with state_mutex:
        db_conn.delete_profile(name)
        for t in health_evaluator.targets.values():
            if t.name == name:
                t.name = "Unknown"
                t.state = "Unregistered Guest"
                t.state_history_window.clear()
        health_evaluator.sync_profiles()
        return jsonify({"message": f"Successfully deleted user profile for {name}."})

def video_stream_generator():
    while True:
        if latest_frame_buffer is not None:
            ret, jpeg = cv2.imencode('.jpg', latest_frame_buffer)
            if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(video_stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

def master_inference_loop():
    global latest_frame_buffer
    while camera_bridge.running:
        frame = camera_bridge.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        h, w, _ = frame.shape
        detections = inference_engine.run_inference(frame)
        
        health_evaluator.update(frame, detections, frame.shape)
        annotated_layer = frame.copy()
        
        with state_mutex:
            for t in health_evaluator.targets.values():
                x1, y1, x2, y2 = int(t.box[0]), int(t.box[1]), int(t.box[2]), int(t.box[3])
                color = (0, 255, 135) if t.is_verified or t.name != "Unknown" else (67, 159, 255)
                cv2.rectangle(annotated_layer, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_layer, f"{t.name} [{t.state}]", (x1, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            latest_frame_buffer = annotated_layer

if __name__ == "__main__":
    try:
        camera_bridge.start()
        threading.Thread(target=master_inference_loop, daemon=True).start()
        logger.info("Server pipeline active at: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[!] SIGINT received. Commencing clean resource teardown...")
    finally:
        camera_bridge.stop()
        print("[*] Clean Teardown Complete.")