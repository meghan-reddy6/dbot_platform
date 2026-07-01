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
        is_db_empty = (len(health_evaluator.profiles) == 0)
        for p in list(health_evaluator.tracked_persons.values()):
            if is_db_empty or p.track_id == health_evaluator.primary_user_track_id:
                snapshot.append({
                    "id": p.track_id, 
                    "name": p.name, 
                    "state": p.state,
                    "sitting_time": p.sitting_duration_clock, 
                    "standing_time": p.standing_duration_clock, 
                    "pitch": p.pitch,
                    "session_limit": p.session_limit,
                    "stand_requirement": p.stand_requirement,
                    "screen_gaze_current": round(p.screen_gaze_accumulation_timer),
                    "screen_gaze_max": p.screen_gaze_limit,
                    "ocular_break_current": round(p.ocular_break_timer),
                    "ocular_break_max": p.gaze_away_limit
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
                "screen_gaze_limit": p.get("screen_gaze_limit", 1200),
                "ocular_break_duration": p.get("ocular_break_duration", 20)
            })
        return jsonify(profiles_list)

@app.route('/api/profile/create', methods=['POST'])
def create_profile_endpoint():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Profile name is required."}), 400
        
    with state_mutex:
        person_to_register = None
        for p in health_evaluator.tracked_persons.values():
            if p.name == "Unknown (Ready for Registration)" or "Unknown" in p.name:
                person_to_register = p
                break
                
        if person_to_register is None:
            return jsonify({"error": "No unregistered skeleton targets currently visible in camera frame."}), 400
            
        try:
            db_conn.create_profile(name, person_to_register.embedding)
            person_to_register.name = name
            person_to_register.state = "Calibrating"
            person_to_register.calibration_start = time.time()
            person_to_register.calibration_accumulator = []
            person_to_register.calibration_announced = False
            health_evaluator.sync_profiles()
            return jsonify({"message": f"Successfully registered user profile for {name}."})
        except Exception as e:
            return jsonify({"error": f"Database insertion failed: {str(e)}"}), 500

@app.route('/api/profile/update', methods=['POST'])
def update_profile():
    data = request.get_json()
    db_conn.update_profile(
        data["name"], data["slouch_sensitivity"], data["session_limit"], data["stand_requirement"], data["screen_gaze_limit"], data["ocular_break_duration"]
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
        health_evaluator.system_was_manually_cleared = True
        db_conn.delete_profile(name)
        for p in health_evaluator.tracked_persons.values():
            if p.name == name:
                p.name = "Unknown"
                p.state = "Unregistered Guest"
                p.state_history_window.clear()
        health_evaluator.sync_profiles()
        return jsonify({"message": f"Successfully deleted user profile for {name}."})

@app.route('/api/profile/recalibrate', methods=['POST'])
def trigger_manual_recalibration():
    with state_mutex:
        health_evaluator.manual_recalibration_requested = True
    return jsonify({"status": "success", "message": "Manual recalibration cycle triggered successfully."})

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
            for p in health_evaluator.tracked_persons.values():
                x1, y1, x2, y2 = int(p.box[0]), int(p.box[1]), int(p.box[2]), int(p.box[3])
                color = (0, 255, 135) if p.is_verified or p.name != "Unknown" else (67, 159, 255)
                cv2.rectangle(annotated_layer, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_layer, f"{p.name} [{p.state}]", (x1, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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