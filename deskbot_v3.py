# d:/Thundersoft/dbot/deskbot_v3.py
import time
import cv2
import logging
import threading
from typing import Any
from flask import Flask, render_template, Response, request, jsonify

# SURGICAL LOG SILENCER: Disables high-frequency HTTP request polling print streams completely
logging.getLogger('werkzeug').setLevel(logging.ERROR)
import flask.cli
flask.cli.show_server_banner = lambda *args: None

from database.crud import DatabaseManager
from core.ingestion import DynamicCameraIngestion
from core.tracking import TrackerEngine, state_mutex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeskBotV3.MainOrchestrator")

app = Flask(__name__)
db_conn = DatabaseManager()
health_evaluator = TrackerEngine(db_conn)
camera_bridge = DynamicCameraIngestion()

latest_frame_buffer = None

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/metrics_slice')
def get_metrics_slice() -> Any:
    """Streams JSON metrics for the primary anchor track."""
    with state_mutex:
        snapshot = []
        is_db_empty = (len(health_evaluator.profiles) == 0)
        for tracked_person in list(health_evaluator.tracked_persons.values()):
            if is_db_empty or tracked_person.track_id == health_evaluator.primary_user_track_id:
                snapshot.append({
                    "id": tracked_person.track_id, 
                    "name": tracked_person.name, 
                    "state": tracked_person.state,
                    "sitting_time": tracked_person.sitting_duration_clock, 
                    "standing_time": tracked_person.standing_duration_clock, 
                    "pitch": tracked_person.pitch,
                    "session_limit": tracked_person.session_limit,
                    "stand_requirement": tracked_person.stand_requirement,
                    "screen_gaze_current": round(tracked_person.screen_gaze_accumulation_timer),
                    "screen_gaze_max": tracked_person.screen_gaze_limit,
                    "ocular_break_current": round(tracked_person.ocular_break_timer),
                    "ocular_break_max": tracked_person.gaze_away_limit
                })
        return jsonify(snapshot)

@app.route('/api/profiles')
def get_profiles() -> Any:
    """Returns all registered user profiles."""
    with state_mutex:
        profiles_list = []
        raw_profiles = db_conn.load_all_profiles()
        for name, profile_data in raw_profiles.items():
            profiles_list.append({
                "user_name": name,
                "slouch_sensitivity": profile_data["slouch_sensitivity"],
                "session_limit": profile_data["session_limit"],
                "stand_requirement": profile_data["stand_requirement"],
                "screen_gaze_limit": profile_data.get("screen_gaze_limit", 1200),
                "ocular_break_duration": profile_data.get("ocular_break_duration", 20)
            })
        return jsonify(profiles_list)

@app.route('/api/profile/create', methods=['POST'])
def create_profile_endpoint() -> Any:
    """Creates a new biometric user profile from an unregistered target."""
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Profile name is required."}), 400
        
    with state_mutex:
        person_to_register = None
        for tracked_person in health_evaluator.tracked_persons.values():
            if tracked_person.name == "Unknown (Ready for Registration)" or "Unknown" in tracked_person.name:
                person_to_register = tracked_person
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
def update_profile() -> Any:
    """Updates configuration thresholds for an existing user."""
    data = request.get_json()
    db_conn.update_profile(
        data["name"], data["slouch_sensitivity"], data["session_limit"], data["stand_requirement"], data["screen_gaze_limit"], data["ocular_break_duration"]
    )
    health_evaluator.sync_profiles()
    return jsonify({"message": "Synchronized profile configurations successfully."})

@app.route('/api/profile/delete', methods=['POST'])
def delete_profile_endpoint() -> Any:
    """Deletes an existing user profile."""
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Profile name is required."}), 400
    with state_mutex:
        health_evaluator.system_was_manually_cleared = True
        db_conn.delete_profile(name)
        for tracked_person in health_evaluator.tracked_persons.values():
            if tracked_person.name == name:
                tracked_person.name = "Unknown"
                tracked_person.state = "Unregistered Guest"
                tracked_person.state_history_window.clear()
        health_evaluator.sync_profiles()
        return jsonify({"message": f"Successfully deleted user profile for {name}."})

@app.route('/api/profile/recalibrate', methods=['POST'])
def trigger_manual_recalibration() -> Any:
    """Triggers an explicit manual recalibration of the primary anchor."""
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
        
        health_evaluator.process_frame_mot(frame, frame.shape)
        annotated_layer = frame.copy()
        
        with state_mutex:
            for tracked_person in health_evaluator.tracked_persons.values():
                x1, y1, x2, y2 = int(tracked_person.box[0]), int(tracked_person.box[1]), int(tracked_person.box[2]), int(tracked_person.box[3])
                color = (0, 255, 135) if tracked_person.is_verified or tracked_person.name != "Unknown" else (67, 159, 255)
                cv2.rectangle(annotated_layer, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_layer, f"{tracked_person.name} [{tracked_person.state}]", (x1, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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