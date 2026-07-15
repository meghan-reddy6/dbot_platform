import time
import cv2
import logging
import threading
import flask.cli
from flask import Flask

from database.crud import DatabaseManager
from core.tracking import TrackerEngine, state_mutex
from camera.camera_manager import CameraManager
from api.routes import api_bp

# SURGICAL LOG SILENCER: Disables high-frequency HTTP request polling print streams completely
logging.getLogger("werkzeug").setLevel(logging.ERROR)
flask.cli.show_server_banner = lambda *args: None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeskBotV3.MainOrchestrator")

app = Flask(__name__)
db_conn = None
health_evaluator = None
camera_bridge = None

latest_frame_buffer = None
app.register_blueprint(api_bp)


def get_latest_frame():
    global latest_frame_buffer
    return latest_frame_buffer


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
                if tracked_person.state in ["Searching / Re-acquiring", "Absent"]:
                    continue
                x1, y1, x2, y2 = (
                    int(tracked_person.box[0]),
                    int(tracked_person.box[1]),
                    int(tracked_person.box[2]),
                    int(tracked_person.box[3]),
                )
                color = (
                    (0, 255, 135)
                    if tracked_person.is_verified or tracked_person.name != "Unknown"
                    else (67, 159, 255)
                )
                cv2.rectangle(annotated_layer, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_layer,
                    f"{tracked_person.name} [{tracked_person.state}]",
                    (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            latest_frame_buffer = annotated_layer


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    
    # ---------------------------------------------------------
    # MODEL MANAGER BOOTSTRAP (Validates & downloads AI weights)
    # ---------------------------------------------------------
    from core.model_manager import ModelManager
    ModelManager().bootstrap()

    db_conn = DatabaseManager()
    health_evaluator = TrackerEngine(db_conn)
    camera_bridge = CameraManager(camera_index=0)

    # Inject dependencies into Flask config for Blueprint to access
    app.config["DB_CONN"] = db_conn
    app.config["HEALTH_EVALUATOR"] = health_evaluator
    app.config["GET_LATEST_FRAME"] = get_latest_frame

    try:
        camera_bridge.start()
        threading.Thread(target=master_inference_loop, daemon=True).start()
        logger.info("Server pipeline active at: http://localhost:5050")
        app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[!] SIGINT received. Commencing clean resource teardown...")
    finally:
        camera_bridge.stop()
        print("[*] Clean Teardown Complete.")
