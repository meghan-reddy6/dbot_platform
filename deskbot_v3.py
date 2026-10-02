import logging
import os
import threading

import flask.cli
from flask import Flask

from database.crud import DatabaseManager
from core.pipeline import PipelineStatus, inference_loop
from core.tracking import TrackerEngine
from camera.camera_manager import CameraManager
from api.routes import MAX_UPLOAD_BYTES, api_bp

# SURGICAL LOG SILENCER: Disables high-frequency HTTP request polling print streams completely
logging.getLogger("werkzeug").setLevel(logging.ERROR)
flask.cli.show_server_banner = lambda *args: None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeskBotV3.MainOrchestrator")

# Loopback only: the dashboard serves camera video and biometric data without authentication.
HOST = "127.0.0.1"
PORT = int(os.environ.get("DESKBOT_PORT", "5050"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.register_blueprint(api_bp)

# Each open video feed holds a thread for as long as the dashboard is open. Waitress buffers a whole
# request body (1 GiB by default) before Flask sees it, so the upload limit must be enforced here too.
WAITRESS_OPTIONS = {"threads": 8, "max_request_body_size": MAX_UPLOAD_BYTES + 1}  # waitress rejects at >=


def serve(flask_app):
    try:
        from waitress import serve as waitress_serve
    except ImportError:
        logger.warning("waitress is not installed; falling back to the Flask development server.")
        flask_app.run(host=HOST, port=PORT, debug=False, threaded=True)
        return
    waitress_serve(flask_app, host=HOST, port=PORT, **WAITRESS_OPTIONS)


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
    pipeline_status = PipelineStatus()
    stop_event = threading.Event()

    # Inject dependencies into Flask config for Blueprint to access
    app.config["DB_CONN"] = db_conn
    app.config["HEALTH_EVALUATOR"] = health_evaluator
    app.config["CAMERA_BRIDGE"] = camera_bridge
    app.config["PIPELINE_STATUS"] = pipeline_status

    try:
        camera_bridge.start()
        threading.Thread(
            target=inference_loop,
            args=(camera_bridge, health_evaluator, pipeline_status, stop_event),
            daemon=True,
            name="inference",
        ).start()
        logger.info(f"Server pipeline active at: http://localhost:{PORT}")
        serve(app)
    except KeyboardInterrupt:
        print("\n[!] SIGINT received. Commencing clean resource teardown...")
    finally:
        stop_event.set()
        camera_bridge.stop()
        health_evaluator.shutdown()
        db_conn.flush()
        print("[*] Clean Teardown Complete.")