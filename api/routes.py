import csv
import io
import json
from flask import (
    Blueprint,
    abort,
    jsonify,
    request,
    render_template,
    Response,
    current_app,
    make_response,
)
from werkzeug.exceptions import HTTPException
from core.tracking import state_mutex
from analytics.session_manager import SessionManager
from api.validation import (
    attachment_header,
    sanitize_settings,
    validate_profile_key,
    validate_profile_name,
    validate_profile_update,
    validate_settings_keys,
)
from config.settings_manager import settings
from utils.hardware import HardwareDetector

api_bp = Blueprint("api", __name__)

MAX_UPLOAD_BYTES = 1024 * 1024
MAX_VIDEO_VIEWERS = 3
BLACK_FRAME_GRACE_S = 3.0
CSRF_HEADER = "X-DeskBot-Request"
_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_placeholder_jpeg = None


def _hostname(host: str) -> str:
    host = (host or "").strip().lower()
    if host.startswith("["):
        return host[: host.find("]") + 1]
    return host.split(":", 1)[0]


@api_bp.record_once
def _limit_upload_size(state):
    # Flask pre-populates this key with None, so setdefault() would never apply the limit.
    if state.app.config.get("MAX_CONTENT_LENGTH") is None:
        state.app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@api_bp.before_app_request
def _reject_foreign_requests():
    # A foreign Host means DNS rebinding: another site's page talking to this loopback server.
    if _hostname(request.host) not in _ALLOWED_HOSTS:
        abort(400, "Unexpected Host header.")
    # Other sites may link to the dashboard, but not load its video, data or camera probes as subresources.
    if request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site") and not (
        request.method == "GET" and request.headers.get("Sec-Fetch-Mode") == "navigate"
    ):
        abort(403, "Cross-site request refused.")
    if request.method in _SAFE_METHODS:
        return None
    # Browsers only send custom headers cross-origin after a CORS preflight, which this app never approves.
    if request.headers.get(CSRF_HEADER) != "1":
        abort(403, f"Missing {CSRF_HEADER} header.")
    origin = request.headers.get("Origin")
    if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
        abort(403, "Cross-origin request refused.")
    return None


@api_bp.after_app_request
def _security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault(
        "Content-Security-Policy",
        "frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'self'; connect-src 'self'",
    )
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    return response


def _json_body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, "Request body must be a JSON object.")
    return data


def _public_settings() -> dict:
    current = settings.get_all()
    camera = current.get("camera", {})
    current["camera"] = {k: v for k, v in camera.items() if not k.startswith("cached_pipeline_")}
    return current


@api_bp.route("/")
def index():
    return render_template("dashboard.html")


@api_bp.route("/api/metrics_slice")
def get_metrics_slice():
    """Streams JSON metrics for the primary anchor track."""
    health_evaluator = current_app.config["HEALTH_EVALUATOR"]

    with state_mutex:
        snapshot = SessionManager.extract_live_metrics(health_evaluator)
        return jsonify(snapshot)


@api_bp.route("/api/profiles")
def get_profiles():
    """Returns all registered user profiles directly from the tracker cache."""
    health_evaluator = current_app.config["HEALTH_EVALUATOR"]
    with state_mutex:
        profiles_dict = {}
        for name, profile_data in health_evaluator.profiles.items():
            profiles_dict[name] = {
                "slouch_sensitivity": profile_data.get("slouch_sensitivity", 15.0),
                "session_limit": profile_data.get("session_limit", 2400),
                "stand_requirement": profile_data.get("stand_requirement", 120),
                "screen_gaze_limit": profile_data.get("screen_gaze_limit", 1200),
                "ocular_break_duration": profile_data.get("ocular_break_duration", 20),
            }
        return jsonify(profiles_dict)


@api_bp.route("/api/profile/create", methods=["POST"])
def create_profile_endpoint():
    """Creates a new biometric user profile from an unregistered target."""
    health_evaluator = current_app.config["HEALTH_EVALUATOR"]
    try:
        name = validate_profile_name(_json_body().get("name"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    with state_mutex:
        if name in health_evaluator.profiles:
            return jsonify({"error": f"A profile named {name} already exists."}), 409
        problem = health_evaluator.request_registration(name)
    if problem:
        return jsonify({"error": problem}), 409
    return jsonify({"message": f"Registration of {name} started; face the camera until it completes."})


@api_bp.route("/api/profile/update", methods=["POST"])
def update_profile():
    """Updates configuration thresholds for an existing user."""
    health_evaluator = current_app.config["HEALTH_EVALUATOR"]
    db_conn = current_app.config["DB_CONN"]

    try:
        data = validate_profile_update(_json_body())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    name = data.pop("name")
    if not db_conn.update_profile(name, **data):
        return jsonify({"error": f"No profile named {name}."}), 404
    health_evaluator.sync_profiles()
    return jsonify({"message": "Synchronized profile configurations successfully."})


@api_bp.route("/api/profile/delete", methods=["POST"])
def delete_profile_endpoint():
    """Deletes a profile together with its stored analytics."""
    health_evaluator = current_app.config["HEALTH_EVALUATOR"]
    try:
        name = validate_profile_key(_json_body().get("name"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not health_evaluator.remove_profile(name):
        return jsonify({"error": f"No profile named {name}."}), 404
    return jsonify({"message": f"Successfully deleted user profile for {name}."})


@api_bp.route("/api/profile/recalibrate", methods=["POST"])
def trigger_manual_recalibration():
    """Triggers an explicit manual recalibration of the primary anchor."""
    current_app.config["HEALTH_EVALUATOR"].request_recalibration()
    return jsonify({"message": "Recalibration flagged in background loop."})


@api_bp.route("/api/profile/restart_session", methods=["POST"])
def trigger_manual_session_restart():
    """Restarts the active tracking session timers."""
    current_app.config["HEALTH_EVALUATOR"].request_session_restart()
    return jsonify({"message": "Session restart flagged in background loop."})


@api_bp.route("/api/history/<user_name>")
def get_user_history(user_name):
    db_conn = current_app.config["DB_CONN"]
    logs = db_conn.get_user_analytics(user_name)
    return jsonify(logs)


@api_bp.route("/api/export/json/<user_name>")
def export_user_history_json(user_name):
    db_conn = current_app.config["DB_CONN"]
    logs = db_conn.get_user_analytics(user_name)
    return jsonify(logs)


@api_bp.route("/api/export/csv/<user_name>")
def export_user_history_csv(user_name):
    db_conn = current_app.config["DB_CONN"]
    logs = db_conn.get_user_analytics(user_name)

    output = io.StringIO()
    if logs:
        writer = csv.DictWriter(output, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = attachment_header(f"{user_name}_analytics.csv")
    response.headers["Content-type"] = "text/csv"
    return response


def _health_reasons(health_evaluator, camera_bridge, pipeline_status, db_conn) -> list:
    """Components that are configured but not working, in words an operator can act on."""
    reasons = []
    if camera_bridge is not None and camera_bridge.metrics.get("state") != "streaming":
        reasons.append(f"camera is {camera_bridge.metrics.get('state', 'unknown')}")
    if pipeline_status is not None:
        snapshot = pipeline_status.snapshot()
        age = snapshot["last_frame_age_s"]
        if pipeline_status.consecutive_errors:
            reasons.append(f"inference failing: {pipeline_status.last_error}")
        elif age is not None and age > 5.0:
            reasons.append(f"no frame processed for {age:.0f}s")
        elif age is None and snapshot["uptime_s"] > 10.0:
            reasons.append("no frame processed since start")
        if snapshot["black_for_s"] >= BLACK_FRAME_GRACE_S and age is not None and age <= 5.0:
            reasons.append(
                f"camera image has been black for {snapshot['black_for_s']:.0f}s "
                "(another app may be using the camera, or the lens is covered or in the dark)"
            )
    detector = getattr(health_evaluator, "person_detector", None)
    if detector is not None and getattr(detector, "session", True) is None:
        reasons.append("pose model is not loaded")
    worker = health_evaluator.worker_status() if hasattr(health_evaluator, "worker_status") else None
    if worker is not None and not worker["alive"]:
        reasons.append(f"face recognition worker is down (restarts: {worker['restarts']})")
    if db_conn is not None and hasattr(db_conn, "writer_alive") and not db_conn.writer_alive():
        reasons.append("database writer has stopped; telemetry is not being saved")
    return reasons


@api_bp.route("/api/system/health")
def system_health():
    hardware = HardwareDetector.detect()
    health_evaluator = current_app.config.get("HEALTH_EVALUATOR", None)
    camera_bridge = current_app.config.get("CAMERA_BRIDGE", None)
    pipeline_status = current_app.config.get("PIPELINE_STATUS", None)
    with state_mutex:
        total_tracked = len(health_evaluator.tracked_persons) if health_evaluator else 0
        last_message = getattr(health_evaluator, "last_system_message", "")
    reasons = _health_reasons(health_evaluator, camera_bridge, pipeline_status, current_app.config.get("DB_CONN"))
    detector = getattr(health_evaluator, "person_detector", None)
    worker = health_evaluator.worker_status() if hasattr(health_evaluator, "worker_status") else None
    return jsonify(
        {
            "status": "degraded" if reasons else "healthy",
            "reasons": reasons,
            "platform": hardware.platform_system,
            "arch": hardware.platform_machine,
            "is_embedded": hardware.is_embedded,
            "has_gpu": hardware.has_gpu,
            "has_npu": hardware.has_npu,
            "ram_gb": hardware.total_ram_gb,
            "ort_providers": hardware.get_ort_providers(),
            "pose_providers": list(getattr(detector, "providers", []) or []),
            "tracked_targets": total_tracked,
            "last_system_message": last_message,
            "camera_metrics": dict(camera_bridge.metrics) if camera_bridge else {},
            "pipeline": pipeline_status.snapshot() if pipeline_status else {},
            "biometric_worker": worker or {},
        }
    )

@api_bp.route("/api/cameras", methods=["GET"])
def get_cameras():
    camera_bridge = current_app.config.get("CAMERA_BRIDGE")
    if not camera_bridge:
        return jsonify({"error": "Camera subsystem not available"}), 500
    
    cameras = camera_bridge.discover_cameras()
    return jsonify({
        "available_cameras": cameras,
        "active_index": camera_bridge.active_camera_info["index"] if camera_bridge.active_camera_info else None
    })

@api_bp.route("/api/cameras/select", methods=["POST"])
def select_camera():
    camera_bridge = current_app.config.get("CAMERA_BRIDGE")
    if not camera_bridge:
        return jsonify({"error": "Camera subsystem not available"}), 500

    new_index = _json_body().get("index")
    if isinstance(new_index, bool) or not isinstance(new_index, int):
        return jsonify({"error": "Camera index must be an integer."}), 400
    if new_index not in {c["index"] for c in camera_bridge.discover_cameras()}:
        return jsonify({"error": f"No camera with index {new_index}."}), 404

    settings.update({"camera": {"preferred_camera": new_index}})
    camera_bridge.restart_async()
    return jsonify({"message": f"Switching to camera index {new_index}."}), 202


def _waiting_jpeg() -> bytes:
    global _placeholder_jpeg
    if _placeholder_jpeg is None:
        import cv2
        import numpy as np

        _placeholder_jpeg = cv2.imencode(".jpg", np.zeros((36, 64, 3), np.uint8))[1].tobytes()
    return _placeholder_jpeg


def video_stream_generator(pipeline_status):
    """Yields annotated JPEGs as an MJPEG stream, at least one part per second."""
    sequence = 0
    while True:
        sequence, jpeg = pipeline_status.wait_for_frame(sequence, timeout=1.0)
        # Repeat the last frame when nothing new arrived: the server only notices a closed client on write.
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + (jpeg or _waiting_jpeg()) + b"\r\n\r\n"


@api_bp.route("/video_feed")
def video_feed():
    pipeline_status = current_app.config.get("PIPELINE_STATUS")
    if pipeline_status is None:
        return jsonify({"error": "Video pipeline not available"}), 503
    if not pipeline_status.try_add_viewer(MAX_VIDEO_VIEWERS):
        return jsonify({"error": "Too many open video streams."}), 503
    response = Response(
        video_stream_generator(pipeline_status),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    # The server calls close() even if the client leaves before the first frame, so the slot is always freed.
    response.call_on_close(pipeline_status.remove_viewer)
    return response

@api_bp.route("/config")
def configuration_page():
    return render_template("configuration.html")

@api_bp.route("/api/settings", methods=["GET", "POST"])
def manage_settings():
    if request.method == "POST":
        try:
            clean = sanitize_settings(_json_body())
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        settings.update(clean)
        return jsonify({"status": "success", "settings": _public_settings()})
    return jsonify(_public_settings())

@api_bp.route("/api/settings/reset", methods=["POST"])
def reset_settings_section():
    payload = _json_body()
    try:
        keys = validate_settings_keys(payload.get("keys", []))
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    if not keys and payload.get("restore_all") is True:
        settings.restore_defaults()
    else:
        settings.reset_section(keys)
    return jsonify({"status": "success", "settings": _public_settings()})

@api_bp.route("/api/settings/export", methods=["GET"])
def export_settings():
    settings_json = json.dumps(_public_settings(), indent=4)
    return Response(
        settings_json,
        mimetype="application/json",
        headers={"Content-disposition": "attachment; filename=deskbot_settings.json"}
    )

@api_bp.route("/api/settings/import", methods=["POST"])
def import_settings():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    try:
        payload = json.loads(file.read().decode('utf-8'))
        clean = sanitize_settings(payload)
    except (UnicodeDecodeError, ValueError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    settings.update(clean)
    return jsonify({"status": "success"})


@api_bp.app_errorhandler(HTTPException)
def _json_errors(error):
    return jsonify({"error": error.description}), error.code

