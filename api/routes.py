import time
import cv2
import csv
import io
from flask import (
    Blueprint,
    jsonify,
    request,
    render_template,
    Response,
    current_app,
    make_response,
)
from core.tracking import state_mutex
from analytics.session_manager import SessionManager
from utils.hardware import HardwareDetector

api_bp = Blueprint("api", __name__)


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
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Profile name is required."}), 400

    health_evaluator.pending_registration_name = name
    return jsonify({"message": f"Successfully queued registration for {name}."})


@api_bp.route("/api/profile/update", methods=["POST"])
def update_profile():
    """Updates configuration thresholds for an existing user."""
    health_evaluator = current_app.config["HEALTH_EVALUATOR"]
    db_conn = current_app.config["DB_CONN"]

    data = request.get_json()
    db_conn.update_profile(
        data["name"],
        data["slouch_sensitivity"],
        data["session_limit"],
        data["stand_requirement"],
        data["screen_gaze_limit"],
        data["ocular_break_duration"],
    )
    health_evaluator.sync_profiles()
    return jsonify({"message": "Synchronized profile configurations successfully."})


@api_bp.route("/api/profile/delete", methods=["POST"])
def delete_profile_endpoint():
    """Deletes an existing user profile."""
    health_evaluator = current_app.config["HEALTH_EVALUATOR"]
    db_conn = current_app.config["DB_CONN"]

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


@api_bp.route("/api/profile/recalibrate", methods=["POST"])
def trigger_manual_recalibration():
    """Triggers an explicit manual recalibration of the primary anchor."""
    health_evaluator = current_app.config["HEALTH_EVALUATOR"]
    with state_mutex:
        health_evaluator.manual_recalibration_requested = True
    return jsonify({"message": "Recalibration flagged in background loop."})


@api_bp.route("/api/profile/restart_session", methods=["POST"])
def trigger_manual_session_restart():
    """Restarts the active tracking session timers."""
    health_evaluator = current_app.config["HEALTH_EVALUATOR"]
    with state_mutex:
        health_evaluator.manual_session_restart_requested = True
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
    response.headers["Content-Disposition"] = (
        f"attachment; filename={user_name}_analytics.csv"
    )
    response.headers["Content-type"] = "text/csv"
    return response


@api_bp.route("/api/system/health")
def system_health():
    hardware = HardwareDetector.detect()
    health_evaluator = current_app.config.get("HEALTH_EVALUATOR", None)
    total_tracked = len(health_evaluator.tracked_persons) if health_evaluator else 0
    return jsonify(
        {
            "status": "healthy",
            "platform": hardware.platform_system,
            "arch": hardware.platform_machine,
            "is_embedded": hardware.is_embedded,
            "has_gpu": hardware.has_gpu,
            "has_npu": hardware.has_npu,
            "ram_gb": hardware.total_ram_gb,
            "tracked_targets": total_tracked,
            "last_system_message": health_evaluator.last_system_message if hasattr(health_evaluator, "last_system_message") else "",
        }
    )


def video_stream_generator(get_frame_func):
    """Generates JPEG frames from the global frame buffer for the dashboard."""
    while True:
        # In this refactored setup, we assume current_app has a method to fetch the latest frame
        # Or we can read it from a globally managed object passed via config
        latest_frame_buffer = get_frame_func()
        if latest_frame_buffer is not None:
            ret, jpeg = cv2.imencode(".jpg", latest_frame_buffer)
            if ret:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n\r\n"
                )
        time.sleep(0.04)


@api_bp.route("/video_feed")
def video_feed():
    func = current_app.config.get("GET_LATEST_FRAME")
    return Response(
        video_stream_generator(func),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@api_bp.route("/config")
def configuration_page():
    return render_template("configuration.html")

@api_bp.route("/api/settings", methods=["GET", "POST"])
def manage_settings():
    from config.settings_manager import settings
    if request.method == "POST":
        payload = request.get_json()
        settings.update(payload)
        return jsonify({"status": "success", "settings": settings.get_all()})
    return jsonify(settings.get_all())

@api_bp.route("/api/settings/reset", methods=["POST"])
def reset_settings_section():
    from config.settings_manager import settings
    payload = request.get_json()
    keys = payload.get("keys", [])
    if not keys and payload.get("restore_all"):
        settings.restore_defaults()
    else:
        settings.reset_section(keys)
    return jsonify({"status": "success", "settings": settings.get_all()})

@api_bp.route("/api/settings/export", methods=["GET"])
def export_settings():
    from config.settings_manager import settings
    import json
    settings_json = json.dumps(settings.get_all(), indent=4)
    return Response(
        settings_json,
        mimetype="application/json",
        headers={"Content-disposition": "attachment; filename=deskbot_settings.json"}
    )

@api_bp.route("/api/settings/import", methods=["POST"])
def import_settings():
    from config.settings_manager import settings
    import json
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    try:
        content = file.read().decode('utf-8')
        payload = json.loads(content)
        settings.update(payload)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

