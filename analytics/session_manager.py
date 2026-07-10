from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Handles extracting real-time metrics and historical telemetry data
    from active tracked targets, decoupling presentation logic from tracking state.
    """

    @staticmethod
    def extract_live_metrics(health_evaluator) -> List[Dict[str, Any]]:
        """
        Parses current tracked state into a JSON-friendly snapshot for dashboards.
        """
        snapshot = []
        is_db_empty = len(health_evaluator.profiles) == 0

        for tracked_person in list(health_evaluator.tracked_persons.values()):
            is_primary = (tracked_person.track_id == getattr(health_evaluator, "primary_user_track_id", None))
            has_active_session = tracked_person.name in health_evaluator.active_sessions and not tracked_person.name.startswith("Unknown")

            if is_db_empty or is_primary or has_active_session:
                # Calculate unified state string for the frontend HTML dashboard
                dashboard_state = tracked_person.state
                health = getattr(tracked_person, "health_status", "Healthy")

                if dashboard_state == "Tracking Active":
                    if "Posture Deficit Alert" in health:
                        dashboard_state = "Posture Deficit Alert"
                    elif "Session Limit Reached" in health:
                        dashboard_state = "Session Limit Reached"
                    elif "Ocular Break" in health:
                        dashboard_state = "Ocular Break"
                    elif getattr(tracked_person, "is_looking_away", False):
                        dashboard_state = "Looking Away"

                session = health_evaluator.active_sessions.get(tracked_person.name)
                
                sitting_time = getattr(session, "sitting_duration_clock", 0.0) if session else 0.0
                standing_time = getattr(session, "standing_duration_clock", 0.0) if session else 0.0
                gaze_current = getattr(session, "screen_gaze_accumulation_timer", 0.0) if session else 0.0
                ocular_current = getattr(session, "ocular_break_timer", 0.0) if session else 0.0

                snapshot.append(
                    {
                        "id": tracked_person.track_id,
                        "name": tracked_person.name,
                        "state": dashboard_state,
                        "sitting_time": sitting_time,
                        "standing_time": standing_time,
                        "pitch": tracked_person.pitch,
                        "spine_alignment": getattr(
                            tracked_person, "spine_alignment", 0.0
                        ),
                        "shoulder_alignment": getattr(
                            tracked_person, "shoulder_alignment", 0.0
                        ),
                        "session_limit": tracked_person.session_limit,
                        "stand_requirement": tracked_person.stand_requirement,
                        "screen_gaze_current": round(gaze_current),
                        "screen_gaze_max": tracked_person.screen_gaze_limit,
                        "ocular_break_current": round(ocular_current),
                        "ocular_break_max": tracked_person.gaze_away_limit,
                    }
                )
        return snapshot
