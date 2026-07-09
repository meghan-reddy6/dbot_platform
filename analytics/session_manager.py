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
            if (
                is_db_empty
                or tracked_person.track_id == health_evaluator.primary_user_track_id
            ):
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

                snapshot.append(
                    {
                        "id": tracked_person.track_id,
                        "name": tracked_person.name,
                        "state": dashboard_state,
                        "sitting_time": tracked_person.sitting_duration_clock,
                        "standing_time": tracked_person.standing_duration_clock,
                        "pitch": tracked_person.pitch,
                        "spine_alignment": getattr(
                            tracked_person, "spine_alignment", 0.0
                        ),
                        "shoulder_alignment": getattr(
                            tracked_person, "shoulder_alignment", 0.0
                        ),
                        "session_limit": tracked_person.session_limit,
                        "stand_requirement": tracked_person.stand_requirement,
                        "screen_gaze_current": round(
                            tracked_person.screen_gaze_accumulation_timer
                        ),
                        "screen_gaze_max": tracked_person.screen_gaze_limit,
                        "ocular_break_current": round(
                            tracked_person.ocular_break_timer
                        ),
                        "ocular_break_max": tracked_person.gaze_away_limit,
                    }
                )
        return snapshot
