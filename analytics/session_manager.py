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
                slouch_time = getattr(session, "slouch_timer", 0.0) if session else 0.0
                
                # -------------------------------------------------------------
                # 5-Indicator Dashboard Subsystem State Evaluation
                # -------------------------------------------------------------
                raw_state = tracked_person.state or ""
                sit_limit = getattr(tracked_person, "session_limit", 1200)
                sit_pct = min(100, (sitting_time / sit_limit) * 100) if sit_limit > 0 else 0
                stand_req = getattr(tracked_person, "stand_requirement", 180)
                stand_pct = min(100, (standing_time / stand_req) * 100) if stand_req > 0 else 0

                # 1. Tracking State
                tracking_ind = "active"
                if "Calibrating" in raw_state:
                    tracking_ind = "calibrating"
                elif "Searching" in raw_state or "Unregistered" in raw_state or "Verifying" in raw_state:
                    tracking_ind = "searching"

                # 2. Posture State
                posture_ind = "healthy"
                if "Posture Deficit" in health:
                    if slouch_time > 60:
                        posture_ind = "critical"
                    else:
                        posture_ind = "slouching"
                
                if tracking_ind == "searching" or "Standing" in raw_state:
                    posture_ind = "inactive"

                # 3. Eyes State
                eyes_ind = "normal"
                if "Ocular Break" in health or "Looking Away" in raw_state or getattr(tracked_person, "is_looking_away", False):
                    eyes_ind = "ocular_break"
                if tracking_ind == "searching":
                    eyes_ind = "inactive"

                # 4. Sitting State
                sitting_ind = "normal"
                if "Session Limit Reached" in health or sit_pct >= 100:
                    sitting_ind = "exceeded"
                elif sit_pct >= 85:
                    sitting_ind = "warning"
                if "Standing" in raw_state:
                    sitting_ind = "inactive"

                # 5. Movement State
                movement_ind = "inactive"
                if "Standing" in raw_state:
                    if stand_pct >= 100:
                        movement_ind = "standing_complete"
                    else:
                        movement_ind = "standing_break"

                dashboard_indicators = {
                    "tracking": tracking_ind,
                    "posture": posture_ind,
                    "eyes": eyes_ind,
                    "sitting": sitting_ind,
                    "movement": movement_ind
                }

                snapshot.append(
                    {
                        "id": tracked_person.track_id,
                        "name": tracked_person.name,
                        "state": dashboard_state,
                        "sitting_time": sitting_time,
                        "standing_time": standing_time,
                        "pitch": tracked_person.pitch,
                        "slouch_time": slouch_time,
                        "dashboard_indicators": dashboard_indicators,
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
