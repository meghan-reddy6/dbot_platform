"""Tracker states. StrEnum members equal (and hash like) the strings older code and the dashboard use."""
from enum import StrEnum


class TrackState(StrEnum):
    UNREGISTERED = "Unregistered Guest"
    UNREGISTERED_TARGET = "Unregistered Target"
    VERIFYING = "Verifying Identity"
    CALIBRATING = "Calibrating"
    TRACKING = "Tracking Active"
    STANDING = "Standing"
    LOOKING_AWAY = "Looking Away"
    SEARCHING = "Searching / Re-acquiring"
    ABSENT = "Absent"
    # Listed as active by older code but never assigned by the tracker.
    POSTURE_DEFICIT = "Posture Deficit"
    OCULAR_BREAK = "Ocular Break"
    SESSION_LIMIT = "Session Limit"


class Verification(StrEnum):
    UNKNOWN = "UNKNOWN"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"


class HealthStatus(StrEnum):
    HEALTHY = "Healthy"
    POSTURE_DEFICIT = "Posture Deficit Alert"
    OCULAR_BREAK = "Ocular Break Recommended"
    SESSION_LIMIT = "Session Limit Reached - Stand Up!"


ACTIVE_STATES = frozenset({
    TrackState.TRACKING,
    TrackState.CALIBRATING,
    TrackState.POSTURE_DEFICIT,
    TrackState.OCULAR_BREAK,
    TrackState.SESSION_LIMIT,
    TrackState.STANDING,
})

UNKNOWN_NAME = "Unknown"
UNREGISTERED_GUEST_NAME = "Unknown [Unregistered Guest]"
AMBIGUOUS_MATCH_NAME = "Unknown [Ambiguous Match]"
LOW_CONFIDENCE_NAME = "Unknown [Low Confidence]"
UNKNOWN_CANDIDATES = frozenset({
    UNKNOWN_NAME,
    "Unknown [Collision Risk Reject]",
    AMBIGUOUS_MATCH_NAME,
    LOW_CONFIDENCE_NAME,
    UNREGISTERED_GUEST_NAME,
})
