from .state import state_mutex, Person

__all__ = ["Person", "state_mutex", "TrackerEngine"]


def __getattr__(name):
    # Imported lazily: orchestrator -> detection.person_detector -> core.tracking.geometry would
    # otherwise make "import detection.person_detector" fail when it is the first import.
    if name == "TrackerEngine":
        from .orchestrator import TrackerEngine

        return TrackerEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
