import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir() -> str:
    """Directory for all runtime state (settings, database). Override with DESKBOT_DATA_DIR."""
    path = os.environ.get("DESKBOT_DATA_DIR") or os.path.join(REPO_ROOT, "data")
    os.makedirs(path, exist_ok=True)
    return path


def data_path(name: str) -> str:
    return os.path.join(data_dir(), name)


def uses_default_data_dir() -> bool:
    return not os.environ.get("DESKBOT_DATA_DIR")


def repo_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)
