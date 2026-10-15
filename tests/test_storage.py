import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from config.settings_manager import SettingsManager
from database.crud import DatabaseManager, ProfileExistsError


def _vec(i):
    v = np.zeros(128, np.float32)
    v[i] = 1.0
    return v


def test_only_overrides_are_saved_and_levels_drive_thresholds(tmp_path):
    path = tmp_path / "settings.json"
    manager = SettingsManager(filepath=str(path))
    manager.update({"slouch_sensitivity": "High", "eye_tracking_sensitivity": "Low", "camera": {"fps": 15}})

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["slouch_sensitivity"] == "High" and saved["camera"]["fps"] == 15
    assert "detection_fps" not in saved
    assert (manager.slouch_ratio_threshold, manager.gaze_away_threshold) == (0.90, 0.90)
    assert not list(tmp_path.glob(".settings-*"))


def test_corrupt_settings_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert SettingsManager(filepath=str(path)).get("detection_fps") == 30


def test_callers_cannot_mutate_settings_through_returned_values(tmp_path):
    manager = SettingsManager(filepath=str(tmp_path / "s.json"))
    manager.get("camera")["fps"] = 999
    manager.get_all()["camera"]["fps"] = 999
    assert manager.get("camera")["fps"] == 30


def test_invalid_values_in_the_settings_file_are_ignored_one_by_one(tmp_path):
    path = tmp_path / "settings.json"
    cached = {"init_arg": 0, "backend": 1400, "desc": "MSMF"}
    path.write_text(json.dumps({
        "detection_fps": "fast",
        "reminder_cooldown": 120,
        "screen_gaze_limit_seconds": 20,  # the old UI slider allowed 5-60 s
        "camera": {"fps": 0, "resolution": [640, 480], "cached_pipeline_0": cached},
        "unknown_key": 1,
    }), encoding="utf-8")

    manager = SettingsManager(filepath=str(path))

    assert (manager.get("detection_fps"), manager.get("reminder_cooldown")) == (30, 120)
    assert manager.get("screen_gaze_limit_seconds") == 1200
    camera = manager.get("camera")
    assert (camera["fps"], camera["resolution"], camera["cached_pipeline_0"]) == (30, [640, 480], cached)
    assert manager.get("unknown_key") is None


def test_wrongly_typed_or_hostile_settings_files_never_stop_startup(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"slouch_sensitivity": ["High"], "detection_fps": 10}), encoding="utf-8")
    assert SettingsManager(filepath=str(path)).get("detection_fps") == 10

    path.write_text("[" * 100000, encoding="utf-8")
    assert SettingsManager(filepath=str(path)).get("detection_fps") == 30


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(str(tmp_path / "t.db"))


def test_database_runs_in_wal_mode(db):
    with closing(sqlite3.connect(db.db_path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_duplicate_profile_is_rejected(db):
    db.create_profile("Alice", _vec(0))
    with pytest.raises(ProfileExistsError):
        db.create_profile("Alice", _vec(1))


def test_embedding_cluster_round_trips(db):
    cluster = np.stack([_vec(0), _vec(1), _vec(2)])
    db.create_profile("Alice", cluster.mean(axis=0), embeddings=cluster)
    assert np.array_equal(db.load_all_profiles()["Alice"]["embeddings"], cluster)


def test_telemetry_is_kept_only_for_registered_users(db):
    db.create_profile("Alice", _vec(0))
    db.log_analytics_flush("Mallory", "Tracking Active", 10, 0, 0, 0, 0)
    db.log_analytics_flush("Alice", "Tracking Active", 10, 0, 0, 0, 0)
    assert db.flush()
    assert len(db.get_user_analytics("Alice")) == 1
    assert db.get_user_analytics("Mallory") == []

    assert db.delete_profile("Alice")
    # Writes still in flight when a profile is deleted must not recreate its history.
    db.log_analytics_flush("Alice", "Tracking Active", 10, 0, 0, 0, 0)
    db.log_session_metrics("Alice", "session_ended", 5.0)
    assert db.flush()
    assert db.get_user_analytics("Alice") == []
    with closing(sqlite3.connect(db.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 0
    assert not db.delete_profile("Alice")


def test_retention_purges_old_telemetry(db):
    db.create_profile("Alice", _vec(0))
    with closing(sqlite3.connect(db.db_path)) as conn, conn:
        conn.execute("INSERT INTO user_analytics_logs (user_name, timestamp) VALUES ('Alice', datetime('now', '-400 days'))")
        conn.execute("INSERT INTO user_analytics_logs (user_name, timestamp) VALUES ('Alice', datetime('now'))")
    db.purge_expired_analytics()
    assert len(db.get_user_analytics("Alice")) == 1


def test_legacy_json_cache_is_imported_for_existing_users_then_removed(db, tmp_path):
    db.create_profile("Alice", _vec(0))
    with closing(sqlite3.connect(db.db_path)) as conn, conn:
        conn.execute("UPDATE users SET embedding_cluster = NULL, embedding_dim = NULL")  # as written by old builds
    cache = tmp_path / "profiles_cache.json"
    cache.write_text(json.dumps({"Alice": {"embeddings": [_vec(3).tolist()]}, "Ghost": [_vec(4).tolist()]}))

    assert db.import_legacy_embedding_cache(str(cache)) == 1
    assert not cache.exists()
    assert np.array_equal(db.load_all_profiles()["Alice"]["embeddings"], _vec(3)[None, :])
    assert "Ghost" not in db.load_all_profiles()


def test_unreadable_legacy_cache_is_left_alone(db, tmp_path):
    cache = tmp_path / "profiles_cache.json"
    cache.write_text('{"Alice": [[0.1, 0.2', encoding="utf-8")  # truncated, as the old writer could leave it
    assert db.import_legacy_embedding_cache(str(cache)) == 0
    assert cache.exists()
