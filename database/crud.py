import json
import logging
import os
import queue
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager

import numpy as np

from config.paths import data_path, repo_path, uses_default_data_dir
from config.settings_manager import settings

logger = logging.getLogger(__name__)

_DB_NAME = "wellness_logs.db"
_FLUSH = object()
_PURGE_INTERVAL_SECONDS = 24 * 3600
_PROFILE_LIMIT_COLUMNS = (
    "slouch_sensitivity", "session_limit", "stand_requirement", "screen_gaze_limit", "ocular_break_duration",
)


class ProfileExistsError(ValueError):
    pass


def default_db_path() -> str:
    path = data_path(_DB_NAME)
    legacy = repo_path(_DB_NAME)
    # Databases created before the data/ directory existed live in the repo root.
    if uses_default_data_dir() and not os.path.exists(path) and os.path.exists(legacy):
        shutil.move(legacy, path)
        logger.info(f"Moved database from {legacy} to {path}")
    return path


class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or default_db_path()
        self._write_queue = queue.Queue()
        self._init_db()
        self.purge_expired_analytics()
        self._last_purge = time.monotonic()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=10.0)

    @contextmanager
    def _transaction(self):
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._transaction() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_name TEXT PRIMARY KEY,
                    face_embedding BLOB,
                    session_limit INTEGER,
                    slouch_sensitivity REAL,
                    biometric_cutoff REAL,
                    stand_requirement INTEGER DEFAULT 180,
                    ocular_break_duration INTEGER DEFAULT 20,
                    screen_gaze_limit INTEGER DEFAULT 1200
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT,
                    duration REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_analytics_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_name TEXT,
                    session_state TEXT,
                    duration_seconds REAL,
                    continuous_sitting_seconds REAL,
                    continuous_gaze_seconds REAL,
                    average_head_pitch REAL,
                    ocular_break_accumulator REAL,
                    spine_alignment REAL DEFAULT 0.0,
                    shoulder_alignment REAL DEFAULT 0.0
                )
            """)
            for table, column in (
                ("users", "stand_requirement INTEGER DEFAULT 180"),
                ("users", "ocular_break_duration INTEGER DEFAULT 20"),
                ("users", "screen_gaze_limit INTEGER DEFAULT 1200"),
                ("users", "embedding_cluster BLOB"),
                ("users", "embedding_dim INTEGER"),
                ("user_analytics_logs", "spine_alignment REAL DEFAULT 0.0"),
                ("user_analytics_logs", "shoulder_alignment REAL DEFAULT 0.0"),
            ):
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
                except sqlite3.OperationalError:
                    pass

    def _writer_loop(self):
        """Serializes all telemetry writes off the inference thread, one commit per batch."""
        while True:
            batch = [self._write_queue.get()]
            while True:
                try:
                    batch.append(self._write_queue.get_nowait())
                except queue.Empty:
                    break
            writes = [item for item in batch if item[0] is not _FLUSH]
            try:
                if writes:
                    with self._transaction() as conn:
                        for sql, params in writes:
                            conn.execute(sql, params)
                if time.monotonic() - self._last_purge > _PURGE_INTERVAL_SECONDS:
                    self._last_purge = time.monotonic()
                    self.purge_expired_analytics()
            except Exception:
                # The writer must outlive any single bad batch, or every later flush() would block.
                logger.exception(f"Dropped {len(writes)} telemetry writes")
            finally:
                for item in batch:
                    if item[0] is _FLUSH:
                        item[1].set()

    def writer_alive(self) -> bool:
        return self._writer.is_alive()

    def flush(self, timeout: float = 10.0) -> bool:
        """Blocks until every queued telemetry write has been committed."""
        done = threading.Event()
        self._write_queue.put((_FLUSH, done))
        return done.wait(timeout)

    def purge_expired_analytics(self):
        days = int(settings.get("analytics_retention_days", 90))
        cutoff = f"-{days} days"
        try:
            with self._transaction() as conn:
                conn.execute("DELETE FROM user_analytics_logs WHERE timestamp < datetime('now', ?)", (cutoff,))
                conn.execute("DELETE FROM metrics WHERE timestamp < datetime('now', ?)", (cutoff,))
        except sqlite3.Error as e:
            logger.error(f"Analytics retention purge failed: {e}")

    def profile_exists(self, user_name) -> bool:
        with self._transaction() as conn:
            row = conn.execute("SELECT 1 FROM users WHERE user_name = ?", (user_name,)).fetchone()
        return row is not None

    def create_profile(
        self,
        user_name,
        face_embedding,
        embeddings=None,
        session_limit=None,
        slouch_sensitivity=None,
        biometric_cutoff=None,
        stand_requirement=None,
        ocular_break_duration=None,
        screen_gaze_limit=None,
    ):
        session_limit = session_limit if session_limit is not None else settings.get("session_limit_seconds", 1200)
        slouch_sensitivity = slouch_sensitivity if slouch_sensitivity is not None else settings.get("neck_pitch_tolerance_degrees", 15.0)
        biometric_cutoff = biometric_cutoff if biometric_cutoff is not None else settings.get("recognition_threshold_initial", 0.25)
        stand_requirement = stand_requirement if stand_requirement is not None else settings.get("stand_requirement_seconds", 180)
        ocular_break_duration = ocular_break_duration if ocular_break_duration is not None else settings.get("ocular_break_duration_seconds", 20)
        screen_gaze_limit = screen_gaze_limit if screen_gaze_limit is not None else settings.get("screen_gaze_limit_seconds", 1200)

        composite = np.asarray(face_embedding, dtype=np.float32)
        cluster = np.asarray(embeddings if embeddings is not None else [composite], dtype=np.float32)
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT INTO users (user_name, face_embedding, session_limit, slouch_sensitivity, biometric_cutoff, "
                    "stand_requirement, ocular_break_duration, screen_gaze_limit, embedding_cluster, embedding_dim) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_name,
                        composite.tobytes(),
                        session_limit,
                        slouch_sensitivity,
                        biometric_cutoff,
                        stand_requirement,
                        ocular_break_duration,
                        screen_gaze_limit,
                        cluster.tobytes(),
                        int(cluster.shape[1]),
                    ),
                )
        except sqlite3.IntegrityError as e:
            raise ProfileExistsError(f"A profile named {user_name!r} already exists.") from e

    def load_all_profiles(self):
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT user_name, face_embedding, session_limit, slouch_sensitivity, biometric_cutoff, "
                "stand_requirement, ocular_break_duration, screen_gaze_limit, embedding_cluster, embedding_dim FROM users"
            ).fetchall()

        profiles = {}
        for row in rows:
            embedding = np.frombuffer(row[1], dtype=np.float32)
            if row[8] is not None and row[9]:
                cluster = np.frombuffer(row[8], dtype=np.float32).reshape(-1, row[9])
            else:
                cluster = embedding.reshape(1, -1)
            profiles[row[0]] = {
                "embedding": embedding,
                "embeddings": cluster,
                "session_limit": row[2],
                "slouch_sensitivity": row[3],
                "biometric_cutoff": row[4],
                "stand_requirement": row[5] if row[5] is not None else 180,
                "ocular_break_duration": row[6] if row[6] is not None else 20,
                "screen_gaze_limit": row[7] if row[7] is not None else 1200,
            }
        return profiles

    def import_legacy_embedding_cache(self, path) -> int:
        """Moves clusters from the retired JSON cache into the DB, only for users that still exist."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, RecursionError) as e:
            logger.warning(f"Leaving unreadable legacy embedding cache {path} in place: {e}")
            return 0
        if not isinstance(data, dict):
            logger.warning(f"Leaving {path} in place: not a DeskBot embedding cache.")
            return 0

        imported = 0
        with self._transaction() as conn:
            existing = {row[0] for row in conn.execute("SELECT user_name FROM users")}
            for name, entry in data.items():
                if name not in existing:
                    continue
                vectors = entry.get("embeddings", entry) if isinstance(entry, dict) else entry
                try:
                    cluster = np.asarray(vectors, dtype=np.float32)
                except (TypeError, ValueError):
                    continue
                if cluster.ndim != 2 or cluster.shape[0] == 0:
                    continue
                cursor = conn.execute(
                    "UPDATE users SET embedding_cluster = ?, embedding_dim = ? "
                    "WHERE user_name = ? AND embedding_cluster IS NULL",
                    (cluster.tobytes(), int(cluster.shape[1]), name),
                )
                imported += cursor.rowcount
        os.remove(path)
        logger.info(f"Imported {imported} embedding clusters from {path} and removed the file.")
        return imported

    def update_profile(self, user_name, **limits) -> bool:
        columns = [c for c in _PROFILE_LIMIT_COLUMNS if c in limits]
        if not columns:
            raise ValueError("No profile limits to update.")
        assignments = ", ".join(f"{c} = ?" for c in columns)  # column names come from the allow-list only
        with self._transaction() as conn:
            cursor = conn.execute(
                f"UPDATE users SET {assignments} WHERE user_name = ?",
                [limits[c] for c in columns] + [user_name],
            )
            return cursor.rowcount > 0

    def delete_profile(self, user_name) -> bool:
        """Removes the biometric profile and every telemetry row recorded for it."""
        self.flush()
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM users WHERE user_name = ?", (user_name,))
            conn.execute("DELETE FROM user_analytics_logs WHERE user_name = ?", (user_name,))
            conn.execute("DELETE FROM metrics WHERE user_name = ?", (user_name,))
            return cursor.rowcount > 0

    def log_session_metrics(self, user_name, event_type, duration):
        # EXISTS guard: a write queued just before a delete must not recreate the user's history.
        self._write_queue.put((
            "INSERT INTO metrics (user_name, event_type, duration) "
            "SELECT ?, ?, ? WHERE EXISTS (SELECT 1 FROM users WHERE user_name = ?)",
            (user_name, event_type, duration, user_name),
        ))

    def log_analytics_flush(
        self,
        user_name,
        session_state,
        duration_seconds,
        continuous_sitting_seconds,
        continuous_gaze_seconds,
        average_head_pitch,
        ocular_break_accumulator,
        spine_alignment=0.0,
        shoulder_alignment=0.0,
    ):
        self._write_queue.put((
            """
            INSERT INTO user_analytics_logs (
                user_name, session_state, duration_seconds, continuous_sitting_seconds, continuous_gaze_seconds, average_head_pitch, ocular_break_accumulator, spine_alignment, shoulder_alignment
            ) SELECT ?, ?, ?, ?, ?, ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM users WHERE user_name = ?)
            """,
            (
                user_name,
                session_state,
                float(duration_seconds),
                float(continuous_sitting_seconds),
                float(continuous_gaze_seconds),
                float(average_head_pitch),
                float(ocular_break_accumulator),
                float(spine_alignment),
                float(shoulder_alignment),
                user_name,
            ),
        ))

    def get_user_analytics(self, user_name, limit=100):
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT timestamp, session_state, duration_seconds, continuous_sitting_seconds,
                       continuous_gaze_seconds, average_head_pitch, ocular_break_accumulator,
                       spine_alignment, shoulder_alignment
                FROM user_analytics_logs
                WHERE user_name = ?
                ORDER BY timestamp DESC LIMIT ?
            """,
                (user_name, limit),
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]
