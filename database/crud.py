import sqlite3
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'wellness_logs.db')

class DatabaseManager:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_name TEXT PRIMARY KEY,
                face_embedding BLOB,
                session_limit INTEGER,
                slouch_sensitivity REAL,
                biometric_cutoff REAL,
                stand_requirement INTEGER DEFAULT 180,
                gaze_away_limit INTEGER DEFAULT 20
            )
        ''')
        try:
            c.execute("ALTER TABLE users ADD COLUMN stand_requirement INTEGER DEFAULT 180")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute("ALTER TABLE users ADD COLUMN gaze_away_limit INTEGER DEFAULT 20")
        except sqlite3.OperationalError:
            pass
            
        c.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                duration REAL
            )
        ''')
        conn.commit()
        conn.close()

    def create_profile(self, user_name, face_embedding, session_limit=1200, slouch_sensitivity=15.0, biometric_cutoff=0.35, stand_requirement=180, gaze_away_limit=20):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (user_name, face_embedding, session_limit, slouch_sensitivity, biometric_cutoff, stand_requirement, gaze_away_limit) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (user_name, face_embedding.tobytes(), session_limit, slouch_sensitivity, biometric_cutoff, stand_requirement, gaze_away_limit))
        conn.commit()
        conn.close()

    def load_all_profiles(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT user_name, face_embedding, session_limit, slouch_sensitivity, biometric_cutoff, stand_requirement, gaze_away_limit FROM users")
        rows = c.fetchall()
        conn.close()
        
        profiles = {}
        for row in rows:
            user_name = row[0]
            embedding = np.frombuffer(row[1], dtype=np.float32)
            profiles[user_name] = {
                "embedding": embedding,
                "session_limit": row[2],
                "slouch_sensitivity": row[3],
                "biometric_cutoff": row[4],
                "stand_requirement": row[5] if row[5] is not None else 180,
                "gaze_away_limit": row[6] if row[6] is not None else 20
            }
        return profiles

    def update_profile(self, user_name, slouch_sensitivity, session_limit, stand_requirement, gaze_away_limit):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET slouch_sensitivity = ?, session_limit = ?, stand_requirement = ?, gaze_away_limit = ? WHERE user_name = ?", 
                  (slouch_sensitivity, session_limit, stand_requirement, gaze_away_limit, user_name))
        conn.commit()
        conn.close()

    def delete_profile(self, user_name):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE user_name = ?", (user_name,))
        conn.commit()
        conn.close()
        
    def log_session_metrics(self, user_name, event_type, duration):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO metrics (user_name, event_type, duration) VALUES (?, ?, ?)", 
                  (user_name, event_type, duration))
        conn.commit()
        conn.close()
