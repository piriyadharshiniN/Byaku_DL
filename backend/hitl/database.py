import sqlite3
from pathlib import Path
from typing import Optional
from backend.config import config

def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_database(db_path: Optional[Path] = None):
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                input_source TEXT NOT NULL CHECK (input_source IN ('live_camera', 'uploaded_image', 'uploaded_video')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP NULL,
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'terminated'))
            );

            CREATE TABLE IF NOT EXISTS reviews (
                review_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                submission_id TEXT NOT NULL,
                track_id INTEGER NULL,
                original_label TEXT NOT NULL,
                corrected_label TEXT NULL,
                original_confidence REAL NOT NULL CHECK (original_confidence >= 0.0 AND original_confidence <= 1.0),
                bbox_x1 REAL NOT NULL,
                bbox_y1 REAL NOT NULL,
                bbox_x2 REAL NOT NULL,
                bbox_y2 REAL NOT NULL,
                corrected_bbox_json TEXT NULL,
                action TEXT NOT NULL CHECK (action IN ('confirmed', 'corrected', 'rejected')),
                feedback_type TEXT NOT NULL DEFAULT 'detection_review',
                input_source TEXT NOT NULL CHECK (input_source IN ('live_camera', 'uploaded_image', 'uploaded_video')),
                frame_cache_token TEXT NULL,
                frame_id TEXT NULL,
                frame_number INTEGER NULL,
                video_timestamp_ms INTEGER NULL,
                crop_sha256 TEXT NULL,
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'undone')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                review_id TEXT NULL,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL CHECK (event_type IN ('review_created', 'label_corrected', 'confirmed', 'rejected', 'undone')),
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE RESTRICT
            );

            -- Indexes for fast query performance
            CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(input_source);
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
            CREATE INDEX IF NOT EXISTS idx_reviews_session ON reviews(session_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_source ON reviews(input_source);
            CREATE INDEX IF NOT EXISTS idx_reviews_action ON reviews(action);
            CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status);
            CREATE INDEX IF NOT EXISTS idx_reviews_created ON reviews(created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events(session_id);
            CREATE INDEX IF NOT EXISTS idx_audit_review ON audit_events(review_id);
            CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type);
            """)
            review_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(reviews)").fetchall()
            }
            if "frame_id" not in review_columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN frame_id TEXT NULL")
            if "frame_cache_token" not in review_columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN frame_cache_token TEXT NULL")
            if "feedback_type" not in review_columns:
                conn.execute(
                    "ALTER TABLE reviews ADD COLUMN feedback_type TEXT NOT NULL DEFAULT 'detection_review'"
                )
            if "corrected_bbox_json" not in review_columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN corrected_bbox_json TEXT NULL")
    finally:
        conn.close()
