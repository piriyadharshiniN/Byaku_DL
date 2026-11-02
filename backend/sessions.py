import uuid
import time
from typing import Dict, Optional, Any
from backend.models.tracker import ByteTrackSessionTracker
from backend.spatial.scene_memory import SceneMemory
from backend.hitl.review import ensure_session
from backend.hitl.database import get_db_connection

class LiveSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.tracker = ByteTrackSessionTracker()
        self.memory = SceneMemory(session_id=session_id)
        self.created_at = time.time()
        self.last_active_at = time.time()
        self.last_scene_signature = ""
        self.last_speech_timestamp = 0.0
        self.frame_counter = 0
        self.is_active = True

    def touch(self):
        self.last_active_at = time.time()

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, LiveSession] = {}

    def start_live_session(self, session_id: Optional[str] = None) -> LiveSession:
        sid = session_id or f"live_{uuid.uuid4().hex[:12]}"
        session = LiveSession(sid)
        self._sessions[sid] = session
        ensure_session(sid, input_source="live_camera")
        return session

    def get_live_session(self, session_id: str) -> Optional[LiveSession]:
        return self._sessions.get(session_id)

    def stop_live_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session:
            session.is_active = False
            session.memory.clear()
            session.tracker.reset()
            # Update session status in SQLite
            try:
                conn = get_db_connection()
                with conn:
                    conn.execute(
                        "UPDATE sessions SET status = 'completed', ended_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                        (session_id,)
                    )
                conn.close()
            except Exception:
                pass
            return True
        return False

session_manager = SessionManager()
