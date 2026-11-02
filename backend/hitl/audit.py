import json
import uuid
import sqlite3
from typing import Dict, Any, Optional
from backend.hitl.database import get_db_connection

def log_audit_event(
    session_id: str,
    event_type: str,
    payload: Dict[str, Any],
    review_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> str:
    """
    Appends an immutable audit event to audit_events.
    This table is append-only: never UPDATE or DELETE rows.
    """
    event_id = f"aud_{uuid.uuid4().hex[:16]}"
    payload_str = json.dumps(payload, ensure_ascii=False)

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        conn.execute(
            """
            INSERT INTO audit_events (event_id, review_id, session_id, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, review_id, session_id, event_type, payload_str)
        )
        if should_close:
            conn.commit()
    finally:
        if should_close:
            conn.close()

    return event_id
