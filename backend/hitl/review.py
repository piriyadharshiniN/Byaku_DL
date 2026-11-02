import uuid
import sqlite3
import json
from typing import Dict, Any, Optional
from backend.hitl.database import get_db_connection
from backend.hitl.audit import log_audit_event
from backend.hitl.cache import compute_sha256, save_cache_frame
from backend.exceptions import ValidationException, NotFoundException

def ensure_session(
    session_id: str,
    input_source: str,
    conn: Optional[sqlite3.Connection] = None
) -> str:
    valid_sources = {"live_camera", "uploaded_image", "uploaded_video"}
    if input_source not in valid_sources:
        raise ValidationException(
            f"Invalid input_source '{input_source}'. Allowed: {', '.join(sorted(valid_sources))}.",
            code="INVALID_INPUT_SOURCE"
        )

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        row = conn.execute("SELECT session_id, status FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO sessions (session_id, input_source, status) VALUES (?, ?, 'active')",
                (session_id, input_source)
            )
            if should_close:
                conn.commit()
    finally:
        if should_close:
            conn.close()

    return session_id

def submit_review(
    session_id: str,
    submission_id: str,
    original_label: str,
    original_confidence: float,
    bbox: Dict[str, float],
    action: str,
    input_source: str,
    corrected_label: Optional[str] = None,
    track_id: Optional[int] = None,
    frame_id: Optional[str] = None,
    frame_cache_token: Optional[str] = None,
    frame_number: Optional[int] = None,
    video_timestamp_ms: Optional[int] = None,
    frame_dimensions: Optional[Dict[str, int]] = None,
    feedback_type: str = "detection_review",
    corrected_bbox: Optional[Dict[str, float]] = None,
    crop_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    """
    Submits a review record with strict validation, crop fingerprinting, and audit logging.
    """
    valid_actions = {"confirmed", "corrected", "rejected"}
    if action not in valid_actions:
        raise ValidationException(
            f"Invalid review action '{action}'. Must be one of: {', '.join(sorted(valid_actions))}.",
            code="INVALID_ACTION"
        )
    valid_feedback_types = {
        "detection_review",
        "class_correction",
        "false_positive",
        "missed_object",
        "bbox_correction",
        "class_and_bbox_correction",
    }
    if feedback_type not in valid_feedback_types:
        raise ValidationException(
            f"Invalid feedback_type '{feedback_type}'.",
            code="INVALID_FEEDBACK_TYPE"
        )

    if action == "corrected":
        if not corrected_label or not corrected_label.strip():
            raise ValidationException(
                "A non-empty 'corrected_label' is required when action is 'corrected'.",
                code="CORRECTED_LABEL_REQUIRED"
            )
        corrected_label = corrected_label.strip()
    else:
        if corrected_label is not None and corrected_label.strip() != "":
            raise ValidationException(
                f"Action '{action}' must not carry a corrected_label.",
                code="UNEXPECTED_CORRECTED_LABEL"
            )
        corrected_label = None

    if not (0.0 <= original_confidence <= 1.0):
        raise ValidationException(
            f"Confidence {original_confidence} out of bounds [0.0, 1.0].",
            code="INVALID_CONFIDENCE"
        )

    x1 = float(bbox.get("x1", 0))
    y1 = float(bbox.get("y1", 0))
    x2 = float(bbox.get("x2", 0))
    y2 = float(bbox.get("y2", 0))

    if x2 <= x1 or y2 <= y1:
        raise ValidationException(f"Invalid bounding box coordinates: ({x1}, {y1}, {x2}, {y2}).", code="INVALID_BBOX")

    if frame_dimensions:
        fw = frame_dimensions.get("width", 0)
        fh = frame_dimensions.get("height", 0)
        if fw > 0 and fh > 0:
            if x1 < 0 or y1 < 0 or x2 > fw + 10 or y2 > fh + 10:
                raise ValidationException(
                    f"Bounding box ({x1},{y1},{x2},{y2}) exceeds frame dimensions ({fw}x{fh}).",
                    code="BBOX_OUT_OF_BOUNDS"
                )

    normalized_corrected_bbox = None
    if corrected_bbox is not None:
        corrected_x1 = float(corrected_bbox.get("x1", 0))
        corrected_y1 = float(corrected_bbox.get("y1", 0))
        corrected_x2 = float(corrected_bbox.get("x2", 0))
        corrected_y2 = float(corrected_bbox.get("y2", 0))
        if corrected_x2 <= corrected_x1 or corrected_y2 <= corrected_y1:
            raise ValidationException(
                "Corrected bounding box must have positive width and height.",
                code="INVALID_CORRECTED_BBOX"
            )
        if frame_dimensions:
            fw = frame_dimensions.get("width", 0)
            fh = frame_dimensions.get("height", 0)
            if fw > 0 and fh > 0 and (
                corrected_x1 < 0 or corrected_y1 < 0 or corrected_x2 > fw or corrected_y2 > fh
            ):
                raise ValidationException(
                    f"Corrected bounding box ({corrected_x1},{corrected_y1},{corrected_x2},{corrected_y2}) "
                    f"exceeds frame dimensions ({fw}x{fh}).",
                    code="CORRECTED_BBOX_OUT_OF_BOUNDS"
                )
        normalized_corrected_bbox = {
            "x1": corrected_x1,
            "y1": corrected_y1,
            "x2": corrected_x2,
            "y2": corrected_y2,
        }

    # Compute crop fingerprint if crop bytes available
    crop_sha256 = None
    if crop_bytes:
        crop_sha256 = compute_sha256(crop_bytes)
        # Store crop in review cache using submission_id / hash
        save_cache_frame(f"crop_{crop_sha256[:16]}", crop_bytes)

    review_id = f"rev_{uuid.uuid4().hex[:16]}"

    conn = get_db_connection()
    try:
        with conn:
            ensure_session(session_id, input_source, conn=conn)

            conn.execute(
                """
                INSERT INTO reviews (
                    review_id, session_id, submission_id, track_id,
                    original_label, corrected_label, original_confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    corrected_bbox_json, action, feedback_type, input_source,
                    frame_cache_token, frame_id, frame_number, video_timestamp_ms, crop_sha256, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    review_id, session_id, submission_id, track_id,
                    original_label, corrected_label, original_confidence,
                    x1, y1, x2, y2,
                    json.dumps(normalized_corrected_bbox) if normalized_corrected_bbox else None,
                    action, feedback_type, input_source, frame_cache_token, frame_id, frame_number,
                    video_timestamp_ms, crop_sha256
                )
            )

            # Audit event
            event_type = "label_corrected" if action == "corrected" else ("confirmed" if action == "confirmed" else "rejected")
            log_audit_event(
                session_id=session_id,
                review_id=review_id,
                event_type=event_type,
                payload={
                    "submission_id": submission_id,
                    "action": action,
                    "original_label": original_label,
                    "corrected_label": corrected_label,
                    "feedback_type": feedback_type,
                    "original_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "corrected_bbox": normalized_corrected_bbox,
                    "frame_cache_token": frame_cache_token,
                    "frame_id": frame_id,
                    "frame_number": frame_number,
                    "confidence": original_confidence,
                    "crop_sha256": crop_sha256
                },
                conn=conn
            )
    finally:
        conn.close()

    return {
        "review_id": review_id,
        "session_id": session_id,
        "submission_id": submission_id,
        "action": action,
        "original_label": original_label,
        "corrected_label": corrected_label,
        "feedback_type": feedback_type,
        "frame_cache_token": frame_cache_token,
        "frame_id": frame_id,
        "frame_number": frame_number,
        "corrected_bbox": normalized_corrected_bbox,
        "status": "active",
        "crop_sha256": crop_sha256
    }

def undo_review(review_id: str) -> Dict[str, Any]:
    """
    Undoes a review action: updates status to 'undone' and appends an 'undone' audit event.
    Nothing is deleted from reviews or audit_events.
    """
    conn = get_db_connection()
    try:
        with conn:
            row = conn.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
            if not row:
                raise NotFoundException("Review record", review_id)

            if row["status"] == "undone":
                return {
                    "review_id": review_id,
                    "status": "undone",
                    "already_undone": True
                }

            conn.execute("UPDATE reviews SET status = 'undone' WHERE review_id = ?", (review_id,))

            log_audit_event(
                session_id=row["session_id"],
                review_id=review_id,
                event_type="undone",
                payload={
                    "previous_action": row["action"],
                    "original_label": row["original_label"],
                    "corrected_label": row["corrected_label"]
                },
                conn=conn
            )
    finally:
        conn.close()

    return {
        "review_id": review_id,
        "session_id": row["session_id"],
        "status": "undone",
        "message": "Review marked undone; audit log updated."
    }
