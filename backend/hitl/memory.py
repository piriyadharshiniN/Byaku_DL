import json
from typing import List, Dict, Any, Optional
from backend.hitl.database import get_db_connection
from backend.config import config

def query_reviews(
    session_id: Optional[str] = None,
    source: Optional[str] = None,
    action: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        query = "SELECT * FROM reviews WHERE 1=1"
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if source:
            query += " AND input_source = ?"
            params.append(source)
        if action:
            query += " AND action = ?"
            params.append(action)
        if status:
            query += " AND status = ?"
            params.append(status)
        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND created_at <= ?"
            params.append(date_to)

        query += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            results.append({
                "review_id": r["review_id"],
                "session_id": r["session_id"],
                "submission_id": r["submission_id"],
                "track_id": r["track_id"],
                "original_label": r["original_label"],
                "corrected_label": r["corrected_label"],
                "original_confidence": r["original_confidence"],
                "bbox": {
                    "x1": r["bbox_x1"],
                    "y1": r["bbox_y1"],
                    "x2": r["bbox_x2"],
                    "y2": r["bbox_y2"],
                },
                "corrected_bbox": json.loads(r["corrected_bbox_json"]) if r["corrected_bbox_json"] else None,
                "action": r["action"],
                "feedback_type": r["feedback_type"],
                "input_source": r["input_source"],
                "frame_cache_token": r["frame_cache_token"],
                "frame_id": r["frame_id"],
                "frame_number": r["frame_number"],
                "video_timestamp_ms": r["video_timestamp_ms"],
                "crop_sha256": r["crop_sha256"],
                "status": r["status"],
                "created_at": r["created_at"]
            })
        return results
    finally:
        conn.close()

def query_audit_events(
    session_id: Optional[str] = None,
    review_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if review_id:
            query += " AND review_id = ?"
            params.append(review_id)

        query += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            try:
                payload = json.loads(r["payload_json"])
            except Exception:
                payload = r["payload_json"]
            results.append({
                "event_id": r["event_id"],
                "review_id": r["review_id"],
                "session_id": r["session_id"],
                "event_type": r["event_type"],
                "payload": payload,
                "created_at": r["created_at"]
            })
        return results
    finally:
        conn.close()

def export_reviews_coco() -> Dict[str, Any]:
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM reviews
            WHERE status = 'active' AND action IN ('confirmed', 'corrected')
            ORDER BY created_at ASC, rowid ASC
            """
        ).fetchall()

        categories_map = {}
        images = []
        annotations = []
        seen_images = {}

        for r in rows:
            label = r["corrected_label"] if r["action"] == "corrected" and r["corrected_label"] else r["original_label"]
            if label not in categories_map:
                categories_map[label] = len(categories_map) + 1

            img_id = r["session_id"]
            if img_id not in seen_images:
                seen_images[img_id] = len(seen_images) + 1
                images.append({
                    "id": seen_images[img_id],
                    "file_name": f"{r['session_id']}.jpg",
                    "source": r["input_source"]
                })

            bbox = json.loads(r["corrected_bbox_json"]) if r["corrected_bbox_json"] else {
                "x1": r["bbox_x1"],
                "y1": r["bbox_y1"],
                "x2": r["bbox_x2"],
                "y2": r["bbox_y2"],
            }
            w = bbox["x2"] - bbox["x1"]
            h = bbox["y2"] - bbox["y1"]
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": seen_images[img_id],
                "category_id": categories_map[label],
                "bbox": [round(bbox["x1"], 1), round(bbox["y1"], 1), round(w, 1), round(h, 1)],
                "area": round(w * h, 1),
                "iscrowd": 0,
                "review_action": r["action"],
                "original_label": r["original_label"],
                "corrected_label": r["corrected_label"],
                "confidence": r["original_confidence"]
            })

        categories = [{"id": cid, "name": name} for name, cid in categories_map.items()]

        return {
            "info": {
                "description": "BYAKU Human-in-the-Loop Curated Dataset",
                "version": "1.0",
                "year": 2026
            },
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
    finally:
        conn.close()

def get_hitl_status() -> Dict[str, Any]:
    conn = get_db_connection()
    try:
        total_reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        active_reviews = conn.execute("SELECT COUNT(*) FROM reviews WHERE status = 'active'").fetchone()[0]
        undone_reviews = conn.execute("SELECT COUNT(*) FROM reviews WHERE status = 'undone'").fetchone()[0]
        total_audit = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        cache_count = 0
        if config.REVIEW_CACHE_DIR.exists():
            cache_count = sum(1 for f in config.REVIEW_CACHE_DIR.iterdir() if f.is_file() and not f.name.startswith("."))

        return {
            "database_ok": True,
            "total_sessions": total_sessions,
            "total_reviews": total_reviews,
            "active_reviews": active_reviews,
            "undone_reviews": undone_reviews,
            "total_audit_events": total_audit,
            "cache_files_count": cache_count,
            "cache_ttl_seconds": config.REVIEW_CACHE_TTL_SECONDS
        }
    finally:
        conn.close()
