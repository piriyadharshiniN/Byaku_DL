import sys
import os
import time
import logging
import binascii
import base64
import uuid
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from backend.config import config
from backend.exceptions import (
    ByakuException, ModelUnavailableException, ValidationException,
    NotFoundException, GoneException
)
from backend.envelope import api_success, api_error, byaku_exception_handler, generic_exception_handler
from backend.validation import validate_image_bytes, validate_video_file
from backend.models.detector import ObjectDetector
from backend.models.depth import DepthAnythingV2Estimator
from backend.models.tracker import ByteTrackSessionTracker
from backend.spatial.reasoning import (
    compute_direction, compute_horizontal_position, compute_vertical_position,
    compute_combined_position, build_scene_graph, extract_relationships
)
from backend.spatial.scene_memory import SceneMemory
from backend.voice.narration import generate_scene_narration, generate_object_phrase
from backend.voice.qa import SceneQAEngine
from backend.hitl.database import init_database, get_db_connection
from backend.hitl.review import submit_review, undo_review, ensure_session
from backend.hitl.memory import query_reviews, query_audit_events, export_reviews_coco, get_hitl_status
from backend.hitl.cache import save_cache_frame, get_cache_frame, sweep_review_cache, sweep_uploads
from backend.sessions import session_manager
from backend.jobs import video_jobs

logger = logging.getLogger(__name__)


class LivePipelineError(ByakuException):
    def __init__(
        self,
        stage: str,
        message: str,
        code: str = "LIVE_PIPELINE_FAILED",
        status_code: int = 500,
    ):
        super().__init__(code, message, status_code, details={"stage": stage})
        self.stage = stage


def encode_depth_preview(depth_map: Optional[np.ndarray]) -> Optional[str]:
    if depth_map is None:
        return None
    depth_u8 = np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)
    color_map = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    encoded, buffer = cv2.imencode(".jpg", color_map, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not encoded:
        raise RuntimeError("Could not encode the predicted depth map for display.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")

# Global state for lifespan & route access
models_state: Dict[str, Any] = {
    "detector": None,
    "depth_estimator": None,
    "qa_engine": None,
    "device": "cpu",
    "startup_time": 0.0
}

def init_models():
    """Initializes models and shared state once. Idempotent."""
    if models_state.get("detector") is not None:
        return models_state["detector"], models_state["depth_estimator"], models_state["qa_engine"]

    models_state["startup_time"] = time.time()
    init_database()
    sweep_review_cache()
    sweep_uploads()

    detector = ObjectDetector()
    depth_estimator = DepthAnythingV2Estimator()
    qa_engine = SceneQAEngine()

    if hasattr(app, "state"):
        app.state.detector = detector
        app.state.depth_estimator = depth_estimator
        app.state.qa_engine = qa_engine

    models_state["detector"] = detector
    models_state["depth_estimator"] = depth_estimator
    models_state["qa_engine"] = qa_engine
    models_state["device"] = detector.device

    return detector, depth_estimator, qa_engine


def cleanup_models():
    detector = models_state.get("detector")
    if detector and hasattr(detector, "cleanup"):
        detector.cleanup()
    depth_estimator = models_state.get("depth_estimator")
    if depth_estimator and hasattr(depth_estimator, "cleanup"):
        depth_estimator.cleanup()

    models_state.update({
        "detector": None,
        "depth_estimator": None,
        "qa_engine": None,
        "device": "cpu",
        "startup_time": 0.0,
    })
    for name in ("detector", "depth_estimator", "qa_engine"):
        if hasattr(app.state, name):
            setattr(app.state, name, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    yield
    cleanup_models()
    sweep_review_cache()

app = FastAPI(
    title="BYAKU Vision & Spatial Audio API",
    description="Assistive perception aid providing relative spatial orientation and audio guidance. Not a mobility aid or obstacle avoidance system. Depth is relative, never metres.",
    version="1.0.0",
    lifespan=lifespan
)

# Also register standard startup/shutdown event handlers for compatibility with Uvicorn 0.17.6 / Starlette 0.26
@app.on_event("startup")
def on_startup():
    init_models()

@app.on_event("shutdown")
def on_shutdown():
    cleanup_models()
    sweep_review_cache()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers for standard envelope
app.add_exception_handler(ByakuException, byaku_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# -------------------------------------------------------------
# Health & Status Endpoints
# -------------------------------------------------------------
@app.get("/health")
@app.get("/api/health")
async def health_check():
    init_models()
    detector = getattr(app.state, "detector", models_state.get("detector"))
    depth_estimator = getattr(app.state, "depth_estimator", models_state.get("depth_estimator"))

    db_ok = False
    try:
        conn = get_db_connection()
        res = conn.execute("PRAGMA quick_check;").fetchone()
        db_ok = (res[0] == "ok")
        conn.close()
    except Exception:
        db_ok = False

    data = {
        "status": "healthy" if db_ok else "degraded",
        "device": detector.device if detector else models_state.get("device", "cpu"),
        "models_loaded": {
            "yolo_detector": bool(detector and detector.loaded),
            "depth_anything_v2": bool(depth_estimator and depth_estimator.loaded)
        },
        "model_status": {
            "yolo": {
                "loaded": bool(detector and detector.loaded),
                "backend": detector.backend_type if detector else None,
                "model": Path(detector.model_path).name if detector else None,
                "error": detector.load_error if detector and not detector.loaded else None,
            },
            "depth": {
                "loaded": bool(depth_estimator and depth_estimator.loaded),
                "backend": depth_estimator.backend_type if depth_estimator else None,
                "model": depth_estimator.model_name if depth_estimator else None,
                "last_inference_error": depth_estimator.last_inference_error if depth_estimator else None,
                "error": depth_estimator.load_error if depth_estimator and not depth_estimator.loaded else None,
            },
        },
        "model_paths": {
            "yolo": config.YOLO_MODEL_PATH,
            "depth": config.DEPTH_MODEL_NAME
        },
        "versions": {
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "fastapi": "0.92.0"
        },
        "database_ok": db_ok,
        "disclaimer": "Assistive perception aid, NOT a mobility aid or guaranteed obstacle avoidance. Depth is relative, never metres."
    }
    return api_success(data)

@app.get("/api/status")
async def api_pipeline_status():
    """Endpoint consumed by the dashboard frontend."""
    init_models()
    detector = getattr(app.state, "detector", models_state.get("detector"))
    depth_estimator = getattr(app.state, "depth_estimator", models_state.get("depth_estimator"))
    hitl_stat = get_hitl_status()

    det_loaded = bool(detector and detector.loaded)
    depth_loaded = bool(depth_estimator and depth_estimator.loaded)

    return {
        "backend": "connected",
        "camera": "ready",
        "detection": "loaded" if det_loaded else "unloaded",
        "depth": "loaded" if depth_loaded else "fallback_unknown",
        "tracking": "ready",
        "tts": "ready",
        "device": detector.device if detector else models_state.get("device", "cpu"),
        "models_loaded": {
            "yolo_detector": det_loaded,
            "depth_anything_v2": depth_loaded
        },
        "model_status": {
            "yolo": {
                "loaded": det_loaded,
                "backend": detector.backend_type if detector else None,
                "model": Path(detector.model_path).name if detector else None,
                "error": detector.load_error if detector and not det_loaded else None,
            },
            "depth": {
                "loaded": depth_loaded,
                "backend": depth_estimator.backend_type if depth_estimator else None,
                "model": depth_estimator.model_name if depth_estimator else None,
                "last_inference_error": depth_estimator.last_inference_error if depth_estimator else None,
                "error": depth_estimator.load_error if depth_estimator and not depth_loaded else None,
            },
        },
        "stage_status": {
            "camera": "ready",
            "detection": "loaded" if det_loaded else "unloaded",
            "tracking": "ready",
            "depth": "loaded" if depth_loaded else "fallback_unknown",
            "spatial": "ready",
            "scene_graph": "ready",
            "reasoning": "ready",
            "narration": "ready",
            "review": "active",
            "tts": "ready"
        },
        "hitl": hitl_stat
    }


# -------------------------------------------------------------
# Live Camera Session Endpoints
# -------------------------------------------------------------
@app.post("/api/sessions/live/start")
async def start_live_session():
    session = session_manager.start_live_session()
    return api_success({
        "session_id": session.session_id,
        "status": "active",
        "created_at": session.created_at
    })

@app.post("/api/sessions/live/{session_id}/stop")
async def stop_live_session(session_id: str):
    success = session_manager.stop_live_session(session_id)
    if not success:
        raise NotFoundException("Live session", session_id)
    return api_success({
        "session_id": session_id,
        "status": "terminated"
    })


async def analyze_live_frame(
    session: Any,
    frame_id: int,
    frame_bytes: bytes,
    image_bgr: np.ndarray,
    width: int,
    height: int,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    init_models()
    detector = getattr(app.state, "detector", models_state.get("detector"))
    depth_estimator = getattr(app.state, "depth_estimator", models_state.get("depth_estimator"))
    if not detector or not detector.loaded:
        raise LivePipelineError(
            "detection",
            (detector.load_error if detector else None) or "YOLO detector is not initialized.",
            code="MODEL_UNAVAILABLE",
            status_code=503,
        )

    try:
        raw_detections, detector_meta = await run_in_threadpool(
            detector.predict_with_meta, image_bgr
        )
    except Exception as error:
        logger.exception("YOLO inference failed for live session %s.", session.session_id)
        raise LivePipelineError("detection", f"YOLO inference failed: {error}") from error

    tracked = session.tracker.update(raw_detections)
    depth_map = None
    depth_error = None
    depth_inference_ms = None
    if depth_estimator and depth_estimator.loaded:
        try:
            depth_map = await run_in_threadpool(depth_estimator.estimate_frame, image_bgr)
            depth_error = depth_estimator.last_inference_error
            depth_inference_ms = depth_estimator.last_inference_ms
        except Exception as error:
            depth_error = f"{type(error).__name__}: {error}"
            logger.exception("Depth inference failed for live session %s.", session.session_id)
    else:
        depth_error = depth_estimator.load_error if depth_estimator else "Depth model is not initialized."

    frame_objects = []
    for item in tracked:
        detection = item["detection"]
        track_id = item["track_id"]
        horizontal, vertical = compute_direction(detection.bbox, width, height)
        relative_depth = (
            depth_estimator.evaluate_object_depth(depth_map, detection.bbox, (height, width))
            if depth_estimator
            else {
                "depth_value": None,
                "depth_category": "UNKNOWN",
                "depth_confidence": "low",
            }
        )
        x1 = round(detection.bbox["x1"], 1)
        y1 = round(detection.bbox["y1"], 1)
        x2 = round(detection.bbox["x2"], 1)
        y2 = round(detection.bbox["y2"], 1)
        frame_objects.append({
            "id": str(track_id) if track_id is not None else f"untracked_{len(frame_objects) + 1}",
            "track_id": track_id,
            "track_status": item["state"],
            "movement_direction": item["movement_direction"],
            "velocity_px_per_frame": item["velocity_px_per_frame"],
            "label": detection.label,
            "class_name": detection.label,
            "model_class_name": detection.label,
            "confidence": round(detection.confidence, 4),
            "bbox": detection.bbox,
            "bounding_box": [x1, y1, x2, y2],
            "box": [x1, y1, x2, y2],
            "center_x": round(detection.center_x, 1),
            "center_y": round(detection.center_y, 1),
            "position": {
                "horizontal": horizontal,
                "vertical": vertical,
                "combined": compute_combined_position(horizontal, vertical),
            },
            "direction_h": horizontal,
            "direction_v": vertical,
            "horizontal_position": horizontal,
            "vertical_position": vertical,
            "combined_position": compute_combined_position(horizontal, vertical),
            "depth": {
                "category": relative_depth["depth_category"],
                "value": relative_depth["depth_value"],
            },
            "depth_category": relative_depth["depth_category"],
            "depth_class": relative_depth["depth_category"],
            "depth_value": relative_depth["depth_value"],
            "relative_depth_value": relative_depth["depth_value"],
            "depth_confidence": relative_depth["depth_confidence"],
            "status": "current",
        })

    scene_graph = build_scene_graph(frame_objects, width, height)
    timestamp = time.time()
    session.memory.update(
        frame_number=frame_id,
        detected_objects=frame_objects,
        scene_graph=scene_graph,
        timestamp=timestamp,
    )
    memory_objects = session.memory.get_all()
    narration, signature = generate_scene_narration(
        memory_objects[:3],
        relationships=scene_graph.get("relationships"),
        recent_changes=session.memory.recent_changes,
    )
    should_speak = signature != session.last_scene_signature
    session.last_scene_signature = signature
    review_frame_token = f"frame_{session.session_id}_{frame_id}"
    save_cache_frame(review_frame_token, frame_bytes)

    depth_status = (
        "ready" if depth_map is not None
        else "fallback_unknown" if depth_estimator and depth_estimator.loaded
        else "unavailable"
    )
    processing_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    return {
        "session_id": session.session_id,
        "frame_id": frame_id,
        "frame_number": frame_id,
        "timestamp": timestamp,
        "frame_size": {"width": width, "height": height},
        "image_width": width,
        "image_height": height,
        "inference_ms": detector_meta.get("inference_ms", 0.0),
        "depth_inference_ms": depth_inference_ms,
        "processing_ms": processing_ms,
        "device": detector_meta.get("device", detector.device),
        "model": detector_meta.get("model", Path(detector.model_path).name),
        "num_detections": len(frame_objects),
        "detections": frame_objects,
        "objects": frame_objects,
        "tracks": [obj for obj in frame_objects if obj["track_id"] is not None],
        "track_events": session.tracker.last_events,
        "scene_graph": scene_graph,
        "graph": scene_graph,
        "relationships": scene_graph["relationships"],
        "scene_memory": [obj.to_dict() for obj in memory_objects],
        "scene_changes": session.memory.recent_changes,
        "narration": narration,
        "scene_narration": narration,
        "scene_description": narration,
        "should_speak": should_speak,
        "review_frame_token": review_frame_token,
        "depth_preview": encode_depth_preview(depth_map),
        "depth_error": depth_error,
        "depth_model": {
            "loaded": bool(depth_estimator and depth_estimator.loaded),
            "backend": depth_estimator.backend_type if depth_estimator else None,
            "model": depth_estimator.model_name if depth_estimator else None,
            "error": depth_error,
        },
        "stage_status": {
            "camera": "connected",
            "detection": "ready",
            "tracking": "ready",
            "depth": depth_status,
            "spatial": "ready",
            "scene_graph": "ready",
            "memory": "ready",
            "narration": "ready" if narration else "unavailable",
            "review": "active",
        },
        "stage_timings_ms": {
            "detection": detector_meta.get("inference_ms", 0.0),
            "depth": depth_inference_ms,
            "total": processing_ms,
        },
    }


@app.post("/api/live/{session_id}/frame")
async def process_live_frame(
    session_id: str,
    request: Request,
    frame_id: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    session = session_manager.get_live_session(session_id)
    if not session:
        raise NotFoundException("Live session", session_id)

    frame_bytes = None
    if file:
        frame_bytes = await file.read()
    else:
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError) as error:
            raise ValidationException(
                f"Invalid frame request JSON: {error}",
                code="INVALID_FRAME_JSON",
            ) from error
        if not isinstance(body, dict):
            raise ValidationException("Frame request must be a JSON object.", code="INVALID_FRAME_JSON")
        raw_b64 = body.get("data")
        if isinstance(raw_b64, str):
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                frame_bytes = base64.b64decode(raw_b64, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValidationException("Frame data is not valid base64.", code="INVALID_FRAME_DATA") from error
        elif raw_b64 is not None:
            raise ValidationException("Frame data must be a base64 string.", code="INVALID_FRAME_DATA")
        if frame_id is None:
            frame_id = body.get("frame_id")
        if frame_id is not None:
            try:
                frame_id = int(frame_id)
                if frame_id < 1:
                    raise ValueError("must be positive")
            except (TypeError, ValueError) as error:
                raise ValidationException("Frame ID must be a positive integer.", code="INVALID_FRAME_ID") from error

    if not frame_bytes:
        raise ValidationException("Missing frame data.", code="NO_FRAME_DATA")

    image_bgr, w, h = validate_image_bytes(frame_bytes)
    session.touch()
    session.frame_counter += 1
    fid = frame_id if frame_id is not None else session.frame_counter
    data = await analyze_live_frame(session, fid, frame_bytes, image_bgr, w, h)
    return api_success(data)

@app.get("/api/sessions/{session_id}/scene")
async def get_session_scene(session_id: str):
    session = session_manager.get_live_session(session_id)
    if not session:
        raise NotFoundException("Live session", session_id)
    objs = session.memory.get_all()
    narration, _ = generate_scene_narration(
        objs[:3],
        relationships=session.memory.latest_relationships,
        recent_changes=session.memory.recent_changes
    )
    return api_success({
        "session_id": session_id,
        "frame_number": session.memory.current_frame,
        "objects": [o.to_dict() for o in objs],
        "scene_graph": session.memory.latest_scene_graph,
        "relationships": session.memory.latest_relationships,
        "scene_description": narration
    })

@app.post("/api/sessions/{session_id}/ask")
async def ask_session_question(session_id: str, request: Request):
    session = session_manager.get_live_session(session_id)
    if not session:
        raise NotFoundException("Live session", session_id)

    body = await request.json()
    question = body.get("question", body.get("text", "")).strip()
    if not question:
        raise ValidationException("Question text is required.", code="EMPTY_QUESTION")

    init_models()
    qa_engine = getattr(app.state, "qa_engine", models_state.get("qa_engine")) or SceneQAEngine()
    answer = qa_engine.answer_question(question, session.memory)
    return api_success({
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "narration": answer
    })


# -------------------------------------------------------------
# Image Upload Endpoints (POST /api/image and POST /api/upload/image)
# -------------------------------------------------------------
async def process_image_upload_internal(file: UploadFile) -> Dict[str, Any]:
    file_bytes = await file.read()
    image_bgr, w, h = validate_image_bytes(file_bytes, filename=file.filename or "", content_type=file.content_type or "")

    session_id = f"img_{uuid.uuid4().hex[:12]}"
    ensure_session(session_id, input_source="uploaded_image")

    init_models()
    detector = getattr(app.state, "detector", models_state.get("detector"))
    depth_estimator = getattr(app.state, "depth_estimator", models_state.get("depth_estimator"))

    # 1. Detection
    raw_dets = []
    meta = {}
    if detector and detector.loaded:
        raw_dets, meta = await run_in_threadpool(detector.predict_with_meta, image_bgr)

    # 2. Depth Estimation (Depth Anything V2)
    depth_map = None
    if depth_estimator and depth_estimator.loaded:
        depth_map = await run_in_threadpool(depth_estimator.estimate_frame, image_bgr)

    # 3. Object Depth & Spatial Reasoning
    objects = []
    for idx, det in enumerate(raw_dets):
        h_dir, v_dir = compute_direction(det.bbox, w, h)
        combined_pos = compute_combined_position(h_dir, v_dir)
        depth_info = depth_estimator.evaluate_object_depth(depth_map, det.bbox, (h, w)) if depth_estimator else {
            "depth_value": None,
            "relative_depth_value": None,
            "depth_category": "UNKNOWN",
            "depth_class": "UNKNOWN",
            "depth_confidence": "low"
        }

        x1 = round(det.bbox["x1"], 1)
        y1 = round(det.bbox["y1"], 1)
        x2 = round(det.bbox["x2"], 1)
        y2 = round(det.bbox["y2"], 1)
        cx = round(getattr(det, "center_x", (x1 + x2) / 2.0), 1)
        cy = round(getattr(det, "center_y", (y1 + y2) / 2.0), 1)

        sub_id = f"sub_{session_id}_{idx}"
        obj_dict = {
            "id": f"object_{idx+1}",
            "label": det.label,
            "class_name": det.label,
            "model_class_name": det.label,
            "confidence": round(det.confidence, 4),
            "track_id": None, # Explicitly null for still image
            "bbox": det.bbox,
            "bounding_box": [x1, y1, x2, y2],
            "box": [x1, y1, x2, y2],
            "center_x": cx,
            "center_y": cy,
            "position": {
                "horizontal": h_dir,
                "vertical": v_dir,
                "combined": combined_pos
            },
            "depth": {
                "category": depth_info["depth_category"],
                "value": depth_info["depth_value"]
            },
            "direction_h": h_dir,
            "direction_v": v_dir,
            "horizontal_position": h_dir,
            "vertical_position": v_dir,
            "combined_position": combined_pos,
            "depth_category": depth_info["depth_category"],
            "depth_class": depth_info["depth_category"],
            "depth_value": depth_info["depth_value"],
            "relative_depth_value": depth_info["depth_value"],
            "submission_id": sub_id
        }
        objects.append(obj_dict)

    # 4. Draw annotated image
    annotated = image_bgr.copy()
    for obj in objects:
        b = obj["bbox"]
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 105, 180), 2)
        label_str = f"{obj['class_name']} ({obj['depth_category']})"
        cv2.putText(annotated, label_str, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    _, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    b64_annotated = base64.b64encode(enc.tobytes()).decode("utf-8")

    # 5. Scene graph & Narration
    scene_graph = build_scene_graph(objects, w, h)
    narration_text, _ = generate_scene_narration(objects[:3], relationships=scene_graph.get("relationships"))

    # Cache frame for review
    cache_token = f"frame_{session_id}"
    save_cache_frame(cache_token, file_bytes)

    return {
        "media_type": "image",
        "session_id": session_id,
        "image_width": w,
        "image_height": h,
        "inference_ms": meta.get("inference_ms", 0.0),
        "device": meta.get("device", detector.device if detector else "cpu"),
        "model": meta.get("model", Path(detector.model_path).name if detector else ""),
        "num_detections": len(objects),
        "scene_description": narration_text,
        "objects": objects,
        "scene_graph": scene_graph,
        "graph": scene_graph,
        "relationships": scene_graph["relationships"],
        "annotated_image": b64_annotated,
        "frames_processed": 1,
        "detection_count": len(objects),
        "review_frame_token": cache_token,
        "width": w,
        "height": h
    }

@app.post("/api/image")
async def upload_image_api(upload: Optional[UploadFile] = File(None), file: Optional[UploadFile] = File(None)):
    target = upload or file
    if not target:
        raise ValidationException("No image file uploaded.", code="NO_FILE")
    result = await process_image_upload_internal(target)
    return api_success(result)

@app.post("/api/upload/image")
async def upload_image_for_ui(upload: Optional[UploadFile] = File(None), file: Optional[UploadFile] = File(None)):
    target = upload or file
    if not target:
        return JSONResponse(status_code=400, content={"error": "No image file uploaded."})
    result = await process_image_upload_internal(target)
    return JSONResponse(status_code=200, content=result)


# -------------------------------------------------------------
# Video Upload Endpoints (POST /api/video and POST /api/upload/video)
# -------------------------------------------------------------
@app.post("/api/video", status_code=202)
async def upload_video_async(upload: Optional[UploadFile] = File(None), file: Optional[UploadFile] = File(None)):
    target = upload or file
    if not target:
        raise ValidationException("No video file uploaded.", code="NO_FILE")

    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = config.UPLOADS_DIR / f"vid_{uuid.uuid4().hex[:12]}_{target.filename}"
    content = await target.read()
    temp_path.write_bytes(content)

    validate_video_file(temp_path, filename=target.filename or "", content_type=target.content_type or "")

    job = video_jobs.create_job(temp_path)
    ensure_session(job.session_id, input_source="uploaded_video")

    init_models()
    detector = getattr(app.state, "detector", models_state.get("detector"))
    depth_estimator = getattr(app.state, "depth_estimator", models_state.get("depth_estimator"))
    video_jobs.start_job(job.job_id, detector, depth_estimator)

    return api_success({
        "job_id": job.job_id,
        "session_id": job.session_id,
        "status": "processing",
        "message": "Video accepted for asynchronous analysis."
    }, status_code=202)

@app.get("/api/video/{job_id}/status")
async def get_video_job_status(job_id: str):
    job = video_jobs.jobs.get(job_id)
    if not job:
        raise NotFoundException("Video job", job_id)
    return api_success({
        "job_id": job.job_id,
        "session_id": job.session_id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error_message
    })

@app.get("/api/video/{job_id}/result")
async def get_video_job_result(job_id: str):
    job = video_jobs.jobs.get(job_id)
    if not job:
        raise NotFoundException("Video job", job_id)
    if job.status == "processing" or job.status == "pending":
        return api_success({
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress
        }, status_code=200)
    if job.status == "failed":
        raise ByakuException(code="VIDEO_PROCESSING_FAILED", message=job.error_message or "Video processing failed.", status_code=500)
    return api_success(job.result)

@app.post("/api/upload/video")
async def upload_video_ui(upload: Optional[UploadFile] = File(None), file: Optional[UploadFile] = File(None)):
    target = upload or file
    if not target:
        return JSONResponse(status_code=400, content={"error": "No video file uploaded."})

    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = config.UPLOADS_DIR / f"vid_{uuid.uuid4().hex[:12]}_{target.filename}"
    content = await target.read()
    temp_path.write_bytes(content)

    try:
        validate_video_file(temp_path, filename=target.filename or "", content_type=target.content_type or "")
        job = video_jobs.create_job(temp_path)
        ensure_session(job.session_id, input_source="uploaded_video")
        init_models()
        detector = getattr(app.state, "detector", models_state.get("detector"))
        depth_estimator = getattr(app.state, "depth_estimator", models_state.get("depth_estimator"))

        video_jobs._process_video(job, detector, depth_estimator)

        if job.status == "failed":
            return JSONResponse(status_code=500, content={"error": job.error_message or "Video processing failed."})

        return JSONResponse(status_code=200, content=job.result)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return JSONResponse(status_code=400, content={"error": str(e)})


# -------------------------------------------------------------
# HITL Review & Feedback Endpoints
# -------------------------------------------------------------
@app.post("/api/reviews")
async def create_review_api(request: Request):
    body = await request.json()
    res = submit_review(
        session_id=body["session_id"],
        submission_id=body.get("submission_id", str(uuid.uuid4())),
        original_label=body["original_label"],
        original_confidence=float(body.get("original_confidence", 1.0)),
        bbox=body["bbox"],
        action=body["action"],
        input_source=body.get("input_source", "live_camera"),
        corrected_label=body.get("corrected_label"),
        track_id=body.get("track_id"),
        frame_id=str(body["frame_id"]) if body.get("frame_id") is not None else None,
        frame_cache_token=body.get("review_frame_token") or body.get("frame_cache_token"),
        frame_number=body.get("frame_number"),
        video_timestamp_ms=body.get("video_timestamp_ms"),
        frame_dimensions=body.get("frame_dimensions"),
        feedback_type=body.get("feedback_type", "detection_review"),
        corrected_bbox=body.get("corrected_bbox")
    )
    return api_success(res, status_code=201)

@app.get("/api/reviews")
async def list_reviews_api(
    session_id: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(100),
    offset: int = Query(0)
):
    reviews = query_reviews(
        session_id=session_id,
        source=source,
        action=action,
        status=status,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset
    )
    return api_success(reviews)

@app.get("/api/sessions/{session_id}/reviews")
async def list_session_reviews(session_id: str):
    reviews = query_reviews(session_id=session_id)
    return api_success(reviews)

@app.post("/api/reviews/{review_id}/undo")
async def undo_review_api(review_id: str):
    res = undo_review(review_id)
    return api_success(res)

@app.get("/api/reviews/status")
async def get_reviews_status_api():
    stats = get_hitl_status()
    return api_success(stats)

@app.get("/api/reviews/export")
async def export_reviews_dataset():
    data = export_reviews_coco()
    return api_success(data)

@app.get("/api/audit")
async def list_audit_events(
    session_id: Optional[str] = Query(None),
    review_id: Optional[str] = Query(None),
    limit: int = Query(100),
    offset: int = Query(0)
):
    events = query_audit_events(session_id=session_id, review_id=review_id, limit=limit, offset=offset)
    return api_success(events)

@app.get("/api/review-frames/{frame_id}")
async def get_cached_review_frame(frame_id: str):
    data_bytes, mime = get_cache_frame(frame_id)
    return Response(content=data_bytes, media_type=mime)


# -------------------------------------------------------------
# Aliases for UI Compatibility (/api/feedback)
# -------------------------------------------------------------
@app.post("/api/feedback")
async def handle_feedback_ui(request: Request):
    """Bridge for UI's /api/feedback format."""
    body = await request.json()
    orig = body.get("original_prediction") or {}
    orig_label = orig.get("label") or orig.get("class_name") or body.get("original_label")
    if not isinstance(orig_label, str) or not orig_label.strip():
        raise ValidationException("Original object label is required.", code="ORIGINAL_LABEL_REQUIRED")
    orig_conf = orig.get("confidence", body.get("original_confidence", 1.0))
    bbox = orig.get("bbox")
    if not bbox and "box" in orig:
        b = orig["box"]
        bbox = {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
    elif not bbox and "bounding_box" in orig:
        b = orig["bounding_box"]
        bbox = {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
    if not bbox:
        raise ValidationException("Original detection bounding box is required.", code="BBOX_REQUIRED")

    action = body.get("review_status") or body.get("action")
    action = {"accepted": "confirmed", "correct": "confirmed", "incorrect": "rejected", "wrong": "rejected"}.get(action, action)
    if action not in {"confirmed", "corrected", "rejected"}:
        raise ValidationException("Review status must be confirmed, corrected, or rejected.", code="INVALID_ACTION")

    corrected = body.get("human_corrected_name") or body.get("corrected_value") or body.get("corrected_label")
    correction_type = body.get("correction_type", "detection_review")
    if correction_type == "class_correction" and action == "corrected" and not corrected:
        raise ValidationException("Corrected class label is required.", code="CORRECTED_LABEL_REQUIRED")
    corrected_bbox = body.get("corrected_bbox")

    sid = body.get("session_id") or f"sess_{uuid.uuid4().hex[:8]}"
    sub_id = body.get("submission_id") or f"sub_{uuid.uuid4().hex[:8]}"

    res = submit_review(
        session_id=sid,
        submission_id=sub_id,
        original_label=orig_label.strip(),
        original_confidence=float(orig_conf),
        bbox=bbox,
        action=action,
        input_source=body.get("input_source", "live_camera"),
        corrected_label=corrected if action == "corrected" else None,
        track_id=body.get("track_id") if body.get("track_id") is not None else orig.get("track_id"),
        frame_id=str(body["frame_id"]) if body.get("frame_id") is not None else None,
        frame_cache_token=body.get("review_frame_token") or body.get("frame_cache_token"),
        frame_number=body.get("frame_number"),
        frame_dimensions=body.get("image_dimensions"),
        feedback_type=correction_type,
        corrected_bbox=corrected_bbox
    )
    return JSONResponse(status_code=200, content={"status": "ok", "feedback_id": res["review_id"], **res})

@app.get("/api/feedback")
async def list_feedback_ui(limit: int = Query(100), session_id: Optional[str] = Query(None)):
    reviews = query_reviews(limit=limit, session_id=session_id)
    formatted = []
    for r in reviews:
        formatted.append({
            "feedback_id": r["review_id"],
            "session_id": r["session_id"],
            "track_id": r["track_id"],
            "frame_id": r["frame_id"],
            "frame_cache_token": r["frame_cache_token"],
            "frame_number": r["frame_number"],
            "original_prediction": {
                "label": r["original_label"],
                "class_name": r["original_label"],
                "confidence": r["original_confidence"],
                "track_id": r["track_id"],
                "bbox": r["bbox"],
                "bounding_box": [r["bbox"]["x1"], r["bbox"]["y1"], r["bbox"]["x2"], r["bbox"]["y2"]]
            },
            "corrected_value": r["corrected_label"],
            "corrected_bbox": r["corrected_bbox"],
            "feedback_type": r["feedback_type"],
            "human_corrected_name": r["corrected_label"],
            "review_status": "undone" if r["status"] == "undone" else r["action"],
            "source": r["input_source"],
            "timestamp": r["created_at"]
        })
    return JSONResponse(status_code=200, content={"feedback": formatted})

@app.post("/api/feedback/{feedback_id}/undo")
async def undo_feedback_ui(feedback_id: str):
    res = undo_review(feedback_id)
    return JSONResponse(status_code=200, content=res)


# -------------------------------------------------------------
# WebSocket for Live Camera & Voice Q&A (/ws)
# -------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = session_manager.start_live_session()
    init_models()
    detector = getattr(app.state, "detector", models_state.get("detector"))
    depth_estimator = getattr(app.state, "depth_estimator", models_state.get("depth_estimator"))
    qa_engine = getattr(app.state, "qa_engine", models_state.get("qa_engine")) or SceneQAEngine()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "ts": data.get("ts")})

            elif msg_type == "start_camera":
                session.touch()
                await websocket.send_json({
                    "type": "camera_started",
                    "session_id": session.session_id,
                    "stage_status": {"camera": "connected"}
                })

            elif msg_type == "stop_camera":
                session_manager.stop_live_session(session.session_id)
                await websocket.send_json({
                    "type": "camera_stopped",
                    "session_id": session.session_id,
                    "stage_status": {"camera": "disconnected"}
                })

            elif msg_type == "frame":
                session.touch()
                frame_id = data.get("frame_id", 0)
                raw_b64 = data.get("data", "")
                stage = "camera"
                try:
                    if not isinstance(raw_b64, str):
                        raise ValidationException(
                            "Frame data must be a base64 string.",
                            code="INVALID_FRAME_DATA",
                        )
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    try:
                        frame_bytes = base64.b64decode(raw_b64, validate=True)
                    except (binascii.Error, ValueError) as error:
                        raise ValidationException(
                            "Frame data is not valid base64.",
                            code="INVALID_FRAME_DATA",
                        ) from error
                    image_bgr, w, h = validate_image_bytes(frame_bytes)
                    session.frame_counter += 1
                    fid = frame_id if isinstance(frame_id, int) and frame_id > 0 else session.frame_counter
                    result = await analyze_live_frame(session, fid, frame_bytes, image_bgr, w, h)
                    await websocket.send_json({"type": "frame_result", **result})
                except Exception as err:
                    logger.exception(
                        "Live frame %s failed for session %s.",
                        frame_id,
                        session.session_id,
                    )
                    await websocket.send_json({
                        "type": "error",
                        "stage": getattr(err, "stage", stage),
                        "frame_id": frame_id,
                        "code": getattr(err, "code", "LIVE_FRAME_FAILED"),
                        "message": str(err),
                        "recoverable": True
                    })

            elif msg_type == "query":
                session.touch()
                q_text = data.get("text", "").strip()
                ans = qa_engine.answer_question(q_text, session.memory)
                await websocket.send_json({
                    "type": "query_result",
                    "session_id": session.session_id,
                    "text": q_text,
                    "answer": ans,
                    "narration": ans,
                    "scene_memory": [obj.to_dict() for obj in session.memory.get_all()],
                    "frame_number": session.memory.current_frame,
                })

    except WebSocketDisconnect:
        session_manager.stop_live_session(session.session_id)
    except Exception:
        logger.exception("WebSocket session failed: %s.", session.session_id)
        session_manager.stop_live_session(session.session_id)
