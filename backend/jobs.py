import uuid
import time
import logging
import cv2
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from backend.config import config
from backend.models.tracker import ByteTrackSessionTracker
from backend.spatial.reasoning import (
    compute_direction, compute_combined_position,
    build_scene_graph, extract_relationships
)
from backend.voice.narration import generate_scene_narration

logger = logging.getLogger(__name__)


class VideoJob:
    def __init__(self, job_id: str, session_id: str, file_path: Path):
        self.job_id = job_id
        self.session_id = session_id
        self.file_path = file_path
        self.status = "pending" # 'pending', 'processing', 'completed', 'failed'
        self.progress = 0.0
        self.created_at = time.time()
        self.completed_at = 0.0
        self.error_message: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None

class VideoJobManager:
    def __init__(self):
        self.jobs: Dict[str, VideoJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    def create_job(self, file_path: Path, session_id: Optional[str] = None) -> VideoJob:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        sid = session_id or f"vid_{uuid.uuid4().hex[:12]}"
        job = VideoJob(job_id=job_id, session_id=sid, file_path=file_path)
        self.jobs[job_id] = job
        return job

    def start_job(self, job_id: str, detector: Any, depth_estimator: Any):
        job = self.jobs.get(job_id)
        if not job:
            return
        self.executor.submit(self._process_video, job, detector, depth_estimator)

    def _process_video(self, job: VideoJob, detector: Any, depth_estimator: Any):
        job.status = "processing"
        job.progress = 0.0
        cap = cv2.VideoCapture(str(job.file_path))

        try:
            if not cap.isOpened():
                raise RuntimeError("Could not open video file.")

            if not detector or not detector.loaded:
                raise RuntimeError(
                    (detector.load_error if detector else None)
                    or "YOLO detector is unavailable; video analysis cannot continue."
                )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            duration_sec = total_frames / max(1.0, fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Sampling step
            step_frames = max(1, int(round(config.VIDEO_SAMPLE_INTERVAL_SECONDS * fps)))
            # Frame indices to sample
            target_indices = list(range(0, total_frames, step_frames))[:config.VIDEO_MAX_FRAMES]
            if not target_indices:
                target_indices = [0]

            tracker = ByteTrackSessionTracker() # Isolated tracker per video session!

            samples = []
            all_detected_objects = []
            warnings = []
            frames_failed = 0

            for sample_idx, f_idx in enumerate(target_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    frames_failed += 1
                    warning = f"Source frame {f_idx} failed to decode."
                    warnings.append(warning)
                    samples.append({
                        "frame_number": f_idx,
                        "timestamp_seconds": round(f_idx / max(1.0, fps), 2),
                        "status": "decode_error",
                        "error": warning,
                        "objects": [],
                        "annotated_image": None,
                    })
                    job.progress = round((sample_idx + 1) / len(target_indices), 2)
                    continue

                ts_sec = round(f_idx / max(1.0, fps), 2)
                h, w = frame_bgr.shape[:2]

                try:
                    raw_dets = detector.predict(frame_bgr)
                except Exception as error:
                    raise RuntimeError(
                        f"Object detection failed on source frame {f_idx}: {error}"
                    ) from error

                # Update session tracker
                tracked = tracker.update(raw_dets)

                # Depth estimation
                depth_map = depth_estimator.estimate_frame(frame_bgr) if depth_estimator else None
                depth_error = (
                    depth_estimator.last_inference_error
                    if depth_estimator and depth_estimator.loaded
                    else (depth_estimator.load_error if depth_estimator else "Depth model is unavailable.")
                )

                frame_objs = []
                for item in tracked:
                    det = item["detection"]
                    tid = item["track_id"]
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

                    obj_dict = {
                        "id": str(tid),
                        "track_id": tid,
                        "track_status": item["state"],
                        "movement_direction": item["movement_direction"],
                        "velocity_px_per_frame": item["velocity_px_per_frame"],
                        "class_name": det.label,
                        "model_class_name": det.label,
                        "label": det.label,
                        "confidence": det.confidence,
                        "bbox": det.bbox,
                        "bounding_box": [x1, y1, x2, y2],
                        "box": [x1, y1, x2, y2],
                        "center_x": round(getattr(det, "center_x", (x1 + x2)/2.0), 1),
                        "center_y": round(getattr(det, "center_y", (y1 + y2)/2.0), 1),
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
                        "frame_number": f_idx,
                        "submission_id": f"sub_{job.session_id}_{f_idx}_{tid}"
                    }
                    frame_objs.append(obj_dict)
                    all_detected_objects.append(obj_dict)

                # Draw annotated frame
                annotated_bgr = frame_bgr.copy()
                for obj in frame_objs:
                    b = obj["bbox"]
                    x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
                    cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (255, 105, 180), 2)
                    label_str = f"#{obj['track_id']} {obj['class_name']} {obj['depth_category']}"
                    cv2.putText(annotated_bgr, label_str, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                _, enc = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                b64_frame = base64.b64encode(enc.tobytes()).decode("utf-8")

                sample_scene_graph = build_scene_graph(frame_objs, w, h)

                samples.append({
                    "frame_number": f_idx,
                    "timestamp_seconds": ts_sec,
                    "status": "success",
                    "annotated_size": {"width": w, "height": h},
                    "objects": frame_objs,
                    "scene_graph": sample_scene_graph,
                    "relationships": sample_scene_graph["relationships"],
                    "annotated_image": b64_frame,
                    "track_events": tracker.last_events,
                    "depth_status": "ready" if depth_map is not None else "fallback_unknown",
                    "depth_error": depth_error,
                })

                job.progress = round((sample_idx + 1) / len(target_indices), 2)

            frames_processed = sum(sample["status"] == "success" for sample in samples)
            if frames_processed == 0:
                raise RuntimeError("No sampled video frames could be decoded for analysis.")

            cap.release()

            # Global scene narration
            narration_text, _ = generate_scene_narration(all_detected_objects[:5])
            summary_statement = f"Analyzed {frames_processed} of {total_frames} frames (sampling every {config.VIDEO_SAMPLE_INTERVAL_SECONDS:.1f} s, capped at {config.VIDEO_MAX_FRAMES} frames)."

            overall_scene_graph = build_scene_graph(all_detected_objects[:10], width, height)

            job.result = {
                "media_type": "video",
                "session_id": job.session_id,
                "summary": summary_statement,
                "frames_processed": frames_processed,
                "frames_total": total_frames,
                "frames_failed": frames_failed,
                "warnings": warnings,
                "fps": round(fps, 1),
                "duration_seconds": round(duration_sec, 2),
                "scene_description": narration_text,
                "detection_count": len(all_detected_objects),
                "scene_graph": overall_scene_graph,
                "relationships": overall_scene_graph["relationships"],
                "samples": samples
            }
            job.status = "partial" if frames_failed else "completed"
            job.progress = 1.0
            job.completed_at = time.time()

        except Exception as e:
            logger.exception("Video job %s failed while processing.", job.job_id)
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = time.time()
        finally:
            if cap.isOpened():
                cap.release()
            # Clean up uploaded video temp file
            try:
                if job.file_path.exists():
                    job.file_path.unlink()
            except OSError:
                logger.exception("Could not remove temporary video file for job %s.", job.job_id)

video_jobs = VideoJobManager()
