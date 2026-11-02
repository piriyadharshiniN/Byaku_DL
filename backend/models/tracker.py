from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two sets of boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a_x1, a_y1, a_x2, a_y2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    b_x1, b_y1, b_x2, b_y2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, a_x2 - a_x1) * np.maximum(0.0, a_y2 - a_y1)
    area_b = np.maximum(0.0, b_x2 - b_x1) * np.maximum(0.0, b_y2 - b_y1)
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)

class STrack:
    def __init__(self, bbox: List[float], score: float, class_id: int, label: str, track_id: int):
        self.bbox = np.array(bbox, dtype=np.float32) # [x1, y1, x2, y2]
        self.last_observed_bbox = self.bbox.copy()
        self.score = float(score)
        self.class_id = int(class_id)
        self.label = label
        self.track_id = int(track_id)
        self.state = "tracked" # "tracked", "lost", "removed"
        self.time_since_update = 0
        self.hit_streak = 1
        self.age = 1
        # Motion model (velocity of centre coordinates)
        self.vx = 0.0
        self.vy = 0.0

    def predict(self):
        """Predicts position in next frame using constant-velocity motion model."""
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update <= 3:
            # Shift predicted bounding box with current velocity
            self.bbox[0] += self.vx
            self.bbox[1] += self.vy
            self.bbox[2] += self.vx
            self.bbox[3] += self.vy
            # Apply momentum damping
            self.vx *= 0.85
            self.vy *= 0.85

    def update(self, new_bbox: List[float], new_score: float):
        """Updates track with verified matched detection and updates velocity."""
        old_cx = (self.last_observed_bbox[0] + self.last_observed_bbox[2]) / 2.0
        old_cy = (self.last_observed_bbox[1] + self.last_observed_bbox[3]) / 2.0
        new_cx = (new_bbox[0] + new_bbox[2]) / 2.0
        new_cy = (new_bbox[1] + new_bbox[3]) / 2.0

        frame_gap = max(1, self.time_since_update)
        inst_vx = (new_cx - old_cx) / frame_gap
        inst_vy = (new_cy - old_cy) / frame_gap

        if self.hit_streak > 1:
            self.vx = 0.6 * inst_vx + 0.4 * self.vx
            self.vy = 0.6 * inst_vy + 0.4 * self.vy
        else:
            self.vx = inst_vx
            self.vy = inst_vy

        self.bbox = np.array(new_bbox, dtype=np.float32)
        self.last_observed_bbox = self.bbox.copy()
        self.score = float(new_score)
        self.state = "tracked"
        self.time_since_update = 0
        self.hit_streak += 1

    def mark_lost(self):
        self.state = "lost"

    def mark_removed(self):
        self.state = "removed"

class ByteTrackSessionTracker:
    """
    Independent ByteTrack instance per session.
    Keeps persistent IDs across frames, tolerates brief disappearance up to track_buffer frames,
    enforces class consistency so unrelated objects never share IDs, and isolates state per session.
    """
    def __init__(
        self,
        track_buffer: int = 30,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.7
    ):
        self.track_buffer = track_buffer
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.next_id = 1
        self.frame_count = 0
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.last_events: List[Dict[str, Any]] = []

    def reset(self):
        """Resets tracker state on session end or new video start."""
        self.next_id = 1
        self.frame_count = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.last_events = []

    def update(self, detections: List[Any]) -> List[Dict[str, Any]]:
        """
        Updates tracks with detections for the current frame.
        detections: list of ObjectDetection or dicts with bbox, confidence, class_id, label.
        Returns list of matched active tracks with persistent track_id.
        """
        self.frame_count += 1
        self.last_events = []

        # Predict positions
        for track in self.tracked_stracks + self.lost_stracks:
            track.predict()

        if not detections:
            # Mark all tracked as lost
            for track in self.tracked_stracks:
                track.mark_lost()
                self.lost_stracks.append(track)
                self.last_events.append({"track_id": track.track_id, "state": "lost", "label": track.label})
            self.tracked_stracks = []

            # Evict expired lost tracks
            self.lost_stracks = [t for t in self.lost_stracks if t.time_since_update <= self.track_buffer]
            return []

        # Extract detection boxes, scores, and attributes
        det_boxes = []
        det_scores = []
        det_classes = []
        det_labels = []
        orig_objects = []

        for d in detections:
            if hasattr(d, "bbox") and isinstance(d.bbox, dict):
                b = d.bbox
                det_boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                det_scores.append(float(d.confidence))
                det_classes.append(int(d.class_id))
                det_labels.append(str(d.label))
            elif isinstance(d, dict) and "bbox" in d:
                b = d["bbox"]
                if isinstance(b, dict):
                    det_boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                else:
                    det_boxes.append(list(b))
                det_scores.append(float(d.get("confidence", 1.0)))
                det_classes.append(int(d.get("class_id", 0)))
                det_labels.append(str(d.get("label", "object")))
            elif isinstance(d, dict) and "bounding_box" in d:
                det_boxes.append(list(d["bounding_box"]))
                det_scores.append(float(d.get("confidence", 1.0)))
                det_classes.append(int(d.get("class_id", 0)))
                det_labels.append(str(d.get("label", "object")))
            orig_objects.append(d)

        det_boxes = np.array(det_boxes, dtype=np.float32)
        det_scores = np.array(det_scores, dtype=np.float32)

        # Split into high and low score detections
        high_idx = np.where(det_scores >= self.high_thresh)[0]
        low_idx = np.where((det_scores >= self.low_thresh) & (det_scores < self.high_thresh))[0]

        # Stage 1: Match high score detections with tracked + lost tracks
        candidate_tracks = self.tracked_stracks + self.lost_stracks
        matched_tracks: List[STrack] = []
        unmatched_tracks: List[STrack] = []
        unmatched_high_dets = list(high_idx)

        track_assignment_map = {} # det_idx -> track_id
        track_states: Dict[int, str] = {}

        if len(candidate_tracks) > 0 and len(high_idx) > 0:
            cand_boxes = np.array([t.bbox for t in candidate_tracks], dtype=np.float32)
            high_boxes = det_boxes[high_idx]
            ious = box_iou_batch(cand_boxes, high_boxes)
            cost_matrix = 1.0 - ious

            # Class compatibility constraint: prevent mismatched classes from ever matching
            for r in range(len(candidate_tracks)):
                for c in range(len(high_idx)):
                    det_i = high_idx[c]
                    if candidate_tracks[r].class_id != det_classes[det_i]:
                        cost_matrix[r, c] = 1000.0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_cand_indices = set()
            for r, c in zip(row_ind, col_ind):
                # Valid match only if cost <= match_thresh (i.e. IoU >= 1.0 - match_thresh) and class matches
                det_i = high_idx[c]
                if cost_matrix[r, c] <= self.match_thresh and candidate_tracks[r].class_id == det_classes[det_i]:
                    track = candidate_tracks[r]
                    was_lost = track.state == "lost"
                    track.update(det_boxes[det_i].tolist(), det_scores[det_i])
                    matched_tracks.append(track)
                    track_states[track.track_id] = (
                        "reacquired" if was_lost else ("continuing" if track.hit_streak > 1 else "new")
                    )
                    matched_cand_indices.add(r)
                    if det_i in unmatched_high_dets:
                        unmatched_high_dets.remove(det_i)
                    track_assignment_map[det_i] = track.track_id

            unmatched_tracks = [candidate_tracks[i] for i in range(len(candidate_tracks)) if i not in matched_cand_indices]
        else:
            unmatched_tracks = list(candidate_tracks)

        # Stage 2: Match low score detections with remaining tracks
        if len(unmatched_tracks) > 0 and len(low_idx) > 0:
            cand_boxes = np.array([t.bbox for t in unmatched_tracks], dtype=np.float32)
            low_boxes = det_boxes[low_idx]
            ious = box_iou_batch(cand_boxes, low_boxes)
            cost_matrix = 1.0 - ious

            for r in range(len(unmatched_tracks)):
                for c in range(len(low_idx)):
                    det_i = low_idx[c]
                    if unmatched_tracks[r].class_id != det_classes[det_i]:
                        cost_matrix[r, c] = 1000.0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_second = set()
            for r, c in zip(row_ind, col_ind):
                det_i = low_idx[c]
                if cost_matrix[r, c] <= 0.5 and unmatched_tracks[r].class_id == det_classes[det_i]:
                    track = unmatched_tracks[r]
                    was_lost = track.state == "lost"
                    track.update(det_boxes[det_i].tolist(), det_scores[det_i])
                    matched_tracks.append(track)
                    track_states[track.track_id] = (
                        "reacquired" if was_lost else ("continuing" if track.hit_streak > 1 else "new")
                    )
                    matched_second.add(r)
                    track_assignment_map[det_i] = track.track_id

            unmatched_tracks = [unmatched_tracks[i] for i in range(len(unmatched_tracks)) if i not in matched_second]

        # Stage 3: Initiate new tracks for unmatched high confidence detections
        for det_i in unmatched_high_dets:
            new_id = self.next_id
            self.next_id += 1
            new_track = STrack(
                bbox=det_boxes[det_i].tolist(),
                score=det_scores[det_i],
                class_id=det_classes[det_i],
                label=det_labels[det_i],
                track_id=new_id
            )
            matched_tracks.append(new_track)
            track_states[new_track.track_id] = "new"
            track_assignment_map[det_i] = new_id

        # Update tracked and lost lists
        new_lost: List[STrack] = []
        for track in unmatched_tracks:
            track.mark_lost()
            if track.time_since_update <= self.track_buffer:
                new_lost.append(track)
                self.last_events.append({"track_id": track.track_id, "state": "lost", "label": track.label})

        self.tracked_stracks = matched_tracks
        self.lost_stracks = new_lost

        # Build output structure aligned with input detections
        results = []
        for idx, det in enumerate(orig_objects):
            tid = track_assignment_map.get(idx, None)
            res = {
                "detection": det,
                "track_id": tid,
                "state": track_states.get(tid, "untracked"),
            }
            track = next((candidate for candidate in matched_tracks if candidate.track_id == tid), None)
            if track is not None and track.hit_streak >= 2:
                res["velocity_px_per_frame"] = {
                    "x": round(float(track.vx), 2),
                    "y": round(float(track.vy), 2),
                }
                res["movement_direction"] = (
                    "right" if track.vx > 1 else "left" if track.vx < -1
                    else "down" if track.vy > 1 else "up" if track.vy < -1
                    else "stationary"
                )
            else:
                res["velocity_px_per_frame"] = None
                res["movement_direction"] = "unknown"
            results.append(res)

        for track in matched_tracks:
            self.last_events.append({
                "track_id": track.track_id,
                "state": track_states[track.track_id],
                "label": track.label,
            })
        return results
