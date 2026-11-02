import time
import hashlib
from typing import Dict, Any, List, Optional
from backend.config import config

class MemoryObject:
    def __init__(
        self,
        key: str,
        label: str,
        bbox: Any,
        direction_h: str,
        direction_v: str,
        depth_class: str,
        depth_value: Optional[float],
        confidence: float,
        last_seen_frame: int,
        track_id: Optional[int] = None,
        combined_position: Optional[str] = None,
        last_seen_time: Optional[float] = None
    ):
        self.key = key
        self.label = label
        self.class_name = label
        self.track_id = track_id
        self.id = str(track_id if track_id is not None else key)

        # Ensure bbox is dict and list
        if isinstance(bbox, dict):
            self.bbox = {
                "x1": round(float(bbox.get("x1", 0.0)), 1),
                "y1": round(float(bbox.get("y1", 0.0)), 1),
                "x2": round(float(bbox.get("x2", 0.0)), 1),
                "y2": round(float(bbox.get("y2", 0.0)), 1)
            }
            self.bounding_box = [self.bbox["x1"], self.bbox["y1"], self.bbox["x2"], self.bbox["y2"]]
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            self.bounding_box = [round(float(bbox[0]), 1), round(float(bbox[1]), 1), round(float(bbox[2]), 1), round(float(bbox[3]), 1)]
            self.bbox = {
                "x1": self.bounding_box[0],
                "y1": self.bounding_box[1],
                "x2": self.bounding_box[2],
                "y2": self.bounding_box[3]
            }
        else:
            self.bbox = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}
            self.bounding_box = [0.0, 0.0, 0.0, 0.0]

        self.direction_h = direction_h
        self.direction_v = direction_v
        self.horizontal_position = direction_h
        self.vertical_position = direction_v
        self.combined_position = combined_position or f"{direction_v}-{direction_h}"

        self.depth_class = depth_class
        self.depth_category = depth_class
        self.depth_value = round(float(depth_value), 3) if depth_value is not None else None
        self.relative_depth_value = self.depth_value

        self.confidence = round(float(confidence), 4)
        self.last_seen_frame = int(last_seen_frame)
        self.last_seen_time = float(last_seen_time if last_seen_time is not None else time.time())
        self.missed_frames = 0
        self.status = "current" # 'current' (visible in latest frame) or 'recent' (temporarily missing)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "track_id": self.track_id,
            "label": self.label,
            "class_name": self.label,
            "model_class_name": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "bounding_box": self.bounding_box,
            "box": self.bounding_box,
            "position": {
                "horizontal": self.direction_h,
                "vertical": self.direction_v,
                "combined": self.combined_position
            },
            "depth": {
                "category": self.depth_category,
                "value": self.depth_value
            },
            "direction_h": self.direction_h,
            "direction_v": self.direction_v,
            "horizontal_position": self.direction_h,
            "vertical_position": self.direction_v,
            "combined_position": self.combined_position,
            "depth_class": self.depth_category,
            "depth_category": self.depth_category,
            "depth_value": self.depth_value,
            "relative_depth_value": self.depth_value,
            "last_seen_frame": self.last_seen_frame,
            "last_seen_time": self.last_seen_time,
            "missed_frames": self.missed_frames,
            "status": self.status
        }

class SceneMemory:
    """
    Intelligent per-session temporal scene memory.
    - Manages object lifecycles: visible ('current') vs temporarily missing ('recent').
    - Prevents duplicates and maintains bounded memory size.
    - Tracks recent scene changes (new object, moved object, disappeared object).
    - Stores the latest structured scene graph and relationships.
    - Evicts stale objects after SCENE_RETENTION_FRAMES (default 5).
    """
    def __init__(self, session_id: str, retention_frames: Optional[int] = None, max_objects: int = 50):
        self.session_id = session_id
        self.retention_frames = retention_frames if retention_frames is not None else config.SCENE_RETENTION_FRAMES
        self.max_objects = max_objects
        self.current_frame = 0
        self._store: Dict[str, MemoryObject] = {}
        self.latest_scene_graph: Dict[str, Any] = {"objects": [], "relationships": [], "nodes": [], "edges": []}
        self.latest_relationships: List[Dict[str, Any]] = []
        self.recent_changes: List[Dict[str, Any]] = []
        self.last_updated_time: float = 0.0

    def _generate_key(self, obj: Dict[str, Any]) -> str:
        tid = obj.get("track_id")
        if tid is not None:
            return f"track_{tid}"
        # For untracked still images or unassigned detections, hash label + spatial region
        b = obj.get("bbox") or obj.get("bounding_box") or [0, 0, 0, 0]
        if isinstance(b, dict):
            x1, y1, x2, y2 = b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0)
        else:
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        label = obj.get("label") or obj.get("class_name") or "unknown"
        bx = int(round(float(x1) / 30.0))
        by = int(round(float(y1) / 30.0))
        bw = int(round(float(x2 - x1) / 30.0))
        bh = int(round(float(y2 - y1) / 30.0))
        raw = f"{label}_{bx}_{by}_{bw}_{bh}"
        return f"pos_{hashlib.md5(raw.encode()).hexdigest()[:10]}"

    def update(
        self,
        frame_number: int,
        detected_objects: List[Dict[str, Any]],
        scene_graph: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ):
        """
        Updates memory with current frame observations.
        - Existing objects are updated in-place with latest bounding box, depth, and position.
        - Missing objects transition to status='recent' and increment missed_frames.
        - Objects missing > retention_frames are evicted.
        - Records scene changes (new, moved, disappeared).
        """
        self.current_frame = frame_number
        now = float(timestamp if timestamp is not None else time.time())
        self.last_updated_time = now
        seen_keys = set()
        current_frame_changes: List[Dict[str, Any]] = []

        for obj in detected_objects:
            key = self._generate_key(obj)
            seen_keys.add(key)
            label = obj.get("label") or obj.get("class_name") or "object"
            bbox = obj.get("bbox") or obj.get("bounding_box") or [0, 0, 0, 0]
            h_dir = obj.get("direction_h") or obj.get("horizontal_position") or "CENTER"
            v_dir = obj.get("direction_v") or obj.get("vertical_position") or "MIDDLE"
            combined = obj.get("combined_position") or (obj.get("position", {}).get("combined") if isinstance(obj.get("position"), dict) else None)
            d_cat = obj.get("depth_category") or obj.get("depth_class") or "UNKNOWN"
            d_val = obj.get("depth_value") or obj.get("relative_depth_value")
            conf = float(obj.get("confidence", 1.0))
            tid = obj.get("track_id")

            if key in self._store:
                mem = self._store[key]
                # Check for significant movement
                if mem.direction_h != h_dir or mem.direction_v != v_dir:
                    current_frame_changes.append({
                        "type": "object_moved",
                        "label": label,
                        "track_id": tid,
                        "from_position": mem.direction_h,
                        "to_position": h_dir,
                        "frame": frame_number
                    })
                # Update memory object attributes
                mem.bbox = bbox if isinstance(bbox, dict) else {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
                mem.bounding_box = [mem.bbox["x1"], mem.bbox["y1"], mem.bbox["x2"], mem.bbox["y2"]]
                mem.box = mem.bounding_box
                mem.label = label
                mem.class_name = label
                mem.direction_h = h_dir
                mem.direction_v = v_dir
                mem.horizontal_position = h_dir
                mem.vertical_position = v_dir
                mem.combined_position = combined or f"{v_dir}-{h_dir}"
                mem.depth_class = d_cat
                mem.depth_category = d_cat
                mem.depth_value = round(float(d_val), 3) if d_val is not None else None
                mem.relative_depth_value = mem.depth_value
                mem.confidence = round(conf, 4)
                mem.last_seen_frame = frame_number
                mem.last_seen_time = now
                mem.missed_frames = 0
                mem.status = "current"
            else:
                # New object detected
                new_mem = MemoryObject(
                    key=key,
                    label=label,
                    bbox=bbox,
                    direction_h=h_dir,
                    direction_v=v_dir,
                    depth_class=d_cat,
                    depth_value=d_val,
                    confidence=conf,
                    last_seen_frame=frame_number,
                    track_id=tid,
                    combined_position=combined,
                    last_seen_time=now
                )
                self._store[key] = new_mem
                current_frame_changes.append({
                    "type": "new_object",
                    "label": label,
                    "track_id": tid,
                    "position": h_dir,
                    "depth": d_cat,
                    "frame": frame_number
                })

        # Update missed frames and evict stale objects
        keys_to_remove = []
        for key, mem in self._store.items():
            if key not in seen_keys:
                mem.missed_frames += 1
                mem.status = "recent"
                if mem.missed_frames > self.retention_frames:
                    keys_to_remove.append(key)
                    current_frame_changes.append({
                        "type": "object_disappeared",
                        "label": mem.label,
                        "track_id": mem.track_id,
                        "frame": frame_number
                    })

        for key in keys_to_remove:
            del self._store[key]

        # Enforce bounded memory size
        if len(self._store) > self.max_objects:
            # Evict oldest missed objects first
            sorted_by_staleness = sorted(self._store.items(), key=lambda item: (-item[1].missed_frames, item[1].confidence))
            excess = len(self._store) - self.max_objects
            for k, _ in sorted_by_staleness[:excess]:
                del self._store[k]

        # Record changes (keep last 10 changes)
        if current_frame_changes:
            self.recent_changes.extend(current_frame_changes)
            self.recent_changes = self.recent_changes[-10:]

        # Store latest scene graph and relationships
        if scene_graph:
            self.latest_scene_graph = scene_graph
            self.latest_relationships = scene_graph.get("relationships", [])

    def get_all(self) -> List[MemoryObject]:
        """Returns all remembered objects (current and recent), sorted nearest first."""
        objs = list(self._store.values())
        depth_order = {"NEAR": 0, "MID-RANGE": 1, "FAR": 2, "UNKNOWN": 3}
        objs.sort(key=lambda x: (0 if x.status == "current" else 1, depth_order.get(x.depth_category, 3), -x.confidence))
        return objs

    def get_current(self) -> List[MemoryObject]:
        """Returns only objects observed in the latest frame."""
        return [o for o in self.get_all() if o.status == "current"]

    def get_recent(self) -> List[MemoryObject]:
        """Returns objects temporarily missing within retention window."""
        return [o for o in self.get_all() if o.status == "recent"]

    def find_objects_by_label(self, label: str) -> List[MemoryObject]:
        """Finds objects matching a specific class label, nearest first."""
        target = label.strip().lower()
        return [o for o in self.get_all() if o.label.lower() == target or target in o.label.lower()]

    def clear(self):
        """Clears memory on session end."""
        self._store.clear()
        self.current_frame = 0
        self.latest_scene_graph = {"objects": [], "relationships": [], "nodes": [], "edges": []}
        self.latest_relationships = []
        self.recent_changes.clear()
        self.last_updated_time = 0.0
