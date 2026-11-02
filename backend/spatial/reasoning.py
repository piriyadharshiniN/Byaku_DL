import math
from typing import Dict, Any, List, Optional, Tuple, Union
from backend.config import config

# Direction calculation functions
def compute_horizontal_position(center_x: float, frame_w: int) -> str:
    """
    Classifies horizontal position as LEFT, CENTER, or RIGHT based on bbox centre relative to width.
    """
    if frame_w <= 0:
        return "CENTER"
    norm_x = center_x / float(frame_w)
    if norm_x < config.HORIZ_LEFT_MAX:
        return "LEFT"
    elif norm_x > config.HORIZ_RIGHT_MIN:
        return "RIGHT"
    return "CENTER"

def compute_vertical_position(center_y: float, frame_h: int) -> str:
    """
    Classifies vertical position as TOP, MIDDLE, or BOTTOM based on bbox centre relative to height.
    """
    if frame_h <= 0:
        return "MIDDLE"
    norm_y = center_y / float(frame_h)
    if norm_y < config.VERT_TOP_MAX:
        return "TOP"
    elif norm_y > config.VERT_BOTTOM_MIN:
        return "BOTTOM"
    return "MIDDLE"

def compute_combined_position(horizontal: str, vertical: str) -> str:
    """
    Combines horizontal and vertical positions into standard descriptor.
    Examples: UPPER-LEFT, UPPER-CENTER, UPPER-RIGHT, MIDDLE-LEFT, CENTER,
              MIDDLE-RIGHT, LOWER-LEFT, LOWER-CENTER, LOWER-RIGHT.
    """
    h = (horizontal or "CENTER").upper()
    v = (vertical or "MIDDLE").upper()

    if h == "CENTER" and v == "MIDDLE":
        return "CENTER"
    elif v == "TOP":
        return f"UPPER-{h}" if h != "CENTER" else "UPPER-CENTER"
    elif v == "BOTTOM":
        return f"LOWER-{h}" if h != "CENTER" else "LOWER-CENTER"
    else: # MIDDLE
        return f"MIDDLE-{h}"

def compute_direction(bbox: Any, frame_w: int, frame_h: int) -> Tuple[str, str]:
    """
    Computes horizontal (LEFT, CENTER, RIGHT) and vertical (TOP, MIDDLE, BOTTOM) direction from bbox.
    """
    if isinstance(bbox, dict):
        x1, y1 = float(bbox.get("x1", 0)), float(bbox.get("y1", 0))
        x2, y2 = float(bbox.get("x2", 0)), float(bbox.get("y2", 0))
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    elif hasattr(bbox, "bbox"):
        return compute_direction(bbox.bbox, frame_w, frame_h)
    else:
        return "CENTER", "MIDDLE"

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    h_dir = compute_horizontal_position(cx, frame_w)
    v_dir = compute_vertical_position(cy, frame_h)
    return h_dir, v_dir

def format_direction_phrase(h_dir: str, v_dir: str) -> str:
    """Natural phrasing for spatial orientation and voice audio assistance."""
    h = (h_dir or "CENTER").upper()
    v = (v_dir or "MIDDLE").upper()

    if h == "CENTER":
        if v == "TOP":
            return "directly ahead above"
        elif v == "BOTTOM":
            return "in front of you lower down"
        else:
            return "directly in front of you"
    elif h == "LEFT":
        if v == "TOP":
            return "upper left"
        elif v == "BOTTOM":
            return "lower left"
        else:
            return "on your left"
    else: # RIGHT
        if v == "TOP":
            return "upper right"
        elif v == "BOTTOM":
            return "lower right"
        else:
            return "on your right"

def _get_box_coords(obj: Any) -> Optional[Tuple[float, float, float, float]]:
    """Helper to extract (x1, y1, x2, y2) from object dict or instance."""
    if isinstance(obj, dict):
        if "bbox" in obj and isinstance(obj["bbox"], dict):
            b = obj["bbox"]
            return float(b.get("x1", 0)), float(b.get("y1", 0)), float(b.get("x2", 0)), float(b.get("y2", 0))
        elif "bounding_box" in obj and isinstance(obj["bounding_box"], (list, tuple)) and len(obj["bounding_box"]) >= 4:
            b = obj["bounding_box"]
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        elif "box" in obj and isinstance(obj["box"], (list, tuple)) and len(obj["box"]) >= 4:
            b = obj["box"]
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    elif hasattr(obj, "bbox") and isinstance(obj.bbox, dict):
        b = obj.bbox
        return float(b.get("x1", 0)), float(b.get("y1", 0)), float(b.get("x2", 0)), float(b.get("y2", 0))
    return None

def compute_object_relationships(objects: List[Any], frame_w: int, frame_h: int) -> List[Dict[str, Any]]:
    """
    Computes pairwise spatial relationships between detected objects in 2D image space.
    Supported relationships:
    - LEFT_OF: Object A is to the left of Object B with vertical overlap
    - RIGHT_OF: Object A is to the right of Object B with vertical overlap
    - ABOVE: Object A is above Object B with horizontal overlap
    - BELOW: Object A is below Object B with horizontal overlap
    - OVERLAPPING: Object A and B intersect significantly (area overlap > threshold)
    - NEAR_IN_IMAGE: Normalized centre distance <= 0.25 of image diagonal
    - FAR_IN_IMAGE: Normalized centre distance >= 0.55 of image diagonal
    - on: Resting on surface (backward-compatible)
    """
    relationships: List[Dict[str, Any]] = []
    n = len(objects)
    if n < 2 or frame_w <= 0 or frame_h <= 0:
        return relationships

    diag = math.hypot(frame_w, frame_h)

    for i in range(n):
        obj_a = objects[i]
        coords_a = _get_box_coords(obj_a)
        if not coords_a:
            continue
        ax1, ay1, ax2, ay2 = coords_a
        aw = max(1.0, ax2 - ax1)
        ah = max(1.0, ay2 - ay1)
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        area_a = aw * ah

        if isinstance(obj_a, dict) and obj_a.get("track_id") is not None:
            id_a = obj_a["track_id"]
        elif hasattr(obj_a, "track_id") and getattr(obj_a, "track_id", None) is not None:
            id_a = obj_a.track_id
        else:
            id_a = obj_a.get("id", f"object_{i+1}") if isinstance(obj_a, dict) else f"object_{i+1}"

        for j in range(n):
            if i == j:
                continue
            obj_b = objects[j]
            coords_b = _get_box_coords(obj_b)
            if not coords_b:
                continue
            bx1, by1, bx2, by2 = coords_b
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)
            bcx = (bx1 + bx2) / 2.0
            bcy = (by1 + by2) / 2.0
            area_b = bw * bh

            if isinstance(obj_b, dict) and obj_b.get("track_id") is not None:
                id_b = obj_b["track_id"]
            elif hasattr(obj_b, "track_id") and getattr(obj_b, "track_id", None) is not None:
                id_b = obj_b.track_id
            else:
                id_b = obj_b.get("id", f"object_{j+1}") if isinstance(obj_b, dict) else f"object_{j+1}"

            # 1. Normalized Euclidean distance between bounding-box centres
            dist = math.hypot(acx - bcx, acy - bcy)
            norm_dist = dist / max(1.0, diag)

            # NEAR_IN_IMAGE / FAR_IN_IMAGE
            if norm_dist <= config.NEAR_IMAGE_DIST_THRESHOLD:
                relationships.append({
                    "subject": id_a,
                    "relation": "NEAR_IN_IMAGE",
                    "object": id_b,
                    "source": id_a,
                    "target": id_b,
                    "relationship": "NEAR_IN_IMAGE",
                    "distance": round(norm_dist, 3)
                })
            elif norm_dist >= config.FAR_IMAGE_DIST_THRESHOLD:
                relationships.append({
                    "subject": id_a,
                    "relation": "FAR_IN_IMAGE",
                    "object": id_b,
                    "source": id_a,
                    "target": id_b,
                    "relationship": "FAR_IN_IMAGE",
                    "distance": round(norm_dist, 3)
                })

            # 2. OVERLAPPING
            inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter_area = inter_w * inter_h
            min_box_area = min(area_a, area_b)

            if inter_area > (config.OVERLAP_AREA_THRESHOLD * min_box_area):
                relationships.append({
                    "subject": id_a,
                    "relation": "OVERLAPPING",
                    "object": id_b,
                    "source": id_a,
                    "target": id_b,
                    "relationship": "OVERLAPPING"
                })

            # 3. ABOVE / BELOW
            # Significant horizontal alignment (overlap ratio > 25% of narrower box)
            if inter_w > (0.25 * min(aw, bw)):
                if acy < bcy and ay2 <= (by1 + 0.3 * bh):
                    relationships.append({
                        "subject": id_a,
                        "relation": "ABOVE",
                        "object": id_b,
                        "source": id_a,
                        "target": id_b,
                        "relationship": "ABOVE"
                    })
                elif acy > bcy and ay1 >= (by2 - 0.3 * bh):
                    relationships.append({
                        "subject": id_a,
                        "relation": "BELOW",
                        "object": id_b,
                        "source": id_a,
                        "target": id_b,
                        "relationship": "BELOW"
                    })

            # Support resting surface "on"
            if inter_w > (0.4 * min(aw, bw)):
                v_delta = ay2 - by1
                if -0.15 * bh <= v_delta <= 0.35 * bh and acy < bcy:
                    relationships.append({
                        "subject": id_a,
                        "relation": "on",
                        "object": id_b,
                        "source": id_a,
                        "target": id_b,
                        "relationship": "on"
                    })

            # 4. LEFT_OF / RIGHT_OF
            # Significant vertical alignment (overlap ratio > 25% of shorter box)
            if inter_h > (0.25 * min(ah, bh)):
                if acx < bcx and ax2 <= (bx1 + 0.3 * bw):
                    relationships.append({
                        "subject": id_a,
                        "relation": "LEFT_OF",
                        "object": id_b,
                        "source": id_a,
                        "target": id_b,
                        "relationship": "LEFT_OF"
                    })
                elif acx > bcx and ax1 >= (bx2 - 0.3 * bw):
                    relationships.append({
                        "subject": id_a,
                        "relation": "RIGHT_OF",
                        "object": id_b,
                        "source": id_a,
                        "target": id_b,
                        "relationship": "RIGHT_OF"
                    })

    return relationships

def build_scene_graph(objects: List[Any], frame_w: int, frame_h: int) -> Dict[str, Any]:
    """
    Constructs a structured scene graph with detected objects and their pairwise spatial relationships.
    Format adheres to Prompt 4.E specification while remaining backward compatible with dashboard UI:
    {
        "objects": [
            {
                "id": "object_1",
                "label": "person",
                "confidence": 0.92,
                "position": {
                    "horizontal": "LEFT",
                    "vertical": "MIDDLE",
                    "combined": "MIDDLE-LEFT"
                },
                "depth": {
                    "category": "NEAR",
                    "value": 0.82
                }
            }
        ],
        "relationships": [
            {
                "subject": "object_1",
                "relation": "LEFT_OF",
                "object": "object_2"
            }
        ],
        "nodes": [...], # for dashboard SVG rendering
        "edges": [...]  # for dashboard SVG rendering
    }
    """
    graph_objects = []
    nodes = []

    for idx, obj in enumerate(objects):
        coords = _get_box_coords(obj)
        if coords:
            x1, y1, x2, y2 = coords
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            h_dir = compute_horizontal_position(cx, frame_w)
            v_dir = compute_vertical_position(cy, frame_h)
            combined = compute_combined_position(h_dir, v_dir)
            bbox_list = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        else:
            h_dir = "CENTER"
            v_dir = "MIDDLE"
            combined = "CENTER"
            bbox_list = [0.0, 0.0, 0.0, 0.0]

        # Extract attributes
        if isinstance(obj, dict):
            label = obj.get("label") or obj.get("class_name") or "object"
            conf = float(obj.get("confidence", 1.0))
            tid = obj.get("track_id")
            depth_cat = obj.get("depth_category") or obj.get("depth_class") or "UNKNOWN"
            depth_val = obj.get("depth_value") or obj.get("relative_depth_value")
            obj_id = str(tid if tid is not None else obj.get("id", f"object_{idx+1}"))
        else:
            label = getattr(obj, "label", "object")
            conf = float(getattr(obj, "confidence", 1.0))
            tid = getattr(obj, "track_id", None)
            depth_cat = getattr(obj, "depth_class", "UNKNOWN")
            depth_val = getattr(obj, "depth_value", None)
            obj_id = str(tid if tid is not None else f"object_{idx+1}")

        obj_entry = {
            "id": obj_id,
            "track_id": tid,
            "label": label,
            "class_name": label,
            "confidence": round(conf, 4),
            "bounding_box": bbox_list,
            "position": {
                "horizontal": h_dir,
                "vertical": v_dir,
                "combined": combined
            },
            "depth": {
                "category": depth_cat,
                "value": round(depth_val, 3) if depth_val is not None else None
            }
        }
        graph_objects.append(obj_entry)

        # Node representation for dashboard SVG
        nodes.append({
            "id": obj_id,
            "object_id": obj_id,
            "track_id": tid,
            "label": label,
            "class_name": label,
            "confidence": round(conf, 4),
            "horizontal_position": h_dir,
            "vertical_position": v_dir,
            "relative_depth_value": round(depth_val, 3) if depth_val is not None else None,
            "depth_category": depth_cat
        })

    # Relationships
    relationships = compute_object_relationships(objects, frame_w, frame_h)

    return {
        "objects": graph_objects,
        "relationships": relationships,
        "nodes": nodes,
        "edges": relationships
    }

def extract_relationships(graph_or_objects: Any, frame_w: int = 640, frame_h: int = 480) -> List[Dict[str, Any]]:
    """
    Extracts structured relationships list from a scene graph dictionary or raw object list.
    """
    if isinstance(graph_or_objects, dict) and "relationships" in graph_or_objects:
        return graph_or_objects["relationships"]
    elif isinstance(graph_or_objects, list):
        return compute_object_relationships(graph_or_objects, frame_w, frame_h)
    return []
