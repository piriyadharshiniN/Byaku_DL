from typing import List, Dict, Any, Tuple, Optional
from backend.spatial.reasoning import format_direction_phrase

def generate_object_phrase(obj: Any) -> str:
    """Generates a concise accessible natural description for an object."""
    label = obj.label if hasattr(obj, "label") else (obj.get("label") or obj.get("class_name") or "object")
    h_dir = obj.direction_h if hasattr(obj, "direction_h") else (obj.get("direction_h") or obj.get("horizontal_position") or "CENTER")
    v_dir = obj.direction_v if hasattr(obj, "direction_v") else (obj.get("direction_v") or obj.get("vertical_position") or "MIDDLE")
    depth_cat = (getattr(obj, "depth_category", None) or getattr(obj, "depth_class", None) or
                 (obj.get("depth_category") or obj.get("depth_class") if isinstance(obj, dict) else "UNKNOWN"))
    status = getattr(obj, "status", obj.get("status", "current") if isinstance(obj, dict) else "current")

    article = "An" if label[0].lower() in "aeiou" else "A"
    dir_phrase = format_direction_phrase(h_dir, v_dir)

    depth_phrase = ""
    if depth_cat == "NEAR":
        depth_phrase = ", nearby"
    elif depth_cat == "MID-RANGE":
        depth_phrase = " at a mid-range distance"
    elif depth_cat == "FAR":
        depth_phrase = " further away"

    if status == "recent":
        return f"I saw {article.lower()} {label} {dir_phrase}{depth_phrase} a moment ago, but can't confirm it now."

    return f"{article} {label} is {dir_phrase}{depth_phrase}."

def compute_scene_signature(objects: List[Any], relationships: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Computes a deterministic signature of the scene based on labels, positions, depth categories, and key relations.
    Used for suppressing unchanged scene speech.
    """
    parts = []
    for obj in objects:
        label = obj.label if hasattr(obj, "label") else (obj.get("label") or obj.get("class_name") or "")
        h_dir = obj.direction_h if hasattr(obj, "direction_h") else (obj.get("direction_h") or obj.get("horizontal_position") or "")
        depth = (getattr(obj, "depth_category", None) or getattr(obj, "depth_class", None) or
                 (obj.get("depth_category") or obj.get("depth_class") if isinstance(obj, dict) else ""))
        parts.append(f"{label}:{h_dir}:{depth}")
    parts.sort()

    rel_parts = []
    if relationships:
        for r in relationships[:5]:
            sub = r.get("subject") or r.get("source")
            rel = r.get("relation") or r.get("relationship")
            obj_t = r.get("object") or r.get("target")
            rel_parts.append(f"{sub}-{rel}-{obj_t}")
        rel_parts.sort()

    return "|".join(parts) + ("#" + ";".join(rel_parts) if rel_parts else "")

def generate_scene_narration(
    objects: List[Any],
    relationships: Optional[List[Dict[str, Any]]] = None,
    recent_changes: Optional[List[Dict[str, Any]]] = None,
    max_objects: int = 3
) -> Tuple[str, str]:
    """
    Generates template-based speech text, nearest objects first, capped at max_objects.
    Includes important spatial relationships (e.g. cup on table) and recent change notices.
    Returns: (narration_text, scene_signature)
    """
    if not objects:
        return "I don't see any objects I can identify.", ""

    # Sort nearest first (NEAR -> MID-RANGE -> FAR -> UNKNOWN)
    depth_ranks = {"NEAR": 0, "MID-RANGE": 1, "FAR": 2, "UNKNOWN": 3}
    sorted_objs = sorted(
        objects,
        key=lambda x: depth_ranks.get(
            getattr(x, "depth_category", getattr(x, "depth_class", x.get("depth_category", x.get("depth_class", "UNKNOWN")) if isinstance(x, dict) else "UNKNOWN")),
            3
        )
    )

    sig = compute_scene_signature(sorted_objs, relationships)
    top_objects = sorted_objs[:max_objects]

    sentences = []

    # 1. Check for notable change announcements (e.g. newly appeared object or movement)
    if recent_changes:
        latest_change = recent_changes[-1]
        change_type = latest_change.get("type")
        ch_label = latest_change.get("label", "object")
        ch_pos = latest_change.get("position", "")
        if change_type == "new_object" and ch_pos:
            dir_phrase = "on your left" if ch_pos == "LEFT" else ("on your right" if ch_pos == "RIGHT" else "in front of you")
            sentences.append(f"A new {ch_label} appeared {dir_phrase}.")
        elif change_type == "object_moved":
            from_p = latest_change.get("from_position", "").lower()
            to_p = latest_change.get("to_position", "").lower()
            if from_p and to_p and from_p != to_p:
                sentences.append(f"The {ch_label} moved from {from_p} to {to_p}.")

    # 2. Add object descriptions
    for obj in top_objects:
        sentences.append(generate_object_phrase(obj))

    # 3. Incorporate key geometric relationships if present (e.g. resting "on", "OVERLAPPING", "ABOVE")
    if relationships:
        obj_map = {}
        for o in objects:
            oid = str(getattr(o, "track_id", None) if getattr(o, "track_id", None) is not None
                      else (o.get("track_id") if isinstance(o, dict) and o.get("track_id") is not None
                            else (o.get("id") if isinstance(o, dict) else getattr(o, "id", None))))
            lbl = getattr(o, "label", o.get("label") if isinstance(o, dict) else "object")
            obj_map[oid] = lbl

        for r in relationships:
            rel = r.get("relation") or r.get("relationship")
            sub_id = str(r.get("subject") or r.get("source"))
            obj_id = str(r.get("object") or r.get("target"))
            if sub_id in obj_map and obj_id in obj_map and sub_id != obj_id:
                sub_label = obj_map[sub_id]
                target_label = obj_map[obj_id]
                if rel == "on":
                    sentences.append(f"The {sub_label} is resting on the {target_label}.")
                    break
                elif rel == "OVERLAPPING":
                    sentences.append(f"The {sub_label} is overlapping with the {target_label}.")
                    break

    narration_text = " ".join(sentences)
    return narration_text, sig
