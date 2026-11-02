import re
from typing import List, Dict, Any, Optional
from backend.spatial.reasoning import format_direction_phrase
from backend.spatial.scene_memory import SceneMemory, MemoryObject

LABEL_SYNONYMS = {
    "couch": "sofa",
    "sofa": "couch",
    "phone": "cell phone",
    "smartphone": "cell phone",
    "mobile": "cell phone",
    "cellphone": "cell phone",
    "people": "person",
    "human": "person",
    "man": "person",
    "woman": "person",
    "child": "person",
    "guy": "person",
    "tv": "tv",
    "television": "tv",
    "screen": "tv",
    "monitor": "tv",
    "laptop": "laptop",
    "computer": "laptop",
    "notebook": "laptop",
    "pc": "laptop",
    "cup": "cup",
    "mug": "cup",
    "glass": "wine glass",
    "bottle": "bottle",
    "water": "bottle",
    "drink": "bottle",
    "chair": "chair",
    "seat": "chair",
    "table": "dining table",
    "desk": "dining table",
    "keys": "keys",
    "backpack": "backpack",
    "bag": "backpack",
    "book": "book"
}

QUESTION_STOPWORDS = {
    "where", "wheres", "where's", "what", "whats", "what's", "which", "is", "are", "there",
    "here", "see", "find", "locate", "tell", "closest", "nearest", "around", "nearby",
    "close", "the", "my", "a", "an", "any", "me", "in", "front", "of", "do", "you", "can",
    "has", "have", "anything", "did", "changed", "move", "moved", "object", "objects", "side",
    "on", "to"
}

def normalize_target_label(query_token: str) -> str:
    cleaned = query_token.strip().lower()
    return LABEL_SYNONYMS.get(cleaned, cleaned)

class SceneQAEngine:
    """
    Deterministic rule-based Voice Q&A engine strictly grounded in Session Scene Memory.
    Never hallucinates objects or distances; strictly respects perceptual boundaries.
    """
    def __init__(self):
        pass

    def answer_question(self, question: str, memory: SceneMemory) -> str:
        q = question.strip().lower()
        if not q:
            return "Please ask a question about your surroundings."

        all_objects = memory.get_all()

        # Handle empty scene immediately if no objects were ever recorded
        if not all_objects and not memory.recent_changes:
            return "I do not see any objects in the scene right now."

        # -------------------------------------------------------------
        # Intent 1: "Has anything changed?" / "Did anything move?"
        # -------------------------------------------------------------
        if any(re.search(pat, q) for pat in [
            r"has anything changed",
            r"any (changes|change)",
            r"did anything move",
            r"what changed",
            r"anything new"
        ]):
            if not memory.recent_changes:
                return "The scene appears unchanged."
            latest = memory.recent_changes[-1]
            c_type = latest.get("type")
            c_lbl = latest.get("label", "object")
            if c_type == "new_object":
                pos = latest.get("position", "")
                dir_str = "on your left" if pos == "LEFT" else ("on your right" if pos == "RIGHT" else "in front of you")
                return f"A new {c_lbl} appeared {dir_str}."
            elif c_type == "object_moved":
                fp = latest.get("from_position", "").lower()
                tp = latest.get("to_position", "").lower()
                return f"The {c_lbl} moved from {fp} to {tp}."
            elif c_type == "object_disappeared":
                return f"The {c_lbl} is no longer visible."
            return "The scene appears unchanged."

        # If all objects are gone and wasn't a change question
        if not all_objects:
            return "I do not see any objects in the scene right now."

        # -------------------------------------------------------------
        # Intent 2: "What objects are nearby?" / "What is close to me?"
        # -------------------------------------------------------------
        if any(re.search(pat, q) for pat in [
            r"what (objects |items |things )?(are |is )?(nearby|close|near me)",
            r"(is |are )?anything (nearby|close|near me)",
            r"what('s| is) nearby",
            r"what('s| is) close"
        ]):
            near_objs = [o for o in all_objects if getattr(o, "depth_category", "UNKNOWN") == "NEAR" or getattr(o, "depth_class", "UNKNOWN") == "NEAR"]
            if not near_objs:
                return "There are no objects nearby right now."
            top_near = near_objs[:3]
            phrases = []
            for obj in top_near:
                art = "an" if obj.label[0].lower() in "aeiou" else "a"
                dp = format_direction_phrase(obj.direction_h, obj.direction_v)
                phrases.append(f"{art} {obj.label} {dp}")
            joined = ", and ".join(phrases) if len(phrases) <= 2 else f"{', '.join(phrases[:-1])}, and {phrases[-1]}"
            return f"Nearby, I see {joined}."

        # -------------------------------------------------------------
        # Intent 3: "Which object is closest?" / "What is closest?"
        # -------------------------------------------------------------
        if any(re.search(pat, q) for pat in [
            r"which (object |item |thing )?(is )?(closest|nearest)",
            r"what('s| is) (the )?(closest|nearest)( thing| object)?"
        ]):
            # all_objects is already sorted nearest first
            closest = all_objects[0]
            art = "an" if closest.label[0].lower() in "aeiou" else "a"
            dp = format_direction_phrase(closest.direction_h, closest.direction_v)
            depth_desc = ", nearby" if closest.depth_category == "NEAR" else ""
            if closest.status == "recent":
                return f"The closest object was {art} {closest.label} {dp}{depth_desc} a moment ago."
            return f"The closest object is {art} {closest.label} {dp}{depth_desc}."

        # -------------------------------------------------------------
        # Intent 4: "What is on my left?" / "What is on my right?"
        # -------------------------------------------------------------
        left_match = re.search(r"what('s| is) (on (my |the )?left|to the left)", q)
        right_match = re.search(r"what('s| is) (on (my |the )?right|to the right)", q)
        if left_match or right_match:
            target_side = "LEFT" if left_match else "RIGHT"
            side_name = "left" if target_side == "LEFT" else "right"
            side_objs = [o for o in all_objects if getattr(o, "direction_h", "") == target_side]
            if not side_objs:
                return f"There are no objects on your {side_name} right now."
            first_obj = side_objs[0]
            art = "an" if first_obj.label[0].lower() in "aeiou" else "a"
            depth_desc = ", nearby" if first_obj.depth_category == "NEAR" else (" at a mid-range distance" if first_obj.depth_category == "MID-RANGE" else "")
            return f"On your {side_name}, there is {art} {first_obj.label}{depth_desc}."

        # -------------------------------------------------------------
        # Intent 5: "What is in front of me?" / "What's ahead?"
        # -------------------------------------------------------------
        if any(re.search(pat, q) for pat in [
            r"what('s| is) in front",
            r"what('s| is) ahead",
            r"what('s| is) (directly )?center",
            r"anything in front"
        ]):
            front_objs = [o for o in all_objects if getattr(o, "direction_h", "") == "CENTER"]
            if not front_objs:
                return "There are no objects directly in front of you."
            closest = front_objs[0]
            dir_str = format_direction_phrase(closest.direction_h, closest.direction_v)
            depth_str = f", {closest.depth_category.lower()}" if closest.depth_category != "UNKNOWN" else ""
            if closest.status == "recent":
                return f"I saw a {closest.label} {dir_str}{depth_str} a moment ago, but cannot confirm it now."
            return f"There is a {closest.label} {dir_str}{depth_str}."

        # -------------------------------------------------------------
        # Intent 6: "Describe the surroundings." / "What do you see?"
        # -------------------------------------------------------------
        if any(re.search(pat, q) for pat in [
            r"describe (the )?(surroundings|scene|environment)",
            r"what('s| is) (around|here|in the scene)",
            r"(what|what all) do you see",
            r"(list|tell me) (everything|the scene|objects|all)"
        ]):
            current_objs = [o for o in all_objects if o.status == "current"]
            if not current_objs:
                return "I don't see any active objects right now, though I saw items a moment ago."
            labels = [o.label for o in current_objs[:5]]
            unique_labels = list(dict.fromkeys(labels))
            if len(unique_labels) == 1:
                return f"I see a {unique_labels[0]} in the scene."
            elif len(unique_labels) == 2:
                return f"I see a {unique_labels[0]} and a {unique_labels[1]} in the scene."
            else:
                items_str = ", ".join(f"a {l}" for l in unique_labels[:-1]) + f", and a {unique_labels[-1]}"
                return f"I see {items_str} around you."

        # -------------------------------------------------------------
        # Intent 7: "Is there a [label]?" (Existence question)
        # -------------------------------------------------------------
        existence_match = re.search(r"(?:is there|do you see|are there)(?: any| a| an)? ([\w\s]+?)(?:\?|$)", q)
        is_existence_question = bool(existence_match) and not ("what" in q or "where" in q)

        # Extract target object candidate token
        target_token = None
        if existence_match:
            candidate = existence_match.group(1).strip()
            # Remove trailing adverbs like "nearby", "here", "close"
            candidate = re.sub(r"\b(nearby|here|close|around|now)\b", "", candidate).strip()
            if candidate:
                target_token = candidate

        if not target_token:
            clean_q = re.sub(r"[^\w\s]", "", q)
            words = clean_q.split()
            for w in words:
                if w not in QUESTION_STOPWORDS and (w in LABEL_SYNONYMS or len(w) >= 3):
                    target_token = w
                    break

        if not target_token:
            return "I did not understand the question. You can ask where an object is, what is in front of you, or which object is closest."

        norm_target = normalize_target_label(target_token)

        # Match against objects in scene memory
        matching_objs = [
            o for o in all_objects
            if o.label.lower() == norm_target or norm_target in o.label.lower() or LABEL_SYNONYMS.get(o.label.lower()) == norm_target
        ]

        # Object not found
        if not matching_objs:
            if is_existence_question:
                return f"No, I do not see a {target_token} in the scene right now."
            return f"I cannot currently identify a {target_token} in the scene."

        # Object is found
        best_match = matching_objs[0]
        dir_phrase = format_direction_phrase(best_match.direction_h, best_match.direction_v)
        depth_phrase = ""
        if best_match.depth_category == "NEAR":
            depth_phrase = ", nearby"
        elif best_match.depth_category == "MID-RANGE":
            depth_phrase = " at a mid-range distance"
        elif best_match.depth_category == "FAR":
            depth_phrase = " further away"

        # Check for existence affirmation
        if is_existence_question:
            if best_match.status == "recent":
                return f"I saw a {best_match.label} {dir_phrase}{depth_phrase} a moment ago, but cannot confirm it now."
            return f"Yes, there is a {best_match.label} {dir_phrase}{depth_phrase}."

        # Nearest / Locate response
        is_nearest_intent = "nearest" in q or "closest" in q
        prefix = "The closest " if is_nearest_intent else "A "
        if best_match.status == "recent":
            return f"I saw {prefix.lower()}{best_match.label} {dir_phrase}{depth_phrase} a moment ago, but can't confirm it now."

        return f"{prefix}{best_match.label} is {dir_phrase}{depth_phrase}."
