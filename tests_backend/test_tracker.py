from backend.models.tracker import ByteTrackSessionTracker


def _detection(x1, x2):
    return {
        "bbox": [x1, 0, x2, 10],
        "confidence": 0.9,
        "class_id": 1,
        "label": "person",
    }


def test_tracker_reports_pixel_motion_and_lifecycle():
    tracker = ByteTrackSessionTracker()

    first = tracker.update([_detection(0, 10)])
    track_id = first[0]["track_id"]
    assert first[0]["state"] == "new"
    assert tracker.last_events[0]["state"] == "new"

    second = tracker.update([_detection(5, 15)])
    assert second[0]["track_id"] == track_id
    assert second[0]["movement_direction"] == "right"
    assert second[0]["velocity_px_per_frame"]["x"] == 5.0
    assert tracker.last_events[0]["state"] == "continuing"

    third = tracker.update([_detection(10, 20)])
    assert third[0]["track_id"] == track_id
    assert third[0]["movement_direction"] == "right"
    assert 4.0 < third[0]["velocity_px_per_frame"]["x"] < 5.0

    assert tracker.update([]) == []
    assert tracker.last_events[0]["state"] == "lost"

    reacquired = tracker.update([_detection(20, 30)])
    assert reacquired[0]["track_id"] == track_id
    assert tracker.last_events[0]["state"] == "reacquired"
