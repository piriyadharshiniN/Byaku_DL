from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2

from backend.jobs import VideoJob, VideoJobManager


class FakeVideoCapture:
    def __init__(self, _file_path, frames):
        self.frames = frames
        self.current_frame = 0
        self.open = True

    def isOpened(self):
        return self.open

    def get(self, property_id):
        values = {
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
            cv2.CAP_PROP_FPS: 2.0,
            cv2.CAP_PROP_FRAME_WIDTH: 20,
            cv2.CAP_PROP_FRAME_HEIGHT: 20,
        }
        return values.get(property_id, 0)

    def set(self, _property_id, frame_number):
        self.current_frame = int(frame_number)
        return True

    def read(self):
        frame = self.frames[self.current_frame]
        if frame is None:
            return False, None
        return True, frame.copy()

    def release(self):
        self.open = False


def _job(frames, detector):
    manager = VideoJobManager()
    job = VideoJob("job_test", "video_test", Path("test-video.mp4"))
    capture = FakeVideoCapture("test-video.mp4", frames)
    depth_estimator = SimpleNamespace(
        loaded=False,
        load_error="Depth model unavailable.",
        estimate_frame=lambda _frame: None,
    )
    with patch("backend.jobs.cv2.VideoCapture", return_value=capture), patch(
        "backend.jobs.generate_scene_narration", return_value=("No objects visible.", None)
    ):
        manager._process_video(job, detector, depth_estimator)
    return job


def test_video_job_fails_instead_of_reporting_detector_errors_as_empty_detections():
    detector = SimpleNamespace(
        loaded=True,
        predict=lambda _frame: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )
    job = _job([np.zeros((20, 20, 3), dtype=np.uint8)], detector)

    assert job.status == "failed"
    assert job.result is None
    assert "Object detection failed on source frame 0: inference failed" == job.error_message


def test_video_job_reports_decode_failures_and_preserves_successful_samples():
    detector = SimpleNamespace(loaded=True, predict=lambda _frame: [])
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    job = _job([frame, None], detector)

    assert job.status == "partial"
    assert job.result["frames_processed"] == 1
    assert job.result["frames_failed"] == 1
    assert job.result["warnings"] == ["Source frame 1 failed to decode."]
    assert [sample["status"] for sample in job.result["samples"]] == [
        "success",
        "decode_error",
    ]
