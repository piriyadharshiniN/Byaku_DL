import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from backend.config import config
from backend.exceptions import ValidationException

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png"}

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_VIDEO_MIMES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
    "application/octet-stream"
}

def validate_image_bytes(data: bytes, filename: str = "", content_type: str = "") -> Tuple[np.ndarray, int, int]:
    """
    Validates image bytes against size, extension, MIME type, magic bytes, OpenCV decodability, and dimensions.
    Returns: (decoded_bgr_image, width, height)
    """
    if not data:
        raise ValidationException("Image data is empty.", code="EMPTY_FILE", status_code=400)

    if len(data) > config.MAX_IMAGE_BYTES:
        raise ValidationException(
            f"Image exceeds maximum allowed size of {config.MAX_IMAGE_BYTES // (1024*1024)} MB.",
            code="FILE_TOO_LARGE",
            status_code=413
        )

    ext = Path(filename).suffix.lower() if filename else ""
    if ext and ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValidationException(
            f"Unsupported image extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}.",
            code="UNSUPPORTED_EXTENSION",
            status_code=400
        )

    # Magic bytes verification
    is_jpeg = data.startswith(b"\xff\xd8\xff")
    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpeg or is_png):
        raise ValidationException("Invalid image header/magic bytes. File is not a valid JPEG or PNG.", code="INVALID_MAGIC_BYTES", status_code=400)

    # OpenCV decoding
    np_arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValidationException("Image could not be decoded. File may be corrupt.", code="IMAGE_DECODE_FAILED", status_code=400)

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValidationException(f"Invalid image dimensions: {w}x{h}.", code="INVALID_DIMENSIONS", status_code=400)

    return image, w, h

def validate_video_file(file_path: Path, filename: str = "", content_type: str = "") -> Tuple[int, int, int, float]:
    """
    Validates a video file against size, extension, and OpenCV decodability.
    Returns: (width, height, total_frames, fps)
    """
    if not file_path.exists():
        raise ValidationException("Video file does not exist.", code="FILE_NOT_FOUND", status_code=404)

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValidationException("Video file is empty.", code="EMPTY_FILE", status_code=400)

    if file_size > config.MAX_VIDEO_BYTES:
        raise ValidationException(
            f"Video exceeds maximum allowed size of {config.MAX_VIDEO_BYTES // (1024*1024)} MB.",
            code="FILE_TOO_LARGE",
            status_code=413
        )

    ext = Path(filename or file_path.name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValidationException(
            f"Unsupported video extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}.",
            code="UNSUPPORTED_EXTENSION",
            status_code=400
        )

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise ValidationException("Video could not be opened or decoded.", code="VIDEO_DECODE_FAILED", status_code=400)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    # Read at least one frame to verify codec compatibility
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None or w <= 0 or h <= 0:
        raise ValidationException("Unable to decode video frames. Codec might be unsupported or file corrupt.", code="VIDEO_READ_FAILED", status_code=400)

    return w, h, total_frames, fps
