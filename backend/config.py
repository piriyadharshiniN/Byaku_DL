import os
from pathlib import Path

class Config:
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8765"))

    # Storage paths
    STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", "storage"))
    DB_PATH: Path = Path(os.getenv("DB_PATH", "storage/byaku.db"))
    UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
    REVIEW_CACHE_DIR: Path = STORAGE_DIR / "review_cache"
    WEIGHTS_DIR: Path = Path(os.getenv("WEIGHTS_DIR", "weights"))

    # YOLO Detector configuration
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "weights/yolo11n.pt")
    YOLO_CONF_THRESHOLD: float = float(os.getenv("YOLO_CONF_THRESHOLD", os.getenv("CONFIDENCE_THRESHOLD", "0.25")))
    YOLO_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU_THRESHOLD", os.getenv("IOU_THRESHOLD", "0.45")))
    YOLO_IMGSZ: int = int(os.getenv("YOLO_IMGSZ", os.getenv("MAX_SIDE", "640")))
    YOLO_DEVICE: str = os.getenv("YOLO_DEVICE", "auto")
    YOLO_MAX_DET: int = int(os.getenv("YOLO_MAX_DET", "100"))
    YOLO_CONCURRENCY_LIMIT: int = int(os.getenv("YOLO_CONCURRENCY_LIMIT", "1"))
    YOLO_DOWNLOAD_TIMEOUT_SECONDS: int = int(os.getenv("YOLO_DOWNLOAD_TIMEOUT_SECONDS", "10"))

    # Legacy aliases
    CONFIDENCE_THRESHOLD: float = YOLO_CONF_THRESHOLD
    IOU_THRESHOLD: float = YOLO_IOU_THRESHOLD
    MAX_SIDE: int = YOLO_IMGSZ

    # Relative depth estimation (Depth Anything V2 Small)
    DEPTH_MODEL_NAME: str = os.getenv("DEPTH_MODEL_NAME", "depth-anything/Depth-Anything-V2-Small-hf")
    DEPTH_WEIGHTS_PATH: str = os.getenv("DEPTH_WEIGHTS_PATH", "weights/depth_anything_v2_vits.pth")
    DEPTH_DEVICE: str = os.getenv("DEPTH_DEVICE", "auto")
    DEPTH_INPUT_SIZE: int = int(os.getenv("DEPTH_INPUT_SIZE", "518"))
    NEAR_THRESHOLD: float = float(os.getenv("NEAR_THRESHOLD", "0.66"))
    MID_THRESHOLD: float = float(os.getenv("MID_THRESHOLD", "0.33"))
    FLAT_DEPTH_EPSILON: float = float(os.getenv("FLAT_DEPTH_EPSILON", "0.02"))

    # Horizontal direction boundaries (thirds of frame width: LEFT [0, 0.33), CENTER [0.33, 0.67], RIGHT (0.67, 1.0])
    HORIZ_LEFT_MAX: float = float(os.getenv("HORIZ_LEFT_MAX", "0.33"))
    HORIZ_RIGHT_MIN: float = float(os.getenv("HORIZ_RIGHT_MIN", "0.67"))

    # Vertical direction boundaries (thirds: TOP [0, 0.33), MIDDLE [0.33, 0.67], BOTTOM (0.67, 1.0])
    VERT_TOP_MAX: float = float(os.getenv("VERT_TOP_MAX", "0.33"))
    VERT_BOTTOM_MIN: float = float(os.getenv("VERT_BOTTOM_MIN", "0.67"))

    # Spatial relationship geometric thresholds
    OVERLAP_AREA_THRESHOLD: float = float(os.getenv("OVERLAP_AREA_THRESHOLD", "0.15"))
    NEAR_IMAGE_DIST_THRESHOLD: float = float(os.getenv("NEAR_IMAGE_DIST_THRESHOLD", "0.25"))
    FAR_IMAGE_DIST_THRESHOLD: float = float(os.getenv("FAR_IMAGE_DIST_THRESHOLD", "0.55"))

    # Retention & Memory
    SCENE_RETENTION_FRAMES: int = int(os.getenv("SCENE_RETENTION_FRAMES", "5"))
    REVIEW_CACHE_TTL_SECONDS: int = int(os.getenv("REVIEW_CACHE_TTL_SECONDS", "180"))
    UPLOAD_RETENTION_HOURS: int = int(os.getenv("UPLOAD_RETENTION_HOURS", "24"))

    # Upload size limits
    MAX_IMAGE_BYTES: int = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))) # 10 MB
    MAX_VIDEO_BYTES: int = int(os.getenv("MAX_VIDEO_BYTES", str(100 * 1024 * 1024))) # 100 MB

    # Video processing
    VIDEO_SAMPLE_INTERVAL_SECONDS: float = float(os.getenv("VIDEO_SAMPLE_INTERVAL_SECONDS", "0.5"))
    VIDEO_MAX_FRAMES: int = int(os.getenv("VIDEO_MAX_FRAMES", "30"))

config = Config()
