import os
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from backend.config import config
from backend.exceptions import GoneException

def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def ensure_cache_dir():
    config.REVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def save_cache_frame(frame_id: str, frame_bytes: bytes, ext: str = ".jpg") -> Path:
    ensure_cache_dir()
    file_path = config.REVIEW_CACHE_DIR / f"{frame_id}{ext}"
    file_path.write_bytes(frame_bytes)
    return file_path

def get_cache_frame(frame_id: str) -> Tuple[bytes, str]:
    """
    Retrieves cached frame/crop bytes. If expired or missing, raises 410 Gone.
    """
    ensure_cache_dir()
    # Check possible extensions
    found = None
    for ext in [".jpg", ".jpeg", ".png"]:
        p = config.REVIEW_CACHE_DIR / f"{frame_id}{ext}"
        if p.exists():
            found = p
            break

    if not found:
        raise GoneException("Preview expired, review record is preserved.")

    # Check TTL
    file_age = time.time() - found.stat().st_mtime
    if file_age > config.REVIEW_CACHE_TTL_SECONDS:
        try:
            found.unlink()
        except OSError:
            pass
        raise GoneException("Preview expired, review record is preserved.")

    ext = found.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return found.read_bytes(), mime

def sweep_review_cache() -> int:
    """
    Deletes files in review_cache that are older than REVIEW_CACHE_TTL_SECONDS.
    Never touches reviews or audit_events in the database.
    """
    ensure_cache_dir()
    now = time.time()
    deleted_count = 0
    for f in config.REVIEW_CACHE_DIR.iterdir():
        if f.is_file() and not f.name.startswith("."):
            age = now - f.stat().st_mtime
            if age > config.REVIEW_CACHE_TTL_SECONDS:
                try:
                    f.unlink()
                    deleted_count += 1
                except OSError:
                    pass
    return deleted_count

def sweep_uploads() -> int:
    """
    Deletes files in storage/uploads older than UPLOAD_RETENTION_HOURS.
    """
    ensure_cache_dir()
    now = time.time()
    retention_secs = config.UPLOAD_RETENTION_HOURS * 3600
    deleted_count = 0
    for f in config.UPLOADS_DIR.iterdir():
        if f.is_file() and not f.name.startswith("."):
            age = now - f.stat().st_mtime
            if age > retention_secs:
                try:
                    f.unlink()
                    deleted_count += 1
                except OSError:
                    pass
    return deleted_count
