"""Shared media path detection and collection helpers."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def collect_media_files(input_path: str, extensions: set[str]) -> list[str]:
    """Return matching files from a single file or non-recursive folder."""
    path = Path(input_path).expanduser()
    if path.is_file():
        return [str(path)] if path.suffix.lower() in extensions else []
    if path.is_dir():
        return [
            str(child)
            for child in sorted(path.iterdir())
            if child.is_file() and child.suffix.lower() in extensions
        ]
    raise FileNotFoundError(f"Input not found: {input_path}")


def collect_images(input_path: str) -> list[str]:
    """Return supported image files from a file or non-recursive folder."""
    return collect_media_files(input_path, IMAGE_EXTENSIONS)


def collect_videos(input_path: str) -> list[str]:
    """Return supported video files from a file or non-recursive folder."""
    return collect_media_files(input_path, VIDEO_EXTENSIONS)


def detect_input_kind(input_path: str) -> str:
    """Return image, video, mixed, folder, missing, unknown, or empty."""
    if not input_path.strip():
        return "empty"

    path = Path(input_path).expanduser()
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        return "unknown"

    if path.is_dir():
        has_images = False
        has_videos = False
        try:
            for child in path.iterdir():
                if not child.is_file():
                    continue
                suffix = child.suffix.lower()
                has_images = has_images or suffix in IMAGE_EXTENSIONS
                has_videos = has_videos or suffix in VIDEO_EXTENSIONS
                if has_images and has_videos:
                    return "mixed"
        except OSError:
            return "unknown"
        if has_videos:
            return "video"
        if has_images:
            return "image"
        return "folder"

    return "missing"
