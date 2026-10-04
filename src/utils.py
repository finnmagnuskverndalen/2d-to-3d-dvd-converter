"""Shared helpers: logging, metadata probing, filesystem checks."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: Path, name: str = "stereoscopy", level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with file + console handlers. Idempotent."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


@dataclass
class VideoMetadata:
    path: Path
    fps: float
    width: int
    height: int
    duration_s: float
    total_frames: int
    codec: str
    pixel_format: str
    has_audio: bool

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


def _parse_fps(rate: str) -> float:
    """ffprobe reports frame rates as fractions like '24000/1001'."""
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate)


def get_video_metadata(path: Path) -> VideoMetadata:
    """Probe a video file with ffprobe. Raises on invalid input."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate", "0/1"))
    duration = float(data["format"].get("duration", 0.0))
    nb_frames = video_stream.get("nb_frames")
    total_frames = int(nb_frames) if nb_frames and nb_frames != "N/A" else int(round(duration * fps))

    return VideoMetadata(
        path=path,
        fps=fps,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        duration_s=duration,
        total_frames=total_frames,
        codec=video_stream.get("codec_name", "unknown"),
        pixel_format=video_stream.get("pix_fmt", "unknown"),
        has_audio=audio_stream is not None,
    )


def check_disk_space(path: Path, min_free_gb: float) -> tuple[bool, float]:
    """Return (ok, free_gb) for the filesystem containing `path`."""
    path = Path(path)
    probe = path if path.exists() else path.parent
    free_bytes = shutil.disk_usage(probe).free
    free_gb = free_bytes / (1024 ** 3)
    return free_gb >= min_free_gb, free_gb


def ensure_tool(name: str) -> str:
    """Return the absolute path to a required system binary, or raise."""
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Required binary '{name}' not found on PATH")
    return found


def human_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"
