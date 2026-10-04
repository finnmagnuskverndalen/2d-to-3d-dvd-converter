"""Encode stereo frames back into a video, optionally re-muxing original audio (Phase 5)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


log = logging.getLogger("stereoscopy.writer")


class VideoWriter:
    def __init__(
        self,
        output_path: Path,
        fps: float,
        size: tuple[int, int],
        codec: str = "libx264",
        bitrate: str = "10000k",
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.size = size   # (width, height)
        self.codec = codec
        self.bitrate = bitrate

    def write_frames(self, frames: Iterable[np.ndarray]) -> None:
        """Stream BGR frames into an ffmpeg encoder via stdin pipe."""
        raise NotImplementedError("Phase 5: ffmpeg rawvideo pipe → h264")

    def add_audio(self, source_media: Path, audio_track_index: int = 0) -> None:
        """Re-mux an audio stream from the source onto the encoded video."""
        raise NotImplementedError("Phase 5: ffmpeg -c:a copy -map 0:v:0 -map 1:a:X")
