"""Smoke tests for VideoReader. Generates a short synthetic MP4 and reads it back."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.video_reader import VideoReader
from src.utils import get_video_metadata


def make_test_clip(path: Path, seconds: int = 2, fps: int = 24, size: tuple[int, int] = (320, 240)) -> Path:
    """Use ffmpeg's testsrc filter to generate a predictable sample video."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size[0]}x{size[1]}:rate={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


def test_metadata_and_frames(tmp_path: Path) -> None:
    clip = make_test_clip(tmp_path / "clip.mp4", seconds=2, fps=24, size=(320, 240))

    md = get_video_metadata(clip)
    assert md.width == 320 and md.height == 240
    assert abs(md.fps - 24.0) < 0.01
    assert md.total_frames == 48

    with VideoReader(clip, cache_dir=tmp_path / ".cache") as vr:
        assert vr.metadata.total_frames == 48

        first = vr.get_frame(0)
        assert first.shape == (240, 320, 3)
        assert first.dtype == np.uint8

        collected = list(vr.frames(sample_rate=4))
        assert len(collected) == 12  # 48 / 4


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_metadata_and_frames(Path(td))
        print("OK")
