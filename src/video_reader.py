"""Video / DVD input: unified frame-level reader for files, VIDEO_TS folders and ISOs.

Three input shapes are supported:
  * A regular video file (mp4, mkv, avi, mov, ...)   → opened directly with OpenCV
  * A VIDEO_TS directory                             → VOB files concatenated via ffmpeg
  * A DVD ISO file                                   → read directly by ffmpeg (`-f dvdvideo` or stream)

For VIDEO_TS / ISO sources, ffmpeg produces a single transport stream that we pipe
into OpenCV through a temporary MKV in a cache directory. This keeps the per-frame
interface identical regardless of source type.
"""

from __future__ import annotations

import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .utils import VideoMetadata, ensure_tool, get_video_metadata


log = logging.getLogger("stereoscopy.video_reader")

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".ts", ".vob"}


class VideoReader:
    """Frame-level reader with a uniform interface across file / VIDEO_TS / ISO sources.

    Usage::

        with VideoReader(path) as vr:
            print(vr.metadata)
            for i, frame in enumerate(vr.frames()):
                ...
    """

    def __init__(self, source: Path | str, cache_dir: Optional[Path] = None):
        self.source = Path(source)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache")
        self._video_path: Optional[Path] = None   # concrete file OpenCV opens
        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._owns_video_path = False             # true when we materialised a temp file

    # ------------------------------------------------------------------ open

    def open(self) -> "VideoReader":
        if self._capture is not None:
            return self

        if self.source.is_file() and self.source.suffix.lower() in VIDEO_SUFFIXES:
            self._video_path = self.source
        elif self.source.is_dir() and self._looks_like_video_ts(self.source):
            self._video_path = self._extract_video_ts(self.source)
            self._owns_video_path = True
        elif self.source.is_file() and self.source.suffix.lower() == ".iso":
            self._video_path = self._extract_iso(self.source)
            self._owns_video_path = True
        else:
            raise ValueError(
                f"Unsupported input: {self.source}. Expected a video file, VIDEO_TS directory, or .iso"
            )

        self._metadata = get_video_metadata(self._video_path)
        self._capture = cv2.VideoCapture(str(self._video_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"OpenCV failed to open {self._video_path}")

        log.info(
            "Opened %s (%dx%d @ %.3f fps, %d frames, codec=%s)",
            self._video_path.name,
            self._metadata.width,
            self._metadata.height,
            self._metadata.fps,
            self._metadata.total_frames,
            self._metadata.codec,
        )
        if self._metadata.height < 720:
            log.warning(
                "Source height %dpx is below 720p — depth estimation quality may suffer.",
                self._metadata.height,
            )
        return self

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        # we intentionally keep extracted temp files unless caller cleans the cache dir

    def __enter__(self) -> "VideoReader":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------------------------------------------------------------- metadata

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("VideoReader not opened — call .open() first")
        return self._metadata

    # ---------------------------------------------------------------- frames

    def get_frame(self, index: int) -> np.ndarray:
        """Return the BGR frame at `index` (0-based). Random access via SET_POS_FRAMES."""
        cap = self._require_capture()
        if index < 0 or index >= self.metadata.total_frames:
            raise IndexError(f"frame {index} out of range [0, {self.metadata.total_frames})")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {index}")
        return frame

    def frames(self, sample_rate: int = 1, start: int = 0, end: Optional[int] = None) -> Iterator[np.ndarray]:
        """Yield frames sequentially. `sample_rate=N` yields every Nth frame.

        Prefers sequential reads over random seeks (orders of magnitude faster on
        most codecs). Only seeks on the initial jump to `start`.
        """
        if sample_rate < 1:
            raise ValueError("sample_rate must be >= 1")

        cap = self._require_capture()
        total = self.metadata.total_frames
        end = total if end is None else min(end, total)
        if start >= end:
            return

        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        idx = start
        while idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % sample_rate == 0:
                yield frame
            idx += 1

    def extract_frames(
        self,
        output_dir: Path,
        sample_rate: int = 1,
        image_format: str = "png",
        limit: Optional[int] = None,
    ) -> list[Path]:
        """Dump frames to disk. Useful for debugging the depth estimator.

        Returns the list of written paths in order.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for i, frame in enumerate(self.frames(sample_rate=sample_rate)):
            path = output_dir / f"frame_{i:06d}.{image_format}"
            cv2.imwrite(str(path), frame)
            written.append(path)
            if limit is not None and len(written) >= limit:
                break
        log.info("Wrote %d frames to %s", len(written), output_dir)
        return written

    # ----------------------------------------------------- source-type helpers

    @staticmethod
    def _looks_like_video_ts(path: Path) -> bool:
        """A VIDEO_TS directory contains VTS_*_*.VOB and VIDEO_TS.IFO."""
        if path.name.upper() == "VIDEO_TS":
            return True
        return (path / "VIDEO_TS").is_dir() or any(path.glob("VTS_*.VOB"))

    def _extract_video_ts(self, path: Path) -> Path:
        """Concatenate the main-feature VOBs into a single remuxed MKV."""
        ensure_tool("ffmpeg")
        vts_dir = path if path.name.upper() == "VIDEO_TS" else path / "VIDEO_TS"
        if not vts_dir.is_dir():
            raise ValueError(f"No VIDEO_TS directory under {path}")

        # Heuristic: pick the title set with the most/largest VOBs as the main feature.
        titles: dict[str, list[Path]] = {}
        for vob in sorted(vts_dir.glob("VTS_*_[1-9].VOB")):
            key = vob.name.split("_")[1]
            titles.setdefault(key, []).append(vob)
        if not titles:
            raise ValueError(f"No feature VOBs found in {vts_dir}")
        main_key = max(titles, key=lambda k: sum(v.stat().st_size for v in titles[k]))
        vobs = sorted(titles[main_key])
        log.info("Selected title set VTS_%s (%d VOBs, %.1f GB)",
                 main_key, len(vobs),
                 sum(v.stat().st_size for v in vobs) / 1024 ** 3)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        concat_list = self.cache_dir / f"vts_{main_key}_concat.txt"
        concat_list.write_text("".join(f"file '{v}'\n" for v in vobs))
        out = self.cache_dir / f"vts_{main_key}_remux.mkv"
        if out.exists() and out.stat().st_mtime > concat_list.stat().st_mtime:
            log.info("Using cached remux: %s", out)
            return out

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", "-map", "0:v:0", "-map", "0:a:0?",
            str(out),
        ]
        log.info("Remuxing VIDEO_TS → %s", out.name)
        subprocess.run(cmd, check=True)
        return out

    def _extract_iso(self, iso_path: Path) -> Path:
        """Pull the main title out of a DVD ISO via ffmpeg's dvdvideo demuxer."""
        ensure_tool("ffmpeg")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        out = self.cache_dir / f"{iso_path.stem}_title1.mkv"
        if out.exists() and out.stat().st_mtime > iso_path.stat().st_mtime:
            log.info("Using cached ISO extraction: %s", out)
            return out

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "dvdvideo", "-title", "1", "-i", str(iso_path),
            "-c", "copy", "-map", "0:v:0", "-map", "0:a:0?",
            str(out),
        ]
        log.info("Extracting DVD title → %s", out.name)
        subprocess.run(cmd, check=True)
        return out

    # ----------------------------------------------------------------- misc

    def _require_capture(self) -> cv2.VideoCapture:
        if self._capture is None:
            raise RuntimeError("VideoReader not opened — call .open() or use `with`")
        return self._capture


@contextmanager
def open_video(source: Path | str, cache_dir: Optional[Path] = None) -> Iterator[VideoReader]:
    """Shorthand context manager."""
    reader = VideoReader(source, cache_dir=cache_dir)
    reader.open()
    try:
        yield reader
    finally:
        reader.close()
