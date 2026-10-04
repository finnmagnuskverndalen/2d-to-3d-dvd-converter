"""DVD authoring: encode to MPEG-2, build VIDEO_TS, wrap in ISO (Phase 6)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional


log = logging.getLogger("stereoscopy.dvd")

Region = Literal["NTSC", "PAL"]


class DVDAuthorer:
    def __init__(self, region: Region = "NTSC", video_bitrate: str = "6000k", audio_codec: str = "ac3"):
        self.region = region
        self.video_bitrate = video_bitrate
        self.audio_codec = audio_codec

    def prepare_video(self, video_file: Path, out_dir: Path) -> Path:
        """Transcode to DVD-spec MPEG-2 (720x480 @ 29.97 NTSC / 720x576 @ 25 PAL)."""
        raise NotImplementedError("Phase 6: ffmpeg -target ntsc-dvd / pal-dvd")

    def prepare_audio(self, source_media: Path, out_dir: Path) -> Path:
        """Transcode source audio to AC3 at 48 kHz."""
        raise NotImplementedError("Phase 6: ffmpeg -c:a ac3 -ar 48000")

    def build_video_ts(self, mpeg_video: Path, audio: Optional[Path], out_dir: Path) -> Path:
        """Run `dvdauthor` to produce a VIDEO_TS directory structure."""
        raise NotImplementedError("Phase 6: dvdauthor -o VIDEO_TS/ -t video.mpg [audio.ac3]")

    def build_iso(self, video_ts_dir: Path, iso_output: Path) -> Path:
        """Wrap VIDEO_TS into a bootable DVD ISO using mkisofs/genisoimage."""
        raise NotImplementedError("Phase 6: mkisofs -dvd-video -o output.iso VIDEO_TS/")
