"""2D → 3D DVD converter — CLI entry point.

Current status: Phase 1-2 complete. Running the pipeline currently exercises
the video reader (file / VIDEO_TS / ISO) and prints metadata. Depth estimation,
stereo synthesis, encoding, and DVD authoring stages raise NotImplementedError
until their respective phases land.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import CACHE_DIR, DEFAULT_CONFIG, LOG_DIR, OUTPUT_DIR
from src.utils import check_disk_space, human_duration, setup_logging
from src.video_reader import VideoReader


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stereoscopy",
        description="Convert a 2D video / DVD to 3D (stereoscopic) with optional DVD authoring.",
    )
    p.add_argument("input", type=Path, help="Input video file, VIDEO_TS folder, or DVD .iso")
    p.add_argument("-o", "--output", type=Path, default=OUTPUT_DIR / "output.iso",
                   help="Output path (ISO, MP4, MKV — format inferred from extension)")
    p.add_argument("-f", "--format",
                   choices=["side-by-side", "top-bottom", "anaglyph", "frame-sequential"],
                   default=DEFAULT_CONFIG["video"]["output_format"])
    p.add_argument("--depth-model",
                   choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                   default=DEFAULT_CONFIG["depth"]["model"])
    p.add_argument("--baseline", type=int, default=DEFAULT_CONFIG["stereo"]["baseline_px"],
                   help="Stereo baseline in pixels (horizontal shift budget)")
    p.add_argument("--quality", choices=["low", "medium", "high"], default="high")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"],
                   default=DEFAULT_CONFIG["depth"]["device"])
    p.add_argument("--fps", type=float, default=None, help="Override output FPS (auto-detect if unset)")
    p.add_argument("--sample-rate", type=int, default=DEFAULT_CONFIG["runtime"]["sample_rate"],
                   help="Process every Nth frame (1 = every frame)")
    p.add_argument("--keep-intermediate", action="store_true")
    p.add_argument("--probe-only", action="store_true",
                   help="Open the input, print metadata, and exit without processing")
    p.add_argument("--extract-frames", type=int, metavar="N",
                   help="Dump first N frames to output/frames/ as PNGs and exit (debug helper)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = setup_logging(LOG_DIR, level=logging.DEBUG if args.verbose else logging.INFO)
    log.info("stereoscopy starting — input=%s", args.input)

    ok, free_gb = check_disk_space(args.output.parent if args.output.parent.exists() else OUTPUT_DIR,
                                   DEFAULT_CONFIG["runtime"]["min_free_gb"])
    if not ok:
        log.warning("Only %.1f GB free near output — recommended >= %d GB",
                    free_gb, DEFAULT_CONFIG["runtime"]["min_free_gb"])

    try:
        with VideoReader(args.input, cache_dir=CACHE_DIR) as vr:
            md = vr.metadata
            log.info("INPUT  : %s", md.path)
            log.info("        codec=%s  pix_fmt=%s  audio=%s", md.codec, md.pixel_format, md.has_audio)
            log.info("        %dx%d @ %.3f fps  →  %d frames, %s",
                     md.width, md.height, md.fps, md.total_frames, human_duration(md.duration_s))

            if args.probe_only:
                return 0

            if args.extract_frames:
                out = OUTPUT_DIR / "frames"
                vr.extract_frames(out, sample_rate=args.sample_rate, limit=args.extract_frames)
                return 0

            log.error("Pipeline stages beyond VideoReader are not yet implemented.")
            log.error("Use --probe-only to inspect inputs, or --extract-frames N for a frame dump.")
            return 2

    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    except Exception as e:
        log.exception("Unexpected failure: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
