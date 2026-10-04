"""Default configuration for the 2D→3D converter pipeline.

Values here are intended as sensible starting points. CLI arguments and an
optional user `config.yaml` override these at runtime (see main.py).
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"


DEFAULT_CONFIG: dict = {
    "video": {
        "output_format": "side-by-side",  # side-by-side | top-bottom | anaglyph | frame-sequential
        "codec": "libx264",
        "bitrate": "10000k",
        "container": "mp4",
    },
    "depth": {
        "model": "MiDaS_small",           # DPT_Large | DPT_Hybrid | MiDaS_small
        "device": "auto",                 # auto | cuda | cpu
        "batch_size": 4,
        "temporal_smoothing": True,
        "smoothing_alpha": 0.5,
    },
    "stereo": {
        "baseline_px": 20,                # horizontal shift budget in pixels
        "max_disparity": 50,
        "inpainting_method": "telea",     # telea | navier_stokes
        "convergence": 0.5,               # 0.0 = all pop-out, 1.0 = all push-in
    },
    "dvd": {
        "region": "NTSC",                 # NTSC (720x480@29.97) | PAL (720x576@25)
        "fps_override": None,
        "audio_codec": "ac3",             # ac3 | mp2
        "video_bitrate": "6000k",
        "audio_bitrate": "192k",
    },
    "runtime": {
        "sample_rate": 1,                 # process every Nth frame
        "chunk_size": 30,                 # frames held in memory at once
        "keep_intermediate": False,
        "min_free_gb": 10,
    },
}
