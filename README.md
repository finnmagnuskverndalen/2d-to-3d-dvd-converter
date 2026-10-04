# stereoscopy — 2D → 3D DVD converter

Convert 2D video sources (video files, VIDEO_TS folders, DVD ISOs) into stereoscopic 3D
and optionally author a playable 3D DVD.

## Status

| Phase | Feature | Status |
|-------|-------------------------------------|--------|
| 1     | Project scaffolding                 | done   |
| 2     | VideoReader (file / VIDEO_TS / ISO) | done   |
| 3     | Depth estimation (MiDaS)            | stub   |
| 4     | Stereo view synthesis               | stub   |
| 5     | Video encoding                      | stub   |
| 6     | DVD authoring                       | stub   |

## System dependencies

```
sudo apt install ffmpeg dvdauthor genisoimage
```

## Python setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Probe an input (works today):

```
python main.py path/to/movie.mp4 --probe-only
python main.py path/to/VIDEO_TS/   --probe-only
python main.py path/to/movie.iso   --probe-only
```

Dump the first 50 frames as PNGs (useful for testing depth models by hand):

```
python main.py path/to/movie.mp4 --extract-frames 50
```

Run the smoke test for the video reader:

```
python tests/test_video_reader.py
```
