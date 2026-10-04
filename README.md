# stereoscopy

Convert 2D video sources (video files, VIDEO_TS folders, DVD ISOs) into stereoscopic
3D and optionally author a playable 3D DVD. Native Rust implementation; see
[`RUST_PLAN.md`](RUST_PLAN.md) for the full design.

## Status

| Phase | Feature                              | Status |
|-------|--------------------------------------|--------|
| 1     | Project scaffolding, CLI, logging    | done   |
| 2     | VideoReader (file / VIDEO_TS / ISO)  | done   |
| 3     | Depth estimation (ort + DAv2)        | stub   |
| 4     | Stereo view synthesis                | stub   |
| 5     | Video encoding                       | stub   |
| 6     | DVD authoring                        | stub   |

## System dependencies

```
sudo apt install ffmpeg dvdauthor genisoimage
```

## Build

```
cargo build            # dev build
cargo build --release  # optimized
cargo test             # unit + integration tests
```

## Quick start

Probe an input (works today):

```
cargo run -- path/to/movie.mp4   --probe-only
cargo run -- path/to/VIDEO_TS/   --probe-only
cargo run -- path/to/movie.iso   --probe-only
```

Dump the first 50 frames as PNGs:

```
cargo run -- path/to/movie.mp4 --extract-frames 50
```

## Logging

- Default: `stereoscopy=info,warn`
- Verbose: `cargo run -- --verbose ...`
- Full control: `RUST_LOG=stereoscopy=debug,trace cargo run -- ...`
