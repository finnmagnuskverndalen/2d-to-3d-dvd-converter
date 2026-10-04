# stereoscopy

Convert 2D video sources (video files, VIDEO_TS folders, DVD ISOs) into stereoscopic
3D and optionally author a playable 3D DVD. Native Rust implementation; see
[`RUST_PLAN.md`](RUST_PLAN.md) for the full design.

## Status

| Phase | Feature                              | Status |
|-------|--------------------------------------|--------|
| 1     | Project scaffolding, CLI, logging    | done   |
| 2     | VideoReader (file / VIDEO_TS / ISO)  | done   |
| 3     | Depth estimation (ONNX Runtime)      | done   |
| 4     | Stereo view synthesis                | stub   |
| 5     | Video encoding                       | stub   |
| 6     | DVD authoring                        | stub   |

## System dependencies

```
sudo apt install ffmpeg dvdauthor genisoimage
```

## Build

```
cargo build                   # dev build (CPU only)
cargo build --release         # optimized
cargo build --features cuda   # enable CUDA execution provider (requires CUDA runtime)
cargo test                    # unit + integration tests
```

The first build downloads the ONNX Runtime binary (~50 MB) via `ort`'s
`download-binaries` feature, so expect a few extra minutes on the initial compile.

## Depth models

Depth estimation requires an ONNX-exported depth model. Weights are **not** shipped
— download one of the variants below and place the `.onnx` file in `models/` (or set
`STEREOSCOPY_MODEL_DIR` to a directory containing it).

| CLI value               | Expected filename                     | Suggested source                                                    |
|-------------------------|---------------------------------------|---------------------------------------------------------------------|
| `depth-anything-v2-small` *(default)* | `depth_anything_v2_vits.onnx` | <https://huggingface.co/onnx-community/depth-anything-v2-small> |
| `depth-anything-v2-base`  | `depth_anything_v2_vitb.onnx`         | <https://huggingface.co/onnx-community/depth-anything-v2-base>  |
| `depth-anything-v2-large` | `depth_anything_v2_vitl.onnx`         | <https://huggingface.co/onnx-community/depth-anything-v2-large> |
| `midas-small`             | `midas_small.onnx`                    | <https://github.com/isl-org/MiDaS>                              |

If a model is missing at runtime, the CLI emits a `ModelMissing` error naming
the expected path and the download URL.

## Quick start

Probe an input (metadata only — works without any model):

```
cargo run -- path/to/movie.mp4   --probe-only
cargo run -- path/to/VIDEO_TS/   --probe-only
cargo run -- path/to/movie.iso   --probe-only
```

Dump the first 50 frames as PNGs for inspection:

```
cargo run -- path/to/movie.mp4 --extract-frames 50
```

Run depth estimation on a single frame and save the depth map as a grayscale PNG
(near = white, far = black):

```
cargo run -- path/to/movie.mp4 --save-depth-preview 100
# → output/depth_000100.png

# Pick a bigger model + GPU:
cargo run --features cuda -- path/to/movie.mp4 \
    --depth-model depth-anything-v2-large \
    --device cuda \
    --save-depth-preview 100
```

## Logging

- Default: `stereoscopy=info,warn`
- Verbose: `cargo run -- --verbose …`
- Full control: `RUST_LOG=stereoscopy=debug,trace cargo run -- …`
