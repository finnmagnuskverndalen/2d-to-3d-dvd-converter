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
| 4     | Stereo view synthesis (SBS/TB/anaglyph) | done |
| 5     | Video encoding + audio re-mux        | done   |
| 6     | DVD authoring (dvdauthor + mkisofs)  | done   |

All phases from the original plan are implemented. See `RUST_PLAN.md` §Phase 10 for
the still-open optimisation work (pipeline parallelism via crossbeam channels,
criterion benches, release packaging).

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

Generate a stereoscopic preview for one frame (runs depth + stereo warp + packing):

```
# Default side-by-side output at 2x source width
cargo run -- path/to/movie.mp4 --save-stereo-preview 100

# Red/cyan anaglyph for quick viewing with glasses
cargo run -- path/to/movie.mp4 --save-stereo-preview 100 --format anaglyph

# Tune the stereo effect
cargo run -- path/to/movie.mp4 --save-stereo-preview 100 \
    --baseline 30 --convergence 0.4 --max-disparity 60
```

Formats (`--format`):

| Value              | Output                              | Preview supported? | Video supported? |
|--------------------|-------------------------------------|--------------------|------------------|
| `side-by-side`     | `[left | right]`, width × 2         | yes                | yes              |
| `top-bottom`       | `[left / right]`, height × 2        | yes                | yes              |
| `anaglyph`         | Red/cyan single frame               | yes                | yes              |
| `frame-sequential` | Alternating L/R frames in a video   | no                 | not yet          |

## Full-pipeline encode

Running without any `--…-preview` or `--probe-only` flag kicks off the full
stereoscopic encode: every source frame is passed through depth → stereo pair →
packing → ffmpeg encoder. If the source has audio, it is re-muxed from the
original into the final container at the end.

```
# End-to-end: encode the whole movie as side-by-side H.264 with audio
cargo run --release -- path/to/movie.mp4 --output output/movie_3d.mkv

# Anaglyph for playback on a standard screen, trimmed to the first 240 frames
cargo run --release -- path/to/movie.mp4 \
    --format anaglyph \
    --frames 240 \
    --output output/preview.mkv

# No audio, lower quality for faster iteration
cargo run --release -- path/to/movie.mp4 \
    --quality low --no-audio --no-progress \
    --output output/draft.mkv
```

Relevant flags:

| Flag                   | Default            | Effect                                                 |
|------------------------|--------------------|--------------------------------------------------------|
| `--output PATH`        | `output/output.mkv`| `.mkv`/`.mp4` → stereo video; `.iso` → authored DVD    |
| `--frames N`           | all frames         | Encode only the first N source frames                  |
| `--sample-rate N`      | 1                  | Keep every Nth frame; output FPS is scaled accordingly |
| `--quality low|medium|high` | high (CRF 18) | H.264 CRF: 28 / 23 / 18                                |
| `--no-audio`           | off                | Skip the audio re-mux pass                             |
| `--no-progress`        | off                | Suppress the progress bar (useful for logs / CI)       |
| `--keep-intermediate`  | off                | Keep the video-only encode after re-muxing audio       |

## DVD authoring

Any `--output` path ending in `.iso` triggers DVD authoring after the stereo
encode: the stereo video is transcoded to a DVD-spec MPEG-2 program stream
(720×480 NTSC / 720×576 PAL, AC3 48 kHz audio), `dvdauthor` builds the
VIDEO_TS tree, and `mkisofs` wraps it into an ISO 9660 image.

```
# Burn-ready NTSC widescreen DVD, side-by-side stereo
cargo run --release -- path/to/movie.mp4 --output output/movie.iso

# PAL, red/cyan anaglyph (best choice for DVD — stays at source resolution)
cargo run --release -- path/to/movie.mp4 \
    --format anaglyph \
    --region pal \
    --output output/movie.iso
```

Relevant DVD flags:

| Flag               | Default      | Values                       |
|--------------------|--------------|------------------------------|
| `--region`         | `ntsc`       | `ntsc` (720×480 @ 29.97) · `pal` (720×576 @ 25) |
| `--dvd-aspect`     | `widescreen` | `widescreen` (16:9) · `standard` (4:3)          |

### DVD caveats

- DVD is strictly 720×480 (NTSC) or 720×576 (PAL). A side-by-side stereoscopic
  source gets downscaled; the 3D effect is preserved but at low resolution.
  For a DVD-only workflow prefer `--format anaglyph` — it keeps full source
  resolution and plays on any TV with red/cyan glasses.
- Max image size is 4.7 GB (single-layer) / 8.5 GB (dual-layer). Long encodes
  may need `--quality low` to fit.
- The DVD authoring integration test is marked `#[ignore]` so routine
  `cargo test` runs stay fast. Run the full chain with
  `cargo test --test dvd_authoring -- --ignored`.

## Logging

- Default: `stereoscopy=info,warn`
- Verbose: `cargo run -- --verbose …`
- Full control: `RUST_LOG=stereoscopy=debug,trace cargo run -- …`
