# 2D → 3D DVD Converter — Rust

Native Rust implementation of the stereoscopic converter. Single static binary;
no runtime dependencies beyond `ffmpeg`, `dvdauthor`, and `mkisofs` on `PATH`.

---

## Phase 1 — Core Architecture & Setup

### Project Layout (single-crate first; promote to workspace later if needed)

```
.
├── Cargo.toml
├── Cargo.lock
├── config.default.toml
├── src/
│   ├── main.rs              # CLI entry + pipeline wiring
│   ├── lib.rs               # module declarations + re-exports
│   ├── config.rs            # serde-deserialized settings
│   ├── errors.rs            # thiserror types
│   ├── video_reader.rs      # file / VIDEO_TS / ISO
│   ├── depth.rs             # ort / candle inference (stub)
│   ├── stereo.rs            # disparity warp + inpainting + packing (stub)
│   ├── video_writer.rs      # ffmpeg encode (stub)
│   ├── dvd.rs               # dvdauthor + mkisofs orchestration (stub)
│   └── util.rs              # ffprobe, disk space, logging init
├── models/                  # downloaded ONNX weights (gitignored)
├── tests/
│   └── video_reader.rs      # integration tests
└── benches/
    └── stereo.rs            # criterion benches (Phase 10)
```

### Core Dependencies

```toml
# CLI + config
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
figment = { version = "0.10", features = ["toml", "env"] }

# Errors + logging
anyhow = "1"              # app-level errors
thiserror = "2"           # library-level errors
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
tracing-appender = "0.2"

# Imaging / math (Phase 4+)
image = "0.25"
imageproc = "0.25"
ndarray = { version = "0.16", features = ["rayon"] }
rayon = "1"

# ML inference (Phase 3)
ort = { version = "2", features = ["cuda", "load-dynamic"] }

# UX / misc
indicatif = "0.17"
tempfile = "3"
sha2 = "0.10"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
assert_cmd = "2"
```

### System Dependencies (unchanged)

`ffmpeg`, `ffprobe`, `dvdauthor`, `mkisofs`/`genisoimage` on `PATH`.

---

## Phase 2 — Video Input & Extraction

### Decision: `ffmpeg-next` vs shelling out

|                    | `ffmpeg-next` crate | shell-out to `ffmpeg` |
|--------------------|---------------------|-----------------------|
| Build friction     | Needs matching libav headers; brittle on new distros | None |
| Perf               | No subprocess overhead | Negligible for whole-file ops |
| Control            | Per-packet, per-frame | Coarse (full-stream in/out) |
| Debugging          | Rust stack traces | Reproducible by hand on the shell |

**Recommendation**: shell out for Phase 2 (VIDEO_TS concat, ISO extraction, metadata via
`ffprobe`). Only reach for `ffmpeg-next` later if we need zero-copy frame pipelining into
the encoder.

### `video_reader.rs` API

```rust
pub struct VideoReader { /* source, resolved file, metadata, cache_dir */ }

impl VideoReader {
    pub fn open(source: &Path, cache: &Path) -> Result<Self>;
    pub fn metadata(&self) -> &VideoMetadata;
    pub fn frames(&self, opts: FrameOpts) -> FrameIter;
    pub fn get_frame(&self, index: u64) -> Result<Frame>;
    pub fn extract_frames(&self, out: &Path, opts: FrameOpts, limit: Option<usize>) -> Result<Vec<PathBuf>>;
}

pub struct Frame { pub index: u64, pub width: u32, pub height: u32, pub bgr: Vec<u8> }
pub struct FrameOpts { pub sample_rate: u32, pub start: u64, pub end: Option<u64> }
```

**Decoding path**: spawn `ffmpeg -i <src> -f rawvideo -pix_fmt bgr24 -` once, read fixed
`(w*h*3)`-byte chunks from stdout. Simple, fast, avoids `ffmpeg-next` build pain.

**VIDEO_TS / ISO**: same heuristic as the Python version — pick the largest title set,
concat VOBs via `ffmpeg -f concat`, cache the remuxed MKV under `.cache/`. For ISOs use
`-f dvdvideo -title 1`.

### Metadata via `ffprobe` + `serde_json`

Deserialize `ffprobe -print_format json` output into typed structs. Direct translation of
the Python `get_video_metadata`.

---

## Phase 3 — Depth Estimation (the real decision point)

### Backend Comparison

|                        | `ort` (ONNX Runtime)                              | `candle`                                | `tch` (libtorch)                |
|------------------------|---------------------------------------------------|-----------------------------------------|---------------------------------|
| Model availability     | Any model you can export to ONNX                  | Needs Rust-side model code (DAv2 in-repo) | Any TorchScript model         |
| GPU support            | CUDA, DirectML, CoreML, TensorRT via EPs          | CUDA, Metal                             | Full libtorch (CUDA, ROCm)      |
| Build cost             | Dynamic lib — small                               | Pure Rust, longer compile               | Pulls in libtorch (~2 GB)       |
| Pre-processing         | Manual (ndarray)                                  | Native tensors                          | Native tensors                  |
| Weight distribution    | Just an `.onnx` file (first-run download)         | HF Hub integration built in             | Download + convert to TorchScript |

**Recommendation**: `ort` + **Depth Anything V2** ONNX. Fastest to get working, broadest
hardware support, single-file model. Reserve the right to swap to `candle` later for a
pure-Rust / zero-C++ dependency chain.

### `depth.rs` API

```rust
pub struct DepthEstimator { /* ort session, input name, input size, EMA buffer */ }

impl DepthEstimator {
    pub fn load(model: ModelChoice, device: Device) -> Result<Self>;
    pub fn infer(&mut self, frame: &Frame) -> Result<Array2<f32>>;        // normalized [0,1]
    pub fn infer_batch(&mut self, frames: &[Frame]) -> Result<Vec<Array2<f32>>>;
}

pub enum ModelChoice { DepthAnythingV2Small, DepthAnythingV2Base, DepthAnythingV2Large, MidasSmall }
```

### Model fetch

First-run download helper with SHA-256 verification into `models/`. Cache + checksum skip
re-download. Honor `STEREOSCOPY_MODEL_DIR` for offline / air-gapped setups.

### Pre/post-processing (ndarray)

- Resize BGR → RGB → f32 → ImageNet mean/std normalization → `[1, 3, H, W]`
- Output `[1, 1, H, W]` → bilinear resize to source resolution → min-max normalize per frame
- Temporal EMA: `depth_t = α·depth_t + (1-α)·depth_{t-1}` (single-frame memory, cheap)

---

## Phase 4 — Stereo View Generation

### `stereo.rs` API

```rust
pub struct StereoGenerator { /* baseline_px, max_disparity, convergence, inpaint */ }

impl StereoGenerator {
    pub fn generate_pair(&self, frame: &Frame, depth: &Array2<f32>) -> (Frame, Frame);
    pub fn pack(&self, left: &Frame, right: &Frame, fmt: OutputFormat) -> Frame;
}

pub enum OutputFormat { SideBySide, TopBottom, Anaglyph, FrameSequential }
pub enum InpaintMethod { Telea, NavierStokes, FastMarching }
```

### Implementation Notes

- **Warping loop**: `rayon` on row iterators — horizontal shift is embarrassingly parallel
  per row. Near-linear scaling to physical cores. This is where Rust meaningfully beats a
  naive Python implementation.
- **Occlusion mask**: pixels no left-eye source pixel landed in → inpaint.
- **Inpainting**: port Telea's fast-marching inpaint ourselves (~200 lines) OR add the
  `opencv` crate behind a feature flag for parity with Python. Default lean: pure port.
- **Anaglyph**: `R = left.R`, `G = right.G`, `B = right.B` — trivial channel multiplex.

---

## Phase 5 — Video Encoding

### `video_writer.rs` — stdin pipe to `ffmpeg`

```rust
pub struct VideoWriter { /* ffmpeg Child, stdin pipe, size, fps */ }

impl VideoWriter {
    pub fn create(path: &Path, size: (u32,u32), fps: f64, opts: EncodeOpts) -> Result<Self>;
    pub fn write_frame(&mut self, frame: &Frame) -> Result<()>;
    pub fn finish(self) -> Result<()>;
}
```

Spawn: `ffmpeg -y -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i - -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4`.
Write raw BGR into stdin, drop → `wait()` on finish.

### Audio Re-mux

Second ffmpeg pass: `-i encoded.mp4 -i source.mkv -c copy -map 0:v:0 -map 1:a:0 final.mkv`.

---

## Phase 6 — DVD Authoring

All subprocess calls — Rust gives nothing new here except a cleaner error type.

```rust
pub struct DvdAuthorer { region: Region, video_bitrate: String, audio_codec: AudioCodec }

impl DvdAuthorer {
    pub fn prepare_video(&self, src: &Path, out: &Path) -> Result<PathBuf>;   // ffmpeg -target ntsc-dvd
    pub fn prepare_audio(&self, src: &Path, out: &Path) -> Result<PathBuf>;   // ffmpeg -c:a ac3
    pub fn build_video_ts(&self, mpeg: &Path, audio: Option<&Path>, out: &Path) -> Result<PathBuf>;
    pub fn build_iso(&self, video_ts: &Path, iso: &Path) -> Result<PathBuf>;
}
```

Specs unchanged (720×480 NTSC / 720×576 PAL, AC3 48 kHz, 4.7 GB / 8.5 GB caps).

---

## Phase 7 — CLI & Config

### `clap` derive

```rust
#[derive(Parser)]
struct Cli {
    input: PathBuf,
    #[arg(short, long, default_value = "output/output.iso")]
    output: PathBuf,
    #[arg(short, long, value_enum, default_value_t = Format::SideBySide)]
    format: Format,
    #[arg(long, value_enum, default_value_t = Model::DepthAnythingV2Small)]
    depth_model: Model,
    #[arg(long, default_value_t = 20)]
    baseline: i32,
    #[arg(long, value_enum, default_value_t = Device::Auto)]
    device: Device,
    #[arg(long)]
    fps: Option<f64>,
    #[arg(long, default_value_t = 1)]
    sample_rate: u32,
    #[arg(long)]
    probe_only: bool,
    #[arg(long, value_name = "N")]
    extract_frames: Option<u32>,
    #[arg(short, long)]
    verbose: bool,
}
```

### Layered config via `figment`

Precedence: CLI flags → env vars (`STEREOSCOPY_*`) → user `~/.config/stereoscopy/config.toml`
→ bundled `config.default.toml`. Each layer deserializes into the same `Config` struct.

---

## Phase 8 — Errors & Observability

### Error types (`thiserror`)

```rust
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("input not found: {0}")]
    InputMissing(PathBuf),
    #[error("required binary {0} not found on PATH")]
    MissingBinary(&'static str),
    #[error("ffmpeg failed (exit {0}): {1}")]
    FfmpegFailed(i32, String),
    #[error("depth model inference failed: {0}")]
    DepthInference(String),
}
```

### `tracing`

- Console + rolling file subscriber (`tracing-appender`)
- Spans around phases: `info_span!("depth", frame = i)` — grep-friendly
- `RUST_LOG=stereoscopy=debug` toggles verbosity

### Progress

`indicatif::ProgressBar` wrapped around the frame iterator. Multibar for depth + stereo +
encode when running concurrently.

---

## Phase 9 — Testing

### Unit tests (`#[cfg(test)]` inline)

- Metadata parsing against a captured ffprobe JSON fixture
- Stereo warping on a synthetic depth map with known expected shifts
- Anaglyph channel multiplex: 1-pixel assert

### Integration tests (`tests/`)

- Generate a `testsrc` clip via `ffmpeg -f lavfi` (same as Python)
- Full pipeline on a 30-frame 720p clip end-to-end
- Assert the output MP4 opens with correct SBS dimensions

### Benches (`criterion`)

- `stereo::generate_pair` on 1080p — track regressions in the hot loop
- `depth::infer` latency, CPU vs CUDA

---

## Phase 10 — Performance

Where Rust meaningfully beats Python:

| Stage                        | Expected speedup                                |
|------------------------------|-------------------------------------------------|
| Depth inference              | ~0 % (GPU / ORT bound regardless)               |
| Stereo warping + inpainting  | 10–30× (rayon-parallel vs single-threaded NumPy)|
| Frame I/O orchestration      | 2–5× (no GIL, zero-copy `Vec<u8>`)              |
| Startup / model load         | 5–20× (no Python interpreter boot)              |

### Pipeline parallelism (channels)

```
reader ──frames──▶ depth ──(frame, depth)──▶ stereo ──SBS frame──▶ encoder
       crossbeam channel        crossbeam channel          crossbeam channel
```

Bounded channels back-pressure cleanly without unbounded memory growth.

### Memory budget

Target: < 2 GB RSS even on a 2-hour source. Channel capacities of ~30 frames each cap
in-flight buffers.

---

## Phase 11 — Distribution

### Release build

`cargo build --release --features cuda` → single ~15 MB binary + the ONNX Runtime shared lib.

### Cross-compile targets

- `x86_64-unknown-linux-gnu` (primary)
- `aarch64-apple-darwin` (CoreML EP for depth; swap `cuda` feature for `coreml`)
- Windows via GitHub Actions matrix

### Packaging

Later: `cargo-dist` for GitHub Releases, or a Nix flake if fully reproducible builds
become a requirement.

---

## Implementation Roadmap

### Week 1 — Foundation
- [ ] `cargo new`, Cargo.toml, CI skeleton
- [ ] `tracing` + `clap` + `figment` wired
- [ ] `util::ffprobe()` + `ensure_tool()`

### Week 2 — Video I/O
- [ ] `VideoReader` for files via `ffmpeg -f rawvideo` pipe
- [ ] VIDEO_TS concat + ISO extraction with `.cache/` reuse
- [ ] `--probe-only` + `--extract-frames` CLI parity with Python version

### Week 3 — Depth
- [ ] `ort` session boot + model download helper
- [ ] Depth Anything V2 Small on a single frame
- [ ] CUDA EP + batched inference
- [ ] Temporal EMA

### Week 4 — Stereo + Encode
- [ ] Rayon-parallel disparity warp
- [ ] Telea inpainting port (or `opencv` crate behind a feature flag)
- [ ] SBS / TB / Anaglyph packing
- [ ] `VideoWriter` stdin-pipe encoder + audio re-mux

### Week 5 — DVD + Polish
- [ ] `DvdAuthorer` three-stage pipeline
- [ ] Pipeline-parallel stages via `crossbeam` channels
- [ ] Criterion benches + CI performance gate
- [ ] Release binary + `cargo-dist`

---

## Key Risks

| Risk                                                         | Mitigation                                                 |
|--------------------------------------------------------------|------------------------------------------------------------|
| `ort` CUDA EP mismatches with system CUDA                    | `load-dynamic` feature; document required `libonnxruntime.so` |
| `ffmpeg-next` build pain on bleeding-edge distros            | Shell out instead — no pure-Rust parity loss              |
| Telea inpainting quality below OpenCV reference              | Feature-flag `opencv` crate fallback                      |
| Depth model fetch needs network at first run                 | Offline mode: honor `STEREOSCOPY_MODEL_DIR`               |
| `ort` + `candle` build time                                  | Keep deps minimal; `cargo chef` in CI                     |
