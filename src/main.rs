//! `stereoscopy` CLI entry point.
//!
//! Phase 1-3 scope: open the input (file / VIDEO_TS / ISO), print metadata, extract
//! frames, and run depth estimation on a single frame via `--save-depth-preview`.
//! Phases 4-6 (stereo synthesis, encoding, DVD authoring) still exit with
//! "not implemented".

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use tracing::{error, info, warn};

use stereoscopy::{
    DepthEstimator, PipelineError, VideoReader, human_duration, init_logging,
    depth,
    video_reader::FrameOpts,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum Format { SideBySide, TopBottom, Anaglyph, FrameSequential }

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum DepthModel { DepthAnythingV2Small, DepthAnythingV2Base, DepthAnythingV2Large, MidasSmall }

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum Device { Auto, Cuda, Cpu }

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum Quality { Low, Medium, High }

#[derive(Parser, Debug)]
#[command(
    name = "stereoscopy",
    version,
    about = "Convert 2D video / DVDs to stereoscopic 3D with optional DVD authoring",
    long_about = None,
)]
struct Cli {
    /// Input video file, VIDEO_TS directory, or DVD .iso
    input: PathBuf,

    /// Output path (extension determines format: .iso / .mp4 / .mkv)
    #[arg(short, long, default_value = "output/output.iso")]
    output: PathBuf,

    #[arg(short, long, value_enum, default_value_t = Format::SideBySide)]
    format: Format,

    #[arg(long, value_enum, default_value_t = DepthModel::DepthAnythingV2Small)]
    depth_model: DepthModel,

    /// Stereo baseline in pixels (horizontal shift budget)
    #[arg(long, default_value_t = 20)]
    baseline: i32,

    #[arg(long, value_enum, default_value_t = Quality::High)]
    quality: Quality,

    #[arg(long, value_enum, default_value_t = Device::Auto)]
    device: Device,

    /// Override output FPS (auto-detect from source if unset)
    #[arg(long)]
    fps: Option<f64>,

    /// Process every Nth frame (1 = every frame)
    #[arg(long, default_value_t = 1)]
    sample_rate: u32,

    #[arg(long)]
    keep_intermediate: bool,

    /// Open the input, print metadata, and exit without processing
    #[arg(long)]
    probe_only: bool,

    /// Dump first N frames to output/frames/ as PNGs and exit
    #[arg(long, value_name = "N")]
    extract_frames: Option<usize>,

    /// Run depth estimation on frame N and write it to output/depth_<N>.png
    #[arg(long, value_name = "N")]
    save_depth_preview: Option<u64>,

    #[arg(short, long)]
    verbose: bool,
}

impl DepthModel {
    fn as_choice(self) -> depth::ModelChoice {
        match self {
            DepthModel::DepthAnythingV2Small => depth::ModelChoice::DepthAnythingV2Small,
            DepthModel::DepthAnythingV2Base  => depth::ModelChoice::DepthAnythingV2Base,
            DepthModel::DepthAnythingV2Large => depth::ModelChoice::DepthAnythingV2Large,
            DepthModel::MidasSmall           => depth::ModelChoice::MidasSmall,
        }
    }
}

impl Device {
    fn as_backend(self) -> depth::Device {
        match self {
            Device::Auto => depth::Device::Auto,
            Device::Cuda => depth::Device::Cuda,
            Device::Cpu  => depth::Device::Cpu,
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    info!(input = %cli.input.display(), "stereoscopy starting");

    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            error!(error = %e, "pipeline failure");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, PipelineError> {
    let cache_dir = PathBuf::from(".cache");
    let output_root = cli.output.parent()
        .map(|p| if p.as_os_str().is_empty() { PathBuf::from(".") } else { p.to_path_buf() })
        .unwrap_or_else(|| PathBuf::from("output"));

    if !output_root.exists() {
        std::fs::create_dir_all(&output_root)?;
    }

    const MIN_FREE_GB: f64 = 10.0;
    if let Ok(free) = stereoscopy::util::disk_free_gb(&output_root) {
        if free < MIN_FREE_GB {
            warn!(free_gb = free, min = MIN_FREE_GB, "low disk space near output");
        }
    }

    let reader = VideoReader::open(&cli.input, &cache_dir)?;
    let md = reader.metadata();
    info!(path = %md.path.display(), "INPUT");
    info!(codec = %md.codec, pix_fmt = %md.pixel_format, audio = md.has_audio, "codec info");
    info!(
        width = md.width, height = md.height, fps = md.fps,
        frames = md.total_frames, duration = %human_duration(md.duration_s),
        "stream info"
    );

    if cli.probe_only {
        return Ok(ExitCode::SUCCESS);
    }

    if let Some(n) = cli.extract_frames {
        let out = PathBuf::from("output/frames");
        let paths = reader.extract_frames(
            &out,
            FrameOpts { sample_rate: cli.sample_rate, start: 0, end: None },
            Some(n),
        )?;
        info!(count = paths.len(), dir = %out.display(), "extracted frames");
        return Ok(ExitCode::SUCCESS);
    }

    if let Some(frame_idx) = cli.save_depth_preview {
        let frame = reader.get_frame(frame_idx)?;
        info!(frame = frame_idx, model = ?cli.depth_model, "running depth estimation");
        let mut estimator = DepthEstimator::load(cli.depth_model.as_choice(), cli.device.as_backend())?;
        let depth_map = estimator.infer(&frame)?;
        let out_path = PathBuf::from(format!("output/depth_{frame_idx:06}.png"));
        depth_map.save_png(&out_path)?;
        info!(path = %out_path.display(), "wrote depth preview");
        return Ok(ExitCode::SUCCESS);
    }

    error!("pipeline stages beyond depth estimation are not yet implemented");
    error!("use --probe-only, --extract-frames N, or --save-depth-preview N");
    Ok(ExitCode::from(2))
}
