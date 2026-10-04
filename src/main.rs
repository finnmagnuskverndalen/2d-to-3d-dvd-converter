//! `stereoscopy` CLI entry point.
//!
//! Phase 1-5 scope: open the input (file / VIDEO_TS / ISO), print metadata, extract
//! frames, run depth estimation, generate a packed stereo preview for one frame, and
//! encode a full stereoscopic video with optional audio re-mux. Phase 6 (DVD
//! authoring) still exits with "not implemented" — the output path currently
//! produces .mkv instead.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use tracing::{error, info, warn};

use stereoscopy::{
    DepthEstimator, EncodeOpts, PipelineError, StereoGenerator, VideoReader,
    human_duration, init_logging,
    depth, stereo, video_writer,
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

    /// Output path (.mkv / .mp4 for Phase 5; .iso falls back to .mkv until Phase 6)
    #[arg(short, long, default_value = "output/output.mkv")]
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

    /// Process every Nth frame (output FPS is scaled accordingly to preserve timing)
    #[arg(long, default_value_t = 1)]
    sample_rate: u32,

    /// Cap the full-pipeline encode to the first N source frames (for incremental testing)
    #[arg(long, value_name = "N")]
    frames: Option<usize>,

    /// Skip audio re-mux even if the source has audio
    #[arg(long)]
    no_audio: bool,

    /// Suppress the progress bar
    #[arg(long)]
    no_progress: bool,

    /// Keep intermediate files (video-only encode before audio re-mux)
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

    /// Run depth + stereo on frame N and write the packed result (see --format)
    /// to output/stereo_<N>_<format>.png
    #[arg(long, value_name = "N")]
    save_stereo_preview: Option<u64>,

    /// Convergence plane for stereo synthesis: 0.0 = everything pops out,
    /// 1.0 = everything sits behind the screen. 0.5 (default) centres the scene.
    #[arg(long, default_value_t = 0.5)]
    convergence: f32,

    /// Clamp per-pixel disparity to +/- this many pixels
    #[arg(long, default_value_t = 50)]
    max_disparity: i32,

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

impl Format {
    fn as_output(self) -> stereo::OutputFormat {
        match self {
            Format::SideBySide      => stereo::OutputFormat::SideBySide,
            Format::TopBottom       => stereo::OutputFormat::TopBottom,
            Format::Anaglyph        => stereo::OutputFormat::Anaglyph,
            Format::FrameSequential => stereo::OutputFormat::FrameSequential,
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Format::SideBySide      => "sbs",
            Format::TopBottom       => "tb",
            Format::Anaglyph        => "anaglyph",
            Format::FrameSequential => "fs",
        }
    }

    fn output_size(self, src: (u32, u32)) -> (u32, u32) {
        match self {
            Format::SideBySide      => (src.0 * 2, src.1),
            Format::TopBottom       => (src.0,     src.1 * 2),
            Format::Anaglyph        => src,
            Format::FrameSequential => src,
        }
    }
}

impl Quality {
    fn as_video_quality(self) -> video_writer::Quality {
        match self {
            Quality::Low    => video_writer::Quality::Low,
            Quality::Medium => video_writer::Quality::Medium,
            Quality::High   => video_writer::Quality::High,
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

    if let Some(frame_idx) = cli.save_stereo_preview {
        let frame = reader.get_frame(frame_idx)?;
        info!(frame = frame_idx, model = ?cli.depth_model, "running depth + stereo");
        let mut estimator = DepthEstimator::load(cli.depth_model.as_choice(), cli.device.as_backend())?;
        let depth_map = estimator.infer(&frame)?;

        let generator = StereoGenerator {
            baseline_px: cli.baseline,
            max_disparity: cli.max_disparity,
            convergence: cli.convergence.clamp(0.0, 1.0),
            inpaint: stereo::InpaintMethod::Simple,
        };
        let (left, right) = generator.generate_pair(&frame, &depth_map)?;
        let packed = generator.pack(&left, &right, cli.format.as_output())?;
        let out_path = PathBuf::from(format!(
            "output/stereo_{frame_idx:06}_{}.png",
            cli.format.slug()
        ));
        stereo::write_frame_as_png(&packed, &out_path)?;
        info!(
            path = %out_path.display(),
            size = format!("{}x{}", packed.width, packed.height),
            "wrote stereo preview"
        );
        return Ok(ExitCode::SUCCESS);
    }

    // ---- Full pipeline: encode a stereoscopic video --------------------------

    if matches!(cli.format, Format::FrameSequential) {
        return Err(PipelineError::NotImplemented(
            "frame-sequential encoding — needs per-frame L/R alternation (future iteration)",
        ));
    }

    // Rewrite .iso → .mkv until Phase 6 lands.
    let final_output = if cli.output.extension().and_then(|e| e.to_str()).map_or(false, |e| e.eq_ignore_ascii_case("iso")) {
        let rewritten = cli.output.with_extension("mkv");
        warn!(
            requested = %cli.output.display(),
            using = %rewritten.display(),
            "Phase 5 encodes to .mkv; DVD ISO authoring arrives in Phase 6"
        );
        rewritten
    } else {
        cli.output.clone()
    };

    let has_audio = md.has_audio && !cli.no_audio;
    let encode_target: PathBuf = if has_audio {
        cache_dir.join("stereo_video_only.mkv")
    } else {
        final_output.clone()
    };
    std::fs::create_dir_all(&cache_dir)?;

    let out_size = cli.format.output_size((md.width, md.height));
    let effective_fps = cli.fps.unwrap_or(md.fps / cli.sample_rate.max(1) as f64);

    info!(model = ?cli.depth_model, device = ?cli.device, "loading depth model");
    let mut estimator = DepthEstimator::load(
        cli.depth_model.as_choice(),
        cli.device.as_backend(),
    )?;

    let generator = StereoGenerator {
        baseline_px: cli.baseline,
        max_disparity: cli.max_disparity,
        convergence: cli.convergence.clamp(0.0, 1.0),
        inpaint: stereo::InpaintMethod::Simple,
    };

    let opts = EncodeOpts::default().with_quality(cli.quality.as_video_quality());
    let mut writer = stereoscopy::VideoWriter::create(&encode_target, out_size, effective_fps, opts)?;

    let planned_frames = {
        let after_sampling = (md.total_frames + cli.sample_rate as u64 - 1) / cli.sample_rate.max(1) as u64;
        match cli.frames {
            Some(n) => (n as u64).min(after_sampling),
            None => after_sampling,
        }
    };

    let progress = make_progress_bar(planned_frames, cli.no_progress);
    progress.set_message("encoding stereo video");

    let mut written: u64 = 0;
    for frame_res in reader.frames(FrameOpts { sample_rate: cli.sample_rate, start: 0, end: None })? {
        if let Some(limit) = cli.frames {
            if written as usize >= limit {
                break;
            }
        }
        let frame = frame_res?;
        let depth_map = estimator.infer(&frame)?;
        let (left, right) = generator.generate_pair(&frame, &depth_map)?;
        let packed = generator.pack(&left, &right, cli.format.as_output())?;
        writer.write_frame(&packed)?;
        written += 1;
        progress.inc(1);
    }
    progress.finish_with_message("encode complete");

    let encoded_path = writer.finish()?;
    info!(path = %encoded_path.display(), frames = written, "video encode done");

    if has_audio {
        info!("re-muxing audio from source");
        video_writer::remux_audio(&encoded_path, reader.video_path(), &final_output)?;
        if !cli.keep_intermediate && encoded_path != final_output {
            let _ = std::fs::remove_file(&encoded_path);
        }
    }

    info!(output = %final_output.display(), "pipeline complete");
    Ok(ExitCode::SUCCESS)
}

fn make_progress_bar(total: u64, hidden: bool) -> ProgressBar {
    let pb = if total > 0 {
        ProgressBar::new(total)
    } else {
        ProgressBar::new_spinner()
    };
    if hidden {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    } else {
        pb.set_draw_target(ProgressDrawTarget::stderr());
        pb.enable_steady_tick(Duration::from_millis(200));
        let style = ProgressStyle::with_template(
            "{spinner:.cyan} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("##-");
        pb.set_style(style);
    }
    pb
}
