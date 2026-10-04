use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("input not found: {0}")]
    InputMissing(PathBuf),

    #[error("unsupported input: {0} (expected video file, VIDEO_TS dir, or .iso)")]
    UnsupportedInput(PathBuf),

    #[error("required binary '{0}' not found on PATH")]
    MissingBinary(&'static str),

    #[error("ffmpeg failed (exit {code}): {stderr}")]
    FfmpegFailed { code: i32, stderr: String },

    #[error("ffprobe failed (exit {code}): {stderr}")]
    FfprobeFailed { code: i32, stderr: String },

    #[error("ffprobe produced no video stream in {0}")]
    NoVideoStream(PathBuf),

    #[error("frame index {index} out of range [0, {total})")]
    FrameOutOfRange { index: u64, total: u64 },

    #[error("stage not yet implemented: {0}")]
    NotImplemented(&'static str),

    #[error("depth model error: {0}")]
    DepthInference(String),

    #[error("model file missing: {path}. {hint}")]
    ModelMissing { path: PathBuf, hint: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Image(#[from] image::ImageError),
}

pub type Result<T> = std::result::Result<T, PipelineError>;
