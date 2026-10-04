//! 2D → 3D video / DVD conversion pipeline (Rust port).

pub mod config;
pub mod depth;
pub mod dvd;
pub mod errors;
pub mod stereo;
pub mod util;
pub mod video_reader;
pub mod video_writer;

pub use errors::{PipelineError, Result};
pub use util::{VideoMetadata, get_video_metadata, human_duration, init_logging};
pub use video_reader::{Frame, FrameOpts, VideoReader};
