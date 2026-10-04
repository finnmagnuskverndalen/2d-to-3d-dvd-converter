//! 2D → 3D video / DVD conversion pipeline (Rust port).

pub mod config;
pub mod depth;
pub mod dvd;
pub mod errors;
pub mod stereo;
pub mod util;
pub mod video_reader;
pub mod video_writer;

pub use depth::{DepthEstimator, DepthMap, Device, ModelChoice};
pub use dvd::{AspectRatio, AudioCodec, DvdAuthorer, Region};
pub use errors::{PipelineError, Result};
pub use stereo::{InpaintMethod, OutputFormat, StereoGenerator};
pub use util::{VideoMetadata, get_video_metadata, human_duration, init_logging};
pub use video_reader::{Frame, FrameOpts, VideoReader};
pub use video_writer::{EncodeOpts, Quality, VideoWriter};
