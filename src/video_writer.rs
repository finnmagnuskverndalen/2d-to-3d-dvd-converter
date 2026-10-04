//! Encode stereo frames into a video via ffmpeg stdin pipe (Phase 5 — not yet implemented).

use std::path::{Path, PathBuf};

use crate::errors::{PipelineError, Result};
use crate::video_reader::Frame;

pub struct EncodeOpts {
    pub codec: String,
    pub bitrate: String,
    pub crf: Option<u8>,
    pub preset: String,
}

impl Default for EncodeOpts {
    fn default() -> Self {
        Self {
            codec: "libx264".into(),
            bitrate: "10000k".into(),
            crf: Some(18),
            preset: "medium".into(),
        }
    }
}

pub struct VideoWriter;

impl VideoWriter {
    pub fn create(_path: &Path, _size: (u32, u32), _fps: f64, _opts: EncodeOpts) -> Result<Self> {
        Err(PipelineError::NotImplemented("video_writer::create (Phase 5)"))
    }

    pub fn write_frame(&mut self, _frame: &Frame) -> Result<()> {
        Err(PipelineError::NotImplemented("video_writer::write_frame (Phase 5)"))
    }

    pub fn finish(self) -> Result<()> {
        Err(PipelineError::NotImplemented("video_writer::finish (Phase 5)"))
    }

    pub fn remux_audio(_video: &Path, _source_media: &Path, _out: &Path) -> Result<PathBuf> {
        Err(PipelineError::NotImplemented("video_writer::remux_audio (Phase 5)"))
    }
}
