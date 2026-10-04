//! DVD authoring: MPEG-2 encode → VIDEO_TS → ISO (Phase 6 — not yet implemented).

use std::path::{Path, PathBuf};

use crate::errors::{PipelineError, Result};

#[derive(Debug, Clone, Copy)]
pub enum Region { Ntsc, Pal }

#[derive(Debug, Clone, Copy)]
pub enum AudioCodec { Ac3, Mp2 }

pub struct DvdAuthorer {
    pub region: Region,
    pub video_bitrate: String,
    pub audio_codec: AudioCodec,
}

impl DvdAuthorer {
    pub fn prepare_video(&self, _src: &Path, _out: &Path) -> Result<PathBuf> {
        Err(PipelineError::NotImplemented("dvd::prepare_video (Phase 6)"))
    }

    pub fn prepare_audio(&self, _src: &Path, _out: &Path) -> Result<PathBuf> {
        Err(PipelineError::NotImplemented("dvd::prepare_audio (Phase 6)"))
    }

    pub fn build_video_ts(&self, _mpeg: &Path, _audio: Option<&Path>, _out: &Path) -> Result<PathBuf> {
        Err(PipelineError::NotImplemented("dvd::build_video_ts (Phase 6)"))
    }

    pub fn build_iso(&self, _video_ts: &Path, _iso: &Path) -> Result<PathBuf> {
        Err(PipelineError::NotImplemented("dvd::build_iso (Phase 6)"))
    }
}
