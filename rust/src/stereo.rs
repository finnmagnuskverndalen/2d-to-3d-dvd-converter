//! Stereo view synthesis (Phase 4 — not yet implemented).

use crate::depth::DepthMap;
use crate::errors::{PipelineError, Result};
use crate::video_reader::Frame;

#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    SideBySide,
    TopBottom,
    Anaglyph,
    FrameSequential,
}

#[derive(Debug, Clone, Copy)]
pub enum InpaintMethod {
    Telea,
    NavierStokes,
}

pub struct StereoGenerator {
    pub baseline_px: i32,
    pub max_disparity: i32,
    pub convergence: f32,
    pub inpaint: InpaintMethod,
}

impl StereoGenerator {
    pub fn generate_pair(&self, _frame: &Frame, _depth: &DepthMap) -> Result<(Frame, Frame)> {
        Err(PipelineError::NotImplemented("stereo::generate_pair (Phase 4)"))
    }

    pub fn pack(&self, _left: &Frame, _right: &Frame, _fmt: OutputFormat) -> Result<Frame> {
        Err(PipelineError::NotImplemented("stereo::pack (Phase 4)"))
    }
}
