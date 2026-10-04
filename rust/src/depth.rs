//! Monocular depth estimation (Phase 3 — not yet implemented).
//!
//! Planned backend: `ort` (ONNX Runtime) with Depth Anything V2 Small as the default
//! model. See `RUST_PLAN.md` §Phase 3 for the backend trade-off analysis.

use crate::errors::{PipelineError, Result};
use crate::video_reader::Frame;

#[derive(Debug, Clone, Copy)]
pub enum ModelChoice {
    DepthAnythingV2Small,
    DepthAnythingV2Base,
    DepthAnythingV2Large,
    MidasSmall,
}

#[derive(Debug, Clone, Copy)]
pub enum Device {
    Auto,
    Cuda,
    Cpu,
}

/// Normalised depth map: `(width, height, Vec<f32>)` with values in `[0, 1]`,
/// row-major, near = 1.0, far = 0.0. Promoted to `ndarray::Array2<f32>` in Phase 3.
#[derive(Debug, Clone)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

pub struct DepthEstimator;

impl DepthEstimator {
    pub fn load(_model: ModelChoice, _device: Device) -> Result<Self> {
        Err(PipelineError::NotImplemented("depth::DepthEstimator::load (Phase 3)"))
    }

    pub fn infer(&mut self, _frame: &Frame) -> Result<DepthMap> {
        Err(PipelineError::NotImplemented("depth::infer (Phase 3)"))
    }

    pub fn infer_batch(&mut self, _frames: &[Frame]) -> Result<Vec<DepthMap>> {
        Err(PipelineError::NotImplemented("depth::infer_batch (Phase 3)"))
    }
}
