//! Monocular depth estimation via ONNX Runtime (Phase 3).
//!
//! The default backend is `ort` with Depth Anything V2 Small as the recommended model.
//! Model weights are not shipped — place the `.onnx` file under `models/` (or set
//! `STEREOSCOPY_MODEL_DIR`) and the loader will pick it up.
//!
//! Pipeline per frame:
//!   1. BGR → RGB → resize to the model's native square input
//!   2. ImageNet mean/std normalisation
//!   3. ONNX session run → raw inverse-depth tensor
//!   4. Bilinear resize back to source resolution
//!   5. Min-max normalise to `[0, 1]`
//!   6. Temporal EMA against the previous frame's depth to reduce flicker

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ndarray::Array4;
use ort::ep::ExecutionProviderDispatch;
use ort::inputs;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use tracing::info;

use crate::errors::{PipelineError, Result};
use crate::video_reader::Frame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Static metadata for each supported model variant.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub choice: ModelChoice,
    pub filename: &'static str,
    pub input_size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub source_hint: &'static str,
}

impl ModelMeta {
    pub fn for_choice(choice: ModelChoice) -> Self {
        match choice {
            ModelChoice::DepthAnythingV2Small => Self {
                choice,
                filename: "depth_anything_v2_vits.onnx",
                input_size: 518,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
                source_hint: "https://huggingface.co/onnx-community/depth-anything-v2-small",
            },
            ModelChoice::DepthAnythingV2Base => Self {
                choice,
                filename: "depth_anything_v2_vitb.onnx",
                input_size: 518,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
                source_hint: "https://huggingface.co/onnx-community/depth-anything-v2-base",
            },
            ModelChoice::DepthAnythingV2Large => Self {
                choice,
                filename: "depth_anything_v2_vitl.onnx",
                input_size: 518,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
                source_hint: "https://huggingface.co/onnx-community/depth-anything-v2-large",
            },
            ModelChoice::MidasSmall => Self {
                choice,
                filename: "midas_small.onnx",
                input_size: 256,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
                source_hint: "https://github.com/isl-org/MiDaS",
            },
        }
    }
}

/// Normalised depth map at the frame's native resolution. Near = 1.0, far = 0.0.
#[derive(Debug, Clone)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

impl DepthMap {
    /// Save as an 8-bit grayscale PNG. Useful for eyeballing model output.
    pub fn save_png(&self, path: &Path) -> Result<()> {
        let mut img = image::GrayImage::new(self.width, self.height);
        for (i, &v) in self.data.iter().enumerate() {
            let x = (i as u32) % self.width;
            let y = (i as u32) / self.width;
            let byte = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel(x, y, image::Luma([byte]));
        }
        img.save(path)?;
        Ok(())
    }
}

pub struct DepthEstimator {
    session: Session,
    meta: ModelMeta,
    ema_alpha: f32,
    prev_depth: Option<Vec<f32>>,
}

impl DepthEstimator {
    pub fn load(choice: ModelChoice, device: Device) -> Result<Self> {
        let meta = ModelMeta::for_choice(choice);
        let path = resolve_model_path(&meta)?;

        static INIT: OnceLock<()> = OnceLock::new();
        INIT.get_or_init(|| {
            let _ = ort::init().with_name("stereoscopy").commit();
        });

        let mut builder = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;

        let providers = execution_providers(device);
        if !providers.is_empty() {
            builder = builder.with_execution_providers(providers).map_err(ort_err)?;
        }

        let session = builder.commit_from_file(&path).map_err(ort_err)?;

        info!(
            model = ?choice,
            path = %path.display(),
            input_size = meta.input_size,
            device = ?device,
            "loaded depth model"
        );

        Ok(Self { session, meta, ema_alpha: 0.5, prev_depth: None })
    }

    pub fn with_temporal_alpha(mut self, alpha: f32) -> Self {
        self.ema_alpha = alpha.clamp(0.0, 1.0);
        self
    }

    pub fn meta(&self) -> &ModelMeta { &self.meta }

    pub fn infer(&mut self, frame: &Frame) -> Result<DepthMap> {
        let input = preprocess(frame, &self.meta);
        let tensor = Tensor::from_array(input).map_err(ort_err)?;
        let outputs = self.session
            .run(inputs![tensor])
            .map_err(ort_err)?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let (model_h, model_w) = infer_output_hw(&shape[..])?;
        let flat: Vec<f32> = data.to_vec();
        if flat.len() != (model_h as usize) * (model_w as usize) {
            return Err(PipelineError::DepthInference(format!(
                "depth output length {} doesn't match shape {}x{}",
                flat.len(), model_w, model_h
            )));
        }

        let resized = resize_bilinear(&flat, model_w, model_h, frame.width, frame.height);
        let normalised = minmax_normalise(resized);

        let smoothed = match &self.prev_depth {
            Some(prev) if prev.len() == normalised.len() => {
                let a = self.ema_alpha;
                normalised.iter().zip(prev.iter()).map(|(&n, &p)| a * n + (1.0 - a) * p).collect()
            }
            _ => normalised,
        };
        self.prev_depth = Some(smoothed.clone());

        Ok(DepthMap { width: frame.width, height: frame.height, data: smoothed })
    }

    pub fn infer_batch(&mut self, frames: &[Frame]) -> Result<Vec<DepthMap>> {
        frames.iter().map(|f| self.infer(f)).collect()
    }
}

// --------------------------------------------------------- helpers

fn ort_err<E: std::fmt::Display>(e: E) -> PipelineError {
    PipelineError::DepthInference(e.to_string())
}

fn model_dir() -> PathBuf {
    std::env::var("STEREOSCOPY_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models"))
}

fn resolve_model_path(meta: &ModelMeta) -> Result<PathBuf> {
    let path = model_dir().join(meta.filename);
    if !path.exists() {
        return Err(PipelineError::ModelMissing {
            path: path.clone(),
            hint: format!(
                "download the ONNX export from {} and place it at the above path \
                 (or set STEREOSCOPY_MODEL_DIR to point at a directory containing it)",
                meta.source_hint
            ),
        });
    }
    Ok(path)
}

fn execution_providers(device: Device) -> Vec<ExecutionProviderDispatch> {
    use ort::ep::CPU;
    match device {
        Device::Cpu => vec![CPU::default().build()],
        Device::Cuda | Device::Auto => {
            #[cfg(feature = "cuda")]
            {
                use ort::ep::CUDA;
                vec![CUDA::default().build(), CPU::default().build()]
            }
            #[cfg(not(feature = "cuda"))]
            vec![CPU::default().build()]
        }
    }
}

fn infer_output_hw(shape: &[i64]) -> Result<(u32, u32)> {
    match shape {
        [_, h, w] => Ok((*h as u32, *w as u32)),
        [_, _, h, w] => Ok((*h as u32, *w as u32)),
        other => Err(PipelineError::DepthInference(format!(
            "unexpected depth output shape {:?}; expected [1, H, W] or [1, 1, H, W]",
            other
        ))),
    }
}

/// BGR frame → `[1, 3, S, S]` float tensor normalised with ImageNet stats.
fn preprocess(frame: &Frame, meta: &ModelMeta) -> Array4<f32> {
    let s = meta.input_size;

    // Pack BGR into an RGB image so we can lean on the image crate's bilinear resize.
    let mut rgb = image::RgbImage::new(frame.width, frame.height);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = ((y * frame.width + x) * 3) as usize;
            let b = frame.bgr[i];
            let g = frame.bgr[i + 1];
            let r = frame.bgr[i + 2];
            rgb.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    let resized = image::imageops::resize(&rgb, s, s, image::imageops::FilterType::Triangle);

    let mut arr = Array4::<f32>::zeros((1, 3, s as usize, s as usize));
    for y in 0..s {
        for x in 0..s {
            let p = resized.get_pixel(x, y);
            for c in 0..3 {
                let v = (p[c] as f32) / 255.0;
                arr[[0, c, y as usize, x as usize]] = (v - meta.mean[c]) / meta.std[c];
            }
        }
    }
    arr
}

fn resize_bilinear(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut out = vec![0f32; (dw as usize) * (dh as usize)];
    let sx_ratio = sw as f32 / dw as f32;
    let sy_ratio = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = ((y as f32) + 0.5) * sy_ratio - 0.5;
        let y0 = fy.floor().max(0.0) as u32;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - fy.floor();
        for x in 0..dw {
            let fx = ((x as f32) + 0.5) * sx_ratio - 0.5;
            let x0 = fx.floor().max(0.0) as u32;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - fx.floor();

            let p00 = src[(y0 * sw + x0) as usize];
            let p01 = src[(y0 * sw + x1) as usize];
            let p10 = src[(y1 * sw + x0) as usize];
            let p11 = src[(y1 * sw + x1) as usize];

            let top = p00 * (1.0 - wx) + p01 * wx;
            let bot = p10 * (1.0 - wx) + p11 * wx;
            out[(y * dw + x) as usize] = top * (1.0 - wy) + bot * wy;
        }
    }
    out
}

fn minmax_normalise(mut v: Vec<f32>) -> Vec<f32> {
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in &v {
        if x < min { min = x; }
        if x > max { max = x; }
    }
    let range = max - min;
    if range < 1e-6 {
        for x in &mut v { *x = 0.5; }
    } else {
        for x in &mut v { *x = (*x - min) / range; }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_handles_flat_input() {
        let out = minmax_normalise(vec![5.0; 10]);
        assert!(out.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn minmax_scales_to_unit_range() {
        let out = minmax_normalise(vec![0.0, 1.0, 2.0, 4.0]);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn resize_bilinear_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = resize_bilinear(&input, 2, 2, 2, 2);
        assert_eq!(out, input);
    }

    #[test]
    fn resize_bilinear_scales_up() {
        let input = vec![0.0, 1.0, 0.0, 1.0];
        let out = resize_bilinear(&input, 2, 2, 4, 4);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn output_shape_parses_both_rank3_and_rank4() {
        assert_eq!(infer_output_hw(&[1, 100, 200]).unwrap(), (100, 200));
        assert_eq!(infer_output_hw(&[1, 1, 100, 200]).unwrap(), (100, 200));
        assert!(infer_output_hw(&[]).is_err());
        assert!(infer_output_hw(&[100, 200]).is_err());
        assert!(infer_output_hw(&[1, 1, 1, 100, 200]).is_err());
    }
}
