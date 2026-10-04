//! Pipeline configuration. For Phase 1 this is just a serde-deserializable struct
//! with built-in defaults. Layered loading (CLI → env → user TOML → bundled default)
//! lands in Phase 7 via `figment`.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub video: VideoConfig,
    #[serde(default)]
    pub depth: DepthConfig,
    #[serde(default)]
    pub stereo: StereoConfig,
    #[serde(default)]
    pub dvd: DvdConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            video: VideoConfig::default(),
            depth: DepthConfig::default(),
            stereo: StereoConfig::default(),
            dvd: DvdConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VideoConfig {
    pub output_format: String,
    pub codec: String,
    pub bitrate: String,
    pub container: String,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            output_format: "side-by-side".into(),
            codec: "libx264".into(),
            bitrate: "10000k".into(),
            container: "mp4".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DepthConfig {
    pub model: String,
    pub device: String,
    pub batch_size: usize,
    pub temporal_smoothing: bool,
    pub smoothing_alpha: f32,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            model: "depth-anything-v2-small".into(),
            device: "auto".into(),
            batch_size: 4,
            temporal_smoothing: true,
            smoothing_alpha: 0.5,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct StereoConfig {
    pub baseline_px: i32,
    pub max_disparity: i32,
    pub convergence: f32,
    pub inpainting_method: String,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            baseline_px: 20,
            max_disparity: 50,
            convergence: 0.5,
            inpainting_method: "telea".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DvdConfig {
    pub region: String,
    pub audio_codec: String,
    pub video_bitrate: String,
    pub audio_bitrate: String,
}

impl Default for DvdConfig {
    fn default() -> Self {
        Self {
            region: "NTSC".into(),
            audio_codec: "ac3".into(),
            video_bitrate: "6000k".into(),
            audio_bitrate: "192k".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeConfig {
    pub sample_rate: u32,
    pub chunk_size: usize,
    pub keep_intermediate: bool,
    pub min_free_gb: f64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 1,
            chunk_size: 30,
            keep_intermediate: false,
            min_free_gb: 10.0,
        }
    }
}
