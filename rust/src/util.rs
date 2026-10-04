//! Shared helpers: logging init, ffprobe, disk-space + binary checks.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

use crate::errors::{PipelineError, Result};

// ------------------------------------------------------------------ logging

/// Install a tracing subscriber (console + optional file) exactly once.
pub fn init_logging(verbose: bool) {
    let default_filter = if verbose { "stereoscopy=debug,info" } else { "stereoscopy=info,warn" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));

    // `.try_init()` is a no-op if a subscriber is already installed (e.g. in tests).
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true).with_level(true))
        .try_init();
}

// --------------------------------------------------------------- ffprobe

#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub path: PathBuf,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
    pub duration_s: f64,
    pub total_frames: u64,
    pub codec: String,
    pub pixel_format: String,
    pub has_audio: bool,
}

#[derive(Deserialize)]
struct FfprobeOut {
    streams: Vec<Stream>,
    format: Format,
}

#[derive(Deserialize)]
struct Stream {
    codec_type: String,
    #[serde(default)]
    codec_name: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    pix_fmt: Option<String>,
    #[serde(default)]
    avg_frame_rate: Option<String>,
    #[serde(default)]
    r_frame_rate: Option<String>,
    #[serde(default)]
    nb_frames: Option<String>,
}

#[derive(Deserialize)]
struct Format {
    #[serde(default)]
    duration: Option<String>,
}

/// Parse ffmpeg's fractional rate strings (e.g. "24000/1001" → 23.976).
fn parse_rate(raw: &str) -> f64 {
    if let Some((num, den)) = raw.split_once('/') {
        let n: f64 = num.parse().unwrap_or(0.0);
        let d: f64 = den.parse().unwrap_or(0.0);
        if d != 0.0 { n / d } else { 0.0 }
    } else {
        raw.parse().unwrap_or(0.0)
    }
}

pub fn get_video_metadata(path: &Path) -> Result<VideoMetadata> {
    if !path.exists() {
        return Err(PipelineError::InputMissing(path.to_path_buf()));
    }
    ensure_tool("ffprobe")?;

    let output = Command::new("ffprobe")
        .args(["-v", "error", "-print_format", "json", "-show_format", "-show_streams"])
        .arg(path)
        .output()?;

    if !output.status.success() {
        return Err(PipelineError::FfprobeFailed {
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    let parsed: FfprobeOut = serde_json::from_slice(&output.stdout)?;

    let video = parsed
        .streams
        .iter()
        .find(|s| s.codec_type == "video")
        .ok_or_else(|| PipelineError::NoVideoStream(path.to_path_buf()))?;
    let has_audio = parsed.streams.iter().any(|s| s.codec_type == "audio");

    let rate_raw = video.avg_frame_rate.as_deref().unwrap_or(
        video.r_frame_rate.as_deref().unwrap_or("0/1")
    );
    let fps = parse_rate(rate_raw);

    let duration_s = parsed.format.duration.as_deref().and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let total_frames = match video.nb_frames.as_deref() {
        Some(s) if s != "N/A" => s.parse().unwrap_or_else(|_| (duration_s * fps).round() as u64),
        _ => (duration_s * fps).round() as u64,
    };

    Ok(VideoMetadata {
        path: path.to_path_buf(),
        fps,
        width: video.width.unwrap_or(0),
        height: video.height.unwrap_or(0),
        duration_s,
        total_frames,
        codec: video.codec_name.clone().unwrap_or_else(|| "unknown".into()),
        pixel_format: video.pix_fmt.clone().unwrap_or_else(|| "unknown".into()),
        has_audio,
    })
}

// ---------------------------------------------------------- disk + tools

pub fn ensure_tool(name: &'static str) -> Result<()> {
    // Portable `which` via env PATH walk.
    let found = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join(name);
            if candidate.is_file() { Some(candidate) } else { None }
        })
    });
    if found.is_some() {
        Ok(())
    } else {
        Err(PipelineError::MissingBinary(name))
    }
}

#[cfg(unix)]
pub fn disk_free_gb(path: &Path) -> std::io::Result<f64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::mem::MaybeUninit;

    let probe = if path.exists() { path } else { path.parent().unwrap_or(Path::new(".")) };
    let c_path = CString::new(probe.as_os_str().as_bytes())?;
    let mut buf = MaybeUninit::<libc::statvfs>::uninit();
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), buf.as_mut_ptr()) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let st = unsafe { buf.assume_init() };
    let free_bytes = st.f_bavail as u64 * st.f_frsize as u64;
    Ok(free_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(not(unix))]
pub fn disk_free_gb(_path: &Path) -> std::io::Result<f64> {
    // Phase 1 scope: Linux-first. Return a conservative value until a Windows backend lands.
    Ok(f64::INFINITY)
}

// --------------------------------------------------------- misc helpers

pub fn human_duration(seconds: f64) -> String {
    let s = seconds as i64;
    let (h, rem) = (s / 3600, s % 3600);
    let (m, s) = (rem / 60, rem % 60);
    format!("{h}:{m:02}:{s:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rate_fraction() {
        assert!((parse_rate("24000/1001") - 23.976023976).abs() < 1e-6);
        assert_eq!(parse_rate("30/1"), 30.0);
        assert_eq!(parse_rate("25"), 25.0);
        assert_eq!(parse_rate("0/0"), 0.0);
    }

    #[test]
    fn human_duration_formats() {
        assert_eq!(human_duration(0.0), "0:00:00");
        assert_eq!(human_duration(65.0), "0:01:05");
        assert_eq!(human_duration(3725.0), "1:02:05");
    }
}
