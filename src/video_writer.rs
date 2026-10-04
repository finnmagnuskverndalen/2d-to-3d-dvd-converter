//! Phase 5: stereoscopic video encoding.
//!
//! Rather than linking `libav` directly we spawn an `ffmpeg` subprocess per encode
//! job and pipe raw BGR frames into its stdin. Keeps the dependency surface tiny
//! and the pipeline reproducible by hand from the shell.
//!
//! The command we spawn is roughly:
//! ```text
//! ffmpeg -y -hide_banner -loglevel error \
//!   -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i - \
//!   -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
//!   out.mkv
//! ```
//!
//! Audio is re-muxed in a second pass (`remux_audio`) so the main encoder stays
//! stateless and we can skip audio entirely with `--no-audio`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::thread::{self, JoinHandle};

use tracing::{debug, info};

use crate::errors::{PipelineError, Result};
use crate::util::ensure_tool;
use crate::video_reader::Frame;

#[derive(Debug, Clone)]
pub struct EncodeOpts {
    pub codec: String,       // libx264 | libx265 | …
    pub crf: Option<u8>,     // 0 (lossless) – 51; mutually exclusive with `bitrate`
    pub bitrate: Option<String>, // e.g. "10000k"
    pub preset: String,      // ultrafast … veryslow
    pub pix_fmt: String,     // typically yuv420p for compatibility
}

impl Default for EncodeOpts {
    fn default() -> Self {
        Self {
            codec: "libx264".into(),
            crf: Some(18),
            bitrate: None,
            preset: "medium".into(),
            pix_fmt: "yuv420p".into(),
        }
    }
}

impl EncodeOpts {
    /// Map a human-ish quality knob (low/medium/high) to a CRF value.
    pub fn with_quality(mut self, quality: Quality) -> Self {
        self.crf = Some(match quality {
            Quality::Low => 28,
            Quality::Medium => 23,
            Quality::High => 18,
        });
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Quality { Low, Medium, High }

pub struct VideoWriter {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr_pump: Option<JoinHandle<Vec<u8>>>,
    size: (u32, u32),
    output: PathBuf,
    frames_written: u64,
}

impl VideoWriter {
    pub fn create(path: &Path, size: (u32, u32), fps: f64, opts: EncodeOpts) -> Result<Self> {
        ensure_tool("ffmpeg")?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let (w, h) = size;
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-hide_banner", "-loglevel", "error"])
            .args(["-f", "rawvideo", "-pix_fmt", "bgr24"])
            .args(["-s", &format!("{w}x{h}")])
            .args(["-r", &format!("{:.6}", fps)])
            .args(["-i", "-"])
            .args(["-c:v", &opts.codec]);

        if let Some(crf) = opts.crf {
            cmd.args(["-crf", &crf.to_string()]);
        }
        if let Some(bitrate) = &opts.bitrate {
            cmd.args(["-b:v", bitrate]);
        }
        cmd.args(["-preset", &opts.preset])
            .args(["-pix_fmt", &opts.pix_fmt])
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        debug!(args = ?cmd.get_args().collect::<Vec<_>>(), "spawning ffmpeg encoder");
        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        // Drain stderr on a background thread so a verbose ffmpeg log never
        // blocks us on a full pipe buffer. We hold on to the handle for the
        // final diagnostic dump in `finish`.
        let stderr_pump = thread::spawn(move || {
            use std::io::Read;
            let mut buf = Vec::new();
            let mut reader = stderr;
            let _ = reader.read_to_end(&mut buf);
            buf
        });

        info!(
            path = %path.display(),
            size = format!("{w}x{h}"),
            fps,
            codec = %opts.codec,
            crf = ?opts.crf,
            "video encoder ready"
        );

        Ok(Self {
            child,
            stdin: Some(stdin),
            stderr_pump: Some(stderr_pump),
            size,
            output: path.to_path_buf(),
            frames_written: 0,
        })
    }

    pub fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        if (frame.width, frame.height) != self.size {
            return Err(PipelineError::FfmpegFailed {
                code: 0,
                stderr: format!(
                    "frame size {}x{} doesn't match encoder {}x{}",
                    frame.width, frame.height, self.size.0, self.size.1
                ),
            });
        }
        let stdin = self.stdin.as_mut().ok_or_else(|| PipelineError::FfmpegFailed {
            code: 0,
            stderr: "write_frame called after finish".into(),
        })?;
        stdin.write_all(&frame.bgr).map_err(|e| self.wrap_broken_pipe(e))?;
        self.frames_written += 1;
        Ok(())
    }

    /// Close stdin, wait for ffmpeg to finish flushing, and surface any encoder error.
    pub fn finish(mut self) -> Result<PathBuf> {
        drop(self.stdin.take()); // flush EOF to ffmpeg
        let status = self.child.wait()?;
        let stderr_bytes = self.stderr_pump
            .take()
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default();
        let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();

        if !status.success() {
            return Err(PipelineError::FfmpegFailed {
                code: status.code().unwrap_or(-1),
                stderr: if stderr.is_empty() { "(ffmpeg stderr empty)".into() } else { stderr },
            });
        }

        info!(
            path = %self.output.display(),
            frames = self.frames_written,
            "encoder finished"
        );
        Ok(self.output.clone())
    }

    pub fn output(&self) -> &Path { &self.output }
    pub fn frames_written(&self) -> u64 { self.frames_written }

    fn wrap_broken_pipe(&mut self, err: std::io::Error) -> PipelineError {
        // Reap the encoder immediately so we can quote its stderr.
        let _ = self.child.wait();
        let stderr_bytes = self.stderr_pump
            .take()
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default();
        let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();
        PipelineError::FfmpegFailed {
            code: self.child.try_wait().ok().flatten().and_then(|s| s.code()).unwrap_or(-1),
            stderr: format!("stdin write failed: {err}\nencoder stderr:\n{stderr}"),
        }
    }
}

/// Re-mux: copy the encoded video stream + the source's first audio stream into `out`.
/// If `source_media` has no audio, the output has no audio track (not an error).
pub fn remux_audio(video: &Path, source_media: &Path, out: &Path) -> Result<PathBuf> {
    ensure_tool("ffmpeg")?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let output = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-i"]).arg(video)
        .args(["-i"]).arg(source_media)
        .args(["-c", "copy", "-map", "0:v:0", "-map", "1:a:0?", "-shortest"])
        .arg(out)
        .output()?;
    if !output.status.success() {
        return Err(PipelineError::FfmpegFailed {
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    info!(video = %video.display(), source = %source_media.display(), out = %out.display(), "re-muxed audio");
    Ok(out.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_encode_opts_have_sensible_h264_defaults() {
        let o = EncodeOpts::default();
        assert_eq!(o.codec, "libx264");
        assert_eq!(o.crf, Some(18));
        assert_eq!(o.pix_fmt, "yuv420p");
        assert!(o.bitrate.is_none());
    }

    #[test]
    fn quality_knob_maps_to_crf_scale() {
        assert_eq!(EncodeOpts::default().with_quality(Quality::High).crf, Some(18));
        assert_eq!(EncodeOpts::default().with_quality(Quality::Medium).crf, Some(23));
        assert_eq!(EncodeOpts::default().with_quality(Quality::Low).crf, Some(28));
    }
}
