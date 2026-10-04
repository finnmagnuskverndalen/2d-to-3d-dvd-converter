//! Phase 6: DVD authoring.
//!
//! Three subprocess stages, each a thin wrapper around a system binary:
//!
//! 1. `prepare_video` — `ffmpeg -target {ntsc|pal}-dvd` converts the stereoscopic
//!    source into a DVD-compliant MPEG-2 program stream (720×480 @ 29.97 for NTSC,
//!    720×576 @ 25 for PAL, AC3 48 kHz audio).
//! 2. `build_video_ts` — `dvdauthor` builds the IFO/VOB/BUP tree under VIDEO_TS/
//!    (with an empty AUDIO_TS/ sibling for strict-player compatibility).
//! 3. `build_iso` — `mkisofs -dvd-video` (or `genisoimage`) wraps the directory
//!    into an ISO 9660 image a DVD player will boot from.
//!
//! `author()` chains all three via a cache dir.
//!
//! ## Caveats
//!
//! DVD is 720×480 (NTSC) / 720×576 (PAL). A side-by-side stereoscopic source gets
//! downscaled — the stereo effect is preserved but at low resolution. For a
//! DVD-only workflow, `--format anaglyph` is usually the better choice: it keeps
//! the source resolution and plays back on any screen with red/cyan glasses.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::{debug, info};

use crate::errors::{PipelineError, Result};
use crate::util::ensure_tool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region { Ntsc, Pal }

impl Region {
    pub fn ffmpeg_target(self) -> &'static str {
        match self { Region::Ntsc => "ntsc-dvd", Region::Pal => "pal-dvd" }
    }
    pub fn fps(self) -> f64 {
        match self { Region::Ntsc => 30000.0 / 1001.0, Region::Pal => 25.0 }
    }
    pub fn resolution(self) -> (u32, u32) {
        match self { Region::Ntsc => (720, 480), Region::Pal => (720, 576) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec { Ac3, Mp2 }

impl AudioCodec {
    pub fn ffmpeg_name(self) -> &'static str {
        match self { AudioCodec::Ac3 => "ac3", AudioCodec::Mp2 => "mp2" }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectRatio { Standard, Widescreen }

impl AspectRatio {
    pub fn ffmpeg_arg(self) -> &'static str {
        match self { AspectRatio::Standard => "4:3", AspectRatio::Widescreen => "16:9" }
    }
}

#[derive(Debug, Clone)]
pub struct DvdAuthorer {
    pub region: Region,
    pub audio_codec: AudioCodec,
    pub video_bitrate: String,     // ignored unless we want to override `-target`
    pub audio_bitrate: String,
    pub aspect: AspectRatio,
}

impl Default for DvdAuthorer {
    fn default() -> Self {
        Self {
            region: Region::Ntsc,
            audio_codec: AudioCodec::Ac3,
            video_bitrate: "6000k".into(),
            audio_bitrate: "192k".into(),
            aspect: AspectRatio::Widescreen,
        }
    }
}

impl DvdAuthorer {
    /// Convert the stereoscopic encode into a DVD-compliant MPEG-2 program stream
    /// containing both video and (optionally) audio.
    pub fn prepare_video(&self, src: &Path, out: &Path, include_audio: bool) -> Result<PathBuf> {
        ensure_tool("ffmpeg")?;
        ensure_parent(out)?;

        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-hide_banner", "-loglevel", "error"])
            .args(["-i"]).arg(src)
            .args(["-target", self.region.ffmpeg_target()])
            .args(["-aspect", self.aspect.ffmpeg_arg()]);

        if include_audio {
            cmd.args(["-c:a", self.audio_codec.ffmpeg_name()])
                .args(["-b:a", &self.audio_bitrate])
                .args(["-ar", "48000"]);
        } else {
            cmd.arg("-an");
        }

        cmd.arg(out);

        debug!(args = ?cmd.get_args().collect::<Vec<_>>(), "ffmpeg prepare_video");
        let output = cmd.output()?;
        if !output.status.success() {
            return Err(PipelineError::FfmpegFailed {
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        info!(
            src = %src.display(),
            out = %out.display(),
            target = self.region.ffmpeg_target(),
            audio = include_audio,
            "prepared DVD MPEG stream"
        );
        Ok(out.to_path_buf())
    }

    /// Build the DVD filesystem (VIDEO_TS/AUDIO_TS tree) under `out_dir`.
    ///
    /// Runs `dvdauthor -o <out_dir> -t <mpeg>` to add the title, then
    /// `dvdauthor -o <out_dir> -T` to generate the TOC. The `VIDEO_FORMAT`
    /// env var is set so dvdauthor doesn't abort on the TOC pass with
    /// "no video format specified for VMGM".
    pub fn build_video_ts(&self, mpeg: &Path, out_dir: &Path) -> Result<PathBuf> {
        ensure_tool("dvdauthor")?;
        if out_dir.exists() {
            // dvdauthor refuses to overwrite an existing structure cleanly.
            fs::remove_dir_all(out_dir)?;
        }
        fs::create_dir_all(out_dir)?;

        let video_format = match self.region {
            Region::Ntsc => "NTSC",
            Region::Pal  => "PAL",
        };

        // Title pass.
        run_tool(
            "dvdauthor",
            Command::new("dvdauthor")
                .env("VIDEO_FORMAT", video_format)
                .args(["-o"]).arg(out_dir)
                .args(["-t"]).arg(mpeg),
        )?;

        // Finalise with the top-level TOC (-T).
        run_tool(
            "dvdauthor",
            Command::new("dvdauthor")
                .env("VIDEO_FORMAT", video_format)
                .args(["-o"]).arg(out_dir)
                .arg("-T"),
        )?;

        // Some strict standalone players want AUDIO_TS to exist (even empty).
        let audio_ts = out_dir.join("AUDIO_TS");
        if !audio_ts.exists() {
            fs::create_dir(&audio_ts)?;
        }

        let ifo = out_dir.join("VIDEO_TS").join("VIDEO_TS.IFO");
        if !ifo.exists() {
            return Err(PipelineError::ExternalToolFailed {
                tool: "dvdauthor",
                code: 0,
                stderr: format!("expected {} after authoring; not found", ifo.display()),
            });
        }

        info!(dir = %out_dir.display(), "built VIDEO_TS");
        Ok(out_dir.to_path_buf())
    }

    /// Wrap the DVD filesystem into an ISO 9660 image. Prefers `mkisofs`,
    /// falls back to `genisoimage` (the Debian replacement with the same CLI).
    pub fn build_iso(&self, dvd_root: &Path, iso: &Path) -> Result<PathBuf> {
        let tool = if ensure_tool("mkisofs").is_ok() {
            "mkisofs"
        } else {
            ensure_tool("genisoimage")?;
            "genisoimage"
        };
        ensure_parent(iso)?;

        run_tool(
            tool,
            Command::new(tool)
                .args(["-dvd-video", "-o"]).arg(iso)
                .arg(dvd_root),
        )?;

        let size = fs::metadata(iso).map(|m| m.len()).unwrap_or(0);
        info!(
            iso = %iso.display(),
            size_mb = size as f64 / (1024.0 * 1024.0),
            tool,
            "wrote DVD ISO"
        );
        Ok(iso.to_path_buf())
    }

    /// Convenience: full chain stereoscopic encode → DVD ISO. Intermediates land
    /// in `cache_dir`; callers can delete them afterwards.
    pub fn author(
        &self,
        stereo_video: &Path,
        iso: &Path,
        cache_dir: &Path,
        include_audio: bool,
    ) -> Result<PathBuf> {
        fs::create_dir_all(cache_dir)?;
        let mpeg = cache_dir.join("dvd_source.mpg");
        let dvd_root = cache_dir.join("dvd_build");

        self.prepare_video(stereo_video, &mpeg, include_audio)?;
        self.build_video_ts(&mpeg, &dvd_root)?;
        self.build_iso(&dvd_root, iso)?;
        Ok(iso.to_path_buf())
    }
}

// ---- shared helpers ---------------------------------------------------------

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn run_tool(tool: &'static str, cmd: &mut Command) -> Result<()> {
    debug!(tool, args = ?cmd.get_args().collect::<Vec<_>>(), "running external tool");
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(PipelineError::ExternalToolFailed {
            tool,
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_constants_are_correct() {
        assert_eq!(Region::Ntsc.resolution(), (720, 480));
        assert_eq!(Region::Pal.resolution(), (720, 576));
        assert!((Region::Ntsc.fps() - 29.97).abs() < 0.01);
        assert_eq!(Region::Pal.fps(), 25.0);
        assert_eq!(Region::Ntsc.ffmpeg_target(), "ntsc-dvd");
        assert_eq!(Region::Pal.ffmpeg_target(), "pal-dvd");
    }

    #[test]
    fn aspect_ratio_formats() {
        assert_eq!(AspectRatio::Standard.ffmpeg_arg(), "4:3");
        assert_eq!(AspectRatio::Widescreen.ffmpeg_arg(), "16:9");
    }

    #[test]
    fn audio_codec_names() {
        assert_eq!(AudioCodec::Ac3.ffmpeg_name(), "ac3");
        assert_eq!(AudioCodec::Mp2.ffmpeg_name(), "mp2");
    }

    #[test]
    fn default_is_ntsc_widescreen_ac3() {
        let a = DvdAuthorer::default();
        assert_eq!(a.region, Region::Ntsc);
        assert_eq!(a.aspect, AspectRatio::Widescreen);
        assert_eq!(a.audio_codec, AudioCodec::Ac3);
    }
}
