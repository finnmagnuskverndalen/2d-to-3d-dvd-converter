//! Phase 2: frame-level reader covering plain video files, VIDEO_TS directories, and DVD ISOs.
//!
//! All three input shapes resolve to a single concrete file we can hand to ffmpeg:
//!   * video file         → used directly
//!   * VIDEO_TS directory → VOBs concatenated into a cached MKV via `ffmpeg -f concat`
//!   * `.iso` file        → extracted via `ffmpeg -f dvdvideo -title 1` into a cached MKV
//!
//! Frames are decoded by piping `ffmpeg -f rawvideo -pix_fmt bgr24 -` into stdout and
//! reading fixed-size `(w*h*3)` byte chunks. This keeps the dependency footprint tiny
//! (no `ffmpeg-next` linkage) and makes the pipeline reproducible by hand on the shell.

use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use tracing::{info, warn};

use crate::errors::{PipelineError, Result};
use crate::util::{VideoMetadata, ensure_tool, get_video_metadata};

const VIDEO_EXTS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "m4v", "webm", "mpg", "mpeg", "ts", "vob",
];

#[derive(Debug, Clone)]
pub struct Frame {
    pub index: u64,
    pub width: u32,
    pub height: u32,
    pub bgr: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct FrameOpts {
    pub sample_rate: u32,
    pub start: u64,
    pub end: Option<u64>,
}

impl Default for FrameOpts {
    fn default() -> Self {
        Self { sample_rate: 1, start: 0, end: None }
    }
}

pub struct VideoReader {
    source: PathBuf,
    video_path: PathBuf,
    metadata: VideoMetadata,
}

impl VideoReader {
    pub fn open(source: &Path, cache_dir: &Path) -> Result<Self> {
        if !source.exists() {
            return Err(PipelineError::InputMissing(source.to_path_buf()));
        }
        fs::create_dir_all(cache_dir)?;

        let video_path = resolve_source(source, cache_dir)?;
        let metadata = get_video_metadata(&video_path)?;

        info!(
            file = %video_path.display(),
            width = metadata.width,
            height = metadata.height,
            fps = metadata.fps,
            frames = metadata.total_frames,
            codec = %metadata.codec,
            "opened source"
        );
        if metadata.height < 720 {
            warn!(
                height = metadata.height,
                "source height is below 720p — depth estimation quality may suffer"
            );
        }

        Ok(Self {
            source: source.to_path_buf(),
            video_path,
            metadata,
        })
    }

    pub fn metadata(&self) -> &VideoMetadata { &self.metadata }
    pub fn source(&self) -> &Path { &self.source }
    pub fn video_path(&self) -> &Path { &self.video_path }

    /// Return the frame at `index` by seeking the decoder. One subprocess per call —
    /// fine for debugging / random access, but prefer `frames()` for bulk iteration.
    pub fn get_frame(&self, index: u64) -> Result<Frame> {
        if index >= self.metadata.total_frames {
            return Err(PipelineError::FrameOutOfRange {
                index,
                total: self.metadata.total_frames,
            });
        }
        let ts = index as f64 / self.metadata.fps.max(1e-6);
        let (w, h) = (self.metadata.width, self.metadata.height);
        let bytes_per_frame = (w as usize) * (h as usize) * 3;

        let output = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error"])
            .args(["-ss", &format!("{ts:.6}")])
            .args(["-i"])
            .arg(&self.video_path)
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
            .output()?;
        if !output.status.success() {
            return Err(PipelineError::FfmpegFailed {
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        if output.stdout.len() < bytes_per_frame {
            return Err(PipelineError::FfmpegFailed {
                code: 0,
                stderr: format!(
                    "short read: got {} bytes, expected {}",
                    output.stdout.len(), bytes_per_frame
                ),
            });
        }

        Ok(Frame {
            index,
            width: w,
            height: h,
            bgr: output.stdout.into_iter().take(bytes_per_frame).collect(),
        })
    }

    /// Stream frames sequentially. `opts.sample_rate = N` yields every Nth frame.
    /// Decoding happens in a single ffmpeg subprocess for the entire iteration.
    pub fn frames(&self, opts: FrameOpts) -> Result<FrameIter> {
        if opts.sample_rate == 0 {
            return Err(PipelineError::FfmpegFailed {
                code: 0,
                stderr: "sample_rate must be >= 1".into(),
            });
        }
        ensure_tool("ffmpeg")?;

        let (w, h) = (self.metadata.width, self.metadata.height);
        let total = self.metadata.total_frames;
        let end = opts.end.unwrap_or(total).min(total);

        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error"]);
        if opts.start > 0 {
            let ts = opts.start as f64 / self.metadata.fps.max(1e-6);
            cmd.args(["-ss", &format!("{ts:.6}")]);
        }
        cmd.args(["-i"]).arg(&self.video_path);
        cmd.args(["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());

        let mut child = cmd.spawn()?;
        let stdout = child.stdout.take().expect("stdout was piped");

        Ok(FrameIter {
            child: Some(child),
            reader: BufReader::with_capacity((w as usize) * (h as usize) * 3 * 2, stdout),
            width: w,
            height: h,
            frames_in_window: 0,
            next_index: opts.start,
            end,
            sample_rate: opts.sample_rate,
        })
    }

    /// Dump frames as PNGs for inspection — delegated to ffmpeg so we don't need
    /// an image-encoding dep in Phase 2.
    pub fn extract_frames(&self, out_dir: &Path, opts: FrameOpts, limit: Option<usize>) -> Result<Vec<PathBuf>> {
        ensure_tool("ffmpeg")?;
        fs::create_dir_all(out_dir)?;
        let pattern = out_dir.join("frame_%06d.png");

        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-hide_banner", "-loglevel", "error"]);
        if opts.start > 0 {
            let ts = opts.start as f64 / self.metadata.fps.max(1e-6);
            cmd.args(["-ss", &format!("{ts:.6}")]);
        }
        cmd.args(["-i"]).arg(&self.video_path);

        if opts.sample_rate > 1 {
            // Keep every Nth frame via the `select` filter.
            cmd.args(["-vf", &format!("select='not(mod(n\\,{}))'", opts.sample_rate), "-vsync", "vfr"]);
        }
        if let Some(n) = limit {
            cmd.args(["-frames:v", &n.to_string()]);
        }
        cmd.arg(&pattern);

        let output = cmd.output()?;
        if !output.status.success() {
            return Err(PipelineError::FfmpegFailed {
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let mut written: Vec<PathBuf> = fs::read_dir(out_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |x| x == "png"))
            .filter(|p| p.file_name().and_then(|f| f.to_str()).map_or(false, |n| n.starts_with("frame_")))
            .collect();
        written.sort();
        info!(count = written.len(), dir = %out_dir.display(), "extracted frames");
        Ok(written)
    }
}

// ------------------------------------------------------------- FrameIter

pub struct FrameIter {
    child: Option<Child>,
    reader: BufReader<ChildStdout>,
    width: u32,
    height: u32,
    frames_in_window: u64,  // frames consumed since `start`
    next_index: u64,        // frame number of the next raw frame about to be read
    end: u64,
    sample_rate: u32,
}

impl FrameIter {
    fn read_one_frame(&mut self) -> Result<Option<Vec<u8>>> {
        let n = (self.width as usize) * (self.height as usize) * 3;
        let mut buf = vec![0u8; n];
        let mut filled = 0;
        while filled < n {
            match self.reader.read(&mut buf[filled..]) {
                Ok(0) => {
                    if filled == 0 {
                        return Ok(None);
                    } else {
                        return Err(PipelineError::FfmpegFailed {
                            code: 0,
                            stderr: format!("short frame: {filled}/{n} bytes before EOF"),
                        });
                    }
                }
                Ok(k) => filled += k,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(PipelineError::Io(e)),
            }
        }
        Ok(Some(buf))
    }
}

impl Iterator for FrameIter {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.next_index >= self.end {
                return None;
            }
            let frame_bytes = match self.read_one_frame() {
                Ok(Some(b)) => b,
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            };
            let current = self.next_index;
            self.next_index += 1;

            let yield_this = self.frames_in_window % self.sample_rate as u64 == 0;
            self.frames_in_window += 1;
            if !yield_this {
                continue;
            }
            return Some(Ok(Frame {
                index: current,
                width: self.width,
                height: self.height,
                bgr: frame_bytes,
            }));
        }
    }
}

impl Drop for FrameIter {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // Close ffmpeg's stdin (none here) and reap; swallow errors in Drop.
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

// ------------------------------------------------------- source resolution

fn resolve_source(source: &Path, cache_dir: &Path) -> Result<PathBuf> {
    if source.is_file() {
        let ext = source.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
        if VIDEO_EXTS.contains(&ext.as_str()) {
            return Ok(source.to_path_buf());
        }
        if ext == "iso" {
            return extract_iso(source, cache_dir);
        }
    }
    if source.is_dir() && looks_like_video_ts(source) {
        return extract_video_ts(source, cache_dir);
    }
    Err(PipelineError::UnsupportedInput(source.to_path_buf()))
}

fn looks_like_video_ts(path: &Path) -> bool {
    let name_matches = path.file_name()
        .and_then(|n| n.to_str())
        .map_or(false, |n| n.eq_ignore_ascii_case("VIDEO_TS"));
    if name_matches { return true; }
    if path.join("VIDEO_TS").is_dir() { return true; }
    fs::read_dir(path)
        .map(|dir| {
            dir.filter_map(|e| e.ok()).any(|e| {
                e.file_name().to_str().map_or(false, |n| {
                    n.to_ascii_uppercase().starts_with("VTS_") && n.to_ascii_uppercase().ends_with(".VOB")
                })
            })
        })
        .unwrap_or(false)
}

fn extract_video_ts(path: &Path, cache_dir: &Path) -> Result<PathBuf> {
    ensure_tool("ffmpeg")?;
    let vts_dir = if path.file_name().and_then(|n| n.to_str()).map_or(false, |n| n.eq_ignore_ascii_case("VIDEO_TS")) {
        path.to_path_buf()
    } else {
        path.join("VIDEO_TS")
    };
    if !vts_dir.is_dir() {
        return Err(PipelineError::UnsupportedInput(path.to_path_buf()));
    }

    // Group VOBs by title-set number ("01" in VTS_01_1.VOB), pick largest set.
    let mut titles: std::collections::BTreeMap<String, Vec<PathBuf>> = Default::default();
    for entry in fs::read_dir(&vts_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_ascii_uppercase();
        if !(name.starts_with("VTS_") && name.ends_with(".VOB")) {
            continue;
        }
        // Skip menu VOBs (VTS_XX_0.VOB).
        let parts: Vec<&str> = name.trim_end_matches(".VOB").split('_').collect();
        if parts.len() != 3 { continue; }
        if parts[2] == "0" { continue; }
        titles.entry(parts[1].to_string()).or_default().push(entry.path());
    }
    if titles.is_empty() {
        return Err(PipelineError::UnsupportedInput(path.to_path_buf()));
    }

    let main_key = titles
        .iter()
        .max_by_key(|(_, v)| v.iter().filter_map(|p| p.metadata().ok()).map(|m| m.len()).sum::<u64>())
        .map(|(k, _)| k.clone())
        .unwrap();
    let mut vobs = titles.remove(&main_key).unwrap();
    vobs.sort();

    let total_bytes: u64 = vobs.iter().filter_map(|p| p.metadata().ok()).map(|m| m.len()).sum();
    info!(title = %main_key, vobs = vobs.len(), size_gb = total_bytes as f64 / 1024f64.powi(3), "selected title set");

    let concat_list = cache_dir.join(format!("vts_{main_key}_concat.txt"));
    let mut f = fs::File::create(&concat_list)?;
    for v in &vobs {
        writeln!(f, "file '{}'", v.display())?;
    }
    let out = cache_dir.join(format!("vts_{main_key}_remux.mkv"));
    if out.exists() {
        let out_mtime = out.metadata().and_then(|m| m.modified()).ok();
        let list_mtime = concat_list.metadata().and_then(|m| m.modified()).ok();
        if let (Some(o), Some(l)) = (out_mtime, list_mtime) {
            if o >= l {
                info!(cache = %out.display(), "using cached remux");
                return Ok(out);
            }
        }
    }

    info!(out = %out.display(), "remuxing VIDEO_TS");
    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "warning"])
        .args(["-f", "concat", "-safe", "0", "-i"])
        .arg(&concat_list)
        .args(["-c", "copy", "-map", "0:v:0", "-map", "0:a:0?"])
        .arg(&out)
        .status()?;
    if !status.success() {
        return Err(PipelineError::FfmpegFailed {
            code: status.code().unwrap_or(-1),
            stderr: "VIDEO_TS concat failed".into(),
        });
    }
    Ok(out)
}

fn extract_iso(iso: &Path, cache_dir: &Path) -> Result<PathBuf> {
    ensure_tool("ffmpeg")?;
    let stem = iso.file_stem().and_then(|s| s.to_str()).unwrap_or("iso");
    let out = cache_dir.join(format!("{stem}_title1.mkv"));
    if out.exists() {
        let out_mtime = out.metadata().and_then(|m| m.modified()).ok();
        let iso_mtime = iso.metadata().and_then(|m| m.modified()).ok();
        if let (Some(o), Some(i)) = (out_mtime, iso_mtime) {
            if o >= i {
                info!(cache = %out.display(), "using cached ISO extraction");
                return Ok(out);
            }
        }
    }

    info!(out = %out.display(), "extracting DVD title from ISO");
    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "warning"])
        .args(["-f", "dvdvideo", "-title", "1", "-i"])
        .arg(iso)
        .args(["-c", "copy", "-map", "0:v:0", "-map", "0:a:0?"])
        .arg(&out)
        .status()?;
    if !status.success() {
        return Err(PipelineError::FfmpegFailed {
            code: status.code().unwrap_or(-1),
            stderr: "ISO extraction failed".into(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_opts_default() {
        let o = FrameOpts::default();
        assert_eq!(o.sample_rate, 1);
        assert_eq!(o.start, 0);
        assert!(o.end.is_none());
    }
}
