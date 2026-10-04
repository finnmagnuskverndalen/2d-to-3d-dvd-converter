//! End-to-end smoke test: generate a predictable testsrc clip with ffmpeg, then verify
//! the reader can probe it and stream frames through at the expected rate.

use std::path::Path;
use std::process::Command;

use stereoscopy::video_reader::{FrameOpts, VideoReader};

fn make_test_clip(path: &Path, seconds: u32, fps: u32, width: u32, height: u32) {
    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi"])
        .args(["-i", &format!("testsrc=duration={seconds}:size={width}x{height}:rate={fps}")])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(path)
        .status()
        .expect("failed to invoke ffmpeg");
    assert!(status.success(), "ffmpeg clip generation failed");
}

#[test]
fn probes_metadata_and_streams_frames() {
    let tmp = tempfile::tempdir().unwrap();
    let clip = tmp.path().join("clip.mp4");
    make_test_clip(&clip, 2, 24, 320, 240);

    let cache = tmp.path().join(".cache");
    let reader = VideoReader::open(&clip, &cache).expect("open");

    let md = reader.metadata();
    assert_eq!(md.width, 320);
    assert_eq!(md.height, 240);
    assert!((md.fps - 24.0).abs() < 0.01, "fps was {}", md.fps);
    assert_eq!(md.total_frames, 48);
    assert_eq!(md.codec, "h264");

    // Random access
    let f0 = reader.get_frame(0).expect("get_frame 0");
    assert_eq!(f0.width, 320);
    assert_eq!(f0.height, 240);
    assert_eq!(f0.bgr.len(), 320 * 240 * 3);

    // Sequential, every 4th frame → 48 / 4 = 12
    let frames: Vec<_> = reader.frames(FrameOpts { sample_rate: 4, start: 0, end: None })
        .expect("frames iterator")
        .map(|r| r.expect("frame"))
        .collect();
    assert_eq!(frames.len(), 12, "expected 12 frames, got {}", frames.len());

    for (expected_index, f) in frames.iter().enumerate() {
        assert_eq!(f.width, 320);
        assert_eq!(f.height, 240);
        assert_eq!(f.bgr.len(), 320 * 240 * 3);
        assert_eq!(f.index, (expected_index as u64) * 4);
    }
}

#[test]
fn extract_frames_writes_pngs() {
    let tmp = tempfile::tempdir().unwrap();
    let clip = tmp.path().join("clip.mp4");
    make_test_clip(&clip, 1, 24, 320, 240);

    let cache = tmp.path().join(".cache");
    let reader = VideoReader::open(&clip, &cache).expect("open");

    let out_dir = tmp.path().join("frames");
    let written = reader
        .extract_frames(&out_dir, FrameOpts::default(), Some(5))
        .expect("extract_frames");
    assert_eq!(written.len(), 5);
    for p in &written {
        assert!(p.exists());
        assert_eq!(p.extension().and_then(|s| s.to_str()), Some("png"));
    }
}
