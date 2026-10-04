//! End-to-end test: feed synthetic BGR frames through `VideoWriter`, then re-probe
//! the output with ffprobe to confirm the encoder actually produced a valid stream.

use std::path::Path;
use std::process::Command;

use stereoscopy::util::get_video_metadata;
use stereoscopy::{EncodeOpts, Frame, Quality, VideoWriter};

fn make_frame(index: u64, width: u32, height: u32, bgr: [u8; 3]) -> Frame {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for _ in 0..(width as usize * height as usize) {
        data.extend_from_slice(&bgr);
    }
    Frame { index, width, height, bgr: data }
}

#[test]
fn encodes_synthetic_frames_to_mp4() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("synth.mp4");

    let (w, h, fps) = (64u32, 48u32, 24.0f64);
    let opts = EncodeOpts::default().with_quality(Quality::High);
    let mut writer = VideoWriter::create(&out, (w, h), fps, opts).expect("create writer");

    // 24 solid frames of varying brightness.
    for i in 0..24 {
        let shade = (i * 10) as u8;
        let frame = make_frame(i as u64, w, h, [shade, shade, shade]);
        writer.write_frame(&frame).expect("write_frame");
    }
    let produced = writer.finish().expect("finish");
    assert_eq!(produced, out);
    assert!(out.exists(), "encoder did not produce {}", out.display());

    let md = get_video_metadata(&out).expect("probe output");
    assert_eq!(md.width, w);
    assert_eq!(md.height, h);
    assert!((md.fps - fps).abs() < 0.01, "fps {} vs expected {fps}", md.fps);
    assert_eq!(md.total_frames, 24);
    assert_eq!(md.codec, "h264");
}

#[test]
fn frame_size_mismatch_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("mismatch.mp4");
    let mut writer = VideoWriter::create(&out, (32, 24), 24.0, EncodeOpts::default()).expect("create");

    let bad = make_frame(0, 48, 24, [0, 0, 0]);
    let err = writer.write_frame(&bad).unwrap_err();
    assert!(matches!(err, stereoscopy::PipelineError::FfmpegFailed { .. }));
    // Still need to drop the writer cleanly; finish will fail because the encoder
    // never got data, which is fine for the test's scope.
    let _ = writer.finish();
}

#[test]
fn audio_remux_without_audio_source_still_succeeds() {
    // ffmpeg's `-map 1:a:0?` makes the audio stream optional, so re-muxing a
    // silent source onto a silent-input video should produce a valid file.
    let tmp = tempfile::tempdir().unwrap();
    let silent_source = tmp.path().join("silent_src.mp4");
    make_silent_clip(&silent_source, 1, 24, 64, 48);

    let encoded = tmp.path().join("encoded.mp4");
    let mut w = VideoWriter::create(&encoded, (64, 48), 24.0, EncodeOpts::default()).expect("create");
    for i in 0..6u8 {
        let f = make_frame(i as u64, 64, 48, [i * 40, i * 40, i * 40]);
        w.write_frame(&f).expect("write");
    }
    w.finish().expect("finish");

    let out = tmp.path().join("final.mp4");
    stereoscopy::video_writer::remux_audio(&encoded, &silent_source, &out).expect("remux");
    assert!(out.exists());
}

fn make_silent_clip(path: &Path, seconds: u32, fps: u32, w: u32, h: u32) {
    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi"])
        .args(["-i", &format!("testsrc=duration={seconds}:size={w}x{h}:rate={fps}")])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(path)
        .status()
        .expect("ffmpeg");
    assert!(status.success());
}
