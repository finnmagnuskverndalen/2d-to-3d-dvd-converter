//! End-to-end DVD authoring: synthetic clip → DVD MPEG → VIDEO_TS → ISO.
//!
//! Requires `ffmpeg`, `dvdauthor`, and `mkisofs`/`genisoimage` on PATH.
//! Marked `#[ignore]` so plain `cargo test` stays fast; run explicitly with
//! `cargo test --test dvd_authoring -- --ignored`.

use std::path::Path;
use std::process::Command;

use stereoscopy::{DvdAuthorer, Region};

fn make_test_clip(path: &Path, seconds: u32, fps: u32, w: u32, h: u32) {
    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi"])
        .args(["-i", &format!("testsrc=duration={seconds}:size={w}x{h}:rate={fps}")])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(path)
        .status()
        .expect("ffmpeg");
    assert!(status.success(), "clip generation failed");
}

#[test]
#[ignore = "slow: runs ffmpeg + dvdauthor + mkisofs"]
fn full_dvd_chain_produces_iso() {
    let tmp = tempfile::tempdir().unwrap();

    // 1-second NTSC-ish clip so dvdauthor has valid DVD-spec input to chew on.
    let src = tmp.path().join("stereo.mkv");
    make_test_clip(&src, 1, 30, 720, 480);

    let authorer = DvdAuthorer { region: Region::Ntsc, ..DvdAuthorer::default() };

    let mpeg = tmp.path().join("src.mpg");
    authorer.prepare_video(&src, &mpeg, false).expect("prepare_video");
    assert!(mpeg.exists());

    let dvd_root = tmp.path().join("dvd_build");
    authorer.build_video_ts(&mpeg, &dvd_root).expect("build_video_ts");
    assert!(dvd_root.join("VIDEO_TS").join("VIDEO_TS.IFO").exists());
    assert!(dvd_root.join("AUDIO_TS").is_dir());

    let iso = tmp.path().join("out.iso");
    authorer.build_iso(&dvd_root, &iso).expect("build_iso");
    assert!(iso.exists());

    let size = std::fs::metadata(&iso).unwrap().len();
    assert!(size > 100_000, "iso suspiciously small: {size} bytes");
}

#[test]
#[ignore = "slow: runs ffmpeg + dvdauthor + mkisofs"]
fn author_convenience_chains_all_three_stages() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("stereo.mkv");
    make_test_clip(&src, 1, 30, 720, 480);

    let iso = tmp.path().join("movie.iso");
    let cache = tmp.path().join(".cache");

    let authorer = DvdAuthorer { region: Region::Ntsc, ..DvdAuthorer::default() };
    let produced = authorer.author(&src, &iso, &cache, false).expect("author");
    assert_eq!(produced, iso);
    assert!(iso.exists());
    assert!(cache.join("dvd_source.mpg").exists());
    assert!(cache.join("dvd_build").join("VIDEO_TS").join("VIDEO_TS.IFO").exists());
}
