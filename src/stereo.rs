//! Stereo view synthesis (Phase 4).
//!
//! Given a BGR frame and a normalised depth map (near = 1.0, far = 0.0), produce a
//! left/right eye pair via forward-mapping disparity warping, then pack them into
//! one of the standard stereoscopic layouts.
//!
//! ## Algorithm
//!
//! For each source column `x` with depth `d ∈ [0, 1]` we compute a signed shift
//! `s = baseline · (d − convergence)`, clamped to `±max_disparity`. The source
//! pixel is written to:
//!   * left-eye column `x + s/2`
//!   * right-eye column `x − s/2`
//!
//! A per-destination-column z-buffer resolves overlaps: a closer source pixel
//! (higher `d`) wins over a farther one that already wrote to the same slot.
//! Destination columns never written to by any source pixel are "disocclusions"
//! and are filled by horizontal interpolation between the surrounding valid
//! pixels on the same row.
//!
//! Row-level parallelism via `rayon` — each row is independent of every other.

use rayon::prelude::*;
use tracing::debug;

use crate::depth::DepthMap;
use crate::errors::{PipelineError, Result};
use crate::video_reader::Frame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    SideBySide,
    TopBottom,
    Anaglyph,
    FrameSequential,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InpaintMethod {
    /// Horizontal interpolation between the nearest valid pixels on the same row.
    /// Fast, dependency-free, adequate for small disocclusions. Default.
    Simple,
    /// Placeholders for OpenCV-parity algorithms; not yet implemented.
    Telea,
    NavierStokes,
}

#[derive(Debug, Clone, Copy)]
pub struct StereoGenerator {
    pub baseline_px: i32,
    pub max_disparity: i32,
    pub convergence: f32,
    pub inpaint: InpaintMethod,
}

impl Default for StereoGenerator {
    fn default() -> Self {
        Self { baseline_px: 20, max_disparity: 50, convergence: 0.5, inpaint: InpaintMethod::Simple }
    }
}

impl StereoGenerator {
    /// Produce the (left, right) eye pair at the source resolution.
    pub fn generate_pair(&self, frame: &Frame, depth: &DepthMap) -> Result<(Frame, Frame)> {
        if frame.width != depth.width || frame.height != depth.height {
            return Err(PipelineError::DepthInference(format!(
                "frame {}x{} and depth map {}x{} must match",
                frame.width, frame.height, depth.width, depth.height
            )));
        }

        let w = frame.width as usize;
        let h = frame.height as usize;
        let bpr = w * 3;

        let mut left_bgr = vec![0u8; bpr * h];
        let mut left_filled = vec![false; w * h];
        let mut right_bgr = vec![0u8; bpr * h];
        let mut right_filled = vec![false; w * h];

        // Row-parallel forward warp. Each row writes to its own slice of the output
        // buffers, so there's no cross-row aliasing.
        left_bgr
            .par_chunks_mut(bpr)
            .zip(left_filled.par_chunks_mut(w))
            .zip(right_bgr.par_chunks_mut(bpr))
            .zip(right_filled.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (((lb, lm), rb), rm))| {
                let src_row = &frame.bgr[y * bpr..(y + 1) * bpr];
                let depth_row = &depth.data[y * w..(y + 1) * w];
                warp_row(src_row, depth_row, self.params(), lb, lm, rb, rm);
            });

        // Inpaint disocclusions per row.
        match self.inpaint {
            InpaintMethod::Simple => {
                left_bgr
                    .par_chunks_mut(bpr)
                    .zip(left_filled.par_chunks(w))
                    .for_each(|(row, mask)| inpaint_row_simple(row, mask));
                right_bgr
                    .par_chunks_mut(bpr)
                    .zip(right_filled.par_chunks(w))
                    .for_each(|(row, mask)| inpaint_row_simple(row, mask));
            }
            InpaintMethod::Telea | InpaintMethod::NavierStokes => {
                return Err(PipelineError::NotImplemented(
                    "stereo::InpaintMethod::{Telea,NavierStokes} — port TBD",
                ));
            }
        }

        debug!(width = w, height = h, baseline = self.baseline_px, "generated stereo pair");

        Ok((
            Frame { index: frame.index, width: frame.width, height: frame.height, bgr: left_bgr },
            Frame { index: frame.index, width: frame.width, height: frame.height, bgr: right_bgr },
        ))
    }

    /// Combine a stereo pair into a single frame in the requested layout.
    pub fn pack(&self, left: &Frame, right: &Frame, fmt: OutputFormat) -> Result<Frame> {
        if left.width != right.width || left.height != right.height {
            return Err(PipelineError::DepthInference(format!(
                "stereo pair size mismatch: left {}x{} vs right {}x{}",
                left.width, left.height, right.width, right.height
            )));
        }
        match fmt {
            OutputFormat::SideBySide => Ok(pack_sbs(left, right)),
            OutputFormat::TopBottom => Ok(pack_tb(left, right)),
            OutputFormat::Anaglyph => Ok(pack_anaglyph(left, right)),
            OutputFormat::FrameSequential => Err(PipelineError::NotImplemented(
                "stereo::pack(FrameSequential) — meaningful only during video encoding (Phase 5)",
            )),
        }
    }

    fn params(&self) -> WarpParams {
        WarpParams {
            baseline: self.baseline_px,
            convergence: self.convergence,
            max_disparity: self.max_disparity,
        }
    }
}

// ---- row-level warp ---------------------------------------------------------

#[derive(Clone, Copy)]
struct WarpParams { baseline: i32, convergence: f32, max_disparity: i32 }

fn warp_row(
    src: &[u8],
    depth: &[f32],
    p: WarpParams,
    left_out: &mut [u8],
    left_mask: &mut [bool],
    right_out: &mut [u8],
    right_mask: &mut [bool],
) {
    let w = depth.len();
    // Per-destination-column z-buffer (track best depth written so far).
    let mut left_z = vec![-1f32; w];
    let mut right_z = vec![-1f32; w];

    for x in 0..w {
        let d = depth[x];
        let signed_shift = (p.baseline as f32 * (d - p.convergence)).round() as i32;
        let shift = signed_shift.clamp(-p.max_disparity, p.max_disparity);
        let half = shift.div_euclid(2); // symmetric split

        let lx = x as i32 + half;
        if lx >= 0 && (lx as usize) < w {
            let i = lx as usize;
            if d >= left_z[i] {
                let s = x * 3;
                let o = i * 3;
                left_out[o..o + 3].copy_from_slice(&src[s..s + 3]);
                left_z[i] = d;
                left_mask[i] = true;
            }
        }

        let rx = x as i32 - half;
        if rx >= 0 && (rx as usize) < w {
            let i = rx as usize;
            if d >= right_z[i] {
                let s = x * 3;
                let o = i * 3;
                right_out[o..o + 3].copy_from_slice(&src[s..s + 3]);
                right_z[i] = d;
                right_mask[i] = true;
            }
        }
    }
}

// ---- inpainting -------------------------------------------------------------

fn inpaint_row_simple(row: &mut [u8], mask: &[bool]) {
    let w = mask.len();
    let mut x = 0;
    while x < w {
        if mask[x] { x += 1; continue; }
        let hole_start = x;
        while x < w && !mask[x] { x += 1; }
        let hole_end = x;

        // Snapshot neighbour colours to avoid borrow conflicts during write-back.
        let left: Option<[u8; 3]> = if hole_start > 0 {
            let o = (hole_start - 1) * 3;
            Some([row[o], row[o + 1], row[o + 2]])
        } else { None };
        let right: Option<[u8; 3]> = if hole_end < w {
            let o = hole_end * 3;
            Some([row[o], row[o + 1], row[o + 2]])
        } else { None };

        let span = (hole_end - hole_start) as f32;
        for px in hole_start..hole_end {
            let o = px * 3;
            match (left, right) {
                (Some(l), Some(r)) => {
                    let t = ((px - hole_start) as f32 + 1.0) / (span + 1.0);
                    for c in 0..3 {
                        row[o + c] = ((1.0 - t) * l[c] as f32 + t * r[c] as f32).round() as u8;
                    }
                }
                (Some(c), None) | (None, Some(c)) => {
                    row[o..o + 3].copy_from_slice(&c);
                }
                (None, None) => { /* row is entirely holes — leave as zeros */ }
            }
        }
    }
}

// ---- packing ----------------------------------------------------------------

fn pack_sbs(left: &Frame, right: &Frame) -> Frame {
    let w = left.width as usize;
    let h = left.height as usize;
    let bpr = w * 3;
    let out_bpr = bpr * 2;
    let mut bgr = vec![0u8; out_bpr * h];
    for y in 0..h {
        let src_l = &left.bgr[y * bpr..(y + 1) * bpr];
        let src_r = &right.bgr[y * bpr..(y + 1) * bpr];
        let dst = &mut bgr[y * out_bpr..(y + 1) * out_bpr];
        dst[..bpr].copy_from_slice(src_l);
        dst[bpr..].copy_from_slice(src_r);
    }
    Frame { index: left.index, width: left.width * 2, height: left.height, bgr }
}

fn pack_tb(left: &Frame, right: &Frame) -> Frame {
    let bpr = (left.width as usize) * 3;
    let h = left.height as usize;
    let mut bgr = Vec::with_capacity(bpr * h * 2);
    bgr.extend_from_slice(&left.bgr);
    bgr.extend_from_slice(&right.bgr);
    Frame { index: left.index, width: left.width, height: left.height * 2, bgr }
}

/// Red/cyan anaglyph: R from left eye, G+B from right eye. BGR → idx 0=B, 1=G, 2=R.
fn pack_anaglyph(left: &Frame, right: &Frame) -> Frame {
    let n = left.bgr.len();
    let mut bgr = vec![0u8; n];
    for i in (0..n).step_by(3) {
        bgr[i]     = right.bgr[i];     // B
        bgr[i + 1] = right.bgr[i + 1]; // G
        bgr[i + 2] = left.bgr[i + 2];  // R
    }
    Frame { index: left.index, width: left.width, height: left.height, bgr }
}

// ---- helpers for CLI preview -----------------------------------------------

/// Write a packed `Frame` as a PNG. Deliberately placed here so Phase 5's video
/// writer can own the moving-image side cleanly.
pub fn write_frame_as_png(frame: &Frame, path: &std::path::Path) -> Result<()> {
    let mut img = image::RgbImage::new(frame.width, frame.height);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = ((y * frame.width + x) * 3) as usize;
            let b = frame.bgr[i];
            let g = frame.bgr[i + 1];
            let r = frame.bgr[i + 2];
            img.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    img.save(path)?;
    Ok(())
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, bgr: [u8; 3]) -> Frame {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w as usize * h as usize) {
            data.extend_from_slice(&bgr);
        }
        Frame { index: 0, width: w, height: h, bgr: data }
    }

    fn uniform_depth(w: u32, h: u32, v: f32) -> DepthMap {
        DepthMap { width: w, height: h, data: vec![v; (w * h) as usize] }
    }

    #[test]
    fn uniform_depth_at_convergence_yields_identical_eyes() {
        let frame = solid_frame(16, 4, [10, 20, 30]);
        let depth = uniform_depth(16, 4, 0.5);
        let gen = StereoGenerator::default();
        let (l, r) = gen.generate_pair(&frame, &depth).unwrap();
        assert_eq!(l.bgr, frame.bgr);
        assert_eq!(r.bgr, frame.bgr);
    }

    #[test]
    fn pack_sbs_doubles_width() {
        let left = solid_frame(4, 2, [1, 1, 1]);
        let right = solid_frame(4, 2, [2, 2, 2]);
        let packed = StereoGenerator::default()
            .pack(&left, &right, OutputFormat::SideBySide)
            .unwrap();
        assert_eq!(packed.width, 8);
        assert_eq!(packed.height, 2);
        assert_eq!(&packed.bgr[0..12], &[1u8; 12]);
        assert_eq!(&packed.bgr[12..24], &[2u8; 12]);
    }

    #[test]
    fn pack_tb_doubles_height() {
        let left = solid_frame(2, 2, [1, 1, 1]);
        let right = solid_frame(2, 2, [2, 2, 2]);
        let packed = StereoGenerator::default()
            .pack(&left, &right, OutputFormat::TopBottom)
            .unwrap();
        assert_eq!(packed.width, 2);
        assert_eq!(packed.height, 4);
        assert_eq!(&packed.bgr[..12], &[1u8; 12]);
        assert_eq!(&packed.bgr[12..], &[2u8; 12]);
    }

    #[test]
    fn pack_anaglyph_multiplexes_channels() {
        // Left is pure red (BGR = 0,0,255); right is pure cyan (BGR = 255,255,0).
        // The anaglyph overlay should give pure white (255,255,255) in BGR.
        let left = solid_frame(2, 1, [0, 0, 255]);
        let right = solid_frame(2, 1, [255, 255, 0]);
        let packed = StereoGenerator::default()
            .pack(&left, &right, OutputFormat::Anaglyph)
            .unwrap();
        for px in packed.bgr.chunks(3) {
            assert_eq!(px, &[255u8, 255, 255]);
        }
    }

    #[test]
    fn frame_sequential_is_not_implemented() {
        let left = solid_frame(2, 2, [0, 0, 0]);
        let right = solid_frame(2, 2, [0, 0, 0]);
        let err = StereoGenerator::default()
            .pack(&left, &right, OutputFormat::FrameSequential)
            .unwrap_err();
        assert!(matches!(err, PipelineError::NotImplemented(_)));
    }

    #[test]
    fn inpaint_row_blends_between_neighbours() {
        // Row of 4 pixels: filled red | hole | hole | filled blue.
        let mut row: Vec<u8> = vec![0, 0, 255,  0, 0, 0,  0, 0, 0,  255, 0, 0];
        let mask = vec![true, false, false, true];
        inpaint_row_simple(&mut row, &mask);
        for px in [1, 2] {
            let o = px * 3;
            let (b, _g, r) = (row[o], row[o + 1], row[o + 2]);
            assert!(b > 0 || r > 0, "hole {px} unfilled: BGR = ({b},{},{r})", row[o + 1]);
        }
    }

    #[test]
    fn warp_shifts_near_pixel_outward() {
        // 7-wide row with a bright pixel at column 3 and depth 1 (near plane).
        // baseline=20, convergence=0, max_disparity=5 → shift clamps to 5, half=2.
        // So the left eye should see the pixel at x=5 and the right eye at x=1.
        let mut src = vec![0u8; 7 * 3];
        for c in 0..3 { src[3 * 3 + c] = 200; }
        let mut depth = vec![0f32; 7];
        depth[3] = 1.0;

        let frame = Frame { index: 0, width: 7, height: 1, bgr: src };
        let dm = DepthMap { width: 7, height: 1, data: depth };
        let gen = StereoGenerator {
            baseline_px: 20, max_disparity: 5, convergence: 0.0,
            inpaint: InpaintMethod::Simple,
        };
        let (l, r) = gen.generate_pair(&frame, &dm).unwrap();
        assert_eq!(&l.bgr[5 * 3..5 * 3 + 3], &[200u8, 200, 200]);
        assert_eq!(&r.bgr[1 * 3..1 * 3 + 3], &[200u8, 200, 200]);
    }
}
