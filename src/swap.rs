//! Face swap via InsightFace's `inswapper_128.onnx`.
//!
//! Inputs:
//!   - target: the painting face aligned to 128x128 via the ArcFace template
//!   - source: the user's 512-d ArcFace embedding
//! Output: a 128x128 face-swapped crop, pasted back into the painting using
//! the inverse alignment transform and a feathered oval mask.

use crate::align::{self, INSWAPPER_TEMPLATE_128};
use crate::face::Face;
use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("ort: {e}")
}

/// Per-phase accumulated timings across a capture batch.
#[derive(Default, Clone, Copy)]
pub struct Timings {
    pub align: std::time::Duration,
    pub infer: std::time::Duration,
    pub paste: std::time::Duration,
    pub count: usize,
}

pub struct Swapper {
    session: Session,
    /// 512×512 row-major float32 matrix baked into the inswapper ONNX graph.
    /// The source embedding must be `latent = embedding @ emap` (and then
    /// L2-normalised) before being fed to the model — without this the
    /// network emits garbage.
    emap: Vec<f32>,
}

impl Swapper {
    pub fn load(models_dir: &Path) -> Result<Self> {
        let path = models_dir.join("inswapper_128.onnx");
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(threads)
            .map_err(ort_err)?
            .commit_from_file(&path)
            .map_err(ort_err)
            .with_context(|| format!("load swapper from {}", path.display()))?;

        let emap_path = models_dir.join("emap.bin");
        let emap_bytes = std::fs::read(&emap_path).with_context(|| {
            format!(
                "load emap from {} — run scripts/extract_emap.py to generate",
                emap_path.display()
            )
        })?;
        if emap_bytes.len() != 512 * 512 * 4 {
            anyhow::bail!(
                "emap.bin size {} != expected {} (512×512 f32)",
                emap_bytes.len(),
                512 * 512 * 4
            );
        }
        let emap: Vec<f32> = emap_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(Self { session, emap })
    }

    /// Apply emap matrix to an L2-normalised 512-d embedding and re-normalise.
    /// This is the "latent" tensor fed to inswapper as the source input.
    /// Call it once per capture and pass the result into `swap_with_latent`.
    pub fn latent_from_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        let mut latent = vec![0.0f32; 512];
        for j in 0..512 {
            let mut s = 0.0;
            for i in 0..512 {
                s += embedding[i] * self.emap[i * 512 + j];
            }
            latent[j] = s;
        }
        let norm: f32 = latent.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in &mut latent {
            *v /= norm;
        }
        latent
    }

    pub fn swap(
        &mut self,
        target_image: &RgbImage,
        target_face: &Face,
        source_embedding: &[f32],
    ) -> Result<RgbImage> {
        let latent = self.latent_from_embedding(source_embedding);
        self.swap_with_latent(target_image, target_face, &latent, None)
    }

    /// Like `swap` but takes a pre-computed latent (reuse across N paintings
    /// in one capture). Also optionally reports per-stage timings.
    pub fn swap_with_latent(
        &mut self,
        target_image: &RgbImage,
        target_face: &Face,
        latent: &[f32],
        timings: Option<&mut Timings>,
    ) -> Result<RgbImage> {
        use std::time::Instant;
        let t_align = Instant::now();
        let m = align::umeyama_similarity(&target_face.landmarks, &INSWAPPER_TEMPLATE_128);
        let aligned = align::warp_affine(target_image, m, 128);
        let align_ms = t_align.elapsed();
        if std::env::var("CLASSIC_ME_DEBUG").is_ok() {
            let _ = std::fs::create_dir_all("debug");
            let _ = aligned.save("debug/aligned_target.png");
        }

        let target_data = image_to_chw_01(&aligned);
        let mut source_data = vec![0.0f32; 512];
        source_data[..latent.len().min(512)]
            .copy_from_slice(&latent[..latent.len().min(512)]);
        let target_tensor =
            Tensor::from_array(([1i64, 3, 128, 128], target_data)).map_err(ort_err)?;
        let source_tensor = Tensor::from_array(([1i64, 512], source_data)).map_err(ort_err)?;

        let out_name = self.session.outputs()[0].name().to_string();
        let t_infer = std::time::Instant::now();
        let outputs = self
            .session
            .run(ort::inputs![target_tensor, source_tensor])
            .map_err(ort_err)
            .context("run inswapper")?;
        let infer_ms = t_infer.elapsed();
        let (_, out_data) = outputs[out_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;

        let swapped = chw_to_image(out_data, 128, 128);
        if std::env::var("CLASSIC_ME_DEBUG").is_ok() {
            let _ = swapped.save("debug/swapped_crop.png");
        }

        let t_paste = std::time::Instant::now();
        let result = paste_back(target_image, &swapped, m);
        let paste_ms = t_paste.elapsed();

        if let Some(t) = timings {
            t.align += align_ms;
            t.infer += infer_ms;
            t.paste += paste_ms;
            t.count += 1;
        }
        Ok(result)
    }
}

fn image_to_chw_01(img: &RgbImage) -> Vec<f32> {
    let w = img.width();
    let h = img.height();
    let plane = (w * h) as usize;
    let mut data = vec![0.0f32; 3 * plane];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            let i = (y * w + x) as usize;
            data[i] = p[0] as f32 / 255.0;
            data[plane + i] = p[1] as f32 / 255.0;
            data[2 * plane + i] = p[2] as f32 / 255.0;
        }
    }
    data
}

fn chw_to_image(data: &[f32], w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let plane = (w * h) as usize;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let r = (data[i].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (data[plane + i].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (data[2 * plane + i].clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    img
}

fn paste_back(base: &RgbImage, crop: &RgbImage, m: [[f32; 3]; 2]) -> RgbImage {
    use rayon::prelude::*;
    let mask = build_oval_mask(128, 128, 0.85);
    let inv = invert_affine(m);
    let warped_crop = warp_rgb_with_affine(crop, inv, base.width(), base.height());
    let warped_mask = warp_gray_with_affine(&mask, inv, base.width(), base.height());

    let w = base.width() as usize;
    let h = base.height() as usize;
    let base_buf = base.as_raw(); // &[u8], RGB RGB RGB...
    let crop_buf = warped_crop.as_raw();
    let mask_buf = warped_mask.as_raw();

    // Row-parallel alpha composite. For a 1000×1000 painting this drops the
    // blend step from ~80ms to ~15ms on 8 cores.
    let mut out_vec = vec![0u8; w * h * 3];
    out_vec
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let mi = y * w + x;
                let a = mask_buf[mi] as f32 * (1.0 / 255.0);
                let pi = mi * 3;
                if a <= 0.0 {
                    row[x * 3] = base_buf[pi];
                    row[x * 3 + 1] = base_buf[pi + 1];
                    row[x * 3 + 2] = base_buf[pi + 2];
                } else {
                    let ia = 1.0 - a;
                    row[x * 3] =
                        (crop_buf[pi] as f32 * a + base_buf[pi] as f32 * ia) as u8;
                    row[x * 3 + 1] =
                        (crop_buf[pi + 1] as f32 * a + base_buf[pi + 1] as f32 * ia) as u8;
                    row[x * 3 + 2] =
                        (crop_buf[pi + 2] as f32 * a + base_buf[pi + 2] as f32 * ia) as u8;
                }
            }
        });
    RgbImage::from_raw(w as u32, h as u32, out_vec).expect("RgbImage from_raw")
}

fn build_oval_mask(w: u32, h: u32, radius_frac: f32) -> image::GrayImage {
    let mut mask = image::GrayImage::new(w, h);
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let rx = w as f32 * 0.5 * radius_frac;
    let ry = h as f32 * 0.55 * radius_frac;
    let feather = 0.15 * w.max(h) as f32;
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            let r = (dx * dx + dy * dy).sqrt();
            let v = if r < 1.0 - feather / rx {
                1.0
            } else if r > 1.0 {
                0.0
            } else {
                let t = (1.0 - r) / (feather / rx);
                t.clamp(0.0, 1.0)
            };
            mask.put_pixel(x, y, image::Luma([(v * 255.0) as u8]));
        }
    }
    mask
}

fn warp_rgb_with_affine(src: &RgbImage, m: [[f32; 3]; 2], w: u32, h: u32) -> RgbImage {
    use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
    let proj = Projection::from_matrix([
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], 0.0, 0.0, 1.0,
    ])
    .unwrap_or(Projection::scale(1.0, 1.0));
    let mut out = RgbImage::from_pixel(w, h, Rgb([0, 0, 0]));
    warp_into(src, &proj, Interpolation::Bilinear, Rgb([0, 0, 0]), &mut out);
    out
}

fn warp_gray_with_affine(
    src: &image::GrayImage,
    m: [[f32; 3]; 2],
    w: u32,
    h: u32,
) -> image::GrayImage {
    use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
    let proj = Projection::from_matrix([
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], 0.0, 0.0, 1.0,
    ])
    .unwrap_or(Projection::scale(1.0, 1.0));
    let mut out = image::GrayImage::from_pixel(w, h, image::Luma([0]));
    warp_into(src, &proj, Interpolation::Bilinear, image::Luma([0]), &mut out);
    out
}

fn invert_affine(m: [[f32; 3]; 2]) -> [[f32; 3]; 2] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[1][0];
    let d = m[1][1];
    let tx = m[0][2];
    let ty = m[1][2];
    let det = a * d - b * c;
    let inv_det = if det.abs() < 1e-8 { 0.0 } else { 1.0 / det };
    let ia = d * inv_det;
    let ib = -b * inv_det;
    let ic = -c * inv_det;
    let id = a * inv_det;
    let itx = -(ia * tx + ib * ty);
    let ity = -(ic * tx + id * ty);
    [[ia, ib, itx], [ic, id, ity]]
}
