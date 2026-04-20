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
use image::RgbImage;
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
    /// 128×128 feathered oval mask. Identical for every swap, so we build it
    /// once at load time and reuse it in `paste_back`.
    mask: image::GrayImage,
}

impl Swapper {
    pub fn load(models_dir: &Path) -> Result<Self> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self::load_with_threads(models_dir, threads)
    }

    /// Like `load`, but lets the caller control intra-op thread count — used
    /// when building a pool of parallel swappers so their threadpools don't
    /// oversubscribe the CPU.
    ///
    /// On Windows the session additionally registers the DirectML execution
    /// provider, which pushes inswapper inference onto any DX12-capable GPU.
    /// Ops DML can't run fall back to CPU transparently. Set
    /// `FACENAISSANCE_NO_DML=1` to disable DML at startup (useful if a
    /// driver/GPU combo produces glitched output).
    pub fn load_with_threads(models_dir: &Path, threads: usize) -> Result<Self> {
        let path = models_dir.join("inswapper_128.onnx");
        let mut builder = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(threads)
            .map_err(ort_err)?;

        #[cfg(target_os = "windows")]
        let mut builder = {
            let disable_dml = std::env::var("FACENAISSANCE_NO_DML")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if disable_dml {
                log::info!("inswapper: DirectML disabled via FACENAISSANCE_NO_DML");
                builder
            } else {
                use ort::execution_providers::DirectMLExecutionProvider;
                match builder
                    .with_execution_providers([DirectMLExecutionProvider::default().build()])
                {
                    Ok(b) => {
                        log::info!("inswapper: DirectML execution provider enabled");
                        b
                    }
                    Err(e) => {
                        log::warn!(
                            "inswapper: could not register DirectML EP ({e}); falling back to CPU"
                        );
                        Session::builder()
                            .map_err(ort_err)?
                            .with_optimization_level(GraphOptimizationLevel::Level3)
                            .map_err(ort_err)?
                            .with_intra_threads(threads)
                            .map_err(ort_err)?
                    }
                }
            }
        };

        let session = builder
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

        let mask = build_oval_mask(128, 128, 0.85);

        Ok(Self { session, emap, mask })
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
        let result = paste_back(target_image, &swapped, &self.mask, m);
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
    let w = img.width() as usize;
    let h = img.height() as usize;
    let plane = w * h;
    let raw = img.as_raw(); // packed RGBRGB...
    let mut data = vec![0.0f32; 3 * plane];
    let (r_plane, rest) = data.split_at_mut(plane);
    let (g_plane, b_plane) = rest.split_at_mut(plane);
    let scale = 1.0 / 255.0;
    for i in 0..plane {
        let p = i * 3;
        r_plane[i] = raw[p] as f32 * scale;
        g_plane[i] = raw[p + 1] as f32 * scale;
        b_plane[i] = raw[p + 2] as f32 * scale;
    }
    data
}

fn chw_to_image(data: &[f32], w: u32, h: u32) -> RgbImage {
    let plane = (w * h) as usize;
    let r_plane = &data[0..plane];
    let g_plane = &data[plane..2 * plane];
    let b_plane = &data[2 * plane..3 * plane];
    let mut out = vec![0u8; plane * 3];
    for i in 0..plane {
        let p = i * 3;
        out[p] = (r_plane[i].clamp(0.0, 1.0) * 255.0) as u8;
        out[p + 1] = (g_plane[i].clamp(0.0, 1.0) * 255.0) as u8;
        out[p + 2] = (b_plane[i].clamp(0.0, 1.0) * 255.0) as u8;
    }
    RgbImage::from_raw(w, h, out).expect("chw_to_image from_raw")
}

/// Fused bbox-limited paste: warp the 128×128 crop+mask into the base frame
/// *only within the aligned crop's footprint*, doing bilinear sampling and
/// alpha blending in a single pass. Avoids the two full-frame `warp_into`
/// calls and the full-frame blend that the old implementation did.
fn paste_back(
    base: &RgbImage,
    crop: &RgbImage,
    mask: &image::GrayImage,
    m: [[f32; 3]; 2],
) -> RgbImage {
    let bw = base.width() as i32;
    let bh = base.height() as i32;
    let cw = crop.width() as i32; // 128
    let ch = crop.height() as i32; // 128

    // Compute the base-image bbox that the 128×128 aligned square projects
    // back to, by sending the four corners through the inverse affine.
    let inv = invert_affine(m);
    let corners = [
        (0.0f32, 0.0f32),
        (cw as f32, 0.0),
        (0.0, ch as f32),
        (cw as f32, ch as f32),
    ];
    let (mut minx, mut miny) = (f32::INFINITY, f32::INFINITY);
    let (mut maxx, mut maxy) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (ax, ay) in corners {
        let px = inv[0][0] * ax + inv[0][1] * ay + inv[0][2];
        let py = inv[1][0] * ax + inv[1][1] * ay + inv[1][2];
        if px < minx { minx = px; }
        if py < miny { miny = py; }
        if px > maxx { maxx = px; }
        if py > maxy { maxy = py; }
    }
    let x0 = (minx.floor() as i32).clamp(0, bw);
    let y0 = (miny.floor() as i32).clamp(0, bh);
    let x1 = ((maxx.ceil() as i32) + 1).clamp(0, bw);
    let y1 = ((maxy.ceil() as i32) + 1).clamp(0, bh);

    let base_buf = base.as_raw();
    let crop_buf = crop.as_raw();
    let mask_buf = mask.as_raw();
    let mut out_vec: Vec<u8> = base_buf.clone();

    let cw_us = cw as usize;
    // Per-row we only need to recompute (ax,ay) from scratch at x=x0, then
    // increment by (m[0][0], m[1][0]) for each x++.
    for y in y0..y1 {
        let base_ax = m[0][1] * (y as f32) + m[0][2];
        let base_ay = m[1][1] * (y as f32) + m[1][2];
        let mut ax = base_ax + m[0][0] * (x0 as f32);
        let mut ay = base_ay + m[1][0] * (x0 as f32);
        for x in x0..x1 {
            let px = ax;
            let py = ay;
            ax += m[0][0];
            ay += m[1][0];
            // Skip if outside the 128×128 aligned square. Use strict `>=` on
            // the upper bound because we sample at (ix, ix+1) / (iy, iy+1)
            // for bilinear interp, so the last valid source coord is cw-1
            // *exclusive*.
            if px < 0.0 || py < 0.0 || px >= (cw as f32) - 1.0 || py >= (ch as f32) - 1.0 {
                continue;
            }
            let ix = px as i32;
            let iy = py as i32;
            let dx = px - ix as f32;
            let dy = py - iy as f32;
            let w00 = (1.0 - dx) * (1.0 - dy);
            let w10 = dx * (1.0 - dy);
            let w01 = (1.0 - dx) * dy;
            let w11 = dx * dy;

            let m00 = mask_buf[(iy as usize) * cw_us + ix as usize] as f32;
            let m10 = mask_buf[(iy as usize) * cw_us + (ix + 1) as usize] as f32;
            let m01 = mask_buf[((iy + 1) as usize) * cw_us + ix as usize] as f32;
            let m11 = mask_buf[((iy + 1) as usize) * cw_us + (ix + 1) as usize] as f32;
            let alpha = (m00 * w00 + m10 * w10 + m01 * w01 + m11 * w11) * (1.0 / 255.0);
            if alpha <= 0.0 {
                continue;
            }

            let c00 = ((iy as usize) * cw_us + ix as usize) * 3;
            let c10 = ((iy as usize) * cw_us + (ix + 1) as usize) * 3;
            let c01 = (((iy + 1) as usize) * cw_us + ix as usize) * 3;
            let c11 = (((iy + 1) as usize) * cw_us + (ix + 1) as usize) * 3;
            let r = crop_buf[c00] as f32 * w00
                + crop_buf[c10] as f32 * w10
                + crop_buf[c01] as f32 * w01
                + crop_buf[c11] as f32 * w11;
            let g = crop_buf[c00 + 1] as f32 * w00
                + crop_buf[c10 + 1] as f32 * w10
                + crop_buf[c01 + 1] as f32 * w01
                + crop_buf[c11 + 1] as f32 * w11;
            let b = crop_buf[c00 + 2] as f32 * w00
                + crop_buf[c10 + 2] as f32 * w10
                + crop_buf[c01 + 2] as f32 * w01
                + crop_buf[c11 + 2] as f32 * w11;

            let pi = ((y as usize) * (bw as usize) + x as usize) * 3;
            let ia = 1.0 - alpha;
            out_vec[pi] = (r * alpha + base_buf[pi] as f32 * ia) as u8;
            out_vec[pi + 1] = (g * alpha + base_buf[pi + 1] as f32 * ia) as u8;
            out_vec[pi + 2] = (b * alpha + base_buf[pi + 2] as f32 * ia) as u8;
        }
    }

    RgbImage::from_raw(bw as u32, bh as u32, out_vec).expect("RgbImage from_raw")
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
