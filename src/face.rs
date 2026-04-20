//! Face detection (SCRFD) and recognition (ArcFace) via ONNX Runtime.
//!
//! Both models come from the InsightFace project:
//!   - `det_10g.onnx`   — SCRFD 10GF detection + 5 landmarks (from `buffalo_l`)
//!   - `w600k_r50.onnx` — 512-d face embedding (ArcFace, from `buffalo_l`)

use crate::align::{self, Landmarks5, ARCFACE_TEMPLATE_112};
use anyhow::{Context, Result};
use image::RgbImage;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

/// `ort::Error`'s generic parameter pulls in !Send/!Sync types, so it doesn't
/// satisfy `anyhow::Error`'s bounds. Stringify it instead.
fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("ort: {e}")
}

fn build_session(path: &Path) -> Result<Session> {
    // Use all physical cores for intra-op parallelism. Each inswapper /
    // ArcFace / SCRFD run becomes several×  faster on multi-core CPUs.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) as i16;
    Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_intra_threads(threads as usize)
        .map_err(ort_err)?
        .commit_from_file(path)
        .map_err(ort_err)
        .with_context(|| format!("load ONNX from {}", path.display()))
}

/// A single detected face.
#[derive(Clone, Debug)]
pub struct Face {
    pub bbox: [f32; 4],       // x1, y1, x2, y2 (original-image pixels)
    pub landmarks: Landmarks5,
    pub score: f32,
}

pub struct FaceAnalyzer {
    detector: Session,
    recognizer: Session,
    det_size: u32,
}

impl FaceAnalyzer {
    pub fn load(models_dir: &Path) -> Result<Self> {
        let detector = build_session(&models_dir.join("det_10g.onnx"))?;
        let recognizer = build_session(&models_dir.join("w600k_r50.onnx"))?;
        Ok(Self {
            detector,
            recognizer,
            det_size: 640,
        })
    }

    /// Detect faces in `image`. Returns faces in original-image coordinates,
    /// sorted by score descending.
    pub fn detect(&mut self, image: &RgbImage) -> Result<Vec<Face>> {
        let (w, h) = (image.width(), image.height());
        let (resized, scale, pad_x, pad_y) = letterbox(image, self.det_size);

        // Cache output names before mutably borrowing the session with .run().
        // SCRFD has 9 outputs: {score, bbox, kps} × 3 FPN strides (8,16,32).
        let names: Vec<String> = self
            .detector
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // SCRFD input: [1,3,H,W], pixels in [-1,1] (mean 127.5 / std 128).
        let (shape, data) = image_to_chw(&resized, 127.5, 128.0);
        let input_tensor = Tensor::from_array((shape, data)).map_err(ort_err)?;

        let outputs = self
            .detector
            .run(ort::inputs![input_tensor])
            .map_err(ort_err)
            .context("run detector")?;

        let strides = [8u32, 16, 32];
        let mut detections: Vec<Face> = Vec::new();

        for (si, &stride) in strides.iter().enumerate() {
            let (_, scores) = outputs[names[si].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            let (_, bboxes) = outputs[names[si + 3].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            let (_, kps) = outputs[names[si + 6].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;

            let grid_w = self.det_size / stride;
            let grid_h = self.det_size / stride;
            let num_anchors = 2usize;

            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    for a in 0..num_anchors {
                        let idx = (((gy * grid_w) + gx) as usize) * num_anchors + a;
                        let score = scores[idx];
                        if score < 0.5 {
                            continue;
                        }
                        let bx = idx * 4;
                        let cx = (gx as f32 + 0.5) * stride as f32;
                        let cy = (gy as f32 + 0.5) * stride as f32;
                        let x1 = cx - bboxes[bx] * stride as f32;
                        let y1 = cy - bboxes[bx + 1] * stride as f32;
                        let x2 = cx + bboxes[bx + 2] * stride as f32;
                        let y2 = cy + bboxes[bx + 3] * stride as f32;

                        let kx = idx * 10;
                        let mut lm: Landmarks5 = [[0.0; 2]; 5];
                        for p in 0..5 {
                            lm[p][0] = cx + kps[kx + p * 2] * stride as f32;
                            lm[p][1] = cy + kps[kx + p * 2 + 1] * stride as f32;
                        }

                        // Undo letterbox.
                        let inv = |x: f32, pad: f32| (x - pad) / scale;
                        let x1o = inv(x1, pad_x).clamp(0.0, w as f32 - 1.0);
                        let y1o = inv(y1, pad_y).clamp(0.0, h as f32 - 1.0);
                        let x2o = inv(x2, pad_x).clamp(0.0, w as f32 - 1.0);
                        let y2o = inv(y2, pad_y).clamp(0.0, h as f32 - 1.0);
                        for p in 0..5 {
                            lm[p][0] = inv(lm[p][0], pad_x);
                            lm[p][1] = inv(lm[p][1], pad_y);
                        }

                        detections.push(Face {
                            bbox: [x1o, y1o, x2o, y2o],
                            landmarks: lm,
                            score,
                        });
                    }
                }
            }
        }

        detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(nms(detections, 0.4))
    }

    /// Compute a 512-d L2-normalised ArcFace embedding for the given face.
    pub fn embed(&mut self, image: &RgbImage, face: &Face) -> Result<Vec<f32>> {
        let (embedding, _) = self.embed_with_crop(image, face)?;
        Ok(embedding)
    }

    /// Same as `embed`, but also returns the 112×112 aligned face crop that
    /// was fed to ArcFace. Useful for sanity-checking alignment in the UI.
    pub fn embed_with_crop(
        &mut self,
        image: &RgbImage,
        face: &Face,
    ) -> Result<(Vec<f32>, RgbImage)> {
        let m = align::umeyama_similarity(&face.landmarks, &ARCFACE_TEMPLATE_112);
        let aligned = align::warp_affine(image, m, 112);
        let name = self.recognizer.outputs()[0].name().to_string();
        let (shape, data) = image_to_chw(&aligned, 127.5, 127.5); // ArcFace: (p-127.5)/127.5
        let input_tensor = Tensor::from_array((shape, data)).map_err(ort_err)?;

        let outputs = self
            .recognizer
            .run(ort::inputs![input_tensor])
            .map_err(ort_err)?;
        let (_, out) = outputs[name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        Ok((l2_normalize(out.to_vec()), aligned))
    }
}

fn letterbox(img: &RgbImage, size: u32) -> (RgbImage, f32, f32, f32) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let scale = (size as f32 / w).min(size as f32 / h);
    let new_w = (w * scale) as u32;
    let new_h = (h * scale) as u32;
    let resized = image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Triangle);
    let mut out = RgbImage::from_pixel(size, size, image::Rgb([0, 0, 0]));
    let pad_x = ((size - new_w) / 2) as f32;
    let pad_y = ((size - new_h) / 2) as f32;
    image::imageops::overlay(&mut out, &resized, pad_x as i64, pad_y as i64);
    (out, scale, pad_x, pad_y)
}

/// Return a normalised image as ([1,3,H,W] shape, channel-major Vec<f32>).
fn image_to_chw(img: &RgbImage, mean: f32, std: f32) -> ([i64; 4], Vec<f32>) {
    let w = img.width() as i64;
    let h = img.height() as i64;
    let plane = (w * h) as usize;
    let raw = img.as_raw(); // packed RGBRGB...
    let mut data = vec![0.0f32; 3 * plane];
    let (r_plane, rest) = data.split_at_mut(plane);
    let (g_plane, b_plane) = rest.split_at_mut(plane);
    let inv_std = 1.0 / std;
    for i in 0..plane {
        let p = i * 3;
        r_plane[i] = (raw[p] as f32 - mean) * inv_std;
        g_plane[i] = (raw[p + 1] as f32 - mean) * inv_std;
        b_plane[i] = (raw[p + 2] as f32 - mean) * inv_std;
    }
    ([1, 3, h, w], data)
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

fn nms(mut dets: Vec<Face>, iou_thresh: f32) -> Vec<Face> {
    let mut keep: Vec<Face> = Vec::new();
    while !dets.is_empty() {
        let best = dets.remove(0);
        dets.retain(|f| iou(&best.bbox, &f.bbox) < iou_thresh);
        keep.push(best);
    }
    keep
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    let inter = w * h;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter + 1e-6;
    inter / union
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

pub fn largest_face(faces: &[Face]) -> Option<&Face> {
    faces.iter().max_by(|a, b| {
        let aa = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
        let bb = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
        aa.partial_cmp(&bb).unwrap()
    })
}
