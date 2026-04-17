//! Face alignment helpers.
//!
//! InsightFace's pipeline aligns faces to a canonical template of 5 landmarks
//! (left eye, right eye, nose tip, left mouth corner, right mouth corner)
//! using a similarity transform. We compute that 2x3 affine matrix and warp
//! the source image into a fixed-size crop (112x112 for ArcFace,
//! 128x128 for inswapper).

use image::{Rgb, RgbImage};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};

/// Canonical ArcFace 5-point landmark template at 112x112 (pixel coords).
pub const ARCFACE_TEMPLATE_112: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Template for inswapper_128. InsightFace uses the ArcFace 112 template
/// shifted by +8 px in x (diff_x = 8 when image_size=128), NOT scaled.
pub const INSWAPPER_TEMPLATE_128: [[f32; 2]; 5] = [
    [38.2946 + 8.0, 51.6963],
    [73.5318 + 8.0, 51.5014],
    [56.0252 + 8.0, 71.7366],
    [41.5493 + 8.0, 92.3655],
    [70.7299 + 8.0, 92.2041],
];

/// 5 points in image space.
pub type Landmarks5 = [[f32; 2]; 5];

/// Estimate a 2D similarity transform (uniform scale + rotation + translation)
/// mapping `src` points to `dst` points, least-squares.
///
/// Treating each point as a complex number z = x + i·y, a similarity transform
/// is z' = a·z + b where `a` is complex (encoding scale·rotation) and `b` is
/// the complex translation. The closed-form least-squares solution is
///   a = Σ conj(p_i)·q_i / Σ |p_i|²
///   b = mean(dst) − a·mean(src)
/// (where p, q are mean-centred). This is numerically stable and avoids
/// the sign-ambiguity that trips up hand-rolled 2×2 SVDs.
///
/// Returns the 2×3 matrix M such that `M · [x,y,1]ᵀ ≈ dst` point for point.
pub fn umeyama_similarity(src: &Landmarks5, dst: &Landmarks5) -> [[f32; 3]; 2] {
    let (msx, msy) = mean(src);
    let (mdx, mdy) = mean(dst);

    let mut num_r = 0.0f32; // Re(Σ conj(p)·q)
    let mut num_i = 0.0f32; // Im(Σ conj(p)·q)
    let mut denom = 0.0f32; // Σ |p|²
    for i in 0..5 {
        let px = src[i][0] - msx;
        let py = src[i][1] - msy;
        let qx = dst[i][0] - mdx;
        let qy = dst[i][1] - mdy;
        // conj(p)·q = (px − i·py)·(qx + i·qy) = (px·qx + py·qy) + i·(px·qy − py·qx)
        num_r += px * qx + py * qy;
        num_i += px * qy - py * qx;
        denom += px * px + py * py;
    }
    let denom = denom.max(1e-12);
    let ax = num_r / denom; // scale·cosθ
    let ay = num_i / denom; // scale·sinθ

    // z' = a·z + b with a = ax + i·ay:
    //   x' = ax·x − ay·y + tx
    //   y' = ay·x + ax·y + ty
    let tx = mdx - (ax * msx - ay * msy);
    let ty = mdy - (ay * msx + ax * msy);

    [[ax, -ay, tx], [ay, ax, ty]]
}

fn mean(pts: &Landmarks5) -> (f32, f32) {
    let mut mx = 0.0;
    let mut my = 0.0;
    for p in pts {
        mx += p[0];
        my += p[1];
    }
    (mx / 5.0, my / 5.0)
}

/// Warp `src` into a new `size x size` image using the 2x3 affine `m` that
/// maps *source* coords → *destination* coords. Bilinear, black background.
pub fn warp_affine(src: &RgbImage, m: [[f32; 3]; 2], size: u32) -> RgbImage {
    // imageproc's Projection is the src→dst mapping; `warp_into` inverts it
    // internally to iterate over destination pixels.
    let proj = Projection::from_matrix([
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], 0.0, 0.0, 1.0,
    ])
    .unwrap_or(Projection::scale(1.0, 1.0));

    let mut out = RgbImage::from_pixel(size, size, Rgb([0, 0, 0]));
    warp_into(src, &proj, Interpolation::Bilinear, Rgb([0, 0, 0]), &mut out);
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

/// Compose `a` then `b` (i.e. `b * a`) as 2x3 affines (last row implicit).
pub fn compose(b: [[f32; 3]; 2], a: [[f32; 3]; 2]) -> [[f32; 3]; 2] {
    [
        [
            b[0][0] * a[0][0] + b[0][1] * a[1][0],
            b[0][0] * a[0][1] + b[0][1] * a[1][1],
            b[0][0] * a[0][2] + b[0][1] * a[1][2] + b[0][2],
        ],
        [
            b[1][0] * a[0][0] + b[1][1] * a[1][0],
            b[1][0] * a[0][1] + b[1][1] * a[1][1],
            b[1][0] * a[0][2] + b[1][1] * a[1][2] + b[1][2],
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_src_eq_dst() {
        let pts: Landmarks5 = ARCFACE_TEMPLATE_112;
        let m = umeyama_similarity(&pts, &pts);
        // expect near-identity
        assert!((m[0][0] - 1.0).abs() < 1e-3);
        assert!((m[1][1] - 1.0).abs() < 1e-3);
        assert!(m[0][1].abs() < 1e-3);
        assert!(m[0][2].abs() < 1e-1);
    }

    #[test]
    fn scale_2x() {
        let src = ARCFACE_TEMPLATE_112;
        let dst: Landmarks5 = std::array::from_fn(|i| [src[i][0] * 2.0, src[i][1] * 2.0]);
        let m = umeyama_similarity(&src, &dst);
        assert!((m[0][0] - 2.0).abs() < 1e-2);
        assert!((m[1][1] - 2.0).abs() < 1e-2);
    }
}
