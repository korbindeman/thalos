use glam::Vec3;
use crate::cubemap::{CubemapFace, face_uv_to_dir};

/// Normal vector for a cubemap face.
pub fn face_normal(face: CubemapFace) -> Vec3 {
    face_basis(face).0
}

/// Orthonormal face basis `(n, u, v)` matching `face_uv_to_dir`. For a texel
/// with sc = 2u−1, tc = 2v−1, the unnormalized direction is `n + sc·u + tc·v`.
#[inline]
pub fn face_basis(face: CubemapFace) -> (Vec3, Vec3, Vec3) {
    match face {
        // PosX: dir = ( 1, -tc, -sc)
        CubemapFace::PosX => (Vec3::X, Vec3::NEG_Z, Vec3::NEG_Y),
        // NegX: dir = (-1, -tc,  sc)
        CubemapFace::NegX => (Vec3::NEG_X, Vec3::Z, Vec3::NEG_Y),
        // PosY: dir = ( sc,  1,  tc)
        CubemapFace::PosY => (Vec3::Y, Vec3::X, Vec3::Z),
        // NegY: dir = ( sc, -1, -tc)
        CubemapFace::NegY => (Vec3::NEG_Y, Vec3::X, Vec3::NEG_Z),
        // PosZ: dir = ( sc, -tc,  1)
        CubemapFace::PosZ => (Vec3::Z, Vec3::X, Vec3::NEG_Y),
        // NegZ: dir = (-sc, -tc, -1)
        CubemapFace::NegZ => (Vec3::NEG_Z, Vec3::NEG_X, Vec3::NEG_Y),
    }
}

/// Direction for texel center at (x, y) on a face of given resolution.
pub fn texel_dir(face: CubemapFace, x: u32, y: u32, res: u32) -> Vec3 {
    let u = (x as f32 + 0.5) / res as f32;
    let v = (y as f32 + 0.5) / res as f32;
    face_uv_to_dir(face, u, v)
}

/// Build parallel x/y/z position arrays for every texel on a face, as
/// normalized unit-sphere directions. Row-major (y outer, x inner), matching
/// the storage order of `Cubemap<T>::face_data`. Used by fastnoise2 bulk
/// position-array sampling.
pub fn face_position_arrays(
    face: CubemapFace,
    resolution: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = (resolution * resolution) as usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    for y in 0..resolution {
        for x in 0..resolution {
            let d = texel_dir(face, x, y, resolution).normalize();
            xs.push(d.x);
            ys.push(d.y);
            zs.push(d.z);
        }
    }
    (xs, ys, zs)
}

/// Half-diagonal angle of a cubemap face in radians (~54.7 degrees).
const FACE_HALF_DIAG: f32 = 0.9553; // atan(sqrt(2))

/// Fast per-face cull: true if the cap around `center` with `half_angle`
/// may intersect `face`. Used to skip faces entirely.
pub fn face_may_intersect_cap(face: CubemapFace, center: Vec3, half_angle: f32) -> bool {
    let cos_cull = (half_angle + FACE_HALF_DIAG).min(std::f32::consts::PI).cos();
    center.dot(face_normal(face)) >= cos_cull
}

/// Conservative texel bbox `(x0, x1, y0, y1)` inclusive for the portion of
/// `face` that could contain texels within `half_angle` of `center`. Returns
/// `None` if the face can't be hit at all. Falls back to the full face
/// when the cap straddles a face edge (`c_n ≤ sin(half_angle)`) or the cap
/// is wide enough that the analytical bound breaks down.
pub fn face_cap_texel_bbox(
    face: CubemapFace,
    resolution: u32,
    center: Vec3,
    half_angle: f32,
) -> Option<(u32, u32, u32, u32)> {
    if !face_may_intersect_cap(face, center, half_angle) {
        return None;
    }
    let res_max = resolution - 1;
    let full = Some((0, res_max, 0, res_max));

    // Wide caps: analytical bound breaks down beyond ~45°, rare in practice.
    if half_angle >= std::f32::consts::FRAC_PI_4 {
        return full;
    }

    let (n_axis, u_axis, v_axis) = face_basis(face);
    let c_n = center.dot(n_axis);
    let sin_h = half_angle.sin();
    // Cap straddles the face edge — no tight analytical bound.
    if c_n <= sin_h + 1e-4 {
        return full;
    }
    let c_u = center.dot(u_axis);
    let c_v = center.dot(v_axis);
    // Project cap center onto the face plane (sc, tc in [-1, 1]).
    let sc_c = c_u / c_n;
    let tc_c = c_v / c_n;
    // Conservative uv radius: r = tan(α) / (c_n − sin(α)). Overestimates
    // near the face edge; the per-texel dot check rejects the false positives.
    let r_uv = half_angle.tan() / (c_n - sin_h).max(1e-4);

    let sc_min = (sc_c - r_uv).max(-1.0);
    let sc_max = (sc_c + r_uv).min(1.0);
    let tc_min = (tc_c - r_uv).max(-1.0);
    let tc_max = (tc_c + r_uv).min(1.0);
    if sc_min > sc_max || tc_min > tc_max {
        return None;
    }

    let res_f = resolution as f32;
    // Texel center i has sc = 2(i+0.5)/res − 1  ⇒  i = (sc+1)·res/2 − 0.5
    let to_idx = |sc: f32| (sc + 1.0) * 0.5 * res_f - 0.5;
    let x_min = (to_idx(sc_min).floor() as i32 - 1).max(0) as u32;
    let x_max = (to_idx(sc_max).ceil() as i32 + 1).min(res_max as i32) as u32;
    let y_min = (to_idx(tc_min).floor() as i32 - 1).max(0) as u32;
    let y_max = (to_idx(tc_max).ceil() as i32 + 1).min(res_max as i32) as u32;
    if x_min > x_max || y_min > y_max {
        return None;
    }
    Some((x_min, x_max, y_min, y_max))
}

/// Iterate texels on a single face within `half_angle` radians of `center`.
///
/// Intended for per-face parallel bake loops. Skips the whole face if its
/// normal is outside the cull angle. Calls `f(x, y, dir, angular_dist)`.
pub fn for_face_texels_in_cap<F>(
    face: CubemapFace,
    resolution: u32,
    center: Vec3,
    half_angle: f32,
    f: F,
) where
    F: FnMut(u32, u32, Vec3, f32),
{
    let Some((x0, x1, y0, y1)) = face_cap_texel_bbox(face, resolution, center, half_angle)
    else {
        return;
    };
    iter_face_bbox(face, resolution, center, half_angle, x0, x1, y0, y1, f);
}

/// Like [`for_face_texels_in_cap`] but restricted to rows `[y_start, y_end_excl)`.
/// Used by strip-parallel bake loops where each worker owns a row range.
pub fn for_face_texels_in_cap_rows<F>(
    face: CubemapFace,
    resolution: u32,
    center: Vec3,
    half_angle: f32,
    y_start: u32,
    y_end_excl: u32,
    f: F,
) where
    F: FnMut(u32, u32, Vec3, f32),
{
    if y_start >= y_end_excl {
        return;
    }
    let Some((x0, x1, y0, y1)) = face_cap_texel_bbox(face, resolution, center, half_angle)
    else {
        return;
    };
    let y0 = y0.max(y_start);
    let y1 = y1.min(y_end_excl - 1);
    if y0 > y1 {
        return;
    }
    iter_face_bbox(face, resolution, center, half_angle, x0, x1, y0, y1, f);
}

/// Inner bbox iteration. Walks `base = n + sc·u + tc·v` incrementally so
/// the hot rejection path avoids `sqrt` / `divide` / `acos` — those run only
/// when a texel is inside the cap.
///
/// Requires `half_angle < π/2` so `cos_half > 0` and the squared comparison
/// `dot² ≥ cos_half² · |base|²` is a valid stand-in for `dot/|base| ≥ cos_half`.
/// For `half_angle ≥ π/2` (not used in this codebase) it falls back to the
/// explicit path.
#[inline]
#[allow(clippy::too_many_arguments)]
fn iter_face_bbox<F>(
    face: CubemapFace,
    resolution: u32,
    center: Vec3,
    half_angle: f32,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    mut f: F,
) where
    F: FnMut(u32, u32, Vec3, f32),
{
    let (n, u_ax, v_ax) = face_basis(face);
    let res_f = resolution as f32;
    let inv_res = 1.0 / res_f;
    let du = 2.0 * inv_res; // Δsc per texel
    let cos_half = half_angle.cos();
    let cos_half_sq = cos_half * cos_half;
    let step = u_ax * du;
    let fast = cos_half > 0.0;
    for y in y0..=y1 {
        let tc = (2.0 * y as f32 + 1.0) * inv_res - 1.0;
        let sc0 = (2.0 * x0 as f32 + 1.0) * inv_res - 1.0;
        let mut base = n + v_ax * tc + u_ax * sc0;
        for x in x0..=x1 {
            let dot = center.dot(base);
            let len2 = base.length_squared();
            let inside = if fast {
                dot >= 0.0 && dot * dot >= cos_half_sq * len2
            } else {
                // half_angle ≥ π/2 — explicit compare.
                let len = len2.sqrt();
                dot >= cos_half * len
            };
            if inside {
                let len = len2.sqrt();
                let dir = base / len;
                let cos_d = (dot / len).clamp(-1.0, 1.0);
                f(x, y, dir, cos_d.acos());
            }
            base += step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubemap::CubemapFace;

    #[test]
    fn basis_matches_face_uv_to_dir() {
        // face_uv_to_dir at (u,v) should equal normalize(n + sc·u_ax + tc·v_ax)
        // where sc = 2u−1, tc = 2v−1.
        let samples = [(0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.25, 0.75), (0.9, 0.1)];
        for face in CubemapFace::ALL {
            let (n, u_ax, v_ax) = face_basis(face);
            for (u, v) in samples {
                let sc = 2.0 * u - 1.0;
                let tc = 2.0 * v - 1.0;
                let reconstructed = (n + u_ax * sc + v_ax * tc).normalize();
                let expected = face_uv_to_dir(face, u, v);
                assert!(
                    reconstructed.distance(expected) < 1e-5,
                    "{face:?} @ ({u},{v}): got {reconstructed}, want {expected}"
                );
            }
        }
    }

    #[test]
    fn bbox_covers_naive_iteration() {
        // For a handful of caps, every texel the naive (full-face) scan would
        // visit must also be inside the bbox version.
        let res = 64u32;
        let caps = [
            (Vec3::Z, 0.05f32),
            (Vec3::Z, 0.2),
            (Vec3::new(0.3, 0.2, 0.9).normalize(), 0.1),
            (Vec3::new(0.7, 0.1, 0.7).normalize(), 0.15),
            (Vec3::X, 0.08),
            (Vec3::NEG_Y, 0.12),
        ];
        for (center, half_angle) in caps {
            for face in CubemapFace::ALL {
                let mut naive: Vec<(u32, u32)> = Vec::new();
                let cos_half = half_angle.cos();
                for y in 0..res {
                    for x in 0..res {
                        let dir = texel_dir(face, x, y, res);
                        if center.dot(dir) >= cos_half {
                            naive.push((x, y));
                        }
                    }
                }
                let mut bbox_hits: Vec<(u32, u32)> = Vec::new();
                for_face_texels_in_cap(face, res, center, half_angle, |x, y, _, _| {
                    bbox_hits.push((x, y));
                });
                for h in &naive {
                    assert!(
                        bbox_hits.contains(h),
                        "missed texel {h:?} on {face:?} for center {center} α {half_angle}"
                    );
                }
            }
        }
    }
}

