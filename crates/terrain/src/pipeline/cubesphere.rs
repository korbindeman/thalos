//! Cube-sphere parameterisation for the pipeline's storage and addressing.
//!
//! Six faces, each with a face-local `(u, v)` in `[0, 1]`. This is the
//! pipeline's *internal* storage mapping (spec §5) — distinct from the legacy
//! [`crate::cubemap`] used by the baked surface, and distinct from UDLOD's
//! tiling mapping (which stays canonical at the renderer; the Query API
//! evaluates by direction, see [`crate::query`]). Any consistent bijection
//! works here; the only contract is that [`face_uv_to_dir`] and
//! [`dir_to_face_uv`] round-trip.

use glam::Vec3;

/// Number of cube faces.
pub const FACE_COUNT: u32 = 6;

/// `(normal, s-axis, t-axis)` basis for a face. `u` runs along `s`, `v` along
/// `t`, each remapped from `[0, 1]` to `[-1, 1]`.
fn face_basis(face: u32) -> (Vec3, Vec3, Vec3) {
    match face {
        0 => (Vec3::X, Vec3::Z, Vec3::Y),
        1 => (Vec3::NEG_X, Vec3::NEG_Z, Vec3::Y),
        2 => (Vec3::Y, Vec3::X, Vec3::Z),
        3 => (Vec3::NEG_Y, Vec3::X, Vec3::NEG_Z),
        4 => (Vec3::Z, Vec3::NEG_X, Vec3::Y),
        _ => (Vec3::NEG_Z, Vec3::X, Vec3::Y),
    }
}

/// Map a face id + face-local `(u, v)` in `[0, 1]` to a unit direction.
pub fn face_uv_to_dir(face: u32, u: f32, v: f32) -> Vec3 {
    let (n, s, t) = face_basis(face);
    (n + s * (2.0 * u - 1.0) + t * (2.0 * v - 1.0)).normalize()
}

/// Map a unit direction to its `(face, u, v)`. Inverse of [`face_uv_to_dir`].
pub fn dir_to_face_uv(dir: Vec3) -> (u32, f32, f32) {
    let dir = dir.normalize_or_zero();
    // The containing face is the one whose outward normal `dir` projects onto
    // most strongly (positively).
    let mut best_face = 0u32;
    let mut best_dot = f32::NEG_INFINITY;
    for face in 0..FACE_COUNT {
        let (n, _, _) = face_basis(face);
        let d = dir.dot(n);
        if d > best_dot {
            best_dot = d;
            best_face = face;
        }
    }
    let (n, s, t) = face_basis(best_face);
    let dn = dir.dot(n);
    // Guard the (degenerate) case of a direction tangent to the face plane.
    let inv = if dn.abs() > 1e-6 { 1.0 / dn } else { 0.0 };
    let a = (dir.dot(s) * inv).clamp(-1.0, 1.0);
    let b = (dir.dot(t) * inv).clamp(-1.0, 1.0);
    (best_face, (a + 1.0) * 0.5, (b + 1.0) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_face_uv() {
        // Interior points only: exact face edges/corners are shared by
        // multiple faces and are intentionally ambiguous.
        for face in 0..FACE_COUNT {
            for &(u, v) in &[(0.5, 0.5), (0.1, 0.9), (0.25, 0.75), (0.3, 0.2), (0.7, 0.85)] {
                let dir = face_uv_to_dir(face, u, v);
                let (f2, u2, v2) = dir_to_face_uv(dir);
                assert_eq!(f2, face, "face mismatch for {face} ({u},{v})");
                assert!((u2 - u).abs() < 1e-4, "u: {u} -> {u2}");
                assert!((v2 - v).abs() < 1e-4, "v: {v} -> {v2}");
            }
        }
    }
}
