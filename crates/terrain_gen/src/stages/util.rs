use glam::Vec3;
use crate::cubemap::{CubemapFace, face_uv_to_dir};

/// Normal vector for a cubemap face.
pub fn face_normal(face: CubemapFace) -> Vec3 {
    match face {
        CubemapFace::PosX => Vec3::X,
        CubemapFace::NegX => Vec3::NEG_X,
        CubemapFace::PosY => Vec3::Y,
        CubemapFace::NegY => Vec3::NEG_Y,
        CubemapFace::PosZ => Vec3::Z,
        CubemapFace::NegZ => Vec3::NEG_Z,
    }
}

/// Direction for texel center at (x, y) on a face of given resolution.
pub fn texel_dir(face: CubemapFace, x: u32, y: u32, res: u32) -> Vec3 {
    let u = (x as f32 + 0.5) / res as f32;
    let v = (y as f32 + 0.5) / res as f32;
    face_uv_to_dir(face, u, v)
}

/// Half-diagonal angle of a cubemap face in radians (~54.7 degrees).
const FACE_HALF_DIAG: f32 = 0.9553; // atan(sqrt(2))

/// Iterate all texels within `half_angle` radians of `center` direction.
///
/// Uses face culling: skips entire faces whose normal is too far from center.
/// Calls `f(face, x, y, dir, angular_dist)` for each texel inside the cap.
pub fn for_texels_in_cap<F>(
    resolution: u32,
    center: Vec3,
    half_angle: f32,
    mut f: F,
) where
    F: FnMut(CubemapFace, u32, u32, Vec3, f32),
{
    let cos_half = half_angle.cos();
    let cos_cull = (half_angle + FACE_HALF_DIAG).min(std::f32::consts::PI).cos();

    for face in CubemapFace::ALL {
        if center.dot(face_normal(face)) < cos_cull {
            continue;
        }
        for y in 0..resolution {
            for x in 0..resolution {
                let dir = texel_dir(face, x, y, resolution);
                let cos_d = center.dot(dir);
                if cos_d >= cos_half {
                    f(face, x, y, dir, cos_d.clamp(-1.0, 1.0).acos());
                }
            }
        }
    }
}

/// Iterate every texel on the cubemap.
pub fn for_all_texels<F>(resolution: u32, mut f: F)
where
    F: FnMut(CubemapFace, u32, u32, Vec3),
{
    for face in CubemapFace::ALL {
        for y in 0..resolution {
            for x in 0..resolution {
                f(face, x, y, texel_dir(face, x, y, resolution));
            }
        }
    }
}

