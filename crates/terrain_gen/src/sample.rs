use glam::Vec3;

use crate::body_data::BodyData;
use crate::cubemap::dir_to_face_uv;

/// Result of sampling the surface at a point.
pub struct SurfaceSample {
    /// Height above the reference sphere, in meters.
    pub height: f32,
    /// World-space normal, derived from the height gradient.
    pub normal: Vec3,
    /// Linear-space albedo color.
    pub albedo: Vec3,
    /// PBR roughness, 0..1.
    pub roughness: f32,
    /// Index into `BodyData::materials`.
    pub material_id: u32,
}

/// Sample the body surface at a direction on the unit sphere.
///
/// `lod` is `log2(meters_per_sample)` at the query point.
/// Larger = coarser.
///
/// ## LOD branching
/// 1. **Always**: read the cubemap layer (height + albedo).
/// 2. **If `lod < cubemap_threshold`**: iterate features via spatial index.
/// 3. **If `lod < detail_threshold`**: evaluate statistical detail noise.
///
/// Steps 2 and 3 are stubbed until stages populate features and detail params.
pub fn sample(body: &BodyData, dir: Vec3, _lod: f32) -> SurfaceSample {
    let dir = dir.normalize();

    // --- Layer 1: cubemap ---
    let height = sample_height(body, dir);
    let normal = compute_normal(body, dir);
    let albedo = sample_albedo(body, dir);

    // TODO: Layer 2 — feature SSBO iteration (when stages produce features)
    // TODO: Layer 3 — detail noise (when stages configure detail_params)

    SurfaceSample {
        height,
        normal,
        albedo,
        roughness: 0.5,
        material_id: 0,
    }
}

/// Decode a height texel from the R16Unorm cubemap.
fn decode_height(texel: u16, range: f32) -> f32 {
    (texel as f32 / 65535.0 * 2.0 - 1.0) * range
}

/// Sample height from the cubemap via bilinear interpolation.
fn sample_height(body: &BodyData, dir: Vec3) -> f32 {
    let (face, u, v) = dir_to_face_uv(dir);
    let res = body.height_cubemap.resolution() as f32;
    let px = (u * res - 0.5).clamp(0.0, res - 1.001);
    let py = (v * res - 0.5).clamp(0.0, res - 1.001);
    let x0 = px.floor() as u32;
    let y0 = py.floor() as u32;
    let x1 = (x0 + 1).min(body.height_cubemap.resolution() - 1);
    let y1 = (y0 + 1).min(body.height_cubemap.resolution() - 1);
    let fx = px - px.floor();
    let fy = py - py.floor();

    let h00 = decode_height(body.height_cubemap.get(face, x0, y0), body.height_range);
    let h10 = decode_height(body.height_cubemap.get(face, x1, y0), body.height_range);
    let h01 = decode_height(body.height_cubemap.get(face, x0, y1), body.height_range);
    let h11 = decode_height(body.height_cubemap.get(face, x1, y1), body.height_range);

    let top = h00 + (h10 - h00) * fx;
    let bot = h01 + (h11 - h01) * fx;
    top + (bot - top) * fy
}

/// Sample albedo from the cubemap.  Returns linear-space color.
fn sample_albedo(body: &BodyData, dir: Vec3) -> Vec3 {
    let (face, u, v) = dir_to_face_uv(dir);
    let res = body.albedo_cubemap.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    let texel = body.albedo_cubemap.get(face, x, y);
    Vec3::new(
        srgb_to_linear(texel[0]),
        srgb_to_linear(texel[1]),
        srgb_to_linear(texel[2]),
    )
}

fn srgb_to_linear(srgb: u8) -> f32 {
    let s = srgb as f32 / 255.0;
    if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

/// Compute the surface normal via finite differences on the height cubemap.
fn compute_normal(body: &BodyData, dir: Vec3) -> Vec3 {
    // Build tangent frame on the sphere at `dir`.
    let up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    let tangent = dir.cross(up).normalize();
    let bitangent = tangent.cross(dir);

    // Offset angle: ~1 texel on the cubemap.
    let texel_angle = 1.0 / body.height_cubemap.resolution() as f32;
    let offset = texel_angle * 1.5; // slightly more than 1 texel for stability

    // Sample height at 4 offset directions.
    let h_east = sample_height(body, (dir + tangent * offset).normalize());
    let h_west = sample_height(body, (dir - tangent * offset).normalize());
    let h_north = sample_height(body, (dir + bitangent * offset).normalize());
    let h_south = sample_height(body, (dir - bitangent * offset).normalize());

    // Convert texel offset to world-space distance.
    let ds = body.radius_m * offset * 2.0;

    let dh_dt = (h_east - h_west) / ds;
    let dh_db = (h_north - h_south) / ds;

    (dir - tangent * dh_dt - bitangent * dh_db).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body_builder::BodyBuilder;
    use crate::types::Composition;

    #[test]
    fn sample_default_body_returns_flat_surface() {
        let builder = BodyBuilder::new(
            869_000.0, 42,
            Composition::new(0.9, 0.05, 0.0, 0.05, 0.0),
            4, // small resolution for test speed
        );
        let body = builder.build();

        let s = sample(&body, Vec3::X, 10.0);
        assert!(s.height.abs() < 1.0, "expected ~0 height, got {}", s.height);
        assert!(s.normal.dot(Vec3::X) > 0.99, "normal should point outward");
    }
}
