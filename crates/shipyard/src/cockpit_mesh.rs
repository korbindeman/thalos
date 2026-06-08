//! Procedural cockpit-nose geometry, shared by the editor and the in-game ship
//! view so a saved aircraft's nose looks identical in both.
//!
//! The [`crate::PodGeometry::AircraftCockpit`] pod renders as a smooth,
//! **rounded ogive** — a half-ellipsoid surface of revolution that meets the
//! fuselage at full diameter and curves through a convex profile to a blunt,
//! rounded tip, like an airliner radome. (The capsule pod keeps a plain
//! truncated cone; a straight `ConicalFrustum` can't make this curve.) This is
//! a deliberately simple placeholder — a real, asymmetric (nose-low) model can
//! replace it later.
//!
//! ## Frame
//!
//! Authored centred on the origin, spanning `[-length/2, +length/2]` along +Y,
//! with the wide base at −Y and the rounded tip at +Y — the same frame the
//! capsule frustum uses. The ship view offsets the body child by `-height/2`,
//! placing the base at the part's bottom attach node (where the fuselage mates)
//! and the tip at the part origin (the nose).
//!
//! ## Normals
//!
//! Normals are computed **analytically** from the ellipsoid, not via
//! `compute_smooth_normals`. Face-normal averaging misbehaves at the two poles:
//! a base end-cap bleeds its `−Y` normal into the rim (a shading "dent"), and
//! the tip fan leaves the apex normal discontinuous with its neighbours (a
//! shading dimple). The exact surface normal avoids both. The base is left
//! **open** — it is hidden inside the fuselage join, so a cap is unnecessary and
//! only reintroduces the rim artifact.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// Radial segments around the nose. Round enough to sit beside the 128-segment
/// fuselage without a visible facet seam at the join.
const RADIAL_SEGMENTS: u32 = 64;
/// Rings along the nose axis, base → (just short of) the tip.
const AXIAL_RINGS: u32 = 32;

/// Build a rounded nose-cone mesh: a half-ellipsoid `length` long and
/// `diameter` wide at the base, tapering through a convex curve to a rounded
/// tip. See the module docs for the authored frame and normal handling.
///
/// Profile in the angular parameter `φ ∈ [0, π/2]`: axial fraction
/// `t = sin φ` (base→tip) and radius `r = R·cos φ`. The outward surface normal
/// in the meridian plane is `(L·cos φ, R·sin φ)` — radial at the base (so it
/// matches the fuselage cylinder seamlessly) and `+Y` at the tip.
pub fn build_cockpit_mesh(diameter: f32, length: f32) -> Mesh {
    let radius = diameter * 0.5;
    let half = length * 0.5;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Rings from the base (φ = 0, full radius) up to just short of the tip.
    for i in 0..AXIAL_RINGS {
        let phi = std::f32::consts::FRAC_PI_2 * (i as f32 / AXIAL_RINGS as f32);
        let (sin_p, cos_p) = phi.sin_cos();
        let r = radius * cos_p;
        let y = -half + sin_p * length;
        // Meridian-plane outward normal (radial, axial).
        let n_radial = length * cos_p;
        let n_axial = radius * sin_p;
        for s in 0..RADIAL_SEGMENTS {
            let a = std::f32::consts::TAU * (s as f32) / (RADIAL_SEGMENTS as f32);
            let (sin_a, cos_a) = a.sin_cos();
            positions.push([r * cos_a, y, r * sin_a]);
            normals.push(
                Vec3::new(n_radial * cos_a, n_axial, n_radial * sin_a)
                    .normalize_or(Vec3::Y)
                    .to_array(),
            );
        }
    }
    // Tip apex (φ = π/2 → r = 0); its exact normal is +Y.
    let apex = positions.len() as u32;
    positions.push([0.0, half, 0.0]);
    normals.push([0.0, 1.0, 0.0]);

    // Barrel quads between consecutive rings. Winding matches the engine
    // frustum: +Y (upper) ring first → outward-facing.
    for i in 0..(AXIAL_RINGS - 1) {
        for s in 0..RADIAL_SEGMENTS {
            let next = (s + 1) % RADIAL_SEGMENTS;
            let l_s = i * RADIAL_SEGMENTS + s;
            let l_next = i * RADIAL_SEGMENTS + next;
            let u_s = (i + 1) * RADIAL_SEGMENTS + s;
            let u_next = (i + 1) * RADIAL_SEGMENTS + next;
            indices.extend_from_slice(&[u_s, u_next, l_next, u_s, l_next, l_s]);
        }
    }
    // Tip fan: last ring → apex (faces +Y).
    let last = (AXIAL_RINGS - 1) * RADIAL_SEGMENTS;
    for s in 0..RADIAL_SEGMENTS {
        let next = (s + 1) % RADIAL_SEGMENTS;
        indices.extend_from_slice(&[apex, last + s, last + next]);
    }
    // No base cap: the base is open and hidden inside the fuselage join.

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    let uv = vec![[0.0_f32, 0.0]; positions.len()];
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extents(mesh: &Mesh) -> (Vec3, Vec3) {
        let pos = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for p in pos {
            let p = Vec3::from_array(*p);
            min = min.min(p);
            max = max.max(p);
        }
        (min, max)
    }

    #[test]
    fn spans_length_and_base_diameter() {
        let (diameter, length) = (1.5_f32, 2.0_f32);
        let m = build_cockpit_mesh(diameter, length);
        let (min, max) = extents(&m);
        // Spans the full length along Y, centred on the origin.
        assert!((min.y + length * 0.5).abs() < 1e-4, "base at -length/2");
        assert!((max.y - length * 0.5).abs() < 1e-4, "tip at +length/2");
        // Widest at the base: reaches the full radius in X/Z, not beyond.
        let r = diameter * 0.5;
        assert!((max.x - r).abs() < 1e-3, "base reaches radius in X");
        assert!((max.z - r).abs() < 1e-3, "base reaches radius in Z");
        assert!(m.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(m.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
    }

    #[test]
    fn tip_is_rounded_not_a_straight_cone() {
        // The ellipsoid is convex: at the axial midpoint the radius is well
        // above the straight-cone radius (which would be half the base). At
        // t = sin φ = 0.5 (the midpoint), r = R·cos φ = R·√(1−0.25) ≈ 0.87·R,
        // not 0.5·R.
        let (diameter, length) = (2.0_f32, 2.0_f32);
        let m = build_cockpit_mesh(diameter, length);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let r = diameter * 0.5;
        let mid_r = pos
            .iter()
            .filter(|p| p[1].abs() < length * 0.05)
            .map(|p| (p[0] * p[0] + p[2] * p[2]).sqrt())
            .fold(0.0_f32, f32::max);
        assert!(
            mid_r > 0.75 * r,
            "midpoint radius {mid_r} should bulge well past a straight cone's 0.5·R ({})",
            0.5 * r
        );
    }

    #[test]
    fn normals_are_unit_and_base_rim_is_radial() {
        // Analytic normals: every one is unit length, and the base rim (ring 0)
        // points purely radial (zero Y) so it meets the fuselage cylinder with
        // no shading seam.
        let m = build_cockpit_mesh(1.5, 1.5);
        let normals = m
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .unwrap()
            .as_float3()
            .unwrap();
        for n in normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "normal {n:?} not unit (len {len})");
        }
        // Ring 0 is the first RADIAL_SEGMENTS vertices.
        for n in &normals[..RADIAL_SEGMENTS as usize] {
            assert!(n[1].abs() < 1e-4, "base rim normal {n:?} should be radial");
        }
    }
}
