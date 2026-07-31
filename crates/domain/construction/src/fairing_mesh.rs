//! Wing-body junction fairing: the belly blister that blends a main wing's
//! root into the fuselage — the reason a real airliner's wing doesn't look
//! bolted onto a tube, and the volume that houses its gear attachment
//! structure so only oleo and wheels show under the hull.
//!
//! **Derived geometry, not a part.** A fairing is generated automatically for
//! a main-wing pair mounted low/mid on a fuselage loft
//! ([`wants_wing_fairing`]); the player never places or edits it. It is
//! visual only — no mass, no aero, no collider — matching how the aero and
//! inertia models already treat the junction.
//!
//! ## Construction
//!
//! Authored in the **host-local frame relative to the wing's mount origin**
//! (the host axis point at the mount station), the same frame
//! [`crate::build_wing_mesh`] uses, so the caller spawns it exactly like the
//! wing mesh (identity transform on the wing part entity).
//!
//! The surface is a loft of the fuselage's own lower cross-section arcs,
//! radially scaled: each fairing station samples the true hull skin via
//! [`crate::skin_radius`] / [`crate::v_offset_at`] and pushes it outward by a
//! bulge that peaks at the belly and mid-length. Everywhere the bulge is zero
//! — the waterline rim and both ends — the surface is pulled a few percent
//! *inside* the hull, so the blister needs no caps, no boolean, and no seam
//! stitching: its entire boundary is submerged in the fuselage.
//!
//! The fairing spans from ahead of the root leading edge to well aft of the
//! trailing edge (real belly fairings trail the wing), clamped to the hull.

use crate::fuselage_mesh::{skin_radius, v_offset_at};
use crate::part::{Fuselage, Wing};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// Lofted stations along the fairing length.
const FAIRING_STATIONS: usize = 26;
/// Segments around the lower cross-section arc.
const ARC_SEGMENTS: usize = 32;
/// Peak radial bulge past the hull skin, as a fraction of the local skin
/// radius. ~13 % on a Ø3.3 m body gives the A320-proportioned blister.
const BULGE_FRAC: f32 = 0.13;
/// How far inside the hull the rim/ends sit, as a fraction of skin radius —
/// deep enough that the open boundary can never peek through the skin.
const TUCK_FRAC: f32 = 0.03;
/// Fairing extent ahead of the mount station, in root chords (the root
/// leading edge sits ~half a chord ahead of the mount).
const FWD_CHORDS: f32 = 0.9;
/// Fairing extent aft of the mount station, in root chords — belly fairings
/// trail well behind the wing.
const AFT_CHORDS: f32 = 1.6;

/// Whether this wing mount earns a junction fairing: the **right-hand panel**
/// of a pair (one fairing per pair — the blister is belly-symmetric), mounted
/// on the side-to-lower hull (a high wing gets no belly blister), and big
/// enough to be a *main* wing — the root chord gate excludes tailplanes,
/// which real aircraft do not fair (Meridian main 5.2 m / 35 m = 0.149,
/// stabilizer 2.6 m / 35 m = 0.074).
pub fn wants_wing_fairing(wing: &Wing, angle: f32, fus: &Fuselage) -> bool {
    angle.sin() > 0.5 && angle.cos() < 0.35 && wing.root_chord >= 0.12 * fus.length
}

/// Build the junction-fairing mesh for a wing pair mounted at `station` on
/// `fus` (rendered at `effective_diameter`). See the module doc for the frame
/// and construction. Call gated by [`wants_wing_fairing`].
pub fn build_wing_fairing_mesh(
    fus: &Fuselage,
    effective_diameter: f32,
    wing: &Wing,
    station: f32,
) -> Mesh {
    let len = fus.length.max(0.01);
    // Fairing y-range in mount-local coordinates (mount origin at y = 0,
    // nose-ward positive), clamped to the hull.
    let y_fwd = (FWD_CHORDS * wing.root_chord).min(station * len);
    let y_aft = -(AFT_CHORDS * wing.root_chord).max(-((1.0 - station) * len));

    let smoothstep = |e0: f32, e1: f32, x: f32| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(FAIRING_STATIONS * (ARC_SEGMENTS + 1));
    for si in 0..FAIRING_STATIONS {
        let t = si as f32 / (FAIRING_STATIONS - 1) as f32;
        let y_local = y_fwd + (y_aft - y_fwd) * t;
        // Station along the whole fuselage for the hull-skin sample.
        let s01 = (station - y_local / len).clamp(0.0, 1.0);
        let v_off = v_offset_at(fus, effective_diameter, s01);
        // Bulge window: zero at both ends (surface tucked inside the hull),
        // full through the middle.
        let end_taper = smoothstep(0.0, 0.22, t) * smoothstep(1.0, 0.78, t);
        for ai in 0..=ARC_SEGMENTS {
            // θ sweeps the lower arc: π/2 (+X waterline) → π (belly) → 3π/2.
            let theta = std::f32::consts::FRAC_PI_2
                + std::f32::consts::PI * (ai as f32 / ARC_SEGMENTS as f32);
            // Belly weight: 0 at the waterline rim, 1 at the keel.
            let mu = (theta - std::f32::consts::FRAC_PI_2).sin().max(0.0);
            let bulge = end_taper * mu.powf(1.2);
            let scale = 1.0 + bulge * BULGE_FRAC - (1.0 - bulge) * TUCK_FRAC;
            let r = skin_radius(fus, effective_diameter, s01, theta) * scale;
            positions.push([theta.sin() * r, y_local, theta.cos() * r + v_off]);
        }
    }

    // Grid triangulation, wound so normals face outward (see the belly-frame
    // derivation in the loft: tangent_t × tangent_θ points out of the hull).
    let stride = (ARC_SEGMENTS + 1) as u32;
    let mut indices: Vec<u32> = Vec::new();
    for si in 0..(FAIRING_STATIONS - 1) as u32 {
        for ai in 0..ARC_SEGMENTS as u32 {
            let a = si * stride + ai;
            let b = a + 1;
            let d = a + stride;
            let c = d + 1;
            indices.extend_from_slice(&[a, d, c, a, c, b]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    let uv = vec![[0.0_f32, 0.0]; positions.len()];
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    crate::part_mesh::add_raytracing_tangents(&mut mesh);
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fuselage() -> Fuselage {
        Fuselage {
            length: 35.0,
            max_width: 3.3,
            max_height: 3.3,
            roundness: 1.0,
            nose_fraction: 0.13,
            nose_bluntness: 0.55,
            tail_fraction: 0.34,
            nose_droop: 0.0,
            tail_upsweep: 1.05,
            tail_tip_diameter: 0.0,
            tail_bluntness: 0.6,
            dry_mass: 0.0,
        }
    }

    fn main_wing() -> Wing {
        Wing {
            span: 15.0,
            root_chord: 5.2,
            tip_chord: 1.5,
            sweep: 0.52,
            dihedral: 0.365,
            thickness: 0.11,
            incidence: 0.0,
            dry_mass: 0.0,
            control_surfaces: Vec::new(),
        }
    }

    #[test]
    fn main_wing_pair_gets_a_fairing_tailplane_and_fin_do_not() {
        let fus = fuselage();
        // Right-hand low-mounted main wing: yes.
        assert!(wants_wing_fairing(&main_wing(), 1.85, &fus));
        // Its mirror (left) must not double-generate.
        assert!(!wants_wing_fairing(&main_wing(), -1.85, &fus));
        // Tailplane: too small a chord to be a main wing.
        let stab = Wing {
            root_chord: 2.6,
            span: 4.6,
            ..main_wing()
        };
        assert!(!wants_wing_fairing(&stab, 1.5708, &fus));
        // Fin (dorsal, angle 0): not a side mount at all.
        assert!(!wants_wing_fairing(&main_wing(), 0.0, &fus));
        // High wing: no belly blister.
        assert!(!wants_wing_fairing(&main_wing(), 0.9, &fus));
    }

    #[test]
    fn fairing_bulges_past_the_hull_only_at_the_belly() {
        let fus = fuselage();
        let m = build_wing_fairing_mesh(&fus, 3.3, &main_wing(), 0.44);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let hull_half = 3.3_f32 * 0.5;
        let mut deepest = 0.0_f32;
        for p in pos {
            // Below the hull keel somewhere (the bulge)…
            deepest = deepest.min(p[2]);
            // …but never wider than the bulged hull anywhere.
            assert!(
                p[0].abs() <= hull_half * (1.0 + BULGE_FRAC) + 1e-3,
                "fairing wider than the bulge allows: {p:?}"
            );
        }
        assert!(
            deepest < -hull_half,
            "fairing keel must drop below the hull keel ({deepest} vs -{hull_half})"
        );
        assert!(m.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(crate::part_mesh::is_raytracing_ready(&m));
    }

    #[test]
    fn fairing_boundary_is_submerged_in_the_hull() {
        // Both end rings and both rim rows must sit inside the hull skin so
        // the open boundary can never show: check the first/last station and
        // the first/last arc column against the local skin radius.
        let fus = fuselage();
        let wing = main_wing();
        let station = 0.44_f32;
        let m = build_wing_fairing_mesh(&fus, 3.3, &wing, station);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let stride = ARC_SEGMENTS + 1;
        let boundary = |i: usize| -> bool {
            let si = i / stride;
            let ai = i % stride;
            si == 0 || si == FAIRING_STATIONS - 1 || ai == 0 || ai == ARC_SEGMENTS
        };
        for (i, p) in pos.iter().enumerate() {
            if !boundary(i) {
                continue;
            }
            let s01 = station - p[1] / fus.length;
            let v = v_offset_at(&fus, 3.3, s01);
            let theta = f32::atan2(p[0], p[2] - v);
            let hull_r = skin_radius(&fus, 3.3, s01, theta);
            let r = (p[0] * p[0] + (p[2] - v) * (p[2] - v)).sqrt();
            assert!(
                r <= hull_r + 1e-3,
                "boundary vertex outside the hull skin: {p:?} (r {r} vs hull {hull_r})"
            );
        }
    }
}
