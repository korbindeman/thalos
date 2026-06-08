//! Procedural wing geometry, shared by the editor and the in-game ship
//! view so a saved aircraft looks identical in both.
//!
//! A [`Wing`] is built as a tapered / swept / dihedral **airfoil loft**: a
//! NACA 4-digit *symmetric* section (00xx) is sampled at the root and tip
//! chords and lofted along the span. The section's maximum thickness is the
//! wing's `thickness` (t/c), so a thicker wing reads as a fuller airfoil.
//! A symmetric section is deliberate for now: it is handed-neutral, so the
//! mirror counterpart (a separate entity built at the reflected `angle`)
//! reflects correctly with no camber sign to track. Camber and trailing-edge
//! control surfaces are the next slice; callers only depend on
//! [`build_wing_mesh`], so those upgrades stay local to this file.
//!
//! ## Airfoil section
//!
//! Half-thickness as a fraction of chord, NACA 4-digit (closed trailing
//! edge so the tail meets cleanly):
//!
//! ```text
//! yt(x) = 5·t·(0.2969·√x − 0.1260·x − 0.3516·x² + 0.2843·x³ − 0.1036·x⁴)
//! ```
//!
//! `x ∈ [0, 1]` is chord fraction from the leading edge; `t` is the wing's
//! t/c, so total section thickness peaks at `t · chord`. Stations are sampled
//! with **cosine spacing** (`x = ½(1 − cos θ)`) to cluster points at the
//! rounded leading edge and the trailing edge where curvature is highest.
//!
//! ## Frame
//!
//! The mesh is authored in the **host's local frame**, origin at the body
//! axis at the mount station (the wing entity's transform places that
//! point; see the transform systems). Host body axis is +Y (fore = toward
//! the nose / "top"); the circular cross-section lies in X/Z. A mount angle
//! `θ` gives the panel's outboard radial `r̂ = (sin θ, 0, cos θ)`
//! (θ = 0 → +Z "up", θ = π/2 → +X "right"). Mirror symmetry is a separate
//! entity with a reflected `θ` (and `Wing.incidence`), not a flag here —
//! see [`crate::SymmetryGroup`].

use crate::part::Wing;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct WingPanelFrame {
    pub r_hat: Vec3,
    pub span_dir: Vec3,
    pub fore_dir: Vec3,
    pub thick_dir: Vec3,
    pub root_center: Vec3,
    pub tip_center: Vec3,
}

impl WingPanelFrame {
    pub fn chord_at(&self, wing: &Wing, span_fraction: f32) -> f32 {
        wing.root_chord + (wing.tip_chord - wing.root_chord) * span_fraction.clamp(0.0, 1.0)
    }

    pub fn center_at(&self, span_fraction: f32) -> Vec3 {
        self.root_center
            .lerp(self.tip_center, span_fraction.clamp(0.0, 1.0))
    }
}

pub fn wing_panel_frame(wing: &Wing, angle: f32, parent_radius: f32) -> WingPanelFrame {
    let r_hat = Vec3::new(angle.sin(), 0.0, angle.cos());
    let (sin_d, cos_d) = wing.dihedral.sin_cos();
    let span_dir = (r_hat * cos_d + Vec3::Z * sin_d).normalize_or(r_hat);
    // Incidence pitches the chord's leading edge up about the span axis.
    let fore_flat = Vec3::Y;
    let fore_dir = Quat::from_axis_angle(span_dir, wing.incidence) * fore_flat;
    // Thickness normal = span × fore, but that cross product flips sign when
    // the span flips (a left wing), which would point the wing's "up" — and
    // anything mounted by it, like a pylon nacelle — *downward*, giving point
    // (radial) symmetry instead of mirror symmetry. Pin the normal to the
    // dorsal (+Z) hemisphere so left and right are true reflections: tops up,
    // nacelles hang down, on both sides.
    let mut thick_dir = span_dir.cross(fore_dir).normalize_or(Vec3::Z);
    if thick_dir.z < 0.0 {
        thick_dir = -thick_dir;
    }

    let root_center = r_hat * parent_radius;
    let tip_center = root_center + span_dir * wing.span - fore_dir * (wing.span * wing.sweep.tan());

    WingPanelFrame {
        r_hat,
        span_dir,
        fore_dir,
        thick_dir,
        root_center,
        tip_center,
    }
}

/// Chordwise samples per surface (upper / lower). Cosine-spaced, so the
/// effective resolution is highest at the leading and trailing edges.
const CHORD_SAMPLES: usize = 24;

/// NACA 4-digit symmetric half-thickness at chord fraction `x ∈ [0, 1]`,
/// as a fraction of chord. `tc` is the wing t/c. Closed-trailing-edge
/// coefficients (`−0.1036·x⁴`) so `yt(1) = 0` and the tail meets cleanly.
fn naca_half_thickness(tc: f32, x: f32) -> f32 {
    5.0 * tc
        * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x
            - 0.1036 * x * x * x * x)
}

/// One closed airfoil perimeter as `(chordwise_fraction, half_thickness_fraction)`
/// pairs, both relative to chord. `s` runs `+0.5` at the leading edge to
/// `−0.5` at the trailing edge (matching `fore_dir`, which points to the
/// nose); `u` is the signed surface offset. Ordered upper LE→TE then lower
/// TE→LE so consecutive entries trace the outline without repeating the
/// shared LE / TE points.
fn airfoil_perimeter(tc: f32) -> Vec<(f32, f32)> {
    let n = CHORD_SAMPLES;
    let mut out = Vec::with_capacity(2 * n);
    // Upper surface, LE (i=0) → TE (i=n).
    for i in 0..=n {
        let theta = std::f32::consts::PI * (i as f32 / n as f32);
        let x = 0.5 * (1.0 - theta.cos());
        out.push((0.5 - x, naca_half_thickness(tc, x)));
    }
    // Lower surface, TE → LE, skipping the shared TE (i=n) and LE (i=0).
    for i in (1..n).rev() {
        let theta = std::f32::consts::PI * (i as f32 / n as f32);
        let x = 0.5 * (1.0 - theta.cos());
        out.push((0.5 - x, -naca_half_thickness(tc, x)));
    }
    out
}

/// Build the host-local mesh for one wing panel mounted at `angle` on a
/// host of radius `parent_radius`.
///
/// **One entity = one panel.** Under KSP-style mirror symmetry the mirror
/// is a *separate* entity whose own `angle` (and `Wing.incidence`) are
/// reflected, so it renders its own correctly-mirrored panel from this same
/// builder — no "draw both sides" flag (see [`crate::SymmetryGroup`]).
///
/// The skin is a loft of the [`airfoil_perimeter`] section between the root
/// (against the host) and the swept/tapered tip, plus a flat cap at each
/// end so the panel is watertight (the root cap is hidden in the host; the
/// tip cap closes the wingtip). Section vertices are shared around the loop
/// so smoothing rounds the leading edge; the caps carry their own vertices
/// so their rims stay crisp.
pub fn build_wing_mesh(wing: &Wing, angle: f32, parent_radius: f32) -> Mesh {
    let frame = wing_panel_frame(wing, angle, parent_radius);
    let section = airfoil_perimeter(wing.thickness);
    let perimeter = section.len();

    // A ring of perimeter points at one spanwise station, in host-local space.
    let ring = |span_fraction: f32| -> Vec<Vec3> {
        let center = frame.center_at(span_fraction);
        let chord = frame.chord_at(wing, span_fraction);
        section
            .iter()
            .map(|&(s, u)| center + frame.fore_dir * (s * chord) + frame.thick_dir * (u * chord))
            .collect()
    };
    let root_ring = ring(0.0);
    let tip_ring = ring(1.0);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let push = |positions: &mut Vec<[f32; 3]>, v: Vec3| positions.push([v.x, v.y, v.z]);

    // Lofted skin: shared section rings, root first then tip.
    let root_base = 0u32;
    for v in &root_ring {
        push(&mut positions, *v);
    }
    let tip_base = positions.len() as u32;
    for v in &tip_ring {
        push(&mut positions, *v);
    }
    for k in 0..perimeter {
        let k2 = (k + 1) % perimeter;
        let (k, k2) = (k as u32, k2 as u32);
        let r0 = root_base + k;
        let r1 = root_base + k2;
        let t0 = tip_base + k;
        let t1 = tip_base + k2;
        // Winding: outward (verified by `airfoil_loft_is_outward_facing`).
        indices.extend_from_slice(&[r0, t1, t0, r0, r1, t1]);
    }

    // End caps — their own vertices so the rim edge stays crisp under smoothing.
    let cap = |ring: &[Vec3], center: Vec3, outward_is_tip: bool, positions: &mut Vec<[f32; 3]>, indices: &mut Vec<u32>| {
        let center_idx = positions.len() as u32;
        push(positions, center);
        let rim_base = positions.len() as u32;
        for v in ring {
            push(positions, *v);
        }
        for k in 0..perimeter {
            let k2 = ((k + 1) % perimeter) as u32;
            let k = k as u32;
            if outward_is_tip {
                indices.extend_from_slice(&[center_idx, rim_base + k, rim_base + k2]);
            } else {
                indices.extend_from_slice(&[center_idx, rim_base + k2, rim_base + k]);
            }
        }
    };
    cap(&root_ring, frame.center_at(0.0), false, &mut positions, &mut indices);
    cap(&tip_ring, frame.center_at(1.0), true, &mut positions, &mut indices);

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    // UVs the standard material expects; a flat default is fine until the
    // wing gets a real skin material.
    let uv = vec![[0.0_f32, 0.0]; positions.len()];
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_indices(Indices::U32(indices));
    // Section ring vertices are shared between adjacent chordwise faces, so
    // smoothing rounds the leading edge; cap vertices are separate copies,
    // so their rims average only within the cap and stay crisp.
    mesh.compute_smooth_normals();
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_wing() -> Wing {
        Wing {
            span: 5.0,
            root_chord: 2.0,
            tip_chord: 1.0,
            sweep: 0.2,
            dihedral: 0.05,
            thickness: 0.12,
            incidence: 0.0,
            dry_mass: 0.0,
        }
    }

    #[test]
    fn single_panel_sits_on_the_clicked_side() {
        // angle = π/2 → +X (right) side.
        let m = build_wing_mesh(&test_wing(), std::f32::consts::FRAC_PI_2, 1.0);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        // The whole panel sits outboard on +X (root at the radius).
        assert!(pos.iter().all(|p| p[0] > 0.0));
        // Normals + UVs are present for the standard material.
        assert!(m.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(m.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
    }

    #[test]
    fn reflected_angle_lands_on_the_opposite_side() {
        // The mirror counterpart renders by building at the reflected angle
        // (−θ), which must put its single panel entirely on the −X side.
        let m = build_wing_mesh(&test_wing(), -std::f32::consts::FRAC_PI_2, 1.0);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        assert!(
            pos.iter().all(|p| p[0] < 0.0),
            "−θ panel should sit entirely on −X"
        );
    }

    #[test]
    fn section_is_an_airfoil_not_a_box() {
        // The NACA section is convex with its max thickness around 30% chord
        // and pinched at both edges — unlike the old box (constant thickness).
        let tc = 0.12_f32;
        let yt_max = naca_half_thickness(tc, 0.3);
        assert!(naca_half_thickness(tc, 0.0).abs() < 1e-6, "sharp/zero LE point");
        assert!(naca_half_thickness(tc, 1.0).abs() < 1e-4, "closed trailing edge");
        // Peak half-thickness is ~t/2 (full thickness ~= t·chord).
        assert!((yt_max - 0.5 * tc).abs() < 0.01 * tc + 0.01, "peak ~ t/2, got {yt_max}");
        // Convex: midspan thickness sits between the edge and the peak.
        let yt_mid = naca_half_thickness(tc, 0.6);
        assert!(yt_mid > 0.0 && yt_mid < yt_max);
    }

    #[test]
    fn airfoil_loft_is_outward_facing() {
        // A top-surface vertex (positive offset along thick_dir) must carry a
        // normal pointing along +thick_dir; a bottom one along −thick_dir.
        // This guards the loft winding.
        let w = test_wing();
        let angle = std::f32::consts::FRAC_PI_2;
        let m = build_wing_mesh(&w, angle, 1.0);
        let frame = wing_panel_frame(&w, angle, 1.0);
        let pos = m.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().as_float3().unwrap();
        let nor = m.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap().as_float3().unwrap();
        let root_center = frame.center_at(0.0);
        let mut checked = 0;
        for (p, n) in pos.iter().zip(nor) {
            let rel = Vec3::from_array(*p) - root_center;
            let u = rel.dot(frame.thick_dir);
            let nd = Vec3::from_array(*n).dot(frame.thick_dir);
            // Only test clearly top/bottom skin points (away from edges/caps).
            if u.abs() > 0.3 * w.thickness * w.root_chord {
                assert!(
                    u.signum() == nd.signum() || nd.abs() < 0.2,
                    "surface normal should face outward (u={u}, n·thick={nd})"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "expected some top/bottom skin vertices");
    }

    #[test]
    fn end_caps_face_outward_along_span() {
        // The cap center vertices are used only by their own fan, so their
        // smoothed normal is the cap's face normal: root cap faces inboard
        // (−span_dir, hidden in the host), tip cap faces outboard (+span_dir).
        let w = test_wing();
        let angle = std::f32::consts::FRAC_PI_2;
        let m = build_wing_mesh(&w, angle, 1.0);
        let frame = wing_panel_frame(&w, angle, 1.0);
        let pos = m.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().as_float3().unwrap();
        let nor = m.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap().as_float3().unwrap();
        let near = |target: Vec3| {
            pos.iter().position(|p| Vec3::from_array(*p).distance(target) < 1e-4)
        };
        let root_c = near(frame.center_at(0.0)).expect("root cap center vertex");
        let tip_c = near(frame.center_at(1.0)).expect("tip cap center vertex");
        assert!(Vec3::from_array(nor[root_c]).dot(frame.span_dir) < -0.8, "root cap inboard");
        assert!(Vec3::from_array(nor[tip_c]).dot(frame.span_dir) > 0.8, "tip cap outboard");
    }

    #[test]
    fn mirror_wing_is_a_reflection_not_a_radial_rotation() {
        // A left/right pair must be a true mirror: both tops face up (+Z), so
        // a pylon nacelle (hung along −thick) drops on both sides. If the
        // mirror's normal flipped to −Z, the nacelle would point up → radial.
        let w = test_wing();
        let right = wing_panel_frame(&w, std::f32::consts::FRAC_PI_2, 1.0);
        let left = wing_panel_frame(&w, -std::f32::consts::FRAC_PI_2, 1.0);
        assert!(right.thick_dir.z > 0.0, "right wing top faces up");
        assert!(left.thick_dir.z > 0.0, "left wing top faces up");
        assert!(right.tip_center.x > 0.0 && left.tip_center.x < 0.0);
    }
}
