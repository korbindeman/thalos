//! Procedural wing geometry, shared by the editor and the in-game ship
//! view so a saved aircraft looks identical in both.
//!
//! A [`Wing`] is built as a tapered / swept / dihedral **extruded box** —
//! a flat-ish slab, not yet a true airfoil. This is the slice-1 stand-in:
//! recognizable as a wing at editor and orbital distance, cheap, and
//! robust. The cross-section will be promoted to an airfoil loft (and grow
//! trailing-edge control surfaces) in a later slice; callers only depend on
//! [`build_wing_mesh`], so that upgrade stays local to this file.
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

/// Build the host-local mesh for one wing panel mounted at `angle` on a
/// host of radius `parent_radius`.
///
/// **One entity = one panel.** Under KSP-style mirror symmetry the mirror
/// is a *separate* entity whose own `angle` (and `Wing.incidence`) are
/// reflected, so it renders its own correctly-mirrored panel from this same
/// builder — no "draw both sides" flag (see [`crate::SymmetryGroup`]).
pub fn build_wing_mesh(wing: &Wing, angle: f32, parent_radius: f32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let r_hat = Vec3::new(angle.sin(), 0.0, angle.cos());
    append_panel(
        wing,
        r_hat,
        parent_radius,
        false,
        &mut positions,
        &mut indices,
    );

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
    // Crisp per-face normals: each quad owns its 4 vertices (no cross-face
    // sharing), so smoothing averages only within a planar face — i.e. it
    // yields the face normal. `compute_smooth_normals` (unlike
    // `compute_flat_normals`) works on indexed geometry, which this is.
    mesh.compute_smooth_normals();
    mesh
}

/// Append one panel's 6 box faces to the buffers. `mirror` negates X and
/// reverses winding so the reflected panel faces outward.
fn append_panel(
    wing: &Wing,
    r_hat: Vec3,
    parent_radius: f32,
    mirror: bool,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let frame = wing_panel_frame(wing, r_hat.x.atan2(r_hat.z), parent_radius);
    let fore_dir = frame.fore_dir;
    let thick_dir = frame.thick_dir;
    let root_center = frame.root_center;
    let tip_center = frame.tip_center;

    let half_t_root = 0.5 * wing.thickness * wing.root_chord;
    let half_t_tip = 0.5 * wing.thickness * wing.tip_chord;
    let half_c_root = 0.5 * wing.root_chord;
    let half_c_tip = 0.5 * wing.tip_chord;

    // 8 corners: [root/tip] × [LE/TE] × [top/bottom].
    let corner = |center: Vec3, half_c: f32, half_t: f32, le: bool, top: bool| -> Vec3 {
        let fore = if le { half_c } else { -half_c };
        let up = if top { half_t } else { -half_t };
        let mut p = center + fore_dir * fore + thick_dir * up;
        if mirror {
            p.x = -p.x;
        }
        p
    };

    let r_le_t = corner(root_center, half_c_root, half_t_root, true, true);
    let r_te_t = corner(root_center, half_c_root, half_t_root, false, true);
    let t_te_t = corner(tip_center, half_c_tip, half_t_tip, false, true);
    let t_le_t = corner(tip_center, half_c_tip, half_t_tip, true, true);
    let r_le_b = corner(root_center, half_c_root, half_t_root, true, false);
    let r_te_b = corner(root_center, half_c_root, half_t_root, false, false);
    let t_te_b = corner(tip_center, half_c_tip, half_t_tip, false, false);
    let t_le_b = corner(tip_center, half_c_tip, half_t_tip, true, false);

    // Six quads, each as its own 4 vertices so flat normals stay crisp.
    // Winding chosen CCW-outward for the primary; reversed for the mirror.
    let mut quad = |a: Vec3, b: Vec3, c: Vec3, d: Vec3| {
        let base = positions.len() as u32;
        for v in [a, b, c, d] {
            positions.push([v.x, v.y, v.z]);
        }
        if mirror {
            indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        } else {
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    };

    quad(r_le_t, r_te_t, t_te_t, t_le_t); // top
    quad(r_le_b, t_le_b, t_te_b, r_te_b); // bottom
    quad(r_le_t, t_le_t, t_le_b, r_le_b); // leading edge
    quad(r_te_t, r_te_b, t_te_b, t_te_t); // trailing edge
    quad(r_le_t, r_le_b, r_te_b, r_te_t); // root
    quad(t_le_t, t_te_t, t_te_b, t_le_b); // tip
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
    fn single_panel_is_one_box_on_the_clicked_side() {
        // angle = π/2 → +X (right) side.
        let m = build_wing_mesh(&test_wing(), std::f32::consts::FRAC_PI_2, 1.0);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        // 6 quads × 4 vertices, one panel.
        assert_eq!(pos.len(), 24);
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
        assert_eq!(pos.len(), 24);
        assert!(
            pos.iter().all(|p| p[0] < 0.0),
            "−θ panel should sit entirely on −X"
        );
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
