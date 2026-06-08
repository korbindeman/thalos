//! Procedural engine geometry shared by the editor and the in-game ship view.
//!
//! Rocket bells still use Bevy's stock frustum primitive at the call sites.
//! This module owns the jet nacelle shape because wing pylons need one mesh
//! that can draw a mirrored pair from a single surface-mounted engine part.

use crate::attach::MountSymmetry;
use crate::part::{Engine, Wing};
use crate::wing_mesh::{WingPanelFrame, wing_panel_frame};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

const NACELLE_SEGMENTS: u32 = 48;

#[derive(Clone, Copy, Debug)]
pub struct JetNacelleMount<'a> {
    pub wing: &'a Wing,
    pub wing_mount_angle: f32,
    pub parent_radius: f32,
    /// Span fraction, root to tip.
    pub span_fraction: f32,
    /// Chord fraction, `-0.5` trailing edge to `0.5` leading edge.
    pub chord_fraction: f32,
    pub symmetry: MountSymmetry,
}

pub fn jet_nacelle_length(engine: &Engine) -> f32 {
    engine.diameter * 2.35
}

pub fn build_jet_nacelle_body_mesh(engine: &Engine) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let length = jet_nacelle_length(engine);
    append_nacelle(
        Vec3::ZERO,
        engine.diameter * 0.5,
        length,
        false,
        &mut positions,
        &mut indices,
    );
    finish_mesh(positions, indices)
}

pub fn build_jet_nacelle_pylon_mesh(engine: &Engine, mount: JetNacelleMount<'_>) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    append_pylon_side(engine, mount, false, &mut positions, &mut indices);
    if mount.symmetry == MountSymmetry::Mirrored {
        append_pylon_side(engine, mount, true, &mut positions, &mut indices);
    }
    finish_mesh(positions, indices)
}

pub fn jet_nacelle_centers(engine: &Engine, mount: JetNacelleMount<'_>) -> Vec<Vec3> {
    let primary = nacelle_center(engine, mount, false);
    if mount.symmetry == MountSymmetry::Mirrored {
        let mut mirror = primary;
        mirror.x = -mirror.x;
        vec![primary, mirror]
    } else {
        vec![primary]
    }
}

fn append_pylon_side(
    engine: &Engine,
    mount: JetNacelleMount<'_>,
    mirror: bool,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let frame = wing_panel_frame(mount.wing, mount.wing_mount_angle, mount.parent_radius);
    let center = nacelle_center_from_frame(engine, mount, &frame);
    let mut nacelle_center = center;
    if mirror {
        nacelle_center.x = -nacelle_center.x;
    }
    append_nacelle(
        nacelle_center,
        engine.diameter * 0.5,
        jet_nacelle_length(engine),
        mirror,
        positions,
        indices,
    );

    let surface = wing_surface_point(mount, &frame);
    let down = -frame.thick_dir.normalize_or(Vec3::Z);
    let gap = engine.diameter * 0.35;
    let mut pylon_center = surface + down * (gap * 0.5);
    if mirror {
        pylon_center.x = -pylon_center.x;
    }
    let reflect_axis = |v: Vec3| {
        if mirror { Vec3::new(-v.x, v.y, v.z) } else { v }
    };
    append_box(
        pylon_center,
        reflect_axis(frame.span_dir).normalize_or(Vec3::X),
        Vec3::Y,
        reflect_axis(down),
        Vec3::new(engine.diameter * 0.08, engine.diameter * 0.16, gap * 0.5),
        mirror,
        positions,
        indices,
    );
}

fn nacelle_center(engine: &Engine, mount: JetNacelleMount<'_>, mirror: bool) -> Vec3 {
    let frame = wing_panel_frame(mount.wing, mount.wing_mount_angle, mount.parent_radius);
    let mut center = nacelle_center_from_frame(engine, mount, &frame);
    if mirror {
        center.x = -center.x;
    }
    center
}

fn nacelle_center_from_frame(
    engine: &Engine,
    mount: JetNacelleMount<'_>,
    frame: &WingPanelFrame,
) -> Vec3 {
    let surface = wing_surface_point(mount, frame);
    let down = -frame.thick_dir.normalize_or(Vec3::Z);
    surface + down * (engine.diameter * 0.35 + engine.diameter * 0.5)
}

fn wing_surface_point(mount: JetNacelleMount<'_>, frame: &WingPanelFrame) -> Vec3 {
    let span_fraction = mount.span_fraction.clamp(0.05, 0.95);
    let chord_fraction = mount.chord_fraction.clamp(-0.45, 0.45);
    let chord = frame.chord_at(mount.wing, span_fraction);
    let half_thickness = 0.5 * mount.wing.thickness * chord;
    frame.center_at(span_fraction) + frame.fore_dir * (chord_fraction * chord)
        - frame.thick_dir * half_thickness
}

fn append_nacelle(
    center: Vec3,
    radius: f32,
    length: f32,
    mirror: bool,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let front = center + Vec3::Y * (length * 0.5);
    let back = center - Vec3::Y * (length * 0.5);
    append_frustum(
        front,
        back,
        radius * 1.05,
        radius * 0.78,
        mirror,
        positions,
        indices,
    );
}

fn append_frustum(
    front: Vec3,
    back: Vec3,
    radius_front: f32,
    radius_back: f32,
    mirror: bool,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let start = positions.len() as u32;
    for i in 0..NACELLE_SEGMENTS {
        let a = std::f32::consts::TAU * (i as f32) / (NACELLE_SEGMENTS as f32);
        let radial = Vec3::new(a.cos(), 0.0, a.sin());
        push_pos(front + radial * radius_front, positions);
        push_pos(back + radial * radius_back, positions);
    }

    for i in 0..NACELLE_SEGMENTS {
        let next = (i + 1) % NACELLE_SEGMENTS;
        let a = start + i * 2;
        let b = start + next * 2;
        let c = b + 1;
        let d = a + 1;
        push_quad_indices(a, b, c, d, mirror, indices);
    }

    append_cap(front, radius_front, true, mirror, positions, indices);
    append_cap(back, radius_back, false, mirror, positions, indices);
}

fn append_cap(
    center: Vec3,
    radius: f32,
    front: bool,
    mirror: bool,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let center_i = positions.len() as u32;
    push_pos(center, positions);
    let ring_start = positions.len() as u32;
    for i in 0..NACELLE_SEGMENTS {
        let a = std::f32::consts::TAU * (i as f32) / (NACELLE_SEGMENTS as f32);
        let radial = Vec3::new(a.cos(), 0.0, a.sin());
        push_pos(center + radial * radius, positions);
    }
    for i in 0..NACELLE_SEGMENTS {
        let next = (i + 1) % NACELLE_SEGMENTS;
        let a = ring_start + i;
        let b = ring_start + next;
        let reverse = mirror ^ !front;
        if reverse {
            indices.extend_from_slice(&[center_i, b, a]);
        } else {
            indices.extend_from_slice(&[center_i, a, b]);
        }
    }
}

fn append_box(
    center: Vec3,
    x_axis: Vec3,
    y_axis: Vec3,
    z_axis: Vec3,
    half: Vec3,
    mirror: bool,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let x = x_axis.normalize_or(Vec3::X) * half.x;
    let y = y_axis.normalize_or(Vec3::Y) * half.y;
    let z = z_axis.normalize_or(Vec3::Z) * half.z;
    let p = [
        center - x - y - z,
        center + x - y - z,
        center + x + y - z,
        center - x + y - z,
        center - x - y + z,
        center + x - y + z,
        center + x + y + z,
        center - x + y + z,
    ];
    let faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ];
    for (a, b, c, d) in faces {
        let base = positions.len() as u32;
        for v in [p[a], p[b], p[c], p[d]] {
            push_pos(v, positions);
        }
        push_quad_indices(base, base + 1, base + 2, base + 3, mirror, indices);
    }
}

fn push_quad_indices(a: u32, b: u32, c: u32, d: u32, mirror: bool, indices: &mut Vec<u32>) {
    if mirror {
        indices.extend_from_slice(&[a, c, b, a, d, c]);
    } else {
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    }
}

fn push_pos(pos: Vec3, positions: &mut Vec<[f32; 3]>) {
    positions.push([pos.x, pos.y, pos.z]);
}

fn finish_mesh(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    let uv = vec![[0.0_f32, 0.0]; positions.len()];
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::EngineGeometry;
    use crate::part::ReactantRatio;
    use crate::resource::Resource;

    fn test_engine() -> Engine {
        Engine {
            model: "test".into(),
            geometry: EngineGeometry::JetNacelle,
            requires_atmosphere: true,
            intake_requirement: None,
            builtin_intake: None,
            diameter: 1.0,
            thrust: 1.0,
            isp: 1.0,
            dry_mass: 1.0,
            reactants: vec![ReactantRatio {
                resource: Resource::Kerosene,
                mass_fraction: 1.0,
            }],
            power_draw_kw: 0.0,
        }
    }

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
    fn mirrored_nacelle_mesh_draws_both_sides() {
        let engine = test_engine();
        let wing = test_wing();
        let mesh = build_jet_nacelle_pylon_mesh(
            &engine,
            JetNacelleMount {
                wing: &wing,
                wing_mount_angle: std::f32::consts::FRAC_PI_2,
                parent_radius: 1.0,
                span_fraction: 0.5,
                chord_fraction: 0.0,
                symmetry: MountSymmetry::Mirrored,
            },
        );
        let pos = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let max_x = pos.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        let min_x = pos.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
        assert!(max_x > 1.0 && min_x < -1.0);
    }
}
