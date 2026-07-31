//! Procedural landing-gear geometry, shared by the editor and the in-game
//! ship view so a saved aircraft's gear looks identical in both.
//!
//! A [`Gear`] is a self-contained **gearbox**: one part draws *all* of its
//! legs in a single mesh (the way [`crate::wing_mesh`] once drew both wing
//! panels in one mesh). `gear_main` draws a left/right main pair spaced by the
//! host-radius-derived track; `gear_nose` draws a single centred leg. This is
//! deliberately **not** the [`crate::SymmetryGroup`] path — the editor places a
//! gear as a single mount regardless of the Mirror toggle, and this builder
//! owns the multiplicity.
//!
//! ## Frame
//!
//! Authored in the **host's local frame**, like the wing builder. Host body
//! axis is +Y (fore / nose / "top"); the circular cross-section lies in X/Z. A
//! mount angle `θ` gives the outboard radial `r̂ = (sin θ, 0, cos θ)`
//! (θ = 0 → +Z dorsal, θ = π → −Z belly, θ = π/2 → +X right). Each wheel hub
//! sits at radial depth `parent_radius + strut_length` at its lateral track
//! offset, with the axle along the lateral axis so it rolls fore/aft. The
//! **oleo strut is always vertical over the hub** — that verticality is what
//! makes gear read as aircraft gear (a hub-to-belly diagonal reads as a
//! glider outrigger; user-rejected). A leg inside the hull silhouette is just
//! that oleo from the skin; a wide-track leg (`track_fraction > 1`, wheels
//! outboard of the hull — the airliner stance) adds the cantilever truss: a
//! horizontal gear beam from the lower skin quadrant out over the wheel,
//! plus a diagonal side-stay. The physics contact frame
//! ([`gear_leg_frames`]) is independent of all of it: the suspension ray is
//! always the outward radial through the hub.
//!
//! The lateral axis (leg spacing + wheel axle) is `r̂ × ŷ`, pinned to the +X
//! hemisphere so a left/right pair is consistent regardless of how the cross
//! product happens to fall — the same handedness guard `wing_panel_frame` uses
//! for its thickness normal.

use crate::part::Gear;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// Radial segment count for the wheel barrels. Round enough at editor zoom,
/// cheap at the part counts we render.
const WHEEL_SEGMENTS: u32 = 24;

/// How far around the lower skin quadrant a strut may anchor, as a fraction
/// of the host radius along the lateral axis (≈ 40° off the mount radial).
/// A wide-track leg's wheels sit outboard of this; the strut slants from the
/// anchor out to the hub instead of floating its root in mid-air.
const MAX_ANCHOR_LATERAL_FRAC: f32 = 0.64;

/// Orthonormal gear frame for a given mount angle, in host-local space.
#[derive(Clone, Copy, Debug)]
struct GearFrame {
    /// Outward radial — the direction struts extend (belly-ward).
    r_hat: Vec3,
    /// Lateral axis — main-leg spacing and the wheel axle. Pinned to +X.
    lateral: Vec3,
    /// Fore axis — completes the right-handed basis (≈ body +Y).
    fore: Vec3,
}

fn gear_frame(angle: f32) -> GearFrame {
    let r_hat = Vec3::new(angle.sin(), 0.0, angle.cos());
    // Lateral = r̂ × ŷ. Pin to the +X hemisphere so the "+lateral" leg is
    // always on the right; otherwise the cross product flips with the mount
    // angle and a left/right pair would swap sides (the wing-thickness footgun).
    let mut lateral = r_hat.cross(Vec3::Y).normalize_or(Vec3::X);
    if lateral.x < 0.0 {
        lateral = -lateral;
    }
    let fore = lateral.cross(r_hat).normalize_or(Vec3::Y);
    GearFrame {
        r_hat,
        lateral,
        fore,
    }
}

/// One landing-gear leg's contact frame in **host-local coordinates**,
/// relative to the gear mount origin (the host axis point at the mount
/// station). Mirrors the per-leg geometry [`build_gear_mesh`] draws, so a
/// physics wheel placed here sits exactly under the rendered wheel.
#[derive(Clone, Copy, Debug)]
pub struct GearLegFrame {
    /// Strut top — where the leg meets the host skin. The suspension ray
    /// origin (and the visual strut's high end).
    pub strut_top: Vec3,
    /// Suspension axis: outward radial `r̂`, belly-ward. Struts extend and the
    /// ground ray casts along this.
    pub susp_dir: Vec3,
    /// Roll axis: `fore` (≈ body +Y). The wheel rolls along this.
    pub roll_dir: Vec3,
    /// Axle axis: `lateral`. The wheel spins about this.
    pub axle_dir: Vec3,
}

/// Per-leg contact frames for a gearbox at `angle` on a host of radius
/// `parent_radius` — one entry for a centred nose leg, two for a main pair.
/// Shared by the game's wheel-physics builder so the collider wheels and the
/// rendered wheels never disagree (the same role [`build_gear_mesh`] plays for
/// the visuals).
pub fn gear_leg_frames(gear: &Gear, angle: f32, parent_radius: f32) -> Vec<GearLegFrame> {
    let frame = gear_frame(angle);
    leg_offsets(gear, parent_radius)
        .into_iter()
        .map(|off| GearLegFrame {
            strut_top: frame.r_hat * parent_radius + frame.lateral * off,
            susp_dir: frame.r_hat,
            roll_dir: frame.fore,
            axle_dir: frame.lateral,
        })
        .collect()
}

/// Lateral offsets of each leg's mount base, in metres along the lateral axis.
/// One centred leg, or a `±track` pair.
fn leg_offsets(gear: &Gear, parent_radius: f32) -> Vec<f32> {
    if gear.legs() >= 2 {
        let track = gear.track_fraction * parent_radius;
        vec![-track, track]
    } else {
        vec![0.0]
    }
}

/// Build the host-local mesh for a gearbox mounted at `angle` on a host of
/// radius `parent_radius`. Draws every leg (one or a pair) in one mesh.
pub fn build_gear_mesh(gear: &Gear, angle: f32, parent_radius: f32) -> Mesh {
    let frame = gear_frame(angle);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Strut half-width: a thin square post, scaled off the wheel so the
    // proportions read at any size.
    let strut_half = (gear.wheel_radius * 0.28).max(0.05);
    let wheel_half_thickness = (gear.wheel_radius * 0.35).max(0.04);

    for off in leg_offsets(gear, parent_radius) {
        let end = frame.r_hat * (parent_radius + gear.strut_length) + frame.lateral * off;

        // Leg structure. What reads as "airliner" is a **vertical oleo** over
        // the wheels — real mains attach out at the wing, so their struts drop
        // straight down; a single hub-to-belly diagonal reads as a glider
        // outrigger (user-rejected 2026-07-30). A leg inside the hull
        // silhouette is just that vertical oleo from the skin. A wide-track
        // leg (`track_fraction > 1`) gets the truss a real cantilever stance
        // has: a **gear beam** running laterally from the lower skin quadrant
        // out over the wheel, the vertical oleo from the beam's end down to
        // the hub, and a thin diagonal **side-stay** bracing beam-root to
        // mid-oleo. The wheel keeps the same drop either way, so the physics
        // contact frame ([`gear_leg_frames`]) is untouched by any of this.
        let leg = leg_structure(parent_radius, off, &frame);
        // Vertical oleo: elbow straight down to the hub. The beam + side-stay
        // that carry a wide leg's elbow live in [`build_gear_struct_mesh`] —
        // airframe structure rendered in hull material, not gear black.
        let oleo_len = (end - leg.elbow).length().max(1.0e-4);
        append_box(
            (leg.elbow + end) * 0.5,
            frame.lateral,
            frame.fore,
            frame.r_hat,
            Vec3::new(strut_half, strut_half, oleo_len * 0.5),
            &mut positions,
            &mut indices,
        );

        // Wheels: one per leg, or a fore/aft **tandem bogie** of several. They
        // hang at the strut end with their axles along the lateral axis (rolls
        // fore/aft). The bogie wheels are spread along `fore`, with a short beam
        // connecting them.
        let wheels = gear.wheels_per_leg.max(1);
        let pitch = gear.wheel_radius * 2.3; // fore/aft centre-to-centre spacing
        let span = pitch * (wheels.saturating_sub(1)) as f32;
        if wheels > 1 {
            // Bogie beam linking the wheel hubs.
            append_box(
                end,
                frame.lateral,
                frame.fore,
                frame.r_hat,
                Vec3::new(strut_half * 0.8, span * 0.5 + strut_half, strut_half * 0.8),
                &mut positions,
                &mut indices,
            );
        }
        for w in 0..wheels {
            let t = w as f32 - (wheels - 1) as f32 * 0.5; // centred row
            let hub = end + frame.fore * (t * pitch);
            append_cylinder(
                hub,
                frame.lateral,
                gear.wheel_radius,
                wheel_half_thickness,
                &mut positions,
                &mut indices,
            );
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
    // Per-quad-unique box verts give crisp strut faces; the wheel barrel shares
    // ring verts so it shades round. `compute_smooth_normals` (unlike
    // `compute_flat_normals`) works on indexed geometry, which this is.
    mesh.compute_smooth_normals();
    crate::part_mesh::add_raytracing_tangents(&mut mesh);
    mesh
}

/// Per-leg cantilever geometry shared by the gear and structure meshes: the
/// hull **anchor** (true skin point at the clamped lateral offset), and the
/// **elbow** directly above the wheel at the anchor's depth — the beam runs
/// anchor→elbow horizontally, the oleo drops elbow→hub vertically.
struct LegStructure {
    anchor: Vec3,
    elbow: Vec3,
    beam_len: f32,
}

fn leg_structure(parent_radius: f32, off: f32, frame: &GearFrame) -> LegStructure {
    let lat_max = parent_radius * MAX_ANCHOR_LATERAL_FRAC;
    let anchor_lat = off.clamp(-lat_max, lat_max);
    let anchor_radial = (parent_radius * parent_radius - anchor_lat * anchor_lat)
        .max(0.0)
        .sqrt();
    LegStructure {
        anchor: frame.r_hat * anchor_radial + frame.lateral * anchor_lat,
        elbow: frame.r_hat * anchor_radial + frame.lateral * off,
        beam_len: (off - anchor_lat).abs(),
    }
}

/// Build the **airframe-structure** half of a gearbox — the horizontal gear
/// beam and diagonal side-stay that carry a wide-track leg's elbow out from
/// the hull. Split from [`build_gear_mesh`] so callers render it in **hull
/// material**: on a real aircraft this structure is wing/fairing, not the
/// matte-black oleo/wheel assembly, and drawing it dark read as scaffolding
/// hanging under the belly (user-rejected 2026-07-30). `None` when every leg
/// sits inside the hull silhouette (nose gear, narrow mains) — there is no
/// beam to draw. Once a wing-body fairing encloses the junction these members
/// are mostly hidden inside it; they remain for the sliver outboard of the
/// fairing and for hulls without one.
pub fn build_gear_struct_mesh(gear: &Gear, angle: f32, parent_radius: f32) -> Option<Mesh> {
    let frame = gear_frame(angle);
    let strut_half = (gear.wheel_radius * 0.28).max(0.05);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for off in leg_offsets(gear, parent_radius) {
        let leg = leg_structure(parent_radius, off, &frame);
        if leg.beam_len <= 0.05 {
            continue;
        }
        // Gear beam (slightly chunkier than the oleo, like the real spar).
        append_box(
            (leg.anchor + leg.elbow) * 0.5,
            frame.lateral,
            frame.fore,
            frame.r_hat,
            Vec3::new(
                leg.beam_len * 0.5 + strut_half,
                strut_half * 1.2,
                strut_half * 1.2,
            ),
            &mut positions,
            &mut indices,
        );
        // Side-stay: thin diagonal from the beam root to two-thirds down the
        // oleo — the brace that makes a wide leg read as engineered rather
        // than propped. Only worth drawing when there is a real beam to brace.
        if leg.beam_len > 0.5 {
            let hub = frame.r_hat * (parent_radius + gear.strut_length) + frame.lateral * off;
            let stay_foot = leg.elbow + (hub - leg.elbow) * 0.65;
            let stay_axis = stay_foot - leg.anchor;
            let stay_len = stay_axis.length().max(1.0e-4);
            let stay_dir = stay_axis / stay_len;
            let stay_side = frame.fore.cross(stay_dir).normalize_or(frame.lateral);
            append_box(
                (leg.anchor + stay_foot) * 0.5,
                stay_side,
                frame.fore,
                stay_dir,
                Vec3::new(strut_half * 0.5, strut_half * 0.5, stay_len * 0.5),
                &mut positions,
                &mut indices,
            );
        }
    }
    if positions.is_empty() {
        return None;
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
    Some(mesh)
}

/// Build the **stow bay** box for a gearbox: the volume *inside* the host that
/// will house the gear when it retracts (`docs/gameplay/construction.md` §4.4 — the
/// future recess morph carves this into the belly). The box is flush with the
/// host skin and extends inward toward the body axis, sized to swallow the
/// folded legs + wheels. Pure geometry, like [`build_gear_mesh`]; the editor
/// renders it as an x-ray ghost. Not used in-game yet.
pub fn build_gear_bay_mesh(gear: &Gear, angle: f32, parent_radius: f32) -> Mesh {
    let frame = gear_frame(angle);
    let track = if gear.legs() >= 2 {
        gear.track_fraction * parent_radius
    } else {
        0.0
    };
    // Lateral: gear retracts inboard, so the stowed footprint is roughly half
    // the deployed track plus a wheel — not the full track. Bounded so the box
    // can fit inside the cylinder. Fore/aft: room for a leg to lie down. Radial
    // depth: a wheel diameter, clamped so it doesn't reach the far side.
    let half_lateral = (track * 0.5 + gear.wheel_radius).min(parent_radius * 0.85);
    let half_fore = gear.strut_length * 0.5 + gear.wheel_radius;
    let depth = (2.0 * gear.wheel_radius + 0.15).min(parent_radius * 0.6);
    let half_radial = depth * 0.5;
    // Inset the outer face below the skin so the box's lateral corners stay
    // *inside* the curved hull rather than poking out the belly sides. At the
    // outer radial `outer`, a corner at ±half_lateral is exactly `parent_radius`
    // from the axis, i.e. flush with the skin; everything inboard is contained.
    let outer = (parent_radius * parent_radius - half_lateral * half_lateral)
        .max(0.0)
        .sqrt();
    let center = frame.r_hat * (outer - half_radial);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    append_box(
        center,
        frame.lateral,
        frame.fore,
        frame.r_hat,
        Vec3::new(half_lateral, half_fore, half_radial),
        &mut positions,
        &mut indices,
    );

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

/// Append a box centred at `center` with the given (already-unit) axes and
/// per-axis half-extents. Each face owns its 4 vertices so smoothing yields
/// crisp face normals.
fn append_box(
    center: Vec3,
    x_axis: Vec3,
    y_axis: Vec3,
    z_axis: Vec3,
    half: Vec3,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let x = x_axis * half.x;
    let y = y_axis * half.y;
    let z = z_axis * half.z;
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
    // CCW-outward winding for each of the 6 faces.
    let faces = [
        (0, 3, 2, 1), // −z
        (4, 5, 6, 7), // +z
        (0, 1, 5, 4), // −y
        (2, 3, 7, 6), // +y
        (1, 2, 6, 5), // +x
        (0, 4, 7, 3), // −x
    ];
    for (a, b, c, d) in faces {
        let base = positions.len() as u32;
        for v in [p[a], p[b], p[c], p[d]] {
            positions.push([v.x, v.y, v.z]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Append a closed cylinder centred at `center` with its axis along `axis`,
/// the given `radius` and `half_length`. The barrel shares ring vertices (so
/// it shades round); the two caps get their own vertices (so the rim stays
/// crisp).
fn append_cylinder(
    center: Vec3,
    axis: Vec3,
    radius: f32,
    half_length: f32,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let n = axis.normalize_or(Vec3::X);
    let u = n.any_orthonormal_vector();
    let v = n.cross(u).normalize_or(Vec3::Y);
    let front = center + n * half_length;
    let back = center - n * half_length;

    // ---- Barrel (shared ring verts → smooth shading around) --------------
    let barrel_start = positions.len() as u32;
    for i in 0..WHEEL_SEGMENTS {
        let a = std::f32::consts::TAU * (i as f32) / (WHEEL_SEGMENTS as f32);
        let radial = u * a.cos() + v * a.sin();
        push_pos(front + radial * radius, positions); // even index: front ring
        push_pos(back + radial * radius, positions); // odd index: back ring
    }
    for i in 0..WHEEL_SEGMENTS {
        let next = (i + 1) % WHEEL_SEGMENTS;
        let f0 = barrel_start + i * 2;
        let b0 = f0 + 1;
        let f1 = barrel_start + next * 2;
        let b1 = f1 + 1;
        // Outward-facing quad (front→back→back_next→front_next).
        indices.extend_from_slice(&[f0, b0, b1, f0, b1, f1]);
    }

    append_cap(front, u, v, radius, true, positions, indices);
    append_cap(back, u, v, radius, false, positions, indices);
}

#[allow(clippy::too_many_arguments)]
fn append_cap(
    center: Vec3,
    u: Vec3,
    v: Vec3,
    radius: f32,
    front: bool,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let center_i = positions.len() as u32;
    push_pos(center, positions);
    let ring_start = positions.len() as u32;
    for i in 0..WHEEL_SEGMENTS {
        let a = std::f32::consts::TAU * (i as f32) / (WHEEL_SEGMENTS as f32);
        let radial = u * a.cos() + v * a.sin();
        push_pos(center + radial * radius, positions);
    }
    for i in 0..WHEEL_SEGMENTS {
        let next = (i + 1) % WHEEL_SEGMENTS;
        let a = ring_start + i;
        let b = ring_start + next;
        // Front cap faces +n, back cap faces −n → opposite winding.
        if front {
            indices.extend_from_slice(&[center_i, a, b]);
        } else {
            indices.extend_from_slice(&[center_i, b, a]);
        }
    }
}

fn push_pos(p: Vec3, positions: &mut Vec<[f32; 3]>) {
    positions.push([p.x, p.y, p.z]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn main_gear() -> Gear {
        Gear {
            strut_length: 1.2,
            wheel_radius: 0.35,
            track_fraction: 0.6,
            wheels_per_leg: 1,
            dry_mass: 0.0,
        }
    }

    fn nose_gear() -> Gear {
        Gear {
            strut_length: 1.0,
            wheel_radius: 0.3,
            track_fraction: 0.0,
            wheels_per_leg: 1,
            dry_mass: 0.0,
        }
    }

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
    fn leg_count_follows_track_fraction() {
        assert_eq!(main_gear().legs(), 2);
        assert_eq!(nose_gear().legs(), 1);
    }

    #[test]
    fn main_gear_box_plus_two_wheels_vertex_count() {
        // Belly mount (angle = π). Two legs: each = 1 strut box (6 faces × 4)
        // + 1 wheel (barrel 2·SEG shared + 2 caps of 1 + SEG).
        let m = build_gear_mesh(&main_gear(), std::f32::consts::PI, 1.25);
        let per_box = 24;
        let per_wheel = (2 * WHEEL_SEGMENTS + 2 * (1 + WHEEL_SEGMENTS)) as usize;
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        assert_eq!(pos.len(), 2 * (per_box + per_wheel));
        assert!(m.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(m.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
    }

    #[test]
    fn bogie_draws_multiple_wheels_spread_fore_aft() {
        // A tandem bogie (wheels_per_leg > 1) renders more geometry than a
        // single-wheel leg and spreads its wheels along the fore/aft (Y) axis.
        let single = build_gear_mesh(&main_gear(), std::f32::consts::PI, 1.25);
        let bogie = Gear {
            wheels_per_leg: 4,
            ..main_gear()
        };
        let m = build_gear_mesh(&bogie, std::f32::consts::PI, 1.25);
        let single_v = single
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap()
            .len();
        let bogie_v = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap()
            .len();
        assert!(
            bogie_v > single_v,
            "bogie adds wheels (+beam): {bogie_v} > {single_v}"
        );
        // Fore/aft spread (Y extent) is larger than a single wheel's thickness.
        let (min, max) = extents(&m);
        assert!(
            max.y - min.y > 2.0 * bogie.wheel_radius,
            "wheels spread fore/aft"
        );
    }

    #[test]
    fn main_gear_reaches_both_sides_and_hangs_below() {
        // Belly mount: legs spread to ±X (the track) and hang to −Z (below the
        // belly), starting from the skin at z = −parent_radius.
        let parent_radius = 1.25;
        let m = build_gear_mesh(&main_gear(), std::f32::consts::PI, parent_radius);
        let (min, max) = extents(&m);
        assert!(max.x > 0.0, "right leg reaches +X");
        assert!(min.x < 0.0, "left leg reaches −X");
        // The deepest geometry is below the skin radius (struts + wheels hang).
        assert!(
            min.z < -parent_radius,
            "gear hangs below the belly skin (min.z {} < −{})",
            min.z,
            parent_radius
        );
    }

    #[test]
    fn stow_bay_stays_inside_the_hull() {
        // Every bay vertex must lie within the host cylinder's cross-section
        // (distance from the body Y axis ≤ parent_radius) so the reserved
        // volume sits *inside* the fuselage, not poking out the belly sides.
        let parent_radius = 1.25_f32;
        for gear in [main_gear(), nose_gear()] {
            let m = build_gear_bay_mesh(&gear, std::f32::consts::PI, parent_radius);
            let pos = m
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .unwrap()
                .as_float3()
                .unwrap();
            for p in pos {
                let r2 = p[0] * p[0] + p[2] * p[2];
                assert!(
                    r2 <= parent_radius * parent_radius + 1e-3,
                    "bay vertex {p:?} (r²={r2}) pokes outside radius {parent_radius}",
                );
            }
        }
    }

    #[test]
    fn leg_frames_match_mesh_geometry() {
        // Belly mount (angle = π): main pair straddles ±X with the strut top at
        // the belly skin (z = −parent_radius), suspension pointing belly-ward
        // (−Z), rolling fore/aft (≈Y), axle lateral (≈X). The nose gear is one
        // centred leg. This is the contract the wheel-physics builder relies on
        // to sit collider wheels under the rendered ones.
        let parent_radius = 1.25;
        let main = gear_leg_frames(&main_gear(), std::f32::consts::PI, parent_radius);
        assert_eq!(main.len(), 2, "main gear is a left/right pair");
        let track = main_gear().track_fraction * parent_radius;
        for leg in &main {
            assert!(
                (leg.strut_top.z + parent_radius).abs() < 1e-4,
                "strut top at belly skin"
            );
            assert!(leg.susp_dir.z < -0.99, "suspension points belly-ward (−Z)");
            assert!(leg.axle_dir.x.abs() > 0.99, "axle is lateral (±X)");
            assert!(leg.roll_dir.y.abs() > 0.99, "rolls fore/aft (≈Y)");
            assert!(
                (leg.strut_top.x.abs() - track).abs() < 1e-4,
                "legs at ±track"
            );
        }
        assert!(
            main[0].strut_top.x * main[1].strut_top.x < 0.0,
            "one leg each side"
        );

        let nose = gear_leg_frames(&nose_gear(), std::f32::consts::PI, parent_radius);
        assert_eq!(nose.len(), 1, "nose gear is a single centred leg");
        assert!(nose[0].strut_top.x.abs() < 1e-4, "centred on X");
    }

    #[test]
    fn wide_track_wheels_outboard_with_hull_anchored_struts() {
        // `track_fraction > 1` (the airliner stance): the wheels sit outboard
        // of the hull silhouette, but every strut still roots on the skin —
        // its topmost vertices stay within the host cross-section radius
        // instead of floating in mid-air beside the fuselage.
        let parent_radius = 1.65_f32;
        let wide = Gear {
            track_fraction: 2.2,
            ..main_gear()
        };
        let m = build_gear_mesh(&wide, std::f32::consts::PI, parent_radius);
        let (min, max) = extents(&m);
        let track = wide.track_fraction * parent_radius;
        assert!(
            max.x > track - wide.wheel_radius && min.x < -(track - wide.wheel_radius),
            "wheels reach ±track ({} vs ±{track})",
            max.x
        );
        // The physics frames put the contact under the hub at ±track.
        let legs = gear_leg_frames(&wide, std::f32::consts::PI, parent_radius);
        for leg in &legs {
            assert!(
                (leg.strut_top.x.abs() - track).abs() < 1e-4,
                "suspension line stays over the wheel"
            );
        }
        // The carrying structure (beam + stay) exists for a wide leg and
        // anchors on the hull: at least one vertex lies within the skin
        // radius of the body axis (cross-section distance).
        let s = build_gear_struct_mesh(&wide, std::f32::consts::PI, parent_radius)
            .expect("wide track has a beam");
        let pos = s
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let anchored = pos
            .iter()
            .any(|p| (p[0] * p[0] + p[2] * p[2]).sqrt() <= parent_radius + 1e-3);
        assert!(anchored, "gear structure roots on the hull skin");
        // A narrow-track gearbox has no beam and therefore no structure mesh.
        assert!(build_gear_struct_mesh(&main_gear(), std::f32::consts::PI, parent_radius).is_none());
    }

    #[test]
    fn nose_gear_is_a_single_centred_leg() {
        // One leg, centred on the mount plane: X is symmetric about 0 and the
        // span is just the wheel/strut width, not a track.
        let m = build_gear_mesh(&nose_gear(), std::f32::consts::PI, 1.25);
        let (min, max) = extents(&m);
        assert!(
            (min.x + max.x).abs() < 1e-4,
            "single leg is centred on X (min {} max {})",
            min.x,
            max.x
        );
        // Width is small (wheel half-thickness), nowhere near a 0.6·radius track.
        assert!(max.x < 0.3, "no lateral track for the nose leg");
    }
    /// See `engine_mesh`'s copy: a mesh that misses the attribute set is
    /// skipped by the BLAS builder silently.
    #[test]
    fn gear_meshes_are_raytracing_ready() {
        assert!(crate::part_mesh::is_raytracing_ready(&build_gear_mesh(
            &main_gear(),
            0.0,
            1.5
        )));
        assert!(crate::part_mesh::is_raytracing_ready(&build_gear_bay_mesh(
            &nose_gear(),
            0.0,
            1.5
        )));
    }
}
