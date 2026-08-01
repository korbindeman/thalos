//! Drives the navball sphere's rotation from craft attitude, and publishes
//! the world→navball rotation as a resource for marker projection.
//!
//! Pipeline per frame:
//! 1. Read craft attitude (body→world) and world position from the sim.
//! 2. Build a **right-handed** local ENU frame at the craft's position
//!    around its current SOI body. Axes:
//!    `+X = East`, `+Y = North`, `+Z = Up` (= radial-out).
//!    Up = radial; North = ecliptic-Y projected orthogonal to Up;
//!    East = North × Up.
//! 3. Express craft body axes (`+Y` nose, `+Z` dorsal, `+X` right) in
//!    the local frame.
//! 4. Build the sphere rotation that maps those body axes onto the
//!    navball-world axes (forward → camera direction, up → screen up,
//!    right → screen right).
//! 5. Compose with `world→local` to produce `world→navball`, used by
//!    marker projection.

use crate::navball::render::NavballSphere;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use thalos_game_state::{SimulationState, SolarSystemState};

/// Body axis pointing out of the nose. Matches `SHIP_NOSE_BODY` in
/// `navigation.rs`.
const BODY_NOSE: DVec3 = DVec3::Y;

/// Body axis pointing through the pilot's overhead (dorsal direction).
/// First consumer of this convention — flip the sign here if the
/// navball rolls the wrong way.
const BODY_DORSAL: DVec3 = DVec3::Z;

/// Resource published every frame by [`drive_navball_attitude`]. Markers
/// project world-space directions onto the navball image via
/// `world_to_navball * d_world`: the resulting `(x, y)` is the position
/// on the navball in unit-sphere coordinates (length 1 = limb), and `z`
/// is positive for the visible hemisphere.
///
/// Stored as a `Mat3` (not a `Quat`) because the full
/// world → navball-image transform composes the sphere rotation with
/// the texture's lat/lon-to-sphere mapping, which is a reflection
/// (det = −1). The sphere mesh's *visual* rotation alone is a proper
/// rotation; only the marker projection needs the reflection.
#[derive(Resource, Debug, Clone, Copy)]
pub struct NavballFrame {
    pub world_to_navball: Mat3,
}

impl Default for NavballFrame {
    fn default() -> Self {
        Self {
            world_to_navball: Mat3::IDENTITY,
        }
    }
}

pub fn drive_navball_attitude(
    sim_state: Res<SimulationState>,
    solar_system: Res<SolarSystemState>,
    mut sphere: Query<&mut Transform, With<NavballSphere>>,
    mut frame: ResMut<NavballFrame>,
) {
    let Ok(mut transform) = sphere.single_mut() else {
        return;
    };

    let sim = &sim_state.simulation;
    let craft = sim.craft_state();
    let craft_pos = craft.translation.position;
    let q_body_to_world = craft.attitude.orientation;

    let soi_body_id = sim.dominant_body();
    let Some(states) = solar_system.states.as_deref() else {
        return;
    };
    let Some(soi_body_pos) = states.get(soi_body_id).map(|s| s.position) else {
        return;
    };

    let computed = compute_navball_frame(q_body_to_world, craft_pos, soi_body_pos);
    transform.rotation = computed.sphere_rotation;
    frame.world_to_navball = computed.world_to_navball;
}

struct ComputedFrame {
    sphere_rotation: Quat,
    world_to_navball: Mat3,
}

/// Pure math: compute the navball's sphere rotation and the world→navball
/// transform from craft attitude and the position of its SOI body.
fn compute_navball_frame(
    q_body_to_world: DQuat,
    craft_pos: DVec3,
    soi_body_pos: DVec3,
) -> ComputedFrame {
    // ---- Local ENU basis at the craft, in world frame ------------------
    // Right-handed: +X=East, +Y=North, +Z=Up. Cross identity: E×N = U.
    let radial = craft_pos - soi_body_pos;
    let radial_len2 = radial.length_squared();
    if radial_len2 < 1e-6 {
        return ComputedFrame {
            sphere_rotation: Quat::IDENTITY,
            world_to_navball: Mat3::IDENTITY,
        };
    }
    let up = radial / radial_len2.sqrt();

    let world_y = DVec3::Y;
    let mut north = world_y - world_y.dot(up) * up;
    if north.length_squared() < 1e-6 {
        let world_x = DVec3::X;
        north = world_x - world_x.dot(up) * up;
    }
    let north = north.normalize();
    let east = north.cross(up);

    // ---- q_world_to_local ---------------------------------------------
    // Standard right-handed ENU: column order (east, north, up). Local
    // +X = East, +Y = North, +Z = Up. det = E·(N×U) = E·E = +1.
    let local_to_world = DMat3::from_cols(east, north, up);
    let world_to_local = local_to_world.transpose();

    // ---- Craft body axes expressed in local frame ----------------------
    let nose_local = world_to_local * (q_body_to_world * BODY_NOSE);
    let dorsal_local = world_to_local * (q_body_to_world * BODY_DORSAL);

    let forward_local = nose_local.normalize();
    // Re-orthogonalise dorsal against forward to guard against drift
    // produced by the attitude integrator.
    let dorsal_local = (dorsal_local - dorsal_local.dot(forward_local) * forward_local).normalize();
    // Right-hand body: X × Y = Z, so right = Y × Z = forward × dorsal.
    let right_local = forward_local.cross(dorsal_local);

    // ---- Texture-aware sphere rotation ---------------------------------
    //
    // The texture is drawn so that, after Bevy's UV-sphere mapping:
    //   * sky pole (lat +90°) lands on sphere-local +Z
    //   * "N" letter (lon 0°)  lands on sphere-local −X
    //   * "E" letter (lon 90°) lands on sphere-local −Y
    //
    // So a local-frame direction `d` is represented by the sphere point
    // `M * d`, where
    //     M : local +X (E)  →  sphere −Y
    //         local +Y (N)  →  sphere −X
    //         local +Z (U)  →  sphere +Z       (det M = −1).
    //
    // We want the sphere transform `Q` such that the texture point
    // representing each body axis ends up on its corresponding
    // navball-image axis:
    //     Q · (M · forward_local) = +Z   (visible centre / toward camera)
    //     Q · (M · dorsal_local)  = +Y   (top of screen)
    //     Q · (M · right_local)   = +X   (right of screen).
    //
    // Let `N` be the fixed matrix with those right-hand-side columns
    // (det N = −1) and `B` = [forward | dorsal | right] (det B = +1).
    // Then `Q = N · B⁻¹ · M⁻¹` = `N · Bᵀ · M⁻¹`, and
    //     det Q = (−1) · (+1) · (−1)⁻¹ = +1
    // — a proper rotation, the right input for `from_mat3`.
    let body_basis = DMat3::from_cols(forward_local, dorsal_local, right_local);
    let m_local_to_sphere = local_to_sphere_texture_mapping();
    let n_targets = nav_axis_targets();
    let q_sphere_mat = n_targets * body_basis.transpose() * m_local_to_sphere.inverse();
    let q_sphere_d = DQuat::from_mat3(&q_sphere_mat);

    // World → navball-image transform for marker projection. Compose
    // (world → local) → (local → sphere via M) → (sphere → navball via
    // Q). Stored as a Mat3 because the composition is a reflection
    // (det = −1) — fine for projecting a direction onto the 2D image,
    // but it can't be expressed as a unit quaternion.
    let q_sphere_d_as_mat = DMat3::from_quat(q_sphere_d);
    let world_to_navball_d = q_sphere_d_as_mat * m_local_to_sphere * world_to_local;

    ComputedFrame {
        sphere_rotation: q_sphere_d.as_quat(),
        world_to_navball: Mat3::from_cols(
            world_to_navball_d.x_axis.as_vec3(),
            world_to_navball_d.y_axis.as_vec3(),
            world_to_navball_d.z_axis.as_vec3(),
        ),
    }
}

/// `M_local_to_sphere`: a local-frame direction in standard ENU coords
/// gets mapped to the sphere position where the texture for that
/// direction is drawn. See [`compute_navball_frame`] for the derivation.
fn local_to_sphere_texture_mapping() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(0.0, -1.0, 0.0), // E → sphere -Y
        DVec3::new(-1.0, 0.0, 0.0), // N → sphere -X
        DVec3::Z,                   // U → sphere +Z
    )
}

/// `N_targets`: which navball-image axis each body axis should land
/// on. Forward → +Z (visible centre), dorsal → +Y (top), right → +X.
fn nav_axis_targets() -> DMat3 {
    DMat3::from_cols(DVec3::Z, DVec3::Y, DVec3::X)
}
