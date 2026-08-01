//! The **surface orientation authority** — body-fixed ↔ world frame
//! resolution from authored bodies + evaluated states alone (no ECS query).
//!
//! This is the frame every surface consumer must share — terrain renderers,
//! height sources, the view anchor, capture framings. Using the raw ephemeris
//! `BodyState::orientation` instead is wrong for tidally-locked moons (the two
//! frames differ by the full lock rotation — the Mira tile-shell 132°
//! misplacement, INC-20260723T232652Z's successor finding).

use bevy::math::{DQuat, Mat3, Quat, Vec3};
use thalos_physics_canonical::types::BodyState;
use thalos_world::{BodyDefinition, BodyId, BodyKind};

use crate::scene::TidallyLocked;

fn tangent_axis(seed: Vec3, normal: Vec3) -> Option<Vec3> {
    let tangent = seed - normal * seed.dot(normal);
    (tangent.length_squared() > 1.0e-8).then(|| tangent.normalize())
}

/// World → body-fixed orientation for a tidally-locked body: body-local +Z
/// faces the parent, +Y follows the orbit normal (negated to keep the terrain
/// generator's north convention).
pub fn tidal_lock_world_to_body_orientation(
    body_state: &BodyState,
    parent_state: &BodyState,
) -> Option<Quat> {
    let to_parent = parent_state.position - body_state.position;
    let len = to_parent.length();
    if len < 1.0 {
        return None;
    }

    let z_world = (to_parent / len).as_vec3();

    // `keplerian_basis` uses XZ as the zero-inclination orbital plane and
    // +Y as ecliptic north. For a prograde zero-inclination orbit,
    // r x v points along -Y, so negate it to keep body-local +Y aligned
    // with the terrain generator's north convention.
    let rel_pos = body_state.position - parent_state.position;
    let rel_vel = body_state.velocity - parent_state.velocity;
    let angular_momentum = rel_pos.cross(rel_vel);
    let y_seed = if angular_momentum.length_squared() > f64::EPSILON {
        (-angular_momentum.normalize()).as_vec3()
    } else {
        Vec3::Y
    };

    let y_world = tangent_axis(y_seed, z_world)
        .or_else(|| tangent_axis(Vec3::Y, z_world))
        .or_else(|| tangent_axis(Vec3::X, z_world))?;
    let x_world = y_world.cross(z_world).normalize();
    let y_world = z_world.cross(x_world).normalize();

    let body_to_world = Mat3::from_cols(x_world, y_world, z_world);
    Some(Quat::from_mat3(&body_to_world).inverse().normalize())
}

/// f32 body-fixed → world surface orientation. Prefer the f64 form for
/// planet-scale placement; see [`surface_body_to_world_orientation_f64`].
pub fn surface_body_to_world_orientation(
    body_id: BodyId,
    lock: Option<&TidallyLocked>,
    states: &[BodyState],
) -> Option<Quat> {
    if let Some(lock) = lock {
        let body_state = states.get(body_id)?;
        let parent_state = states.get(lock.parent_id)?;
        return tidal_lock_world_to_body_orientation(body_state, parent_state)
            .map(|q| q.inverse().normalize());
    }

    states
        .get(body_id)
        .map(|state| state.orientation.as_quat().normalize())
}

/// f64 body-fixed → world surface orientation — the precise source the
/// real-space body grid's f32 `Transform.rotation` is derived from, and the
/// value handed to udlod's high-precision Taylor path. At planet scale this
/// rotation is applied to the camera→body vector (~radius), where f32
/// quaternion ULP is a flickering decimetre.
pub fn surface_body_to_world_orientation_f64(
    body_id: BodyId,
    lock: Option<&TidallyLocked>,
    states: &[BodyState],
) -> Option<DQuat> {
    if lock.is_some() {
        // Tidal-lock orientation is still computed in f32 internally
        // (`tidal_lock_world_to_body_orientation`); widen on the way out. This
        // is the status quo, not a regression: no player stands on a tidally
        // locked body, so the high-precision Taylor path — the only consumer
        // that needs the extra precision — is never exercised for one. Port
        // the tidal math to f64 when that changes.
        return surface_body_to_world_orientation(body_id, lock, states).map(|q| q.as_dquat());
    }

    states
        .get(body_id)
        .map(|state| state.orientation.normalize())
}

/// The authored-data tidal-lock rule — the ONE place that decides which bodies
/// are surface-locked to a parent. The runtime's spawn inserts the
/// [`TidallyLocked`] tag from this, and frame-conversion consumers without ECS
/// access (screenshot framings, saved-viewpoint replay) derive the lock from
/// it directly, so the two can never disagree.
pub fn authored_lock_parent(body: &BodyDefinition) -> Option<usize> {
    matches!(body.kind, BodyKind::Moon)
        .then_some(())
        .and(body.parent)
}

/// Surface body-fixed → world orientation resolved from authored bodies +
/// evaluated states alone (no ECS query). See the module docs — this is the
/// one orientation authority every surface consumer shares.
pub fn surface_orientation_authored(
    bodies: &[BodyDefinition],
    body_id: BodyId,
    states: &[BodyState],
) -> Option<DQuat> {
    let lock = bodies
        .get(body_id)
        .and_then(authored_lock_parent)
        .map(|parent_id| TidallyLocked { parent_id });
    surface_body_to_world_orientation_f64(body_id, lock.as_ref(), states)
}
