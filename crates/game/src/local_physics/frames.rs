//! Inertial <-> bubble-frame conversions (ship SLF / EVA body-centered) and anchors.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/regimes.md`).

#[allow(unused_imports)]
use super::*;

use bevy::math::{DMat3, DQuat, DVec3};
use thalos_body_render::HeightSource;
use thalos_physics_canonical::body_centered::{
    BodyCenteredState, body_centered_to_inertial, inertial_to_body_centered,
};
use thalos_physics_canonical::canonical::TranslationalState;
use thalos_physics_canonical::surface_local::{
    SurfaceAnchor, SurfaceLocalFrame, SurfaceLocalState, inertial_to_surface_local, surface_local_to_inertial,
};
use thalos_physics_canonical::types::{AttitudeState, BodyState, VesselKind};



pub(crate) struct BubbleFrame {
    pub(crate) position_m: DVec3,
    pub(crate) rotation: DQuat,
    pub(crate) linear_velocity_m_s: DVec3,
    pub(crate) angular_velocity_rad_s: DVec3,
}

/// Convert canonical inertial state into the Avian rigid body's frame.
///
/// The Avian body lives in **body-centered inertial** coordinates: the origin
/// tracks the dominant body's centre but the axes are the parent inertial
/// axes (no rotation). Position and velocity are simple offsets from the
/// body's centre, and the craft's attitude / angular velocity are expressed
/// in inertial axes — matching how Avian treats the frame it integrates in.
///
/// Avian's `AngularVelocity` lives in the rigid body's surrounding frame
/// (here, inertial), while [`AttitudeState::angular_velocity`] is expressed
/// in the craft body frame, so we rotate by `orientation`.
pub(crate) fn inertial_to_bubble_frame(
    body_state: &BodyState,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> BubbleFrame {
    let state = inertial_to_body_centered(body_state, translation, attitude);
    BubbleFrame {
        position_m: state.translation_bc.position,
        rotation: state.attitude.orientation.normalize(),
        linear_velocity_m_s: state.translation_bc.velocity,
        angular_velocity_rad_s: state.attitude.orientation * state.attitude.angular_velocity,
    }
}


pub(crate) fn bubble_frame_to_inertial(
    body_state: &BodyState,
    position_m: DVec3,
    rotation: DQuat,
    linear_velocity_m_s: DVec3,
    angular_velocity_rad_s: DVec3,
) -> (TranslationalState, AttitudeState) {
    let orientation = rotation.normalize();
    let state = BodyCenteredState {
        translation_bc: TranslationalState {
            position: position_m,
            velocity: linear_velocity_m_s,
        },
        attitude: AttitudeState {
            orientation,
            angular_velocity: orientation.inverse() * angular_velocity_rad_s,
        },
    };
    body_centered_to_inertial(body_state, state)
}

/// Convert canonical inertial state into the **ship** Avian body's frame.
///
/// Ships use the **surface-local frame (SLF)**: a body-fixed tangent frame
/// anchored at a surface point, Y-up, small coordinates near the anchor
/// (`thalos_physics_canonical::surface_local`, `docs/surface_local.md`). The
/// frame co-rotates with the body, so a craft parked on or taxiing across the
/// surface is ~stationary instead of translating at the surface co-rotation
/// speed (`ω×r`, hundreds of m/s), and the ground colliders are genuinely
/// static geometry (see [`sync_terrain_collider_pose`]). Frame velocity is
/// airspeed (the atmosphere co-rotates).
///
/// EVA keeps the body-centered [`inertial_to_bubble_frame`] seam — its capsule
/// is owned directly by the on-foot controller and never touches a runway.
pub(crate) fn inertial_to_ship_frame(
    body_state: &BodyState,
    frame: &SurfaceLocalFrame,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> BubbleFrame {
    let state = inertial_to_surface_local(body_state, frame, translation, attitude);
    BubbleFrame {
        position_m: state.position_m,
        rotation: state.orientation_frame.normalize(),
        linear_velocity_m_s: state.velocity_m_s,
        // Avian's `AngularVelocity` lives in the surrounding (SLF) axes;
        // `SurfaceLocalState` carries it in the craft body frame.
        angular_velocity_rad_s: state.orientation_frame * state.angular_velocity_body,
    }
}

pub(crate) fn ship_frame_to_inertial(
    body_state: &BodyState,
    frame: &SurfaceLocalFrame,
    position_m: DVec3,
    rotation: DQuat,
    linear_velocity_m_s: DVec3,
    angular_velocity_rad_s: DVec3,
) -> (TranslationalState, AttitudeState) {
    let orientation = rotation.normalize();
    let state = SurfaceLocalState {
        position_m,
        velocity_m_s: linear_velocity_m_s,
        orientation_frame: orientation,
        angular_velocity_body: orientation.inverse() * angular_velocity_rad_s,
    };
    surface_local_to_inertial(body_state, frame, state)
}

/// Pick the inertial→Avian conversion for the craft's kind: ships are
/// surface-local (body-fixed tangent frame), EVA is body-centered inertial.
pub(crate) fn inertial_to_craft_frame(
    kind: VesselKind,
    body_state: &BodyState,
    frame: &SurfaceLocalFrame,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> BubbleFrame {
    match kind {
        VesselKind::Ship => inertial_to_ship_frame(body_state, frame, translation, attitude),
        VesselKind::Eva => inertial_to_bubble_frame(body_state, translation, attitude),
    }
}

/// Inverse of [`inertial_to_craft_frame`].
pub(crate) fn craft_frame_to_inertial(
    kind: VesselKind,
    body_state: &BodyState,
    frame: &SurfaceLocalFrame,
    position_m: DVec3,
    rotation: DQuat,
    linear_velocity_m_s: DVec3,
    angular_velocity_rad_s: DVec3,
) -> (TranslationalState, AttitudeState) {
    match kind {
        VesselKind::Ship => ship_frame_to_inertial(
            body_state,
            frame,
            position_m,
            rotation,
            linear_velocity_m_s,
            angular_velocity_rad_s,
        ),
        VesselKind::Eva => bubble_frame_to_inertial(
            body_state,
            position_m,
            rotation,
            linear_velocity_m_s,
            angular_velocity_rad_s,
        ),
    }
}

/// Build a [`SurfaceAnchor`] at the surface projection of an inertial
/// position, sampling terrain elevation when a height source is available
/// (the anchor elevation only places the frame origin — conversions are
/// exact regardless, so a missing source degrades to reference-radius
/// origin, not to incorrectness).
pub(crate) fn surface_anchor_under(
    body_state: &BodyState,
    height_source: Option<&dyn HeightSource>,
    position_inertial: DVec3,
) -> SurfaceAnchor {
    let dir_body = (body_state.orientation.inverse() * (position_inertial - body_state.position))
        .normalize_or_zero();
    let dir_body = if dir_body == DVec3::ZERO {
        DVec3::Y
    } else {
        dir_body
    };
    let elevation_m = height_source
        .and_then(|hs| hs.sample_height_m(dir_body.as_vec3(), PHYSICS_QUERY_TILE_LOD_M))
        .map(|h| h as f64)
        .unwrap_or(0.0);
    SurfaceAnchor {
        dir_body,
        elevation_m,
    }
}

pub(crate) fn level_attitude_for_body_dir(body_orientation: DQuat, up_body: DVec3) -> DQuat {
    let basis = thalos_body_render::TerrainPatchBasis::from_normal(up_body);
    let nose_body = basis.tangent_z;
    let dorsal_body = up_body.normalize();
    let right_body = nose_body.cross(dorsal_body).normalize();
    let craft_to_body = DMat3::from_cols(right_body, nose_body, dorsal_body);
    (body_orientation * DQuat::from_mat3(&craft_to_body)).normalize()
}

