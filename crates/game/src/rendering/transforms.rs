//! Per-frame transform updates: floating render origin, render frame,
//! body parent translation, ship-layer mesh compensation, ship marker
//! placement, and per-planet orientation (tidal lock + spin).

use bevy::prelude::*;
use thalos_planet_rendering::{PlanetHaloMaterial, PlanetMaterial};

use super::screen_marker_radius;
use super::types::{
    CelestialBody, FrameBodyStates, PlanetMaterials, ShipBodyMesh, ShipMarker, SimulationState,
    TidallyLocked,
};
use crate::camera::{ActiveCamera, CameraFocus, CameraFocusTarget, OrbitCamera};
use crate::coords::{
    MAP_SCALE, RenderFrame, RenderGhostFocus, RenderOrigin, SHIP_SCALE, WorldScale, to_render_pos,
};
use crate::flight_plan_view::FlightPlanView;
use crate::view::ViewMode;

fn ghost_position(
    focus: RenderGhostFocus,
    view: Option<&FlightPlanView>,
    states: &[thalos_physics::types::BodyState],
) -> bevy::math::DVec3 {
    if let Some(view) = view {
        return view.pin_for_ghost_focus(focus, states);
    }

    states
        .get(focus.parent_id)
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO)
        + focus.relative_position
}

/// Sets the render origin to the camera focus body's position so that nearby
/// objects always have small render-space coordinates (full f32 precision).
/// Updates every frame: at ~1 AU focus, the body moves ~480 m per frame
/// heliocentrically; at LEO the ship moves ~125 m per frame. f32 ulp at
/// 1000 render units (~1,000,000 km from origin) is ~120 m, which quantizes
/// per-frame motion into visible steps. A zero-threshold tracking origin
/// keeps render-space coordinates small and full-precision.
///
/// While a focus transition is active (`focus.transition_origin_start`
/// is `Some`), the origin interpolates in f64 from the captured
/// starting position to the new focus target's current position over
/// [`FOCUS_TRANSITION_DURATION_S`](crate::camera::FOCUS_TRANSITION_DURATION_S).
/// Doing the lerp here in physics space — rather than as a
/// render-space `focus_offset` applied to the camera — keeps both the
/// camera and its target near render-space `(0, 0, 0)` throughout
/// the switch, avoiding the f32 cancellation in `looking_at` that
/// otherwise shows up as scene-wide jitter when transitioning between
/// distant bodies.
pub fn update_render_origin(
    cache: Res<FrameBodyStates>,
    focus: Res<CameraFocus>,
    flight_plan: Option<Res<FlightPlanView>>,
    sim: Res<SimulationState>,
    mut origin: ResMut<RenderOrigin>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    let target_position = match focus.target {
        CameraFocusTarget::Body(body_id) => states
            .get(body_id)
            .map(|s| s.position)
            .unwrap_or(bevy::math::DVec3::ZERO),
        CameraFocusTarget::Ship => sim.simulation.ship_state().position,
        CameraFocusTarget::Ghost(ghost_focus) => {
            ghost_position(ghost_focus, flight_plan.as_deref(), states)
        }
        CameraFocusTarget::None => bevy::math::DVec3::ZERO,
    };

    origin.position = match focus.transition_origin_start {
        Some(start) => {
            let progress = crate::camera::focus_transition_progress(&focus);
            start.lerp(target_position, progress)
        }
        None => target_position,
    };
}

/// Resolve [`RenderFrame::focus_body`] — the body whose frame the
/// trajectory and ghost system are conceptually drawn in.
///
/// - Camera target is a celestial body → that body
/// - Camera target is a ghost → the body the ghost represents, plus the
///   selected encounter epoch
/// - Camera target is the player ship → ship's current SOI body
///   (so a Mira-orbiting ship gets a Mira-relative trajectory view)
/// - No target → body 0 (the star)
///
/// Distinct from [`update_render_origin`] because origin tracks the
/// camera pivot (ship for ship-focus) while the frame tracks the
/// physical SOI parent. They differ when the camera follows the ship
/// — a deliberate decoupling so the trajectory's *shape* reads in the
/// SOI body's frame even while the camera tracks the ship.
pub fn update_render_frame(
    cache: Res<FrameBodyStates>,
    focus: Res<CameraFocus>,
    sim: Res<SimulationState>,
    mut frame: ResMut<RenderFrame>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    *frame = match focus.target {
        CameraFocusTarget::Body(body_id) => RenderFrame {
            focus_body: body_id,
            focus_ghost: None,
        },
        CameraFocusTarget::Ghost(ghost_focus) => RenderFrame {
            focus_body: ghost_focus.body_id,
            focus_ghost: Some(ghost_focus),
        },
        CameraFocusTarget::Ship => RenderFrame {
            focus_body: crate::camera::find_reference_body(
                sim.simulation.ship_state().position,
                sim.simulation.bodies(),
                states,
            ),
            focus_ghost: None,
        },
        CameraFocusTarget::None => RenderFrame::default(),
    };
}

/// Drive each [`CelestialBody`] parent's translation at [`MAP_SCALE`].
///
/// The parent acts as the canonical map-side anchor: its [`BodyMesh`]
/// and [`BodyIcon`] children inherit it directly, while
/// [`ShipBodyMesh`] siblings carry a compensating local translation
/// (see [`update_ship_body_meshes`]) so they sit at `phys * SHIP_SCALE`
/// in world space. Locking the parent at MAP_SCALE means both views
/// have a stable, view-independent representation regardless of which
/// camera is active.
pub(super) fn update_body_positions(
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    mut query: Query<(&CelestialBody, &mut Transform)>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    for (body, mut transform) in &mut query {
        if let Some(state) = states.get(body.body_id) {
            transform.translation = ((state.position - origin.position) * MAP_SCALE).as_vec3();
        }
    }
}

/// Per-frame: rewrite each [`ShipBodyMesh`]'s LOCAL translation so its
/// world position lands at `(phys_pos - origin) * SHIP_SCALE` regardless
/// of what scale the parent's translation is computed at.
///
/// Parent translation is `(phys_pos - origin) * MAP_SCALE` (driven by
/// [`update_body_positions`]). For the ship-view sibling we want world
/// = `(phys_pos - origin) * SHIP_SCALE`, so the local translation that
/// achieves that is `(phys_pos - origin) * (SHIP_SCALE - MAP_SCALE)`.
///
/// Scale and rotation are not touched — those are set once at spawn
/// time (sphere placeholder uses `radius_m * SHIP_SCALE`; rings use
/// `Vec3::ONE` because the ring mesh is built at SHIP_SCALE radii;
/// post-swap impostor billboards ignore the model scale because the
/// vertex shader sizes the quad from `params.radius` directly).
pub(super) fn update_ship_body_meshes(
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    parents: Query<&CelestialBody>,
    mut ship_meshes: Query<(&ChildOf, &mut Transform), With<ShipBodyMesh>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    for (parent_link, mut tf) in &mut ship_meshes {
        let Ok(body) = parents.get(parent_link.0) else {
            continue;
        };
        let Some(state) = states.get(body.body_id) else {
            continue;
        };
        let rel = state.position - origin.position;
        tf.translation = (rel * (SHIP_SCALE - MAP_SCALE)).as_vec3();
    }
}

pub(super) fn update_ship_position(
    sim: Res<SimulationState>,
    origin: Res<RenderOrigin>,
    scale: Res<WorldScale>,
    camera_query: Query<&Transform, (With<ActiveCamera>, With<OrbitCamera>)>,
    mut query: Query<&mut Transform, (With<ShipMarker>, Without<OrbitCamera>)>,
) {
    let Ok(cam_tf) = camera_query.single() else {
        return;
    };
    for mut transform in &mut query {
        transform.translation = to_render_pos(
            sim.simulation.ship_state().position - origin.position,
            &scale,
        );
        transform.rotation = cam_tf.rotation;
        transform.scale = Vec3::splat(screen_marker_radius(
            transform.translation,
            cam_tf.translation,
        ));
    }
}

/// Rewrite each baked planet's material `orientation` quaternion every frame.
///
/// Tidally-locked bodies point their baked +Z axis at the parent (mare /
/// tidal asymmetry baked into `BodyBuilder::tidal_axis`). Free-spinning
/// bodies compose `Ry(phase) * Rx(tilt)` so the surface spins under a tilted
/// axis — this mirrors the gas-giant pipeline, where the shader applies
/// `rotation_phase` post-orientation; for the impostor shader there is no
/// such uniform, so spin must be baked into the single orientation quat.
pub(super) fn update_planet_orientations(
    query: Query<(&CelestialBody, Option<&TidallyLocked>, &PlanetMaterials)>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    mut halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    cache: Res<FrameBodyStates>,
    sim: Res<SimulationState>,
    view: Res<ViewMode>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let body_defs = sim.simulation.bodies();
    let sim_time = sim.simulation.sim_time();

    // See note on `update_planet_light_dirs` — gate inactive scale.
    let force_both = view.is_changed();
    let do_map = force_both || matches!(*view, ViewMode::Map);
    let do_ship = force_both || matches!(*view, ViewMode::Ship);

    for (body, lock, mats) in &query {
        let q = if let Some(lock) = lock {
            let Some(body_state) = states.get(body.body_id) else {
                continue;
            };
            let Some(parent_state) = states.get(lock.parent_id) else {
                continue;
            };
            let offset = parent_state.position - body_state.position;
            let len = offset.length();
            if len < 1.0 {
                continue;
            }
            let dir = (offset / len).as_vec3();
            // `from_rotation_arc` produces the shortest rotation that maps
            // `dir` onto +Z. glam's impl handles the antiparallel case with a
            // stable fallback axis, so there's no degenerate pole for the
            // moon's orbit.
            Quat::from_rotation_arc(dir, Vec3::Z)
        } else {
            let body_def = &body_defs[body.body_id];
            let period = body_def.rotation_period_s;
            // Negative periods are retrograde: Rust's `a % b` keeps the sign
            // of `a`, so (positive sim_time) % (negative period) is positive
            // and the subsequent division flips the phase sign.
            let phase = if period.abs() > 1.0 {
                ((sim_time % period) / period) as f32 * std::f32::consts::TAU
            } else {
                0.0
            };
            let tilt = body_def.axial_tilt_rad as f32;
            Quat::from_rotation_y(phase) * Quat::from_rotation_x(tilt)
        };

        // Orientation is scale-independent — same value for both materials.
        let q4 = Vec4::new(q.x, q.y, q.z, q.w);
        for (handle, halo_handle, want) in [
            (&mats.map, &mats.map_halo, do_map),
            (&mats.ship, &mats.ship_halo, do_ship),
        ] {
            if !want {
                continue;
            }
            if let Some(mat) = materials.get_mut(handle) {
                mat.params.orientation = q4;
            }
            if let Some(mat) = halo_materials.get_mut(halo_handle) {
                mat.params.orientation = q4;
            }
        }
    }
}
