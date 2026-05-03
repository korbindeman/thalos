//! Per-frame transform updates: floating render origin, render frame,
//! body parent translation, ship-layer mesh compensation, ship marker
//! placement, and per-planet orientation (tidal lock + spin).

use bevy::prelude::*;
use thalos_physics::types::{BodyDefinition, BodyId, BodyState};
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
    view: Res<ViewMode>,
    focus: Res<CameraFocus>,
    photo_mode: Res<crate::photo_mode::PhotoMode>,
    camera_query: Query<&Transform, (With<ActiveCamera>, With<OrbitCamera>)>,
    mut query: Query<(&mut Transform, &mut Visibility), (With<ShipMarker>, Without<OrbitCamera>)>,
) {
    let Ok(cam_tf) = camera_query.single() else {
        return;
    };
    let ship_soi_body = sim.simulation.dominant_body();
    let marker_visible = matches!(*view, ViewMode::Map)
        && !photo_mode.active
        && ship_marker_visible_in_focus(focus.target, ship_soi_body, sim.simulation.bodies());
    let target_visibility = if marker_visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };

    for (mut transform, mut visibility) in &mut query {
        transform.translation = to_render_pos(
            sim.simulation.ship_state().position - origin.position,
            &scale,
        );
        transform.rotation = cam_tf.rotation;
        transform.scale = Vec3::splat(screen_marker_radius(
            transform.translation,
            cam_tf.translation,
        ));
        if *visibility != target_visibility {
            *visibility = target_visibility;
        }
    }
}

fn ship_marker_visible_in_focus(
    focus_target: CameraFocusTarget,
    ship_soi_body: BodyId,
    bodies: &[BodyDefinition],
) -> bool {
    match focus_target {
        CameraFocusTarget::Ship => true,
        CameraFocusTarget::Body(body_id) => same_local_system(body_id, ship_soi_body, bodies),
        CameraFocusTarget::Ghost(ghost_focus) => {
            same_local_system(ghost_focus.body_id, ship_soi_body, bodies)
        }
        CameraFocusTarget::None => false,
    }
}

fn same_local_system(a: BodyId, b: BodyId, bodies: &[BodyDefinition]) -> bool {
    local_system_root(a, bodies) == local_system_root(b, bodies)
}

/// Return the top-level body that owns this local map context.
///
/// A planet owns its moon system; a moon resolves to that planet. The root
/// star owns interplanetary space, so a ship in the star SOI remains visible
/// only when viewing the root system.
fn local_system_root(mut body_id: BodyId, bodies: &[BodyDefinition]) -> BodyId {
    for _ in 0..bodies.len() {
        let Some(body) = bodies.get(body_id) else {
            return body_id;
        };
        let Some(parent_id) = body.parent else {
            return body_id;
        };
        let Some(parent) = bodies.get(parent_id) else {
            return body_id;
        };
        if parent.parent.is_none() {
            return body_id;
        }
        body_id = parent_id;
    }
    body_id
}

fn tangent_axis(seed: Vec3, normal: Vec3) -> Option<Vec3> {
    let tangent = seed - normal * seed.dot(normal);
    (tangent.length_squared() > 1.0e-8).then(|| tangent.normalize())
}

fn tidal_lock_orientation(body_state: &BodyState, parent_state: &BodyState) -> Option<Quat> {
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

/// Rewrite each baked planet's material `orientation` quaternion every frame.
///
/// Tidally-locked bodies point their baked +Z axis at the parent (mare /
/// tidal asymmetry baked into `BodyBuilder::tidal_axis`) while their local
/// +Y stays tied to the orbit plane. Free-spinning bodies compose
/// `Ry(phase) * Rx(tilt)` so the surface spins under a tilted axis — this
/// mirrors the gas-giant pipeline, where the shader applies `rotation_phase`
/// post-orientation; for the impostor shader there is no such uniform, so
/// spin must be baked into the single orientation quat.
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
            let Some(q) = tidal_lock_orientation(body_state, parent_state) else {
                continue;
            };
            q
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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::DVec3;

    fn body_state(position: DVec3, velocity: DVec3) -> BodyState {
        BodyState {
            position,
            velocity,
            mass_kg: 1.0,
        }
    }

    fn assert_vec3_near(actual: Vec3, expected: Vec3) {
        let err = (actual - expected).length();
        assert!(
            err < 1.0e-5,
            "expected {expected:?}, got {actual:?}, err {err}"
        );
    }

    fn assert_axes_close(a: Quat, b: Quat, max_angle_rad: f32) {
        let a = a.inverse();
        let b = b.inverse();
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let da = a * axis;
            let db = b * axis;
            let angle = da.dot(db).clamp(-1.0, 1.0).acos();
            assert!(
                angle < max_angle_rad,
                "axis {axis:?} changed by {angle} rad, expected < {max_angle_rad}"
            );
        }
    }

    #[test]
    fn tidal_lock_faces_parent_and_preserves_orbital_north() {
        let parent = body_state(DVec3::ZERO, DVec3::ZERO);
        let body = body_state(DVec3::X, DVec3::Z);

        let orientation = tidal_lock_orientation(&body, &parent).unwrap();

        assert_vec3_near(orientation * Vec3::NEG_X, Vec3::Z);
        assert_vec3_near(orientation * Vec3::Y, Vec3::Y);
    }

    #[test]
    fn tidal_lock_orientation_stays_continuous_at_antiparallel_arc() {
        fn circular_state(theta: f64) -> BodyState {
            body_state(
                DVec3::new(theta.cos(), 0.0, theta.sin()),
                DVec3::new(-theta.sin(), 0.0, theta.cos()),
            )
        }

        let parent = body_state(DVec3::ZERO, DVec3::ZERO);
        let epsilon = 1.0e-4;
        let before = tidal_lock_orientation(
            &circular_state(std::f64::consts::FRAC_PI_2 - epsilon),
            &parent,
        )
        .unwrap();
        let after = tidal_lock_orientation(
            &circular_state(std::f64::consts::FRAC_PI_2 + epsilon),
            &parent,
        )
        .unwrap();

        assert_axes_close(before, after, 1.0e-3);
    }
}
