//! Bubble lifecycle: player body spawn, SOI rebase, debug drops, EVA surface placement.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/simulation/regimes.md`).

#[allow(unused_imports)]
use super::*;

use bevy::math::{DQuat, DVec3};
use thalos_physics_canonical::canonical::{AuthorityMode, TranslationalState};
use thalos_physics_canonical::surface_local::SurfaceLocalFrame;
use thalos_physics_canonical::types::{AttitudeState, VesselKind};
use thalos_physics_local::avian::{
    AngularVelocity, CenterOfMass, Collider, CustomPositionIntegration, LinearVelocity, LockedAxes,
    NoAutoCenterOfMass, Position, RigidBody, Rotation,
};
use thalos_physics_local::{
    ActiveLocalBubble, HeightSourceRegistry, LocalBubble, LocalBubbleConfig, LocalCraftBody,
    LocalCraftSpawn, LocalPrimitiveCollider, LocalPrimitiveShape, spawn_local_craft_body,
};
use thalos_shipyard::{AttachNodes, Gear, Part, SurfaceMount};
use thalos_world::BodyId;

use crate::debug::DebugMode;
use crate::player_controller::{EvaMode, PlayerControllerBody};
use crate::rendering::{PlayerShip, SimulationState};
use crate::view::ViewMode;

/// Spawn the player's Avian rigid body the first time the simulation is
/// ready to host it. Ships live in the **surface-local frame** (a body-fixed
/// tangent frame anchored under the craft, Y-up, small coordinates — see
/// `docs/simulation/surface_local.md`); gravity plus the rotating-frame terms come from
/// `surface_local_acceleration`, and ground colliders are static in the
/// frame. The EVA capsule still lives in body-centered inertial coordinates
/// until its SLF fold-in.
///
/// Two vessel kinds spawn through this single seam, KSP-style:
/// - `VesselKind::Ship`: waits for `PlayerShip` + ship params, then
///   spawns a compound collider built from the rendered ship parts.
/// - `VesselKind::Eva`: spawns a 1.8 m capsule with rotation locked and
///   walking-friendly friction. The same entity carries
///   `PlayerControllerBody` so the EVA controller's systems find it.
///
/// Avian owns rotation and live thrust for ships; for EVA, rotation is
/// fully locked and the controller drives translation directly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_player_avian_body(
    mut commands: Commands,
    view: Res<ViewMode>,
    mut active: ResMut<ActiveLocalBubble>,
    mut sim: ResMut<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    player_ship: Query<&GlobalTransform, With<PlayerShip>>,
    parts: PartColliderQuery,
    gear_q: Query<
        (&Gear, &SurfaceMount),
        (
            With<Part>,
            Without<crate::shipyard_editor::core::EditorPart>,
        ),
    >,
    host_nodes: Query<&AttachNodes>,
) {
    if active.bubble.is_some() || *view != ViewMode::Ship {
        return;
    }
    let vessel_kind = sim.simulation.vessel_kind();
    let params = *sim.simulation.ship_params();
    if params.moment_of_inertia.length_squared() <= 0.0 {
        return;
    }
    let body_id = sim.simulation.dominant_body();
    let body_state = body_state_for(&sim, body_id);

    // EVA refines its canonical spawn pose to sit just above the
    // rendered terrain at the sub-stellar point (daylight) before the
    // Avian body is created. main.rs only knows the body radius, so it
    // seeds the rough 12 km drop; once the height source exists, we can
    // plant the player at the actual terrain.
    if vessel_kind == VesselKind::Eva {
        let Some(height_source) = height_sources.get(body_id) else {
            return;
        };
        let body = &sim.system.bodies[body_id];
        let sun_dir_inertial = (-body_state.position).normalize_or_zero();
        let mut dir_body_fixed = if sun_dir_inertial == DVec3::ZERO {
            DVec3::Y
        } else {
            (body_state.orientation.inverse() * sun_dir_inertial).normalize()
        };
        // EVA drop-site selection, searching the daylight hemisphere near the
        // sub-stellar point:
        //   default / `plain` → flattest usable plain (the intended on-foot start),
        //   `relief`          → highest-relief hill site (terrain inspection),
        //   `substellar`      → exact sub-stellar point (legacy behaviour).
        let eva_site = std::env::var("THALOS_EVA_SITE").ok();
        if eva_site.as_deref() != Some("substellar") {
            let seek_hills = eva_site.as_deref() == Some("relief");
            let up = if dir_body_fixed.y.abs() > 0.99 {
                DVec3::X
            } else {
                DVec3::Y
            };
            let east = up.cross(dir_body_fixed).normalize();
            let north = dir_body_fixed.cross(east);
            let probe = 3_000.0 / body.radius_m; // ~3 km cross, in radians
            let h = |d: DVec3| {
                height_source
                    .sample_height_m(d.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
                    .unwrap_or(0.0) as f64
            };
            let n = 24i32;
            let mut best_dir = dir_body_fixed;
            let mut best_score = f64::NEG_INFINITY;
            let mut best_relief = 0.0f64;
            for iy in -n..=n {
                for ix in -n..=n {
                    // Offsets within ~50° of the sub-stellar point keep the site lit.
                    let ax = (ix as f64 / n as f64) * 0.9;
                    let ay = (iy as f64 / n as f64) * 0.9;
                    let cand = (dir_body_fixed + east * ax + north * ay).normalize();
                    let relief = (h((cand + east * probe).normalize())
                        - h((cand - east * probe).normalize()))
                    .abs()
                        + (h((cand + north * probe).normalize())
                            - h((cand - north * probe).normalize()))
                        .abs();
                    // Maximise relief for hills, minimise it for a usable plain.
                    let score = if seek_hills { relief } else { -relief };
                    if score > best_score {
                        best_score = score;
                        best_relief = relief;
                        best_dir = cand;
                    }
                }
            }
            dir_body_fixed = best_dir;
            let kind = if seek_hills { "hill" } else { "plain" };
            eprintln!("EVA {kind} site selected (relief proxy {best_relief:.0} m)");
        }
        let terrain_h = height_source
            .sample_height_m(dir_body_fixed.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
            .unwrap_or(0.0) as f64;
        let stand_clearance_m = 1.0;
        let position_body = dir_body_fixed * (body.radius_m + terrain_h + stand_clearance_m);
        let position_inertial = body_state.position + body_state.orientation * position_body;
        let velocity_inertial = body_state.velocity
            + body_state
                .angular_velocity
                .cross(body_state.orientation * position_body);
        let translation = TranslationalState {
            position: position_inertial,
            velocity: velocity_inertial,
        };
        let attitude = AttitudeState {
            orientation: level_attitude_for_body_dir(body_state.orientation, dir_body_fixed),
            angular_velocity: DVec3::ZERO,
        };
        sim.simulation
            .install_local_rigid_body_state(translation, attitude);
    }

    let craft = sim.simulation.craft_state();
    // Anchor the surface-local frame under the craft's surface projection.
    // The height source may not be registered yet (bakes still loading) —
    // a reference-radius anchor is exact regardless, and the re-anchor
    // system refreshes the elevation as the craft moves.
    let height_source = height_sources.get(body_id);
    let slf = SurfaceLocalFrame::new(
        &body_state,
        surface_anchor_under(
            &body_state,
            height_source.as_deref(),
            craft.translation.position,
        ),
    );
    let frame = inertial_to_craft_frame(
        vessel_kind,
        &body_state,
        &slf,
        craft.translation,
        craft.attitude,
    );

    let craft_entity = match vessel_kind {
        VesselKind::Ship => {
            if player_ship.iter().next().is_none() {
                return;
            }
            let collider_primitives = build_ship_collider_primitives(&parts);
            let part_positions = compute_part_collider_positions(&parts);
            let wheels = build_wheel_set(&gear_q, &host_nodes, &part_positions);
            let entity = spawn_local_craft_body(
                &mut commands,
                LocalCraftSpawn {
                    craft_id: craft.id,
                    position_m: frame.position_m,
                    rotation: frame.rotation,
                    linear_velocity_m_s: frame.linear_velocity_m_s,
                    angular_velocity_rad_s: frame.angular_velocity_rad_s,
                    mass_kg: craft.mass.wet_mass_kg,
                    angular_inertia_kg_m2: params.moment_of_inertia,
                    collider_primitives,
                },
            );
            // Pin the rigid body's rotation pivot to the craft's *real* CoM for
            // every ship, not just gear ships. Two systems depend on it:
            //   - aero: the native aero model (`thalos_physics_canonical::aero`)
            //     takes each surface's moment about the CoM. With Avian's auto
            //     CoM (the collider centroid) the static margin — hence pitch/yaw
            //     stability — is accidental, which is the wingless-craft tumble.
            //   - gear: upward wheel forces aft of the nose origin need the pivot
            //     at the CoM they straddle, or they tip the craft over.
            // `NoAutoCenterOfMass` stops the compound collider from overwriting
            // it; Position still tracks the root origin, so snap/readback are
            // unaffected.
            let com = params.center_of_mass.as_vec3();
            commands
                .entity(entity)
                .insert((CenterOfMass(com), NoAutoCenterOfMass));
            if !wheels.is_empty() {
                info!(
                    "landing gear: {} wheel(s) on player ship, CoM = ({:.2}, {:.2}, {:.2}) m",
                    wheels.len(),
                    com.x,
                    com.y,
                    com.z,
                );
                // Gear is the sole ground interface for a wheeled craft: filter
                // the hull compound collider out of solver contact with the
                // ground so it can't fight the raycast suspension (which flung
                // the craft on its gear). The gear raycast is a SpatialQuery,
                // unaffected by these layers; crash detection switches to the
                // weight-on-wheels signal. See `docs/simulation/surface_local.md`.
                commands.entity(entity).insert((
                    WheelSet { wheels },
                    thalos_physics_local::wheeled_craft_collision_layers(),
                ));
            }
            entity
        }
        VesselKind::Eva => {
            // KSP-on-foot is a kinematic capsule whose position is set
            // each frame by `step_eva_controller` from direct terrain
            // heightmap queries — no Avian contact resolution. Spawn a
            // placeholder cuboid so `spawn_local_craft_body` is happy
            // (it falls back to a 1 m cube if the list is empty, which
            // is fine but loud in inspectors); then immediately remove
            // the `Collider` entirely so writeback_solver_bodies has
            // nothing to integrate, which was producing the visible
            // sliding (delta_position from kinematic↔kinematic contacts
            // was being applied on top of our terrain-clamped writes).
            let entity = spawn_local_craft_body(
                &mut commands,
                LocalCraftSpawn {
                    craft_id: craft.id,
                    position_m: frame.position_m,
                    rotation: frame.rotation,
                    linear_velocity_m_s: frame.linear_velocity_m_s,
                    angular_velocity_rad_s: DVec3::ZERO,
                    mass_kg: craft.mass.wet_mass_kg.max(params.dry_mass_kg),
                    angular_inertia_kg_m2: params.moment_of_inertia,
                    collider_primitives: vec![LocalPrimitiveCollider {
                        offset_m: DVec3::ZERO,
                        rotation: DQuat::IDENTITY,
                        shape: LocalPrimitiveShape::Capsule {
                            radius: 0.32,
                            length: 1.8 - 0.64,
                        },
                    }],
                },
            );
            commands.entity(entity).remove::<Collider>().insert((
                RigidBody::Kinematic,
                CustomPositionIntegration,
                LockedAxes::ROTATION_LOCKED,
                PlayerControllerBody,
                Name::new("EVA player vessel"),
            ));
            entity
        }
    };

    let bubble_id = active.allocate_id();
    active.bubble = Some(LocalBubble {
        id: bubble_id,
        body_id,
        craft_entity,
        frame: slf,
        terrain_entity: None,
        center_dir_body: DVec3::Y,
        center_surface_body_m: DVec3::ZERO,
        basis: thalos_body_render::TerrainPatchBasis::from_normal(DVec3::Y),
        patch_half_extent_m: 0.0,
        terrain_built_at_revision: 0,
    });
    info!(
        "spawned player vessel bubble={} body_id={} kind={:?}",
        bubble_id, body_id, vessel_kind,
    );
}

/// Re-project the Avian rigid body onto the new dominant body's
/// body-centered inertial frame when the ship transits an SOI boundary.
/// `apply_local_forces` computes gravity against `bubble.body_id`, so a
/// stale value would pull the ship toward the body it just left.
///
/// Runs every frame but does work only when the dominant body actually
/// changes — cheap in the common case. The transformation is mediated
/// through canonical inertial state so we don't need to compute
/// body-to-body change-of-basis directly. Any attached terrain patch
/// belongs to the old body and is despawned;
/// `attach_terrain_patch_when_close` re-spawns over the new body on a
/// subsequent frame if the ship is close enough.
pub(crate) fn rebase_bubble_to_dominant_body(
    mut commands: Commands,
    mut active: ResMut<ActiveLocalBubble>,
    height_sources: Res<HeightSourceRegistry>,
    sim: Res<SimulationState>,
    mut craft_q: Query<
        (
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
        ),
        With<LocalCraftBody>,
    >,
) {
    let Some(bubble) = active.bubble.as_mut() else {
        return;
    };
    let new_body_id = sim.simulation.dominant_body();
    if new_body_id == bubble.body_id {
        return;
    }
    let Ok((mut position, mut rotation, mut linear_velocity, mut angular_velocity)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    let kind = sim.simulation.vessel_kind();
    let old_body_state = body_state_for(&sim, bubble.body_id);
    let (translation, attitude) = craft_frame_to_inertial(
        kind,
        &old_body_state,
        &bubble.frame,
        position.0,
        rotation.0,
        linear_velocity.0,
        angular_velocity.0,
    );
    let new_body_state = body_state_for(&sim, new_body_id);
    // Fresh surface-local frame anchored under the craft on the *new* body.
    let height_source = height_sources.get(new_body_id);
    let new_frame = SurfaceLocalFrame::new(
        &new_body_state,
        surface_anchor_under(
            &new_body_state,
            height_source.as_deref(),
            translation.position,
        ),
    );
    let frame = inertial_to_craft_frame(kind, &new_body_state, &new_frame, translation, attitude);
    bubble.frame = new_frame;
    position.0 = frame.position_m;
    rotation.0 = frame.rotation;
    linear_velocity.0 = frame.linear_velocity_m_s;
    angular_velocity.0 = frame.angular_velocity_rad_s;

    if let Some(terrain_entity) = bubble.terrain_entity.take() {
        commands.entity(terrain_entity).despawn();
    }
    bubble.center_dir_body = DVec3::Y;
    bubble.center_surface_body_m = DVec3::ZERO;
    bubble.basis = thalos_body_render::TerrainPatchBasis::from_normal(DVec3::Y);
    let old_body_id = bubble.body_id;
    bubble.body_id = new_body_id;
    info!(
        "rebased local bubble across SOI transit: body_id {} -> {}",
        old_body_id, new_body_id
    );
}

pub(crate) fn debug_surface_drop(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    debug: Option<Res<DebugMode>>,
    height_sources: Res<HeightSourceRegistry>,
    config: Res<LocalBubbleConfig>,
    mut active: ResMut<ActiveLocalBubble>,
    mut eva_mode: ResMut<EvaMode>,
    mut sim: ResMut<SimulationState>,
    mut craft_q: Query<
        (
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
        ),
        With<LocalCraftBody>,
    >,
) {
    if !keys.just_pressed(DEBUG_DROP_KEY) || !debug.as_deref().map(|d| d.enabled).unwrap_or(false) {
        return;
    }
    let Some(body_id) = thalos_body_id(&sim) else {
        return;
    };
    let Some(height_source) = height_sources.get(body_id) else {
        warn!("debug surface drop requested before Thalos height source is available");
        return;
    };

    let is_eva = sim.simulation.vessel_kind() == VesselKind::Eva;
    // For ships the bubble is teardown-and-respawn territory: a teleport
    // while it exists would skew the contact graph and Avian's internal
    // state. EVA spawns its bubble at startup and never tears it down,
    // so we teleport in place and let `maintain_terrain_patch` (or our
    // explicit terrain despawn below) rebuild the surface mesh around
    // the new position.
    if !is_eva && let Some(bubble) = active.bubble.take() {
        warn!(
            "debug surface drop requested while local bubble {} is active; keeping current bubble",
            bubble.id
        );
        active.bubble = Some(bubble);
        return;
    }

    let body = sim.system.bodies[body_id].clone();
    let body_state = body_state_for(&sim, body_id);
    // Body-fixed direction toward the star at the current sim time.
    // Pyros sits at the heliocentric origin, so the sun direction in
    // body-centered inertial coordinates is `-body_state.position`;
    // rotating by `orientation.inverse()` puts it in body-fixed coords
    // so the spawn rotates with the planet (always day-side).
    let sun_dir_inertial = (-body_state.position).normalize_or_zero();
    let dir = if sun_dir_inertial == DVec3::ZERO {
        // Star is at the body's centre — degenerate, fall back to a
        // fixed body-fixed heading rather than dividing by zero.
        DVec3::new(0.271, 0.893, -0.361).normalize()
    } else {
        (body_state.orientation.inverse() * sun_dir_inertial).normalize()
    };
    let height = height_source
        .sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(0.0) as f64;
    let position_body = dir * (body.radius_m + height + config.debug_drop_height_m);
    let surface_velocity = body_state.velocity
        + body_state
            .angular_velocity
            .cross(body_state.orientation * position_body);
    // Ships get a small downward kick so they land instead of hover;
    // EVA arrives at rest (the controller's gravity will pull it down).
    let velocity = if is_eva {
        surface_velocity
    } else {
        surface_velocity + body_state.orientation * (-dir * config.debug_drop_speed_m_s)
    };
    let translation = TranslationalState {
        position: body_state.position + body_state.orientation * position_body,
        velocity,
    };
    let attitude = AttitudeState {
        orientation: level_attitude_for_body_dir(body_state.orientation, dir),
        angular_velocity: DVec3::ZERO,
    };

    if is_eva {
        // Grounded EVA owns its Avian capsule directly (the canonical→Avian
        // snap is short-circuited), so the shared helper writes canonical,
        // marks the player grounded, and plants the capsule in one place.
        if let Some(bubble) = active.bubble.as_mut()
            && let Ok((mut position, mut rotation, mut linear_velocity, mut angular_velocity)) =
                craft_q.get_mut(bubble.craft_entity)
        {
            place_eva_on_surface(
                &mut commands,
                &mut sim,
                &mut eva_mode,
                bubble,
                (
                    &mut position,
                    &mut rotation,
                    &mut linear_velocity,
                    &mut angular_velocity,
                ),
                body_id,
                translation,
                attitude,
            );
        }
    } else {
        sim.simulation
            .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
        sim.simulation
            .install_local_rigid_body_state(translation, attitude);
        sim.simulation.warp.reset();
    }
    // A fresh drop hands back a flyable craft — clear any structural failure.
    sim.simulation.repair();

    info!(
        "debug surface drop placed {:?} {:.0} m above rendered {} terrain (day-side)",
        sim.simulation.vessel_kind(),
        config.debug_drop_height_m,
        body.name,
    );
}

/// Plant a grounded EVA player at a surface pose, in place.
///
/// EVA keeps its persistent bubble across teleports (KSP-on-foot never tears
/// the capsule down), so a surface teleport is a rewrite rather than a
/// respawn: set canonical, mark the EVA grounded, move the bubble onto the
/// target body, drop the old terrain patch, and plant the Avian capsule. The
/// grounded canonical→Avian snap is short-circuited, so this is the only
/// thing that moves the capsule; [`crate::player_controller::step_eva_controller`]
/// takes over next frame and glues it to the rendered surface.
///
/// Shared by the F9 sub-stellar drop and the map-cursor surface teleport so
/// both place EVA the same way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn place_eva_on_surface(
    commands: &mut Commands,
    sim: &mut SimulationState,
    eva_mode: &mut EvaMode,
    bubble: &mut LocalBubble,
    avian: (
        &mut Position,
        &mut Rotation,
        &mut LinearVelocity,
        &mut AngularVelocity,
    ),
    body_id: BodyId,
    translation: TranslationalState,
    attitude: AttitudeState,
) {
    let (position, rotation, linear_velocity, angular_velocity) = avian;
    let body_state = body_state_for(sim, body_id);

    sim.simulation
        .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
    sim.simulation
        .install_local_rigid_body_state(translation, attitude);
    sim.simulation.warp.reset();
    *eva_mode = EvaMode::Grounded;

    // Move the bubble onto the target body and drop the old terrain patch so
    // `attach_terrain_patch_when_close` rebuilds it around the new spot. When
    // the body is unchanged (the common Thalos→Thalos case) these are no-ops.
    if let Some(terrain_entity) = bubble.terrain_entity.take() {
        commands.entity(terrain_entity).despawn();
    }
    bubble.body_id = body_id;
    bubble.center_dir_body = DVec3::Y;
    bubble.center_surface_body_m = DVec3::ZERO;
    bubble.basis = thalos_body_render::TerrainPatchBasis::from_normal(DVec3::Y);
    // Keep the (ship-frame) SLF coherent with the new body even though the EVA
    // seam doesn't read it — ship-only systems consult `bubble.frame` by body.
    bubble.frame = SurfaceLocalFrame::new(
        &body_state,
        surface_anchor_under(&body_state, None, translation.position),
    );

    let frame = inertial_to_bubble_frame(&body_state, translation, attitude);
    position.0 = frame.position_m;
    rotation.0 = frame.rotation;
    linear_velocity.0 = frame.linear_velocity_m_s;
    angular_velocity.0 = DVec3::ZERO;
}
