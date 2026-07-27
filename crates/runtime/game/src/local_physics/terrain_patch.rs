//! Terrain collider patch lifecycle: attach/detach band, streaming rebuilds, SLF pose.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/simulation/regimes.md`).

#[allow(unused_imports)]
use super::*;

use bevy::math::DVec3;
use thalos_physics_canonical::canonical::AuthorityMode;
use thalos_physics_local::avian::{
    AngularVelocity, ContactGraph, LinearVelocity, Position, Rotation,
};
use thalos_physics_local::{
    ActiveLocalBubble, HeightSourceRegistry, LocalBubble, LocalBubbleConfig, LocalCraftBody,
    TerrainColliderPatch, craft_contacts_terrain, spawn_terrain_collider_patch,
};

use crate::player_controller::{PlayerControllerBody, PlayerControllerState};
use crate::rendering::SimulationState;

/// Attach a terrain collider patch when the ship enters the AGL handoff
/// band over a body whose surface is registered. The collider is
/// [`RigidBody::Kinematic`] centered on the patch's surface point. Its
/// local vertices are body-fixed offsets from that center, so
/// `Position + Rotation * local_vertex` lands at the correct
/// body-centered-inertial position while the narrow phase solves against
/// small local coordinates. [`sync_terrain_collider_pose`] re-poses it
/// each frame as the body rotates.
pub(crate) fn attach_terrain_patch_when_close(
    mut commands: Commands,
    height_sources: Res<HeightSourceRegistry>,
    config: Res<LocalBubbleConfig>,
    runway: Option<Res<crate::runway::RunwaySite>>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
    craft_q: Query<&crate::regime::CraftRegimeState, With<LocalCraftBody>>,
) {
    let Some(bubble) = active.bubble.as_mut() else {
        return;
    };
    if bubble.terrain_entity.is_some() {
        return;
    }
    // In a runway scenario the purpose-built flat `RunwayCollider` already backs
    // the surface under the craft. Attaching the generic terrain patch on top
    // would put a *second* flat kinematic collider at the same elevation, so the
    // craft's compound collider resolves contacts against both at once — a
    // double penetration-recovery push that launches it off its gear. Skip the
    // patch on that body and let the runway collider be the sole ground.
    if runway
        .as_deref()
        .is_some_and(|r| r.body_id == bubble.body_id)
    {
        return;
    }
    // Regime gate (A3 port, `docs/simulation/regimes.md`): the record's
    // `terrain_collider_allowed` folds the 1×-warp-lock requirement with the
    // craft-has-a-collider capability. The capability term subsumes the old
    // per-`VesselKind` EVA skip — the EVA capsule's `Collider` is removed at
    // spawn, so a patch would collide with nothing while its per-frame
    // streaming rebuilds cost ~11% of surface frame time (the old EVA
    // "unplayable stutter").
    if !craft_q
        .get(bubble.craft_entity)
        .is_ok_and(|state| state.regime.terrain_collider_allowed)
    {
        return;
    }
    let Some(body_id) = thalos_body_id(&sim) else {
        return;
    };
    if bubble.body_id != body_id {
        return;
    }
    let Some(height_source) = height_sources.get(body_id) else {
        return;
    };
    let body = &sim.system.bodies[body_id];
    let body_state = body_state_for(&sim, body_id);
    let craft = sim.simulation.craft_state();
    let Some((agl_m, center_dir, _)) = agl_above_rendered_surface(
        body,
        &body_state,
        height_source.as_ref(),
        craft.translation.position,
    ) else {
        return;
    };
    if agl_m > config.handoff_agl_m {
        return;
    }
    let built_revision = height_source.revision();
    let slf = bubble.frame;
    let patch = spawn_terrain_collider_patch(
        &mut commands,
        body_id,
        height_source.as_ref(),
        body.radius_m,
        center_dir,
        &config,
        &slf,
    );
    bubble.terrain_entity = Some(patch.entity);
    bubble.center_dir_body = center_dir;
    bubble.center_surface_body_m = patch.center_surface_body_m;
    bubble.basis = patch.basis;
    bubble.patch_half_extent_m = patch.half_extent_m;
    bubble.terrain_built_at_revision = built_revision;
    info!(
        target: "thalos::diagnostic::local_physics",
        event = "terrain_patch_attached",
        body = %body.name,
        agl_m,
        height_source_revision = built_revision,
        "terrain collider patch attached"
    );
}

/// Despawn the terrain collider patch when the ship climbs back above the
/// handoff band (with hysteresis so we don't churn on the boundary).
pub(crate) fn detach_terrain_patch_when_far(
    mut commands: Commands,
    height_sources: Res<HeightSourceRegistry>,
    config: Res<LocalBubbleConfig>,
    contact_graph: Res<ContactGraph>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
    craft_q: Query<&crate::regime::CraftRegimeState, With<LocalCraftBody>>,
) {
    let Some(bubble) = active.bubble.as_mut() else {
        return;
    };
    let Some(terrain_entity) = bubble.terrain_entity else {
        return;
    };
    if matches!(sim.simulation.authority(), AuthorityMode::BodyFixed { .. }) {
        clear_terrain_patch(&mut commands, bubble);
        info!(
            target: "thalos::diagnostic::local_physics",
            event = "terrain_patch_detached",
            reason = "body_fixed",
            "terrain collider patch detached"
        );
        return;
    }
    if !craft_q
        .get(bubble.craft_entity)
        .is_ok_and(|state| state.regime.terrain_collider_allowed)
        && !craft_contacts_terrain(&contact_graph, bubble.craft_entity, terrain_entity)
    {
        clear_terrain_patch(&mut commands, bubble);
        info!(
            target: "thalos::diagnostic::local_physics",
            event = "terrain_patch_detached",
            reason = "outside_warp_lock",
            "terrain collider patch detached"
        );
        return;
    }
    let Some(height_source) = height_sources.get(bubble.body_id) else {
        return;
    };
    let body = &sim.system.bodies[bubble.body_id];
    let body_state = body_state_for(&sim, bubble.body_id);
    let craft = sim.simulation.craft_state();
    let Some((agl_m, _, _)) = agl_above_rendered_surface(
        body,
        &body_state,
        height_source.as_ref(),
        craft.translation.position,
    ) else {
        return;
    };
    // Hysteresis: detach at 1.5× the attach threshold.
    if agl_m <= config.handoff_agl_m * 1.5 {
        return;
    }
    clear_terrain_patch(&mut commands, bubble);
    info!(
        target: "thalos::diagnostic::local_physics",
        event = "terrain_patch_detached",
        reason = "altitude",
        body = %body.name,
        agl_m,
        "terrain collider patch detached"
    );
}

pub(crate) fn clear_terrain_patch(commands: &mut Commands, bubble: &mut LocalBubble) {
    if let Some(terrain_entity) = bubble.terrain_entity.take() {
        commands.entity(terrain_entity).despawn();
    }
    bubble.center_dir_body = DVec3::Y;
    bubble.center_surface_body_m = DVec3::ZERO;
    bubble.basis = thalos_body_render::TerrainPatchBasis::from_normal(DVec3::Y);
    bubble.patch_half_extent_m = 0.0;
    bubble.terrain_built_at_revision = 0;
}

pub(crate) fn maintain_terrain_patch(
    mut commands: Commands,
    height_sources: Res<HeightSourceRegistry>,
    config: Res<LocalBubbleConfig>,
    mut active: ResMut<ActiveLocalBubble>,
    sim: Res<SimulationState>,
    craft_q: Query<&Position, With<LocalCraftBody>>,
    regime_q: Query<&crate::regime::CraftRegimeState, With<LocalCraftBody>>,
    player: Option<Res<PlayerControllerState>>,
    player_q: Query<&Position, With<PlayerControllerBody>>,
) {
    let Some(current) = active.bubble.clone() else {
        return;
    };
    if current.terrain_entity.is_none() {
        return;
    }
    if !regime_q
        .get(current.craft_entity)
        .is_ok_and(|state| state.regime.terrain_collider_allowed)
    {
        return;
    }
    let Some(height_source) = height_sources.get(current.body_id) else {
        return;
    };
    let player_position = player
        .as_deref()
        .and_then(|state| state.is_active().then_some(()))
        .and_then(|_| player_q.iter().next());
    let position = if let Some(position) = player_position {
        position
    } else {
        let Ok(position) = craft_q.get(current.craft_entity) else {
            return;
        };
        position
    };
    // Terrain patches only exist for ships, whose Avian body is in the
    // surface-local frame — recover the body-fixed body-centered position the
    // patch metadata (`center_surface_body_m`, `center_dir_body`) is expressed
    // in. (EVA never attaches a patch, so it never reaches here.)
    let craft_body_fixed = current.frame.rotation_body_to_frame.inverse()
        * current.frame.body_center_offset(position.0);
    let delta = craft_body_fixed - current.center_surface_body_m;
    let along = delta.dot(current.center_dir_body);
    let lateral = (delta - along * current.center_dir_body).length();
    let current_revision = height_source.revision();
    // Re-center before the craft drifts off the patch edge. The tile-based
    // collider window (docs/simulation/surface.md §3.6) is only tens of metres, so cap the
    // global drift distance by a fraction of the patch's own half-extent; the
    // coarse tangent-grid fallback (km-scale half-extent) keeps the global
    // distance. `patch_half_extent_m` is 0 only with no patch attached, which
    // the early return above already excludes.
    let rebuild_distance_m = if current.patch_half_extent_m > 0.0 {
        config
            .patch_rebuild_distance_m
            .min(0.45 * current.patch_half_extent_m)
    } else {
        config.patch_rebuild_distance_m
    };
    let lateral_stale = lateral > rebuild_distance_m;
    let source_stale = current_revision != current.terrain_built_at_revision;
    if !lateral_stale && !source_stale {
        return;
    }
    let body = &sim.system.bodies[current.body_id];
    let center_dir = craft_body_fixed.normalize_or_zero();
    if center_dir == DVec3::ZERO {
        return;
    }
    if let Some(terrain_entity) = current.terrain_entity {
        commands.entity(terrain_entity).despawn();
    }
    let patch = spawn_terrain_collider_patch(
        &mut commands,
        current.body_id,
        height_source.as_ref(),
        body.radius_m,
        center_dir,
        &config,
        &current.frame,
    );
    active.bubble = Some(LocalBubble {
        terrain_entity: Some(patch.entity),
        center_dir_body: center_dir,
        center_surface_body_m: patch.center_surface_body_m,
        basis: patch.basis,
        patch_half_extent_m: patch.half_extent_m,
        terrain_built_at_revision: current_revision,
        ..current
    });
}

/// Hold the kinematic terrain collider **static in the surface-local frame**:
/// its mesh vertices are body-fixed offsets from `center_surface_body_m`, so
/// with `Position` = the patch centre in SLF coordinates and `Rotation` = the
/// constant body-fixed→SLF rotation, every contact point sits exactly where
/// the rotating surface is — with zero velocity, genuinely static geometry.
/// The pose is constant between re-anchors (this is a cheap idempotent write
/// that guarantees consistency after [`reanchor_surface_frame`] swaps the
/// frame), so the contact solver sees a floor that never moves.
pub(crate) fn sync_terrain_collider_pose(
    active: Res<ActiveLocalBubble>,
    mut terrain_q: Query<
        (
            &mut Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
        ),
        (With<TerrainColliderPatch>, Without<LocalCraftBody>),
    >,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Some(terrain_entity) = bubble.terrain_entity else {
        return;
    };
    let Ok((mut position, mut rotation, mut linear_velocity, mut angular_velocity)) =
        terrain_q.get_mut(terrain_entity)
    else {
        return;
    };
    // The heightfield is authored in the patch-tangent frame (height along its
    // local +Y = the patch up-normal), so its SLF rotation composes the
    // body-fixed→SLF rotation with the patch-basis rotation.
    position.0 = bubble.frame.rotation_body_to_frame
        * (bubble.center_surface_body_m - bubble.frame.anchor_point_body_m);
    rotation.0 = bubble.frame.rotation_body_to_frame
        * thalos_physics_local::patch_basis_rotation(&bubble.basis);
    linear_velocity.0 = DVec3::ZERO;
    angular_velocity.0 = DVec3::ZERO;
}
