//! Ghost body ECS entities: spawn, update, lifecycle, and camera handoff.
//!
//! Ghost bodies are translucent spheres at future encounter positions. They
//! are managed via diff against [`GhostSpec`]s in [`FlightPlanView`] — matched
//! by `(body_id, ~epoch)` so entities persist across repredictions without
//! despawn/respawn churn.

use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::camera::CameraFocus;
use crate::coords::{RENDER_SCALE, RenderOrigin};
use crate::rendering::{CelestialBody, FrameBodyStates, SimulationState};

use super::view::{FlightPlanView, GhostPhase};

/// Minimum blend duration in sim-seconds. Scaled up by warp speed so the
/// blend remains visible even at high warp.
const BLEND_LEAD_BASE: f64 = 60.0;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/// Translucent duplicate of a celestial body at a future encounter position.
#[derive(Component, Debug, Clone)]
pub struct GhostBody {
    /// Body position relative to parent at projection epoch.
    pub relative_position: DVec3,
    /// Leg-anchor body whose frame this ghost renders in.
    pub leg_anchor_id: usize,
    /// Body's absolute position at projection epoch (`r_body(t_enc)`).
    pub body_position: DVec3,
    /// Leg anchor's position at projection epoch (`r_anchor(t_enc)`).
    pub leg_anchor_pos: DVec3,
    pub render_radius: f32,
    pub radius_m: f64,
    /// SOI entry epoch — drives handoff timing.
    pub encounter_epoch: f64,
    pub phase: GhostPhase,
}

// ---------------------------------------------------------------------------
// Sync: diff specs vs entities
// ---------------------------------------------------------------------------

pub(super) fn sync_ghost_bodies(
    mut commands: Commands,
    sim: Option<Res<SimulationState>>,
    mut view: ResMut<FlightPlanView>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut existing: Query<(Entity, &mut GhostBody)>,
) {
    let Some(sim) = sim else { return };

    for spec in view.ghost_specs.iter_mut() {
        if spec.phase == GhostPhase::Retired {
            // Retired specs with entities get despawned below.
            continue;
        }

        if let Some(entity) = spec.entity {
            // Entity already exists — update in place.
            if let Ok((_, mut ghost)) = existing.get_mut(entity) {
                ghost.relative_position = spec.relative_position;
                ghost.body_position = spec.body_position;
                ghost.leg_anchor_id = spec.leg_anchor_id;
                ghost.leg_anchor_pos = spec.leg_anchor_pos;
                ghost.encounter_epoch = spec.encounter_epoch;
                ghost.phase = spec.phase;
            }
        } else {
            // Need to spawn a new ghost entity.
            let Some(body_def) = sim.system.bodies.get(spec.body_id) else {
                continue;
            };
            let render_radius = (body_def.radius_m * RENDER_SCALE) as f32;
            let [r, g, b] = body_def.color;

            let mesh = meshes.add(Sphere::new(1.0).mesh().ico(3).unwrap());
            let material = materials.add(StandardMaterial {
                base_color: Color::srgba(r, g, b, 0.35),
                emissive: LinearRgba::new(r, g, b, 1.0) * 1.2,
                unlit: true,
                double_sided: true,
                alpha_mode: AlphaMode::Blend,
                ..default()
            });

            let entity = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(material),
                    Transform::from_translation(Vec3::ZERO)
                        .with_scale(Vec3::splat(render_radius)),
                    NotShadowCaster,
                    NotShadowReceiver,
                    GhostBody {
                        relative_position: spec.relative_position,
                        leg_anchor_id: spec.leg_anchor_id,
                        body_position: spec.body_position,
                        leg_anchor_pos: spec.leg_anchor_pos,
                        render_radius,
                        radius_m: body_def.radius_m,
                        encounter_epoch: spec.encounter_epoch,
                        phase: GhostPhase::Active,
                    },
                    Name::new(format!("Ghost: {}", body_def.name)),
                ))
                .id();
            spec.entity = Some(entity);
        }
    }

    // Despawn retired ghosts.
    let retired_entities: Vec<Entity> = view
        .ghost_specs
        .iter()
        .filter(|s| s.phase == GhostPhase::Retired && s.entity.is_some())
        .filter_map(|s| s.entity)
        .collect();
    for entity in &retired_entities {
        commands.entity(*entity).despawn();
    }

    // Remove retired specs from the list.
    view.ghost_specs
        .retain(|s| s.phase != GhostPhase::Retired);
}

// ---------------------------------------------------------------------------
// Lifecycle: blend + handoff
// ---------------------------------------------------------------------------

pub(super) fn update_ghost_lifecycle(
    sim: Option<Res<SimulationState>>,
    mut view: ResMut<FlightPlanView>,
    mut focus: ResMut<CameraFocus>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    celestials: Query<(Entity, &CelestialBody, &Transform)>,
    mut ghosts: Query<(&GhostBody, &MeshMaterial3d<StandardMaterial>, &Transform)>,
) {
    let Some(sim) = sim else { return };
    let sim_time = sim.simulation.sim_time();
    let warp_speed = sim.simulation.warp.speed().max(1.0);

    let blend_lead_time = BLEND_LEAD_BASE.max(warp_speed * 2.0);

    for spec in view.ghost_specs.iter_mut() {
        let Some(entity) = spec.entity else { continue };

        let time_to_encounter = spec.encounter_epoch - sim_time;

        // Transition Active → Blending when sim time approaches encounter.
        if time_to_encounter <= blend_lead_time && time_to_encounter > 0.0 {
            let progress =
                ((blend_lead_time - time_to_encounter) / blend_lead_time).clamp(0.0, 1.0) as f32;
            spec.phase = GhostPhase::Blending { progress };
        } else if time_to_encounter <= 0.0 {
            spec.phase = GhostPhase::Retired;
        }

        // Update material alpha during blend.
        if let GhostPhase::Blending { progress } = spec.phase
            && let Ok((_, mat_handle, _)) = ghosts.get_mut(entity)
            && let Some(mat) = materials.get_mut(&mat_handle.0)
        {
            let alpha = 0.35 * (1.0 - progress);
            mat.base_color = mat.base_color.with_alpha(alpha);
        }

        // Camera handoff: if focused on this ghost and it's retired, transfer
        // to the real body with smooth offset.
        if spec.phase == GhostPhase::Retired
            && focus.target == Some(entity)
            && let Some((real_entity, _, real_tf)) =
                celestials.iter().find(|(_, cb, _)| cb.body_id == spec.body_id)
        {
            let ghost_pos = ghosts
                .get(entity)
                .map(|(_, _, tf)| tf.translation)
                .unwrap_or(Vec3::ZERO);
            let old_pos = ghost_pos + focus.focus_offset;
            let new_pos = real_tf.translation;
            focus.focus_offset = old_pos - new_pos;
            focus.target = Some(real_entity);
        }
    }
}

// ---------------------------------------------------------------------------
// Transform update
// ---------------------------------------------------------------------------

pub(super) fn update_ghost_transforms(
    origin: Res<RenderOrigin>,
    focus: Res<CameraFocus>,
    cache: Res<FrameBodyStates>,
    view: Res<super::view::FlightPlanView>,
    mut query: Query<(&GhostBody, &mut Transform)>,
) {
    let cam_dist_render = (focus.distance * RENDER_SCALE) as f32;
    let min_radius = cam_dist_render * 0.008;

    let body_states = match &cache.states {
        Some(s) => s.as_slice(),
        None => return,
    };

    for (ghost, mut transform) in &mut query {
        // Render the ghost using the same rule as trajectory samples in the
        // leg it anchors. If the leg anchor itself has a ghost (e.g. the
        // encounter body IS the leg anchor), `pin_for_body` returns that
        // ghost's fixed world position so the mesh sits exactly where the
        // trajectory curves.
        let pin = view.pin_for_body(ghost.leg_anchor_id, body_states);
        let sim_pos = ghost.body_position - ghost.leg_anchor_pos + pin;

        transform.translation = ((sim_pos - origin.position) * RENDER_SCALE).as_vec3();
        let radius = ghost.render_radius.max(min_radius);
        transform.scale = Vec3::splat(radius);
    }
}
