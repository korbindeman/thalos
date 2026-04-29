//! Ghost body ECS entities: spawn, update, lifecycle, and camera handoff.
//!
//! Ghost bodies are translucent spheres at future encounter positions. They
//! are managed via diff against [`Ghost`]s in [`FlightPlanView`] — matched
//! by `(body_id, ~epoch)` so entities persist across repredictions without
//! despawn/respawn churn.

use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::camera::CameraFocus;
use crate::coords::{RenderOrigin, WorldScale};
use crate::photo_mode::HideInPhotoMode;
use crate::rendering::{CelestialBody, FrameBodyStates, SimulationState};
use crate::view::HideInShipView;

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
    /// The body this ghost represents.
    pub body_id: usize,
    /// SOI parent body — the live pin we anchor the ghost to.
    pub parent_id: usize,
    /// Body's offset from its parent at the projection epoch.
    pub relative_position: DVec3,
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
    scale: Res<WorldScale>,
    mut view: ResMut<FlightPlanView>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut existing: Query<(Entity, &mut GhostBody)>,
) {
    let Some(sim) = sim else { return };

    for ghost in view.ghosts_mut().iter_mut() {
        if ghost.phase == GhostPhase::Retired {
            // Retired ghosts with entities get despawned below.
            continue;
        }

        if let Some(entity) = ghost.entity {
            // Entity already exists — update in place.
            if let Ok((_, mut comp)) = existing.get_mut(entity) {
                comp.parent_id = ghost.parent_id;
                comp.relative_position = ghost.relative_position;
                comp.encounter_epoch = ghost.encounter_epoch;
                comp.phase = ghost.phase;
            }
        } else {
            // Need to spawn a new ghost entity.
            let Some(body_def) = sim.system.bodies.get(ghost.body_id) else {
                continue;
            };
            let render_radius = (body_def.radius_m * scale.0) as f32;
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
                        body_id: ghost.body_id,
                        parent_id: ghost.parent_id,
                        relative_position: ghost.relative_position,
                        radius_m: body_def.radius_m,
                        encounter_epoch: ghost.encounter_epoch,
                        phase: GhostPhase::Active,
                    },
                    HideInPhotoMode,
                    HideInShipView,
                    Name::new(format!("Ghost: {}", body_def.name)),
                ))
                .id();
            ghost.entity = Some(entity);
        }
    }

    // Despawn retired ghosts.
    let retired_entities: Vec<Entity> = view
        .ghosts()
        .iter()
        .filter(|g| g.phase == GhostPhase::Retired && g.entity.is_some())
        .filter_map(|g| g.entity)
        .collect();
    for entity in &retired_entities {
        commands.entity(*entity).despawn();
    }

    // Remove retired ghosts from the list.
    view.ghosts_mut().retain(|g| g.phase != GhostPhase::Retired);
}

// ---------------------------------------------------------------------------
// Lifecycle: blend + handoff
// ---------------------------------------------------------------------------

pub(super) fn update_ghost_lifecycle(
    sim: Option<Res<SimulationState>>,
    origin: Res<RenderOrigin>,
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

    for ghost in view.ghosts_mut().iter_mut() {
        let Some(entity) = ghost.entity else { continue };

        let time_to_encounter = ghost.encounter_epoch - sim_time;

        // Transition Active → Blending when sim time approaches encounter.
        if time_to_encounter <= blend_lead_time && time_to_encounter > 0.0 {
            let progress =
                ((blend_lead_time - time_to_encounter) / blend_lead_time).clamp(0.0, 1.0) as f32;
            ghost.phase = GhostPhase::Blending { progress };
        } else if time_to_encounter <= 0.0 {
            ghost.phase = GhostPhase::Retired;
        }

        // Update material alpha during blend.
        if let GhostPhase::Blending { progress } = ghost.phase
            && let Ok((_, mat_handle, _)) = ghosts.get_mut(entity)
            && let Some(mat) = materials.get_mut(&mat_handle.0)
        {
            let alpha = 0.35 * (1.0 - progress);
            mat.base_color = mat.base_color.with_alpha(alpha);
        }

        // Camera handoff: if focused on this ghost and it's retired, transfer
        // to the real body with a smooth physics-space origin lerp.
        if ghost.phase == GhostPhase::Retired
            && focus.target == Some(entity)
            && let Some((real_entity, _, _)) =
                celestials.iter().find(|(_, cb, _)| cb.body_id == ghost.body_id)
        {
            focus.focus_on(real_entity, origin.position);
        }
    }
}

// ---------------------------------------------------------------------------
// Transform update
// ---------------------------------------------------------------------------

pub(super) fn update_ghost_transforms(
    origin: Res<RenderOrigin>,
    focus: Res<CameraFocus>,
    scale: Res<WorldScale>,
    cache: Res<FrameBodyStates>,
    view: Res<FlightPlanView>,
    mut query: Query<(&GhostBody, &mut Transform)>,
) {
    let cam_dist_render = (focus.distance * scale.0) as f32;
    let min_radius = cam_dist_render * 0.008;

    let body_states = match &cache.states {
        Some(s) => s.as_slice(),
        None => return,
    };

    for (ghost_comp, mut transform) in &mut query {
        // Anchor the ghost to its parent's pin recursively. The parent
        // recursion grounds out at the focus body (rule 1 in
        // pin_for_body), so a Mira ghost composes its offset onto
        // current Thalos when the camera focus is Thalos.
        //
        // We pass `encounter_epoch` as the sample time so multi-encounter
        // dispatch resolves the parent ghost active at this body's
        // encounter moment, matching the trajectory pin's recursion
        // entry point exactly.
        let parent_pin =
            view.pin_for_body(ghost_comp.parent_id, ghost_comp.encounter_epoch, body_states);
        let sim_pos = parent_pin + ghost_comp.relative_position;

        transform.translation = ((sim_pos - origin.position) * scale.0).as_vec3();
        let body_radius_render = (ghost_comp.radius_m * scale.0) as f32;
        let radius = body_radius_render.max(min_radius);
        transform.scale = Vec3::splat(radius);
    }
}
