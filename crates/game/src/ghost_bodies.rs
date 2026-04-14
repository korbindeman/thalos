//! Ghost bodies: translucent duplicates of celestial bodies projected to
//! their future positions along the player's [`FlightPlan`].
//!
//! Two sources feed ghost projection:
//!
//! 1. **Target closest approach** — for the currently-selected [`TargetBody`],
//!    spawn one ghost of the target at the flight plan's minimum-distance
//!    epoch. This is the marker the player uses to aim a transfer.
//! 2. **Sphere-of-influence entries** — every detected `SoiEntry` encounter
//!    gets a ghost of the entered body at the entry epoch. Cheap situational
//!    awareness for where SOI boundaries will be crossed.
//!
//! Ghosts are rebuilt whenever the prediction generation or target changes,
//! and their render-space transforms are refreshed every frame against the
//! camera's floating origin.

use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use thalos_physics::trajectory::EncounterKind;
use thalos_physics::types::BodyKind;

use crate::coords::{RENDER_SCALE, RenderOrigin};
use crate::rendering::SimulationState;
use crate::target::TargetBody;

/// Sim-space position in metres (f64) + body id. The transform system
/// converts to render-space each frame.
#[derive(Component, Debug, Clone, Copy)]
pub struct GhostBody {
    #[allow(dead_code)]
    pub body_id: usize,
    pub sim_position: bevy::math::DVec3,
}

/// Cache of the state this crate used last rebuild. Rebuild fires when any
/// of these changes.
#[derive(Resource, Default)]
struct GhostCache {
    last_epoch: u64,
    last_target: Option<usize>,
    /// Trip the initial rebuild on first run regardless of epoch.
    built_once: bool,
}

pub struct GhostBodiesPlugin;

impl Plugin for GhostBodiesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GhostCache>().add_systems(
            Update,
            (
                rebuild_ghosts,
                update_ghost_transforms.after(rebuild_ghosts),
            )
                .in_set(crate::SimStage::Sync),
        );
    }
}

fn rebuild_ghosts(
    mut commands: Commands,
    sim: Option<Res<SimulationState>>,
    target: Res<TargetBody>,
    mut cache: ResMut<GhostCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing: Query<Entity, With<GhostBody>>,
) {
    let Some(sim) = sim else { return };
    let Some(plan) = sim.simulation.prediction() else {
        return;
    };

    let epoch = sim.simulation.prediction_epoch();
    let target_changed = cache.last_target != target.target;
    let epoch_changed = cache.last_epoch != epoch;
    if cache.built_once && !epoch_changed && !target_changed {
        return;
    }
    cache.last_epoch = epoch;
    cache.last_target = target.target;
    cache.built_once = true;

    for entity in &existing {
        commands.entity(entity).despawn();
    }

    // Collected (body_id, epoch, kind) tuples to spawn.
    let mut jobs: Vec<(usize, f64)> = Vec::new();

    // Ghost for every SOI entry the plan crosses.
    for enc in plan.encounters() {
        if enc.kind == EncounterKind::SoiEntry {
            jobs.push((enc.body, enc.epoch));
        }
    }

    // Closest-approach ghost for the player-selected target.
    if let Some(target_id) = target.target {
        let ephemeris = sim.ephemeris.as_ref();
        if let Some(enc) = plan.closest_approach_to(target_id, ephemeris) {
            push_unique(&mut jobs, target_id, enc.epoch);
        }
    }

    if jobs.is_empty() {
        return;
    }

    // Spawn ghosts.
    let ephemeris = sim.ephemeris.as_ref();
    for (body_id, epoch) in jobs {
        let Some(body_def) = sim.system.bodies.get(body_id) else {
            continue;
        };
        if body_def.kind == BodyKind::Star {
            continue;
        }
        let body_state = ephemeris.query_body(body_id, epoch);
        spawn_ghost(
            &mut commands,
            &mut meshes,
            &mut materials,
            body_def,
            body_state.position,
        );
    }
}

fn push_unique(jobs: &mut Vec<(usize, f64)>, body: usize, epoch: f64) {
    for existing in jobs.iter() {
        if existing.0 == body && (existing.1 - epoch).abs() < 1.0 {
            return;
        }
    }
    jobs.push((body, epoch));
}

fn spawn_ghost(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    body: &thalos_physics::types::BodyDefinition,
    sim_position: bevy::math::DVec3,
) {
    let render_radius = (body.radius_m * RENDER_SCALE) as f32;
    let render_radius = render_radius.max(0.01);

    let [r, g, b] = body.color;
    let mesh = meshes.add(Sphere::new(render_radius).mesh().ico(3).unwrap());
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba(r, g, b, 0.35),
        emissive: LinearRgba::new(r, g, b, 1.0) * 1.2,
        unlit: true,
        double_sided: true,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_translation(Vec3::ZERO),
        NotShadowCaster,
        NotShadowReceiver,
        GhostBody {
            body_id: body.id,
            sim_position,
        },
        Name::new(format!("Ghost: {}", body.name)),
    ));
}

fn update_ghost_transforms(
    origin: Res<RenderOrigin>,
    mut query: Query<(&GhostBody, &mut Transform)>,
) {
    for (ghost, mut transform) in &mut query {
        transform.translation = ((ghost.sim_position - origin.position) * RENDER_SCALE).as_vec3();
    }
}
