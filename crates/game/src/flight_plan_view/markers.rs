//! Trajectory markers (apoapsis, periapsis, …).
//!
//! Markers are derived from [`thalos_physics::trajectory::TrajectoryEvent`]s in
//! the active flight plan. The system is intentionally generic: adding a new
//! marker kind (ascending/descending node, SOI entry, closest approach) means
//! extending [`MarkerKind`] + the visual table — no plumbing changes.
//!
//! Lifecycle: on prediction-version change every marker is despawned and
//! respawned from the current event list, since [`TrajectoryEvent::id`]s are
//! only stable within a single propagation. Within a prediction, transforms
//! are recomputed every frame so markers track ghost-pinned legs and follow
//! the camera.

use std::collections::HashSet;

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::body_state_provider::BodyStateProvider;
use thalos_physics::trajectory::{EncounterId, FlightPlan, TrajectoryEventKind};
use thalos_physics::types::BodyState;

use crate::camera::{ActiveCamera, CameraFocus};
use crate::coords::{RenderOrigin, WorldScale};
use crate::photo_mode::HideInPhotoMode;
use crate::rendering::{FrameBodyStates, SimulationState};
use crate::view::HideInShipView;

use super::FlightPlanView;

const MARKER_RADIUS: f32 = 0.006;

// ---------------------------------------------------------------------------
// Public marker kind
// ---------------------------------------------------------------------------

/// Kinds of marker the renderer knows how to draw. Extend by adding a variant
/// + an arm in [`MarkerKind::base_color`] and [`MarkerKind::from_event_kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkerKind {
    Apoapsis,
    Periapsis,
    // Future: AscendingNode, DescendingNode, SoiEntry, SoiExit, ClosestApproach.
}

impl MarkerKind {
    fn from_event_kind(kind: TrajectoryEventKind) -> Option<Self> {
        match kind {
            TrajectoryEventKind::Apoapsis => Some(MarkerKind::Apoapsis),
            TrajectoryEventKind::Periapsis => Some(MarkerKind::Periapsis),
            TrajectoryEventKind::SoiEntry
            | TrajectoryEventKind::SoiExit
            | TrajectoryEventKind::SurfaceImpact => None,
        }
    }

    fn base_color(self) -> Color {
        match self {
            MarkerKind::Apoapsis => Color::srgb(0.95, 0.30, 0.30),
            MarkerKind::Periapsis => Color::srgb(0.30, 0.75, 1.00),
        }
    }
}

#[derive(Component, Debug, Clone, Copy)]
pub struct TrajectoryMarker {
    pub event_id: EncounterId,
}

// ---------------------------------------------------------------------------
// Shared assets
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub(super) struct TrajectoryMarkerAssets {
    mesh: Handle<Mesh>,
    apoapsis: KindMaterials,
    periapsis: KindMaterials,
}

struct KindMaterials {
    main: Handle<StandardMaterial>,
    ghost: Handle<StandardMaterial>,
}

impl TrajectoryMarkerAssets {
    fn material(&self, kind: MarkerKind, ghost: bool) -> Handle<StandardMaterial> {
        let kind_mats = match kind {
            MarkerKind::Apoapsis => &self.apoapsis,
            MarkerKind::Periapsis => &self.periapsis,
        };
        if ghost {
            kind_mats.ghost.clone()
        } else {
            kind_mats.main.clone()
        }
    }
}

pub(super) fn setup_trajectory_marker_assets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(Circle::new(1.0));
    let make = |color: Color, materials: &mut Assets<StandardMaterial>| {
        let lin: LinearRgba = color.into();
        materials.add(StandardMaterial {
            base_color: color,
            emissive: lin * 2.0,
            unlit: true,
            cull_mode: None,
            ..default()
        })
    };
    let kind_mats = |kind: MarkerKind, materials: &mut Assets<StandardMaterial>| KindMaterials {
        main: make(kind.base_color(), materials),
        ghost: make(ghost_color(kind.base_color()), materials),
    };
    let apoapsis = kind_mats(MarkerKind::Apoapsis, &mut materials);
    let periapsis = kind_mats(MarkerKind::Periapsis, &mut materials);
    commands.insert_resource(TrajectoryMarkerAssets {
        mesh,
        apoapsis,
        periapsis,
    });
}

/// Mirror of `flight_plan_view::render::ghost_adjust` for marker base colors:
/// pull toward white and drop alpha so future-leg markers read as a soft echo.
fn ghost_color(color: Color) -> Color {
    let srgba = color.to_srgba();
    let mix = 0.3;
    Color::srgba(
        srgba.red + (1.0 - srgba.red) * mix,
        srgba.green + (1.0 - srgba.green) * mix,
        srgba.blue + (1.0 - srgba.blue) * mix,
        srgba.alpha * 0.6,
    )
}

// ---------------------------------------------------------------------------
// Spec construction (events → world-space marker specs)
// ---------------------------------------------------------------------------

struct MarkerSpec {
    kind: MarkerKind,
    event_id: EncounterId,
    world_pos: Vec3,
    is_ghost: bool,
}

fn compute_marker_specs(
    flight_plan: &FlightPlan,
    ephemeris: &dyn BodyStateProvider,
    flight_plan_view: &FlightPlanView,
    body_states: &[BodyState],
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> Vec<MarkerSpec> {
    let mut specs = Vec::new();
    // Multi-revolution legs (fast moon orbit, long heliocentric horizon) emit
    // a Pe + Ap pair per revolution. Keep only the first of each kind per
    // (kind, body, leg) — events arrive in time order so `insert` returns
    // false for duplicates after the earliest is taken.
    let mut seen: HashSet<(MarkerKind, usize, usize)> = HashSet::new();

    for event in flight_plan.events() {
        let Some(kind) = MarkerKind::from_event_kind(event.kind) else {
            continue;
        };
        if !seen.insert((kind, event.body, event.leg_index)) {
            continue;
        }

        // Match the trajectory's leg-anchor render frame so the marker stays
        // glued to the line. Per-leg relock (flight_plan.rs §"Per-leg anchor
        // relock") locks every sample's anchor to the first sample's anchor;
        // the marker must use that same anchor body, not `event.body`, which
        // is the SOI body where the apsis was detected and may differ on a
        // multi-SOI leg.
        let Some(leg) = flight_plan.legs().get(event.leg_index) else {
            continue;
        };
        let Some(leg_anchor_id) = leg
            .coast_segment
            .samples
            .first()
            .or_else(|| leg.burn_segment.as_ref().and_then(|s| s.samples.first()))
            .map(|s| s.anchor_body)
        else {
            continue;
        };

        let leg_anchor_pos = ephemeris.query_body(leg_anchor_id, event.epoch).position;
        let pin = flight_plan_view.pin_for_body(leg_anchor_id, body_states);
        let world_pos = render_pos(event.craft_state.position, leg_anchor_pos, pin, origin, scale);

        specs.push(MarkerSpec {
            kind,
            event_id: event.id,
            world_pos,
            is_ghost: event.leg_index > 0,
        });
    }

    specs
}

#[inline]
fn billboard_transform(world_pos: Vec3, camera_rotation: Quat, marker_scale: f32) -> Transform {
    Transform {
        translation: world_pos,
        rotation: camera_rotation,
        scale: Vec3::splat(marker_scale),
    }
}

#[inline]
fn render_pos(
    craft_pos: DVec3,
    leg_anchor_pos: DVec3,
    pin: DVec3,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> Vec3 {
    let world = (craft_pos - leg_anchor_pos) + pin - origin.position;
    (world * scale.0).as_vec3()
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

pub(super) fn manage_trajectory_markers(
    mut commands: Commands,
    assets: Option<Res<TrajectoryMarkerAssets>>,
    sim: Option<Res<SimulationState>>,
    body_states: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    scale: Res<WorldScale>,
    flight_plan_view: Res<FlightPlanView>,
    focus: Res<CameraFocus>,
    camera_q: Query<&Transform, (With<ActiveCamera>, With<crate::camera::OrbitCamera>, Without<TrajectoryMarker>)>,
    mut markers: Query<(Entity, &TrajectoryMarker, &mut Transform, &mut Visibility)>,
    mut last_version: Local<Option<u64>>,
) {
    let Some(assets) = assets.as_deref() else { return };
    let Some(sim) = sim.as_deref() else { return };
    let Some(flight_plan) = sim.simulation.prediction() else {
        return;
    };
    let Some(states) = body_states.states.as_deref() else {
        return;
    };

    let cam_dist = (focus.distance * scale.0) as f32;
    let cam_rot = camera_q
        .single()
        .map(|t| t.rotation)
        .unwrap_or(Quat::IDENTITY);

    let version = sim.simulation.prediction_version();
    let specs = compute_marker_specs(
        flight_plan,
        sim.ephemeris.as_ref(),
        &flight_plan_view,
        states,
        &origin,
        &scale,
    );

    let version_changed = *last_version != Some(version);
    if version_changed {
        // Event ids reset across predictions, so reusing entities by id would
        // alias markers across unrelated apsides. Drop everything and respawn.
        for (entity, _, _, _) in &markers {
            commands.entity(entity).despawn();
        }
        for spec in &specs {
            spawn_marker(&mut commands, assets, spec, cam_rot, cam_dist);
        }
        *last_version = Some(version);
        return;
    }

    // Same prediction: update transforms in-place, keyed by event_id.
    let mut by_id: std::collections::HashMap<EncounterId, &MarkerSpec> = specs
        .iter()
        .map(|s| (s.event_id, s))
        .collect();
    for (_, marker, mut tf, mut vis) in &mut markers {
        if let Some(spec) = by_id.remove(&marker.event_id) {
            *vis = Visibility::Inherited;
            *tf = billboard_transform(spec.world_pos, cam_rot, cam_dist * MARKER_RADIUS);
        }
    }
}

fn spawn_marker(
    commands: &mut Commands,
    assets: &TrajectoryMarkerAssets,
    spec: &MarkerSpec,
    cam_rot: Quat,
    cam_dist: f32,
) {
    commands.spawn((
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(assets.material(spec.kind, spec.is_ghost)),
        billboard_transform(spec.world_pos, cam_rot, cam_dist * MARKER_RADIUS),
        TrajectoryMarker {
            event_id: spec.event_id,
        },
        HideInPhotoMode,
        HideInShipView,
    ));
}
