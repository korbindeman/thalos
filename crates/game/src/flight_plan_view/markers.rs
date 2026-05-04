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

use bevy::prelude::*;
use thalos_physics::trajectory::{
    EncounterId, FlightPlan, NumericSegment, Trajectory, TrajectoryEventKind,
};
use thalos_physics::types::BodyId;

use crate::camera::ActiveCamera;
use crate::coords::{RenderOrigin, WorldScale, sample_render_pos};
use crate::map_view::MapSnapshot;
use crate::photo_mode::HideInPhotoMode;
use crate::rendering::screen_marker_radius;
use crate::view::HideInShipView;

use super::FlightPlanView;

// ---------------------------------------------------------------------------
// Public marker kind
// ---------------------------------------------------------------------------

/// Kinds of marker the renderer knows how to draw. Extend by adding a variant
/// + an arm in [`MarkerKind::base_color`] and [`MarkerKind::from_event_kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkerKind {
    Apoapsis,
    Periapsis,
    SoiEntry,
    SoiExit,
    ClosestApproach,
    // Future: AscendingNode, DescendingNode.
}

impl MarkerKind {
    fn from_event_kind(kind: TrajectoryEventKind) -> Option<Self> {
        match kind {
            TrajectoryEventKind::Apoapsis => Some(MarkerKind::Apoapsis),
            TrajectoryEventKind::Periapsis => Some(MarkerKind::Periapsis),
            TrajectoryEventKind::SoiEntry => Some(MarkerKind::SoiEntry),
            TrajectoryEventKind::SoiExit => Some(MarkerKind::SoiExit),
            TrajectoryEventKind::SurfaceImpact => None,
        }
    }

    fn base_color(self) -> Color {
        match self {
            MarkerKind::Apoapsis => Color::srgb(0.95, 0.30, 0.30),
            MarkerKind::Periapsis => Color::srgb(0.30, 0.75, 1.00),
            MarkerKind::SoiEntry => Color::srgb(0.30, 0.85, 1.00),
            MarkerKind::SoiExit => Color::srgb(1.00, 0.48, 0.20),
            MarkerKind::ClosestApproach => Color::srgb(1.00, 0.88, 0.25),
        }
    }

    fn is_apsis(self) -> bool {
        matches!(self, MarkerKind::Apoapsis | MarkerKind::Periapsis)
    }
}

#[derive(Component, Debug, Clone, Copy)]
pub struct TrajectoryMarker {
    key: MarkerKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MarkerKey {
    kind: MarkerKind,
    source_id: EncounterId,
}

#[derive(Debug, Clone, Copy)]
struct TrajectoryAttachment {
    kind: MarkerKind,
    body: BodyId,
    leg_index: usize,
    epoch: f64,
    source_id: EncounterId,
}

// ---------------------------------------------------------------------------
// Shared assets
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub(super) struct TrajectoryMarkerAssets {
    mesh: Handle<Mesh>,
    apoapsis: KindMaterials,
    periapsis: KindMaterials,
    soi_entry: KindMaterials,
    soi_exit: KindMaterials,
    closest_approach: KindMaterials,
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
            MarkerKind::SoiEntry => &self.soi_entry,
            MarkerKind::SoiExit => &self.soi_exit,
            MarkerKind::ClosestApproach => &self.closest_approach,
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
    let soi_entry = kind_mats(MarkerKind::SoiEntry, &mut materials);
    let soi_exit = kind_mats(MarkerKind::SoiExit, &mut materials);
    let closest_approach = kind_mats(MarkerKind::ClosestApproach, &mut materials);
    commands.insert_resource(TrajectoryMarkerAssets {
        mesh,
        apoapsis,
        periapsis,
        soi_entry,
        soi_exit,
        closest_approach,
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
// Spec construction (events → trajectory-attached marker specs)
// ---------------------------------------------------------------------------

struct MarkerSpec {
    key: MarkerKey,
    kind: MarkerKind,
    world_pos: Vec3,
    is_ghost: bool,
}

fn compute_marker_specs(
    flight_plan: &FlightPlan,
    snapshot: &MapSnapshot,
    flight_plan_view: &FlightPlanView,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> Vec<MarkerSpec> {
    let mut specs = Vec::new();
    let focused_ghost = flight_plan_view.focused_ghost();

    // Multi-revolution legs (fast moon orbit, long heliocentric horizon) emit
    // a Pe + Ap pair per revolution. Keep only the first visible, resolved
    // marker of each kind per (kind, body, leg); events arrive in time order.
    let mut seen_apsides: HashSet<(MarkerKind, usize, usize)> = HashSet::new();

    if focused_ghost.is_none() {
        for event in flight_plan.events() {
            let Some(kind) = MarkerKind::from_event_kind(event.kind) else {
                continue;
            };
            if !kind.is_apsis() {
                continue;
            }
            let seen_key = (kind, event.body, event.leg_index);
            if seen_apsides.contains(&seen_key) {
                continue;
            }
            if flight_plan_view.epoch_hidden_in_focus(flight_plan, &snapshot.body_defs, event.epoch)
            {
                continue;
            }

            let attachment = TrajectoryAttachment {
                kind,
                body: event.body,
                leg_index: event.leg_index,
                epoch: event.epoch,
                source_id: event.id,
            };
            if push_attached_marker(
                &mut specs,
                flight_plan,
                attachment,
                flight_plan_view,
                snapshot,
                origin,
                scale,
            ) {
                seen_apsides.insert(seen_key);
            }
        }

        supplement_derived_apsides(
            flight_plan,
            flight_plan_view,
            snapshot,
            origin,
            scale,
            &mut seen_apsides,
            &mut specs,
        );
    }
    supplement_encounter_markers(
        flight_plan,
        flight_plan_view,
        snapshot,
        origin,
        scale,
        &mut specs,
    );
    supplement_approach_markers(
        flight_plan,
        flight_plan_view,
        snapshot,
        origin,
        scale,
        &mut specs,
    );

    specs
}

fn supplement_derived_apsides(
    flight_plan: &FlightPlan,
    flight_plan_view: &FlightPlanView,
    snapshot: &MapSnapshot,
    origin: &RenderOrigin,
    scale: &WorldScale,
    seen: &mut HashSet<(MarkerKind, usize, usize)>,
    specs: &mut Vec<MarkerSpec>,
) {
    for (leg_index, leg) in flight_plan.legs().iter().enumerate() {
        let Some(anchor_body) = leg
            .coast_segment
            .samples
            .first()
            .map(|sample| sample.anchor_body)
        else {
            continue;
        };
        for kind in [MarkerKind::Periapsis, MarkerKind::Apoapsis] {
            let seen_key = (kind, anchor_body, leg_index);
            if seen.contains(&seen_key) {
                continue;
            }
            let Some(epoch) = derived_apsis_epoch(&leg.coast_segment, kind) else {
                continue;
            };
            let attachment = TrajectoryAttachment {
                kind,
                body: anchor_body,
                leg_index,
                epoch,
                source_id: derived_source_id(leg_index, kind),
            };
            if push_attached_marker(
                specs,
                flight_plan,
                attachment,
                flight_plan_view,
                snapshot,
                origin,
                scale,
            ) {
                seen.insert(seen_key);
            }
        }
    }
}

fn supplement_encounter_markers(
    flight_plan: &FlightPlan,
    flight_plan_view: &FlightPlanView,
    snapshot: &MapSnapshot,
    origin: &RenderOrigin,
    scale: &WorldScale,
    specs: &mut Vec<MarkerSpec>,
) {
    let focused_ghost = flight_plan_view.focused_ghost();
    for encounter in flight_plan.encounters() {
        if let Some(focus) = focused_ghost
            && !focus.matches(encounter.body, encounter.entry_epoch)
        {
            continue;
        }
        for (kind, epoch) in [
            (MarkerKind::SoiEntry, Some(encounter.entry_epoch)),
            (MarkerKind::ClosestApproach, Some(encounter.closest_epoch)),
            (MarkerKind::SoiExit, encounter.exit_epoch),
        ] {
            let Some(epoch) = epoch else {
                continue;
            };
            let attachment = TrajectoryAttachment {
                kind,
                body: encounter.body,
                leg_index: encounter.leg_index,
                epoch,
                source_id: encounter.id,
            };
            push_attached_marker(
                specs,
                flight_plan,
                attachment,
                flight_plan_view,
                snapshot,
                origin,
                scale,
            );
        }
    }
}

fn supplement_approach_markers(
    flight_plan: &FlightPlan,
    flight_plan_view: &FlightPlanView,
    snapshot: &MapSnapshot,
    origin: &RenderOrigin,
    scale: &WorldScale,
    specs: &mut Vec<MarkerSpec>,
) {
    let Some(target_body) = snapshot.target_body else {
        return;
    };
    let focused_ghost = flight_plan_view.focused_ghost();
    for approach in flight_plan.approaches() {
        if approach.body != target_body {
            continue;
        }
        if let Some(focus) = focused_ghost
            && !focus.matches(approach.body, approach.epoch)
        {
            continue;
        }
        let Some(leg_index) = flight_plan.leg_at_time(approach.epoch) else {
            continue;
        };
        let attachment = TrajectoryAttachment {
            kind: MarkerKind::ClosestApproach,
            body: approach.body,
            leg_index,
            epoch: approach.epoch,
            source_id: approach_source_id(approach.body, approach.epoch),
        };
        push_attached_marker(
            specs,
            flight_plan,
            attachment,
            flight_plan_view,
            snapshot,
            origin,
            scale,
        );
    }
}

fn push_attached_marker(
    specs: &mut Vec<MarkerSpec>,
    flight_plan: &FlightPlan,
    attachment: TrajectoryAttachment,
    flight_plan_view: &FlightPlanView,
    snapshot: &MapSnapshot,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> bool {
    let Some((world_pos, is_ghost)) = resolve_marker_attachment(
        flight_plan,
        attachment,
        flight_plan_view,
        snapshot,
        origin,
        scale,
    ) else {
        return false;
    };
    specs.push(MarkerSpec {
        key: MarkerKey {
            kind: attachment.kind,
            source_id: attachment.source_id,
        },
        kind: attachment.kind,
        world_pos,
        is_ghost,
    });
    true
}

fn derived_apsis_epoch(segment: &NumericSegment, kind: MarkerKind) -> Option<f64> {
    let selected = match kind {
        MarkerKind::Periapsis => segment.samples.iter().min_by(|a, b| {
            apsis_radius_sq(a)
                .partial_cmp(&apsis_radius_sq(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        MarkerKind::Apoapsis => segment.samples.iter().max_by(|a, b| {
            apsis_radius_sq(a)
                .partial_cmp(&apsis_radius_sq(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        MarkerKind::SoiEntry | MarkerKind::SoiExit | MarkerKind::ClosestApproach => None,
    }?;
    Some(selected.time)
}

fn apsis_radius_sq(sample: &thalos_physics::types::TrajectorySample) -> f64 {
    (sample.position - sample.ref_pos).length_squared()
}

fn derived_source_id(leg_index: usize, kind: MarkerKind) -> EncounterId {
    let kind_bit = match kind {
        MarkerKind::Periapsis => 0,
        MarkerKind::Apoapsis => 1,
        MarkerKind::SoiEntry | MarkerKind::SoiExit | MarkerKind::ClosestApproach => 2,
    };
    (1_u64 << 63) | ((leg_index as u64) << 1) | kind_bit
}

fn approach_source_id(body: BodyId, epoch: f64) -> EncounterId {
    (1_u64 << 62)
        | (((body as u64) & 0xffff) << 40)
        | (epoch.max(0.0).round() as u64 & ((1_u64 << 40) - 1))
}

fn resolve_marker_attachment(
    flight_plan: &FlightPlan,
    attachment: TrajectoryAttachment,
    flight_plan_view: &FlightPlanView,
    snapshot: &MapSnapshot,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> Option<(Vec3, bool)> {
    let (segment, is_ghost) = segment_for_attachment(flight_plan, attachment)?;
    if let Some(world_pos) = focused_ghost_marker_position(
        segment,
        attachment,
        flight_plan_view,
        snapshot,
        origin,
        scale,
    ) {
        return Some((world_pos, true));
    }
    let first = segment.samples.first()?;
    let pin = flight_plan_view.pin_for_body(first.anchor_body, first.time, &snapshot.body_states);
    let world_pos = rendered_segment_point(segment, attachment.epoch, pin, origin, scale)?;
    Some((world_pos, is_ghost))
}

fn focused_ghost_marker_position(
    segment: &NumericSegment,
    attachment: TrajectoryAttachment,
    flight_plan_view: &FlightPlanView,
    snapshot: &MapSnapshot,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> Option<Vec3> {
    let focus = flight_plan_view.focused_ghost()?;
    if attachment.body != focus.body_id {
        return None;
    }
    let ghost = flight_plan_view.focused_ghost_ref()?;
    if let Some(window) = ghost.trajectory_window {
        if attachment.epoch < window.start_epoch - 1.0 || attachment.epoch > window.end_epoch + 1.0
        {
            return None;
        }
    } else if (attachment.epoch - ghost.encounter_epoch).abs() > 1.0 {
        return None;
    }

    let state = segment.state_at(attachment.epoch)?;
    let body_state = snapshot.body_state_at(attachment.body, attachment.epoch)?;
    let relative = state.position - body_state.position;
    let relative = if matches!(attachment.kind, MarkerKind::SoiEntry | MarkerKind::SoiExit) {
        let soi_radius = snapshot
            .body_defs
            .get(attachment.body)
            .map(|body| body.soi_radius_m)
            .unwrap_or(f64::INFINITY);
        if soi_radius.is_finite() && relative.length() > soi_radius * 1.000_001 {
            relative
                .try_normalize()
                .map(|direction| direction * soi_radius)
                .unwrap_or(relative)
        } else {
            relative
        }
    } else {
        relative
    };
    let pin = flight_plan_view.pin_for_ghost_focus(focus, &snapshot.body_states);
    Some(((relative + pin - origin.position) * scale.0).as_vec3())
}

fn segment_for_attachment(
    flight_plan: &FlightPlan,
    attachment: TrajectoryAttachment,
) -> Option<(&NumericSegment, bool)> {
    let leg = flight_plan.legs().get(attachment.leg_index)?;
    match attachment.kind {
        MarkerKind::SoiEntry => {
            if let Some(segment) = leg
                .segments()
                .find(|segment| segment_start_matches(segment, attachment.epoch))
            {
                return Some((segment, attachment.leg_index > 0));
            }
        }
        MarkerKind::SoiExit => {
            if let Some(segment) = leg
                .segments()
                .find(|segment| segment_end_matches(segment, attachment.epoch))
            {
                return Some((segment, attachment.leg_index > 0));
            }
        }
        MarkerKind::Apoapsis | MarkerKind::Periapsis | MarkerKind::ClosestApproach => {}
    }
    for segment in leg.segments() {
        if segment_contains_epoch(segment, attachment.epoch) {
            return Some((segment, attachment.leg_index > 0));
        }
    }
    None
}

fn segment_start_matches(segment: &NumericSegment, epoch: f64) -> bool {
    segment
        .start_time()
        .map(|start| (start - epoch).abs() <= 1e-6)
        .unwrap_or(false)
}

fn segment_end_matches(segment: &NumericSegment, epoch: f64) -> bool {
    segment
        .end_time()
        .map(|end| (end - epoch).abs() <= 1e-6)
        .unwrap_or(false)
}

fn segment_contains_epoch(segment: &NumericSegment, epoch: f64) -> bool {
    let (start, end) = segment.epoch_range();
    epoch >= start - 1e-6 && epoch <= end + 1e-6
}

fn rendered_segment_point(
    segment: &NumericSegment,
    epoch: f64,
    pin: bevy::math::DVec3,
    origin: &RenderOrigin,
    scale: &WorldScale,
) -> Option<Vec3> {
    let samples = segment.samples.as_slice();
    match samples {
        [] => None,
        [only] if (epoch - only.time).abs() <= 1e-6 => {
            Some(sample_render_pos(only, pin, origin, scale))
        }
        [_] => None,
        _ => {
            for pair in samples.windows(2) {
                let a = &pair[0];
                let b = &pair[1];
                if epoch < a.time - 1e-6 || epoch > b.time + 1e-6 {
                    continue;
                }
                let span = b.time - a.time;
                if span <= 0.0 {
                    return Some(sample_render_pos(a, pin, origin, scale));
                }
                let t = ((epoch - a.time) / span).clamp(0.0, 1.0) as f32;
                let a_pos = sample_render_pos(a, pin, origin, scale);
                let b_pos = sample_render_pos(b, pin, origin, scale);
                return Some(a_pos.lerp(b_pos, t));
            }
            None
        }
    }
}

#[inline]
fn billboard_transform(world_pos: Vec3, camera_rotation: Quat, marker_scale: f32) -> Transform {
    Transform {
        translation: world_pos,
        rotation: camera_rotation,
        scale: Vec3::splat(marker_scale),
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

pub(super) fn manage_trajectory_markers(
    mut commands: Commands,
    assets: Option<Res<TrajectoryMarkerAssets>>,
    snapshot: Res<MapSnapshot>,
    origin: Res<RenderOrigin>,
    scale: Res<WorldScale>,
    flight_plan_view: Res<FlightPlanView>,
    camera_q: Query<
        &Transform,
        (
            With<ActiveCamera>,
            With<crate::camera::OrbitCamera>,
            Without<TrajectoryMarker>,
        ),
    >,
    mut markers: Query<(Entity, &TrajectoryMarker, &mut Transform, &mut Visibility)>,
    mut last_version: Local<Option<u64>>,
) {
    let Some(assets) = assets.as_deref() else {
        return;
    };
    let Some(flight_plan) = snapshot.flight_plan.as_ref() else {
        return;
    };
    if snapshot.body_states.is_empty() {
        return;
    }
    let Ok(cam_tf) = camera_q.single() else {
        return;
    };
    let cam_rot = cam_tf.rotation;
    let cam_pos = cam_tf.translation;

    let version = snapshot.prediction_version;
    let specs = compute_marker_specs(flight_plan, &snapshot, &flight_plan_view, &origin, &scale);

    let version_changed = *last_version != Some(version);
    if version_changed {
        // Event ids reset across predictions, so reusing entities by id would
        // alias markers across unrelated apsides. Drop everything and respawn.
        for (entity, _, _, _) in &markers {
            commands.entity(entity).despawn();
        }
        for spec in &specs {
            spawn_marker(&mut commands, assets, spec, cam_rot, cam_pos);
        }
        *last_version = Some(version);
        return;
    }

    // Same prediction: update transforms in-place, keyed by marker source.
    let mut by_key: std::collections::HashMap<MarkerKey, &MarkerSpec> =
        specs.iter().map(|s| (s.key, s)).collect();
    for (_, marker, mut tf, mut vis) in &mut markers {
        if let Some(spec) = by_key.remove(&marker.key) {
            *vis = Visibility::Inherited;
            *tf = billboard_transform(
                spec.world_pos,
                cam_rot,
                screen_marker_radius(spec.world_pos, cam_pos),
            );
        } else {
            *vis = Visibility::Hidden;
        }
    }
    for spec in by_key.values() {
        spawn_marker(&mut commands, assets, spec, cam_rot, cam_pos);
    }
}

fn spawn_marker(
    commands: &mut Commands,
    assets: &TrajectoryMarkerAssets,
    spec: &MarkerSpec,
    cam_rot: Quat,
    cam_pos: Vec3,
) {
    commands.spawn((
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(assets.material(spec.kind, spec.is_ghost)),
        billboard_transform(
            spec.world_pos,
            cam_rot,
            screen_marker_radius(spec.world_pos, cam_pos),
        ),
        TrajectoryMarker { key: spec.key },
        HideInPhotoMode,
        HideInShipView,
    ));
}

#[cfg(test)]
mod tests {
    use bevy::math::{DQuat, DVec3};
    use thalos_physics::canonical::Epoch;
    use thalos_physics::trajectory::{
        CaptureStatus, ClosestApproach, Encounter, Leg, TrajectoryEvent,
    };
    use thalos_physics::types::{BodyState, StateVector, TrajectorySample};

    use crate::coords::RenderGhostFocus;
    use crate::map_view::ProjectedBodyState;

    use super::super::view::{Ghost, GhostPhase, TrajectoryWindow};
    use super::*;

    fn sample(time: f64, position: DVec3, ref_pos: DVec3) -> TrajectorySample {
        TrajectorySample {
            time,
            position,
            velocity: DVec3::ZERO,
            anchor_body: 0,
            ref_pos,
        }
    }

    fn body_state(position: DVec3) -> BodyState {
        body_state_at(0, 0.0, position)
    }

    fn body_state_at(id: BodyId, epoch: f64, position: DVec3) -> BodyState {
        BodyState {
            id,
            epoch: Epoch(epoch),
            position,
            velocity: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
            mass_kg: 1.0,
            gm: 1.0,
            radius_m: 1.0,
        }
    }

    fn projected_body_state(id: BodyId, epoch: f64, position: DVec3) -> ProjectedBodyState {
        ProjectedBodyState {
            body: id,
            epoch: Epoch(epoch),
            state: body_state_at(id, epoch, position),
        }
    }

    fn snapshot_with_body_states(body_states: Vec<BodyState>) -> MapSnapshot {
        MapSnapshot {
            body_states,
            ..default()
        }
    }

    fn segment() -> NumericSegment {
        NumericSegment {
            samples: vec![
                sample(0.0, DVec3::new(0.0, 0.0, 0.0), DVec3::new(100.0, 0.0, 0.0)),
                sample(
                    10.0,
                    DVec3::new(20.0, 0.0, 0.0),
                    DVec3::new(110.0, 0.0, 0.0),
                ),
            ],
            is_stable_orbit: false,
            stable_orbit_start_index: None,
            collision_body: None,
        }
    }

    fn flight_plan(segment: NumericSegment) -> FlightPlan {
        FlightPlan {
            initial_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            initial_time: 0.0,
            legs: vec![Leg {
                start_state: StateVector {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                },
                start_time: 0.0,
                applied_delta_v: None,
                burn_segment: None,
                coast_segment: segment.clone(),
            }],
            segments: vec![segment],
            events: vec![TrajectoryEvent {
                id: 7,
                body: 0,
                epoch: 5.0,
                kind: TrajectoryEventKind::Apoapsis,
                craft_state: StateVector {
                    position: DVec3::new(-9_999.0, 0.0, 0.0),
                    velocity: DVec3::ZERO,
                },
                body_state: StateVector {
                    position: DVec3::new(-9_999.0, 0.0, 0.0),
                    velocity: DVec3::ZERO,
                },
                leg_index: 0,
            }],
            encounters: Vec::new(),
            approaches: Vec::new(),
            baseline: None,
        }
    }

    fn boundary_plan() -> FlightPlan {
        let burn = NumericSegment {
            samples: vec![
                sample(0.0, DVec3::new(0.0, 0.0, 0.0), DVec3::new(100.0, 0.0, 0.0)),
                sample(
                    10.0,
                    DVec3::new(20.0, 0.0, 0.0),
                    DVec3::new(110.0, 0.0, 0.0),
                ),
            ],
            is_stable_orbit: false,
            stable_orbit_start_index: None,
            collision_body: None,
        };
        let coast = NumericSegment {
            samples: vec![
                sample(
                    10.0,
                    DVec3::new(120.0, 0.0, 0.0),
                    DVec3::new(110.0, 0.0, 0.0),
                ),
                sample(
                    20.0,
                    DVec3::new(130.0, 0.0, 0.0),
                    DVec3::new(120.0, 0.0, 0.0),
                ),
            ],
            is_stable_orbit: false,
            stable_orbit_start_index: None,
            collision_body: None,
        };
        FlightPlan {
            initial_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            initial_time: 0.0,
            legs: vec![Leg {
                start_state: StateVector {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                },
                start_time: 0.0,
                applied_delta_v: None,
                burn_segment: Some(burn.clone()),
                coast_segment: coast.clone(),
            }],
            segments: vec![burn, coast],
            events: Vec::new(),
            encounters: Vec::new(),
            approaches: Vec::new(),
            baseline: None,
        }
    }

    #[test]
    fn marker_attachment_resolves_to_displayed_trajectory_polyline() {
        let segment = segment();
        let plan = flight_plan(segment.clone());
        let view = FlightPlanView::default();
        let body_states = vec![body_state(DVec3::new(1_000.0, 0.0, 0.0))];
        let snapshot = snapshot_with_body_states(body_states.clone());
        let origin = RenderOrigin::default();
        let scale = WorldScale(1.0);

        let attachment = TrajectoryAttachment {
            kind: MarkerKind::Apoapsis,
            body: 0,
            leg_index: 0,
            epoch: 5.0,
            source_id: 7,
        };
        let (world_pos, is_ghost) =
            resolve_marker_attachment(&plan, attachment, &view, &snapshot, &origin, &scale)
                .expect("marker should resolve");

        let pin = view.pin_for_body(0, 0.0, &body_states);
        let start = sample_render_pos(&segment.samples[0], pin, &origin, &scale);
        let end = sample_render_pos(&segment.samples[1], pin, &origin, &scale);

        assert!(!is_ghost);
        assert_eq!(world_pos, start.lerp(end, 0.5));
        assert_eq!(world_pos, Vec3::new(905.0, 0.0, 0.0));
    }

    #[test]
    fn soi_boundary_markers_choose_the_visible_side_of_boundary() {
        let plan = boundary_plan();
        let view = FlightPlanView::default();
        let snapshot = snapshot_with_body_states(vec![body_state(DVec3::new(1_000.0, 0.0, 0.0))]);
        let origin = RenderOrigin::default();
        let scale = WorldScale(1.0);

        let exit = TrajectoryAttachment {
            kind: MarkerKind::SoiExit,
            body: 0,
            leg_index: 0,
            epoch: 10.0,
            source_id: 1,
        };
        let entry = TrajectoryAttachment {
            kind: MarkerKind::SoiEntry,
            body: 0,
            leg_index: 0,
            epoch: 10.0,
            source_id: 2,
        };

        let (exit_pos, _) =
            resolve_marker_attachment(&plan, exit, &view, &snapshot, &origin, &scale)
                .expect("exit marker should resolve");
        let (entry_pos, _) =
            resolve_marker_attachment(&plan, entry, &view, &snapshot, &origin, &scale)
                .expect("entry marker should resolve");

        assert_eq!(exit_pos, Vec3::new(910.0, 0.0, 0.0));
        assert_eq!(entry_pos, Vec3::new(1_010.0, 0.0, 0.0));
    }

    #[test]
    fn event_craft_state_cannot_move_marker_off_trajectory() {
        let plan = flight_plan(segment());
        let view = FlightPlanView::default();
        let snapshot = snapshot_with_body_states(vec![body_state(DVec3::new(1_000.0, 0.0, 0.0))]);
        let origin = RenderOrigin::default();
        let scale = WorldScale(1.0);

        let specs = compute_marker_specs(&plan, &snapshot, &view, &origin, &scale);

        let event_spec = specs
            .iter()
            .find(|spec| spec.key.source_id == 7)
            .expect("event-backed marker should resolve");
        assert_eq!(event_spec.world_pos, Vec3::new(905.0, 0.0, 0.0));
    }

    #[test]
    fn missing_apsis_events_are_derived_from_segment_samples() {
        let plan = flight_plan(segment());
        let view = FlightPlanView::default();
        let snapshot = snapshot_with_body_states(vec![body_state(DVec3::new(1_000.0, 0.0, 0.0))]);
        let origin = RenderOrigin::default();
        let scale = WorldScale(1.0);

        let specs = compute_marker_specs(&plan, &snapshot, &view, &origin, &scale);

        assert!(
            specs
                .iter()
                .any(|spec| spec.kind == MarkerKind::Apoapsis && spec.key.source_id == 7)
        );
        let derived_periapsis = specs
            .iter()
            .find(|spec| {
                spec.kind == MarkerKind::Periapsis
                    && spec.key.source_id == derived_source_id(0, MarkerKind::Periapsis)
            })
            .expect("missing periapsis should be derived from the trajectory segment");
        assert_eq!(derived_periapsis.world_pos, Vec3::new(910.0, 0.0, 0.0));
    }

    #[test]
    fn encounter_markers_attach_to_displayed_trajectory() {
        let mut plan = flight_plan(segment());
        plan.encounters.push(Encounter {
            id: 9,
            body: 0,
            leg_index: 0,
            entry_epoch: 0.0,
            exit_epoch: Some(10.0),
            closest_epoch: 5.0,
            closest_distance: 1.0,
            periapsis_altitude: 0.0,
            relative_velocity: 0.0,
            eccentricity: 1.0,
            inclination_rad: 0.0,
            capture: CaptureStatus::Flyby,
            craft_state: StateVector {
                position: DVec3::new(-9_999.0, 0.0, 0.0),
                velocity: DVec3::ZERO,
            },
            body_state: StateVector {
                position: DVec3::new(-9_999.0, 0.0, 0.0),
                velocity: DVec3::ZERO,
            },
        });
        let view = FlightPlanView::default();
        let snapshot = snapshot_with_body_states(vec![body_state(DVec3::new(1_000.0, 0.0, 0.0))]);
        let origin = RenderOrigin::default();
        let scale = WorldScale(1.0);

        let specs = compute_marker_specs(&plan, &snapshot, &view, &origin, &scale);

        let by_kind = |kind| {
            specs
                .iter()
                .find(|spec| spec.kind == kind && spec.key.source_id == 9)
                .map(|spec| spec.world_pos)
        };
        assert_eq!(
            by_kind(MarkerKind::SoiEntry),
            Some(Vec3::new(900.0, 0.0, 0.0))
        );
        assert_eq!(
            by_kind(MarkerKind::ClosestApproach),
            Some(Vec3::new(905.0, 0.0, 0.0))
        );
        assert_eq!(
            by_kind(MarkerKind::SoiExit),
            Some(Vec3::new(910.0, 0.0, 0.0))
        );
    }

    #[test]
    fn focused_ghost_hides_apsides_but_keeps_focused_encounter_markers() {
        let mut plan = flight_plan(segment());
        plan.encounters.push(Encounter {
            id: 9,
            body: 1,
            leg_index: 0,
            entry_epoch: 0.0,
            exit_epoch: Some(10.0),
            closest_epoch: 5.0,
            closest_distance: 1.0,
            periapsis_altitude: 0.0,
            relative_velocity: 0.0,
            eccentricity: 1.0,
            inclination_rad: 0.0,
            capture: CaptureStatus::Flyby,
            craft_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            body_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
        });

        let focus = RenderGhostFocus {
            body_id: 1,
            parent_id: 0,
            relative_position: DVec3::new(100.0, 0.0, 0.0),
            projection_epoch: 5.0,
            encounter_epoch: 0.0,
        };
        let mut view = FlightPlanView::default();
        view.set_focused_ghost(focus);
        view.ghosts_mut().push(Ghost {
            body_id: 1,
            parent_id: 0,
            relative_position: focus.relative_position,
            projection_epoch: focus.projection_epoch,
            encounter_epoch: focus.encounter_epoch,
            entity: None,
            phase: GhostPhase::Active,
            trajectory_window: Some(TrajectoryWindow {
                start_epoch: 0.0,
                end_epoch: 10.0,
                exit_epoch: Some(10.0),
            }),
        });

        let mut snapshot = snapshot_with_body_states(vec![
            body_state_at(0, 0.0, DVec3::ZERO),
            body_state_at(1, 0.0, DVec3::new(100.0, 0.0, 0.0)),
        ]);
        snapshot.projected_body_states = vec![
            projected_body_state(1, 5.0, DVec3::new(110.0, 0.0, 0.0)),
            projected_body_state(1, 10.0, DVec3::new(120.0, 0.0, 0.0)),
        ];
        let origin = RenderOrigin::default();
        let scale = WorldScale(1.0);

        let specs = compute_marker_specs(&plan, &snapshot, &view, &origin, &scale);

        assert!(
            specs
                .iter()
                .all(|spec| !matches!(spec.kind, MarkerKind::Apoapsis | MarkerKind::Periapsis))
        );
        for kind in [
            MarkerKind::SoiEntry,
            MarkerKind::ClosestApproach,
            MarkerKind::SoiExit,
        ] {
            assert!(
                specs
                    .iter()
                    .any(|spec| spec.kind == kind && spec.key.source_id == 9),
                "{kind:?} marker should remain visible for the focused encounter"
            );
        }
    }

    #[test]
    fn non_target_closest_approaches_do_not_create_markers() {
        let mut plan = flight_plan(segment());
        plan.approaches.push(ClosestApproach {
            body: 1,
            epoch: 5.0,
            distance: 1.0,
            craft_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            body_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
        });
        let view = FlightPlanView::default();
        let snapshot = snapshot_with_body_states(vec![
            body_state(DVec3::new(1_000.0, 0.0, 0.0)),
            body_state(DVec3::new(2_000.0, 0.0, 0.0)),
        ]);
        let origin = RenderOrigin::default();
        let scale = WorldScale(1.0);

        let specs = compute_marker_specs(&plan, &snapshot, &view, &origin, &scale);

        assert!(
            specs
                .iter()
                .all(|spec| spec.kind != MarkerKind::ClosestApproach)
        );
    }

    #[test]
    fn target_closest_approach_creates_one_marker() {
        let mut plan = flight_plan(segment());
        plan.approaches.push(ClosestApproach {
            body: 1,
            epoch: 5.0,
            distance: 1.0,
            craft_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            body_state: StateVector {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
        });
        let view = FlightPlanView::default();
        let mut snapshot = snapshot_with_body_states(vec![
            body_state(DVec3::new(1_000.0, 0.0, 0.0)),
            body_state(DVec3::new(2_000.0, 0.0, 0.0)),
        ]);
        snapshot.target_body = Some(1);
        let origin = RenderOrigin::default();
        let scale = WorldScale(1.0);

        let specs = compute_marker_specs(&plan, &snapshot, &view, &origin, &scale);

        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.kind == MarkerKind::ClosestApproach)
                .count(),
            1
        );
    }
}
