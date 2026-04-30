use bevy::prelude::*;

use super::super::helpers::{node_world_position, overlay_marker_transform};
use super::super::state::{
    InteractionMode, ManeuverPlan, NodeMarkerDisc, SelectedNode, SnapIndicator,
};
use crate::camera::ActiveCamera;
use crate::coords::{RenderOrigin, WorldScale};
use crate::flight_plan_view::FlightPlanView;
use crate::photo_mode::HideInPhotoMode;
use crate::rendering::{FrameBodyStates, SimulationState, screen_marker_radius};
use crate::view::HideInShipView;

/// Spawn the snap indicator (hidden by default).
pub(in crate::maneuver) fn spawn_snap_indicator(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(Circle::new(1.0));
    let mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 1.0, 0.0),
        emissive: {
            let lin: LinearRgba = Color::srgb(1.0, 1.0, 0.0).into();
            (lin * 2.0).into()
        },
        unlit: true,
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(mat),
        Transform::default(),
        Visibility::Hidden,
        SnapIndicator,
        HideInPhotoMode,
        HideInShipView,
    ));
}

/// Update snap indicator position/visibility.
pub(in crate::maneuver) fn update_snap_indicator(
    mode: Res<InteractionMode>,
    camera_q: Query<
        &Transform,
        (
            With<ActiveCamera>,
            With<crate::camera::OrbitCamera>,
            Without<SnapIndicator>,
        ),
    >,
    mut indicators: Query<(&mut Transform, &mut Visibility), With<SnapIndicator>>,
) {
    let Ok((mut tf, mut vis)) = indicators.single_mut() else {
        return;
    };

    if let InteractionMode::PlacingNode {
        snap_world_pos: Some(pos),
        ..
    } = *mode
    {
        let Ok(cam_tf) = camera_q.single() else {
            *vis = Visibility::Hidden;
            return;
        };
        *vis = Visibility::Inherited;
        *tf = overlay_marker_transform(
            pos,
            cam_tf.rotation,
            screen_marker_radius(pos, cam_tf.translation),
        );
        return;
    }
    *vis = Visibility::Hidden;
}

/// Manage flat circle markers for unselected maneuver nodes.
pub(in crate::maneuver) fn manage_node_markers(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    plan: Res<ManeuverPlan>,
    selected: Res<SelectedNode>,
    sim: Option<Res<SimulationState>>,
    body_states: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    flight_plan_view: Res<FlightPlanView>,
    scale: Res<WorldScale>,
    camera_q: Query<
        &Transform,
        (
            With<ActiveCamera>,
            With<crate::camera::OrbitCamera>,
            Without<NodeMarkerDisc>,
        ),
    >,
    mut markers: Query<(Entity, &NodeMarkerDisc, &mut Transform, &mut Visibility)>,
) {
    let Ok(cam_tf) = camera_q.single() else {
        return;
    };
    let cam_rot = cam_tf.rotation;
    let cam_pos = cam_tf.translation;

    let prediction = sim.as_ref().and_then(|s| s.simulation.prediction());
    let states = body_states.states.as_deref();
    let sim_ref = sim.as_deref();

    for (entity, marker, mut tf, mut vis) in &mut markers {
        let selected = selected.id == Some(marker.node_id);
        let node_exists = plan.nodes.iter().any(|n| n.id == marker.node_id);

        if !node_exists || selected {
            commands.entity(entity).despawn();
            continue;
        }

        let node = plan.nodes.iter().find(|n| n.id == marker.node_id).unwrap();
        if let (Some(pred), Some(states), Some(sim)) = (prediction, states, sim_ref) {
            if let Some(world_pos) = node_world_position(
                node,
                pred,
                states,
                &origin,
                &scale,
                &sim.system,
                sim.ephemeris.as_ref(),
                &flight_plan_view,
            ) {
                *vis = Visibility::Inherited;
                *tf = overlay_marker_transform(
                    world_pos,
                    cam_rot,
                    screen_marker_radius(world_pos, cam_pos),
                );
                continue;
            }
        }
        *vis = Visibility::Hidden;
    }

    for node in &plan.nodes {
        if selected.id == Some(node.id) {
            continue;
        }
        let already_exists = markers
            .iter()
            .any(|(_, marker, _, _)| marker.node_id == node.id);
        if already_exists {
            continue;
        }

        let mesh = meshes.add(Circle::new(1.0));
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.6, 0.0),
            emissive: {
                let lin: LinearRgba = Color::srgb(0.8, 0.6, 0.0).into();
                (lin * 2.0).into()
            },
            unlit: true,
            cull_mode: None,
            ..default()
        });

        let (world_pos, visibility) = prediction
            .zip(states)
            .zip(sim_ref)
            .and_then(|((pred, states), sim)| {
                node_world_position(
                    node,
                    pred,
                    states,
                    &origin,
                    &scale,
                    &sim.system,
                    sim.ephemeris.as_ref(),
                    &flight_plan_view,
                )
            })
            .map(|pos| (pos, Visibility::Inherited))
            .unwrap_or((Vec3::ZERO, Visibility::Hidden));

        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            overlay_marker_transform(world_pos, cam_rot, screen_marker_radius(world_pos, cam_pos)),
            visibility,
            NodeMarkerDisc { node_id: node.id },
            HideInPhotoMode,
            HideInShipView,
        ));
    }
}
