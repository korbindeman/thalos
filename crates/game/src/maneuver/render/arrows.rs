use bevy::picking::hover::HoverMap;
use bevy::picking::prelude::Pickable;
use bevy::prelude::*;

use super::super::state::{
    ARROW_COLORS, ARROW_STRETCH, ArrowCone, ArrowHandle, ArrowHitbox, ArrowShaft,
    ArrowStretchState, ArrowVisual, BASE_ARROW_LEN, CONE_HEIGHT, CONE_RADIUS,
    HITBOX_CAPSULE_RADIUS, HOVER_BRIGHTNESS, InteractionMode, NodeSlideSphere, SHAFT_RADIUS,
    SLIDE_SPHERE_RADIUS, STRETCH_LERP_SPEED, SelectedNodeView, SphereVisual,
};
use crate::camera::{CameraFocus, OrbitCamera};
use crate::coords::RENDER_SCALE;
use crate::photo_mode::HideInPhotoMode;
use crate::view::HideInShipView;

/// Spawn/despawn arrow handles and the slide sphere when node selection changes.
pub(in crate::maneuver) fn manage_arrow_handles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    selected_view: Res<SelectedNodeView>,
    mode: Res<InteractionMode>,
    existing_handles: Query<Entity, With<ArrowHandle>>,
    existing_spheres: Query<Entity, With<NodeSlideSphere>>,
) {
    let has_node = selected_view.world_pos.is_some() && selected_view.frame.is_some();

    if !has_node {
        // Keep handles alive while the user is actively dragging. Despawning
        // mid-drag makes Bevy fire DragEnd on the despawned entity, which
        // exits the interaction mode and silently ends the user's slide —
        // perceived as the node "disappearing". The next frame's prediction
        // rebuild will restore a valid position.
        if matches!(
            *mode,
            InteractionMode::SlidingNode | InteractionMode::DraggingArrow { .. }
        ) {
            return;
        }
        for entity in &existing_handles {
            commands.entity(entity).despawn();
        }
        for entity in &existing_spheres {
            commands.entity(entity).despawn();
        }
        return;
    }

    if !existing_handles.is_empty() {
        return;
    }

    let hitbox_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.0, 0.0, 0.0, 0.0),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    let hitbox_mesh = meshes.add(Capsule3d::new(HITBOX_CAPSULE_RADIUS, BASE_ARROW_LEN));

    for axis in 0..3 {
        let color = ARROW_COLORS[axis];
        let base_bright: LinearRgba = color.into();
        let base_dim = LinearRgba::new(
            base_bright.red * 0.3,
            base_bright.green * 0.3,
            base_bright.blue * 0.3,
            1.0,
        );

        let shaft_mesh = meshes.add(Cylinder::new(SHAFT_RADIUS, 1.0));
        let cone_mesh = meshes.add(Cone {
            radius: CONE_RADIUS,
            height: CONE_HEIGHT,
        });

        for positive in [true, false] {
            let base_color = if positive { base_bright } else { base_dim };
            let arrow_material = materials.add(StandardMaterial {
                base_color: base_color.into(),
                emissive: base_color.into(),
                unlit: true,
                alpha_mode: AlphaMode::Blend,
                ..default()
            });

            commands
                .spawn((
                    Mesh3d(hitbox_mesh.clone()),
                    MeshMaterial3d(hitbox_mat.clone()),
                    Transform::default(),
                    ArrowHandle { axis, positive },
                    ArrowHitbox,
                    ArrowVisual {
                        material: arrow_material.clone(),
                        base_color,
                    },
                    Pickable::default(),
                    HideInPhotoMode,
                    HideInShipView,
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Mesh3d(shaft_mesh.clone()),
                        MeshMaterial3d(arrow_material.clone()),
                        Transform::from_scale(Vec3::new(1.0, BASE_ARROW_LEN, 1.0)),
                        Pickable::IGNORE,
                        ArrowShaft,
                    ));
                    parent.spawn((
                        Mesh3d(cone_mesh.clone()),
                        MeshMaterial3d(arrow_material),
                        Transform::from_translation(Vec3::new(0.0, BASE_ARROW_LEN / 2.0, 0.0)),
                        Pickable::IGNORE,
                        ArrowCone,
                    ));
                });
        }
    }

    let sphere_mesh = meshes.add(Sphere::new(SLIDE_SPHERE_RADIUS));
    let sphere_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.8, 0.0),
        emissive: Color::srgb(0.5, 0.4, 0.0).into(),
        unlit: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(sphere_mesh),
        MeshMaterial3d(sphere_mat.clone()),
        Transform::default(),
        NodeSlideSphere,
        SphereVisual {
            material: sphere_mat,
        },
        Pickable::default(),
        HideInPhotoMode,
        HideInShipView,
    ));
}

/// Update arrow transforms each frame. Arrows maintain fixed screen size,
/// fade when aligned with the camera, brighten on hover, and stretch when dragged.
pub(in crate::maneuver) fn update_arrow_transforms(
    selected_view: Res<SelectedNodeView>,
    focus: Res<CameraFocus>,
    mode: Res<InteractionMode>,
    hover_map: Res<HoverMap>,
    time: Res<Time>,
    mut stretch_state: ResMut<ArrowStretchState>,
    mut material_assets: ResMut<Assets<StandardMaterial>>,
    camera_q: Query<
        &Transform,
        (
            With<OrbitCamera>,
            Without<ArrowHitbox>,
            Without<NodeSlideSphere>,
            Without<ArrowShaft>,
            Without<ArrowCone>,
        ),
    >,
    mut hitboxes: Query<
        (
            Entity,
            &ArrowHandle,
            &mut Transform,
            &ArrowVisual,
            &mut Pickable,
            Option<&Children>,
        ),
        (
            With<ArrowHitbox>,
            Without<NodeSlideSphere>,
            Without<ArrowShaft>,
            Without<ArrowCone>,
        ),
    >,
    mut children_set: ParamSet<(
        Query<
            &mut Transform,
            (
                With<ArrowShaft>,
                Without<ArrowHitbox>,
                Without<ArrowCone>,
                Without<NodeSlideSphere>,
            ),
        >,
        Query<
            &mut Transform,
            (
                With<ArrowCone>,
                Without<ArrowHitbox>,
                Without<ArrowShaft>,
                Without<NodeSlideSphere>,
            ),
        >,
        Query<
            (Entity, &mut Transform, &SphereVisual),
            (
                With<NodeSlideSphere>,
                Without<ArrowHitbox>,
                Without<ArrowShaft>,
                Without<ArrowCone>,
            ),
        >,
    )>,
) {
    let Some(world_pos) = selected_view.world_pos else {
        return;
    };
    let Some(frame) = selected_view.frame else {
        return;
    };

    let view_dir = camera_q
        .single()
        .map(|t| -t.translation.normalize())
        .unwrap_or(-Vec3::Z);

    let hovered_entities: Vec<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    let s = (focus.distance * RENDER_SCALE) as f32;
    let arrow_len = BASE_ARROW_LEN * s;
    let sphere_gap = (SLIDE_SPHERE_RADIUS + HITBOX_CAPSULE_RADIUS) * s;

    let mut shaft_updates: Vec<(Entity, f32)> = Vec::new();
    let mut cone_updates: Vec<(Entity, f32)> = Vec::new();

    for (entity, handle, mut transform, visual, mut pickable, children) in &mut hitboxes {
        let dir = frame.col(handle.axis).normalize();
        let sign = if handle.positive { 1.0 } else { -1.0 };

        let alignment = dir.dot(view_dir).abs();
        let fade_start = 0.707;
        let opacity = if alignment < fade_start {
            1.0
        } else {
            let t = ((alignment - fade_start) / 0.2).min(1.0);
            1.0 - t * 0.85
        };

        let drag_axis = if let InteractionMode::DraggingArrow { axis, .. } = *mode {
            Some(axis)
        } else {
            None
        };

        let is_hovered = hovered_entities.contains(&entity);
        let is_dragging = drag_axis == Some(handle.axis);
        let brightness = if is_hovered || is_dragging {
            HOVER_BRIGHTNESS
        } else {
            1.0
        };

        if let Some(mat) = material_assets.get_mut(&visual.material) {
            let c = visual.base_color * opacity * brightness;
            mat.base_color = LinearRgba::new(c.red, c.green, c.blue, opacity).into();
            mat.emissive = LinearRgba::new(c.red, c.green, c.blue, opacity).into();
        }

        pickable.is_hoverable = opacity > 0.3;

        let is_this_arrow_dragged = matches!(
            *mode,
            InteractionMode::DraggingArrow { axis, positive, .. }
            if axis == handle.axis && positive == handle.positive
        );
        let target_stretch = if is_this_arrow_dragged {
            let dir_sign = if handle.positive { 1.0 } else { -1.0 };
            let drag_rate_sign = if let InteractionMode::DraggingArrow { rate_sign, .. } = *mode {
                rate_sign
            } else {
                0.0
            };
            ARROW_STRETCH * drag_rate_sign * dir_sign
        } else {
            0.0
        };

        let polarity_idx = if handle.positive { 0 } else { 1 };
        let current = &mut stretch_state.current[handle.axis][polarity_idx];
        let dt = time.delta_secs();
        *current += (target_stretch - *current) * (STRETCH_LERP_SPEED * dt).min(1.0);
        let stretch = *current;

        let base = world_pos + dir * sign * sphere_gap;
        let midpoint = base + dir * sign * (arrow_len * 0.5 + stretch * s * 0.5);
        let rotation = Quat::from_rotation_arc(Vec3::Y, dir * sign);
        *transform = Transform {
            translation: midpoint,
            rotation,
            scale: Vec3::splat(s),
        };

        if let Some(children) = children {
            let shaft_len = BASE_ARROW_LEN + stretch;
            for child in children.iter() {
                shaft_updates.push((child, shaft_len));
                cone_updates.push((child, shaft_len));
            }
        }
    }

    {
        let mut shafts = children_set.p0();
        for (entity, shaft_len) in &shaft_updates {
            if let Ok(mut tf) = shafts.get_mut(*entity) {
                tf.scale = Vec3::new(1.0, *shaft_len, 1.0);
            }
        }
    }
    {
        let mut cones = children_set.p1();
        for (entity, shaft_len) in &cone_updates {
            if let Ok(mut tf) = cones.get_mut(*entity) {
                tf.translation.y = *shaft_len / 2.0;
            }
        }
    }
    {
        let mut spheres = children_set.p2();
        for (entity, mut transform, sphere_visual) in &mut spheres {
            let is_hovered = hovered_entities.contains(&entity);
            if let Some(mat) = material_assets.get_mut(&sphere_visual.material) {
                if is_hovered {
                    mat.base_color = Color::srgb(1.0, 1.0, 0.5);
                    mat.emissive = LinearRgba::from(Color::srgb(0.8, 0.7, 0.2)).into();
                } else {
                    mat.base_color = Color::srgb(1.0, 0.8, 0.0);
                    mat.emissive = LinearRgba::from(Color::srgb(0.5, 0.4, 0.0)).into();
                }
            }
            *transform = Transform {
                translation: world_pos,
                scale: Vec3::splat(s),
                ..default()
            };
        }
    }
}
