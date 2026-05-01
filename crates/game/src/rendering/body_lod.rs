//! Screen-space LOD for celestial bodies: crossfading between the
//! impostor mesh and the icon billboard, hiding moons that have merged
//! with their parent's icon, and double-click-to-focus picking. Also
//! holds the homeworld-focus startup system since it shares the
//! camera-focus code path.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::EguiContexts;
use thalos_physics::types::BodyKind;

use super::screen_marker_radius;
use super::types::{BodyIcon, BodyMesh, CelestialBody, ShipMarker, SimulationState};
use crate::camera::{ActiveCamera, CameraFocus, CameraFocusTarget, OrbitCamera};
use crate::coords::{RenderGhostFocus, RenderOrigin, WorldScale};
use crate::flight_plan_view::GhostBody;

/// Tracks the last left-click time and screen position for double-click detection.
#[derive(Resource, Default)]
pub(super) struct LastClick {
    time: f64,
    position: Vec2,
}

const DOUBLE_CLICK_THRESHOLD: f64 = 0.4; // seconds
const DOUBLE_CLICK_RADIUS: f32 = 10.0; // pixels — tolerance for cursor drift between clicks

type IconFilter = (With<BodyIcon>, Without<CelestialBody>, Without<OrbitCamera>);

/// Width of the crossfade window as a multiple of `icon_radius`. When the
/// body's render radius is between `icon_radius` and `icon_radius * (1 +
/// ICON_FADE_WIDTH)`, the icon alpha smoothly ramps from 0 to 1 while the
/// impostor mesh stays on top. Below `icon_radius` the mesh is hidden and
/// the icon is already fully opaque, so the swap is invisible.
const ICON_FADE_WIDTH: f32 = 0.25;

/// Multiple of `focus.distance` beyond which a body's billboard is hidden.
/// Bodies farther than this from the focus target (in render units) are
/// considered "out of the current neighborhood" — e.g. when framing Thalos
/// at ~20,000 km zoom, Auron is tens of millions of km away and its
/// billboard would just add clutter.
const BILLBOARD_NEIGHBORHOOD: f32 = 30.0;

/// Width of the moon-vs-parent merge fade as a multiple of the combined
/// icon radii. Moon alpha reaches 0 when its world-space separation from
/// the parent drops below `parent_icon_r + moon_icon_r` (i.e. the two icon
/// discs overlap and the moon can no longer be clicked separately), and
/// reaches 1 at `(1 + SEPARATION_FADE_WIDTH)` times that threshold.
const SEPARATION_FADE_WIDTH: f32 = 1.5;

type MeshFilter = (
    With<BodyMesh>,
    Without<BodyIcon>,
    Without<CelestialBody>,
    Without<OrbitCamera>,
);

/// Alpha ramp for moon billboards based on angular separation from parent.
/// Returns 1.0 when moon is clearly separable, 0.0 when icons fully merged.
fn moon_separation_alpha(moon_pos: Vec3, parent_pos: Vec3, cam_pos: Vec3) -> f32 {
    let moon_r = screen_marker_radius(moon_pos, cam_pos);
    let parent_r = screen_marker_radius(parent_pos, cam_pos);
    let merged = moon_r + parent_r;
    let fade_end = merged * (1.0 + SEPARATION_FADE_WIDTH);
    let sep = (moon_pos - parent_pos).length();
    ((sep - merged) / (fade_end - merged)).clamp(0.0, 1.0)
}

pub(super) fn sync_body_icons(
    bodies: Query<(Entity, &CelestialBody, &Transform, &Children)>,
    sim: Res<SimulationState>,
    focus: Res<CameraFocus>,
    scale: Res<WorldScale>,
    photo_mode: Res<crate::photo_mode::PhotoMode>,
    camera_query: Query<&Transform, (With<ActiveCamera>, With<OrbitCamera>)>,
    ghosts: Query<(&GhostBody, &Transform), Without<BodyIcon>>,
    mut icons: Query<
        (
            &mut Transform,
            &mut Visibility,
            &MeshMaterial3d<StandardMaterial>,
        ),
        IconFilter,
    >,
    mut meshes: Query<&mut Visibility, MeshFilter>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
) {
    let Ok(cam_tf) = camera_query.single() else {
        return;
    };

    let cam_rotation = cam_tf.rotation;
    let cam_pos = cam_tf.translation;
    let body_defs = sim.simulation.bodies();

    // Billboard neighborhood: anything farther from the focus target than
    // `focus.distance * BILLBOARD_NEIGHBORHOOD` (all in render units) is
    // outside the current view's scope and gets hidden.
    let focus_pos = match focus.target {
        CameraFocusTarget::Body(focus_body_id) => bodies
            .iter()
            .find(|(_, body, _, _)| body.body_id == focus_body_id)
            .map(|(_, _, tf, _)| tf.translation),
        CameraFocusTarget::Ghost(ghost_focus) => ghosts
            .iter()
            .find(|(ghost, _)| ghost_focus.matches(ghost.body_id, ghost.encounter_epoch))
            .map(|(_, tf)| tf.translation),
        _ => None,
    };
    let neighborhood_radius = (focus.distance * scale.0) as f32 * BILLBOARD_NEIGHBORHOOD;

    for (_, body, body_tf, children) in &bodies {
        let icon_radius = screen_marker_radius(body_tf.translation, cam_pos);
        let is_focus =
            matches!(focus.target, CameraFocusTarget::Body(body_id) if body_id == body.body_id);

        // Fade moons when their icon disc overlaps the parent's: at that
        // point the user can no longer click the moon separately from the
        // parent, so rendering it adds clutter. Focus is exempt so zooming
        // out while focused on a moon doesn't hide the focus target.
        let mut hidden = false;
        let mut separation_alpha = 1.0f32;
        if !is_focus
            && matches!(body_defs[body.body_id].kind, BodyKind::Moon)
            && let Some(parent_id) = body_defs[body.body_id].parent
            && let Some((_, _, parent_tf, _)) =
                bodies.iter().find(|(_, b, _, _)| b.body_id == parent_id)
        {
            separation_alpha =
                moon_separation_alpha(body_tf.translation, parent_tf.translation, cam_pos);
            if separation_alpha <= 0.0 {
                hidden = true;
            }
        }

        // Hide distant billboards: if this body is far outside the current
        // focus neighborhood AND would only render as a billboard anyway
        // (impostor radius below icon size), drop it entirely. Bodies that
        // still resolve as a real impostor mesh stay visible — otherwise
        // zooming in on a moon would make its huge parent disappear.
        if !is_focus
            && !body.is_star
            && body.render_radius < icon_radius
            && let Some(fp) = focus_pos
            && (body_tf.translation - fp).length() > neighborhood_radius
        {
            hidden = true;
        }

        // Icon alpha ramps from 0 at `render_radius >= (1 + WIDTH) *
        // icon_radius` to 1 at `render_radius <= icon_radius`. The impostor
        // stays visible throughout the fade window so the two layers
        // crossfade instead of popping.
        let fade_start = icon_radius * (1.0 + ICON_FADE_WIDTH);
        let icon_alpha = if body.is_star || hidden {
            0.0
        } else {
            ((fade_start - body.render_radius) / (fade_start - icon_radius)).clamp(0.0, 1.0)
                * separation_alpha
        };
        let show_icon = !hidden && icon_alpha > 0.0 && !photo_mode.active;
        let show_mesh = !hidden && (body.is_star || body.render_radius >= icon_radius);

        for child in children.iter() {
            if let Ok((mut icon_tf, mut icon_vis, mat_handle)) = icons.get_mut(child) {
                let target_vis = if show_icon {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                };
                if *icon_vis != target_vis {
                    *icon_vis = target_vis;
                }
                if show_icon {
                    if icon_tf.rotation != cam_rotation {
                        icon_tf.rotation = cam_rotation;
                    }
                    let target_scale = Vec3::splat(icon_radius);
                    if icon_tf.scale != target_scale {
                        icon_tf.scale = target_scale;
                    }
                    if let Some(mat) = std_materials.get_mut(&mat_handle.0) {
                        let current = mat.base_color.alpha();
                        if (current - icon_alpha).abs() > 1e-3 {
                            mat.base_color.set_alpha(icon_alpha);
                            // Emissive ignores material alpha in the forward
                            // shader, so scale rgb directly to fade glow.
                            let lin = mat.base_color.to_linear();
                            mat.emissive = LinearRgba::new(lin.red, lin.green, lin.blue, 1.0)
                                * 2.0
                                * icon_alpha;
                        }
                    }
                }
            }
            if let Ok(mut mesh_vis) = meshes.get_mut(child) {
                let target_vis = if show_mesh {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                };
                if *mesh_vis != target_vis {
                    *mesh_vis = target_vis;
                }
            }
        }
    }
}

pub(super) fn focus_camera_on_homeworld(
    mut focus: ResMut<CameraFocus>,
    bodies: Query<(Entity, &CelestialBody, &Name)>,
) {
    // Find "Thalos" (homeworld) entity and focus the camera on it.
    for (_, body, name) in &bodies {
        if name.as_str() == "Thalos" {
            focus.target = CameraFocusTarget::Body(body.body_id);
            focus.distance = 2e7; // 20,000 km — close enough to see the planet
            focus.target_distance = 2e7;
            return;
        }
    }
}

/// Detects double-clicks and focuses the camera on the nearest body by
/// projecting every body's world position to screen space and picking the
/// closest one to the cursor.  Works for both 3D sphere meshes and billboard
/// icons because we test the parent entity's transform, not the mesh child.
pub(super) fn double_click_focus_system(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut contexts: EguiContexts,
    time: Res<Time>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), (With<ActiveCamera>, With<OrbitCamera>)>,
    bodies: Query<(Entity, &CelestialBody, &Transform)>,
    ghosts: Query<
        (
            Entity,
            &crate::flight_plan_view::GhostBody,
            &Transform,
            &Visibility,
        ),
        Without<CelestialBody>,
    >,
    ship_marker: Query<
        (Entity, &Transform, &Visibility),
        (
            With<ShipMarker>,
            Without<CelestialBody>,
            Without<crate::flight_plan_view::GhostBody>,
        ),
    >,
    sim: Res<SimulationState>,
    scale: Res<WorldScale>,
    origin: Res<RenderOrigin>,
    mut focus: ResMut<CameraFocus>,
    mut last_click: ResMut<LastClick>,
) {
    let focus_target = focus.target;
    let focus_render_dist = (focus.distance * scale.0) as f32;
    let neighborhood_radius = focus_render_dist * BILLBOARD_NEIGHBORHOOD;
    let focus_pos = match focus_target {
        CameraFocusTarget::Body(focus_body_id) => bodies
            .iter()
            .find(|(_, body, _)| body.body_id == focus_body_id)
            .map(|(_, _, t)| t.translation),
        CameraFocusTarget::Ghost(ghost_focus) => ghosts
            .iter()
            .find(|(_, ghost, _, _)| ghost_focus.matches(ghost.body_id, ghost.encounter_epoch))
            .map(|(_, _, transform, _)| transform.translation),
        _ => None,
    };
    if !mouse_buttons.just_pressed(MouseButton::Left) {
        return;
    }
    // Skip clicks consumed by egui — otherwise clicking a button or
    // dragging a window would also pick a body behind it.
    if contexts
        .ctx_mut()
        .map(|ctx| ctx.wants_pointer_input())
        .unwrap_or(false)
    {
        return;
    }

    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_gt)) = camera_q.single() else {
        return;
    };

    let now = time.elapsed_secs_f64();
    let is_double = (now - last_click.time) < DOUBLE_CLICK_THRESHOLD
        && cursor_pos.distance(last_click.position) < DOUBLE_CLICK_RADIUS;

    last_click.time = now;
    last_click.position = cursor_pos;

    if !is_double {
        return;
    }

    // Reset so a third click doesn't trigger another double-click.
    last_click.time = 0.0;

    // Two-pass pick:
    //   1. Bodies currently rendered as visible billboards (icon alpha > 0,
    //      not hidden as a faraway moon) take priority. Among those, the one
    //      closest to the camera wins — a visible billboard is always the
    //      topmost selection target for any cursor inside its disc.
    //   2. Otherwise, fall back to the nearest-center hit across all bodies.
    let body_defs = sim.simulation.bodies();
    let viewport_height = window.height();
    let fov = std::f32::consts::FRAC_PI_4;
    let half_fov_tan = (fov / 2.0).tan();
    let cam_world = cam_gt.translation();

    let fade_end_factor = 1.0 + ICON_FADE_WIDTH;

    let mut billboard_best: Option<(CameraFocusTarget, f32)> = None; // (target, cam_dist)
    let mut fallback_best: Option<(CameraFocusTarget, f32)> = None; // (target, cursor_dist)

    for (_, body, transform) in &bodies {
        let Ok(screen) = camera.world_to_viewport(cam_gt, transform.translation) else {
            continue;
        };
        let cam_dist = cam_world.distance(transform.translation);
        if cam_dist <= 0.0 {
            continue;
        }
        let pixels_per_unit = viewport_height / (2.0 * half_fov_tan * cam_dist);
        let icon_radius_world = screen_marker_radius(transform.translation, cam_world);
        let icon_radius_px = icon_radius_world * pixels_per_unit;
        let sphere_radius_px = body.render_radius * pixels_per_unit;

        // Same visibility rules as `sync_body_icons`: moons fade out as
        // their icon disc merges with the parent's, and far-away bodies
        // outside the focus neighborhood get dropped. A moon that's fully
        // merged is neither visible nor clickable separately.
        let body_target = CameraFocusTarget::Body(body.body_id);

        if focus_target != body_target
            && matches!(body_defs[body.body_id].kind, BodyKind::Moon)
            && let Some(parent_id) = body_defs[body.body_id].parent
            && let Some((_, _, parent_tf)) = bodies.iter().find(|(_, b, _)| b.body_id == parent_id)
            && moon_separation_alpha(transform.translation, parent_tf.translation, cam_world) <= 0.0
        {
            continue;
        }

        // Also skip bodies hidden by the distant-billboard rule (matches
        // `sync_body_icons`): only drop when the body is billboard-sized.
        if focus_target != body_target
            && !body.is_star
            && body.render_radius < icon_radius_world
            && let Some(fp) = focus_pos
            && (transform.translation - fp).length() > neighborhood_radius
        {
            continue;
        }

        let is_visible_billboard =
            !body.is_star && body.render_radius < icon_radius_world * fade_end_factor;

        let dist_px = screen.distance(cursor_pos);

        if is_visible_billboard && dist_px <= icon_radius_px {
            if billboard_best.map(|(_, d)| cam_dist < d).unwrap_or(true) {
                billboard_best = Some((body_target, cam_dist));
            }
            continue;
        }

        // Fallback pass: mesh-sized hit circle for bodies not acting as a
        // billboard, plus a 12px minimum so tiny dots stay clickable.
        let hit_radius = sphere_radius_px.max(icon_radius_px).max(12.0);
        if dist_px > hit_radius {
            continue;
        }
        if fallback_best.map(|(_, d)| dist_px < d).unwrap_or(true) {
            fallback_best = Some((body_target, dist_px));
        }
    }

    // Also check ghost bodies (translucent encounter previews).
    for (_, ghost, transform, visibility) in &ghosts {
        if *visibility == Visibility::Hidden {
            continue;
        }
        let Ok(screen) = camera.world_to_viewport(cam_gt, transform.translation) else {
            continue;
        };
        let dist_px = screen.distance(cursor_pos);
        // Ghost bodies are screen-size-stable, use generous hit radius.
        if dist_px > 20.0 {
            continue;
        }
        if fallback_best.map(|(_, d)| dist_px < d).unwrap_or(true) {
            fallback_best = Some((
                CameraFocusTarget::Ghost(RenderGhostFocus {
                    body_id: ghost.body_id,
                    parent_id: ghost.parent_id,
                    relative_position: ghost.relative_position,
                    projection_epoch: ghost.projection_epoch,
                    encounter_epoch: ghost.encounter_epoch,
                }),
                dist_px,
            ));
        }
    }

    // Ship marker: always a screen-stable billboard in map view. Treated
    // as a billboard so it competes with body billboards by camera
    // distance (a body in front of the ship still wins). 12 px floor
    // mirrors the bodies' fallback hit so the small icon stays clickable.
    if let Ok((_, ship_transform, ship_visibility)) = ship_marker.single()
        && *ship_visibility != Visibility::Hidden
    {
        let cam_dist = cam_world.distance(ship_transform.translation);
        if cam_dist > 0.0
            && let Ok(screen) = camera.world_to_viewport(cam_gt, ship_transform.translation)
        {
            let pixels_per_unit = viewport_height / (2.0 * half_fov_tan * cam_dist);
            let icon_radius_world = screen_marker_radius(ship_transform.translation, cam_world);
            let icon_radius_px = icon_radius_world * pixels_per_unit;
            let hit_radius_px = icon_radius_px.max(12.0);
            if screen.distance(cursor_pos) <= hit_radius_px
                && billboard_best.map(|(_, d)| cam_dist < d).unwrap_or(true)
            {
                billboard_best = Some((CameraFocusTarget::Ship, cam_dist));
            }
        }
    }

    let Some((target, _)) = billboard_best.or(fallback_best) else {
        return;
    };

    focus.focus_on(target, origin.position);
}
