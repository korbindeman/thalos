//! Body hierarchy popup for the map view.
//!
//! Shows every body in `SolarSystemDefinition` as an indented tree
//! (Star → Planets → Moons), with a separate "Minor bodies" collapsing
//! group for dwarf planets, centaurs, and comets. Clicking a row focuses
//! the map camera on that body, mirroring the smooth-transition pattern
//! used by `double_click_focus_system` in `rendering.rs`.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use std::collections::HashMap;
use thalos_physics::types::{BodyDefinition, BodyId, BodyKind};

use crate::camera::CameraFocus;
use crate::hud::time_control_panel;
use crate::photo_mode::not_in_photo_mode;
use crate::rendering::{CelestialBody, RenderOrigin, SimulationState};
use crate::view::in_map_view;

pub struct BodyTreePanelPlugin;

impl Plugin for BodyTreePanelPlugin {
    fn build(&self, app: &mut App) {
        // Run after the HUD top panel so `ctx.available_rect()` already
        // excludes its docked area on the first frame — otherwise
        // `default_pos` (used only on first show) anchors the window
        // under the top bar permanently.
        app.add_systems(
            bevy_egui::EguiPrimaryContextPass,
            body_tree_panel
                .run_if(not_in_photo_mode.and(in_map_view))
                .after(time_control_panel),
        );
    }
}

fn body_tree_panel(
    mut contexts: EguiContexts,
    sim: Res<SimulationState>,
    bodies: Query<(Entity, &CelestialBody, &Transform)>,
    origin: Res<RenderOrigin>,
    mut focus: ResMut<CameraFocus>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let body_entities: HashMap<BodyId, Entity> = bodies
        .iter()
        .map(|(e, cb, _)| (cb.body_id, e))
        .collect();

    let system = &sim.system;
    let mut children_of: HashMap<BodyId, Vec<&BodyDefinition>> = HashMap::new();
    for body in &system.bodies {
        if let Some(parent) = body.parent {
            children_of.entry(parent).or_default().push(body);
        }
    }
    // Stable order: the file's listing order.
    for kids in children_of.values_mut() {
        kids.sort_by_key(|b| b.id);
    }

    let Some(root) = system.bodies.iter().find(|b| b.parent.is_none()) else {
        return;
    };

    let mut clicked: Option<BodyId> = None;

    let initial_pos = ctx.available_rect().left_top() + egui::vec2(8.0, 8.0);

    egui::Window::new("Celestial bodies")
        .default_pos(initial_pos)
        .resizable(false)
        .show(ctx, |ui| {
            ui.set_min_width(180.0);

            // Major tree: star and its non-minor descendants.
            render_row(ui, root, &body_entities, focus.target, &mut clicked, 0);
            if let Some(kids) = children_of.get(&root.id) {
                for child in kids.iter().filter(|b| !is_minor(b.kind)) {
                    render_subtree(
                        ui,
                        child,
                        &children_of,
                        &body_entities,
                        focus.target,
                        &mut clicked,
                        1,
                    );
                }
            }

            // Minor bodies: collapsing group of dwarf planets / centaurs /
            // comets that orbit the star, with their own descendants nested.
            let minor: Vec<&BodyDefinition> = children_of
                .get(&root.id)
                .map(|kids| kids.iter().copied().filter(|b| is_minor(b.kind)).collect())
                .unwrap_or_default();
            if !minor.is_empty() {
                ui.collapsing("Minor bodies", |ui| {
                    for body in minor {
                        render_subtree(
                            ui,
                            body,
                            &children_of,
                            &body_entities,
                            focus.target,
                            &mut clicked,
                            0,
                        );
                    }
                });
            }
        });

    if let Some(body_id) = clicked
        && let Some(&target_entity) = body_entities.get(&body_id)
        && focus.target != Some(target_entity)
    {
        focus.focus_on(target_entity, origin.position);
        focus.frame_for_radius(system.bodies[body_id].radius_m);

        // Aim from the lit side, biased slightly above and to the
        // camera's right so the body has a soft terminator instead of
        // looking flat (true full-phase). Skipped when the target is the
        // star itself (sun_dir collapses to zero).
        if let Some((_, _, sun_t)) =
            bodies.iter().find(|(_, cb, _)| cb.body_id == root.id)
            && let Ok((_, _, target_t)) = bodies.get(target_entity)
        {
            const TILT_UP: f32 = 0.2;
            const TILT_RIGHT: f32 = 0.2;
            let sun_dir = (sun_t.translation - target_t.translation).normalize_or_zero();
            if sun_dir != Vec3::ZERO {
                // Camera-right (world space) when sitting on the Sun side
                // and looking back at the target: `Y × sun_dir`. Falls
                // back to zero when the Sun is directly above/below the
                // target (degenerate); only the up-tilt applies then.
                let camera_right = Vec3::Y.cross(sun_dir).normalize_or_zero();
                let aim_dir =
                    (sun_dir + Vec3::Y * TILT_UP + camera_right * TILT_RIGHT).normalize();
                focus.aim_from(aim_dir);
            }
        }
    }
}

fn is_minor(kind: BodyKind) -> bool {
    matches!(
        kind,
        BodyKind::DwarfPlanet | BodyKind::Centaur | BodyKind::Comet
    )
}

fn render_subtree(
    ui: &mut egui::Ui,
    body: &BodyDefinition,
    children_of: &HashMap<BodyId, Vec<&BodyDefinition>>,
    body_entities: &HashMap<BodyId, Entity>,
    focused_entity: Option<Entity>,
    clicked: &mut Option<BodyId>,
    depth: u32,
) {
    render_row(ui, body, body_entities, focused_entity, clicked, depth);
    if let Some(kids) = children_of.get(&body.id) {
        for child in kids {
            render_subtree(
                ui,
                child,
                children_of,
                body_entities,
                focused_entity,
                clicked,
                depth + 1,
            );
        }
    }
}

fn render_row(
    ui: &mut egui::Ui,
    body: &BodyDefinition,
    body_entities: &HashMap<BodyId, Entity>,
    focused_entity: Option<Entity>,
    clicked: &mut Option<BodyId>,
    depth: u32,
) {
    let entity = body_entities.get(&body.id).copied();
    let is_focused = entity.is_some() && entity == focused_entity;

    ui.horizontal(|ui| {
        ui.add_space(depth as f32 * 14.0);

        let [r, g, b] = body.color;
        let dot_color = egui::Color32::from_rgb(
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
        );
        let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 4.0, dot_color);
        ui.add_space(4.0);

        let label = ui.add_enabled(
            entity.is_some(),
            egui::Button::selectable(is_focused, &body.name).frame(false),
        );
        if label.clicked() {
            *clicked = Some(body.id);
        }
    });
}
