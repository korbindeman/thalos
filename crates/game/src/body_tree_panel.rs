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
use thalos_physics_canonical::canonical::AuthorityMode;
use thalos_world::{BodyDefinition, BodyId, BodyKind};
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::ActiveLocalBubble;

use crate::camera::{CameraFocus, CameraFocusTarget};
use crate::debug::{DebugLaunchMount, DebugMode, DebugSurfaceTeleport, low_orbit_state};
use crate::maneuver::{ManeuverPlan, SelectedNode};
use crate::pause_menu::not_game_paused;
use crate::photo_mode::not_in_photo_mode;
use crate::player_controller::EvaMode;
use crate::rendering::{CelestialBody, RenderOrigin, ShipMarker, SimulationState};
use crate::view::in_map_view;

/// Default framing distance (metres) when the ship row is clicked. Matches
/// the value `sync_view_mode_changed` uses when entering map view, so
/// picking the ship from the tree lands at the same zoom as switching to
/// map view from ship view.
const SHIP_TREE_FOCUS_DISTANCE_M: f64 = 2.0e7;

pub struct BodyTreePanelPlugin;

impl Plugin for BodyTreePanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            bevy_egui::EguiPrimaryContextPass,
            body_tree_panel.run_if(not_game_paused.and(not_in_photo_mode).and(in_map_view)),
        );
    }
}

fn body_tree_panel(
    mut commands: Commands,
    mut contexts: EguiContexts,
    mut sim: ResMut<SimulationState>,
    bodies: Query<(Entity, &CelestialBody, &Transform)>,
    ship_marker: Query<&Name, With<ShipMarker>>,
    origin: Res<RenderOrigin>,
    mut focus: ResMut<CameraFocus>,
    debug: Res<DebugMode>,
    mut surface_teleport: ResMut<DebugSurfaceTeleport>,
    mut active_bubble: Option<ResMut<ActiveLocalBubble>>,
    mut launch_mount: ResMut<DebugLaunchMount>,
    mut eva_mode: ResMut<EvaMode>,
    mut plan: ResMut<ManeuverPlan>,
    mut selected: ResMut<SelectedNode>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let body_entities: HashMap<BodyId, Entity> =
        bodies.iter().map(|(e, cb, _)| (cb.body_id, e)).collect();

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

    // Anchor the ship row under whichever body's SOI currently contains
    // the ship — same rule the propagator uses, so the tree position
    // tracks SOI transitions live.
    let soi_id = sim.simulation.dominant_body();
    let ship: Option<&str> = ship_marker.single().ok().map(|n| n.as_str());

    let mut clicked: Option<BodyId> = None;
    let mut cmd_clicked: Option<BodyId> = None;
    let mut drop_clicked: Option<BodyId> = None;
    let mut clicked_ship = false;

    let initial_pos = ctx.available_rect().left_top() + egui::vec2(8.0, 8.0);

    egui::Window::new("Celestial bodies")
        .default_pos(initial_pos)
        .resizable(false)
        .show(ctx, |ui| {
            ui.set_min_width(180.0);

            // Major tree: star and its non-minor descendants.
            render_row(
                ui,
                root,
                &body_entities,
                focus.target,
                debug.enabled,
                surface_teleport.armed_body,
                &mut clicked,
                &mut cmd_clicked,
                &mut drop_clicked,
                0,
            );
            if root.id == soi_id
                && let Some(name) = ship
            {
                render_ship_row(ui, name, focus.target, &mut clicked_ship, 1);
            }
            if let Some(kids) = children_of.get(&root.id) {
                for child in kids.iter().filter(|b| !is_minor(b.kind)) {
                    render_subtree(
                        ui,
                        child,
                        &children_of,
                        &body_entities,
                        focus.target,
                        debug.enabled,
                        surface_teleport.armed_body,
                        &mut clicked,
                        &mut cmd_clicked,
                        &mut drop_clicked,
                        1,
                        ship,
                        soi_id,
                        &mut clicked_ship,
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
                            debug.enabled,
                            surface_teleport.armed_body,
                            &mut clicked,
                            &mut cmd_clicked,
                            &mut drop_clicked,
                            0,
                            ship,
                            soi_id,
                            &mut clicked_ship,
                        );
                    }
                });
            }
        });

    if clicked_ship && ship.is_some() && focus.target != CameraFocusTarget::Ship {
        focus.focus_on_ship(origin.position);
        focus.target_distance = SHIP_TREE_FOCUS_DISTANCE_M;
    }

    if let Some(body_id) = clicked
        && let Some(&target_entity) = body_entities.get(&body_id)
        && focus.target != CameraFocusTarget::Body(body_id)
    {
        focus.focus_on_body(body_id, origin.position);
        focus.frame_for_radius(system.bodies[body_id].radius_m);

        // Aim from the lit side, biased slightly above and to the
        // camera's right so the body has a soft terminator instead of
        // looking flat (true full-phase). Skipped when the target is the
        // star itself (sun_dir collapses to zero).
        if let Some((_, _, sun_t)) = bodies.iter().find(|(_, cb, _)| cb.body_id == root.id)
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
                let aim_dir = (sun_dir + Vec3::Y * TILT_UP + camera_right * TILT_RIGHT).normalize();
                focus.aim_from(aim_dir);
            }
        }
    }

    if debug.enabled
        && let Some(body_id) = drop_clicked
    {
        if matches!(sim.system.bodies[body_id].kind, BodyKind::Star) {
            warn!(
                "surface drop ignored for star {}",
                sim.system.bodies[body_id].name
            );
        } else {
            surface_teleport.arm(body_id);
        }
    }

    // Debug: cmd-click teleports the craft to a low circular orbit. Surface
    // placement is now explicit via each row's `drop` button and cursor.
    if debug.enabled
        && let Some(body_id) = cmd_clicked
    {
        let is_eva = sim.simulation.vessel_kind() == VesselKind::Eva;
        let sim_time = sim.simulation.sim_time();
        let body_state = sim.ephemeris.state(
            body_id,
            thalos_physics_canonical::canonical::Epoch(sim_time),
        );
        // EVA keeps its persistent bubble (in-place teleport); only ships tear
        // down and respawn. Clearing the EVA bubble would despawn the capsule
        // and the next spawn would re-ground it on the surface.
        if !is_eva {
            clear_active_local_bubble(&mut commands, &mut active_bubble);
        }
        let (state, attitude) = low_orbit_state(&sim.system.bodies[body_id], &body_state);
        sim.simulation
            .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
        sim.simulation.set_ship_state(state);
        sim.simulation.set_attitude(attitude);
        // Teleporting to orbit hands the player a fresh craft — clear any
        // structural failure so a wreck can be recovered. See docs/landing.md.
        sim.simulation.repair();
        sim.simulation.warp.reset();
        launch_mount.active = None;
        // Airborne: Kepler owns translation, the canonical→Avian snap drives
        // the capsule, and `step_eva_controller` stands down.
        if is_eva {
            *eva_mode = EvaMode::Airborne;
        }

        clear_debug_teleport_maneuvers(&mut plan, &mut selected);
    }
}

fn clear_active_local_bubble(
    commands: &mut Commands,
    active_bubble: &mut Option<ResMut<ActiveLocalBubble>>,
) {
    let Some(active) = active_bubble.as_mut() else {
        return;
    };
    let Some(bubble) = active.bubble.take() else {
        return;
    };
    commands.entity(bubble.craft_entity).despawn();
    if let Some(terrain_entity) = bubble.terrain_entity {
        commands.entity(terrain_entity).despawn();
    }
}

fn clear_debug_teleport_maneuvers(plan: &mut ManeuverPlan, selected: &mut SelectedNode) {
    // The pre-teleport flight plan is meaningless once the ship jumps to a
    // new orbit or surface pose. The next bridge sync pushes the empty plan
    // into physics, which dirties prediction and rebuilds from the new state.
    if !plan.nodes.is_empty() {
        plan.nodes.clear();
        plan.dirty = true;
    }
    selected.id = None;
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
    focused_target: CameraFocusTarget,
    debug_enabled: bool,
    armed_drop_body: Option<BodyId>,
    clicked: &mut Option<BodyId>,
    cmd_clicked: &mut Option<BodyId>,
    drop_clicked: &mut Option<BodyId>,
    depth: u32,
    ship: Option<&str>,
    soi_id: BodyId,
    clicked_ship: &mut bool,
) {
    render_row(
        ui,
        body,
        body_entities,
        focused_target,
        debug_enabled,
        armed_drop_body,
        clicked,
        cmd_clicked,
        drop_clicked,
        depth,
    );
    if body.id == soi_id
        && let Some(name) = ship
    {
        render_ship_row(ui, name, focused_target, clicked_ship, depth + 1);
    }
    if let Some(kids) = children_of.get(&body.id) {
        for child in kids {
            render_subtree(
                ui,
                child,
                children_of,
                body_entities,
                focused_target,
                debug_enabled,
                armed_drop_body,
                clicked,
                cmd_clicked,
                drop_clicked,
                depth + 1,
                ship,
                soi_id,
                clicked_ship,
            );
        }
    }
}

/// Row rendering the player ship under its SOI body. Mirrors
/// [`render_row`]'s layout and selection styling, with a white marker
/// dot matching the in-world `ShipMarker` billboard.
fn render_ship_row(
    ui: &mut egui::Ui,
    name: &str,
    focused_target: CameraFocusTarget,
    clicked: &mut bool,
    depth: u32,
) {
    let is_focused = focused_target == CameraFocusTarget::Ship;

    ui.horizontal(|ui| {
        ui.add_space(depth as f32 * 14.0);

        let dot_color = egui::Color32::WHITE;
        let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 4.0, dot_color);
        ui.add_space(4.0);

        let label = ui.add(egui::Button::selectable(is_focused, name).frame(false));
        if label.clicked() {
            *clicked = true;
        }
    });
}

fn render_row(
    ui: &mut egui::Ui,
    body: &BodyDefinition,
    body_entities: &HashMap<BodyId, Entity>,
    focused_target: CameraFocusTarget,
    debug_enabled: bool,
    armed_drop_body: Option<BodyId>,
    clicked: &mut Option<BodyId>,
    cmd_clicked: &mut Option<BodyId>,
    drop_clicked: &mut Option<BodyId>,
    depth: u32,
) {
    let entity = body_entities.get(&body.id).copied();
    let is_focused = focused_target == CameraFocusTarget::Body(body.id);

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
            // `command` is cmd on macOS, ctrl on Windows/Linux — egui's
            // standard cross-platform "primary modifier." Cmd-click
            // teleports to low orbit (handled below) and focuses.
            *clicked = Some(body.id);
            let modifiers = ui.input(|i| i.modifiers);
            if modifiers.command {
                *cmd_clicked = Some(body.id);
            }
        }

        if debug_enabled && !matches!(body.kind, BodyKind::Star) {
            ui.add_space(4.0);
            let armed = armed_drop_body == Some(body.id);
            let text = if armed { "aim" } else { "drop" };
            let drop = ui
                .add_enabled(entity.is_some(), egui::Button::selectable(armed, text))
                .on_hover_text("Arm terrain cursor");
            if drop.clicked() {
                *clicked = Some(body.id);
                *drop_clicked = Some(body.id);
            }
        }
    });
}
