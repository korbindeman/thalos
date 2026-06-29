//! Body hierarchy popup for the map view (native Bevy UI).
//!
//! Shows every body in `SolarSystemDefinition` as an indented tree
//! (Star → Planets → Moons), with a separate "Minor bodies" collapsing group
//! for dwarf planets, centaurs, and comets. Clicking a row focuses the map
//! camera on that body, mirroring the smooth-transition pattern used by
//! `double_click_focus_system` in `rendering`.
//!
//! The row set is rebuilt only when its structure changes (the ship's SOI body,
//! the debug flag, or the minor-group collapse); focus highlighting and the
//! debug drop/aim state are per-frame visual updates. Debug cmd-click teleports
//! a body to low orbit; the per-row drop button arms the surface cursor.

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;
use std::collections::HashMap;
use thalos_physics_canonical::canonical::{AuthorityMode, Epoch};
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::ActiveLocalBubble;
use thalos_world::{BodyDefinition, BodyId, BodyKind};

use crate::camera::{CameraFocus, CameraFocusTarget};
use crate::debug::{DebugMode, DebugSurfaceTeleport, low_orbit_state};
use crate::hud::theme::{HudTheme, panel_frame};
use crate::maneuver::{ManeuverPlan, SelectedNode};
use crate::pause_menu::GamePause;
use crate::photo_mode::PhotoMode;
use crate::player_controller::EvaMode;
use crate::rendering::{CelestialBody, RenderOrigin, ShipMarker, SimulationState};
use crate::scenario_menu::ScenarioMenu;
use crate::shipyard_editor::ShipyardEditor;
use crate::ui_widgets::{ScrollableColumn, UiButton};
use crate::view::ViewMode;

/// Default framing distance (metres) when the ship row is clicked. Matches the
/// value `sync_view_mode_changed` uses when entering map view.
const SHIP_TREE_FOCUS_DISTANCE_M: f64 = 2.0e7;

const INDENT_PX: f32 = 14.0;

// ── Resources / markers ─────────────────────────────────────────────────────────

#[derive(Resource)]
struct BodyTreeState {
    collapsed_minor: bool,
}

impl Default for BodyTreeState {
    fn default() -> Self {
        Self {
            collapsed_minor: true,
        }
    }
}

#[derive(Component)]
struct BodyTreePanelRoot;

#[derive(Component)]
struct BodyTreeContent;

#[derive(Component, Clone, Copy)]
struct BodyRowButton {
    body_id: BodyId,
}

#[derive(Component)]
struct ShipRowButton;

#[derive(Component, Clone, Copy)]
struct DropButton {
    body_id: BodyId,
}

#[derive(Component)]
struct MinorToggle;

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct BodyTreePanelPlugin;

impl Plugin for BodyTreePanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BodyTreeState>()
            .add_systems(Startup, setup_ui.after(crate::hud::theme::init_theme))
            .add_systems(
                Update,
                (
                    update_visibility,
                    rebuild_body_tree,
                    update_row_visuals,
                    handle_minor_toggle,
                    handle_tree_clicks,
                ),
            );
    }
}

fn setup_ui(mut commands: Commands, theme: Res<HudTheme>) {
    let (bg, border) = panel_frame(&theme);
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(12.0),
                top: Val::Px(56.0),
                width: Val::Px(212.0),
                max_height: Val::Percent(70.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            bg,
            border,
            Visibility::Hidden,
            BodyTreePanelRoot,
            Name::new("BodyTreePanel"),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("CELESTIAL BODIES"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 10.0,
                    ..default()
                },
                TextColor(theme.text_subtitle),
            ));
            panel.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(2.0),
                    overflow: Overflow::scroll_y(),
                    flex_grow: 1.0,
                    ..default()
                },
                ScrollPosition::default(),
                RelativeCursorPosition::default(),
                Interaction::None,
                ScrollableColumn,
                BodyTreeContent,
                Name::new("BodyTreeContent"),
            ));
        });
}

// ── Visibility ──────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn update_visibility(
    view: Res<ViewMode>,
    pause: Res<GamePause>,
    scenario: Res<ScenarioMenu>,
    photo: Res<PhotoMode>,
    shipyard: Option<Res<ShipyardEditor>>,
    mut roots: Query<&mut Visibility, With<BodyTreePanelRoot>>,
) {
    let editor_open = shipyard.as_deref().map(|e| e.open).unwrap_or(false);
    let visible = *view == ViewMode::Map
        && !pause.active
        && !scenario.open
        && !photo.active
        && !editor_open;
    let target = if visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut roots {
        if *vis != target {
            *vis = target;
        }
    }
}

// ── Rebuild ───────────────────────────────────────────────────────────────────

fn rebuild_body_tree(
    mut commands: Commands,
    sim: Option<Res<SimulationState>>,
    state: Res<BodyTreeState>,
    debug: Res<DebugMode>,
    theme: Res<HudTheme>,
    ship_marker: Query<&Name, With<ShipMarker>>,
    content: Query<(Entity, Option<&Children>), With<BodyTreeContent>>,
    mut shown: Local<Option<(BodyId, bool, bool, usize)>>,
) {
    let Some(sim) = sim else {
        return;
    };
    let soi = sim.simulation.dominant_body();
    let key = (
        soi,
        debug.enabled,
        state.collapsed_minor,
        sim.system.bodies.len(),
    );
    if *shown == Some(key) {
        return;
    }
    *shown = Some(key);

    let Ok((content_entity, children)) = content.single() else {
        return;
    };
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }

    let system = &sim.system;
    let Some(root) = system.bodies.iter().find(|b| b.parent.is_none()) else {
        return;
    };

    let mut children_of: HashMap<BodyId, Vec<&BodyDefinition>> = HashMap::new();
    for body in &system.bodies {
        if let Some(parent) = body.parent {
            children_of.entry(parent).or_default().push(body);
        }
    }
    for kids in children_of.values_mut() {
        kids.sort_by_key(|b| b.id);
    }

    let ship: Option<&str> = ship_marker.single().ok().map(|n| n.as_str());
    let debug_enabled = debug.enabled;
    let collapsed = state.collapsed_minor;
    let theme = theme.clone();

    commands.entity(content_entity).with_children(|c| {
        // Major tree.
        spawn_body_row(c, &theme, root, debug_enabled, 0);
        if root.id == soi
            && let Some(name) = ship
        {
            spawn_ship_row(c, &theme, name, 1);
        }
        if let Some(kids) = children_of.get(&root.id) {
            for child in kids.iter().filter(|b| !is_minor(b.kind)) {
                build_subtree(c, &theme, child, &children_of, debug_enabled, 1, soi, ship);
            }
        }

        // Minor bodies.
        let minor: Vec<&BodyDefinition> = children_of
            .get(&root.id)
            .map(|kids| kids.iter().copied().filter(|b| is_minor(b.kind)).collect())
            .unwrap_or_default();
        if !minor.is_empty() {
            spawn_minor_toggle(c, &theme, collapsed);
            if !collapsed {
                for body in minor {
                    build_subtree(c, &theme, body, &children_of, debug_enabled, 1, soi, ship);
                }
            }
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn build_subtree(
    c: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    body: &BodyDefinition,
    children_of: &HashMap<BodyId, Vec<&BodyDefinition>>,
    debug_enabled: bool,
    depth: u32,
    soi: BodyId,
    ship: Option<&str>,
) {
    spawn_body_row(c, theme, body, debug_enabled, depth);
    if body.id == soi
        && let Some(name) = ship
    {
        spawn_ship_row(c, theme, name, depth + 1);
    }
    if let Some(kids) = children_of.get(&body.id) {
        for child in kids {
            build_subtree(c, theme, child, children_of, debug_enabled, depth + 1, soi, ship);
        }
    }
}

fn spawn_body_row(
    c: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    body: &BodyDefinition,
    debug_enabled: bool,
    depth: u32,
) {
    let [r, g, b] = body.color;
    let dot = Color::srgb(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0));
    let body_id = body.id;
    let name = body.name.clone();
    let show_drop = debug_enabled && !matches!(body.kind, BodyKind::Star);

    c.spawn(row_node(depth)).with_children(|row| {
        spawn_dot(row, dot);
        // Frameless selectable name button; colour driven by `update_row_visuals`.
        row.spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(2.0), Val::Px(1.0)),
                ..default()
            },
            Interaction::None,
            BodyRowButton { body_id },
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(name),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
        });
        if show_drop {
            spawn_drop_button(row, theme, body_id);
        }
    });
}

/// Small bordered debug button (`drop`/`aim`). Like `ui_widgets::spawn_button`
/// but its label carries [`DropLabel`] so `update_row_visuals` can swap the
/// text; `style_buttons` still owns its colour via [`UiButton`].
fn spawn_drop_button(row: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, body_id: BodyId) {
    row.spawn((
        Button,
        Node {
            height: Val::Px(18.0),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(3.0)),
            padding: UiRect::axes(Val::Px(6.0), Val::Px(1.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(theme.panel_bg),
        BorderColor::all(theme.panel_border),
        Interaction::None,
        UiButton::default(),
        DropButton { body_id },
    ))
    .with_children(|c| {
        c.spawn((
            Text::new("drop"),
            TextFont {
                font: theme.font.clone(),
                font_size: 9.0,
                ..default()
            },
            TextColor(theme.text_primary),
            DropLabel,
        ));
    });
}

fn spawn_ship_row(c: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, name: &str, depth: u32) {
    c.spawn(row_node(depth)).with_children(|row| {
        spawn_dot(row, Color::WHITE);
        row.spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(2.0), Val::Px(1.0)),
                ..default()
            },
            Interaction::None,
            ShipRowButton,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(name.to_string()),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
        });
    });
}

fn spawn_minor_toggle(c: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, collapsed: bool) {
    let glyph = if collapsed { "▸" } else { "▾" };
    c.spawn((
        Button,
        Node {
            padding: UiRect::axes(Val::Px(2.0), Val::Px(2.0)),
            margin: UiRect::top(Val::Px(2.0)),
            ..default()
        },
        Interaction::None,
        MinorToggle,
    ))
    .with_children(|b| {
        b.spawn((
            Text::new(format!("{glyph} Minor bodies")),
            TextFont {
                font: theme.font.clone(),
                font_size: 11.0,
                ..default()
            },
            TextColor(theme.text_dim),
        ));
    });
}

fn row_node(depth: u32) -> Node {
    Node {
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        column_gap: Val::Px(4.0),
        padding: UiRect::left(Val::Px(depth as f32 * INDENT_PX)),
        ..default()
    }
}

fn spawn_dot(row: &mut ChildSpawnerCommands<'_>, color: Color) {
    row.spawn((
        Node {
            width: Val::Px(8.0),
            height: Val::Px(8.0),
            border_radius: BorderRadius::all(Val::Percent(50.0)),
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(color),
    ));
}

// ── Per-frame visuals ───────────────────────────────────────────────────────────

#[allow(clippy::type_complexity)]
fn update_row_visuals(
    focus: Res<CameraFocus>,
    teleport: Res<DebugSurfaceTeleport>,
    theme: Res<HudTheme>,
    body_btns: Query<(&BodyRowButton, &Children)>,
    ship_btns: Query<&Children, With<ShipRowButton>>,
    mut drop_btns: Query<(&DropButton, &mut UiButton, &Children)>,
    mut name_text: Query<&mut TextColor, Without<DropLabel>>,
    mut drop_text: Query<&mut Text, With<DropLabel>>,
) {
    for (button, children) in &body_btns {
        let focused = focus.target == CameraFocusTarget::Body(button.body_id);
        let color = if focused {
            theme.text_accent
        } else {
            theme.text_primary
        };
        if let Some(&child) = children.first()
            && let Ok(mut tc) = name_text.get_mut(child)
            && tc.0 != color
        {
            tc.0 = color;
        }
    }

    let ship_focused = focus.target == CameraFocusTarget::Ship;
    let ship_color = if ship_focused {
        theme.text_accent
    } else {
        theme.text_primary
    };
    for children in &ship_btns {
        if let Some(&child) = children.first()
            && let Ok(mut tc) = name_text.get_mut(child)
            && tc.0 != ship_color
        {
            tc.0 = ship_color;
        }
    }

    for (drop, mut button, children) in &mut drop_btns {
        let armed = teleport.armed_body == Some(drop.body_id);
        if button.latched != armed {
            button.latched = armed;
        }
        let label = if armed { "aim" } else { "drop" };
        if let Some(&child) = children.first()
            && let Ok(mut text) = drop_text.get_mut(child)
            && **text != *label
        {
            **text = label.to_string();
        }
    }
}

/// Marker on the drop-button label so `update_row_visuals` can disambiguate it
/// from the name-button labels (whose colour it owns; the drop label's colour
/// is owned by `ui_widgets::style_buttons` via [`UiButton`]).
#[derive(Component)]
struct DropLabel;

// ── Interaction ───────────────────────────────────────────────────────────────

fn handle_minor_toggle(
    interactions: Query<&Interaction, (Changed<Interaction>, With<MinorToggle>)>,
    mut state: ResMut<BodyTreeState>,
) {
    for interaction in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            state.collapsed_minor = !state.collapsed_minor;
        }
    }
}

type TreeInteractions<'w, 's> = (
    Query<'w, 's, (&'static Interaction, &'static BodyRowButton), Changed<Interaction>>,
    Query<'w, 's, &'static Interaction, (Changed<Interaction>, With<ShipRowButton>)>,
    Query<'w, 's, (&'static Interaction, &'static DropButton), Changed<Interaction>>,
);

#[allow(clippy::too_many_arguments)]
fn handle_tree_clicks(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    interactions: TreeInteractions,
    mut sim: ResMut<SimulationState>,
    mut focus: ResMut<CameraFocus>,
    origin: Res<RenderOrigin>,
    debug: Res<DebugMode>,
    mut surface_teleport: ResMut<DebugSurfaceTeleport>,
    mut active_bubble: Option<ResMut<ActiveLocalBubble>>,
    mut eva_mode: ResMut<EvaMode>,
    mut plan: ResMut<ManeuverPlan>,
    mut selected: ResMut<SelectedNode>,
    bodies: Query<(Entity, &CelestialBody, &Transform)>,
) {
    let (body_q, ship_q, drop_q) = interactions;
    let cmd = keys.any_pressed([
        KeyCode::ControlLeft,
        KeyCode::ControlRight,
        KeyCode::SuperLeft,
        KeyCode::SuperRight,
    ]);

    let body_entities: HashMap<BodyId, Entity> =
        bodies.iter().map(|(e, cb, _)| (cb.body_id, e)).collect();

    // Ship row → focus the ship.
    for interaction in &ship_q {
        if matches!(interaction, Interaction::Pressed)
            && focus.target != CameraFocusTarget::Ship
        {
            focus.focus_on_ship(origin.position);
            focus.target_distance = SHIP_TREE_FOCUS_DISTANCE_M;
        }
    }

    // Drop button → arm the surface cursor (debug).
    for (interaction, drop) in &drop_q {
        if !matches!(interaction, Interaction::Pressed) || !debug.enabled {
            continue;
        }
        if matches!(sim.system.bodies[drop.body_id].kind, BodyKind::Star) {
            warn!(
                "surface drop ignored for star {}",
                sim.system.bodies[drop.body_id].name
            );
        } else {
            surface_teleport.arm(drop.body_id);
        }
    }

    // Body name → focus (+ cmd-click teleport to low orbit, debug).
    for (interaction, button) in &body_q {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let body_id = button.body_id;

        if focus.target != CameraFocusTarget::Body(body_id)
            && let Some(&target_entity) = body_entities.get(&body_id)
        {
            focus.focus_on_body(body_id, origin.position);
            focus.frame_for_radius(sim.system.bodies[body_id].radius_m);

            // Aim from the lit side, biased above/right for a soft terminator.
            if let Some(root) = sim.system.bodies.iter().find(|b| b.parent.is_none())
                && let Some((_, _, sun_t)) =
                    bodies.iter().find(|(_, cb, _)| cb.body_id == root.id)
                && let Ok((_, _, target_t)) = bodies.get(target_entity)
            {
                const TILT_UP: f32 = 0.2;
                const TILT_RIGHT: f32 = 0.2;
                let sun_dir = (sun_t.translation - target_t.translation).normalize_or_zero();
                if sun_dir != Vec3::ZERO {
                    let camera_right = Vec3::Y.cross(sun_dir).normalize_or_zero();
                    let aim_dir =
                        (sun_dir + Vec3::Y * TILT_UP + camera_right * TILT_RIGHT).normalize();
                    focus.aim_from(aim_dir);
                }
            }
        }

        if debug.enabled && cmd {
            teleport_to_low_orbit(
                body_id,
                &mut commands,
                &mut sim,
                &mut active_bubble,
                &mut eva_mode,
                &mut plan,
                &mut selected,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn teleport_to_low_orbit(
    body_id: BodyId,
    commands: &mut Commands,
    sim: &mut SimulationState,
    active_bubble: &mut Option<ResMut<ActiveLocalBubble>>,
    eva_mode: &mut EvaMode,
    plan: &mut ManeuverPlan,
    selected: &mut SelectedNode,
) {
    let is_eva = sim.simulation.vessel_kind() == VesselKind::Eva;
    let sim_time = sim.simulation.sim_time();
    let body_state = sim.ephemeris.state(body_id, Epoch(sim_time));
    // EVA keeps its persistent bubble (in-place teleport); only ships tear down.
    if !is_eva {
        clear_active_local_bubble(commands, active_bubble);
    }
    let (state, attitude) = low_orbit_state(&sim.system.bodies[body_id], &body_state);
    sim.simulation
        .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
    sim.simulation.set_ship_state(state);
    sim.simulation.set_attitude(attitude);
    // A fresh craft — clear any structural failure so a wreck can be recovered.
    sim.simulation.repair();
    sim.simulation.warp.reset();
    if is_eva {
        *eva_mode = EvaMode::Airborne;
    }
    clear_debug_teleport_maneuvers(plan, selected);
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
