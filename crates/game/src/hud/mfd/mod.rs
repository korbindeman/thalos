//! Multi-Function Display (MFD): a contextual, customizable HUD widget slot.
//!
//! The ship-view HUD has one top-right slot that hosts exactly one *widget*
//! at a time. By default the slot **auto-selects** the widget most relevant
//! to the current flight context (an orbital-trajectory plot in vacuum, an
//! airliner-style navigation display in atmosphere, …); a small tab row lets
//! the pilot **pin** a specific widget or hide the slot, an override that
//! persists until cleared. This replaces the old single hardcoded
//! "TRAJECTORY" panel (`system_map_panel`), whose show-gate popped an orbital
//! schematic up during atmospheric flight.
//!
//! ## Shape
//!
//! - One **slot container** (this module) owns the panel frame, the selector
//!   tab row, and a *widget area* the widget roots parent into.
//! - Each [`WidgetKind`] is a module under [`widgets`] exposing `build`
//!   (spawn its root + children under the area, hidden), `relevance`
//!   (a pure priority from [`FlightContext`]), and optionally an `update`
//!   system that refreshes its contents while active.
//! - [`select_active_widget`] is the single owner of per-widget + slot
//!   visibility. It resolves [`MfdSelection`] → an [`ActiveWidget`] and, in
//!   one pass, shows the chosen root and hides every other.
//!
//! ## Invariants
//!
//! - **`HudPanel` lives on the slot container only**, never on widget roots.
//!   `hide_in_photo_mode` flips every `HudPanel`'s visibility; container-only
//!   tagging means photo mode hides the whole slot through inheritance and
//!   the selector keeps sole ownership of which widget shows (no flashes).
//! - **One pass, one visible.** `select_active_widget` sets the chosen root
//!   `Inherited` and all others `Hidden` in a single system, so the
//!   one-widget invariant never races across systems. All visibility writes
//!   are diff-writes so they coexist with the photo-mode / editor writers.
//! - The MFD is **ship-view only** (the map view already draws the full 3D
//!   trajectory); in map view the slot is hidden.

pub mod widgets;

use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use thalos_physics_local::LocalCraftBody;

use crate::aero::AeroReadout;
use crate::fuel::ThrottleState;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::structures::{StructureKind, StructurePlacement, StructureRegistry, StructureSite};
use crate::view::ViewMode;

use super::HudPanel;
use super::theme::{HudTheme, panel_frame, panel_node};

/// Density (kg/m³) above which the craft counts as "in atmosphere" — the
/// same threshold the atmospheric readout panel uses.
const IN_ATMOSPHERE_DENSITY: f64 = 1.0e-6;
/// Keep `recently_burning` latched this long after the last throttle blip so
/// the trajectory widget doesn't flicker between burns.
const BURN_LINGER_SECS: f32 = 5.0;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Which widget occupies the slot. Reflect-registered (for a future debug UI).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Reflect)]
pub enum WidgetKind {
    Trajectory,
    NavDisplay,
    Docking,
    Interplanetary,
}

impl WidgetKind {
    /// Auto-pick scan order; also the tie-break order (earlier wins ties).
    pub const ALL: [WidgetKind; 4] = [
        WidgetKind::Trajectory,
        WidgetKind::NavDisplay,
        WidgetKind::Docking,
        WidgetKind::Interplanetary,
    ];

    /// Compact tab label.
    fn tab_label(self) -> &'static str {
        match self {
            WidgetKind::Trajectory => "TRAJ",
            WidgetKind::NavDisplay => "ND",
            WidgetKind::Docking => "DOCK",
            WidgetKind::Interplanetary => "IPL",
        }
    }

    fn relevance(self, ctx: &FlightContext) -> Option<i32> {
        match self {
            WidgetKind::Trajectory => widgets::trajectory::relevance(ctx),
            WidgetKind::NavDisplay => widgets::nav_display::relevance(ctx),
            WidgetKind::Docking => widgets::docking::relevance(ctx),
            WidgetKind::Interplanetary => widgets::interplanetary::relevance(ctx),
        }
    }
}

/// Pilot's slot selection. **Sole writer:** [`handle_tab_clicks`].
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Resource)]
pub enum MfdSelection {
    /// Show the highest-relevance widget for the current context.
    #[default]
    Auto,
    /// Pin one widget regardless of context (shows its "no data" state when
    /// the context wouldn't otherwise pick it).
    Pinned(WidgetKind),
    /// Hide the slot entirely.
    Hidden,
}

/// The widget currently shown (derived). `None` = empty slot.
/// **Sole writer:** [`select_active_widget`].
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
pub struct ActiveWidget(pub Option<WidgetKind>);

/// Per-frame flight situation the widgets' `relevance` reads.
/// **Sole writer:** [`update_flight_context`].
#[derive(Resource, Default, Clone, Copy)]
pub struct FlightContext {
    pub in_atmosphere: bool,
    pub prediction_shown: bool,
    pub recently_burning: bool,
    pub has_nodes: bool,
    pub altitude_m: f64,
    /// Distance to the nearest runway on the dominant body, if any.
    pub nearest_runway_m: Option<f64>,
}

/// Marker on the slot container (panel frame). Its visibility gates the whole
/// slot; carries [`HudPanel`].
#[derive(Component)]
struct MfdSlotRoot;

/// Marker on each widget's root node. The selector owns its `Visibility`.
#[derive(Component, Clone, Copy)]
pub struct MfdWidgetRoot {
    pub kind: WidgetKind,
}

/// What a selector tab does when pressed.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum MfdTab {
    Auto,
    Pin(WidgetKind),
    Hide,
}

// ---------------------------------------------------------------------------
// Shared geometry helper
// ---------------------------------------------------------------------------

/// Inertial position of a runway site's surface point — the same placement
/// the runway itself uses (`anchor_dir * (radius + elevation)` rotated into
/// the body's spin frame). Shared by the context's nearest-runway scan and
/// the ND widget's projection.
pub(super) fn runway_surface_inertial(
    site: &StructureSite,
    body_radius_m: f64,
    body_pos: DVec3,
    body_orientation: DQuat,
) -> DVec3 {
    let elevation_m = match site.placement {
        StructurePlacement::FlattenTo { elevation_m, .. } => elevation_m,
        StructurePlacement::Drape => 0.0,
    };
    body_pos + body_orientation * (site.anchor_dir * (body_radius_m + elevation_m))
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct MfdPlugin;

impl Plugin for MfdPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(UiMaterialPlugin::<widgets::trajectory::SystemMapMaterial>::default())
            .add_plugins(UiMaterialPlugin::<widgets::nav_display::NavDisplayMaterial>::default())
            .init_resource::<MfdSelection>()
            .register_type::<MfdSelection>()
            .register_type::<WidgetKind>()
            .init_resource::<ActiveWidget>()
            .init_resource::<FlightContext>()
            .add_systems(Startup, setup_mfd.after(super::theme::init_theme))
            .add_systems(
                Update,
                (
                    update_flight_context,
                    select_active_widget,
                    widgets::trajectory::update,
                    widgets::nav_display::update,
                    handle_tab_clicks,
                    update_tab_visuals,
                )
                    .chain()
                    .after(crate::SimStage::Sync)
                    .run_if(
                        crate::photo_mode::not_in_photo_mode
                            .and(crate::shipyard_editor::editor_closed),
                    ),
            );
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/// Builds the whole slot in one system (container → selector → widget area →
/// each widget's subtree) so widget roots never race a deferred parent insert.
fn setup_mfd(
    mut commands: Commands,
    theme: Res<HudTheme>,
    mut system_map_materials: ResMut<Assets<widgets::trajectory::SystemMapMaterial>>,
    mut nav_materials: ResMut<Assets<widgets::nav_display::NavDisplayMaterial>>,
) {
    let theme = &*theme;
    let system_map_materials = &mut *system_map_materials;
    let nav_materials = &mut *nav_materials;

    let mut root = panel_node();
    // Right edge, between the top-right FPS overlay and the bottom-right
    // staging stack — where the legacy TRAJECTORY panel sat.
    root.right = Val::Px(20.0);
    root.top = Val::Px(140.0);
    root.row_gap = Val::Px(6.0);
    root.align_items = AlignItems::Center;
    let (bg, border) = panel_frame(theme);

    commands
        .spawn((
            root,
            bg,
            border,
            // Start hidden; `select_active_widget` reveals the bezel in ship
            // view (the selector stays reachable even with no widget shown).
            Visibility::Hidden,
            MfdSlotRoot,
            HudPanel,
            Name::new("HudMfdSlot"),
        ))
        .with_children(|slot| {
            spawn_selector(slot, theme);
            slot.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    row_gap: Val::Px(6.0),
                    ..default()
                },
                Name::new("MfdWidgetArea"),
            ))
            .with_children(|area| {
                widgets::trajectory::build(area, theme, system_map_materials);
                widgets::nav_display::build(area, theme, nav_materials);
                widgets::docking::build(area, theme);
                widgets::interplanetary::build(area, theme);
            });
        });
}

fn spawn_selector(slot: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    slot.spawn((
        Node {
            width: Val::Px(200.0),
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::Wrap,
            justify_content: JustifyContent::Center,
            column_gap: Val::Px(4.0),
            row_gap: Val::Px(4.0),
            ..default()
        },
        Name::new("MfdSelector"),
    ))
    .with_children(|row| {
        spawn_tab(row, theme, MfdTab::Auto, "AUTO");
        for kind in WidgetKind::ALL {
            spawn_tab(row, theme, MfdTab::Pin(kind), kind.tab_label());
        }
        spawn_tab(row, theme, MfdTab::Hide, "OFF");
    });
}

fn spawn_tab(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, action: MfdTab, label: &str) {
    parent
        .spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            action,
            Name::new(format!("MfdTab_{label}")),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 10.0,
                    ..default()
                },
                TextColor(theme.text_dim),
            ));
        });
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// **Sole writer** of [`FlightContext`]. Reads only always-available signals
/// (no dependency on the regime bubble, which may be absent).
fn update_flight_context(
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    throttle: Res<ThrottleState>,
    structures: Res<StructureRegistry>,
    aero_q: Query<&AeroReadout, With<LocalCraftBody>>,
    time: Res<Time>,
    mut ctx: ResMut<FlightContext>,
    mut last_burn: Local<Option<f32>>,
) {
    let in_atmosphere = aero_q
        .single()
        .map(|r| r.density_kgm3 > IN_ATMOSPHERE_DENSITY)
        .unwrap_or(false);

    let s = &sim.simulation;
    let prediction_shown = s.prediction().is_some();
    let has_nodes = s
        .trajectory_branches()
        .is_some_and(|b| !b.branches.is_empty());

    let now = time.elapsed_secs();
    if throttle.effective > 0.02 {
        *last_burn = Some(now);
    }
    let recently_burning = last_burn.is_some_and(|t| now - t < BURN_LINGER_SECS);

    let dominant = s.dominant_body();
    let body_radius_m = s.bodies()[dominant].radius_m;
    let ship = s.ship_state();

    let mut altitude_m = 0.0;
    let mut nearest_runway_m: Option<f64> = None;
    if let Some(states) = solar.states.as_deref()
        && let Some(bs) = states.get(dominant)
    {
        altitude_m = (ship.position - bs.position).length() - body_radius_m;
        let mut best = f64::INFINITY;
        for site in structures.sites_on(dominant) {
            if site.kind != StructureKind::Runway {
                continue;
            }
            let surf = runway_surface_inertial(site, body_radius_m, bs.position, bs.orientation);
            best = best.min((surf - ship.position).length());
        }
        if best.is_finite() {
            nearest_runway_m = Some(best);
        }
    }

    *ctx = FlightContext {
        in_atmosphere,
        prediction_shown,
        recently_burning,
        has_nodes,
        altitude_m,
        nearest_runway_m,
    };
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Highest-relevance widget for the context, ties broken by [`WidgetKind::ALL`]
/// order (earlier wins).
fn auto_pick(ctx: &FlightContext) -> Option<WidgetKind> {
    let mut best: Option<(WidgetKind, i32)> = None;
    for kind in WidgetKind::ALL {
        if let Some(priority) = kind.relevance(ctx)
            && best.is_none_or(|(_, bp)| priority > bp)
        {
            best = Some((kind, priority));
        }
    }
    best.map(|(kind, _)| kind)
}

/// **Sole writer** of [`ActiveWidget`] and of every MFD root's layout/visibility.
/// One pass: the chosen widget root is laid out + shown, all others are taken
/// out of layout (`Display::None`). `Visibility::Hidden` alone would keep the
/// hidden widgets reserving their column height, stacking the visible one far
/// below them — so the chosen widget is driven by `Display`, not visibility.
fn select_active_widget(
    selection: Res<MfdSelection>,
    ctx: Res<FlightContext>,
    view: Res<ViewMode>,
    mut active: ResMut<ActiveWidget>,
    mut slot_q: Query<&mut Visibility, (With<MfdSlotRoot>, Without<MfdWidgetRoot>)>,
    mut roots: Query<(&MfdWidgetRoot, &mut Node, &mut Visibility), Without<MfdSlotRoot>>,
) {
    // Ship-view only: the map view already draws the full 3D trajectory.
    let ship_view = !matches!(*view, ViewMode::Map);
    let chosen = if ship_view {
        match *selection {
            MfdSelection::Hidden => None,
            MfdSelection::Pinned(kind) => Some(kind),
            MfdSelection::Auto => auto_pick(&ctx),
        }
    } else {
        None
    };

    if active.0 != chosen {
        active.0 = chosen;
    }

    // The slot bezel (the selector tab row) stays visible in ship view so the
    // pilot can always re-select a widget — even with the slot OFF, or when
    // AUTO finds nothing relevant. Only the widget content below it collapses.
    // The whole slot hides only in map view (and photo mode / editor, via the
    // container's `HudPanel`).
    let slot_target = if ship_view {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut slot_q {
        if *vis != slot_target {
            *vis = slot_target;
        }
    }

    for (root, mut node, mut vis) in &mut roots {
        let chosen_now = Some(root.kind) == chosen;
        let display = if chosen_now {
            Display::Flex
        } else {
            Display::None
        };
        if node.display != display {
            node.display = display;
        }
        let target_vis = if chosen_now {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != target_vis {
            *vis = target_vis;
        }
    }
}

// ---------------------------------------------------------------------------
// Selector interaction
// ---------------------------------------------------------------------------

fn handle_tab_clicks(
    tabs: Query<(&Interaction, &MfdTab), Changed<Interaction>>,
    mut selection: ResMut<MfdSelection>,
) {
    for (interaction, tab) in &tabs {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let next = match tab {
            MfdTab::Auto => MfdSelection::Auto,
            MfdTab::Pin(kind) => MfdSelection::Pinned(*kind),
            MfdTab::Hide => MfdSelection::Hidden,
        };
        if *selection != next {
            *selection = next;
        }
    }
}

fn update_tab_visuals(
    selection: Res<MfdSelection>,
    active: Res<ActiveWidget>,
    theme: Res<HudTheme>,
    mut tabs: Query<(
        &MfdTab,
        &Interaction,
        &mut BorderColor,
        &mut BackgroundColor,
        &Children,
    )>,
    mut text_q: Query<&mut TextColor>,
) {
    for (tab, interaction, mut border, mut bg, children) in &mut tabs {
        let selected = match (*selection, tab) {
            (MfdSelection::Auto, MfdTab::Auto) => true,
            (MfdSelection::Pinned(k), MfdTab::Pin(j)) => k == *j,
            (MfdSelection::Hidden, MfdTab::Hide) => true,
            _ => false,
        };
        // In AUTO mode, dim-highlight the tab AUTO actually resolved to.
        let auto_resolved = matches!(*selection, MfdSelection::Auto)
            && matches!(tab, MfdTab::Pin(k) if Some(*k) == active.0);

        let (border_color, bg_color) = match (selected, auto_resolved, interaction) {
            (true, _, _) => (theme.text_accent, theme.panel_bg),
            (false, true, _) => (theme.text_primary, theme.panel_bg),
            (false, false, Interaction::Pressed) => (theme.text_primary, theme.panel_border),
            (false, false, Interaction::Hovered) => (theme.text_primary, theme.panel_bg),
            (false, false, Interaction::None) => (theme.panel_border, theme.panel_bg),
        };
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
        if bg.0 != bg_color {
            bg.0 = bg_color;
        }

        let label_color = if selected {
            theme.text_accent
        } else if auto_resolved {
            theme.text_primary
        } else {
            theme.text_dim
        };
        if let Some(&child) = children.first()
            && let Ok(mut tc) = text_q.get_mut(child)
            && tc.0 != label_color
        {
            tc.0 = label_color;
        }
    }
}
