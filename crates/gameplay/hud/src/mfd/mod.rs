//! Flight-HUD widget launcher and floating workspace.
//!
//! The fixed launcher is the catalogue; each widget lives in its own floating
//! window. Opening ND never replaces TRAJ, and closing a window never removes
//! its remembered placement. All lifecycle and placement mutations flow
//! through [`HudLayoutRequest`] and [`apply_layout_requests`].

pub mod widgets;

use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use serde::{Deserialize, Serialize};
use thalos_physics_local::LocalCraftBody;

use thalos_game_state::flight::{AeroReadout, ThrottleState};
use thalos_game_state::nav::ViewMode;
use thalos_game_state::structures::{StructureKind, StructureRegistry, StructureSite};
use thalos_game_state::{SimulationState, SolarSystemState};

use super::HudPanel;
use super::IN_ATMOSPHERE_DENSITY;
use super::theme::{HudTheme, panel_frame, panel_node};

const BURN_LINGER_SECS: f32 = 5.0;
const LAUNCHER_Z: i32 = 50;
const WINDOW_Z_BASE: i32 = 10;

// ---------------------------------------------------------------------------
// Persisted model
// ---------------------------------------------------------------------------

/// A widget implementation registered in the permanent launcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WidgetKind {
    Trajectory,
    NavDisplay,
    Docking,
    Interplanetary,
}

impl WidgetKind {
    pub const ALL: [WidgetKind; 4] = [
        WidgetKind::Trajectory,
        WidgetKind::NavDisplay,
        WidgetKind::Docking,
        WidgetKind::Interplanetary,
    ];

    const fn index(self) -> usize {
        match self {
            WidgetKind::Trajectory => 0,
            WidgetKind::NavDisplay => 1,
            WidgetKind::Docking => 2,
            WidgetKind::Interplanetary => 3,
        }
    }

    const fn tab_label(self) -> &'static str {
        match self {
            WidgetKind::Trajectory => "TRAJ",
            WidgetKind::NavDisplay => "ND",
            WidgetKind::Docking => "DOCK",
            WidgetKind::Interplanetary => "XFER",
        }
    }

    const fn title(self) -> &'static str {
        match self {
            WidgetKind::Trajectory => "TRAJECTORY",
            WidgetKind::NavDisplay => "NAV DISPLAY",
            WidgetKind::Docking => "DOCKING",
            WidgetKind::Interplanetary => "TRANSFER",
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

    /// Craft capability is catalogue policy, separate from situational
    /// relevance. A saved unavailable window is retained, merely dormant.
    pub fn available(self, ctx: &FlightContext) -> bool {
        match self {
            WidgetKind::NavDisplay => ctx.winged,
            WidgetKind::Trajectory | WidgetKind::Docking | WidgetKind::Interplanetary => true,
        }
    }
}

/// Stable identity of one floating window.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WidgetInstanceId(pub u64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct WidgetWindowSettings {
    id: u64,
    kind: WidgetKind,
    open: bool,
    /// Top-left position as a fraction of the currently usable viewport.
    x_frac: f32,
    y_frac: f32,
    z: u32,
}

impl Default for WidgetWindowSettings {
    fn default() -> Self {
        Self {
            id: 1,
            kind: WidgetKind::Trajectory,
            open: false,
            x_frac: 0.72,
            y_frac: 0.14,
            z: 1,
        }
    }
}

impl WidgetWindowSettings {
    const fn new(id: u64, kind: WidgetKind, x_frac: f32, y_frac: f32, z: u32) -> Self {
        Self {
            id,
            kind,
            open: false,
            x_frac,
            y_frac,
            z,
        }
    }

    fn sanitize(&mut self) {
        self.x_frac = finite_clamp01(self.x_frac);
        self.y_frac = finite_clamp01(self.y_frac);
    }
}

/// Player-owned widget workspace, persisted as the `hud` section of the
/// unified application settings file.
#[derive(Resource, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HudWorkspaceSettings {
    initialized: bool,
    windows: Vec<WidgetWindowSettings>,
}

impl Default for HudWorkspaceSettings {
    fn default() -> Self {
        Self {
            initialized: false,
            windows: default_windows(),
        }
    }
}

impl HudWorkspaceSettings {
    /// Deterministic headless layout. `THALOS_SCREENSHOT_WIDGETS` accepts a
    /// comma-separated set of `traj`, `nd`, `dock`, and `xfer`; absent/empty
    /// keeps the ordinary craft-derived first-run default.
    pub fn for_capture_env() -> Self {
        let mut settings = Self::default();
        let Ok(raw) = std::env::var("THALOS_SCREENSHOT_WIDGETS") else {
            return settings;
        };
        let requested: Vec<_> = raw
            .split(',')
            .filter_map(|token| match token.trim().to_ascii_lowercase().as_str() {
                "traj" | "trajectory" => Some(WidgetKind::Trajectory),
                "nd" | "nav" | "navigation" => Some(WidgetKind::NavDisplay),
                "dock" | "docking" => Some(WidgetKind::Docking),
                "xfer" | "ipl" | "transfer" | "interplanetary" => Some(WidgetKind::Interplanetary),
                _ => None,
            })
            .collect();
        if requested.is_empty() {
            return settings;
        }

        settings.initialized = true;
        for window in &mut settings.windows {
            window.open = requested.contains(&window.kind);
            let position = match window.kind {
                WidgetKind::Trajectory => Vec2::new(0.61, 0.55),
                WidgetKind::NavDisplay => Vec2::new(0.80, 0.08),
                WidgetKind::Docking => Vec2::new(0.03, 0.16),
                WidgetKind::Interplanetary => Vec2::new(0.03, 0.48),
            };
            window.x_frac = position.x;
            window.y_frac = position.y;
        }
        settings
    }

    fn sanitize(&mut self) {
        // This first slice renders one instance of every implemented kind. The
        // stable ID/vector model is already instance-shaped; explicit same-kind
        // duplication lands with responsive layouts, once widget-local query
        // state has also been instance-keyed.
        let mut sanitized = Vec::with_capacity(WidgetKind::ALL.len());
        for default in default_windows() {
            let default_id = default.id;
            let mut value = self
                .windows
                .iter()
                .find(|window| window.kind == default.kind)
                .cloned()
                .unwrap_or(default);
            value.id = default_id;
            value.sanitize();
            sanitized.push(value);
        }
        self.windows = sanitized;
    }
}

fn default_windows() -> Vec<WidgetWindowSettings> {
    vec![
        WidgetWindowSettings::new(1, WidgetKind::Trajectory, 0.76, 0.14, 1),
        WidgetWindowSettings::new(2, WidgetKind::NavDisplay, 0.70, 0.10, 2),
        WidgetWindowSettings::new(3, WidgetKind::Docking, 0.72, 0.18, 3),
        WidgetWindowSettings::new(4, WidgetKind::Interplanetary, 0.72, 0.22, 4),
    ]
}

fn finite_clamp01(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Runtime workspace state
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
struct HudWorkspaceRuntime {
    windows: Vec<WidgetWindowSettings>,
    next_z: u32,
}

impl HudWorkspaceRuntime {
    fn window(&self, id: WidgetInstanceId) -> Option<&WidgetWindowSettings> {
        self.windows.iter().find(|window| window.id == id.0)
    }

    fn window_mut(&mut self, id: WidgetInstanceId) -> Option<&mut WidgetWindowSettings> {
        self.windows.iter_mut().find(|window| window.id == id.0)
    }

    fn first_of_kind(&self, kind: WidgetKind) -> Option<WidgetInstanceId> {
        self.windows
            .iter()
            .find(|window| window.kind == kind)
            .map(|window| WidgetInstanceId(window.id))
    }
}

/// Derived visibility set read by widget update/input systems. It replaces the
/// old scalar `ActiveWidget`: several kinds may be live in the same frame.
#[derive(Resource, Default)]
pub struct ActiveWidgets([bool; 4]);

impl ActiveWidgets {
    pub fn contains(&self, kind: WidgetKind) -> bool {
        self.0[kind.index()]
    }
}

/// Per-frame flight situation used only for default/catalogue policy.
#[derive(Resource, Default, Clone, Copy)]
pub struct FlightContext {
    pub in_atmosphere: bool,
    pub winged: bool,
    pub prediction_shown: bool,
    pub recently_burning: bool,
    pub has_nodes: bool,
    pub altitude_m: f64,
    pub nearest_runway_m: Option<f64>,
}

impl FlightContext {
    pub fn airplane_flight(self) -> bool {
        self.in_atmosphere && self.winged
    }
}

#[derive(Component)]
struct WidgetLauncherRoot;

#[derive(Component, Clone, Copy)]
struct WidgetLauncherButton(WidgetKind);

#[derive(Component, Clone, Copy)]
struct WidgetWindowRoot {
    id: WidgetInstanceId,
    kind: WidgetKind,
}

/// Marker on widget content. The workspace owns its inherited visibility.
#[derive(Component, Clone, Copy)]
pub struct MfdWidgetRoot {
    pub kind: WidgetKind,
}

#[derive(Component, Clone, Copy)]
struct WidgetDragHandle(WidgetInstanceId);

#[derive(Component, Clone, Copy)]
struct WidgetCloseButton(WidgetInstanceId);

#[derive(Resource, Default)]
struct WorkspaceDrag(Option<DragState>);

#[derive(Clone, Copy)]
struct DragState {
    id: WidgetInstanceId,
    start_cursor: Vec2,
    start_position: Vec2,
    current_position: Vec2,
}

/// One canonical mutation path for launcher and floating-window interactions.
#[derive(Message, Clone, Copy, Debug)]
enum HudLayoutRequest {
    InitializeDefault(WidgetKind),
    Open(WidgetKind),
    Focus(WidgetInstanceId),
    Move {
        id: WidgetInstanceId,
        position: Vec2,
        commit: bool,
    },
    Close(WidgetInstanceId),
}

// ---------------------------------------------------------------------------
// Shared geometry helper
// ---------------------------------------------------------------------------

pub(super) fn runway_surface_inertial(
    registry: &StructureRegistry,
    site: &StructureSite,
    body_radius_m: f64,
    body_pos: DVec3,
    body_orientation: DQuat,
) -> DVec3 {
    let elevation_m = registry.site_elevation_m(site);
    body_pos + body_orientation * (site.anchor_dir * (body_radius_m + elevation_m))
}

// ---------------------------------------------------------------------------
// Plugin and construction
// ---------------------------------------------------------------------------

pub struct MfdPlugin;

impl Plugin for MfdPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(UiMaterialPlugin::<widgets::trajectory::SystemMapMaterial>::default())
            .add_plugins(UiMaterialPlugin::<widgets::nav_display::NavDisplayMaterial>::default())
            .init_resource::<HudWorkspaceSettings>()
            .init_resource::<HudWorkspaceRuntime>()
            .init_resource::<WorkspaceDrag>()
            .init_resource::<ActiveWidgets>()
            .init_resource::<widgets::nav_display::NavZoom>()
            .init_resource::<widgets::nav_display::NavRangeState>()
            .init_resource::<FlightContext>()
            .register_type::<WidgetKind>()
            .add_message::<HudLayoutRequest>()
            .add_systems(
                Startup,
                (initialize_workspace, setup_workspace)
                    .chain()
                    .after(super::theme::init_theme),
            )
            .add_systems(
                Update,
                (
                    update_flight_context,
                    initialize_default_workspace,
                    emit_launcher_requests,
                    emit_close_requests,
                    begin_window_drag,
                    update_window_drag,
                    apply_layout_requests,
                    constrain_workspace_to_viewport,
                    sync_workspace_visuals,
                    update_chrome_visuals,
                    widgets::trajectory::update,
                    widgets::nav_display::update,
                    widgets::nav_display::handle_canvas_click,
                    widgets::nav_display::handle_zoom,
                    widgets::nav_display::handle_select_buttons,
                )
                    .chain()
                    .after(thalos_game_state::sched::SimStage::Sync)
                    .run_if(
                        thalos_game_state::ui::not_in_photo_mode
                            .and_then(thalos_game_state::context::flight_or_no_context),
                    ),
            );
    }
}

fn initialize_workspace(
    mut settings: ResMut<HudWorkspaceSettings>,
    mut runtime: ResMut<HudWorkspaceRuntime>,
) {
    settings.sanitize();
    runtime.windows.clone_from(&settings.windows);
    runtime.next_z = runtime
        .windows
        .iter()
        .map(|window| window.z)
        .max()
        .unwrap_or(0)
        + 1;
}

fn setup_workspace(
    mut commands: Commands,
    theme: Res<HudTheme>,
    runtime: Res<HudWorkspaceRuntime>,
    mut system_map_materials: ResMut<Assets<widgets::trajectory::SystemMapMaterial>>,
    mut nav_materials: ResMut<Assets<widgets::nav_display::NavDisplayMaterial>>,
) {
    spawn_launcher(&mut commands, &theme);

    for window in runtime.windows.iter().cloned() {
        spawn_widget_window(
            &mut commands,
            &theme,
            window,
            &mut system_map_materials,
            &mut nav_materials,
        );
    }
}

fn spawn_launcher(commands: &mut Commands, theme: &HudTheme) {
    let mut node = panel_node();
    node.right = Val::Px(16.0);
    node.top = Val::Px(16.0);
    node.flex_direction = FlexDirection::Row;
    node.align_items = AlignItems::Center;
    node.column_gap = Val::Px(5.0);
    let (bg, border) = panel_frame(theme);

    commands
        .spawn((
            node,
            bg,
            border,
            Visibility::Hidden,
            GlobalZIndex(LAUNCHER_Z),
            WidgetLauncherRoot,
            HudPanel,
            Name::new("HudWidgetLauncher"),
        ))
        .with_children(|launcher| {
            launcher.spawn((
                Text::new("WIDGETS"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(9.0),
                    ..default()
                },
                TextColor(theme.text_dim),
            ));
            for kind in WidgetKind::ALL {
                launcher
                    .spawn((
                        Button,
                        Node {
                            min_width: Val::Px(30.0),
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
                        WidgetLauncherButton(kind),
                        Name::new(format!("WidgetLauncher_{}", kind.tab_label())),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new(kind.tab_label()),
                            TextFont {
                                font: theme.font.clone(),
                                font_size: FontSize::Px(10.0),
                                ..default()
                            },
                            TextColor(theme.text_dim),
                        ));
                    });
            }
        });
}

fn spawn_widget_window(
    commands: &mut Commands,
    theme: &HudTheme,
    window: WidgetWindowSettings,
    system_map_materials: &mut Assets<widgets::trajectory::SystemMapMaterial>,
    nav_materials: &mut Assets<widgets::nav_display::NavDisplayMaterial>,
) {
    let id = WidgetInstanceId(window.id);
    let kind = window.kind;
    let mut node = panel_node();
    node.left = Val::Percent(window.x_frac * 100.0);
    node.top = Val::Percent(window.y_frac * 100.0);
    node.align_items = AlignItems::Center;
    node.row_gap = Val::Px(5.0);
    let (bg, border) = panel_frame(theme);

    commands
        .spawn((
            node,
            bg,
            border,
            Visibility::Hidden,
            GlobalZIndex(WINDOW_Z_BASE + window.z as i32),
            WidgetWindowRoot { id, kind },
            HudPanel,
            Name::new(format!("HudWidgetWindow_{}", kind.tab_label())),
        ))
        .with_children(|root| {
            root.spawn((
                Button,
                Node {
                    width: Val::Percent(100.0),
                    min_height: Val::Px(22.0),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(6.0),
                    padding: UiRect::horizontal(Val::Px(2.0)),
                    ..default()
                },
                BackgroundColor(Color::NONE),
                Interaction::None,
                WidgetDragHandle(id),
                Name::new(format!("WidgetDragHandle_{}", kind.tab_label())),
            ))
            .with_children(|header| {
                header.spawn((
                    Text::new(kind.title()),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(10.0),
                        ..default()
                    },
                    TextColor(theme.text_dim),
                ));
                header.spawn(Node {
                    flex_grow: 1.0,
                    ..default()
                });
                header
                    .spawn((
                        Button,
                        Node {
                            width: Val::Px(20.0),
                            height: Val::Px(20.0),
                            border: UiRect::all(Val::Px(1.0)),
                            border_radius: BorderRadius::all(Val::Px(3.0)),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        BackgroundColor(Color::NONE),
                        BorderColor::all(theme.panel_border),
                        Interaction::None,
                        WidgetCloseButton(id),
                        Name::new(format!("WidgetClose_{}", kind.tab_label())),
                    ))
                    .with_children(|close| {
                        close.spawn((
                            Text::new("×"),
                            TextFont {
                                font: theme.font.clone(),
                                font_size: FontSize::Px(13.0),
                                ..default()
                            },
                            TextColor(theme.text_dim),
                        ));
                    });
            });

            root.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    ..default()
                },
                Name::new(format!("WidgetContent_{}", kind.tab_label())),
            ))
            .with_children(|area| match kind {
                WidgetKind::Trajectory => {
                    widgets::trajectory::build(area, theme, system_map_materials)
                }
                WidgetKind::NavDisplay => widgets::nav_display::build(area, theme, nav_materials),
                WidgetKind::Docking => widgets::docking::build(area, theme),
                WidgetKind::Interplanetary => widgets::interplanetary::build(area, theme),
            });
        });
}

// ---------------------------------------------------------------------------
// Context and default
// ---------------------------------------------------------------------------

fn update_flight_context(
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    throttle: Res<ThrottleState>,
    structures: Res<StructureRegistry>,
    aero_layout: Res<thalos_game_state::flight::ShipAeroLayout>,
    aero_q: Query<&AeroReadout, With<LocalCraftBody>>,
    time: Res<Time>,
    mut ctx: ResMut<FlightContext>,
    mut last_burn: Local<Option<f32>>,
) {
    let in_atmosphere = aero_q
        .single()
        .map(|readout| readout.density_kgm3 > IN_ATMOSPHERE_DENSITY)
        .unwrap_or(false);
    let winged = aero_layout.config.lift_slope > 0.0;

    let simulation = &sim.simulation;
    let prediction_shown = simulation.prediction().is_some();
    let has_nodes = simulation
        .trajectory_branches()
        .is_some_and(|branches| !branches.branches.is_empty());

    let now = time.elapsed_secs();
    if throttle.effective > 0.02 {
        *last_burn = Some(now);
    }
    let recently_burning = last_burn.is_some_and(|last| now - last < BURN_LINGER_SECS);

    let dominant = simulation.dominant_body();
    let body_radius_m = simulation.bodies()[dominant].radius_m;
    let ship = simulation.ship_state();
    let mut altitude_m = 0.0;
    let mut nearest_runway_m = None;
    if let Some(states) = solar.states.as_deref()
        && let Some(body) = states.get(dominant)
    {
        altitude_m = (ship.position - body.position).length() - body_radius_m;
        let mut best = f64::INFINITY;
        for site in structures.sites_on(dominant) {
            if !matches!(site.kind, StructureKind::Runway { .. }) {
                continue;
            }
            let position = runway_surface_inertial(
                &structures,
                site,
                body_radius_m,
                body.position,
                body.orientation,
            );
            best = best.min((position - ship.position).length());
        }
        if best.is_finite() {
            nearest_runway_m = Some(best);
        }
    }

    *ctx = FlightContext {
        in_atmosphere,
        winged,
        prediction_shown,
        recently_burning,
        has_nodes,
        altitude_m,
        nearest_runway_m,
    };
}

fn auto_pick(ctx: &FlightContext) -> WidgetKind {
    let mut best = None;
    for kind in WidgetKind::ALL {
        if !kind.available(ctx) {
            continue;
        }
        if let Some(priority) = kind.relevance(ctx)
            && best.is_none_or(|(_, best_priority)| priority > best_priority)
        {
            best = Some((kind, priority));
        }
    }
    best.map(|(kind, _)| kind).unwrap_or(if ctx.winged {
        WidgetKind::NavDisplay
    } else {
        WidgetKind::Trajectory
    })
}

fn initialize_default_workspace(
    settings: Res<HudWorkspaceSettings>,
    ctx: Res<FlightContext>,
    mut requests: MessageWriter<HudLayoutRequest>,
) {
    if !settings.initialized {
        requests.write(HudLayoutRequest::InitializeDefault(auto_pick(&ctx)));
    }
}

// ---------------------------------------------------------------------------
// Input → requests
// ---------------------------------------------------------------------------

fn emit_launcher_requests(
    buttons: Query<(&Interaction, &WidgetLauncherButton), Changed<Interaction>>,
    ctx: Res<FlightContext>,
    mut requests: MessageWriter<HudLayoutRequest>,
) {
    for (interaction, button) in &buttons {
        if matches!(interaction, Interaction::Pressed) && button.0.available(&ctx) {
            requests.write(HudLayoutRequest::Open(button.0));
        }
    }
}

fn emit_close_requests(
    buttons: Query<(&Interaction, &WidgetCloseButton), Changed<Interaction>>,
    mut requests: MessageWriter<HudLayoutRequest>,
) {
    for (interaction, button) in &buttons {
        if matches!(interaction, Interaction::Pressed) {
            requests.write(HudLayoutRequest::Close(button.0));
        }
    }
}

fn begin_window_drag(
    handles: Query<(&Interaction, &WidgetDragHandle), Changed<Interaction>>,
    close_buttons: Query<&Interaction, With<WidgetCloseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    runtime: Res<HudWorkspaceRuntime>,
    mut drag: ResMut<WorkspaceDrag>,
    mut requests: MessageWriter<HudLayoutRequest>,
) {
    // The close button sits inside the drag header. If the pointer press hit
    // that child, close owns the gesture and the parent must not latch a drag.
    if close_buttons
        .iter()
        .any(|interaction| matches!(interaction, Interaction::Pressed))
    {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    for (interaction, handle) in &handles {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let Some(state) = runtime.window(handle.0) else {
            continue;
        };
        let position = Vec2::new(state.x_frac, state.y_frac);
        drag.0 = Some(DragState {
            id: handle.0,
            start_cursor: cursor,
            start_position: position,
            current_position: position,
        });
        requests.write(HudLayoutRequest::Focus(handle.0));
    }
}

fn update_window_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    roots: Query<(&WidgetWindowRoot, &ComputedNode)>,
    mut drag: ResMut<WorkspaceDrag>,
    mut requests: MessageWriter<HudLayoutRequest>,
) {
    let Some(mut state) = drag.0 else {
        return;
    };
    let Ok(window) = windows.single() else {
        drag.0 = None;
        return;
    };

    if mouse.pressed(MouseButton::Left)
        && let Some(cursor) = window.cursor_position()
    {
        let viewport = Vec2::new(window.width().max(1.0), window.height().max(1.0));
        let mut max_position = Vec2::ONE;
        if let Some((_, computed)) = roots.iter().find(|(root, _)| root.id == state.id) {
            let logical_size = computed.size() * computed.inverse_scale_factor;
            max_position = (Vec2::ONE - logical_size / viewport).max(Vec2::ZERO);
        }
        let delta = (cursor - state.start_cursor) / viewport;
        state.current_position = (state.start_position + delta).clamp(Vec2::ZERO, max_position);
        requests.write(HudLayoutRequest::Move {
            id: state.id,
            position: state.current_position,
            commit: false,
        });
        drag.0 = Some(state);
        return;
    }

    requests.write(HudLayoutRequest::Move {
        id: state.id,
        position: state.current_position,
        commit: true,
    });
    drag.0 = None;
}

// ---------------------------------------------------------------------------
// One reducer and one visual projection
// ---------------------------------------------------------------------------

fn apply_layout_requests(
    mut requests: MessageReader<HudLayoutRequest>,
    mut runtime: ResMut<HudWorkspaceRuntime>,
    mut settings: ResMut<HudWorkspaceSettings>,
) {
    for request in requests.read().copied() {
        match request {
            HudLayoutRequest::InitializeDefault(kind) => {
                if settings.initialized {
                    continue;
                }
                settings.initialized = true;
                if let Some(id) = runtime.first_of_kind(kind) {
                    open_and_focus(id, &mut runtime, &mut settings);
                }
            }
            HudLayoutRequest::Open(kind) => {
                if let Some(id) = runtime.first_of_kind(kind) {
                    open_and_focus(id, &mut runtime, &mut settings);
                }
            }
            HudLayoutRequest::Focus(id) => focus(id, &mut runtime, &mut settings),
            HudLayoutRequest::Move {
                id,
                position,
                commit,
            } => {
                let position = Vec2::new(finite_clamp01(position.x), finite_clamp01(position.y));
                if let Some(window) = runtime.window_mut(id) {
                    window.x_frac = position.x;
                    window.y_frac = position.y;
                }
                if commit
                    && let Some(window) =
                        settings.windows.iter_mut().find(|window| window.id == id.0)
                {
                    window.x_frac = position.x;
                    window.y_frac = position.y;
                }
            }
            HudLayoutRequest::Close(id) => {
                if let Some(window) = runtime.window_mut(id) {
                    window.open = false;
                }
                if let Some(window) = settings.windows.iter_mut().find(|window| window.id == id.0) {
                    window.open = false;
                }
            }
        }
    }
}

fn open_and_focus(
    id: WidgetInstanceId,
    runtime: &mut HudWorkspaceRuntime,
    settings: &mut HudWorkspaceSettings,
) {
    if let Some(window) = runtime.window_mut(id) {
        window.open = true;
    }
    if let Some(window) = settings.windows.iter_mut().find(|window| window.id == id.0) {
        window.open = true;
    }
    focus(id, runtime, settings);
}

fn focus(
    id: WidgetInstanceId,
    runtime: &mut HudWorkspaceRuntime,
    settings: &mut HudWorkspaceSettings,
) {
    let z = runtime.next_z;
    runtime.next_z = runtime.next_z.saturating_add(1);
    if let Some(window) = runtime.window_mut(id) {
        window.z = z;
    }
    if let Some(window) = settings.windows.iter_mut().find(|window| window.id == id.0) {
        window.z = z;
    }
}

fn constrain_workspace_to_viewport(
    windows: Query<&Window, With<PrimaryWindow>>,
    roots: Query<(&WidgetWindowRoot, &ComputedNode)>,
    mut runtime: ResMut<HudWorkspaceRuntime>,
    mut settings: ResMut<HudWorkspaceSettings>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let viewport = Vec2::new(window.width().max(1.0), window.height().max(1.0));
    for (root, computed) in &roots {
        if computed.is_empty() {
            continue;
        }
        let logical_size = computed.size() * computed.inverse_scale_factor;
        let max_position = (Vec2::ONE - logical_size / viewport).max(Vec2::ZERO);
        let Some(state) = runtime.window(root.id) else {
            continue;
        };
        let old = Vec2::new(state.x_frac, state.y_frac);
        let clamped = old.clamp(Vec2::ZERO, max_position);
        if clamped.abs_diff_eq(old, 1e-6) {
            continue;
        }
        if let Some(state) = runtime.window_mut(root.id) {
            state.x_frac = clamped.x;
            state.y_frac = clamped.y;
        }
        if let Some(saved) = settings
            .windows
            .iter_mut()
            .find(|window| window.id == root.id.0)
        {
            saved.x_frac = clamped.x;
            saved.y_frac = clamped.y;
        }
    }
}

fn sync_workspace_visuals(
    runtime: Res<HudWorkspaceRuntime>,
    ctx: Res<FlightContext>,
    view: Res<ViewMode>,
    mut active: ResMut<ActiveWidgets>,
    mut launcher: Query<&mut Visibility, (With<WidgetLauncherRoot>, Without<WidgetWindowRoot>)>,
    mut windows: Query<(
        &WidgetWindowRoot,
        &mut Node,
        &mut Visibility,
        &mut GlobalZIndex,
    )>,
    mut contents: Query<
        (&MfdWidgetRoot, &mut Visibility),
        (Without<WidgetWindowRoot>, Without<WidgetLauncherRoot>),
    >,
) {
    let ship_view = !matches!(*view, ViewMode::Map);
    let launcher_visibility = if ship_view {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut visibility in &mut launcher {
        if *visibility != launcher_visibility {
            *visibility = launcher_visibility;
        }
    }

    let mut open = [false; 4];
    for (root, mut node, mut visibility, mut z_index) in &mut windows {
        let Some(state) = runtime.window(root.id) else {
            continue;
        };
        node.left = Val::Percent(state.x_frac * 100.0);
        node.top = Val::Percent(state.y_frac * 100.0);
        let visible = ship_view && state.open && root.kind.available(&ctx);
        let target = if visible {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != target {
            *visibility = target;
        }
        open[root.kind.index()] |= visible;
        let target_z = WINDOW_Z_BASE + state.z.min((LAUNCHER_Z - WINDOW_Z_BASE - 1) as u32) as i32;
        if z_index.0 != target_z {
            z_index.0 = target_z;
        }
    }

    if active.0 != open {
        active.0 = open;
    }
    for (root, mut visibility) in &mut contents {
        let target = if open[root.kind.index()] {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != target {
            *visibility = target;
        }
    }
}

fn update_chrome_visuals(
    runtime: Res<HudWorkspaceRuntime>,
    ctx: Res<FlightContext>,
    theme: Res<HudTheme>,
    mut launcher: Query<(
        &WidgetLauncherButton,
        &Interaction,
        &mut Node,
        &mut BorderColor,
        &mut BackgroundColor,
        &Children,
    )>,
    mut close_buttons: Query<
        (
            &Interaction,
            &mut BorderColor,
            &mut BackgroundColor,
            &Children,
        ),
        (With<WidgetCloseButton>, Without<WidgetLauncherButton>),
    >,
    mut text: Query<&mut TextColor>,
) {
    for (button, interaction, mut node, mut border, mut background, children) in &mut launcher {
        let available = button.0.available(&ctx);
        node.display = if available {
            Display::Flex
        } else {
            Display::None
        };
        if !available {
            continue;
        }
        let is_open = runtime
            .windows
            .iter()
            .any(|window| window.kind == button.0 && window.open);
        let hovered = matches!(interaction, Interaction::Hovered | Interaction::Pressed);
        let border_color = if is_open || hovered {
            theme.text_accent
        } else {
            theme.panel_border
        };
        let background_color = if matches!(interaction, Interaction::Pressed) {
            theme.panel_border
        } else {
            theme.panel_bg
        };
        *border = BorderColor::all(border_color);
        background.0 = background_color;
        if let Some(&child) = children.first()
            && let Ok(mut color) = text.get_mut(child)
        {
            color.0 = if is_open {
                theme.text_accent
            } else if hovered {
                theme.text_primary
            } else {
                theme.text_dim
            };
        }
    }

    for (interaction, mut border, mut background, children) in &mut close_buttons {
        let hovered = matches!(interaction, Interaction::Hovered | Interaction::Pressed);
        *border = BorderColor::all(if hovered {
            theme.text_warn
        } else {
            theme.panel_border
        });
        background.0 = if matches!(interaction, Interaction::Pressed) {
            theme.panel_border
        } else {
            Color::NONE
        };
        if let Some(&child) = children.first()
            && let Ok(mut color) = text.get_mut(child)
        {
            color.0 = if hovered {
                theme.text_warn
            } else {
                theme.text_dim
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_defaults_to_navigation_display() {
        let ctx = FlightContext {
            winged: true,
            ..default()
        };
        assert_eq!(auto_pick(&ctx), WidgetKind::NavDisplay);
    }

    #[test]
    fn rocket_defaults_to_trajectory() {
        assert_eq!(auto_pick(&FlightContext::default()), WidgetKind::Trajectory);
    }

    #[test]
    fn situational_relevance_overrides_craft_default() {
        let ctx = FlightContext {
            winged: true,
            prediction_shown: true,
            recently_burning: true,
            ..default()
        };
        assert_eq!(auto_pick(&ctx), WidgetKind::Trajectory);
    }

    #[test]
    fn corrupt_saved_positions_are_sanitized() {
        let mut settings = HudWorkspaceSettings::default();
        settings.windows[0].x_frac = f32::NAN;
        settings.windows[0].y_frac = 9.0;
        settings.sanitize();
        assert_eq!(settings.windows[0].x_frac, 0.0);
        assert_eq!(settings.windows[0].y_frac, 1.0);
    }

    #[test]
    fn missing_widget_entries_are_repaired_without_losing_known_state() {
        let mut settings = HudWorkspaceSettings {
            initialized: true,
            windows: vec![WidgetWindowSettings {
                kind: WidgetKind::NavDisplay,
                open: true,
                x_frac: 0.25,
                ..default()
            }],
        };
        settings.sanitize();
        assert_eq!(settings.windows.len(), WidgetKind::ALL.len());
        let nav = settings
            .windows
            .iter()
            .find(|window| window.kind == WidgetKind::NavDisplay)
            .unwrap();
        assert!(nav.open);
        assert_eq!(nav.x_frac, 0.25);
    }

    #[test]
    fn workspace_model_keeps_several_kinds_open() {
        let mut settings = HudWorkspaceSettings::default();
        settings.initialized = true;
        for window in &mut settings.windows {
            window.open = matches!(window.kind, WidgetKind::Trajectory | WidgetKind::NavDisplay);
        }
        assert_eq!(
            settings.windows.iter().filter(|window| window.open).count(),
            2
        );
    }
}
