use bevy::{
    diagnostic::{DiagnosticsStore, EntityCountDiagnosticsPlugin},
    prelude::*,
    render::{diagnostic::RenderDiagnosticsPlugin, renderer::RenderAdapterInfo},
    ui_render::prelude::{MaterialNode, UiMaterialPlugin},
    window::PrimaryWindow,
};

use crate::{
    DiagnosticsGraphMaterial, DiagnosticsGraphMode, DiagnosticsPanelPostUpdateSet,
    FRAME_HISTORY_LEN, FrameSamples, format_bytes, samples::collect_frame_samples,
};

const PANEL_WIDTH_PX: f32 = 462.0;
const GRAPH_WIDTH_PX: f32 = 430.0;
const TEXT_REFRESH_FRAMES: u8 = 15;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticsPanelStartupSet {
    Setup,
    Extensions,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticsPanelUpdateSet {
    Gate,
    Input,
    Display,
    Refresh,
}

/// Requested-open state. Application adapters observe this for related debug
/// side effects; temporary gates do not erase the player's choice.
#[derive(Resource, Default)]
pub struct DiagnosticsPanelState {
    pub visible: bool,
}

/// One application-owned aggregate gate for photo modes, modal menus, or
/// other surfaces that must temporarily own F3/input.
#[derive(Resource)]
pub struct DiagnosticsPanelGate {
    pub available: bool,
}

impl Default for DiagnosticsPanelGate {
    fn default() -> Self {
        Self { available: true }
    }
}

#[derive(Component)]
pub struct DiagnosticsPanelRoot;

/// Parent for application-specific text sections.
#[derive(Component)]
pub struct DiagnosticsPanelExtensions;

/// Parent inside the common memory section for richer application-owned
/// accounting widgets and graphs.
#[derive(Component)]
pub struct DiagnosticsPanelMemoryExtensions;

#[derive(Component, Clone, Copy)]
enum CommonBlock {
    Fps,
    Timing,
    Device,
    Memory,
    Scene,
}

#[derive(Component)]
struct FrameGraph;

pub struct DiagnosticsPanelPlugin;

impl Plugin for DiagnosticsPanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            EntityCountDiagnosticsPlugin::default(),
            RenderDiagnosticsPlugin,
            UiMaterialPlugin::<DiagnosticsGraphMaterial>::default(),
        ))
        .init_resource::<DiagnosticsPanelState>()
        .init_resource::<DiagnosticsPanelGate>()
        .init_resource::<FrameSamples>()
        .configure_sets(
            Startup,
            (
                DiagnosticsPanelStartupSet::Setup,
                DiagnosticsPanelStartupSet::Extensions,
            )
                .chain(),
        )
        .configure_sets(
            Update,
            (
                DiagnosticsPanelUpdateSet::Gate,
                DiagnosticsPanelUpdateSet::Input,
                DiagnosticsPanelUpdateSet::Display,
                DiagnosticsPanelUpdateSet::Refresh,
            )
                .chain(),
        )
        .add_systems(
            Startup,
            setup
                .after(thalos_ui::init_ui_theme)
                .in_set(DiagnosticsPanelStartupSet::Setup),
        )
        .add_systems(
            Update,
            (
                toggle.in_set(DiagnosticsPanelUpdateSet::Input),
                sync_display.in_set(DiagnosticsPanelUpdateSet::Display),
                (refresh_common_text, update_frame_graph)
                    .in_set(DiagnosticsPanelUpdateSet::Refresh),
            ),
        )
        .add_systems(
            PostUpdate,
            collect_frame_samples.in_set(DiagnosticsPanelPostUpdateSet::SampleFrame),
        );
    }
}

fn setup(
    mut commands: Commands,
    theme: Res<thalos_ui::UiTheme>,
    mut graph_materials: ResMut<Assets<DiagnosticsGraphMaterial>>,
) {
    let mut root = thalos_ui::floating_panel_node();
    root.display = Display::None;
    root.left = Val::Px(16.0);
    root.top = Val::Px(150.0);
    root.width = Val::Px(PANEL_WIDTH_PX);

    commands
        .spawn((
            root,
            theme.glass(),
            Visibility::Inherited,
            GlobalZIndex(80),
            DiagnosticsPanelRoot,
            Name::new("F3 diagnostics"),
        ))
        .with_children(|panel| {
            panel
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|header| {
                    header.spawn(theme.heading("DIAGNOSTICS"));
                    thalos_ui::spawn_key_hint(header, &theme, "F3");
                });

            let mut fps = theme.mono("— FPS");
            fps.1.font_size = FontSize::Px(28.0);
            panel.spawn((fps, CommonBlock::Fps));
            panel.spawn((
                theme.mono_dim("waiting for frame samples"),
                CommonBlock::Timing,
            ));

            spawn_section_header(panel, &theme, "DEVICE");
            panel.spawn((theme.mono_dim("—"), CommonBlock::Device));

            spawn_section_header(panel, &theme, "MEMORY");
            panel
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(thalos_ui::tokens::SPACE_SM),
                        ..default()
                    },
                    DiagnosticsPanelMemoryExtensions,
                ))
                .with_children(|memory| {
                    memory.spawn((theme.mono_dim("—"), CommonBlock::Memory));
                });

            spawn_section_header(panel, &theme, "SCENE");
            panel.spawn((theme.mono_dim("—"), CommonBlock::Scene));

            panel.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(thalos_ui::tokens::SPACE_SM),
                    ..default()
                },
                DiagnosticsPanelExtensions,
            ));

            panel.spawn(theme.faint("FRAME ms · cpu bars · gpu line · 16.7 / 33.3 marks"));
            panel.spawn((
                Node {
                    width: Val::Px(GRAPH_WIDTH_PX),
                    height: Val::Px(96.0),
                    ..default()
                },
                MaterialNode(graph_materials.add(DiagnosticsGraphMaterial::frame_time())),
                FrameGraph,
            ));

            let session = thalos_diagnostics::session_id();
            panel.spawn(theme.faint(if session == "unstarted" {
                "runtime diagnostic stream unavailable"
            } else {
                session
            }));
        });
}

pub fn spawn_section_header(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &thalos_ui::UiTheme,
    heading: &str,
) {
    thalos_ui::spawn_divider(parent);
    thalos_ui::spawn_heading(parent, theme, heading, false);
}

pub fn spawn_text_section(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &thalos_ui::UiTheme,
    heading: &str,
    marker: impl Bundle,
) {
    spawn_section_header(parent, theme, heading);
    parent.spawn((theme.mono_dim("—"), marker));
}

fn toggle(
    keys: Res<ButtonInput<KeyCode>>,
    gate: Res<DiagnosticsPanelGate>,
    mut state: ResMut<DiagnosticsPanelState>,
) {
    if gate.available && keys.just_pressed(KeyCode::F3) {
        state.visible = !state.visible;
        info!(
            "diagnostics (F3): {}",
            if state.visible { "ON" } else { "off" }
        );
    }
}

fn sync_display(
    state: Res<DiagnosticsPanelState>,
    gate: Res<DiagnosticsPanelGate>,
    mut roots: Query<&mut Node, With<DiagnosticsPanelRoot>>,
) {
    let display = if state.visible && gate.available {
        Display::Flex
    } else {
        Display::None
    };
    for mut node in &mut roots {
        if node.display != display {
            node.display = display;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn refresh_common_text(
    mut tick: Local<u8>,
    state: Res<DiagnosticsPanelState>,
    gate: Res<DiagnosticsPanelGate>,
    samples: Res<FrameSamples>,
    store: Res<DiagnosticsStore>,
    adapter: Option<Res<RenderAdapterInfo>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    meshes: Res<Assets<Mesh>>,
    images: Res<Assets<Image>>,
    mut texts: Query<(&CommonBlock, &mut Text)>,
) {
    if !state.visible || !gate.available {
        return;
    }
    *tick = tick.wrapping_add(1);
    if *tick != 1 && !tick.is_multiple_of(TEXT_REFRESH_FRAMES) {
        return;
    }

    let stats = samples.stats(120);
    let fps = stats.map_or_else(
        || "— FPS".to_string(),
        |stats| format!("{:.0} FPS", 1000.0 / stats.cpu_mean_ms.max(1e-3)),
    );
    let timing = format_timing(stats, top_gpu_passes(&store));
    let device = format_device(adapter.as_deref(), &windows);
    let memory = format_memory();
    let scene = format!(
        "{} entities · {} meshes · {} images",
        entity_count(&store),
        meshes.len(),
        images.len(),
    );

    for (block, mut text) in &mut texts {
        let value = match block {
            CommonBlock::Fps => &fps,
            CommonBlock::Timing => &timing,
            CommonBlock::Device => &device,
            CommonBlock::Memory => &memory,
            CommonBlock::Scene => &scene,
        };
        if text.0 != *value {
            text.0.clone_from(value);
        }
    }
}

fn update_frame_graph(
    state: Res<DiagnosticsPanelState>,
    gate: Res<DiagnosticsPanelGate>,
    samples: Res<FrameSamples>,
    mut materials: ResMut<Assets<DiagnosticsGraphMaterial>>,
    graphs: Query<&MaterialNode<DiagnosticsGraphMaterial>, With<FrameGraph>>,
) {
    if !state.visible || !gate.available {
        return;
    }
    for graph in &graphs {
        let Some(mut material) = materials.get_mut(&graph.0) else {
            continue;
        };
        material.set_series(
            samples.recent(FRAME_HISTORY_LEN).map(|(cpu_ms, _)| cpu_ms),
            samples.recent(FRAME_HISTORY_LEN).map(|(_, gpu_ms)| gpu_ms),
            1000.0 / 30.0 + 1.0,
            DiagnosticsGraphMode::FrameTime,
        );
    }
}

fn format_timing(stats: Option<crate::FrameStats>, passes: Vec<(String, f64)>) -> String {
    let Some(stats) = stats else {
        return "waiting for frame samples".to_string();
    };
    let pass_line = if passes.is_empty() {
        "GPU passes pending".to_string()
    } else {
        passes
            .into_iter()
            .map(|(name, milliseconds)| format!("{name} {milliseconds:.2}"))
            .collect::<Vec<_>>()
            .join(" · ")
    };
    format!(
        "{:.2} cpu · {:.2} gpu ms · p95 {:.2} · max {:.2}\n{pass_line}",
        stats.cpu_mean_ms, stats.gpu_mean_ms, stats.cpu_p95_ms, stats.cpu_max_ms,
    )
}

fn top_gpu_passes(store: &DiagnosticsStore) -> Vec<(String, f64)> {
    let mut passes: Vec<_> = store
        .iter()
        .filter_map(|diagnostic| {
            let path = diagnostic.path().as_str();
            let mut parts = path.split('/');
            let (Some("render"), Some(name), Some("elapsed_gpu"), None) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                return None;
            };
            let value = diagnostic
                .value()
                .filter(|value| value.is_finite() && *value > 0.005)?;
            Some((name.to_string(), value))
        })
        .collect();
    passes.sort_by(|left, right| right.1.total_cmp(&left.1));
    passes.truncate(3);
    passes
}

fn format_device(
    adapter: Option<&RenderAdapterInfo>,
    windows: &Query<&Window, With<PrimaryWindow>>,
) -> String {
    let adapter = adapter.map_or_else(
        || "render adapter pending".to_string(),
        |adapter| {
            let driver = if adapter.driver_info.is_empty() {
                adapter.driver.as_str()
            } else {
                adapter.driver_info.as_str()
            };
            if driver.is_empty() {
                format!("{} · {:?}", adapter.name, adapter.backend)
            } else {
                format!("{} · {:?} · {driver}", adapter.name, adapter.backend)
            }
        },
    );
    let resolution = windows.iter().next().map_or_else(
        || "headless".to_string(),
        |window| {
            format!(
                "{}×{}",
                window.resolution.physical_width(),
                window.resolution.physical_height()
            )
        },
    );
    format!("{adapter}\n{resolution}")
}

fn format_memory() -> String {
    let card = thalos_diagnostics::gpu_memory().map_or_else(
        || "whole-card VRAM unavailable".to_string(),
        |memory| {
            format!(
                "card {} / {} ({:.0} %)",
                format_bytes(memory.used_bytes),
                format_bytes(memory.total_bytes),
                memory.used_frac() * 100.0,
            )
        },
    );
    let resident = thalos_diagnostics::process::self_resident_bytes().map_or_else(
        || "host rss unavailable".to_string(),
        |bytes| format!("host {} rss", format_bytes(bytes)),
    );
    format!("{card}\n{resident}")
}

pub fn entity_count(store: &DiagnosticsStore) -> u64 {
    store
        .get(&EntityCountDiagnosticsPlugin::ENTITY_COUNT)
        .and_then(|diagnostic| diagnostic.value())
        .unwrap_or(0.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<DiagnosticsPanelState>()
            .init_resource::<DiagnosticsPanelGate>()
            .add_systems(Update, toggle);
        app.finish();
        app.cleanup();
        app
    }

    #[test]
    fn f3_toggle_respects_the_application_gate_without_losing_state() {
        let mut app = input_app();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::F3);
        app.update();
        assert!(app.world().resource::<DiagnosticsPanelState>().visible);

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.world_mut()
            .resource_mut::<DiagnosticsPanelGate>()
            .available = false;
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::F3);
        app.update();
        assert!(
            app.world().resource::<DiagnosticsPanelState>().visible,
            "a modal gate should preserve the requested-open state"
        );
    }

    #[test]
    fn pending_timing_is_explicit() {
        assert_eq!(format_timing(None, Vec::new()), "waiting for frame samples");
    }

    #[test]
    fn timing_keeps_frame_distribution_and_gpu_passes_together() {
        let stats = crate::FrameStats {
            count: 120,
            cpu_mean_ms: 12.5,
            cpu_p50_ms: 11.0,
            cpu_p95_ms: 18.0,
            cpu_max_ms: 24.0,
            gpu_mean_ms: 7.25,
        };
        let text = format_timing(Some(stats), vec![("main_pass".to_string(), 4.5)]);
        assert_eq!(
            text,
            "12.50 cpu · 7.25 gpu ms · p95 18.00 · max 24.00\nmain_pass 4.50"
        );
    }
}
