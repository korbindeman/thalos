//! **F3 debug view** — Minecraft-style live perf screen.
//!
//! One F3 press toggles the whole debug surface together: this stats/graph
//! screen, the physics hitbox overlay (`debug::draw_debug_hitboxes`), and the
//! aero force/wind gizmos (`aero::AeroGizmos`). This system is the single F3
//! reader; the previous per-module toggles were deleted so the three surfaces
//! can never drift out of sync.
//!
//! The graphs draw as one `UiMaterial` quad each: the CPU sample ring rides a
//! uniform array into `perf_graph.wgsl`, so a frame renders the whole history
//! with zero per-frame UI entity churn — the debug view must not perturb what
//! it measures. Being an `assets/shaders/` asset, the graph shader hot-reloads
//! in the running capture host for styling iteration.

use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use bevy::ui_render::prelude::{MaterialNode, UiMaterial, UiMaterialPlugin};

use super::{MEM_RING_LEN, PerfSamples, RING_LEN, entity_count, gpu_frame_ms};
use crate::aero::AeroGizmos;
use crate::debug::DebugMode;
use crate::hud::theme::{HudTheme, panel_frame, panel_node};

/// Samples packed 4-per-Vec4 into a uniform array.
const SERIES_VEC4S: usize = RING_LEN / 4;

/// Text stats refresh cadence (frames): ~4 Hz keeps text relayout invisible
/// in the frame graph the screen itself displays.
const TEXT_EVERY_FRAMES: u32 = 15;

/// Whether the F3 debug view is currently shown. Single writer:
/// [`toggle_debug_view`].
///
/// `THALOS_DEBUG_VIEW=1` starts it visible — that's how a headless capture
/// screenshots the view (no keypress channel there).
#[derive(Resource)]
pub struct DebugViewState {
    pub visible: bool,
}

impl Default for DebugViewState {
    fn default() -> Self {
        Self {
            visible: std::env::var_os("THALOS_DEBUG_VIEW").is_some_and(|v| v == "1"),
        }
    }
}

#[derive(Component)]
struct DebugViewRoot;

#[derive(Component)]
struct DebugStatsText;

/// Which graph a quad displays; also selects the material `mode`.
#[derive(Component, Clone, Copy, PartialEq)]
enum GraphKind {
    /// CPU frame-time bars + GPU line, fixed ms scale marks.
    FrameMs,
    /// Tile-resident + mesh-slab MiB lines, autoscaled.
    MemoryMib,
}

/// One uniform-driven graph. `series_a`/`series_b` are the two curves packed
/// 4 samples per Vec4; `params` = (count, scale, mode, unused);
/// `marks` = (mark1, mark2, 0, 0) in value units (0 = no mark).
#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct PerfGraphMaterial {
    #[uniform(0)]
    params: Vec4,
    #[uniform(1)]
    marks: Vec4,
    #[uniform(2)]
    series_a: [Vec4; SERIES_VEC4S],
    #[uniform(3)]
    series_b: [Vec4; SERIES_VEC4S],
}

impl Default for PerfGraphMaterial {
    fn default() -> Self {
        Self {
            params: Vec4::new(0.0, 33.4, 0.0, 0.0),
            marks: Vec4::ZERO,
            series_a: [Vec4::ZERO; SERIES_VEC4S],
            series_b: [Vec4::ZERO; SERIES_VEC4S],
        }
    }
}

impl UiMaterial for PerfGraphMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/perf_graph.wgsl".into()
    }
}

pub struct DebugViewPlugin;

impl Plugin for DebugViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(UiMaterialPlugin::<PerfGraphMaterial>::default())
            .init_resource::<DebugViewState>()
            .add_systems(
                Startup,
                setup
                    .after(crate::hud::theme::init_theme)
                    .run_if(resource_exists::<HudTheme>),
            )
            .add_systems(
                Update,
                (
                    toggle_debug_view,
                    sync_visibility,
                    update_graphs,
                    update_stats_text,
                ),
            );
    }
}

/// **F3** — the one debug-surface toggle (stats screen + hitboxes + aero
/// gizmos flip together).
fn toggle_debug_view(
    keys: Res<ButtonInput<KeyCode>>,
    photo: Res<crate::photo_mode::PhotoMode>,
    mut state: ResMut<DebugViewState>,
    mut debug: ResMut<DebugMode>,
    mut gizmos: ResMut<bevy::gizmos::config::GizmoConfigStore>,
) {
    if photo.active || !keys.just_pressed(KeyCode::F3) {
        return;
    }
    state.visible = !state.visible;
    debug.show_hitboxes = state.visible && debug.enabled;
    gizmos.config_mut::<AeroGizmos>().0.enabled = state.visible;
    info!(
        "debug view (F3): {}",
        if state.visible { "ON" } else { "off" }
    );
}

/// Keep the root's `Visibility` following `visible && !photo_mode`, writing
/// only on change.
fn sync_visibility(
    state: Res<DebugViewState>,
    photo: Res<crate::photo_mode::PhotoMode>,
    mut roots: Query<&mut Visibility, With<DebugViewRoot>>,
) {
    let target = if state.visible && !photo.active {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut roots {
        if *vis != target {
            *vis = target;
        }
    }
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<PerfGraphMaterial>>,
    theme: Res<HudTheme>,
) {
    let mut root = panel_node();
    root.left = Val::Px(16.0);
    root.top = Val::Px(150.0);
    root.row_gap = Val::Px(6.0);
    let (bg, border) = panel_frame(&theme);

    let stats_font = TextFont {
        font: theme.font.clone(),
        font_size: FontSize::Px(12.0),
        ..default()
    };
    let label_font = TextFont {
        font: theme.font.clone(),
        font_size: FontSize::Px(10.0),
        ..default()
    };

    commands
        .spawn((
            root,
            bg,
            border,
            Visibility::Hidden,
            DebugViewRoot,
            Name::new("DebugViewF3"),
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("collecting…"),
                stats_font,
                TextColor(theme.text_primary),
                DebugStatsText,
            ));

            p.spawn((
                Text::new("FRAME ms — cpu bars / gpu line — 16.7 + 33.3 marks"),
                label_font.clone(),
                TextColor(theme.text_subtitle),
            ));
            p.spawn((
                Node {
                    width: Val::Px(430.0),
                    height: Val::Px(96.0),
                    ..default()
                },
                MaterialNode(materials.add(PerfGraphMaterial {
                    params: Vec4::new(0.0, 33.4, 0.0, 0.0),
                    marks: Vec4::new(1000.0 / 60.0, 1000.0 / 30.0, 0.0, 0.0),
                    ..default()
                })),
                GraphKind::FrameMs,
            ));

            p.spawn((
                Text::new("MEMORY MiB — tile-resident / mesh-slab — 2 min"),
                label_font,
                TextColor(theme.text_subtitle),
            ));
            p.spawn((
                Node {
                    width: Val::Px(430.0),
                    height: Val::Px(64.0),
                    ..default()
                },
                MaterialNode(materials.add(PerfGraphMaterial {
                    params: Vec4::new(0.0, 1.0, 1.0, 0.0),
                    ..default()
                })),
                GraphKind::MemoryMib,
            ));
        });
}

fn pack_series(dst: &mut [Vec4; SERIES_VEC4S], values: impl Iterator<Item = f32>) -> (usize, f32) {
    let mut count = 0usize;
    let mut max = 0.0f32;
    for (i, v) in values.enumerate().take(RING_LEN) {
        dst[i / 4][i % 4] = v;
        count = i + 1;
        max = max.max(v);
    }
    (count, max)
}

fn update_graphs(
    state: Res<DebugViewState>,
    samples: Res<PerfSamples>,
    mut materials: ResMut<Assets<PerfGraphMaterial>>,
    graphs: Query<(&MaterialNode<PerfGraphMaterial>, &GraphKind)>,
) {
    if !state.visible {
        return;
    }
    for (node, kind) in &graphs {
        let Some(mut mat) = materials.get_mut(&node.0) else {
            continue;
        };
        match kind {
            GraphKind::FrameMs => {
                let (count, max_a) =
                    pack_series(&mut mat.series_a, samples.recent(RING_LEN).map(|(c, _)| c));
                let (_, max_b) =
                    pack_series(&mut mat.series_b, samples.recent(RING_LEN).map(|(_, g)| g));
                // Keep at least the 30 fps mark on screen; grow for spikes.
                let scale = (1000.0_f32 / 30.0 + 1.0).max(1.15 * max_a.max(max_b));
                mat.params = Vec4::new(count as f32, scale, 0.0, 0.0);
            }
            GraphKind::MemoryMib => {
                let (count, max_a) = pack_series(
                    &mut mat.series_a,
                    samples.recent_mem(MEM_RING_LEN).map(|(t, _)| t),
                );
                let (_, max_b) = pack_series(
                    &mut mat.series_b,
                    samples.recent_mem(MEM_RING_LEN).map(|(_, s)| s),
                );
                let scale = (1.15 * max_a.max(max_b)).max(1.0);
                mat.params = Vec4::new(count as f32, scale, 1.0, 0.0);
            }
        }
    }
}

fn update_stats_text(
    mut tick: Local<u32>,
    state: Res<DebugViewState>,
    samples: Res<PerfSamples>,
    store: Res<DiagnosticsStore>,
    mut texts: Query<&mut Text, With<DebugStatsText>>,
) {
    if !state.visible {
        return;
    }
    *tick += 1;
    if !tick.is_multiple_of(TEXT_EVERY_FRAMES) {
        return;
    }

    // Window stats over the last ~2 s.
    let window = 120usize.min(samples.frame_count()).max(1);
    let mut cpu: Vec<f32> = samples.recent(window).map(|(c, _)| c).collect();
    if cpu.is_empty() {
        return;
    }
    cpu.sort_by(|a, b| a.total_cmp(b));
    let mean = cpu.iter().sum::<f32>() / cpu.len() as f32;
    let p95 = cpu[(cpu.len() * 95 / 100).min(cpu.len() - 1)];
    let max = cpu[cpu.len() - 1];
    let gpu_ms = gpu_frame_ms(&store);

    // Top GPU passes (3-component render paths), heaviest first.
    let mut passes: Vec<(&str, f64)> = store
        .iter()
        .filter_map(|d| {
            let path = d.path().as_str();
            if !path.starts_with("render/") || !path.ends_with("/elapsed_gpu") {
                return None;
            }
            let mut parts = path.split('/');
            let (Some(_), Some(name), Some(_), None) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                return None;
            };
            let v = d.value().filter(|v| v.is_finite() && *v > 0.005)?;
            Some((name, v))
        })
        .collect();
    passes.sort_by(|a, b| b.1.total_cmp(&a.1));
    passes.truncate(4);
    let passes_line = passes
        .iter()
        .map(|(name, ms)| format!("{name} {ms:.2}"))
        .collect::<Vec<_>>()
        .join("  ");

    let text = format!(
        "fps {:>4.0}   frame {mean:.2} ms  p95 {p95:.2}  max {max:.2}\n\
         gpu {gpu_ms:.2} ms   {passes_line}\n\
         sim  physics {:.2}  sync {:.2}  camera {:.2} ms\n\
         entities {}   meshes {}   images {}\n\
         tiles {} ({:.0} MiB)   mesh slabs {:.0} MiB\n\
         session {}",
        1000.0 / mean.max(1e-3),
        samples.stage_physics_ms,
        samples.stage_sync_ms,
        samples.stage_camera_ms,
        entity_count(&store),
        samples.main_meshes,
        samples.main_images,
        samples.tile_resident,
        samples.tile_mib,
        samples.slab_mib(),
        crate::runtime_diagnostics::session_id(),
    );
    for mut t in &mut texts {
        if t.0 != text {
            t.0.clone_from(&text);
        }
    }
}
