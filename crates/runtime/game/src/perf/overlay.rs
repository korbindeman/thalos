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
use bevy::render::renderer::RenderAdapterInfo;
use bevy::shader::ShaderRef;
use bevy::ui_render::prelude::{MaterialNode, UiMaterial, UiMaterialPlugin};
use bevy::window::PrimaryWindow;

use super::{MEM_RING_LEN, PerfSamples, RING_LEN, entity_count, fmt_bytes, fmt_mib, gpu_frame_ms};
use crate::aero::AeroGizmos;
use crate::bridge::CraftStateMirror;
use crate::debug::DebugMode;
use crate::hud::theme::{HudTheme, panel_frame, panel_node};
use crate::rendering::SimulationState;
use crate::rendering::view_anchor::ViewAnchor;
use crate::terrain_registry::BodySurfaceRegistry;

/// Samples packed 4-per-Vec4 into a uniform array.
const SERIES_VEC4S: usize = RING_LEN / 4;

/// Text stats refresh cadence (frames): ~4 Hz keeps text relayout invisible
/// in the frame graph the screen itself displays.
const TEXT_EVERY_FRAMES: u32 = 15;

/// Width of the graphs and the VRAM bar. They share it so the panel has one
/// left and one right edge rather than a ragged stack.
const GRAPH_WIDTH: f32 = 430.0;

/// Whether the F3 debug view is currently shown. Single writer:
/// [`toggle_debug_view`].
///
/// `THALOS_DEBUG_VIEW=1` starts it visible — that's how a headless capture
/// screenshots the view (no keypress channel there). The env var is read at
/// boot *and* per capture request ([`apply_debug_view_override`]), because the
/// capture host is a machine-wide shared process: reading it only at boot meant
/// a shot against a host someone else started came back looking like a
/// successful capture with the view simply off.
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

/// Which block of the stats panel a text node carries.
///
/// The panel used to be one text node holding every line in one colour and one
/// size, which is unreadable at a glance — the fps you check constantly and the
/// session id you read once a month had identical weight. Splitting it lets the
/// headline be large, section captions be small and faint, and bodies sit in
/// between, so the eye can land on the right block without reading all of it.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum StatsBlock {
    /// The one number worth seeing from across the room.
    Fps,
    /// Frame/GPU/stage timing detail under it.
    Timing,
    /// Adapter and window.
    Device,
    /// Memory beyond the VRAM bar's own legend.
    Memory,
    /// Scene object counts.
    Scene,
    /// Where the view is, and what the generator says is there.
    Place,
}

/// Section captions, in draw order. `None` = no caption (the headline block
/// heads the panel on its own).
const STATS_SECTIONS: [(Option<&str>, StatsBlock); 6] = [
    (None, StatsBlock::Fps),
    (None, StatsBlock::Timing),
    (Some("DEVICE"), StatsBlock::Device),
    (Some("MEMORY"), StatsBlock::Memory),
    (Some("SCENE"), StatsBlock::Scene),
    (Some("POSITION"), StatsBlock::Place),
];

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
                    apply_debug_view_override,
                    sync_visibility,
                    update_graphs,
                    update_stats_text,
                )
                    .chain(),
            );
    }
}

/// Honour a capture request's `THALOS_DEBUG_VIEW`, so the F3 view can be
/// screenshotted through the **resident** host rather than only a cold one.
///
/// The alternative — a `ScreenshotConfig` field — would mean a line in every
/// preset literal for a knob no preset wants to set. This reads the request's
/// own override map instead, which is already the per-shot channel.
fn apply_debug_view_override(
    overrides: Option<Res<crate::screenshot::CaptureRuntimeOverrides>>,
    mut state: ResMut<DebugViewState>,
) {
    let Some(overrides) = overrides else {
        return;
    };
    if !overrides.is_changed() {
        return;
    }
    let Some(raw) = overrides.values.get("THALOS_DEBUG_VIEW") else {
        return;
    };
    let visible = matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    );
    if state.visible != visible {
        state.visible = visible;
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
            for (caption, block) in STATS_SECTIONS {
                if let Some(caption) = caption {
                    p.spawn((
                        Node {
                            // Air above a caption, none below it: the caption
                            // has to read as belonging to the block under it,
                            // not floating between two.
                            margin: UiRect::top(Val::Px(7.0)),
                            ..default()
                        },
                        Text::new(caption),
                        TextFont {
                            font: theme.font.clone(),
                            font_size: FontSize::Px(9.0),
                            ..default()
                        },
                        TextColor(theme.text_subtitle),
                    ));
                }
                // The VRAM bar *heads* the memory block — how full the card is
                // and which band grew, before any of the numbers under it.
                if block == StatsBlock::Memory {
                    crate::vram_bar::spawn_vram_bar(
                        p,
                        &crate::vram_bar::VramBarStyle {
                            width: Val::Px(GRAPH_WIDTH),
                            bar_height: 7.0,
                            font: theme.font.clone(),
                            font_size: 10.0,
                            label_color: theme.text_subtitle,
                            value_color: theme.text_primary,
                            // The section caption above already says MEMORY.
                            caption: "",
                        },
                    );
                }

                p.spawn((
                    block,
                    // Never empty: the first text refresh is a quarter second
                    // after F3, and a panel that opens with blank rows reads as
                    // broken rather than as loading.
                    Text::new("—"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(match block {
                            StatsBlock::Fps => 22.0,
                            StatsBlock::Timing => 10.0,
                            _ => 12.0,
                        }),
                        ..default()
                    },
                    TextColor(match block {
                        StatsBlock::Fps => theme.text_accent,
                        StatsBlock::Timing => theme.text_subtitle,
                        _ => theme.text_primary,
                    }),
                ));
            }

            p.spawn((
                Text::new("FRAME ms — cpu bars / gpu line — 16.7 + 33.3 marks"),
                label_font.clone(),
                TextColor(theme.text_subtitle),
            ));
            p.spawn((
                Node {
                    width: Val::Px(GRAPH_WIDTH),
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
                    width: Val::Px(GRAPH_WIDTH),
                    height: Val::Px(64.0),
                    ..default()
                },
                MaterialNode(materials.add(PerfGraphMaterial {
                    params: Vec4::new(0.0, 1.0, 1.0, 0.0),
                    ..default()
                })),
                GraphKind::MemoryMib,
            ));

            // The session id is the join key between this screen and
            // `runtime.jsonl` / `just perf-report`, so it has to be on screen —
            // but it is read once, not watched, so it sits at the bottom in the
            // faintest type on the panel.
            p.spawn((
                Text::new(crate::runtime_diagnostics::session_id()),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(9.0),
                    ..default()
                },
                TextColor(theme.text_subtitle),
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

/// Rebuild the F3 stats block.
///
/// Every source below the perf gauges is `Option`al and degrades to a stated
/// gap rather than to a blank or a zero: the menu has no craft, a boot before
/// the world exists has no view anchor, a non-NVIDIA card has no whole-card
/// VRAM. A debug screen that silently prints `0` for a quantity it could not
/// read is worse than one that says so.
fn update_stats_text(
    mut tick: Local<u32>,
    state: Res<DebugViewState>,
    samples: Res<PerfSamples>,
    store: Res<DiagnosticsStore>,
    adapter: Option<Res<RenderAdapterInfo>>,
    craft: Option<Res<CraftStateMirror>>,
    anchor: Option<Res<ViewAnchor>>,
    sim: Option<Res<SimulationState>>,
    surfaces: Option<Res<BodySurfaceRegistry>>,
    tile_roots: Query<&thalos_body_render::tiles::TileTerrainRoot>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut texts: Query<(&StatsBlock, &mut Text)>,
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
    passes.truncate(3);
    let passes_line = passes
        .iter()
        .map(|(name, ms)| format!("{name} {ms:.2}"))
        .collect::<Vec<_>>()
        .join("  ");

    for (block, mut text) in &mut texts {
        let new_text = match block {
            StatsBlock::Fps => format!("{:.0} FPS", 1000.0 / mean.max(1e-3)),
            StatsBlock::Timing => format!(
                "{mean:.1} cpu · {gpu_ms:.1} gpu ms · p95 {p95:.1} · max {max:.1}\n\
                 physics {:.1} · sync {:.1} · camera {:.1}{}\n\
                 {passes_line}",
                samples.stage_physics_ms,
                samples.stage_sync_ms,
                samples.stage_camera_ms,
                sim_suffix(craft.as_deref()),
            ),
            StatsBlock::Device => device_block(adapter.as_deref(), &windows),
            StatsBlock::Memory => memory_block(&samples),
            StatsBlock::Scene => scene_block(&store, &samples, &tile_roots),
            StatsBlock::Place => {
                place_block(anchor.as_deref(), sim.as_deref(), surfaces.as_deref())
            }
        };
        if text.0 != new_text {
            text.0.clone_from(&new_text);
        }
    }
}

/// `   warp 10×  t+1 284 s`, or nothing when there is no craft yet (the menu,
/// a world-`Absent` boot).
fn sim_suffix(craft: Option<&CraftStateMirror>) -> String {
    match craft {
        Some(craft) => format!(
            "   warp {}×  t+{:.0} s",
            fmt_warp(craft.warp_speed),
            craft.sim_time_s
        ),
        None => String::new(),
    }
}

/// Warp factor without trailing zeros: `1`, `2.5`, `10 000`.
fn fmt_warp(warp: f64) -> String {
    if warp.fract().abs() < 1e-6 {
        format!("{warp:.0}")
    } else {
        format!("{warp:.1}")
    }
}

/// Adapter and window.
///
/// Which card this process *actually* got is not always the one the machine
/// advertises — a laptop that fell back to the integrated GPU, or a
/// `THALOS_WGPU_BACKEND` override, explains a frame-time report that otherwise
/// looks impossible. The instance count rides along because it is the divisor
/// of the tile budget below, and a second instance is exactly what made a 12 GB
/// card run out with both processes reporting themselves comfortably inside
/// budget (INC-20260725T012104Z-tile-residency-had-no-budget).
fn device_block(
    adapter: Option<&RenderAdapterInfo>,
    windows: &Query<&Window, With<PrimaryWindow>>,
) -> String {
    let card = match adapter {
        Some(info) => {
            let driver = if info.driver_info.is_empty() {
                info.driver.as_str()
            } else {
                info.driver_info.as_str()
            };
            let mut line = format!("{} · {:?}", info.name, info.backend);
            if !driver.is_empty() {
                line.push_str(&format!(" · {driver}"));
            }
            line
        }
        None => "no render adapter".to_string(),
    };
    let resolution = windows
        .iter()
        .next()
        .map(|window| {
            format!(
                "{}×{}",
                window.resolution.physical_width(),
                window.resolution.physical_height()
            )
        })
        .unwrap_or_else(|| "headless".to_string());
    let instances = thalos_body_render::tiles::vram_share::live_instances();
    let plural = if instances == 1 { "" } else { "s" };
    format!("{card}\n{resolution} · {instances} renderer instance{plural}")
}

/// The two memory facts the VRAM bar above cannot show.
///
/// The bar owns the GPU side — how full the card is and which band filled it.
/// What it cannot express is a **limit** (terrain residency against the budget
/// that will start coarsening the ground) or the **other side of the bus**: a
/// capture host was killed at 8.1 GiB RSS while every GPU-side gauge summed to
/// ~2 GiB (INC-20260729T081809Z), so a screen showing only VRAM cannot explain
/// that death at all. Both stay, in one line each, explicitly labelled `host`
/// so nobody reads them against the card figure again.
fn memory_block(samples: &PerfSamples) -> String {
    let budget_bytes = thalos_body_render::tiles::residency_budget_bytes();
    // Deliberately a *fraction*, not the byte count again. The bar's legend
    // above already prints terrain bytes, and the two are sampled by separate
    // timers — printing both showed `terrain 1.7 GiB` directly above
    // `terrain budget 1.8 GiB`, one quantity contradicting itself, which is
    // exactly how a diagnostic screen stops being believed.
    let terrain = if budget_bytes == usize::MAX {
        "terrain budget disabled".to_string()
    } else {
        let budget_mib = budget_bytes as f32 / (1024.0 * 1024.0);
        format!(
            "terrain {:.0} % of its {} budget",
            (samples.tile_mib / budget_mib.max(f32::EPSILON) * 100.0).min(999.0),
            fmt_bytes(budget_bytes as u64),
        )
    };
    format!(
        "{terrain}\n\
         host {} rss (cpu-side: {} mesh, {} image)",
        fmt_mib(samples.rss_mib),
        fmt_mib(samples.mesh_cpu_mib),
        fmt_mib(samples.image_cpu_mib),
    )
}

/// What that memory is holding: object counts, and the tile driver's own
/// braking state.
fn scene_block(
    store: &DiagnosticsStore,
    samples: &PerfSamples,
    tile_roots: &Query<&thalos_body_render::tiles::TileTerrainRoot>,
) -> String {
    // `split_scale < 1` means the residency budget is actively coarsening the
    // ground — the tell that separates "the terrain looks blurry" from "the
    // terrain is blurry *because* it is out of VRAM".
    let split_scale = tile_roots
        .iter()
        .map(|root| root.split_scale())
        .fold(1.0, f64::min);
    format!(
        "{} entities · {} meshes · {} images\n\
         {} tiles resident · split {split_scale:.2}",
        entity_count(store),
        samples.main_meshes,
        samples.main_images,
        samples.tile_resident,
    )
}

/// Where the view is, in the frame the ground is actually generated in.
///
/// Two lines: the body and the latitude/longitude the terrain field is being
/// evaluated at, then the altitude triple and the landcover the generator
/// believes is there. That last part is the difference between "the trees are
/// wrong here" and a reproducible coordinate plus the field values that
/// produced them.
fn place_block(
    anchor: Option<&ViewAnchor>,
    sim: Option<&SimulationState>,
    surfaces: Option<&BodySurfaceRegistry>,
) -> String {
    let Some(anchor) = anchor.and_then(|anchor| anchor.resolved) else {
        return "no terrain-backed body under the view".to_string();
    };
    let name = sim
        .and_then(|sim| sim.simulation.bodies().get(anchor.body))
        .map(|body| body.name.clone())
        .unwrap_or_else(|| format!("body {}", anchor.body));

    // Inverse of the `latlon_dir` convention used across the surface code
    // (+y north, longitude measured from +x through +z).
    let dir = anchor.cam_dir;
    let lat_deg = dir.y.clamp(-1.0, 1.0).asin().to_degrees();
    let lon_deg = dir.z.atan2(dir.x).to_degrees();
    let (lat_hemi, lon_hemi) = (
        if lat_deg >= 0.0 { 'N' } else { 'S' },
        if lon_deg >= 0.0 { 'E' } else { 'W' },
    );

    let altitude_m = anchor.cam_body.length() - anchor.radius_m;
    let mut block = format!(
        "{name} · {:.4}°{lat_hemi} {:.4}°{lon_hemi}\n\
         {altitude_m:.0} m · {:.0} agl · {:.0} ground · {:.0} m/s",
        lat_deg.abs(),
        lon_deg.abs(),
        anchor.agl_m,
        anchor.ground_h_m,
        anchor.speed_m_s,
    );

    // Two point queries at 4 Hz — the same fields the ground shader and the
    // scatter placer read, so a disagreement between what is on screen and what
    // is printed here is itself the bug.
    if let Some(surface) = surfaces.and_then(|surfaces| surfaces.surface(anchor.body)) {
        let moisture = surface.landcover_moisture(dir);
        let canopy = surface.canopy_coverage(dir, anchor.ground_h_m as f32, PLACE_QUERY_LOD_M);
        block.push_str(&format!("\nmoisture {moisture:+.2} · canopy {canopy:.2}"));
    }
    block
}

/// Sampling scale for the landcover point queries in [`place_block`]. Metres per
/// sample, at the finest scale the generator is asked for anywhere — this is a
/// single point, so there is nothing to gain from a coarser one.
const PLACE_QUERY_LOD_M: f32 = 1.0;
