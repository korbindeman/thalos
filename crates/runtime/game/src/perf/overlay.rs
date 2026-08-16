//! Thalos adapter for the shared F3 diagnostics surface.
//!
//! `thalos_diagnostics_ui` owns F3, common frame/device/process facts, and the
//! frame graph. This module contributes simulation, terrain memory, planetary
//! position, capture override, and the game's hitbox/aero-gizmo side effects.

use bevy::{prelude::*, ui_render::prelude::MaterialNode};
use thalos_diagnostics_ui::{
    DiagnosticsGraphMaterial, DiagnosticsGraphMode, DiagnosticsPanelExtensions,
    DiagnosticsPanelGate, DiagnosticsPanelMemoryExtensions, DiagnosticsPanelStartupSet,
    DiagnosticsPanelState, DiagnosticsPanelUpdateSet, spawn_text_section,
};

use super::{MEM_RING_LEN, PerfSamples, fmt_bytes, fmt_mib};
use crate::aero::AeroGizmos;
use crate::bridge::CraftStateMirror;
use crate::debug::DebugMode;
use crate::rendering::SimulationState;
use crate::rendering::view_anchor::ViewAnchor;
use crate::terrain_registry::BodySurfaceRegistry;

const TEXT_REFRESH_FRAMES: u8 = 15;
const GRAPH_WIDTH_PX: f32 = 430.0;

#[derive(Component, Clone, Copy)]
enum GameBlock {
    Timing,
    Memory,
    Scene,
    Place,
}

#[derive(Component)]
struct GameMemoryGraph;

pub struct DebugViewPlugin;

impl Plugin for DebugViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Startup,
            (apply_initial_debug_view, setup_extensions)
                .in_set(DiagnosticsPanelStartupSet::Extensions),
        )
        .add_systems(
            Update,
            (
                (sync_gate, apply_debug_view_override).in_set(DiagnosticsPanelUpdateSet::Gate),
                sync_debug_surfaces.in_set(DiagnosticsPanelUpdateSet::Display),
                (update_memory_graph, update_stats_text).in_set(DiagnosticsPanelUpdateSet::Refresh),
            ),
        );
    }
}

fn apply_initial_debug_view(mut state: ResMut<DiagnosticsPanelState>) {
    if std::env::var_os("THALOS_DEBUG_VIEW").is_some_and(|value| value == "1") {
        state.visible = true;
    }
}

/// Honour a request-scoped `THALOS_DEBUG_VIEW` in the resident capture host.
fn apply_debug_view_override(
    overrides: Option<Res<crate::screenshot::CaptureRuntimeOverrides>>,
    mut state: ResMut<DiagnosticsPanelState>,
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

fn sync_gate(
    photo: Res<crate::photo_mode::PhotoMode>,
    settings: Res<thalos_preferences::SettingsMenu>,
    viewpoints: Option<Res<thalos_viewer::ViewpointUiState>>,
    mut gate: ResMut<DiagnosticsPanelGate>,
) {
    gate.available = !photo.active
        && !settings.open
        && !viewpoints
            .as_deref()
            .is_some_and(thalos_viewer::ViewpointUiState::is_open);
}

/// Keep the game's non-panel debug surfaces following the shared F3 state.
fn sync_debug_surfaces(
    state: Res<DiagnosticsPanelState>,
    gate: Res<DiagnosticsPanelGate>,
    mut previous: Local<bool>,
    mut debug: ResMut<DebugMode>,
    mut gizmos: ResMut<bevy::gizmos::config::GizmoConfigStore>,
) {
    let active = state.visible && gate.available;
    debug.show_hitboxes = active && debug.enabled;
    gizmos.config_mut::<AeroGizmos>().0.enabled = active;
    if *previous != active {
        info!("game debug surfaces: {}", if active { "ON" } else { "off" });
        *previous = active;
    }
}

fn setup_extensions(
    mut commands: Commands,
    extensions: Single<Entity, With<DiagnosticsPanelExtensions>>,
    memory_extensions: Single<Entity, With<DiagnosticsPanelMemoryExtensions>>,
    theme: Res<thalos_ui::UiTheme>,
    mut graph_materials: ResMut<Assets<DiagnosticsGraphMaterial>>,
) {
    commands.entity(*memory_extensions).with_children(|memory| {
        crate::vram_bar::spawn_vram_bar(
            memory,
            &crate::vram_bar::VramBarStyle {
                width: Val::Px(GRAPH_WIDTH_PX),
                bar_height: 7.0,
                font: theme.font_mono.clone(),
                font_size: 10.0,
                label_color: thalos_ui::tokens::TEXT_FAINT,
                value_color: thalos_ui::tokens::TEXT_PRIMARY,
                caption: "",
                show_header: false,
            },
        );
        memory.spawn((theme.mono_dim("—"), GameBlock::Memory));
        memory.spawn(theme.faint("MEMORY MiB · tile-resident / mesh-slab · 2 min"));
        memory.spawn((
            Node {
                width: Val::Px(GRAPH_WIDTH_PX),
                height: Val::Px(64.0),
                ..default()
            },
            MaterialNode(graph_materials.add(DiagnosticsGraphMaterial::memory())),
            GameMemoryGraph,
        ));
    });

    commands.entity(*extensions).with_children(|sections| {
        spawn_text_section(sections, &theme, "SIMULATION", GameBlock::Timing);
        spawn_text_section(sections, &theme, "TERRAIN", GameBlock::Scene);
        spawn_text_section(sections, &theme, "POSITION", GameBlock::Place);
    });
}

fn update_memory_graph(
    state: Res<DiagnosticsPanelState>,
    gate: Res<DiagnosticsPanelGate>,
    samples: Res<PerfSamples>,
    mut materials: ResMut<Assets<DiagnosticsGraphMaterial>>,
    graphs: Query<&MaterialNode<DiagnosticsGraphMaterial>, With<GameMemoryGraph>>,
) {
    if !state.visible || !gate.available {
        return;
    }
    for graph in &graphs {
        let Some(mut material) = materials.get_mut(&graph.0) else {
            continue;
        };
        material.set_series(
            samples.recent_mem(MEM_RING_LEN).map(|(tiles, _)| tiles),
            samples.recent_mem(MEM_RING_LEN).map(|(_, slabs)| slabs),
            1.0,
            DiagnosticsGraphMode::Memory,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn update_stats_text(
    mut tick: Local<u8>,
    state: Res<DiagnosticsPanelState>,
    gate: Res<DiagnosticsPanelGate>,
    samples: Res<PerfSamples>,
    craft: Option<Res<CraftStateMirror>>,
    anchor: Option<Res<ViewAnchor>>,
    sim: Option<Res<SimulationState>>,
    surfaces: Option<Res<BodySurfaceRegistry>>,
    tile_roots: Query<&thalos_body_render::tiles::TileTerrainRoot>,
    mut texts: Query<(&GameBlock, &mut Text)>,
) {
    if !state.visible || !gate.available {
        return;
    }
    *tick = tick.wrapping_add(1);
    if *tick != 1 && !tick.is_multiple_of(TEXT_REFRESH_FRAMES) {
        return;
    }

    let timing = format!(
        "physics {:.2} · sync {:.2} · camera {:.2} ms{}",
        samples.stage_physics_ms,
        samples.stage_sync_ms,
        samples.stage_camera_ms,
        sim_suffix(craft.as_deref()),
    );
    let memory = memory_block(&samples);
    let scene = scene_block(&samples, &tile_roots);
    let place = place_block(anchor.as_deref(), sim.as_deref(), surfaces.as_deref());
    for (block, mut text) in &mut texts {
        let value = match block {
            GameBlock::Timing => &timing,
            GameBlock::Memory => &memory,
            GameBlock::Scene => &scene,
            GameBlock::Place => &place,
        };
        if text.0 != *value {
            text.0.clone_from(value);
        }
    }
}

fn sim_suffix(craft: Option<&CraftStateMirror>) -> String {
    craft.map_or_else(String::new, |craft| {
        format!(
            " · warp {}× · t+{:.0} s",
            fmt_warp(craft.warp_speed),
            craft.sim_time_s,
        )
    })
}

fn fmt_warp(warp: f64) -> String {
    if warp.fract().abs() < 1e-6 {
        format!("{warp:.0}")
    } else {
        format!("{warp:.1}")
    }
}

fn memory_block(samples: &PerfSamples) -> String {
    let budget_bytes = thalos_body_render::tiles::residency_budget_bytes();
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
        "{terrain}\ncpu assets: {} mesh · {} image",
        fmt_mib(samples.mesh_cpu_mib),
        fmt_mib(samples.image_cpu_mib),
    )
}

fn scene_block(
    samples: &PerfSamples,
    tile_roots: &Query<&thalos_body_render::tiles::TileTerrainRoot>,
) -> String {
    let split_scale = tile_roots
        .iter()
        .map(|root| root.split_scale())
        .fold(1.0, f64::min);
    let instances = thalos_body_render::tiles::vram_share::live_instances();
    format!(
        "{} tiles resident · split {split_scale:.2}\n{instances} renderer instance{}",
        samples.tile_resident,
        if instances == 1 { "" } else { "s" },
    )
}

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

    let direction = anchor.cam_dir;
    let latitude_deg = direction.y.clamp(-1.0, 1.0).asin().to_degrees();
    let longitude_deg = direction.z.atan2(direction.x).to_degrees();
    let latitude_hemisphere = if latitude_deg >= 0.0 { 'N' } else { 'S' };
    let longitude_hemisphere = if longitude_deg >= 0.0 { 'E' } else { 'W' };
    let altitude_m = anchor.cam_body.length() - anchor.radius_m;
    let mut block = format!(
        "{name} · {:.4}°{latitude_hemisphere} {:.4}°{longitude_hemisphere}\n\
         {altitude_m:.0} m · {:.0} agl · {:.0} ground · {:.0} m/s",
        latitude_deg.abs(),
        longitude_deg.abs(),
        anchor.agl_m,
        anchor.ground_h_m,
        anchor.speed_m_s,
    );

    if let Some(surface) = surfaces.and_then(|surfaces| surfaces.surface(anchor.body)) {
        let moisture = surface.landcover_moisture(direction);
        let canopy = surface.canopy_coverage(direction, anchor.ground_h_m as f32, 1.0);
        block.push_str(&format!("\nmoisture {moisture:+.2} · canopy {canopy:.2}"));
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_format_drops_only_redundant_fractional_digits() {
        assert_eq!(fmt_warp(10.0), "10");
        assert_eq!(fmt_warp(2.5), "2.5");
    }
}
