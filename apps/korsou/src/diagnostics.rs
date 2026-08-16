use bevy::prelude::*;
use thalos_runtime::diagnostics_ui::{
    DiagnosticsPanelExtensions, DiagnosticsPanelGate, DiagnosticsPanelPlugin, DiagnosticsPanelRoot,
    DiagnosticsPanelStartupSet, DiagnosticsPanelState, DiagnosticsPanelUpdateSet,
    spawn_text_section,
};

use crate::{
    camera::TerrainCamera,
    foliage::FoliageStats,
    spatial::TerrainSpatialFrame,
    terrain::{TerrainDataset, TerrainStats},
    viewpoint::ViewpointUiState,
    world::{SolarClock, format_local_time, format_ordinal_date},
};

const REFRESH_EVERY_FRAMES: u8 = 15;

pub struct DiagnosticsPlugin;

impl Plugin for DiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DiagnosticsPanelPlugin)
            .add_systems(
                Startup,
                setup_extensions.in_set(DiagnosticsPanelStartupSet::Extensions),
            )
            .add_systems(
                Update,
                (
                    sync_gate
                        .after(crate::photo_mode::KorsouPhotoModeInput)
                        .in_set(DiagnosticsPanelUpdateSet::Gate),
                    refresh_text.in_set(DiagnosticsPanelUpdateSet::Refresh),
                ),
            );
    }
}

#[derive(Component, Clone, Copy)]
enum KorsouBlock {
    Streaming,
    Position,
    Lighting,
}

fn setup_extensions(
    mut commands: Commands,
    root: Single<Entity, With<DiagnosticsPanelRoot>>,
    extensions: Single<Entity, With<DiagnosticsPanelExtensions>>,
    theme: Res<thalos_runtime::ui::UiTheme>,
) {
    // F2 targets every ViewerUiRoot; F1 targets the ambient diagnostics panel
    // explicitly. The panel drives Node.display while both clean-view paths
    // drive Visibility, so neither races the requested-open state.
    commands.entity(*root).insert((
        thalos_runtime::viewer::ViewerUiRoot,
        thalos_runtime::photo_mode::HideInPhotoMode,
    ));
    commands.entity(*extensions).with_children(|sections| {
        spawn_text_section(sections, &theme, "STREAMING", KorsouBlock::Streaming);
        spawn_text_section(sections, &theme, "POSITION", KorsouBlock::Position);
        spawn_text_section(sections, &theme, "LIGHTING", KorsouBlock::Lighting);
    });
}

fn sync_gate(
    settings: Res<thalos_runtime::preferences::SettingsMenu>,
    viewpoints: Res<ViewpointUiState>,
    photo_mode: Res<thalos_runtime::photo_mode::PhotoMode>,
    mut gate: ResMut<DiagnosticsPanelGate>,
) {
    gate.available = !settings.open && !viewpoints.is_open() && !photo_mode.active;
}

#[allow(clippy::too_many_arguments)]
fn refresh_text(
    mut tick: Local<u8>,
    state: Res<DiagnosticsPanelState>,
    gate: Res<DiagnosticsPanelGate>,
    terrain: Res<TerrainStats>,
    foliage: Option<Res<FoliageStats>>,
    graphics: Res<thalos_runtime::preferences::GraphicsPreferences>,
    camera: Single<&TerrainCamera>,
    dataset: Res<TerrainDataset>,
    spatial: Res<TerrainSpatialFrame>,
    clock: Res<SolarClock>,
    mut texts: Query<(&KorsouBlock, &mut Text)>,
) {
    if !state.visible || !gate.available {
        return;
    }
    *tick = tick.wrapping_add(1);
    if *tick != 1 && !tick.is_multiple_of(REFRESH_EVERY_FRAMES) {
        return;
    }

    let streaming = format_streaming(&terrain, foliage.as_deref(), graphics.foliage);
    let position = format_position(&camera, &dataset, &spatial);
    let lighting = format_lighting(&clock);
    for (block, mut text) in &mut texts {
        let value = match block {
            KorsouBlock::Streaming => &streaming,
            KorsouBlock::Position => &position,
            KorsouBlock::Lighting => &lighting,
        };
        if text.0 != *value {
            text.0.clone_from(value);
        }
    }
}

fn format_lighting(clock: &SolarClock) -> String {
    let sun_elevation_deg = clock.sun_direction().y.asin().to_degrees();
    format!(
        "{} · {} AST\n\
         sun {sun_elevation_deg:+.1}° · {} at {:.0}×",
        format_ordinal_date(f32::from(clock.day_of_year)),
        format_local_time(clock.local_hours() as f32),
        if clock.running { "running" } else { "paused" },
        clock.rate,
    )
}

fn format_streaming(
    terrain: &TerrainStats,
    foliage: Option<&FoliageStats>,
    foliage_enabled: bool,
) -> String {
    let lods = terrain
        .by_lod
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(level, count)| format!("L{level} {count}"))
        .collect::<Vec<_>>()
        .join(" · ");
    let lods = if lods.is_empty() { "no LODs" } else { &lods };
    let foliage = match (foliage, foliage_enabled) {
        (Some(_), false) => "foliage off".to_string(),
        (Some(stats), true) if !stats.bake_ready => "foliage impostor bake pending".to_string(),
        (Some(stats), true) => format!(
            "foliage {}/{} · {} queued · {} plants · {} vertices\n\
             shadows {} cells · {} triangles",
            stats.resident,
            stats.desired,
            stats.queued,
            stats.woody_plants,
            stats.impostor_vertices,
            stats.shadow_cells,
            stats.shadow_triangles,
        ),
        (None, _) => "foliage unavailable in this spatial mode".to_string(),
    };
    format!(
        "terrain {}/{} · {} queued · {} morphing\n\
         {} triangles · {} dense\n\
         {lods}\n\
         {foliage}",
        terrain.resident,
        terrain.desired,
        terrain.queued,
        terrain.transitioning,
        terrain.triangles,
        terrain.dense_triangles,
    )
}

fn format_position(
    camera: &TerrainCamera,
    dataset: &TerrainDataset,
    spatial: &TerrainSpatialFrame,
) -> String {
    let ground_m = f64::from(dataset.dem_height(camera.position_m.x, camera.position_m.z));
    let utm = spatial.local_to_utm(camera.position_m);
    format!(
        "EPSG:32619 · E {:.1} · N {:.1}\n\
         local x {:+.1} · z {:+.1} m\n\
         altitude {:.1} · agl {:.1} m",
        utm.easting_m,
        utm.northing_m,
        camera.position_m.x,
        camera.position_m.z,
        camera.position_m.y,
        camera.position_m.y - ground_m,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_text_distinguishes_disabled_and_unavailable_foliage() {
        let terrain = TerrainStats {
            resident: 12,
            desired: 16,
            queued: 4,
            by_lod: [1, 3, 8, 0, 0, 0, 0],
            triangles: 2_048,
            dense_triangles: 8_192,
            transitioning: 2,
        };
        let foliage = FoliageStats {
            resident: 7,
            desired: 9,
            queued: 2,
            woody_plants: 1_024,
            impostor_vertices: 4_096,
            shadow_cells: 3,
            shadow_triangles: 768,
            bake_ready: true,
        };

        let enabled = format_streaming(&terrain, Some(&foliage), true);
        assert!(enabled.contains("terrain 12/16 · 4 queued · 2 morphing"));
        assert!(enabled.contains("L0 1 · L1 3 · L2 8"));
        assert!(enabled.contains("foliage 7/9 · 2 queued · 1024 plants · 4096 vertices"));
        assert!(enabled.contains("shadows 3 cells · 768 triangles"));
        assert!(format_streaming(&terrain, Some(&foliage), false).contains("foliage off"));
        assert!(format_streaming(&terrain, None, true).contains("foliage unavailable"));
    }

    #[test]
    fn lighting_text_exposes_the_clock_and_solar_elevation() {
        let clock = SolarClock {
            day_of_year: 222,
            local_seconds: 17.5 * 3_600.0,
            running: false,
            rate: 60.0,
        };

        let text = format_lighting(&clock);
        assert!(text.contains("Aug 10 · 17:30 AST"));
        assert!(text.contains("paused at 60×"));
        assert!(text.contains("sun "));
    }
}
