use std::{collections::HashSet, fs, path::Path};

use anyhow::{Context, Result, ensure};
use bevy::{
    math::{DVec2, DVec3},
    prelude::*,
};
use serde::Deserialize;
use thalos_geodetic::{GeographicPosition, wgs84_to_utm_north};

use crate::{
    camera::{TerrainCamera, TerrainCameraSet},
    spatial::{KORSOU_UTM_ZONE, TerrainSpatialFrame},
    terrain::TerrainDataset,
    viewpoint::ViewpointUiState,
};

const CATALOG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/places.json");
const WAYPOINT_PANEL_WIDTH_PX: f32 = 390.0;
const WAYPOINT_MARKER_SIZE_PX: f32 = 13.0;
const WAYPOINT_MARKER_LIFT_M: f64 = 18.0;

pub struct PlacesPlugin {
    interactive: bool,
}

impl PlacesPlugin {
    pub const fn new(interactive: bool) -> Self {
        Self { interactive }
    }
}

impl Plugin for PlacesPlugin {
    fn build(&self, app: &mut App) {
        let catalog = {
            let dataset = app.world().resource::<TerrainDataset>();
            let spatial = app.world().resource::<TerrainSpatialFrame>();
            PlaceCatalog::load(Path::new(CATALOG_PATH), dataset, spatial).unwrap_or_else(|error| {
                panic!("failed to load Kòrsou place catalog {CATALOG_PATH}: {error:#}")
            })
        };
        app.insert_resource(catalog)
            .init_resource::<PlaceState>()
            .add_systems(
                Update,
                update_current_area
                    .in_set(PlaceSet::Locate)
                    .after(TerrainCameraSet::Movement),
            );

        if self.interactive {
            app.add_systems(
                Startup,
                setup_waypoint_ui.after(thalos_runtime::ui::init_ui_theme),
            )
            .add_systems(
                Update,
                (
                    handle_waypoint_input.in_set(PlaceSet::Input),
                    sync_ui_display,
                    refresh_waypoint_hud.after(PlaceSet::Input),
                    project_waypoint_marker
                        .after(PlaceSet::Input)
                        .after(TerrainCameraSet::Projection),
                ),
            );
        }
    }
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaceSet {
    Locate,
    Input,
}

#[derive(Resource)]
pub struct PlaceCatalog {
    places: Vec<Place>,
    waypoint_indices: Vec<usize>,
}

impl PlaceCatalog {
    pub fn load(
        path: &Path,
        dataset: &TerrainDataset,
        spatial: &TerrainSpatialFrame,
    ) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let raw: RawPlaceCatalog =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        ensure!(raw.format_version == 1, "unsupported place catalog version");
        ensure!(raw.source.name == "OpenStreetMap");
        ensure!(!raw.source.snapshot_utc.is_empty());
        ensure!(!raw.source.attribution.is_empty());
        ensure!(raw.source.license == "ODbL 1.0");
        ensure!(!raw.source.url.is_empty());
        ensure!(!raw.source.method.is_empty());
        ensure!(!raw.places.is_empty(), "place catalog is empty");

        let bounds = dataset.metadata.quadtree.domain_bounds_local_m;
        let mut ids = HashSet::new();
        let mut orders = HashSet::new();
        let mut places = Vec::with_capacity(raw.places.len());
        for entry in raw.places {
            ensure!(!entry.id.is_empty() && ids.insert(entry.id.clone()));
            ensure!(!entry.name.is_empty());
            ensure!(entry.osm_id > 0);
            let _ = entry.osm_type;
            ensure!(
                entry.area_radius_m.is_some() == entry.area_specificity.is_some(),
                "{} must provide both area radius and specificity",
                entry.id
            );
            if let Some(radius_m) = entry.area_radius_m {
                ensure!(radius_m.is_finite() && radius_m > 0.0);
            }
            if let Some(order) = entry.waypoint_order {
                ensure!(orders.insert(order), "duplicate waypoint order {order}");
            }

            let geographic = GeographicPosition::new(entry.latitude_deg, entry.longitude_deg)
                .with_context(|| format!("invalid WGS84 coordinate for {}", entry.id))?;
            let utm = wgs84_to_utm_north(geographic, KORSOU_UTM_ZONE)
                .with_context(|| format!("project {} into UTM zone 19N", entry.id))?;
            let local_xz_m = spatial.utm_to_local_xz(utm);
            ensure!(
                local_xz_m.x >= bounds[0]
                    && local_xz_m.y >= bounds[1]
                    && local_xz_m.x <= bounds[2]
                    && local_xz_m.y <= bounds[3],
                "{} lies outside the Kòrsou terrain domain",
                entry.id
            );
            places.push(Place {
                id: entry.id,
                name: entry.name,
                kind: entry.kind,
                local_xz_m,
                area_radius_m: entry.area_radius_m,
                area_specificity: entry.area_specificity,
                waypoint_order: entry.waypoint_order,
            });
        }

        let mut waypoint_indices: Vec<_> = places
            .iter()
            .enumerate()
            .filter_map(|(index, place)| place.waypoint_order.map(|order| (order, index)))
            .collect();
        waypoint_indices.sort_unstable_by_key(|(order, _)| *order);
        let waypoint_indices: Vec<usize> = waypoint_indices
            .into_iter()
            .map(|(_, index)| index)
            .collect();
        ensure!(
            !waypoint_indices.is_empty(),
            "place catalog has no waypoints"
        );

        Ok(Self {
            places,
            waypoint_indices,
        })
    }

    pub fn find(&self, id: &str) -> Option<&Place> {
        self.places.iter().find(|place| place.id == id)
    }

    fn area_index_at(&self, local_xz_m: DVec2) -> Option<usize> {
        self.places
            .iter()
            .enumerate()
            .filter_map(|(index, place)| {
                let radius_m = place.area_radius_m?;
                let specificity = place.area_specificity?;
                let distance_squared = place.local_xz_m.distance_squared(local_xz_m);
                (distance_squared <= radius_m * radius_m).then_some((
                    index,
                    specificity,
                    distance_squared,
                ))
            })
            .max_by(|left, right| {
                left.1
                    .cmp(&right.1)
                    .then_with(|| right.2.total_cmp(&left.2))
            })
            .map(|(index, _, _)| index)
    }

    fn cycle_waypoint(&self, current: Option<usize>, direction: i32) -> Option<usize> {
        let count = self.waypoint_indices.len();
        if count == 0 {
            return None;
        }
        let position = current.and_then(|current| {
            self.waypoint_indices
                .iter()
                .position(|index| *index == current)
        });
        let next = match (position, direction.is_negative()) {
            (None, false) => 0,
            (None, true) => count - 1,
            (Some(position), false) => (position + 1) % count,
            (Some(position), true) => (position + count - 1) % count,
        };
        Some(self.waypoint_indices[next])
    }
}

pub struct Place {
    pub id: String,
    pub name: String,
    pub kind: PlaceKind,
    pub local_xz_m: DVec2,
    pub area_radius_m: Option<f64>,
    pub area_specificity: Option<u8>,
    waypoint_order: Option<u16>,
}

impl Place {
    fn kind_label(&self) -> &'static str {
        match self.kind {
            PlaceKind::Locality => "AREA",
            PlaceKind::Beach => "BEACH",
            PlaceKind::Bay => "BAY",
            PlaceKind::Landmark => "LANDMARK",
            PlaceKind::Lookout => "LOOKOUT",
        }
    }
}

#[derive(Resource, Default)]
pub struct PlaceState {
    current_area: Option<usize>,
    active_waypoint: Option<usize>,
}

impl PlaceState {
    pub fn current_area<'a>(&self, catalog: &'a PlaceCatalog) -> Option<&'a Place> {
        self.current_area
            .and_then(|index| catalog.places.get(index))
    }

    fn active_waypoint<'a>(&self, catalog: &'a PlaceCatalog) -> Option<&'a Place> {
        self.active_waypoint
            .and_then(|index| catalog.places.get(index))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PlaceKind {
    Locality,
    Beach,
    Bay,
    Landmark,
    Lookout,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OsmType {
    Node,
    Way,
    Relation,
}

#[derive(Deserialize)]
struct RawPlaceCatalog {
    format_version: u32,
    source: RawPlaceSource,
    places: Vec<RawPlace>,
}

#[derive(Deserialize)]
struct RawPlaceSource {
    name: String,
    snapshot_utc: String,
    attribution: String,
    license: String,
    url: String,
    method: String,
}

#[derive(Deserialize)]
struct RawPlace {
    id: String,
    name: String,
    kind: PlaceKind,
    latitude_deg: f64,
    longitude_deg: f64,
    area_radius_m: Option<f64>,
    area_specificity: Option<u8>,
    waypoint_order: Option<u16>,
    osm_type: OsmType,
    osm_id: u64,
}

fn update_current_area(
    camera: Single<&TerrainCamera>,
    catalog: Res<PlaceCatalog>,
    mut state: ResMut<PlaceState>,
) {
    let current = catalog.area_index_at(camera.position_m.xz());
    if state.current_area != current {
        state.current_area = current;
    }
}

fn handle_waypoint_input(
    keys: Res<ButtonInput<KeyCode>>,
    settings: Res<thalos_runtime::preferences::SettingsMenu>,
    viewpoints: Res<ViewpointUiState>,
    catalog: Res<PlaceCatalog>,
    mut state: ResMut<PlaceState>,
) {
    if settings.open || viewpoints.is_open() {
        return;
    }
    let next = if keys.just_pressed(KeyCode::BracketRight) {
        catalog.cycle_waypoint(state.active_waypoint, 1)
    } else if keys.just_pressed(KeyCode::BracketLeft) {
        catalog.cycle_waypoint(state.active_waypoint, -1)
    } else if keys.just_pressed(KeyCode::KeyX) {
        None
    } else {
        return;
    };
    if state.active_waypoint != next {
        state.active_waypoint = next;
    }
}

#[derive(Component)]
struct PlacesUiRoot;

#[derive(Component)]
struct WaypointMarker;

#[derive(Component, Clone, Copy)]
enum WaypointText {
    Name,
    Detail,
}

fn setup_waypoint_ui(mut commands: Commands, theme: Res<thalos_runtime::ui::UiTheme>) {
    commands
        .spawn((
            Node {
                display: Display::Flex,
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
            Visibility::Inherited,
            GlobalZIndex(70),
            thalos_runtime::viewer::ViewerUiRoot,
            thalos_runtime::photo_mode::HideInPhotoMode,
            PlacesUiRoot,
            Name::new("Kòrsou places UI"),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Percent(50.0),
                    top: Val::Px(16.0),
                    width: Val::Px(WAYPOINT_PANEL_WIDTH_PX),
                    margin: UiRect::left(Val::Px(-WAYPOINT_PANEL_WIDTH_PX * 0.5)),
                    padding: UiRect::axes(
                        Val::Px(thalos_runtime::ui::tokens::SPACE_LG),
                        Val::Px(thalos_runtime::ui::tokens::SPACE_MD),
                    ),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    row_gap: Val::Px(thalos_runtime::ui::tokens::SPACE_XS),
                    border_radius: BorderRadius::all(Val::Px(
                        thalos_runtime::ui::tokens::RADIUS_PANEL,
                    )),
                    ..default()
                },
                theme.glass(),
                Name::new("Waypoint readout"),
            ))
            .with_children(|panel| {
                panel.spawn(theme.heading("WAYPOINT"));
                let mut name = theme.body_strong("OFF");
                name.1.font_size = FontSize::Px(18.0);
                panel.spawn((name, WaypointText::Name));
                panel.spawn((
                    theme.mono_dim("Press [ or ] to choose a destination"),
                    WaypointText::Detail,
                ));
                panel.spawn(theme.faint("[ previous · ] next · X clear"));
            });

            root.spawn((
                Node {
                    display: Display::None,
                    position_type: PositionType::Absolute,
                    width: Val::Px(WAYPOINT_MARKER_SIZE_PX),
                    height: Val::Px(WAYPOINT_MARKER_SIZE_PX),
                    border: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Percent(50.0)),
                    ..default()
                },
                BackgroundColor(thalos_runtime::ui::tokens::ACCENT),
                BorderColor::all(Color::WHITE),
                WaypointMarker,
                Name::new("Waypoint marker"),
            ));
        });
}

fn sync_ui_display(
    settings: Res<thalos_runtime::preferences::SettingsMenu>,
    viewpoints: Res<ViewpointUiState>,
    mut roots: Query<&mut Node, With<PlacesUiRoot>>,
) {
    let display = if settings.open || viewpoints.is_open() {
        Display::None
    } else {
        Display::Flex
    };
    for mut node in &mut roots {
        if node.display != display {
            node.display = display;
        }
    }
}

fn refresh_waypoint_hud(
    camera: Single<&TerrainCamera>,
    catalog: Res<PlaceCatalog>,
    state: Res<PlaceState>,
    mut texts: Query<(&WaypointText, &mut Text)>,
) {
    let waypoint = state.active_waypoint(&catalog);
    let (name, detail) = waypoint.map_or_else(
        || {
            (
                "OFF".to_string(),
                "Press [ or ] to choose a destination".to_string(),
            )
        },
        |waypoint| {
            let (distance_m, bearing_deg) = waypoint_solution(camera.position_m.xz(), waypoint);
            (
                waypoint.name.clone(),
                format!(
                    "{} · {} · {:03.0}° {}",
                    waypoint.kind_label(),
                    format_distance(distance_m),
                    bearing_deg,
                    compass_point(bearing_deg),
                ),
            )
        },
    );
    for (block, mut text) in &mut texts {
        let value = match block {
            WaypointText::Name => &name,
            WaypointText::Detail => &detail,
        };
        if text.0 != *value {
            text.0.clone_from(value);
        }
    }
}

fn project_waypoint_marker(
    camera: Single<(&TerrainCamera, &Camera, &GlobalTransform)>,
    catalog: Res<PlaceCatalog>,
    state: Res<PlaceState>,
    dataset: Res<TerrainDataset>,
    spatial: Res<TerrainSpatialFrame>,
    mut marker: Single<&mut Node, With<WaypointMarker>>,
) {
    let Some(waypoint) = state.active_waypoint(&catalog) else {
        marker.display = Display::None;
        return;
    };
    let local = DVec3::new(
        waypoint.local_xz_m.x,
        f64::from(dataset.dem_height(waypoint.local_xz_m.x, waypoint.local_xz_m.y))
            + WAYPOINT_MARKER_LIFT_M,
        waypoint.local_xz_m.y,
    );
    let world = spatial.project(local).as_vec3();
    let Ok(screen) = camera.1.world_to_viewport(camera.2, world) else {
        marker.display = Display::None;
        return;
    };
    let Some(viewport) = camera.1.logical_viewport_size() else {
        marker.display = Display::None;
        return;
    };
    if screen.x < 0.0 || screen.y < 0.0 || screen.x > viewport.x || screen.y > viewport.y {
        marker.display = Display::None;
        return;
    }
    marker.display = Display::Flex;
    marker.left = Val::Px(screen.x - WAYPOINT_MARKER_SIZE_PX * 0.5);
    marker.top = Val::Px(screen.y - WAYPOINT_MARKER_SIZE_PX * 0.5);
}

fn waypoint_solution(camera_xz_m: DVec2, waypoint: &Place) -> (f64, f64) {
    let delta = waypoint.local_xz_m - camera_xz_m;
    let east_m = delta.x;
    let north_m = -delta.y;
    let bearing_deg = east_m.atan2(north_m).to_degrees().rem_euclid(360.0);
    (delta.length(), bearing_deg)
}

fn format_distance(distance_m: f64) -> String {
    if distance_m < 1_000.0 {
        format!("{distance_m:.0} m")
    } else {
        format!("{:.1} km", distance_m / 1_000.0)
    }
}

fn compass_point(bearing_deg: f64) -> &'static str {
    const POINTS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    let index = ((bearing_deg / 45.0).round() as usize) % POINTS.len();
    POINTS[index]
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::cli::SpatialMode;

    fn dataset() -> TerrainDataset {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        TerrainDataset::load(&asset_dir).unwrap()
    }

    fn catalog(dataset: &TerrainDataset) -> PlaceCatalog {
        let spatial = TerrainSpatialFrame::new(dataset, SpatialMode::Planar).unwrap();
        PlaceCatalog::load(Path::new(CATALOG_PATH), dataset, &spatial).unwrap()
    }

    #[test]
    fn catalog_coordinates_round_trip_through_the_runtime_frame() {
        let dataset = dataset();
        let catalog = catalog(&dataset);
        let spatial = TerrainSpatialFrame::new(&dataset, SpatialMode::Planar).unwrap();
        let raw: RawPlaceCatalog =
            serde_json::from_slice(&fs::read(CATALOG_PATH).unwrap()).unwrap();
        for source in raw.places {
            let place = catalog.find(&source.id).unwrap();
            let round_trip =
                spatial.local_to_wgs84(DVec3::new(place.local_xz_m.x, 0.0, place.local_xz_m.y));
            assert!(
                (round_trip.latitude_deg - source.latitude_deg).abs() < 1.0e-7,
                "{} latitude moved",
                place.id
            );
            assert!(
                (round_trip.longitude_deg - source.longitude_deg).abs() < 1.0e-7,
                "{} longitude moved",
                place.id
            );
        }
    }

    #[test]
    fn specific_areas_win_inside_broader_regions() {
        let dataset = dataset();
        let catalog = catalog(&dataset);
        for expected in ["punda", "otrobanda", "caracasbaai", "westpunt"] {
            let place = catalog.find(expected).unwrap();
            let actual = catalog
                .area_index_at(place.local_xz_m)
                .and_then(|index| catalog.places.get(index))
                .unwrap();
            assert_eq!(actual.id, expected);
        }
    }

    #[test]
    fn waypoint_cycle_is_ordered_and_wraps() {
        let dataset = dataset();
        let catalog = catalog(&dataset);
        let first = catalog.cycle_waypoint(None, 1).unwrap();
        assert_eq!(catalog.places[first].id, "queen-emma-bridge");
        let last = catalog.cycle_waypoint(None, -1).unwrap();
        assert_eq!(catalog.places[last].id, "playa-kalki");
        assert_eq!(catalog.cycle_waypoint(Some(last), 1), Some(first));
        assert_eq!(catalog.cycle_waypoint(Some(first), -1), Some(last));
    }

    #[test]
    fn waypoint_bearing_uses_north_as_zero_and_east_as_ninety() {
        let mut place = Place {
            id: "test".into(),
            name: "Test".into(),
            kind: PlaceKind::Landmark,
            local_xz_m: DVec2::new(0.0, -1_000.0),
            area_radius_m: None,
            area_specificity: None,
            waypoint_order: Some(1),
        };
        assert_eq!(waypoint_solution(DVec2::ZERO, &place).1, 0.0);
        place.local_xz_m = DVec2::new(1_000.0, 0.0);
        assert_eq!(waypoint_solution(DVec2::ZERO, &place).1, 90.0);
    }
}
