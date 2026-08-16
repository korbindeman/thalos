use std::path::PathBuf;

use bevy::{
    math::{DQuat, DVec2, DVec3},
    prelude::*,
};
use thalos_geodetic::UtmPosition;

use crate::{
    camera::{TerrainCamera, TerrainCameraSet, preset_pose},
    places::PlaceCatalog,
    spatial::{KORSOU_UTM_ZONE, TerrainSpatialFrame},
    terrain::TerrainDataset,
};

pub use thalos_runtime::viewer::{
    Viewpoint, ViewpointSet, ViewpointStartupSet, ViewpointStore as ViewpointLibrary,
    ViewpointUiState,
};

const KORSOU_VIEWPOINT_FRAME: &str = "EPSG:32619-curacao-local";

#[derive(Clone, Copy)]
struct CoastalReference {
    name: &'static str,
    anchor: CoastalAnchor,
    camera_altitude_m: f64,
    camera_offshore_m: f64,
    camera_alongshore_m: f64,
    target_inland_m: f64,
    target_height_m: f64,
}

#[derive(Clone, Copy)]
enum CoastalAnchor {
    Place(&'static str),
    /// A composition-specific survey point that is not interchangeable with
    /// the named place's representative catalog coordinate.
    Utm([f64; 2]),
}

const COASTAL_REFERENCES: [CoastalReference; 3] = [
    CoastalReference {
        name: "Reference - Grote Knip beach",
        anchor: CoastalAnchor::Place("grote-knip"),
        camera_altitude_m: 180.0,
        camera_offshore_m: 450.0,
        camera_alongshore_m: -180.0,
        target_inland_m: 65.0,
        target_height_m: 18.0,
    },
    CoastalReference {
        name: "Reference - Boka Tabla cliffs",
        anchor: CoastalAnchor::Place("boka-tabla"),
        camera_altitude_m: 240.0,
        camera_offshore_m: 650.0,
        camera_alongshore_m: 420.0,
        target_inland_m: 70.0,
        target_height_m: 14.0,
    },
    CoastalReference {
        name: "Reference - Blue Bay reef",
        // Blauwbaai reef survey site, 12°08.063'N, 68°59.138'W. The OSM
        // beach centroid is a different semantic point and changes which
        // side of the narrow bay the camera occupies.
        anchor: CoastalAnchor::Utm([501_563.16, 1_341_413.59]),
        camera_altitude_m: 220.0,
        camera_offshore_m: 620.0,
        camera_alongshore_m: -280.0,
        target_inland_m: 35.0,
        target_height_m: 18.0,
    },
];

const NORTH_COAST_WAVE_REFERENCE: CoastalReference = CoastalReference {
    name: "Reference - North coast waves",
    anchor: CoastalAnchor::Place("boka-tabla"),
    camera_altitude_m: 8.0,
    camera_offshore_m: 1_600.0,
    camera_alongshore_m: -500.0,
    target_inland_m: -5_000.0,
    target_height_m: 4.0,
};

pub struct ViewpointPlugin {
    path: PathBuf,
    interactive: bool,
}

impl ViewpointPlugin {
    pub fn new(path: PathBuf, interactive: bool) -> Self {
        Self { path, interactive }
    }
}

impl Plugin for ViewpointPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(
            thalos_runtime::viewer::ViewpointPlugin::new(self.path.clone(), self.interactive)
                .with_legacy_projected(KORSOU_VIEWPOINT_FRAME),
        )
        .add_systems(
            Startup,
            install_default_viewpoints.in_set(ViewpointStartupSet::Defaults),
        )
        .add_systems(
            Update,
            project_current_viewpoint.in_set(ViewpointSet::Snapshot),
        )
        .add_systems(
            Update,
            apply_viewpoint
                .in_set(ViewpointSet::Apply)
                .after(TerrainCameraSet::Movement),
        );
    }
}

fn install_default_viewpoints(
    dataset: Res<TerrainDataset>,
    spatial: Res<TerrainSpatialFrame>,
    places: Res<PlaceCatalog>,
    mut fallbacks: ResMut<thalos_runtime::viewer::ViewpointFallbacks>,
) {
    fallbacks.0 = default_viewpoints(&dataset, &spatial, &places);
}

fn project_current_viewpoint(
    camera: Single<(&TerrainCamera, &thalos_runtime::viewer::CameraOptics)>,
    dataset: Res<TerrainDataset>,
    mut current: ResMut<thalos_runtime::viewer::CurrentViewpoint>,
) {
    let agl_m = camera.0.position_m.y
        - f64::from(dataset.dem_height(camera.0.position_m.x, camera.0.position_m.z));
    let suggested_name = if agl_m.is_finite() {
        if agl_m < 1_000.0 {
            format!("Curaçao {} m", (agl_m.max(0.0) / 10.0).round() as i64 * 10)
        } else {
            format!("Curaçao {} km", (agl_m.max(0.0) / 1_000.0).round() as i64)
        }
    } else {
        "Curaçao".into()
    };
    let snapshot = thalos_runtime::viewer::ViewpointSnapshot {
        frame: thalos_runtime::viewer::ViewpointFrame::ProjectedLocal {
            reference: KORSOU_VIEWPOINT_FRAME.into(),
        },
        camera_position_m: camera.0.position_m.to_array(),
        camera_rotation_xyzw: camera.0.rotation_local.to_array(),
        optics: camera.1.spec(),
        suggested_name,
    };
    if current.0.as_ref() != Some(&snapshot) {
        current.0 = Some(snapshot);
    }
}

fn apply_viewpoint(
    mut pending: ResMut<thalos_runtime::viewer::PendingViewpointApply>,
    mut state: ResMut<ViewpointUiState>,
    mut camera: Single<(
        &mut TerrainCamera,
        &mut thalos_runtime::viewer::CameraOptics,
    )>,
) {
    let Some(target) = pending.take() else {
        return;
    };
    let result = match target {
        thalos_runtime::viewer::ViewpointApplyTarget::Saved(viewpoint) => {
            let (controller, optics) = &mut *camera;
            apply_saved_viewpoint(&viewpoint, controller, optics)
                .map(|()| format!("Viewing {}", viewpoint.id))
        }
        thalos_runtime::viewer::ViewpointApplyTarget::Scripted(viewpoint) => Err(format!(
            "Kòrsou does not provide scripted viewpoint driver {:?}",
            viewpoint.driver
        )),
    };
    state.report(result);
}

fn apply_saved_viewpoint(
    viewpoint: &Viewpoint,
    camera: &mut TerrainCamera,
    optics: &mut thalos_runtime::viewer::CameraOptics,
) -> Result<(), String> {
    match &viewpoint.frame {
        thalos_runtime::viewer::ViewpointFrame::ProjectedLocal { reference }
            if reference == KORSOU_VIEWPOINT_FRAME => {}
        thalos_runtime::viewer::ViewpointFrame::ProjectedLocal { reference } => {
            return Err(format!(
                "viewpoint {} uses projected frame {reference:?}, expected {KORSOU_VIEWPOINT_FRAME:?}",
                viewpoint.id
            ));
        }
        thalos_runtime::viewer::ViewpointFrame::AuthoredBodyFixed { .. } => {
            return Err(format!(
                "viewpoint {} is body-fixed and cannot be applied in Kòrsou",
                viewpoint.id
            ));
        }
    }
    camera.position_m = DVec3::from_array(viewpoint.camera_position_m);
    camera.rotation_local = DQuat::from_array(viewpoint.camera_rotation_xyzw).normalize();
    let (yaw, pitch, _) = camera.rotation_local.to_euler(EulerRot::YXZ);
    camera.yaw = yaw;
    camera.pitch = pitch;
    optics.set_spec(viewpoint.optics)?;
    Ok(())
}

fn default_viewpoints(
    dataset: &TerrainDataset,
    spatial: &TerrainSpatialFrame,
    places: &PlaceCatalog,
) -> Vec<Viewpoint> {
    let mut viewpoints: Vec<_> = [
        (2, "West low flight"),
        (1, "Island aerial"),
        (3, "East low flight"),
    ]
    .into_iter()
    .map(|(preset, name)| {
        let (position, target) = preset_pose(dataset, preset);
        viewpoint_looking_at(name, position, target)
    })
    .collect();
    viewpoints.extend(COASTAL_REFERENCES.map(|reference| {
        let (position, target) = coastal_reference_pose(dataset, spatial, places, reference);
        viewpoint_looking_at(reference.name, position, target)
    }));
    viewpoints.push(caracasbaai_close_reference());
    let reference = NORTH_COAST_WAVE_REFERENCE;
    let (position, target) = coastal_reference_pose(dataset, spatial, places, reference);
    viewpoints.push(viewpoint_looking_at(reference.name, position, target));
    for (index, viewpoint) in viewpoints.iter_mut().enumerate() {
        viewpoint.saved_unix_ms = index as u128;
    }
    viewpoints
}

fn viewpoint_looking_at(name: &str, position_m: DVec3, target_m: DVec3) -> Viewpoint {
    let rotation = Transform::from_translation(position_m.as_vec3())
        .looking_at(target_m.as_vec3(), Vec3::Y)
        .rotation
        .as_dquat();
    Viewpoint {
        id: thalos_runtime::viewer::viewpoint_id_from_name(name),
        name: name.into(),
        description: String::new(),
        saved_unix_ms: 0,
        frame: thalos_runtime::viewer::ViewpointFrame::ProjectedLocal {
            reference: KORSOU_VIEWPOINT_FRAME.into(),
        },
        camera_position_m: position_m.to_array(),
        camera_rotation_xyzw: rotation.to_array(),
        optics: thalos_runtime::viewer::CameraOptics::default().spec(),
    }
}

fn caracasbaai_close_reference() -> Viewpoint {
    let position = DVec3::new(
        8_893.890_319_693_666,
        2.918_633_222_674_541,
        14_708.662_248_452_052,
    );
    let rotation = DQuat::from_euler(
        EulerRot::YXZ,
        (-274.216_189_628_456_45_f64).to_radians(),
        (-3.357_673_278_525_876_f64).to_radians(),
        0.0,
    );
    Viewpoint {
        id: "reference-caracasbaai-close-coast".into(),
        name: "Reference - Caracasbaai close coast".into(),
        description: String::new(),
        saved_unix_ms: 0,
        frame: thalos_runtime::viewer::ViewpointFrame::ProjectedLocal {
            reference: KORSOU_VIEWPOINT_FRAME.into(),
        },
        camera_position_m: position.to_array(),
        camera_rotation_xyzw: rotation.to_array(),
        optics: thalos_runtime::viewer::CameraOptics::default().spec(),
    }
}

fn coastal_reference_pose(
    dataset: &TerrainDataset,
    spatial: &TerrainSpatialFrame,
    places: &PlaceCatalog,
    reference: CoastalReference,
) -> (DVec3, DVec3) {
    let anchor = match reference.anchor {
        CoastalAnchor::Place(place_id) => {
            places
                .find(place_id)
                .unwrap_or_else(|| panic!("missing coastal reference place {place_id}"))
                .local_xz_m
        }
        CoastalAnchor::Utm([easting_m, northing_m]) => spatial.utm_to_local_xz(
            UtmPosition::new_north(KORSOU_UTM_ZONE, easting_m, northing_m)
                .expect("authored Kòrsou reference must be valid UTM zone 19N"),
        ),
    };
    let shoreline = nearest_shoreline_point(dataset, anchor, 900.0);
    let gradient = shore_gradient(dataset, shoreline);
    let landward = if gradient.length_squared() > 1.0e-6 {
        gradient.normalize()
    } else {
        DVec2::X
    };
    let alongshore = DVec2::new(-landward.y, landward.x);
    let camera_xz = shoreline - landward * reference.camera_offshore_m
        + alongshore * reference.camera_alongshore_m;
    let target_xz = shoreline + landward * reference.target_inland_m;
    let target_ground_m = if dataset.is_land(target_xz.x, target_xz.y) {
        f64::from(dataset.dem_height(target_xz.x, target_xz.y)).max(0.0)
    } else {
        0.0
    };
    (
        DVec3::new(camera_xz.x, reference.camera_altitude_m, camera_xz.y),
        DVec3::new(
            target_xz.x,
            target_ground_m + reference.target_height_m,
            target_xz.y,
        ),
    )
}

fn nearest_shoreline_point(dataset: &TerrainDataset, anchor: DVec2, radius_m: f64) -> DVec2 {
    let spacing = dataset.metadata.coastline.distance_spacing_m;
    let radius_steps = (radius_m / spacing).ceil() as i32;
    let mut best = anchor;
    let mut best_distance = f64::INFINITY;
    let mut best_offset_squared = f64::INFINITY;
    for z in -radius_steps..=radius_steps {
        for x in -radius_steps..=radius_steps {
            let offset = DVec2::new(f64::from(x) * spacing, f64::from(z) * spacing);
            if offset.length_squared() > radius_m * radius_m {
                continue;
            }
            let candidate = anchor + offset;
            let distance = f64::from(dataset.shore_distance_m(candidate.x, candidate.y).abs());
            let offset_squared = offset.length_squared();
            if distance < best_distance
                || (distance == best_distance && offset_squared < best_offset_squared)
            {
                best = candidate;
                best_distance = distance;
                best_offset_squared = offset_squared;
            }
        }
    }

    for _ in 0..6 {
        let distance = f64::from(dataset.shore_distance_m(best.x, best.y));
        let gradient = shore_gradient(dataset, best);
        let length_squared = gradient.length_squared();
        if distance.abs() < 0.05 || length_squared < 1.0e-6 {
            break;
        }
        best -= gradient * (distance / length_squared);
    }
    best
}

fn shore_gradient(dataset: &TerrainDataset, point: DVec2) -> DVec2 {
    let step = dataset.metadata.coastline.distance_spacing_m;
    let west = f64::from(dataset.shore_distance_m(point.x - step, point.y));
    let east = f64::from(dataset.shore_distance_m(point.x + step, point.y));
    let north = f64::from(dataset.shore_distance_m(point.x, point.y - step));
    let south = f64::from(dataset.shore_distance_m(point.x, point.y + step));
    DVec2::new(east - west, south - north) / (2.0 * step)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_viewpoint(frame: thalos_runtime::viewer::ViewpointFrame) -> Viewpoint {
        let rotation = DQuat::from_rotation_z(0.4);
        Viewpoint {
            id: "test-view".into(),
            name: "Test view".into(),
            description: String::new(),
            saved_unix_ms: 1,
            frame,
            camera_position_m: [1.0, 2.0, 3.0],
            camera_rotation_xyzw: rotation.to_array(),
            optics: thalos_runtime::viewer::CameraOptics::default().spec(),
        }
    }

    fn spatial(dataset: &TerrainDataset) -> TerrainSpatialFrame {
        TerrainSpatialFrame::new(dataset, crate::cli::SpatialMode::Planar).unwrap()
    }

    fn place_catalog(dataset: &TerrainDataset, spatial: &TerrainSpatialFrame) -> PlaceCatalog {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/places.json");
        PlaceCatalog::load(&path, dataset, spatial).unwrap()
    }

    #[test]
    fn projected_adapter_replays_exact_position_rotation_and_optics() {
        let viewpoint = test_viewpoint(thalos_runtime::viewer::ViewpointFrame::ProjectedLocal {
            reference: KORSOU_VIEWPOINT_FRAME.into(),
        });
        let mut camera = TerrainCamera {
            position_m: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            rotation_local: DQuat::IDENTITY,
        };
        let mut optics = thalos_runtime::viewer::CameraOptics::default();

        apply_saved_viewpoint(&viewpoint, &mut camera, &mut optics).unwrap();

        assert_eq!(camera.position_m, DVec3::new(1.0, 2.0, 3.0));
        assert!(camera.rotation_local.dot(DQuat::from_rotation_z(0.4)).abs() > 1.0 - 1.0e-12);
        assert_eq!(optics.spec(), viewpoint.optics);
    }

    #[test]
    fn projected_adapter_rejects_a_foreign_frame() {
        let viewpoint = test_viewpoint(thalos_runtime::viewer::ViewpointFrame::ProjectedLocal {
            reference: "EPSG:3857".into(),
        });
        let mut camera = TerrainCamera {
            position_m: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            rotation_local: DQuat::IDENTITY,
        };
        let mut optics = thalos_runtime::viewer::CameraOptics::default();

        assert!(apply_saved_viewpoint(&viewpoint, &mut camera, &mut optics).is_err());
    }

    #[test]
    fn coastal_reference_viewpoints_stay_anchored_to_the_shoreline() {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        let spatial = spatial(&dataset);
        let places = place_catalog(&dataset, &spatial);
        let defaults = default_viewpoints(&dataset, &spatial, &places);

        for reference in COASTAL_REFERENCES
            .into_iter()
            .chain([NORTH_COAST_WAVE_REFERENCE])
        {
            let viewpoint = defaults
                .iter()
                .find(|viewpoint| viewpoint.name == reference.name)
                .unwrap();
            viewpoint.validate().unwrap();

            let anchor = match reference.anchor {
                CoastalAnchor::Place(place_id) => places.find(place_id).unwrap().local_xz_m,
                CoastalAnchor::Utm([easting_m, northing_m]) => spatial.utm_to_local_xz(
                    UtmPosition::new_north(KORSOU_UTM_ZONE, easting_m, northing_m).unwrap(),
                ),
            };
            let shoreline = nearest_shoreline_point(&dataset, anchor, 900.0);
            assert!(
                dataset.shore_distance_m(shoreline.x, shoreline.y).abs() < 0.25,
                "{} no longer resolves to the shoreline",
                reference.name
            );

            let (position, target) = coastal_reference_pose(&dataset, &spatial, &places, reference);
            assert!(
                dataset.shore_distance_m(position.x, position.z) < 0.0,
                "{} camera must remain over water",
                reference.name
            );
            if reference.target_inland_m > 0.0 {
                assert!(
                    dataset.shore_distance_m(target.x, target.z) > 0.0,
                    "{} must look toward land",
                    reference.name
                );
            } else {
                assert!(
                    dataset.shore_distance_m(target.x, target.z) < 0.0,
                    "{} must look toward open water",
                    reference.name
                );
            }
        }

        let caracasbaai = defaults
            .iter()
            .find(|viewpoint| viewpoint.name == "Reference - Caracasbaai close coast")
            .unwrap();
        caracasbaai.validate().unwrap();
        let position = DVec3::from_array(caracasbaai.camera_position_m);
        assert!(position.y < 5.0, "Caracasbaai must remain a waterline view");
        let shore_distance_m = dataset.shore_distance_m(position.x, position.z);
        assert!(
            shore_distance_m.abs() < dataset.metadata.coastline.distance_clamp_m as f32,
            "Caracasbaai camera must remain inside the measured coastal strip"
        );
    }

    #[test]
    fn startup_viewpoint_stays_close_to_the_island() {
        let project_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dataset = TerrainDataset::load(&project_dir.join("assets/terrain/curacao")).unwrap();
        let spatial = spatial(&dataset);
        let places = place_catalog(&dataset, &spatial);
        let checked_in =
            thalos_runtime::viewer::read_viewpoint_catalog(&project_dir.join("viewpoints.json"))
                .unwrap()
                .viewpoints;
        let generated = default_viewpoints(&dataset, &spatial, &places);
        for viewpoints in [&checked_in, &generated] {
            let startup = viewpoints.first().unwrap();
            let position = DVec3::from_array(startup.camera_position_m);
            let ground = f64::from(dataset.dem_height(position.x, position.z));
            assert!(
                position.y - ground <= 1_000.0,
                "{} starts {:.0} m above terrain",
                startup.name,
                position.y - ground
            );
        }
    }
}
