use bevy::prelude::*;
use thalos_physics_canonical::canonical::Epoch;

use crate::SimStage;

// The evaluated-solar-system vocabulary moved to
// `thalos_game_state::solar_system` (Phase 5a); this module keeps the
// sole writer and the plugin.
pub use thalos_game_state::solar_system::{
    BodyEnvironmentState, CloudBandEnvironmentState, SimulationState, SolarSystemState,
};

// The weather cube — `CloudWeatherField`, its derivation, and the coverage
// trim — lives in `thalos_weather::cloud_cube` (Phase 5a move,
// ADR-20260731T024003Z). Re-exported here so existing consumers (the spawn
// wiring, the `cloud_weather_probe` example) keep their import paths.
pub use thalos_weather::cloud_cube::{
    CLOUD_WEATHER_FACE_SIZE, CloudWeatherField, SurfaceDensityTrace, WeatherTraceSample,
    cloud_surface_density_traced,
};

/// The one coverage trim, owned by the producer (`thalos_weather::cloud_cube`).
pub const CLOUD_COVERAGE_SCALE: f32 = thalos_weather::cloud_cube::COVERAGE_SCALE;

pub fn sync_solar_system_state(
    sim: Res<SimulationState>,
    mut solar_system: ResMut<SolarSystemState>,
) {
    let epoch = Epoch(sim.simulation.sim_time());
    if solar_system.states.is_some() && (solar_system.time - epoch.0).abs() < f64::EPSILON {
        return;
    }

    if let Some(states) = solar_system.states.as_mut() {
        sim.ephemeris.states_into(epoch, states);
    } else {
        let mut states = Vec::with_capacity(sim.ephemeris.body_count());
        sim.ephemeris.states_into(epoch, &mut states);
        solar_system.states = Some(states);
    }
    solar_system.time = epoch.0;
    solar_system.ensure_body_capacity(sim.ephemeris.body_count());
}

pub struct SolarSystemStatePlugin;

impl Plugin for SolarSystemStatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SolarSystemState>().add_systems(
            Update,
            sync_solar_system_state
                .in_set(SimStage::Sync)
                .in_set(thalos_game_state::sched::SolarSystemSyncSet),
        );
    }
}

#[cfg(test)]
mod cloud_site_probe {
    use super::*;
    use thalos_terrain::cubemap::{CubemapFace, face_uv_to_dir};

    /// Dev probe for the BL-20260723T214730Z thickness-parity protocol: scan
    /// the authored Thalos weather field for *cloudy* sites near the runway's
    /// daylight longitude, so tier A/B captures can frame real cloud (the
    /// default spaceport column is authored nearly clear). Prints
    /// `THALOS_RUNWAY_SITE` candidates.
    ///
    /// Run: `cargo test -p thalos_runtime --lib cloud_site_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "dev probe: prints cloudy THALOS_RUNWAY_SITE candidates"]
    fn print_cloudy_sites() {
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets");
        let system = thalos_world::parsing::load_solar_system_from_dir(&assets)
            .expect("load authored solar system");
        let thalos_id = system.name_to_id["Thalos"];
        let climate = system.bodies[thalos_id]
            .terrestrial_atmosphere
            .as_ref()
            .and_then(|atmosphere| atmosphere.clouds.clone())
            .expect("Thalos authored cloud climate");
        let field = CloudWeatherField::from_climate(&climate);

        // 2°x2° bins over the runway's daylight window (lon near 178°).
        const LAT_MIN: f32 = -45.0;
        const LAT_MAX: f32 = 45.0;
        const LON_MIN: f32 = 150.0;
        const LON_MAX: f32 = 206.0;
        const BIN_DEG: f32 = 2.0;
        let lat_bins = ((LAT_MAX - LAT_MIN) / BIN_DEG) as usize;
        let lon_bins = ((LON_MAX - LON_MIN) / BIN_DEG) as usize;
        #[derive(Clone, Copy, Default)]
        struct Bin {
            n: u32,
            cov: f64,
            sd_col: f64,
            cloudy: u32,
        }
        let mut bins = vec![Bin::default(); lat_bins * lon_bins];
        let size = field.face_size as usize;
        for (face_index, face) in CubemapFace::ALL.into_iter().enumerate() {
            for y in (0..size).step_by(2) {
                let v = (y as f32 + 0.5) / size as f32;
                for x in (0..size).step_by(2) {
                    let u = (x as f32 + 0.5) / size as f32;
                    let dir = face_uv_to_dir(face, u, v).normalize();
                    let lat = dir.y.asin().to_degrees();
                    let lon = dir.z.atan2(dir.x).to_degrees().rem_euclid(360.0);
                    if !(LAT_MIN..LAT_MAX).contains(&lat) || !(LON_MIN..LON_MAX).contains(&lon) {
                        continue;
                    }
                    let bin = &mut bins[((lat - LAT_MIN) / BIN_DEG) as usize * lon_bins
                        + ((lon - LON_MIN) / BIN_DEG) as usize];
                    let index = face_index * size * size + y * size + x;
                    let weather = field.texels[index];
                    let strata = field.surface_density_texels[index];
                    let cov = f64::from(weather[0]) / 255.0;
                    let col = strata
                        .iter()
                        .map(|&s| f64::from(s) / 255.0)
                        .fold(0.0f64, f64::max);
                    bin.n += 1;
                    bin.cov += cov;
                    bin.sd_col += col;
                    bin.cloudy += u32::from(cov > 0.25);
                }
            }
        }

        // Rank by "broken moderate field" suitability: mean column strata near
        // 0.42 with substantial (but not total) cloudy-texel fraction.
        let mut ranked: Vec<(f32, f32, f64, f64, f64)> = Vec::new();
        for (i, bin) in bins.iter().enumerate() {
            if bin.n < 32 {
                continue;
            }
            let lat = LAT_MIN + (i / lon_bins) as f32 * BIN_DEG + BIN_DEG * 0.5;
            let lon = LON_MIN + (i % lon_bins) as f32 * BIN_DEG + BIN_DEG * 0.5;
            let n = f64::from(bin.n);
            ranked.push((
                lat,
                lon,
                bin.cov / n,
                bin.sd_col / n,
                f64::from(bin.cloudy) / n,
            ));
        }
        ranked.sort_by(|a, b| {
            // Broken moderate field wanted: real cloudy texels, mid strata.
            let score = |r: &(f32, f32, f64, f64, f64)| (r.3 - 0.42).abs() - 0.6 * r.4.min(0.6);
            score(a).total_cmp(&score(b))
        });

        // Local sun elevation at the runway morning boot epoch, so candidates
        // are known-daylit before spending a cold capture on them.
        use thalos_physics_canonical::body_trajectory_provider::BodyTrajectoryProvider;
        let provider =
            thalos_physics_canonical::patched_conics::PatchedConics::new(&system, 3.156e11);
        let states = provider.states(Epoch(59_100.0));
        let star = states.first().map(|s| s.position).unwrap_or_default();
        let thalos_state = &states[thalos_id];
        let sun_elevation_deg = |lat_deg: f32, lon_deg: f32| -> f64 {
            let lat = f64::from(lat_deg).to_radians();
            let lon = f64::from(lon_deg).to_radians();
            let dir_body =
                bevy::math::DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin());
            let up_world = thalos_state.orientation * dir_body;
            let to_sun =
                (star - (thalos_state.position + up_world * thalos_state.radius_m)).normalize();
            90.0 - up_world.angle_between(to_sun).to_degrees()
        };

        println!("lat, lon, mean_cov, mean_col_strata, cloudy_frac, sun_elev_deg");
        for (lat, lon, cov, sd, cloudy) in ranked.iter().take(30) {
            println!(
                "{lat:7.1} {lon:7.1}   {cov:5.3}   {sd:5.3}   {cloudy:5.3}   {:6.1}",
                sun_elevation_deg(*lat, *lon)
            );
        }
        // Reference: the default runway site's bin.
        let default_bin = &bins[((7.6 - LAT_MIN) / BIN_DEG) as usize * lon_bins
            + ((178.0 - LON_MIN) / BIN_DEG) as usize];
        if default_bin.n > 0 {
            let n = f64::from(default_bin.n);
            println!(
                "default site (7.6, 178.0): cov {:5.3} col_strata {:5.3} cloudy_frac {:5.3}",
                default_bin.cov / n,
                default_bin.sd_col / n,
                f64::from(default_bin.cloudy) / n,
            );
        }
    }

    /// Dev probe: run the shared fill derivation on the real Thalos field and
    /// print the fitted curve + far response, without booting a capture.
    /// The per-bin convergence table lands in the log output (init a
    /// subscriber below so it prints).
    ///
    /// Run: `cargo test -p thalos_runtime --lib derive_fill -- --ignored --nocapture`
    #[test]
    #[ignore = "dev probe: prints the derived cloud fill calibration"]
    fn derive_fill_calibration_probe() {
        let subscriber = tracing_subscriber_fmt();
        let _guard = tracing::subscriber::set_default(subscriber);
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets");
        let system = thalos_world::parsing::load_solar_system_from_dir(&assets)
            .expect("load authored solar system");
        let thalos_id = system.name_to_id["Thalos"];
        let body = &system.bodies[thalos_id];
        let climate = body
            .terrestrial_atmosphere
            .as_ref()
            .and_then(|atmosphere| atmosphere.clouds.clone())
            .expect("Thalos authored cloud climate");
        let field = CloudWeatherField::from_climate(&climate);
        let start = std::time::Instant::now();
        let calibration = crate::rendering::derive_body_fill_calibration_for_probe(
            &field,
            &climate,
            body.radius_m as f32,
        );
        println!(
            "derived in {:?}: threshold nodes {:?}\nfar_response {:?}",
            start.elapsed(),
            calibration.threshold_nodes,
            calibration.far_response,
        );

        // Cross-check the CPU mirror against the pixel-measured tier A/B at
        // the measurement site (22.0 N, 153.0 E, ~15 km crop).
        let lat = 22.0f32.to_radians();
        let lon = 153.0f32.to_radians();
        let site = Vec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin());
        let climate_bottom = climate.base_altitude_m.max(0.0);
        let input = thalos_body_render::FillCalibrationInput {
            weather_texels: &field.texels,
            strata_texels: &field.surface_density_texels,
            face_size: field.face_size,
            coverage_scale: crate::rendering::clouds::COVERAGE_SCALE,
            density: 0.0026 * climate.density.max(0.0),
            detail_strength: 0.16,
            base_edge_softness: 0.055,
            bottom_softness: 0.16,
            base_shape_scale_m: climate.base_shape_scale_m.max(500.0),
            detail_scale_m: climate.detail_scale_m.max(50.0),
            bottom_height_m: climate_bottom,
            top_height_m: (climate.base_altitude_m + climate.thickness_m).max(climate_bottom + 1.0),
            planet_radius_m: body.radius_m as f32,
            seed: field.seed,
        };
        for radius_km in [8.0f32, 20.0, 60.0] {
            let cos_radius = (radius_km * 1000.0 / body.radius_m as f32).cos();
            let stats = thalos_body_render::fill_lut::predict_region_fill(
                &input,
                &calibration,
                site,
                cos_radius,
                4000,
            );
            println!("site prediction r={radius_km} km: {stats:?}");
        }
    }

    fn tracing_subscriber_fmt() -> impl tracing::Subscriber + Send + Sync {
        use bevy::log::tracing_subscriber::{self, layer::SubscriberExt};
        tracing_subscriber::registry().with(tracing_subscriber::fmt::layer())
    }
}
