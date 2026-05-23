//! Player spawn situations.
//!
//! Three ways the player can start a session, selected by `just game [mode]`
//! (CLI arg) or the `THALOS_SPAWN` env var:
//!
//! - `orbit` (default): a ship in the authored low Thalos parking orbit.
//! - `eva`: the player on foot at the Thalos sub-stellar point.
//! - `landing`: a ship on a powered-descent approach, coming down over Thalos
//!   land. (Thalos has no atmosphere yet, so this is a vacuum / lunar-style
//!   suicide-burn descent — there is no aerobraking to lean on.)
//!
//! `orbit` and `eva` resolve fully in `main.rs` from the body state alone.
//! `landing` needs terrain data to place the ship *over land* at a true
//! above-ground altitude, neither of which is known until the bakes load — so
//! `main.rs` seeds the ship in the parking orbit (hidden behind the loading
//! screen) and [`refine_landing_spawn`] installs the real descent state on the
//! first `Running` frame, after searching the daylight hemisphere for a land
//! site. The loading gate guarantees terrain is resident by then.

use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;

use thalos_physics_canonical::canonical::{AuthorityMode, Epoch};
use thalos_physics_canonical::types::{AttitudeState, BodyState, StateVector};
use thalos_physics_local::{HeightSourceRegistry, TerrainSurfaceRegistry};
use thalos_terrain_render::HeightSource;

use crate::SimStage;
use crate::loading::AppState;
use crate::solar_system_state::SimulationState;

/// Which scenario the player is dropped into. Selected once at startup in
/// `main.rs`; inserted as a resource so deferred spawn finishers (today only
/// [`refine_landing_spawn`]) can tell which path they belong to.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnSituation {
    /// Ship in the low parking orbit (default).
    ShipOrbit,
    /// Player on foot at the sub-stellar point.
    Eva,
    /// Ship descending toward a landing site over land.
    Landing,
}

impl SpawnSituation {
    /// Parse the `just game [mode]` argument / `THALOS_SPAWN` value. Unknown
    /// values warn and fall back to the ship orbit.
    pub fn from_request(request: &str) -> Self {
        match request.trim().to_ascii_lowercase().as_str() {
            "eva" => Self::Eva,
            "land" | "landing" | "descent" => Self::Landing,
            "" | "orbit" | "ship" => Self::ShipOrbit,
            other => {
                eprintln!("  Unknown spawn mode '{other}'; defaulting to ship orbit.");
                Self::ShipOrbit
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Landing-approach tuning
// ---------------------------------------------------------------------------

/// Above-ground altitude (m) of the landing-approach spawn. Set just above the
/// 20 km Avian/Kepler handoff (`LocalBubbleConfig::handoff_agl_m`) so the ship
/// starts coasting on rails with a visible impact trajectory, then hands off to
/// the local-physics bubble for the powered touchdown as it sinks below 20 km.
const LANDING_APPROACH_ALTITUDE_M: f64 = 25_000.0;

/// Surface-relative descent rate (m/s) at spawn. Gentle — Thalos's surface
/// gravity is ~9 m/s² and there is no atmosphere to slow the fall, so the drop
/// builds speed fast on its own; the player still has ~60 s to set up the burn.
const LANDING_DESCENT_RATE_M_S: f64 = 60.0;

/// Surface-relative cross-track speed (m/s). A small lead so the approach reads
/// as flight over the terrain rather than a dead vertical drop, while staying
/// close enough to come down on the chosen site.
const LANDING_CROSS_TRACK_SPEED_M_S: f64 = 40.0;

/// Coarse LOD (m) for the land/ocean search. We want the baked macro height
/// that defines the coastline as seen from orbit, not fine procedural relief,
/// and most search directions fall back to the baked cubemap anyway (the GPU
/// height mirror only holds tiles near the camera).
const LANDING_SITE_QUERY_LOD_M: f32 = 2_000.0;

/// Reject landing sites poleward of this latitude (`|sin(lat)|`). Thalos's ice
/// caps begin near 68°; sin(55°) ≈ 0.82 keeps the player on bare ground with
/// margin. Latitude is measured from body-fixed +Y, matching the terrain
/// compiler's `asin(dir.y)` convention.
const LANDING_SITE_MAX_ABS_LAT_SIN: f32 = 0.82;

/// Required freeboard (m) above sea level for a site to count as dry land, so
/// the spawn isn't planted right on the waterline.
const LANDING_SITE_FREEBOARD_M: f32 = 50.0;

pub struct SpawnPlugin;

impl Plugin for SpawnPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            refine_landing_spawn
                .run_if(in_state(AppState::Running))
                .before(SimStage::Physics),
        );
    }
}

/// Build the canonical descent state for a powered landing approach toward a
/// body-fixed surface direction.
///
/// `dir_body_fixed` is the unit surface direction (body-fixed frame) the ship
/// comes down over; `terrain_h_m` is the rendered terrain height there. The
/// ship is placed [`LANDING_APPROACH_ALTITUDE_M`] above the local surface,
/// sinking at [`LANDING_DESCENT_RATE_M_S`] with a small surface-relative
/// cross-track lead, nose held retrograde to the surface-relative velocity so
/// the first throttle pulse decelerates the fall. Velocities are built in the
/// co-rotating surface frame so the player isn't fighting the planet's spin.
fn landing_approach_state(
    body_state: &BodyState,
    body_radius_m: f64,
    dir_body_fixed: DVec3,
    terrain_h_m: f64,
) -> (StateVector, AttitudeState) {
    let dir_body_fixed = dir_body_fixed.try_normalize().unwrap_or(DVec3::Y);
    // Radial-out unit in the inertial frame.
    let up = (body_state.orientation * dir_body_fixed)
        .try_normalize()
        .unwrap_or(DVec3::Y);
    let radius = body_radius_m + terrain_h_m + LANDING_APPROACH_ALTITUDE_M;
    let position = body_state.position + up * radius;
    let r_rel = position - body_state.position;

    // Velocity of the co-rotating surface point directly below.
    let surface_velocity = body_state.velocity + body_state.angular_velocity.cross(r_rel);

    // A surface tangent for the cross-track lead: local "east" (the direction
    // the surface rotates), falling back to an arbitrary tangent at the poles.
    let east = body_state
        .angular_velocity
        .cross(up)
        .try_normalize()
        .unwrap_or_else(|| {
            let seed = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
            (seed - up * seed.dot(up)).normalize()
        });

    let descent = -up * LANDING_DESCENT_RATE_M_S + east * LANDING_CROSS_TRACK_SPEED_M_S;
    let velocity = surface_velocity + descent;

    // Nose (+Y) retrograde to the surface-relative velocity (= `-descent`),
    // dorsal (+Z) toward radial-out — same convention as the orbit spawn.
    let nose = (-descent).try_normalize().unwrap_or(up);
    let dorsal = (up - nose * up.dot(nose))
        .try_normalize()
        .unwrap_or(east);
    let right = nose.cross(dorsal).normalize();
    let basis = DMat3::from_cols(right, nose, dorsal);
    let attitude = AttitudeState {
        orientation: DQuat::from_mat3(&basis),
        angular_velocity: DVec3::ZERO,
    };

    (StateVector { position, velocity }, attitude)
}

/// Search the daylight hemisphere around `sun_dir_body_fixed` for a body-fixed
/// surface direction that is dry land away from the ice caps. Returns the
/// closest such direction to the sub-stellar point, the highest point seen if
/// no dry land turns up, or `sun_dir_body_fixed` itself for an ocean-free body.
fn find_landing_site(
    height_source: &dyn HeightSource,
    sea_level_m: Option<f32>,
    sun_dir_body_fixed: DVec3,
) -> DVec3 {
    let sun = sun_dir_body_fixed.try_normalize().unwrap_or(DVec3::Y);
    // No ocean authored (airless body): the sub-stellar point is already land.
    let Some(sea_level_m) = sea_level_m else {
        return sun;
    };
    let land_threshold = sea_level_m + LANDING_SITE_FREEBOARD_M;

    // Orthonormal tangent basis around the sub-stellar axis.
    let t1 = {
        let seed = if sun.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
        (seed - sun * seed.dot(sun)).normalize()
    };
    let t2 = sun.cross(t1).normalize();

    // Spiral outward in concentric rings; the first dry-land hit is the nearest
    // land to local noon.
    const RINGS: usize = 24;
    const MAX_ANGLE_RAD: f64 = 1.30; // ~75° from sub-stellar — comfortably lit.
    let mut best_fallback: Option<(f32, DVec3)> = None;
    for ring in 0..=RINGS {
        let theta = MAX_ANGLE_RAD * ring as f64 / RINGS as f64;
        let (st, ct) = theta.sin_cos();
        // ~12° azimuth spacing, denser on wider rings.
        let spokes = ((st * 30.0).ceil() as usize).max(1);
        for spoke in 0..spokes {
            let phi = std::f64::consts::TAU * spoke as f64 / spokes as f64;
            let (sp, cp) = phi.sin_cos();
            let dir = (sun * ct + (t1 * cp + t2 * sp) * st)
                .try_normalize()
                .unwrap_or(sun);
            if dir.y.abs() as f32 > LANDING_SITE_MAX_ABS_LAT_SIN {
                continue;
            }
            let Some(h) = height_source.sample_height_m(dir.as_vec3(), LANDING_SITE_QUERY_LOD_M)
            else {
                continue;
            };
            if h > land_threshold {
                return dir;
            }
            if best_fallback.is_none_or(|(bh, _)| h > bh) {
                best_fallback = Some((h, dir));
            }
        }
    }
    best_fallback.map(|(_, d)| d).unwrap_or(sun)
}

/// Finalize the `Landing` spawn once terrain is resident (guaranteed by the
/// loading gate before `AppState::Running`). `main.rs` parked the ship in orbit
/// behind the loading screen; here we drop it onto a descent over the nearest
/// daylight land at a true above-ground altitude and leave it coasting on
/// rails. Runs exactly once.
fn refine_landing_spawn(
    mut done: Local<bool>,
    situation: Res<SpawnSituation>,
    mut sim: ResMut<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    surfaces: Res<TerrainSurfaceRegistry>,
) {
    if *done || *situation != SpawnSituation::Landing {
        return;
    }
    let body_id = sim.simulation.dominant_body();
    let Some(height_source) = height_sources.get(body_id) else {
        return; // Terrain not registered yet — retry next frame.
    };

    let body_radius_m = sim.system.bodies[body_id].radius_m;
    let epoch = Epoch(sim.simulation.sim_time());
    let body_state = sim.ephemeris.state(body_id, epoch);

    // Sub-stellar (daylight) direction in the body-fixed frame: Pyros sits at
    // the heliocentric origin, so `-body_position` points at the local noon.
    let sun_dir_inertial = (-body_state.position).normalize_or_zero();
    let sun_dir_body_fixed = if sun_dir_inertial == DVec3::ZERO {
        DVec3::Y
    } else {
        (body_state.orientation.inverse() * sun_dir_inertial).normalize()
    };

    let sea_level_m = surfaces
        .get(body_id)
        .and_then(|surface| surface.static_surface.sea_level_m);
    let site_dir = find_landing_site(height_source.as_ref(), sea_level_m, sun_dir_body_fixed);
    let terrain_h_m = height_source
        .sample_height_m(site_dir.as_vec3(), LANDING_SITE_QUERY_LOD_M)
        .unwrap_or(0.0) as f64;

    let (state, attitude) =
        landing_approach_state(&body_state, body_radius_m, site_dir, terrain_h_m);
    sim.simulation.set_ship_state(state);
    sim.simulation.set_attitude(attitude);
    // Coast on rails until the descent crosses the local-physics handoff band.
    sim.simulation
        .transition_authority(AuthorityMode::OnRails { trajectory: 0 });

    let lat_deg = site_dir.y.clamp(-1.0, 1.0).asin().to_degrees();
    info!(
        "landing approach: descending over {} at {:.0} km AGL, lat {:.0}°, ground {:.0} m",
        sim.system.bodies[body_id].name,
        LANDING_APPROACH_ALTITUDE_M / 1000.0,
        lat_deg,
        terrain_h_m,
    );
    *done = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;

    /// Minimal `HeightSource` for search tests: height is whatever the closure
    /// returns for a body-fixed direction.
    struct MockHeight<F>(F);

    impl<F: Fn(Vec3) -> Option<f32> + Send + Sync> HeightSource for MockHeight<F> {
        fn sample_height_m(&self, dir: Vec3, _tile_lod_m: f32) -> Option<f32> {
            (self.0)(dir)
        }
    }

    fn body_at(position: DVec3, radius_m: f64) -> BodyState {
        BodyState {
            id: 0,
            epoch: Epoch(0.0),
            position,
            velocity: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
            mass_kg: 1.0,
            gm: 1.0,
            radius_m,
        }
    }

    #[test]
    fn landing_state_sits_above_ground_and_descends_retrograde() {
        let radius_m = 1_000_000.0;
        let terrain_h_m = 1_000.0;
        let body = body_at(DVec3::ZERO, radius_m);
        let (state, attitude) = landing_approach_state(&body, radius_m, DVec3::X, terrain_h_m);

        // Placed exactly AGL above the local surface, radially over the site.
        let expected_radius = radius_m + terrain_h_m + LANDING_APPROACH_ALTITUDE_M;
        assert!((state.position.length() - expected_radius).abs() < 1.0);
        assert!(state.position.normalize().dot(DVec3::X) > 0.999_9);

        // Descending: the radial (up = +X) velocity component is the negative
        // descent rate; the rest is the cross-track lead.
        let up = DVec3::X;
        assert!((state.velocity.dot(up) + LANDING_DESCENT_RATE_M_S).abs() < 1e-6);
        let expected_speed =
            (LANDING_DESCENT_RATE_M_S.powi(2) + LANDING_CROSS_TRACK_SPEED_M_S.powi(2)).sqrt();
        assert!((state.velocity.length() - expected_speed).abs() < 1e-6);

        // Nose (+Y) points retrograde to the surface-relative velocity.
        let nose = attitude.orientation * DVec3::Y;
        let retrograde = (-state.velocity).normalize();
        assert!((nose - retrograde).length() < 1e-6);
        // Dorsal (+Z) leans toward radial-out, not into the ground.
        assert!((attitude.orientation * DVec3::Z).dot(up) > 0.0);
    }

    #[test]
    fn landing_state_cancels_surface_rotation() {
        // A spinning body: the spawn velocity must ride with the surface so the
        // surface-relative descent is exactly what we authored, not surface
        // speed plus descent.
        let radius_m = 1_000_000.0;
        let mut body = body_at(DVec3::ZERO, radius_m);
        body.angular_velocity = DVec3::Y * 1e-4; // spin about the pole
        let (state, _) = landing_approach_state(&body, radius_m, DVec3::X, 0.0);

        let r_rel = state.position; // body at origin
        let surface_velocity = body.angular_velocity.cross(r_rel);
        let surface_relative = state.velocity - surface_velocity;
        let up = DVec3::X;
        assert!((surface_relative.dot(up) + LANDING_DESCENT_RATE_M_S).abs() < 1e-6);
        let expected_speed =
            (LANDING_DESCENT_RATE_M_S.powi(2) + LANDING_CROSS_TRACK_SPEED_M_S.powi(2)).sqrt();
        assert!((surface_relative.length() - expected_speed).abs() < 1e-6);
    }

    #[test]
    fn find_site_returns_substellar_when_it_is_land() {
        let land = MockHeight(|dir: Vec3| {
            Some(if dir.dot(Vec3::X) > 0.5 { 5_000.0 } else { -3_000.0 })
        });
        let site = find_landing_site(&land, Some(0.0), DVec3::X);
        assert!((site - DVec3::X).length() < 1e-6);
    }

    #[test]
    fn find_site_walks_off_ocean_to_nearby_land() {
        // Sub-stellar point is ocean; land is a cap ~30° away in the X–Z plane.
        let land_dir = Vec3::new(30.0_f32.to_radians().cos(), 0.0, 30.0_f32.to_radians().sin());
        let height = MockHeight(move |dir: Vec3| {
            Some(if dir.dot(land_dir) > 0.9 { 4_000.0 } else { -2_000.0 })
        });
        let site = find_landing_site(&height, Some(0.0), DVec3::X);
        let h = height.sample_height_m(site.as_vec3(), 0.0).unwrap();
        assert!(h > 0.0, "expected the search to settle on land, got h = {h}");
    }

    #[test]
    fn find_site_skips_polar_land() {
        // Only the poles are above water; the latitude filter must refuse them
        // so the player never wakes up on the ice cap.
        let height = MockHeight(|dir: Vec3| {
            Some(if dir.y.abs() > 0.95 { 6_000.0 } else { -1_000.0 })
        });
        let site = find_landing_site(&height, Some(0.0), DVec3::X);
        assert!(
            site.y.abs() as f32 <= LANDING_SITE_MAX_ABS_LAT_SIN,
            "site latitude {} exceeded the ice-cap guard",
            site.y
        );
    }

    #[test]
    fn find_site_returns_substellar_for_ocean_free_body() {
        let height = MockHeight(|_dir: Vec3| Some(-9_999.0));
        let site = find_landing_site(&height, None, DVec3::X);
        assert!((site - DVec3::X).length() < 1e-6);
    }
}
