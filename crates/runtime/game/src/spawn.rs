//! Player spawn situations.
//!
//! Ways the player can start a session, selected by `just game [mode]`
//! (CLI arg) or the `THALOS_SPAWN` env var:
//!
//! - `orbit` (default): a ship in the authored low Thalos parking orbit
//!   (equatorial).
//! - `polar`: same parking altitude as `orbit`, but a polar (i ≈ 90°) path.
//! - `eva`: the player on foot at the Thalos sub-stellar point.
//! - `landing`: a ship on a powered-descent approach, coming down over Thalos
//!   land. (Thalos has no atmosphere yet, so this is a vacuum / lunar-style
//!   suicide-burn descent — there is no aerobraking to lean on.)
//! - `final`: the same ship already on final approach, very low over a flat
//!   dry patch of Thalos.
//! - `cruise`: the Meridian aircraft at ~15,000 ft (~4,600 m AGL), flying
//!   level at cruise speed over dry land.
//! - `launch`: the Saturn rocket standing vertically on the default spaceport's
//!   launchpad, engines cold and ready for staging.
//!
//! `orbit`, `polar`, and `eva` resolve fully in `main.rs` from the body state
//! alone.
//! `landing`, `final`, and `cruise` need terrain data to place the ship *over
//! land* at a true above-ground altitude, neither of which is known until the
//! bakes load — so `main.rs` seeds the ship in the parking orbit (hidden behind
//! the loading screen) and [`refine_descent_spawn`] installs the real state on
//! the first `Running` frame, after searching the daylight hemisphere for a land
//! site. The loading gate guarantees terrain is resident by then.

use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;

use thalos_body_render::HeightSource;
use thalos_physics_canonical::canonical::{AuthorityMode, Epoch};
use thalos_physics_canonical::types::{AttitudeState, BodyState};
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry};
use thalos_world::{BodyId, StateVector};

use crate::SimStage;
use crate::loading::AppState;
use crate::solar_system_state::SimulationState;

/// The canonical placement of the player craft: where it is, how it's oriented,
/// and which authority owns it.
pub(crate) struct CraftPlacement {
    pub state: StateVector,
    pub attitude: AttitudeState,
    pub authority: AuthorityMode,
}

/// The **one canonical way** to seat the player craft's canonical state — every
/// spawn / respawn / relaunch / teleport / launch path routes through here, so
/// the ordering invariant lives in exactly one place instead of being
/// re-assembled (and occasionally mis-assembled) at each site.
///
/// The order is the invariant: if `teardown` is given, tear down the live Avian
/// bubble **first** — so [`local_physics::spawn_player_avian_body`] rebuilds it
/// seeded from the *placed* pose next frame instead of fighting a bubble still
/// carrying the pre-teleport velocity (the class of jitter / "buzzing" bug that
/// every teleport that forgot to clear the bubble produced). Then set the
/// canonical state, attitude, and authority.
///
/// Pass `teardown = Some((commands, active))` for a runtime teleport of an
/// existing craft; `None` at boot / before any bubble exists, or where the caller
/// tears the bubble down itself. Site-specific extras (throttle, target body,
/// engine lighting, gear/brake state) stay at the call site — this owns only the
/// canonical-state ritual.
pub(crate) fn place_craft(
    sim: &mut SimulationState,
    placement: CraftPlacement,
    teardown: Option<(&mut Commands, &mut ActiveLocalBubble)>,
) {
    if let Some((commands, active)) = teardown {
        crate::scenario_menu::clear_bubble(commands, active);
    }
    sim.simulation.set_ship_state(placement.state);
    sim.simulation.set_attitude(placement.attitude);
    sim.simulation.transition_authority(placement.authority);
}

/// A [`CraftPlacement`] for a craft coasting on rails — the common orbit /
/// airborne / approach case (`OnRails` authority at trajectory 0). Landed poses
/// (`BodyFixed`) build their `CraftPlacement` at the call site since the pose is
/// derived from the state.
pub(crate) fn coast_placement(state: StateVector, attitude: AttitudeState) -> CraftPlacement {
    CraftPlacement {
        state,
        attitude,
        authority: AuthorityMode::OnRails { trajectory: 0 },
    }
}

/// Whether a fresh session resumes at 1× immediately instead of holding the
/// paused-on-spawn default.
///
/// **Every** spawn situation starts paused (warp 0×) so the player gets a beat
/// to orient before sim time advances. Setting `THALOS_AUTO_RUN` (truthy) flips
/// this on so the sim is live at 1× the instant the loading screen clears; this
/// exists mainly for unattended runs that want motion without first issuing a
/// warp input.
///
/// Sole consumer: [`apply_initial_warp`].
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct AutoRun {
    pub enabled: bool,
}

impl AutoRun {
    /// Read `THALOS_AUTO_RUN`. Truthy = `1` / `true` / `yes` / `on`
    /// (case-insensitive); anything else, or unset, keeps the paused default.
    pub fn from_env() -> Self {
        let enabled = std::env::var("THALOS_AUTO_RUN")
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        Self { enabled }
    }
}

// `SpawnSituation` moved to `thalos_game_state::scenario` (Phase 5b);
// placement, site search, and descent tuning stay in this module.
pub use thalos_game_state::scenario::SpawnSituation;

/// Descent tuning for the scenarios the deferred descent finisher places.
/// (Was a private method on `SpawnSituation`; the enum moved to the
/// blackboard and the tuning stayed here with its consumer.)
fn descent_profile(situation: SpawnSituation) -> Option<DescentProfile> {
    match situation {
        SpawnSituation::Landing => Some(LANDING_PROFILE),
        SpawnSituation::FinalApproach => Some(FINAL_APPROACH_PROFILE),
        SpawnSituation::Cruise => Some(CRUISE_PROFILE),
        _ => None,
    }
}

/// The body the authored parking-orbit / on-foot scenarios are anchored to
/// (Thalos, or the first non-star fallback). Resolved once at startup in
/// `main.rs` and stored so the destruction scenario menu can rebuild the
/// "ship in orbit" start without re-deriving the homeworld.
#[derive(Resource, Debug, Clone, Copy)]
pub struct Homeworld(pub BodyId);

/// Absolute parking-orbit state + attitude for the authored ship scenario.
///
/// `ship_rel` is the homeworld-relative state authored in
/// `system.ship.initial_state`; `homeworld_state` is the homeworld's
/// heliocentric state at the spawn epoch. Nose (+Y) along prograde, dorsal
/// (+Z) radial-out — the shared "level orbital flight" convention used by the
/// navball and control axes. Shared by `main.rs` (startup) and the destruction
/// scenario menu (respawn) so both produce the identical orbit.
/// The Thalos debug parking orbit at the current epoch — the `just game`
/// default ship start, rebuilt for a destruction respawn or an editor Launch.
/// Shared so a respawn / relaunch into orbit matches the `just game` boot.
pub(crate) fn orbit_respawn_state(
    sim: &SimulationState,
    homeworld: BodyId,
) -> (StateVector, AttitudeState) {
    let epoch = Epoch(sim.simulation.sim_time());
    let homeworld_state = sim.ephemeris.state(homeworld, epoch);
    let rel = thalos_physics_canonical::debug_orbits::debug_parking_orbit_relative_state(
        &sim.system.bodies[homeworld],
    );
    orbit_parking_state(rel, &homeworld_state)
}

/// Polar variant of [`orbit_respawn_state`] — same altitude, inclination 90°.
/// Shared by `just game polar`, the start-screen shortcut, and destruction
/// respawn so all three stay identical.
pub(crate) fn polar_orbit_respawn_state(
    sim: &SimulationState,
    homeworld: BodyId,
) -> (StateVector, AttitudeState) {
    let epoch = Epoch(sim.simulation.sim_time());
    let homeworld_state = sim.ephemeris.state(homeworld, epoch);
    let rel = thalos_physics_canonical::debug_orbits::debug_polar_parking_orbit_relative_state(
        &sim.system.bodies[homeworld],
    );
    orbit_parking_state(rel, &homeworld_state)
}

pub(crate) fn orbit_parking_state(
    ship_rel: StateVector,
    homeworld_state: &BodyState,
) -> (StateVector, AttitudeState) {
    let state = StateVector {
        position: homeworld_state.position + ship_rel.position,
        velocity: homeworld_state.velocity + ship_rel.velocity,
    };
    let prograde = ship_rel.velocity.normalize();
    let radial = ship_rel.position.normalize();
    let dorsal = (radial - radial.dot(prograde) * prograde).normalize();
    let right = prograde.cross(dorsal);
    let basis = DMat3::from_cols(right, prograde, dorsal);
    let attitude = AttitudeState {
        orientation: DQuat::from_mat3(&basis),
        angular_velocity: DVec3::ZERO,
    };
    (state, attitude)
}

// ---------------------------------------------------------------------------
// Landing-approach tuning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct DescentProfile {
    label: &'static str,
    altitude_m: f64,
    descent_rate_m_s: f64,
    cross_track_speed_m_s: f64,
    surface_query_lod_m: f32,
    site_search: SiteSearch,
    /// When true the nose (+Y) points *forward* along the velocity vector
    /// instead of retrograde. Used for the cruise scenario.
    nose_forward: bool,
}

#[derive(Debug, Clone, Copy)]
enum SiteSearch {
    NearestDryLand,
    FlatDryPatch {
        probe_radius_m: f64,
        max_relief_m: f32,
        distance_penalty_m_per_rad: f32,
    },
}

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

const LANDING_PROFILE: DescentProfile = DescentProfile {
    label: "landing approach",
    altitude_m: LANDING_APPROACH_ALTITUDE_M,
    descent_rate_m_s: LANDING_DESCENT_RATE_M_S,
    cross_track_speed_m_s: LANDING_CROSS_TRACK_SPEED_M_S,
    surface_query_lod_m: LANDING_SITE_QUERY_LOD_M,
    site_search: SiteSearch::NearestDryLand,
    nose_forward: false,
};

/// Very low final-approach altitude (m). This starts inside the local-physics
/// handoff band, so terrain contact is live almost immediately.
const FINAL_APPROACH_ALTITUDE_M: f64 = 1_500.0;

/// Surface-relative descent rate (m/s) for `just game final`. Low enough to
/// give a short flare window, high enough that the touchdown practice starts
/// with real urgency.
const FINAL_APPROACH_DESCENT_RATE_M_S: f64 = 12.0;

/// Surface-relative cross-track speed (m/s) for final approach. This keeps the
/// craft moving visibly across the chosen flats without outrunning them.
const FINAL_APPROACH_CROSS_TRACK_SPEED_M_S: f64 = 18.0;

/// LOD hint (m) for final-approach site and height samples. Finer than the
/// ordinary landing search so the chosen "flat" patch is useful near touchdown.
const FINAL_APPROACH_SITE_QUERY_LOD_M: f32 = 250.0;

/// Metric radius sampled around a candidate final-approach site to decide
/// whether the local terrain patch is flat.
const FINAL_APPROACH_FLAT_PROBE_RADIUS_M: f64 = 1_000.0;

/// Desired height relief (m) across the flatness probe. If no daylight dry site
/// meets this, the least-bad dry site still wins.
const FINAL_APPROACH_MAX_RELIEF_M: f32 = 25.0;

/// Gentle preference for nearer-to-noon sites when flatness is similar. The
/// flatness score is height relief in metres plus `theta * penalty`.
const FINAL_APPROACH_DISTANCE_PENALTY_M_PER_RAD: f32 = 8.0;

const FINAL_APPROACH_PROFILE: DescentProfile = DescentProfile {
    label: "final approach",
    altitude_m: FINAL_APPROACH_ALTITUDE_M,
    descent_rate_m_s: FINAL_APPROACH_DESCENT_RATE_M_S,
    cross_track_speed_m_s: FINAL_APPROACH_CROSS_TRACK_SPEED_M_S,
    surface_query_lod_m: FINAL_APPROACH_SITE_QUERY_LOD_M,
    site_search: SiteSearch::FlatDryPatch {
        probe_radius_m: FINAL_APPROACH_FLAT_PROBE_RADIUS_M,
        max_relief_m: FINAL_APPROACH_MAX_RELIEF_M,
        distance_penalty_m_per_rad: FINAL_APPROACH_DISTANCE_PENALTY_M_PER_RAD,
    },
    nose_forward: false,
};

/// Cruise altitude (m AGL) — 15,000 ft rounded slightly.
const CRUISE_ALTITUDE_M: f64 = 4_600.0;

/// Cruise speed (m/s) — ~160 m/s (~Mach 0.5 at sea level) for the Meridian.
const CRUISE_SPEED_M_S: f64 = 160.0;

/// LOD for the cruise site search — coarse enough to find coastlines and pick
/// a dry-land site, fine enough to avoid landing over a lone reef.
const CRUISE_SITE_QUERY_LOD_M: f32 = 2_000.0;

const CRUISE_PROFILE: DescentProfile = DescentProfile {
    label: "cruise",
    altitude_m: CRUISE_ALTITUDE_M,
    descent_rate_m_s: 0.0,
    cross_track_speed_m_s: CRUISE_SPEED_M_S,
    surface_query_lod_m: CRUISE_SITE_QUERY_LOD_M,
    site_search: SiteSearch::NearestDryLand,
    nose_forward: true,
};

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

/// Sea-level datum (m above the reference radius). The runtime
/// `ProceduralSurface` has no water layer; its continent-mask shoreline sits at
/// the reference radius, so sea level is height 0.
const SEA_LEVEL_M: f32 = 0.0;

/// Re-armable trigger for the deferred descent/cruise placement, mirroring
/// [`crate::runway::RunwayPlacement`]. Armed at startup when the boot
/// scenario is a descent; **not** armed by runtime scenario starts (the start
/// screen and the destruction picker place descents synchronously via
/// [`compute_descent_state`], so arming this would double-place). The
/// explicit flag exists because [`SpawnSituation`] is mutable at runtime now —
/// a `Local<bool>` keyed off the resource would fire spuriously after a
/// scenario switch.
#[derive(Resource, Debug, Default)]
pub struct DescentPlacement {
    pub pending: bool,
}

pub struct SpawnPlugin;

impl Plugin for SpawnPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DescentPlacement>()
            .add_systems(Startup, arm_boot_descent_placement)
            .add_systems(
                Update,
                // Runs during `AppState::Loading` (not gated on `Running`) so the
                // descent is placed and the camera reaches the approach altitude
                // behind the loading screen, letting terrain stream in before the
                // reveal. Self-gated by `DescentPlacement` + terrain residency.
                refine_descent_spawn.before(SimStage::Physics),
            )
            // Single source of truth for the initial warp level, applied once
            // when the loading screen clears (after every deferred placement
            // has run).
            .add_systems(OnEnter(AppState::Running), apply_initial_warp);
    }
}

/// Arm the deferred placement for a descent boot scenario.
fn arm_boot_descent_placement(
    situation: Res<SpawnSituation>,
    mut placement: ResMut<DescentPlacement>,
) {
    placement.pending = situation.is_descent();
}

/// Apply the paused-on-spawn policy once the loading screen clears.
///
/// Every situation spawns paused (warp 0×); this resumes to 1× only when
/// [`AutoRun`] is set. Centralising it on the `Loading → Running` transition
/// keeps the deferred placement flows (runway, descent) from each having to
/// decide a warp level — they install canonical state and leave the clock
/// alone, and this fires after all of them so it always has the last word.
fn apply_initial_warp(auto_run: Res<AutoRun>, mut sim: ResMut<SimulationState>) {
    if auto_run.enabled {
        sim.simulation.warp.reset_immediate();
    } else {
        sim.simulation.warp.pause_immediate();
    }
}

/// Build the canonical descent state for a powered descent approach toward a
/// body-fixed surface direction.
///
/// `dir_body_fixed` is the unit surface direction (body-fixed frame) the ship
/// comes down over; `terrain_h_m` is the rendered terrain height there. The
/// ship is placed above the local surface according to `profile`, sinking with
/// a small surface-relative cross-track lead, nose held retrograde to the
/// surface-relative velocity so the first throttle pulse decelerates the fall.
/// Velocities are built in the co-rotating surface frame so the player isn't
/// fighting the planet's spin.
fn descent_approach_state(
    body_state: &BodyState,
    body_radius_m: f64,
    dir_body_fixed: DVec3,
    terrain_h_m: f64,
    profile: DescentProfile,
) -> (StateVector, AttitudeState) {
    let dir_body_fixed = dir_body_fixed.try_normalize().unwrap_or(DVec3::Y);
    // Radial-out unit in the inertial frame.
    let up = (body_state.orientation * dir_body_fixed)
        .try_normalize()
        .unwrap_or(DVec3::Y);
    let radius = body_radius_m + terrain_h_m + profile.altitude_m;
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

    let descent = -up * profile.descent_rate_m_s + east * profile.cross_track_speed_m_s;
    let velocity = surface_velocity + descent;

    // Nose (+Y): forward (prograde) for cruise, retrograde for descents.
    // Dorsal (+Z) toward radial-out — same convention as the orbit spawn.
    let flight_dir = descent.try_normalize().unwrap_or(east);
    let nose = if profile.nose_forward {
        flight_dir
    } else {
        -flight_dir
    };
    let dorsal = (up - nose * up.dot(nose)).try_normalize().unwrap_or(east);
    let right = nose.cross(dorsal).normalize();
    let basis = DMat3::from_cols(right, nose, dorsal);
    let attitude = AttitudeState {
        orientation: DQuat::from_mat3(&basis),
        angular_velocity: DVec3::ZERO,
    };

    (StateVector { position, velocity }, attitude)
}

/// Search the daylight hemisphere around `sun_dir_body_fixed` for a body-fixed
/// surface direction that is dry land away from the ice caps. The ordinary
/// landing profile returns the closest dry direction to the sub-stellar point;
/// the final-approach profile scores dry directions by local relief so it can
/// start over a flat patch. If no dry land turns up, returns the highest point
/// seen; for ocean-free bodies it returns `sun_dir_body_fixed`.
fn find_landing_site(
    height_source: &dyn HeightSource,
    sea_level_m: Option<f32>,
    sun_dir_body_fixed: DVec3,
    body_radius_m: f64,
    profile: DescentProfile,
) -> DVec3 {
    let sun = sun_dir_body_fixed.try_normalize().unwrap_or(DVec3::Y);
    // No ocean authored (airless body): the sub-stellar point is already land.
    let Some(sea_level_m) = sea_level_m else {
        return sun;
    };
    let land_threshold = sea_level_m + LANDING_SITE_FREEBOARD_M;

    // Orthonormal tangent basis around the sub-stellar axis.
    let t1 = {
        let seed = if sun.y.abs() < 0.9 {
            DVec3::Y
        } else {
            DVec3::X
        };
        (seed - sun * seed.dot(sun)).normalize()
    };
    let t2 = sun.cross(t1).normalize();

    // Spiral outward in concentric rings; the first dry-land hit is the nearest
    // land to local noon.
    const RINGS: usize = 24;
    const MAX_ANGLE_RAD: f64 = 1.30; // ~75° from sub-stellar — comfortably lit.
    let mut best_fallback: Option<(f32, DVec3)> = None;
    let mut best_flat: Option<(f32, f32, DVec3)> = None;
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
            let Some(h) = height_source.sample_height_m(dir.as_vec3(), profile.surface_query_lod_m)
            else {
                continue;
            };
            if h > land_threshold {
                match profile.site_search {
                    SiteSearch::NearestDryLand => return dir,
                    SiteSearch::FlatDryPatch {
                        probe_radius_m,
                        max_relief_m,
                        distance_penalty_m_per_rad,
                    } => {
                        let relief_m = sample_site_relief_m(
                            height_source,
                            dir,
                            body_radius_m,
                            probe_radius_m,
                            profile.surface_query_lod_m,
                        )
                        .unwrap_or(f32::INFINITY);
                        let score = relief_m + theta as f32 * distance_penalty_m_per_rad;
                        if relief_m <= max_relief_m {
                            return dir;
                        }
                        if best_flat.is_none_or(|(best_score, _, _)| score < best_score) {
                            best_flat = Some((score, relief_m, dir));
                        }
                    }
                }
            }
            if best_fallback.is_none_or(|(bh, _)| h > bh) {
                best_fallback = Some((h, dir));
            }
        }
    }
    if let Some((_, _, dir)) = best_flat {
        return dir;
    }
    best_fallback.map(|(_, d)| d).unwrap_or(sun)
}

pub(crate) fn sample_site_relief_m(
    height_source: &dyn HeightSource,
    dir_body_fixed: DVec3,
    body_radius_m: f64,
    probe_radius_m: f64,
    tile_lod_m: f32,
) -> Option<f32> {
    let center_dir = dir_body_fixed.try_normalize()?;
    let center_h = height_source.sample_height_m(center_dir.as_vec3(), tile_lod_m)?;
    let center_radius_m = body_radius_m + center_h as f64;
    let seed = if center_dir.y.abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let tangent_x = seed.cross(center_dir).normalize();
    let tangent_z = tangent_x.cross(center_dir).normalize();

    let mut min_h = center_h;
    let mut max_h = center_h;
    // Eight compass probes around the site: axis-aligned plus the diagonals at
    // ±1/√2 so each sample sits on the unit circle at `probe_radius_m`.
    const D: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const PROBES: &[(f64, f64)] = &[
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (D, D),
        (-D, D),
        (D, -D),
        (-D, -D),
    ];
    for &(x, z) in PROBES {
        let tangent_point = center_dir * center_radius_m
            + tangent_x * (x * probe_radius_m)
            + tangent_z * (z * probe_radius_m);
        let probe_dir = tangent_point.try_normalize()?;
        let h = height_source.sample_height_m(probe_dir.as_vec3(), tile_lod_m)?;
        min_h = min_h.min(h);
        max_h = max_h.max(h);
    }
    Some(max_h - min_h)
}

/// Finalize deferred descent spawns once terrain is resident (guaranteed by the
/// loading gate before `AppState::Running`). `main.rs` parked the ship in orbit
/// behind the loading screen; here we drop it onto a descent over the nearest
/// daylight land (or flat daylight land for final approach) at a true
/// above-ground altitude and leave it coasting on rails. Runs exactly once.
fn refine_descent_spawn(
    mut placement: ResMut<DescentPlacement>,
    situation: Res<SpawnSituation>,
    mut sim: ResMut<SimulationState>,
    mut settle: ResMut<crate::surface_settle::SurfaceSettle>,
    mut tracker: ResMut<crate::loading::LoadingTracker>,
    height_sources: Res<HeightSourceRegistry>,
) {
    if !placement.pending || descent_profile(*situation).is_none() {
        return;
    }
    let Some((state, attitude)) = compute_descent_state(*situation, &sim, &height_sources) else {
        return; // Terrain not registered yet — retry next frame.
    };
    // Coast on rails until the descent crosses the local-physics handoff band.
    // No bubble teardown: this is a deferred boot placement behind the loading
    // screen, before any Avian bubble exists.
    place_craft(&mut sim, coast_placement(state, attitude), None);
    placement.pending = false;
    // Surface state installed: start the tile-settle timer at the site and
    // release the loading screen's placement gate.
    settle.mark_placed();
    tracker.complete(crate::loading::step::PLACEMENT);
}

/// Build the descent state for `situation` (a `Landing` / `FinalApproach`
/// scenario) over the current dominant body, or `None` if the situation isn't
/// a descent or terrain isn't resident yet. Searches the daylight hemisphere
/// for a dry, ice-free site (flat for final approach), then places the ship at
/// the profile's above-ground altitude, sinking nose-retrograde.
///
/// Shared by [`refine_descent_spawn`] (deferred startup placement) and the
/// destruction scenario menu (respawn), so a respawn into "landing" matches the
/// `just game landing` start exactly.
pub(crate) fn compute_descent_state(
    situation: SpawnSituation,
    sim: &SimulationState,
    height_sources: &HeightSourceRegistry,
) -> Option<(StateVector, AttitudeState)> {
    let profile = descent_profile(situation)?;
    let body_id = sim.simulation.dominant_body();
    let height_source = height_sources.get(body_id)?;

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

    // Sea level is the reference radius (height 0) — the runtime
    // `ProceduralSurface` has no separate water layer.
    let sea_level_m = Some(SEA_LEVEL_M);
    let site_dir = find_landing_site(
        height_source.as_ref(),
        sea_level_m,
        sun_dir_body_fixed,
        body_radius_m,
        profile,
    );
    let terrain_h_m = height_source
        .sample_height_m(site_dir.as_vec3(), profile.surface_query_lod_m)
        .unwrap_or(0.0) as f64;

    let (state, attitude) =
        descent_approach_state(&body_state, body_radius_m, site_dir, terrain_h_m, profile);

    let lat_deg = site_dir.y.clamp(-1.0, 1.0).asin().to_degrees();
    info!(
        "{}: descending over {} at {:.1} km AGL, lat {:.0}°, ground {:.0} m",
        profile.label,
        sim.system.bodies[body_id].name,
        profile.altitude_m / 1000.0,
        lat_deg,
        terrain_h_m,
    );
    Some((state, attitude))
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
    fn launch_request_selects_saturn_and_deferred_spaceport_placement() {
        let situation = SpawnSituation::from_request("launch");
        assert_eq!(situation, SpawnSituation::Launch);
        assert_eq!(situation.ship_blueprint_path(), "ships/saturn.ron");
        assert!(situation.is_spaceport());
        assert!(situation.has_deferred_placement());
        assert!(!situation.is_runway());
        assert!(!situation.is_aircraft());
    }

    #[test]
    fn landing_state_sits_above_ground_and_descends_retrograde() {
        let radius_m = 1_000_000.0;
        let terrain_h_m = 1_000.0;
        let body = body_at(DVec3::ZERO, radius_m);
        let (state, attitude) =
            descent_approach_state(&body, radius_m, DVec3::X, terrain_h_m, LANDING_PROFILE);

        // Placed exactly AGL above the local surface, radially over the site.
        let expected_radius = radius_m + terrain_h_m + LANDING_PROFILE.altitude_m;
        assert!((state.position.length() - expected_radius).abs() < 1.0);
        assert!(state.position.normalize().dot(DVec3::X) > 0.999_9);

        // Descending: the radial (up = +X) velocity component is the negative
        // descent rate; the rest is the cross-track lead.
        let up = DVec3::X;
        assert!((state.velocity.dot(up) + LANDING_PROFILE.descent_rate_m_s).abs() < 1e-6);
        let expected_speed = (LANDING_PROFILE.descent_rate_m_s.powi(2)
            + LANDING_PROFILE.cross_track_speed_m_s.powi(2))
        .sqrt();
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
        let (state, _) = descent_approach_state(&body, radius_m, DVec3::X, 0.0, LANDING_PROFILE);

        let r_rel = state.position; // body at origin
        let surface_velocity = body.angular_velocity.cross(r_rel);
        let surface_relative = state.velocity - surface_velocity;
        let up = DVec3::X;
        assert!((surface_relative.dot(up) + LANDING_PROFILE.descent_rate_m_s).abs() < 1e-6);
        let expected_speed = (LANDING_PROFILE.descent_rate_m_s.powi(2)
            + LANDING_PROFILE.cross_track_speed_m_s.powi(2))
        .sqrt();
        assert!((surface_relative.length() - expected_speed).abs() < 1e-6);
    }

    #[test]
    fn final_approach_state_is_low_and_slow() {
        let radius_m = 1_000_000.0;
        let terrain_h_m = 250.0;
        let body = body_at(DVec3::ZERO, radius_m);
        let (state, _) = descent_approach_state(
            &body,
            radius_m,
            DVec3::X,
            terrain_h_m,
            FINAL_APPROACH_PROFILE,
        );

        let expected_radius = radius_m + terrain_h_m + FINAL_APPROACH_PROFILE.altitude_m;
        assert!((state.position.length() - expected_radius).abs() < 1.0);
        assert!(
            (state.velocity.dot(DVec3::X) + FINAL_APPROACH_PROFILE.descent_rate_m_s).abs() < 1e-6
        );
    }

    #[test]
    fn find_site_returns_substellar_when_it_is_land() {
        let land = MockHeight(|dir: Vec3| {
            Some(if dir.dot(Vec3::X) > 0.5 {
                5_000.0
            } else {
                -3_000.0
            })
        });
        let site = find_landing_site(&land, Some(0.0), DVec3::X, 1_000_000.0, LANDING_PROFILE);
        assert!((site - DVec3::X).length() < 1e-6);
    }

    #[test]
    fn find_site_walks_off_ocean_to_nearby_land() {
        // Sub-stellar point is ocean; land is a cap ~30° away in the X–Z plane.
        let land_dir = Vec3::new(
            30.0_f32.to_radians().cos(),
            0.0,
            30.0_f32.to_radians().sin(),
        );
        let height = MockHeight(move |dir: Vec3| {
            Some(if dir.dot(land_dir) > 0.9 {
                4_000.0
            } else {
                -2_000.0
            })
        });
        let site = find_landing_site(&height, Some(0.0), DVec3::X, 1_000_000.0, LANDING_PROFILE);
        let h = height.sample_height_m(site.as_vec3(), 0.0).unwrap();
        assert!(
            h > 0.0,
            "expected the search to settle on land, got h = {h}"
        );
    }

    #[test]
    fn final_approach_prefers_flat_land_over_nearby_rough_land() {
        let flat_dir = Vec3::new(
            30.0_f32.to_radians().cos(),
            0.0,
            30.0_f32.to_radians().sin(),
        );
        let height = MockHeight(move |dir: Vec3| {
            if dir.dot(flat_dir) > 0.99 {
                Some(2_000.0)
            } else if dir.dot(Vec3::X) > 0.99 {
                Some(2_000.0 + dir.y * 100_000.0)
            } else {
                Some(-2_000.0)
            }
        });
        let site = find_landing_site(
            &height,
            Some(0.0),
            DVec3::X,
            1_000_000.0,
            FINAL_APPROACH_PROFILE,
        );
        assert!(
            site.as_vec3().dot(flat_dir) > 0.98,
            "expected final approach to choose the flat patch, got {site:?}",
        );
    }

    #[test]
    fn find_site_skips_polar_land() {
        // Only the poles are above water; the latitude filter must refuse them
        // so the player never wakes up on the ice cap.
        let height = MockHeight(|dir: Vec3| {
            Some(if dir.y.abs() > 0.95 {
                6_000.0
            } else {
                -1_000.0
            })
        });
        let site = find_landing_site(&height, Some(0.0), DVec3::X, 1_000_000.0, LANDING_PROFILE);
        assert!(
            site.y.abs() as f32 <= LANDING_SITE_MAX_ABS_LAT_SIN,
            "site latitude {} exceeded the ice-cap guard",
            site.y
        );
    }

    #[test]
    fn find_site_returns_substellar_for_ocean_free_body() {
        let height = MockHeight(|_dir: Vec3| Some(-9_999.0));
        let site = find_landing_site(&height, None, DVec3::X, 1_000_000.0, FINAL_APPROACH_PROFILE);
        assert!((site - DVec3::X).length() < 1e-6);
    }
}
