//! A fixed, flat runway on the Thalos surface, plus the two spawn scenarios
//! that put an aircraft on it (`just game runway`, `just game runway-approach`)
//! or the Saturn rocket on a launchpad (`just game launch`).
//!
//! This is a deferred, terrain-aware spawn just like the descent scenarios in
//! [`crate::spawn`]: `main.rs` parks the ship in the placeholder orbit behind
//! the loading screen, and [`finish_runway_spawn`] installs the real runway +
//! aircraft state on the first `AppState::Running` frame, once the terrain
//! height source is resident.
//!
//! **The terrain itself is flattened into a wide basin; the runway sits flush on
//! it.** The basin is one level area the whole spaceport shares — the runway near
//! one edge, and a launch complex (two pads, tank farms, flame diverters, a VAB
//! and hangars, authored by [`crate::base_editor::spawn_default_base`]) filling
//! the rest — offset toward the complex side so the flattened ground isn't wasted
//! on the empty side of the strip.
//! The runway centre is a **fixed body-fixed location** (constant lat/lon, see
//! [`fixed_runway_site`]) — not an auto-chosen flattest/dry/sunlit site, which
//! could land on the night side. The scenario also seats the world at a
//! **morning boot epoch** ([`RUNWAY_MORNING_EPOCH_S`]) so the fixed site is lit
//! by a low, rising sun instead of the high noon sun the epoch-0 sub-stellar
//! point gives it. A single fixed elevation
//! `E = mean(natural terrain over the basin)` is then chosen — levelling to the
//! *mean* balances cut against fill, so the wide basin sinks into rising ground
//! and fills hollows by roughly equal amounts instead of becoming an all-fill
//! plateau towering over the surroundings.
//! A [`thalos_terrain::TerrainFlatten`] pad is installed via the body's shared
//! [`crate::rendering::ground_terrain::TerrainFlattenRegistry`] handle: the
//! terrain tile provider reads it as it bakes, so the *rendered* ground — and,
//! through the GPU-atlas height mirror, the collider and CPU height queries —
//! level out to `E` across the pad and smoothstep-blend back to natural terrain
//! over a ramp. The pad is set before the aircraft/camera move to the site, so
//! the tiles that stream in there bake flattened from the start.
//!
//! On top of the levelled ground the runway is just a paved strip + markings +
//! posts, lifted a few centimetres so the paving reads on the grass. The strip
//! is a **true flat plane** (the tangent plane at the site centre), not a
//! sphere-draped strip — paving, markings, the kinematic collider slab, and the
//! parked craft's rest pose all share that one plane, so the gear rests exactly
//! on the painted surface. The collider's top face sits at the paved surface
//! (posed each frame like the terrain collider patch) and stays exactly flat
//! regardless of tile residency timing. A parked aircraft spawns landed,
//! resting on its gear at static-sag equilibrium (no launch clamp); the pilot
//! flies it off with the throttle.

use bevy::camera::primitives::MeshAabb;
use bevy::camera::visibility::RenderLayers;
use bevy::math::{DMat3, DQuat, DVec3, Vec3, Vec3A};
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{HeightSource, ShadowedStandardMaterial, shadowed};
// Runway geometry (frame, meshes, markings, designators, materials, site math)
// lives in `thalos_structures` (Phase 5b); this module keeps the drivers —
// deferred placement, the collider, per-frame f64 anchoring, spaceport
// orchestration.
use thalos_physics_canonical::body_fixed::{
    body_fixed_pose_from_inertial, body_fixed_surface_velocity,
};
use thalos_physics_canonical::canonical::{AuthorityMode, Epoch, TranslationalState};
use thalos_physics_canonical::types::{AttitudeState, BodyState};
use thalos_physics_local::{
    ActiveLocalBubble, HeightSourceRegistry, LocalPrimitiveShape, spawn_structure_collider,
};
use thalos_shipyard::{AttachNodes, EngineActivation};
pub use thalos_structures::runway_geometry::*;
use thalos_world::{BodyId, StateVector};

use crate::SimStage;
use crate::camera::ShipCamera;
use crate::coords::SHIP_LAYER;
use crate::local_physics::PHYSICS_QUERY_TILE_LOD_M;
use crate::rendering::ground_terrain::TerrainFlattenRegistry;
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::rendering::sun_shadow::SHADOW_CASTER_LAYER;
use crate::rendering::terrain_residency::TerrainRebuildRequest;
use crate::rendering::{PlayerShip, RealSpaceBody};
use crate::solar_system_state::{SimulationState, SolarSystemState};
use crate::spawn::{CraftPlacement, SpawnSituation, coast_placement, place_craft};

// ---------------------------------------------------------------------------
// Runway dimensions (aerospace research runway: 5 km × 90 m)
// ---------------------------------------------------------------------------
//
// Sized for spaceplane recovery with comfortable margins. The benchmarks are
// the KSC Shuttle Landing Facility (4,572 m × 91 m) and Edwards AFB's main
// paved runway (~4,580 m), both built for high-speed gliding returns; 5 km of
// length leaves room above what the Shuttle actually needed, so any plane or
// spaceplane can take off and land here without running out of strip.

/// Flat margin levelled around the painted strip (the "shoulder" of the pad).
/// The terrain inside `runway + this` is flattened to `E`.
const RUNWAY_PAD_MARGIN_M: f64 = 50.0;
/// The base is one wide flat **basin** the whole spaceport sits inside: the two
/// separated runways plus the launch/airport complex (pads, tank farms, flame
/// diverters, VAB, hangars, aprons — see [`crate::base_editor::spawn_default_base`])
/// filling the space between and beside them. Everything drapes coplanar on the
/// basin at its single elevation `E`.
///
/// Half-length of the basin along the runway (the strip plus generous apron/
/// clearway at each end).
const BASIN_HALF_ALONG_M: f64 = RUNWAY_HALF_LENGTH_M + 400.0;
/// Half-width of the basin across the runway. Wide enough to clear the angled
/// secondary runway (whose far end reaches ~2.2 km off to the `+across` side)
/// and the launch complex (~1.1 km off to the `-across` side) with a green
/// margin, so both runways sit fully inside cleared, flattened ground.
const BASIN_HALF_ACROSS_M: f64 = 1900.0;
/// The basin **rectangle** is offset toward the airside (`+across` =
/// `center_dir × heading`, the secondary-runway side): the angled secondary
/// fans much further out on that side than the launch complex does on its
/// side, so a centred rectangle would waste half a kilometre of flattening
/// behind the pads. This is a rect offset within the plane tangent at the
/// runway centre (`StructurePlacement::FlattenTo::rect_offset_across_m`), not
/// a moved anchor — see the basin registration in [`build_spaceport`].
const BASIN_RECT_OFFSET_ACROSS_M: f64 = 500.0;
/// Wide blend from the basin back to natural terrain. The basin levels to the
/// *mean* terrain over its footprint (balanced cut/fill — see
/// `finish_runway_spawn`), so the edge height is only ~half the basin's relief;
/// this broad ramp then fades that into the surrounding ground so it reads as a
/// graded basin, not a plateau with a wall around it.
const BASIN_RAMP_M: f64 = 500.0;

/// Light the jet engines once the cruise aircraft is placed. Mirrors
/// [`enable_runway_engines`] but triggers on the cruise scenario (no runway
/// site resource needed — the aircraft is already airborne at spawn).
///
/// Keyed per ship root (not a plain one-shot) and gated on the
/// [`crate::staging::StagingPlan`] existing, so a craft swapped in at runtime
/// (start screen / relaunch into cruise) gets its engines lit *after* the
/// staging build has done its disable-at-spawn pass — never the outgoing
/// craft, never re-disabled.
fn enable_cruise_engines(
    mut lit: Local<Option<Entity>>,
    situation: Res<SpawnSituation>,
    ships: Query<Entity, (With<PlayerShip>, With<crate::staging::StagingPlan>)>,
    mut activations: Query<
        &mut EngineActivation,
        Without<crate::shipyard_editor::core::EditorPart>,
    >,
) {
    if !matches!(*situation, SpawnSituation::Cruise) {
        return;
    }
    let Ok(ship) = ships.single() else {
        return; // craft not built / staging plan not derived yet
    };
    if *lit == Some(ship) {
        return;
    }
    let mut count = 0;
    for mut activation in &mut activations {
        activation.enabled = true;
        count += 1;
    }
    if count > 0 {
        *lit = Some(ship);
        info!("cruise: lit {count} engine(s) for cruise flight");
    }
}

// ---------------------------------------------------------------------------
// Fixed runway site (constant body-fixed location)
// ---------------------------------------------------------------------------
//
// The runway lives at one constant spot on the sphere rather than auto-choosing
// a flattest/dry/sunlit site each spawn (which could fall on the night side).
// The terrain under the footprint is flattened into a level pad regardless (see
// the `TerrainFlatten` install in `finish_runway_spawn`), so the only thing the
// coordinates must satisfy is "dry land, sunlit at the spawn epoch" — the
// author's call, since only the rendered map shows where that is. The spawn
// epoch is the morning boot epoch (`RUNWAY_MORNING_EPOCH_S`), not epoch 0, so
// "sunlit" means sunlit *in the morning* (the site sits ~13° below the rising
// sun there). Override the site at runtime for iteration with
// `THALOS_RUNWAY_SITE="lat_deg,lon_deg[,heading_deg]"`.
//
// The default below was chosen with the `runway_site` probe
// (`cargo run --release -p thalos_bake_dump --example runway_site`), which
// samples the **runtime** terrain — `ProceduralSurface::new(radius, body.id)`,
// the same generator `rendering::ground_terrain` builds — computes the
// sub-stellar (local-noon) point at epoch 0 (lat 0°, lon 180°), and ranks
// dry near-equator land by how well-lit and gently-rolling it is. This site is a
// low coastal plain (~80 m above sea level), ~8° north of the equator at ~178°.
// At epoch 0 it sits almost under the noon sun (~82° up); the scenario instead
// boots at `RUNWAY_MORNING_EPOCH_S`, which rotates the planet so the same fixed
// site is lit by a low rising sun. NOTE: the spot is specific to the procedural
// seed (Thalos = `body.id`); if the terrain generator changes, re-run the probe
// and update these constants, or the runway can land below sea level (which also
// pins the chase camera overhead — the camera floor won't follow it under the
// surface).

/// Boot epoch (s) the runway scenario seats the world at, so the fixed site is
/// lit by a low **morning** sun instead of the noon sun the epoch-0 sub-stellar
/// point gives it.
///
/// At epoch 0 the sub-stellar (local-noon) point is lat 0°, lon 180°, putting
/// the runway site (lat 7.6°, lon 178°) almost directly under the sun (~82° up
/// — high noon). Thalos spins about +Y (prograde, 76,680 s day), so the sun
/// climbs across the second half of the day at this site; advancing the clock
/// to ~59,100 s rotates the planet so the same fixed site sees a ~13° sun that
/// is *rising* — early-to-mid morning, long shadows down the strip. The whole
/// authored runway terrain (continent land bias, plains suppression, horizon
/// massifs in `thalos_terrain::procedural`) is body-fixed, so only the lighting
/// changes; the site stays put.
///
/// Derived with the `morning_probe` example
/// (`cargo run -p thalos_physics_canonical --example morning_probe`), which
/// reuses the real ephemeris to sweep one Thalos day and report the sun
/// elevation/trend at the site. Re-run it and update this constant if the site
/// lat/lon or Thalos's rotation/orbit changes.
const RUNWAY_MORNING_EPOCH_S: f64 = 59_100.0;

/// The canonical boot epoch a spawn situation seats, or `None` if it authors no
/// time of day and simply starts at epoch 0.
///
/// One authority for "what time of day is this scenario supposed to be". It
/// exists because the **capture lane needs the answer without booting the
/// scenario**: a resident capture host serves many requests, `build_spaceport`
/// seats the clock exactly once (at placement), and nothing rewound it
/// afterwards — so an untimed shot taken after a `--time` shot silently
/// inherited the previous shot's sun, exited 0, and wrote a plausible PNG at
/// the wrong lighting (BL-20260731T202657Z). Resolving every request against
/// this value makes a shot's time absolute instead of history-dependent.
///
/// Keep it consistent with what [`build_spaceport`] actually calls
/// `set_sim_time` with; a divergence would reset the clock away from the pose
/// the placement was computed at.
pub(crate) fn canonical_epoch_s(situation: SpawnSituation) -> Option<f64> {
    // `build_spaceport` — the sole `set_sim_time` caller among the spawns — runs
    // for exactly the spaceport situations.
    situation.is_spaceport().then_some(RUNWAY_MORNING_EPOCH_S)
}

/// Sea-level datum (m above the reference radius). The runtime
/// `ProceduralSurface` has no water layer; its continent-mask shoreline sits at
/// the reference radius, so sea level is height 0.
const SEA_LEVEL_M: f32 = 0.0;
/// Freeboard above sea level below which the fixed site is warned as likely
/// underwater. A constant spot isn't validated against the ocean mask — this
/// just flags a bad coordinate in the log instead of silently submerging it.
const SITE_FREEBOARD_M: f32 = 50.0;

/// Footprint grid sampled to find the platform elevation (max/min terrain).
/// Denser across than the old runway-only pad, since the basin is now wide
/// enough that a coarse across-sampling could step over a local rise.
const FOOTPRINT_SAMPLES_LEN: usize = 60;
const FOOTPRINT_SAMPLES_W: usize = 24;

// ---------------------------------------------------------------------------
// Orientation markers (raised edge posts)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Aircraft placement
// ---------------------------------------------------------------------------

/// Small gap left between the craft's lowest point and the platform surface so
/// the gear/belly rests just on top without z-fighting. The real clearance is
/// measured per-craft from its geometry (see [`craft_ground_clearance`]); this
/// is only the sliver above it.
const RUNWAY_GEAR_REST_MARGIN_M: f64 = 0.03;
/// Park this far in from the threshold end so the nose is on the numbers.
const PARK_THRESHOLD_INSET_M: f64 = 150.0;

// Short-final spawn geometry: ~3.4° glideslope flown at ~1.3 × the Meridian's
// clean stall speed (~60 m/s at its ~37 t wet mass), the standard approach
// margin. The speed must track the craft's wing loading: spawning at or below
// stall just falls out of the sky.
const APPROACH_BACK_M: f64 = 2500.0;
const APPROACH_ALT_M: f64 = 150.0;
const APPROACH_SPEED_M_S: f64 = 80.0;
const APPROACH_SINK_M_S: f64 = 4.7;

/// Hide the runway beyond this multiple of the body radius (matches the
/// terrain/impostor LOD swap so it isn't a speck poking through the orbital
/// billboard).
const RUNWAY_VIS_RADIUS_FACTOR: f64 = 4.0;

/// The chosen runway, in the body-fixed frame. Inserted once by
/// [`build_spaceport`]; kept around for UI / future reference, and used as the
/// "spaceport already built?" key by the launch-select flow.
#[derive(Resource, Debug, Clone, Copy)]
pub struct RunwaySite {
    pub body_id: BodyId,
    /// Unit body-fixed direction to the (primary) runway centre.
    pub center_dir: DVec3,
    /// Unit body-fixed tangent along the primary takeoff heading at the centre.
    pub heading_tangent: DVec3,
    /// Flat platform elevation (m above the body reference radius). The whole
    /// runway top is this constant radius — level, not draped.
    pub elevation_m: f64,
    /// The [`BaseSite`](crate::structures::StructureKind::BaseSite) basin the
    /// spaceport sits on. The launch-select flow opens the god-view focused on
    /// this site and hit-tests the runways/launchpads registered under it.
    pub basin_id: crate::structures::StructureId,
}

/// Marker on the runway platform entity (a root-grid big_space child).
/// Children (shoulder, runoff, markings, posts) inherit its visibility and
/// transform. [`update_runway_transform`] positions it in f64 each frame.
#[derive(Component, Debug)]
struct RunwayVisual {
    body_id: BodyId,
    swap_radius_m: f32,
    /// Body-fixed position of the platform centre at elevation `E`.
    center_surface_body: DVec3,
}

/// Re-armable trigger for deferred spaceport placement. Armed at startup
/// when the boot scenario is a runway/launchpad start, and again by the start
/// screen when one is picked there ([`crate::main_menu`]).
/// [`finish_runway_spawn`] consumes it (clears `pending`) once the site is
/// built and the craft placed.
#[derive(Resource, Debug, Default)]
pub struct RunwayPlacement {
    pub pending: bool,
}

pub struct RunwayPlugin;

impl Plugin for RunwayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RunwayPlacement>()
            .add_systems(Startup, arm_boot_runway_placement)
            .add_systems(
                Update,
                (
                    // Runs during `AppState::Loading` (not gated on `Running`): the
                    // site selection + flatten install + aircraft placement happen
                    // behind the loading screen so the camera reaches the surface
                    // and the terrain streams/settles before the reveal. Self-gated
                    // by `RunwayPlacement::pending` + the height-source residency
                    // check; `relaunch_idle` keeps it from measuring/parking the
                    // outgoing craft while a runtime craft swap is in flight.
                    finish_runway_spawn
                        .run_if(crate::relaunch::relaunch_idle)
                        .before(SimStage::Physics),
                    update_runway_transform
                        .in_set(SimStage::Sync)
                        .after(crate::solar_system_state::sync_solar_system_state),
                    sync_runway_visibility
                        .in_set(SimStage::Sync)
                        .after(update_runway_transform),
                    enable_runway_engines,
                    enable_cruise_engines,
                ),
            );
        // The runway slab collider is posed each frame by the executor's
        // generic `sync_structure_collider_pose` (scheduled by
        // `local_physics`), so no runway-local pose system is needed — see
        // `docs/simulation/physics.md` (backend seam).
    }
}

/// Arm the deferred placement for a runway boot scenario. Runtime re-arms
/// (start screen → runway) set [`RunwayPlacement::pending`] directly.
fn arm_boot_runway_placement(
    situation: Res<SpawnSituation>,
    mut placement: ResMut<RunwayPlacement>,
) {
    placement.pending = situation.is_spaceport();
}

/// Deferred finisher: resolve the fixed site, build the flat platform +
/// collider, and place the aircraft. Runs once per arming of
/// [`RunwayPlacement`], retrying
/// each frame until the terrain height source is resident. (Each run builds a
/// fresh runway; today it can only ever run once per process — the boot
/// scenario *or* the one-shot start screen — so there is no stale-runway
/// teardown here yet.)
#[allow(clippy::too_many_arguments)]
fn finish_runway_spawn(
    mut placement: ResMut<RunwayPlacement>,
    situation: Res<SpawnSituation>,
    mut sim: ResMut<SimulationState>,
    mut settle: ResMut<crate::surface_settle::SurfaceSettle>,
    mut tracker: ResMut<crate::loading::LoadingTracker>,
    height_sources: Res<HeightSourceRegistry>,
    // Bundled to stay within Bevy's 16-param system limit (like `gear_geometry`).
    registries: (
        ResMut<TerrainFlattenRegistry>,
        ResMut<crate::structures::StructureRegistry>,
        ResMut<ActiveLocalBubble>,
        ResMut<TerrainRebuildRequest>,
        ResMut<crate::base_editor::PavedFootprints>,
    ),
    root: Res<RealSpaceRoot>,
    ship_root_q: Query<(Entity, &GlobalTransform), With<PlayerShip>>,
    children_q: Query<&Children>,
    mesh_q: Query<(&GlobalTransform, &Mesh3d)>,
    // Bundled into one tuple param to stay within Bevy's 16-param system limit.
    // Used (with the gear parts + suspension stiffness) to rest the craft on its
    // landing gear at the loaded static-sag equilibrium.
    gear_geometry: (
        crate::local_physics::PartColliderQuery,
        crate::local_physics::GearPartQuery,
        Query<&AttachNodes>,
        Res<crate::local_physics::GearTuning>,
    ),
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ShadowedStandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    if !placement.pending || !situation.is_spaceport() {
        return;
    }
    let (mut flatten_registry, mut structure_registry, mut active_bubble, mut rebuild, mut paved) =
        registries;
    let body_id = sim.simulation.dominant_body();
    let Some(height_source) = height_sources.get(body_id) else {
        return; // terrain not resident yet — retry next frame
    };
    let hs = height_source.as_ref();

    // How far to lift the parked craft so it rests on the surface. Computed
    // before building anything, so a craft whose gear/geometry isn't ready yet
    // just retries instead of double-spawning the spaceport.
    let park_clearance_m = match *situation {
        SpawnSituation::Runway => {
            let Ok((ship_entity, ship_gt)) = ship_root_q.single() else {
                return; // ship not spawned yet — retry
            };
            let (parts, gear_q, host_nodes, gear_tuning) = &gear_geometry;
            match measure_runway_clearance(
                ship_entity,
                ship_gt,
                &children_q,
                &mesh_q,
                &meshes,
                parts,
                gear_q,
                host_nodes,
                gear_tuning,
            ) {
                Some(c) => c,
                None => return, // craft gear/geometry not ready yet — retry
            }
        }
        SpawnSituation::Launch => {
            let Ok((ship_entity, ship_gt)) = ship_root_q.single() else {
                return; // Saturn root not spawned yet — retry
            };
            let Some(clearance) = craft_extent_below(
                ship_entity,
                ship_gt,
                &children_q,
                &mesh_q,
                &meshes,
                Vec3::NEG_Y,
            ) else {
                return; // part meshes not ready yet — retry
            };
            clearance
        }
        _ => 0.0,
    };

    // Build the spaceport site — basin flatten, both runways (primary + angled
    // secondary) with their geometry/colliders, and the default base — without
    // placing any craft. Shared with the launch-select flow, which builds the
    // same site lazily and then lets the player pick a launch point on it.
    let SpaceportBuild {
        basin_id: _basin_id,
        site,
        body_state,
        body_radius_m,
    } = build_spaceport(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut images,
        &mut structure_registry,
        &mut paved,
        &mut flatten_registry,
        &mut rebuild,
        &mut sim,
        hs,
        root.entity,
        body_id,
    );

    match *situation {
        SpawnSituation::Runway => {
            place_on_runway(
                &mut sim,
                &body_state,
                &site,
                body_radius_m,
                RUNWAY_HALF_LENGTH_M,
                park_clearance_m,
                &mut commands,
            );
            // Hold the freshly-parked craft on the strip. The brakes latch
            // defaults off (airborne spawns must not start with the spoilers
            // out), so the parked placement is the one spot that engages it.
            commands.insert_resource(crate::local_physics::ParkingBrake { engaged: true });
        }
        SpawnSituation::RunwayApproach => {
            place_approach(&mut sim, &body_state, &site, body_radius_m)
        }
        SpawnSituation::Launch => {
            let Some(pad) = structure_registry
                .sites_on(body_id)
                .iter()
                .find(|site| {
                    matches!(
                        site.kind,
                        crate::structures::StructureKind::Launchpad { .. }
                    )
                })
                .copied()
            else {
                error!("launch: default spaceport contains no launchpad");
                return;
            };
            let elevation_m = pad
                .parent_site
                .and_then(|parent| structure_registry.get(parent))
                .and_then(|parent| match parent.placement {
                    crate::structures::StructurePlacement::FlattenTo { elevation_m, .. } => {
                        Some(elevation_m)
                    }
                    crate::structures::StructurePlacement::Drape => None,
                })
                .unwrap_or(0.0);
            crate::base_editor::place_on_launchpad(
                &mut sim,
                &body_state,
                body_id,
                pad.anchor_dir,
                pad.heading_tangent,
                body_radius_m,
                elevation_m,
                park_clearance_m,
                &mut commands,
                &mut active_bubble,
            );
            info!("launch: Saturn placed on launchpad {:?}", pad.id);
        }
        _ => {}
    }

    // Both runway scenarios rest the craft on its wheels, so force the gear
    // down — `GearState` is a persistent resource, and a respawn after the
    // player retracted gear in flight would otherwise spawn the aircraft
    // belly-down on the strip.
    if situation.is_runway() {
        commands.insert_resource(crate::local_physics::GearState { down: true });
    }

    commands.insert_resource(site);
    placement.pending = false;
    // The placement just teleported the canonical craft. Any live Avian bubble
    // was seeded from the *pre-placement* state (the placeholder orbit), and
    // the authority-edge snap can miss a bubble whose spawn commands haven't
    // flushed yet — the render craft (and the camera + tile streamer behind
    // it) would then coast the stale orbit while the canonical stats sit
    // parked. Tear the bubble down like every other ship teleport does;
    // `spawn_player_avian_body` rebuilds it next frame seeded from the placed
    // state.
    crate::scenario_menu::clear_bubble(&mut commands, &mut active_bubble);
    // The surface state + flatten pad are installed and the aircraft is placed:
    // let the settle gate start timing the tile stream at the (now-known) site
    // and release the loading screen's placement gate.
    settle.mark_placed();
    tracker.complete(crate::loading::step::PLACEMENT);
}

/// Result of [`build_spaceport`]: the flattened base site plus the primary
/// runway, ready for a craft to be placed on (dev runway park) or for the
/// player to pick a launch point on (launch-select flow).
pub(crate) struct SpaceportBuild {
    /// The `BaseSite` basin every runway/launchpad drapes on — the launch-select
    /// god-view focuses on it and hit-tests its child launch points.
    pub basin_id: crate::structures::StructureId,
    /// The primary runway (also the [`RunwaySite`] resource the game inserts).
    pub site: RunwaySite,
    /// Body state at the morning boot epoch this build seats the world at, so a
    /// caller placing a craft uses the same epoch the geometry was posed at.
    pub body_state: BodyState,
    pub body_radius_m: f64,
}

/// Build the spaceport at the fixed site: flatten the basin, register + render +
/// collide both runways (primary + angled secondary), and author the default
/// base (launchpads, buildings, tanks, tarmac) on the basin. Seats the world at
/// the morning boot epoch. **Does not place or measure any craft** — the caller
/// decides what to do next (park a craft, or open the launch-select god-view).
///
/// Shared by the dev runway scenario ([`finish_runway_spawn`]) and the
/// launch-select flow's lazy site build, so both produce the identical
/// spaceport. Not idempotent — the caller must guard against building twice
/// (e.g. on the [`RunwaySite`] resource already existing).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_spaceport(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ShadowedStandardMaterial>,
    images: &mut Assets<Image>,
    structure_registry: &mut crate::structures::StructureRegistry,
    paved: &mut crate::base_editor::PavedFootprints,
    flatten_registry: &mut TerrainFlattenRegistry,
    rebuild: &mut TerrainRebuildRequest,
    sim: &mut SimulationState,
    hs: &dyn HeightSource,
    root_entity: Entity,
    body_id: BodyId,
) -> SpaceportBuild {
    let body_radius_m = sim.system.bodies[body_id].radius_m;

    // Fixed body-fixed site (constant lat/lon + heading), instead of the old
    // flattest/dry/sunlit search that could land on the night side. The pad is
    // flattened under the footprint regardless, so the coordinates only need to
    // be dry sunlit land — the author's call (see `fixed_runway_site`).
    let (center_dir, heading_tangent) = fixed_runway_site();
    let center_h = hs
        .sample_height_m(center_dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(0.0) as f64;
    let across = center_dir.cross(heading_tangent).normalize();

    // The runtime `ProceduralSurface` has no separate water layer; its shoreline
    // is the reference radius (height 0). Flag a site that samples at/below it,
    // so a bad coordinate is obvious in the log rather than a mysteriously
    // underwater runway.
    if center_h <= (SEA_LEVEL_M + SITE_FREEBOARD_M) as f64 {
        warn!(
            "runway: fixed site samples {:.0} m, at/below sea level — likely ocean; \
             adjust RUNWAY_SITE_LAT_DEG/LON_DEG or set THALOS_RUNWAY_SITE",
            center_h
        );
    }

    // The base is one wide flat basin the spaceport sits inside. The basin site
    // is **anchored at the primary runway centre** — the flatten's level plane
    // is tangent there, the same plane every paving mesh / runway slab /
    // collider is built in (`RunwayFrame`, `connections::site_anchor`) — while
    // its *rectangle* is pushed `BASIN_OFFSET_ACROSS_M` toward the secondary-
    // runway side (the V fans much further out there than the launch complex
    // does behind the pads). The offset must be a rect offset within the shared
    // plane, never a moved anchor: a plane tangent at the offset centre tilts
    // ~`offset/R` against the pavement — at 500 m that rose past the
    // connections' 0.12 m lift ~350 m out and buried the core apron's far strip
    // (the "dark serrated fringe" bug).
    let basin_across = center_dir.cross(heading_tangent).normalize();
    // Where the offset rectangle actually is — the elevation sampling below
    // must average the ground the flatten will level, not the anchor's
    // surroundings.
    let basin_center_dir =
        (center_dir * body_radius_m + basin_across * BASIN_RECT_OFFSET_ACROSS_M).normalize();
    let basin_center_h = hs
        .sample_height_m(basin_center_dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(center_h as f32) as f64;

    // Basin elevation: the **mean** natural terrain across the whole basin
    // (sampled from natural terrain, *before* the flatten is installed below).
    // Levelling to the mean balances cut against fill, so the basin sinks into
    // rising ground and fills hollows by roughly equal amounts instead of
    // becoming an all-fill plateau towering over its surroundings (which is what
    // levelling to the max produced). The wide `BASIN_RAMP_M` then fades the
    // remaining ~half-relief edge step smoothly into the natural terrain. Because
    // the flatten forces `E` everywhere inside the basin, a natural rise that was
    // above the mean is simply cut down — nothing pokes through.
    let (max_h, min_h, mean_h) = footprint_stats(
        hs,
        basin_center_dir,
        heading_tangent,
        basin_across,
        BASIN_HALF_ALONG_M,
        BASIN_HALF_ACROSS_M,
        body_radius_m,
        basin_center_h,
    );
    let elevation_m = mean_h;
    // A basin wide enough to span the coast could average out below the reference
    // radius even when the runway centre is dry; flag it rather than silently
    // sinking the flat ground under the water renderer.
    if elevation_m <= SEA_LEVEL_M as f64 {
        warn!(
            "runway: basin mean {:.0} m is at/below sea level (terrain {:.0}..{:.0} m) — \
             the flattened basin may sit in water; move the site or shrink BASIN_HALF_ACROSS_M",
            elevation_m, min_h, max_h
        );
    }

    // Register the **basin** as a `BaseSite` and install its flatten through the
    // shared structures path (`crate::structures`). The terrain provider reads
    // the flatten handle as it bakes tiles, so the rendered ground — and, via the
    // GPU-atlas height mirror, the collider and CPU height queries — level out to
    // `elevation_m` across the basin, smoothstep-blending back to natural terrain
    // over the ramp. Done before placing the aircraft / moving the camera so the
    // tiles that stream in bake flattened from the start.
    let basin_id = structure_registry.register(
        body_id,
        // Anchored at the runway centre — the flatten plane must be tangent
        // where the pavement is built (see the basin comment above); the
        // rectangle alone is pushed toward the secondary.
        center_dir,
        heading_tangent,
        crate::structures::StructurePlacement::FlattenTo {
            elevation_m,
            half_along_m: BASIN_HALF_ALONG_M,
            half_across_m: BASIN_HALF_ACROSS_M,
            ramp_m: BASIN_RAMP_M,
            rect_offset_along_m: 0.0,
            rect_offset_across_m: BASIN_RECT_OFFSET_ACROSS_M,
        },
        crate::structures::StructureKind::BaseSite,
        None,
    );
    if let Some(basin) = structure_registry.get(basin_id).copied() {
        crate::structures::apply_structure_flatten(&basin, body_radius_m, flatten_registry);
    }
    // The runways are the first structures on the basin — they drape on the
    // level ground (the basin provides the flatten), at the basin's elevation
    // `E`. Register the primary strip (centred) plus the **angled** secondary:
    // its near threshold sits at the V corner (`SEC_NEAR_*`, near the primary's
    // `−along` threshold on the empty `+across` side) and the strip runs out at
    // `SECONDARY_HEADING_OFFSET_DEG`, fanning away so the two never intersect.
    // Both are plain parametric `StructureKind::Runway` registry entries and
    // render through the same generalized geometry path — the editor can add
    // more the same way.
    let sec_heading = {
        let a = SECONDARY_HEADING_OFFSET_DEG.to_radians();
        // Rotate toward `+across` (the side the secondary sits on) so the strips
        // fan apart with along, never converging.
        (heading_tangent * a.cos() + across * a.sin()).normalize()
    };
    // Centre = the near threshold plus half a length down the angled heading.
    let sec_center_offset = heading_tangent * SEC_NEAR_ALONG_M
        + across * SEC_NEAR_ACROSS_M
        + sec_heading * (SECONDARY_LENGTH_M * 0.5);
    let sec_center_dir = (center_dir * body_radius_m + sec_center_offset).normalize();
    structure_registry.register(
        body_id,
        center_dir,
        heading_tangent,
        crate::structures::StructurePlacement::Drape,
        crate::structures::StructureKind::Runway {
            half_length_m: RUNWAY_HALF_LENGTH_M as f32,
            half_width_m: RUNWAY_HALF_WIDTH_M as f32,
        },
        Some(basin_id),
    );
    structure_registry.register(
        body_id,
        sec_center_dir,
        sec_heading,
        crate::structures::StructurePlacement::Drape,
        crate::structures::StructureKind::Runway {
            half_length_m: (SECONDARY_LENGTH_M * 0.5) as f32,
            half_width_m: (SECONDARY_WIDTH_M * 0.5) as f32,
        },
        Some(basin_id),
    );

    // The primary runway backs the parked-craft spawn + gear collider skip.
    let site = RunwaySite {
        body_id,
        center_dir,
        heading_tangent,
        elevation_m,
        basin_id,
    };

    let lat_deg = center_dir.y.clamp(-1.0, 1.0).asin().to_degrees();
    let lon_deg = center_dir.z.atan2(center_dir.x).to_degrees();
    info!(
        "runway: spaceport on {} at lat {:.1}°, lon {:.1}° — basin levelled to mean {:.0} m (terrain {:.0}..{:.0} m), {:.0} m × {:.0} m",
        sim.system.bodies[body_id].name,
        lat_deg,
        lon_deg,
        elevation_m,
        min_h,
        max_h,
        RUNWAY_LENGTH_M,
        RUNWAY_WIDTH_M,
    );

    // Seat the world at the morning boot epoch so the fixed site is lit by a
    // low, climbing sun rather than the epoch-0 noon sun (see
    // `RUNWAY_MORNING_EPOCH_S`). This must happen before the body state is read
    // below — and before the per-frame lighting/terrain transforms run off
    // `sim_time` — so the placement pose, the rendered ground, and the sun all
    // agree on the same time of day. Idempotent across the placement's retries.
    sim.simulation.set_sim_time(RUNWAY_MORNING_EPOCH_S);

    // Read the body state once (immutable) before any sim mutation below.
    let epoch = Epoch(sim.simulation.sim_time());
    let body_state = sim.ephemeris.state(body_id, epoch);

    // Render + collide every runway on the **shared basin plane** (normal
    // `center_dir`, at `E`): the primary (centred, `center_offset = 0`) and the
    // secondary (offset `across·SEP` within the same plane, `sec_heading`). Both
    // are flat strips flush with the one flattened basin; each gets its own
    // coplanar slab collider. Using the basin plane for the offset strip (rather
    // than its own tangent plane) is what keeps it from sinking into the ground.
    let swap_radius_m = (body_radius_m * RUNWAY_VIS_RADIUS_FACTOR) as f32;
    // `pair_side` is 0 for both: the 30° heading divergence gives each strip its
    // own true-heading designator numbers, so no L/R suffix pair is needed.
    for (hdg, center_offset, pair_side, half_len, half_wid) in [
        (
            heading_tangent,
            DVec3::ZERO,
            0i8,
            RUNWAY_HALF_LENGTH_M,
            RUNWAY_HALF_WIDTH_M,
        ),
        (
            sec_heading,
            sec_center_offset,
            0i8,
            SECONDARY_LENGTH_M * 0.5,
            SECONDARY_WIDTH_M * 0.5,
        ),
    ] {
        let frame = RunwayFrame {
            body_id,
            center_dir,
            heading: hdg,
            across: center_dir.cross(hdg).normalize(),
            body_radius_m,
            elevation_m,
            center_offset,
            pair_side,
            half_length_m: half_len,
            half_width_m: half_wid,
        };
        spawn_runway_geometry(
            commands,
            meshes,
            materials,
            images,
            &frame,
            swap_radius_m,
            root_entity,
            &body_state,
        );
        spawn_runway_collider(commands, &frame, &body_state);
    }

    // Author the rest of the default base on the basin: launchpads + support
    // buildings (coplanar with the runway at `E`) and their connecting tarmac.
    crate::base_editor::spawn_default_base(
        commands,
        meshes,
        materials,
        structure_registry,
        paved,
        root_entity,
        body_id,
        basin_id,
        center_dir,
        heading_tangent,
        sec_heading,
        body_radius_m + elevation_m,
    );

    // Re-bake any already-resident terrain so the basin flatten installed above
    // reaches tiles that streamed in *un-flattened*. The flatten handle is read
    // per tile-pixel at bake time, so tiles baked after this point come out
    // level — but the loading-pass callers only run once the height source is
    // resident, by which point the coarse low-LOD ancestor tiles covering the
    // whole planet (streamed at the placeholder-craft view before the flatten
    // existed) are already resident and stay natural. Note the *rendered*
    // ground no longer depends on this: the terrain vertex stage re-applies
    // the flatten analytically per vertex (`flattened_height` in
    // `body_terrain.wgsl`), so stale/coarse tiles still draw flat inside the
    // basin at every LOD. The rebuild remains load-bearing for everything the
    // shader can't fix — the GPU-atlas height mirror (collider / CPU height
    // queries), the baked albedo/material layers, and scatter placement — and
    // for the ramp band outside the rect, which only the bake levels. Mirrors
    // `base_editor::pick`, which rebuilds for the same reason after a runtime
    // flatten. A no-op if nothing is resident yet (then the first bake is
    // already flattened).
    rebuild.request(body_id);

    SpaceportBuild {
        basin_id,
        site,
        body_state,
        body_radius_m,
    }
}

/// Measure how far to lift a parked craft so its gear (or belly) rests on the
/// flat runway pad. Returns the gear-contact depth minus the static suspension
/// sag (so the gear spawns already loaded at equilibrium — no settle / tip /
/// jump when live physics takes over); a gearless craft falls back to its lowest
/// visual-mesh point plus a sliver. `None` while the craft's gear parts / meshes
/// aren't resident yet, so the caller retries. Shared by the dev runway park
/// ([`finish_runway_spawn`]) and the launch-select runway placement.
#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_runway_clearance(
    ship_entity: Entity,
    ship_gt: &GlobalTransform,
    children_q: &Query<&Children>,
    mesh_q: &Query<(&GlobalTransform, &Mesh3d)>,
    meshes: &Assets<Mesh>,
    parts: &crate::local_physics::PartColliderQuery,
    gear_q: &crate::local_physics::GearPartQuery,
    host_nodes: &Query<&AttachNodes>,
    gear_tuning: &crate::local_physics::GearTuning,
) -> Option<f64> {
    // Rest the craft on its landing gear. Measure the wheel-contact depth from
    // the gear parts — *not* visual meshes, which aren't spawned yet at
    // loading-time placement (a visual measurement saw only the fuselage and
    // buried the gear). Then drop the craft by the static suspension sag so the
    // gear spawns already *loaded* (grounded → WoW aero gate on); spawning at
    // zero compression left the gear unsupported for a frame and the craft tipped
    // before the spring engaged. A craft with no gear falls back to the
    // visual-mesh extent and rests on its belly.
    match crate::local_physics::gear_contact_geometry(parts, gear_q, host_nodes) {
        Some((depth_m, mean_strut_length_m)) => {
            // Stiffness is now derived per wheel to achieve one common loaded
            // stroke fraction regardless of craft mass or axle load. Spawn at
            // that same equilibrium instead of reproducing the retired fixed-k
            // approximation here.
            let sag = crate::local_physics::nominal_static_sag_m(gear_tuning, mean_strut_length_m);
            Some((depth_m - sag).max(0.0))
        }
        None => craft_ground_clearance(ship_entity, ship_gt, children_q, mesh_q, meshes)
            .map(|c| c + RUNWAY_GEAR_REST_MARGIN_M),
    }
}

/// Light the jet engines once the runway aircraft is placed.
///
/// The staging system ([`crate::staging::build_staging_plan`]) disables every
/// engine at spawn (KSP rocket convention: ignite a stage with the stage key).
/// That's wrong for an aircraft on a runway — a real jet is already running and
/// the pilot just advances the throttle. So for the runway scenarios we re-enable
/// the engines once (after the staging plan has been built and the aircraft is
/// placed), giving the documented "throttle-only flight" the staging comment
/// promises. Keyed per ship root + gated on [`crate::staging::StagingPlan`]
/// for the same runtime-craft-swap reasons as [`enable_cruise_engines`];
/// `build_staging_plan` runs only once per ship, so it won't re-disable.
fn enable_runway_engines(
    mut lit: Local<Option<Entity>>,
    situation: Res<SpawnSituation>,
    site: Option<Res<RunwaySite>>,
    ships: Query<Entity, (With<PlayerShip>, With<crate::staging::StagingPlan>)>,
    mut activations: Query<
        &mut EngineActivation,
        Without<crate::shipyard_editor::core::EditorPart>,
    >,
) {
    if !situation.is_runway() || site.is_none() {
        return;
    }
    let Ok(ship) = ships.single() else {
        return; // craft not built / staging plan not derived yet
    };
    if *lit == Some(ship) {
        return;
    }
    let mut count = 0;
    for mut activation in &mut activations {
        activation.enabled = true;
        count += 1;
    }
    if count > 0 {
        *lit = Some(ship);
        info!("runway: lit {count} engine(s) for throttle-only flight");
    }
}

// ---------------------------------------------------------------------------
// Fixed site & heading
// ---------------------------------------------------------------------------

/// Max / min / mean natural terrain height over the basin footprint, sampled on
/// a regular grid at fine LOD. The mean is the level the basin flattens to
/// (balanced cut/fill); max/min are kept for the log / sanity checks.
#[allow(clippy::too_many_arguments)]
fn footprint_stats(
    hs: &dyn HeightSource,
    center_dir: DVec3,
    heading: DVec3,
    across: DVec3,
    half_along_m: f64,
    half_across_m: f64,
    body_radius_m: f64,
    center_h: f64,
) -> (f64, f64, f64) {
    let center_point = center_dir * (body_radius_m + center_h);
    let mut max_h = center_h;
    let mut min_h = center_h;
    let mut sum_h = 0.0;
    let mut count = 0u32;
    for i in 0..=FOOTPRINT_SAMPLES_LEN {
        let along = -half_along_m + 2.0 * half_along_m * (i as f64 / FOOTPRINT_SAMPLES_LEN as f64);
        for j in 0..=FOOTPRINT_SAMPLES_W {
            let across_m =
                -half_across_m + 2.0 * half_across_m * (j as f64 / FOOTPRINT_SAMPLES_W as f64);
            let dir = (center_point + heading * along + across * across_m).normalize();
            if let Some(h) = hs.sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M) {
                let h = h as f64;
                max_h = max_h.max(h);
                min_h = min_h.min(h);
                sum_h += h;
                count += 1;
            }
        }
    }
    let mean_h = if count > 0 {
        sum_h / count as f64
    } else {
        center_h
    };
    (max_h, min_h, mean_h)
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn spawn_runway_geometry(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ShadowedStandardMaterial>,
    images: &mut Assets<Image>,
    frame: &RunwayFrame,
    swap_radius_m: f32,
    parent: Entity,
    body_state: &BodyState,
) {
    // The terrain itself is flattened to `E` across the pad (see the
    // `TerrainFlatten` installed in `finish_runway_spawn`), so the runway is just
    // a paved strip sitting on the levelled ground. The asphalt is lifted a few
    // cm so it reads as paving and never z-fights the flattened tiles; a thin
    // edge skirt (no graded runoff) drops that lift back into the terrain so the
    // strip meets the ground with a curb instead of a floating lip.
    let top = meshes.add(build_top_mesh(frame));
    let skirt = meshes.add(build_skirt_mesh(frame));
    let asphalt = flat_runway_material(materials, Color::srgb(0.055, 0.055, 0.062), 0.95);

    // --- Markings (white, one mesh) ---
    let markings = meshes.add(build_markings_mesh(frame));
    let paint = flat_runway_material(materials, Color::srgb(0.85, 0.85, 0.85), 0.8);

    // Anchor the pad centre via a root-grid big_space cell, positioned in f64
    // (heliocentric) so the multi-Mm surface offset never lands in an f32
    // translation. `update_runway_transform` re-derives this every frame; this
    // is just the first-frame value. The strip rides the f32 `Transform.rotation`
    // (surface orientation) — fine, because only the small child vertex offsets
    // are rotated by it, not the planet-radius position.
    let orientation = body_state.orientation.normalize();
    let center_world = body_state.position + orientation * frame.center_surface();
    let (cell, local) = real_space_grid().translation_to_grid(center_world);
    let runway_entity = commands
        .spawn((
            Mesh3d(top),
            MeshMaterial3d(asphalt.clone()),
            Transform {
                translation: local,
                rotation: orientation.as_quat(),
                scale: Vec3::ONE,
            },
            cell,
            Visibility::Inherited,
            // RECEIVE-ONLY, like the terrain it sits flush on and the tarmac.
            // A km-scale flat surface that both casts into and receives from
            // the cascade maps self-shadow-acnes at grazing sun: the depth the
            // plane spans across ONE shadow texel (texel × cot elevation)
            // exceeds the sampler's hard-capped bias/offset, so the strip
            // shadowed itself in texel-grid blocks and stripes. Its own cast
            // shadow is a 12 cm curb lip — sub-texel in every cascade — so
            // casting bought nothing. The bias model in `thalos::shadow`
            // assumes dominant flat receivers stay out of the maps; keep it so.
            RenderLayers::layer(SHIP_LAYER),
            ChildOf(parent),
            RunwayVisual {
                body_id: frame.body_id,
                swap_radius_m,
                center_surface_body: frame.center_surface(),
            },
            Name::new("Thalos Runway"),
        ))
        .id();

    commands.spawn((
        Mesh3d(markings),
        MeshMaterial3d(paint),
        Transform::IDENTITY,
        Visibility::Inherited,
        // Receive-only: the paint is coplanar with the casting top mesh, so
        // adding it as a second caster would just double-write the same depth.
        RenderLayers::layer(SHIP_LAYER),
        ChildOf(runway_entity),
        Name::new("Runway Markings"),
    ));

    commands.spawn((
        Mesh3d(skirt),
        MeshMaterial3d(asphalt.clone()),
        Transform::IDENTITY,
        Visibility::Inherited,
        // Receive-only for the same reason as the top mesh: the skirt's top
        // edge is coplanar with the paving, so as a caster it re-introduced
        // the same self-shadow acne along the strip edges.
        RenderLayers::layer(SHIP_LAYER),
        ChildOf(runway_entity),
        Name::new("Runway Edge Skirt"),
    ));

    spawn_runway_posts(commands, meshes, materials, frame, runway_entity);
    spawn_runway_numbers(commands, meshes, materials, images, frame, runway_entity);
}

/// Paint the two runway designator numbers (one at each threshold), from the
/// real ICAO font. Each is an alpha texture rasterized from the font and applied
/// to a small quad lying on the runway plane, oriented so it reads upright on
/// approach (the far end rotated 180°). Spawned as children of the runway
/// entity, so they inherit its body-fixed transform.
fn spawn_runway_numbers(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ShadowedStandardMaterial>,
    images: &mut Assets<Image>,
    frame: &RunwayFrame,
    parent: Entity,
) {
    let heading_deg = runway_heading_deg(frame);
    // L/R suffix for a parallel-runway pair. At the near (`rot180 == false`)
    // threshold the strip on the pilot's left (the `+across` side, `pair_side >
    // 0`) is "L"; the far threshold is approached from the opposite side, so it
    // flips. A lone runway (`pair_side == 0`) gets no suffix.
    let suffix = |rot180: bool| -> &'static str {
        match frame.pair_side {
            0 => "",
            s => {
                let left = if rot180 { s < 0 } else { s > 0 };
                if left { "L" } else { "R" }
            }
        }
    };
    // (designator text, along-centre of the number block, rotate-180-for-far-end)
    let ends = [
        (
            format!("{:02}{}", runway_designator(heading_deg), suffix(false)),
            -frame.half_length_m + NUM_THRESHOLD_MARGIN_M + NUM_DIGIT_H_M * 0.5,
            false,
        ),
        (
            format!(
                "{:02}{}",
                runway_designator(heading_deg + 180.0),
                suffix(true)
            ),
            frame.half_length_m - NUM_THRESHOLD_MARGIN_M - NUM_DIGIT_H_M * 0.5,
            true,
        ),
    ];
    for (text, along_center, rot180) in ends {
        let Some((w, h, pixels)) = rasterize_designator(&text) else {
            continue;
        };
        let image = images.add(image_from_alpha_rgba8(w, h, pixels));
        // White paint, masked by the glyph coverage (alpha), receiving the sun
        // shadow like the rest of the strip. Matches the marking paint's tone
        // (`flat_runway_material`) so the digits read as bright as the
        // threshold bar.
        let material = materials.add(shadowed(StandardMaterial {
            base_color: Color::srgb(0.92, 0.92, 0.92),
            base_color_texture: Some(image),
            perceptual_roughness: 0.8,
            metallic: 0.0,
            alpha_mode: AlphaMode::Mask(0.5),
            double_sided: true,
            cull_mode: None,
            ..default()
        }));
        // Quad sized to the glyph aspect: fixed along-height, width follows the
        // rasterized aspect ratio so the digits keep the font's proportions.
        let half_along = NUM_DIGIT_H_M * 0.5;
        let half_across = half_along * (w as f64 / h as f64);
        let mesh = meshes.add(build_number_quad(
            frame,
            along_center,
            half_along,
            half_across,
            rot180,
        ));
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::IDENTITY,
            Visibility::Inherited,
            RenderLayers::layer(SHIP_LAYER),
            ChildOf(parent),
            Name::new("Runway Number"),
        ));
    }
}

/// Raised edge posts at regular intervals down both sides — the 3D references
/// to orient around. Takeoff-threshold posts green, far-end posts red, the rest
/// white (aviation convention).
fn spawn_runway_posts(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ShadowedStandardMaterial>,
    frame: &RunwayFrame,
    parent: Entity,
) {
    let post_mesh = meshes.add(Cuboid::new(POST_SIZE_M, POST_HEIGHT_M, POST_SIZE_M));
    let thresh_mesh = meshes.add(Cuboid::new(
        POST_SIZE_M,
        POST_THRESHOLD_HEIGHT_M,
        POST_SIZE_M,
    ));
    let white = materials.add(post_material(Color::srgb(0.9, 0.9, 0.9)));
    let green = materials.add(post_material(Color::srgb(0.1, 0.8, 0.2)));
    let red = materials.add(post_material(Color::srgb(0.9, 0.15, 0.1)));

    let edge = frame.half_width_m + POST_EDGE_OFFSET_M;
    let stations = (2.0 * frame.half_length_m / POST_SPACING_M).floor() as i32;
    for i in 0..=stations {
        let along = -frame.half_length_m + POST_SPACING_M * i as f64;
        if along > frame.half_length_m + 1.0 {
            break;
        }
        let (mesh, mat, height) = if i == 0 {
            (thresh_mesh.clone(), green.clone(), POST_THRESHOLD_HEIGHT_M)
        } else if i == stations {
            (thresh_mesh.clone(), red.clone(), POST_THRESHOLD_HEIGHT_M)
        } else {
            (post_mesh.clone(), white.clone(), POST_HEIGHT_M)
        };
        for side in [-1.0, 1.0] {
            let (base, up) = frame.level(along, side * edge, 0.0);
            let base_f = Vec3::new(base.x as f32, base.y as f32, base.z as f32);
            let up_f = Vec3::new(up.x as f32, up.y as f32, up.z as f32);
            let translation = base_f + up_f * (height * 0.5);
            let rotation = Quat::from_rotation_arc(Vec3::Y, up_f);
            commands.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(mat.clone()),
                Transform {
                    translation,
                    rotation,
                    scale: Vec3::ONE,
                },
                Visibility::Inherited,
                // Casts into the sun-shadow rig: edge posts throw the small
                // grounding shadows that sell the strip as solid (F6).
                RenderLayers::from_layers(&[SHIP_LAYER, SHADOW_CASTER_LAYER]),
                ChildOf(parent),
                Name::new("Runway Post"),
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Collider — a flat kinematic trimesh at elevation E
// ---------------------------------------------------------------------------

/// Half-thickness of the solid runway slab (m). Generous so a fast or
/// hard-landing craft can never tunnel through the bottom face.
const RUNWAY_SLAB_HALF_THICKNESS_M: f64 = 50.0;

/// Spawn the **solid** runway collider — a cuboid slab whose top face is the
/// flat pad at elevation `E`. A solid (two-sided) shape resolves resting
/// contact gently and pushes a slightly-penetrating craft straight back out,
/// unlike the one-sided trimesh it replaces (whose one-step penetration
/// recovery launched the craft off its gear). Posed each frame from the active
/// surface-local frame by [`sync_runway_collider_pose`] (the spawn-time pose is
/// a placeholder — the bubble may not exist yet during the loading-screen
/// install). See `docs/simulation/surface_local.md` §3.
fn spawn_runway_collider(commands: &mut Commands, frame: &RunwayFrame, _body_state: &BodyState) {
    let half_along = frame.half_length_m + RUNWAY_PAD_MARGIN_M;
    let half_across = frame.half_width_m + RUNWAY_PAD_MARGIN_M;
    // Runway-local tangent frame: X = across, Y = up (center_dir), Z = heading.
    let basis_body_quat = DQuat::from_mat3(&DMat3::from_cols(
        frame.across,
        frame.center_dir,
        frame.heading,
    ));
    // Cuboid centre: drop half the slab thickness below the *paved* surface
    // (`E + asphalt lift`) so the top face coincides with the visible asphalt —
    // the gear raycast then rests the wheels exactly on the painted strip, not
    // on the bare pad a paving-thickness below it.
    let center_body_m = frame.center_surface()
        + frame.center_dir * (RUNWAY_ASPHALT_LIFT_M - RUNWAY_SLAB_HALF_THICKNESS_M);
    // Full side lengths; local axes (X = across, Y = up, Z = along) match
    // `basis_body_quat`. The executor owns the Avian body and its per-frame SLF
    // pose (`sync_structure_collider_pose`) — see `docs/simulation/physics.md` (backend
    // seam). This replaced the bespoke `RunwayCollider` + `sync_runway_collider_pose`.
    spawn_structure_collider(
        commands,
        frame.body_id,
        LocalPrimitiveShape::Cuboid {
            x: 2.0 * half_across,
            y: 2.0 * RUNWAY_SLAB_HALF_THICKNESS_M,
            z: 2.0 * half_along,
        },
        center_body_m,
        basis_body_quat,
        "Runway collider slab",
    );
}

// ---------------------------------------------------------------------------
// Aircraft placement (referenced to the fixed elevation E, never terrain)
// ---------------------------------------------------------------------------

/// Attitude with the nose along `heading_body` and the dorsal along `up_body`
/// — level on the ground, lined up with the runway. Reused by the base editor's
/// launchpad placement.
pub(crate) fn level_heading_attitude(
    body_state: &BodyState,
    up_body: DVec3,
    heading_body: DVec3,
) -> AttitudeState {
    let dorsal = up_body.normalize();
    let nose = (heading_body - dorsal * heading_body.dot(dorsal))
        .try_normalize()
        .unwrap_or_else(|| {
            let seed = if dorsal.x.abs() < 0.9 {
                DVec3::X
            } else {
                DVec3::Z
            };
            (seed - dorsal * seed.dot(dorsal)).normalize()
        });
    let right = nose.cross(dorsal).normalize();
    let craft_to_body = DMat3::from_cols(right, nose, dorsal);
    AttitudeState {
        orientation: (body_state.orientation * DQuat::from_mat3(&craft_to_body)).normalize(),
        angular_velocity: DVec3::ZERO,
    }
}

/// Measure how far the craft's lowest visual point sits below its origin,
/// along the craft's ventral (−Z) axis — so the parked craft can be lifted by
/// exactly this much and rest on the surface for *any* craft (gear, belly,
/// pods, whatever is lowest), with no per-craft constant.
///
/// Walks every mesh under the player-ship root, computes each part's local AABB
/// straight from its **mesh vertices** (`Mesh::compute_aabb` — the part visuals
/// carry `NoFrustumCulling`, so they have no Bevy-computed `Aabb` component),
/// transforms its corners into the ship-root (craft body) frame, and takes the
/// most-negative Z. The craft frame is `+Y` nose / `+Z` dorsal (up), so the
/// lowest point is the minimum Z. Returns `None` if no descendant mesh is ready
/// yet so the caller can retry. Uses local-to-root affines, so the result is
/// independent of the craft's current (placeholder-orbit) pose. Reused by the
/// base editor's launchpad placement to rest any craft on the pad.
pub(crate) fn craft_ground_clearance(
    root_entity: Entity,
    root_gt: &GlobalTransform,
    children_q: &Query<&Children>,
    mesh_q: &Query<(&GlobalTransform, &Mesh3d)>,
    meshes: &Assets<Mesh>,
) -> Option<f64> {
    // The craft frame is `+Z` dorsal (up), so the lowest visual point is the
    // greatest extent along `-Z`.
    craft_extent_below(
        root_entity,
        root_gt,
        children_q,
        mesh_q,
        meshes,
        Vec3::NEG_Z,
    )
}

/// How far the craft's lowest visual point sits below its origin **along the
/// craft-local `down` direction** — generalises [`craft_ground_clearance`] so a
/// vertically-spawned craft (e.g. on a launchpad, nose `+Y` up) can rest on its
/// engine end (`down = -Y`) instead of its belly (`down = -Z`). Returns `None`
/// if no descendant mesh is ready.
pub(crate) fn craft_extent_below(
    root_entity: Entity,
    root_gt: &GlobalTransform,
    children_q: &Query<&Children>,
    mesh_q: &Query<(&GlobalTransform, &Mesh3d)>,
    meshes: &Assets<Mesh>,
    down: Vec3,
) -> Option<f64> {
    let root_inv = root_gt.affine().inverse();
    let down = Vec3A::from(down);
    let mut max_ext = f32::NEG_INFINITY;
    let mut found = false;
    let mut stack: Vec<Entity> = Vec::new();
    if let Ok(c) = children_q.get(root_entity) {
        stack.extend(c.iter());
    }
    while let Some(e) = stack.pop() {
        if let Ok((gt, mesh3d)) = mesh_q.get(e)
            && let Some(aabb) = meshes.get(&mesh3d.0).and_then(Mesh::compute_aabb)
        {
            let local = root_inv * gt.affine();
            let c = aabb.center;
            let h = aabb.half_extents;
            for sx in [-1.0f32, 1.0] {
                for sy in [-1.0f32, 1.0] {
                    for sz in [-1.0f32, 1.0] {
                        let corner = Vec3A::new(c.x + sx * h.x, c.y + sy * h.y, c.z + sz * h.z);
                        max_ext = max_ext.max(local.transform_point3a(corner).dot(down));
                        found = true;
                    }
                }
            }
        }
        if let Ok(c) = children_q.get(e) {
            stack.extend(c.iter());
        }
    }
    found.then(|| (max_ext as f64).max(0.0))
}

/// Park the aircraft **landed** on the runway, resting on its gear, KSP-style —
/// no debug launch clamp.
///
/// The craft spawns already at its landing-gear static-sag equilibrium against
/// the flat runway slab: the origin is lifted by `clearance_m` (the measured
/// gear-contact depth minus the static suspension sag, computed by the caller),
/// so the wheels rest on the paved surface with the suspension pre-loaded — no
/// settle, no tip, and (crucially) **no jump** when live physics takes over,
/// because the spawn pose already *is* the equilibrium the gear would integrate
/// to. Because the runway is now a true flat plane shared by the paving, the
/// collider, and this pose (see [`RunwayFrame::level`]), the wheels sit on the
/// painted strip in every regime instead of being buried by the paving-vs-pad
/// curvature gap the old spherical reference produced.
///
/// Authority is analytic `BodyFixed` — the proper "landed" state (a frozen
/// surface-local pose, per `docs/simulation/surface_local.md` §4), held by
/// [`crate::local_physics::snap_avian_from_canonical`] without integration so it
/// can't drift while the terrain streams in behind the loading screen. The
/// pilot leaves it the moment they advance the throttle:
/// the authority executor's landed throttle release
/// ([`crate::regime::apply_regime_authority`]) hands translation
/// back to the live regimes (`OnRails` → bubble), where thrust + gear take over
/// from the equilibrium pose. Stationary-while-warping is the same `BodyFixed`
/// state the generic stable-landing collapse uses, so time-warp on the runway
/// is stable for free.
///
/// **Warp-neutral**: leaves the time-warp level alone (the dev runway spawn
/// wants the paused-on-spawn default that `spawn::apply_initial_warp` sets on
/// `Loading → Running`). The launch-select caller sets warp to 1× itself, since
/// it places the craft *after* that edge has already fired.
pub(crate) fn place_on_runway(
    sim: &mut SimulationState,
    body_state: &BodyState,
    site: &RunwaySite,
    body_radius_m: f64,
    half_length_m: f64,
    clearance_m: f64,
    commands: &mut Commands,
) {
    let surface_radius = body_radius_m + site.elevation_m;
    let center_surface = site.center_dir * surface_radius;

    // Park inset from the threshold end, on the flat runway plane. Up is the
    // plane normal (`center_dir`), not the local radial, so the craft sits
    // parallel to the flat strip and every wheel reads the same compression.
    // `half_length_m` is the *chosen* strip's half-length (the launch-select
    // flow can park on the shorter secondary runway), so the inset stays on the
    // paved strip + its collider regardless of which runway was picked.
    let along = -(half_length_m - PARK_THRESHOLD_INSET_M);
    let up_body = site.center_dir;
    // The paved (drive) surface at the parked station — the same flat plane the
    // collider's top face and the asphalt mesh use.
    let drive_point =
        center_surface + site.heading_tangent * along + up_body * RUNWAY_ASPHALT_LIFT_M;
    // Rest the gear on it: lift the origin off the paving by the gear-contact
    // depth minus static sag (already folded into `clearance_m`).
    let position_body = drive_point + up_body * clearance_m;
    let nose_body =
        (site.heading_tangent - up_body * site.heading_tangent.dot(up_body)).normalize();

    let position = body_state.position + body_state.orientation * position_body;
    let velocity = body_fixed_surface_velocity(body_state, position_body);
    let state = StateVector { position, velocity };
    let attitude = level_heading_attitude(body_state, up_body, nose_body);

    // Landed: a frozen body-fixed pose at the gear equilibrium. Released to the
    // live bubble by the authority executor's throttle release on throttle-up.
    let pose = body_fixed_pose_from_inertial(body_state, TranslationalState::from(state), attitude);
    place_craft(
        sim,
        CraftPlacement {
            state,
            attitude,
            authority: AuthorityMode::BodyFixed {
                body: site.body_id,
                pose,
            },
        },
        None,
    );
    // Warp is left at the spawn default (paused); `spawn::apply_initial_warp`
    // sets the final level once on `Loading → Running` per `AutoRun`.
    sim.simulation.set_throttle(0.0);
    sim.simulation.set_target_body(Some(site.body_id));
    // A craft already parked at the threshold is ready to roll: command the
    // takeoff detent and start the actuator there, rather than animating from
    // clean after the loading screen clears. Keeping this in the shared runway
    // placement core covers both the dev scenario and launch-select runways.
    commands.insert_resource(crate::flight_config::FlightConfig::runway_takeoff());
}

/// Put the aircraft on short final, lined up with the centreline and sinking,
/// coasting on rails until the local-physics bubble takes over. Referenced to
/// the fixed elevation `E`.
fn place_approach(
    sim: &mut SimulationState,
    body_state: &BodyState,
    site: &RunwaySite,
    body_radius_m: f64,
) {
    let surface_radius = body_radius_m + site.elevation_m;
    let center_surface = site.center_dir * surface_radius;
    let threshold_point = center_surface - site.heading_tangent * RUNWAY_HALF_LENGTH_M;
    let approach_point = threshold_point - site.heading_tangent * APPROACH_BACK_M;
    let approach_dir = approach_point.normalize();

    let up = (body_state.orientation * approach_dir).normalize();
    let local_heading =
        (site.heading_tangent - approach_dir * site.heading_tangent.dot(approach_dir)).normalize();
    let heading_inertial = (body_state.orientation * local_heading).normalize();

    let radius = surface_radius + APPROACH_ALT_M;
    let position = body_state.position + up * radius;
    let r_rel = position - body_state.position;
    let surface_velocity = body_state.velocity + body_state.angular_velocity.cross(r_rel);
    let approach_vel = heading_inertial * APPROACH_SPEED_M_S - up * APPROACH_SINK_M_S;
    let velocity = surface_velocity + approach_vel;

    let nose = approach_vel.try_normalize().unwrap_or(heading_inertial);
    let dorsal = (up - nose * up.dot(nose)).try_normalize().unwrap_or(up);
    let right = nose.cross(dorsal).normalize();
    let basis = DMat3::from_cols(right, nose, dorsal);
    let attitude = AttitudeState {
        orientation: DQuat::from_mat3(&basis),
        angular_velocity: DVec3::ZERO,
    };

    place_craft(
        sim,
        coast_placement(StateVector { position, velocity }, attitude),
        None,
    );
    // Warp is left at the spawn default (paused); `spawn::apply_initial_warp`
    // sets the final level once on `Loading → Running` per `AutoRun`.
    sim.simulation.set_throttle(0.0);
    sim.simulation.set_target_body(Some(site.body_id));
}

// ---------------------------------------------------------------------------
// Per-frame transform + visibility
// ---------------------------------------------------------------------------

/// Position the runway like the player ship: a root-grid big_space child placed
/// in **f64** every frame. Anchoring it as a fixed-cell child of the *rotating*
/// body grid (the natural-looking choice) makes big_space rotate its multi-Mm
/// cell offset by an f32 quaternion, which jitters frame-to-frame at high warp
/// as the body spins fast (≈ decimetre ULP at planet radius). Computing the
/// centre in f64 here and letting the f32 `Transform.rotation` act only on the
/// small (≤ ~1.6 km) child vertex offsets keeps it rock-steady — the same trick
/// udlod uses with `PreciseRotation` for the terrain itself.
fn update_runway_transform(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut runways: Query<(&RunwayVisual, &mut CellCoord, &mut Transform)>,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Ok(grid) = root_grid.single() else {
        return;
    };
    for (rv, mut cell, mut transform) in &mut runways {
        let Some(state) = states.get(rv.body_id) else {
            continue;
        };
        // Thalos is free-spinning, so the surface orientation is the body
        // orientation (the free-spin branch of
        // `surface_body_to_world_orientation_f64` — the only kind a player
        // stands on). Tidally-locked bodies would need that helper's f64 path.
        let orientation = state.orientation.normalize();
        let center_world = state.position + orientation * rv.center_surface_body;
        let (next_cell, local) = grid.translation_to_grid(center_world);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = orientation.as_quat();
    }
}

/// Hide the runway when the camera is far from the body (mirrors the
/// terrain/impostor LOD swap), so it doesn't poke through the orbital impostor.
fn sync_runway_visibility(
    cam_q: Query<&GlobalTransform, With<ShipCamera>>,
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut runways: Query<(&RunwayVisual, &mut Visibility)>,
) {
    let Ok(cam) = cam_q.single() else {
        return;
    };
    let cam_pos = cam.translation();
    for (marker, mut vis) in &mut runways {
        let Some((_, body_tf)) = body_q.iter().find(|(b, _)| b.body_id == marker.body_id) else {
            continue;
        };
        let dist = (cam_pos - body_tf.translation()).length();
        let want = if dist < marker.swap_radius_m {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    }
}
