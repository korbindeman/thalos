//! Resources and components shared across the rendering submodules.
//!
//! Per-submodule resources (e.g. trail caches, click state) live in
//! their respective modules; this file holds only types touched by more
//! than one rendering concern.

use bevy::prelude::*;
use std::collections::HashMap;
use thalos_body_render::{
    GasGiantMaterial, RingMaterial, SolidPlanetHaloMaterial, SolidPlanetMaterial,
};

pub use crate::solar_system_state::{SimulationState, SolarSystemState};

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Linear-RGB tint to use as a body's planetshine emission. Populated when
/// the body's surface info first becomes known: at bake completion for
/// terrain bodies (from `StaticSurfaceData::mean_albedo`), at spawn for gas giants
/// (from cloud albedo). Bodies without an entry contribute no planetshine
/// to their moons.
#[derive(Resource, Default)]
pub struct PlanetshineTints {
    pub by_body: HashMap<usize, [f32; 3]>,
}

/// Camera exposure model. Acts as the semantic "sensor" of the game camera:
/// it owns how focus distance maps to display brightness and how much grain
/// is added in consequence. Every system that cares about "how much flux
/// does the shader see" or "how much noise should the post stack add" reads
/// this resource rather than recomputing from focus distance.
///
/// Linear-in-distance compensation: outer-system focus pulls distant bodies
/// out of black without erasing the distance cue. Concretely, the display
/// flux at the focus body scales as `LIGHT_AT_1AU * (1 AU / focus_d)`,
/// so a body at 40 AU remains roughly 40x dimmer than the same body at
/// 1 AU even when focused.
///
/// The gain applied to each body's raw inverse-square flux in the impostor
/// shader is `exposure.gain = focus_d / 1 AU`. Combined with the raw
/// `(AU/body_d)^2` falloff baked into `update_planet_light_dirs`, this
/// yields the focus-relative display flux above.
#[derive(Resource, Reflect, Default, Clone, Copy, Debug)]
#[reflect(Resource)]
pub struct CameraExposure {
    /// Camera focus body's distance from the star, in meters.
    pub focus_dist_m: f64,
    /// Multiplicative gain applied to per-body raw inverse-square flux.
    pub gain: f32,
    /// Log2(gain). Positive = we're pushing dark outer-system scenes;
    /// negative = we're pulling down bright inner-system scenes. Drives
    /// film grain strength (and, later, lens flare intensity).
    pub ev: f32,
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct CelestialBody {
    pub body_id: usize,
    pub is_star: bool,
    pub render_radius: f32,
    /// True physical radius in metres (not clamped like render_radius).
    pub radius_m: f64,
}

/// Marks a body whose baked surface is tidally locked to its parent. Each
/// frame the shared surface orientation keeps the baked near-side (local +Z,
/// where the mare/tidal asymmetry lives) facing the parent body; impostors use
/// the world→body form and real-space terrain uses the inverse body→world form.
#[derive(Component)]
pub(super) struct TidallyLocked {
    pub(super) parent_id: usize,
}

#[derive(Component)]
pub struct ShipMarker;

/// Root of the player's ship in 3D space. Its children are the ship parts
/// rendered at 1:1 meter scale in the entity's local frame; the entity's
/// `Transform::scale` compensates so the ship renders at real size in the
/// solar-system-wide render-units coordinate space (see [`WorldScale`]).
///
/// Present in both views. In map view it's hidden (the flat `ShipMarker`
/// billboard stands in for it); in ship view it becomes visible and the
/// camera orbits it.
#[derive(Component)]
pub struct PlayerShip;

/// The craft the player is currently controlling — the **N-craft accessor seam**.
///
/// Today the game has exactly one [`PlayerShip`]; this resource simply names it
/// by entity. Its purpose is architectural: it is the single sanctioned answer to
/// "which craft is active", so consumers read `active.0` (an `Option<Entity>`,
/// `None` during the respawn/relaunch rebuild window) instead of assuming exactly
/// one craft via `q.single()` — a call that *panics* the moment a second craft
/// entity exists. When N craft land, this picks the active one and nothing else
/// changes. New per-craft state should be a **component on this entity**, not a
/// new global resource (see `docs/architecture_cleanup.md` §E).
///
/// **Sole writer:** [`track_active_craft`].
#[derive(Resource, Default)]
pub struct ActiveCraft(pub Option<Entity>);

/// Sole writer of [`ActiveCraft`]: mirror the (currently single) [`PlayerShip`]
/// entity into it each frame, `None` when no craft exists (the respawn/relaunch
/// rebuild window). Centralises the one `q.single()` so every other consumer can
/// take the active craft by id without its own single-craft assumption; when N
/// craft exist this is where "which is active" is decided.
pub fn track_active_craft(
    ships: Query<Entity, With<PlayerShip>>,
    mut active: ResMut<ActiveCraft>,
) {
    let current = ships.iter().next();
    if active.0 != current {
        active.0 = current;
    }
}

/// Marker for the directional light that simulates sunlight toward the focus body.
#[derive(Component)]
pub(super) struct SunLight;

/// Marker for the secondary directional light that simulates moonlight — the
/// brightest child moon reflecting the star onto the body the craft is on.
/// Driven by [`crate::rendering::lighting::update_moon_light`]; lights the
/// `StandardMaterial` craft hull + surface structures at night (the terrain has
/// its own moonlight term in `body_terrain.wgsl`). Shadows are disabled — soft
/// moonlight doesn't need a second cascade pass.
#[derive(Component)]
pub(super) struct MoonLight;

/// Marker for the map-view mesh child of a celestial body. Inherits the
/// parent's transform, which is updated at [`MAP_SCALE`](crate::coords::MAP_SCALE)
/// each frame.
#[derive(Component)]
pub(super) struct BodyMesh;

/// Marker for the ship-view mesh child of a celestial body. These meshes
/// live under the body's real-space BigSpace grid and use local
/// [`SHIP_SCALE`](crate::coords::SHIP_SCALE) sizing.
#[derive(Component)]
pub(super) struct ShipBodyMesh;

/// Marker for the flat circle icon child of a celestial body.
#[derive(Component)]
pub(super) struct BodyIcon;

/// Material handles for a [`SolidPlanetMaterial`] placeholder body —
/// the placeholder used by bodies that don't have a terrain pipeline.
#[derive(Component)]
pub struct SolidPlanetMaterials {
    pub map: Handle<SolidPlanetMaterial>,
    pub ship: Handle<SolidPlanetMaterial>,
    /// Map-layer atmosphere rim companion, present only for bodies with a
    /// `terrestrial_atmosphere`. Drawn as a premultiplied sibling billboard
    /// outside the solid disc; kept in lockstep with `map` (radius + scene
    /// lighting) by `update_solid_planet_params`. `None` for airless bodies.
    pub map_halo: Option<Handle<SolidPlanetHaloMaterial>>,
}

/// Material handles for a [`GasGiantMaterial`] body.
#[derive(Component)]
pub struct GasGiantMaterials {
    pub map: Handle<GasGiantMaterial>,
    pub ship: Handle<GasGiantMaterial>,
}

/// Per-ring-entity marker for the map-layer ring child. Carries its own
/// [`RingMaterial`] handle so per-frame updates can find it.
#[derive(Component)]
pub(super) struct MapRingMaterial(pub(super) Handle<RingMaterial>);

/// Per-ring-entity marker for the ship-layer ring child. Mirror of
/// [`MapRingMaterial`] for the ship-scale instance.
#[derive(Component)]
pub(super) struct ShipRingMaterial(pub(super) Handle<RingMaterial>);

// `PendingPlanetGeneration` and `PendingPlanetBake` were removed when the
// game switched to synchronous bake loading. Procedural bodies now spawn
// directly with their final `PlanetMaterial` impostor via
// `super::generation::install_baked_planet`; there is no async task to
// poll. See `crates/game/src/rendering/generation.rs`.

#[derive(Component)]
pub struct RealSpaceBody {
    pub body_id: usize,
}
