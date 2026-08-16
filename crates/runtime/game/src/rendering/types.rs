//! Resources and components shared across the rendering submodules.
//!
//! The engine-agnostic scene vocabulary (`CelestialBody`, `PlayerShip`,
//! `ActiveCraft`, `CameraExposure`, …) lives in `thalos_game_state::scene`
//! (Phase 5a, ADR-20260731T024003Z) and is re-exported here; this file keeps
//! the renderer-specific material-handle components and per-submodule
//! markers. Per-submodule resources (e.g. trail caches, click state) live in
//! their respective modules.

use bevy::prelude::*;
use thalos_body_render::{
    GasGiantMaterial, RingMaterial, SolidPlanetHaloMaterial, SolidPlanetMaterial,
};

pub use crate::solar_system_state::{SimulationState, SolarSystemState};
pub use thalos_game_state::scene::{
    ActiveCraft, CameraExposure, CelestialBody, CraftIdentity, CraftRoot, PlanetshineTints,
    PlayerShip, RealSpaceBody, ShipMarker, TidallyLocked,
};

/// Sole writer of [`ActiveCraft`]: resolve the canonical active [`CraftIdentity`]
/// to its runtime [`CraftRoot`]. `None` is expected during relaunch and before
/// an EVA local body has materialized.
pub fn track_active_craft(
    mut commands: Commands,
    sim: Res<SimulationState>,
    roots: Query<(Entity, &CraftIdentity, Has<PlayerShip>), With<CraftRoot>>,
    mut active: ResMut<ActiveCraft>,
) {
    let active_id = sim.simulation.active_craft_id();
    let mut current = None;
    for (entity, identity, selected_marker) in &roots {
        let selected = identity.0 == active_id;
        if selected {
            current = Some(entity);
        }
        match (selected, selected_marker) {
            (true, false) => {
                commands.entity(entity).insert(PlayerShip);
            }
            (false, true) => {
                commands.entity(entity).remove::<PlayerShip>();
            }
            _ => {}
        }
    }
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
// poll. See `crates/runtime/game/src/rendering/generation.rs`.
