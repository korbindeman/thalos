//! Runtime body-install support + reference-cloud patching.
//!
//! Procedural bodies generate their terrain at runtime via
//! [`thalos_terrain::ProceduralSurface`] behind the `SurfaceQuery` seam — there
//! is no pre-baked artifact to load. [`super::spawn::spawn_bodies`] spawns the
//! body shell + a solid-color impostor synchronously and registers the runtime
//! surface for the near-surface height source ([`HeightSourceRegistry`]) and the
//! propagator ([`crate::GameTerrainRegistry`]); the ground-LOD terrain itself is
//! spawned on demand by [`crate::rendering::terrain_residency`].
//!
//! The [`ProceduralInstallExtras`] / [`WorldStateAssets`] `SystemParam`s bundle
//! the registries `spawn_bodies` writes into so it stays under Bevy 0.18's
//! 16-param limit.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use thalos_physics_local::HeightSourceRegistry;

use super::types::PlanetshineTints;

/// Registry handles `spawn_bodies` needs when registering each procedural
/// body's runtime surface. Bundled to stay under the 16-param limit.
#[derive(SystemParam)]
pub(super) struct ProceduralInstallExtras<'w> {
    pub(super) height_sources: ResMut<'w, HeightSourceRegistry>,
    pub(super) terrain_registry: Res<'w, crate::GameTerrainRegistry>,
}

/// World-level state the spawn path writes into. Bundled to stay under the
/// 16-param limit.
#[derive(SystemParam)]
pub(super) struct WorldStateAssets<'w> {
    pub(super) planetshine: ResMut<'w, PlanetshineTints>,
}
