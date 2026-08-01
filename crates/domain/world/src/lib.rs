//! Authored source of truth for the Thalos world.
//!
//! This crate owns the *data* definition of the system and its bodies — the
//! physical and orbital parameters, the per-body subsystem-config aggregate
//! ([`BodyDefinition`]), and the RON loader ([`parsing`]). It is pure Rust with
//! no Bevy, sitting below physics, terrain generation, and rendering so all
//! three read the same body definitions.
//!
//! Boundary rule (see `docs/architecture.md`): **authored data lives here;
//! algorithms and runtime simulation state live in `thalos_physics_canonical`,
//! which depends on this crate, never the reverse.** Keeping `world` free of
//! physics runtime types is what makes that one-way dependency hold.

pub mod atmosphere;
pub mod body;
pub mod ocean;
pub mod parsing;

pub use body::{
    AU_TO_METERS, BodyDefinition, BodyId, BodyKind, G, OrbitalElements, SolarSystemDefinition,
    StateVector, keplerian_basis, orbital_elements_to_cartesian,
};
pub use ocean::OceanState;

// The atmosphere data schemas live here now (folded in from the former
// `thalos_atmosphere` crate). Re-export the commonly-named types at the crate
// root; the full set is under `thalos_world::atmosphere`.
pub use atmosphere::{
    AtmosphereParams, AtmosphereProfile, AtmosphereSample, CLOUD_BAND_COUNT, CloudClimate,
    RingSystem, TerrestrialAtmosphere,
};

// Re-export the terrain config types the body definition aggregates, so
// consumers (and `physics_canonical`) have a single import surface for world
// data and don't need to depend on `thalos_terrain` directly just to name a
// `BodyDefinition` field.
pub use thalos_terrain::{TectonicConfig, TerrainConfig};
