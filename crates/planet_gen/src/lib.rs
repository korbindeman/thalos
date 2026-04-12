//! Procedural planet surface generation (spec v0.1).
//!
//! This crate is the pure-Rust, dependency-free core of the planet generator.
//! Bevy and rendering concerns live downstream in `thalos_planet_rendering`.
//!
//! The v0.1 scope is airless rocky bodies — moons and small planets with no
//! meaningful atmosphere, liquid surface, or biosphere.  See
//! `docs/stale/thalos_planetgen_spec.md` for the full spec.
//!
//! Entry points:
//! - [`PlanetDescriptor`] — authored physical inputs
//! - [`derived::compute`] — one-shot derivation of [`DerivedProperties`]
//! - [`reference_bodies`] — canonical fixtures (Luna, Mercury, Callisto, Rhea, Deimos)

pub mod crater;
pub mod derived;
pub mod descriptor;
pub mod detail;
pub mod giant_basin;
pub mod main_craters;
pub mod mare;
pub mod maturity;
pub mod noise;
pub mod pipeline;
pub mod primordial;
pub mod reference_bodies;
pub mod sampling;
pub mod seeding;
pub mod surface;

pub use crater::{Crater, SimpleCrater, stamp_crater, stamp_simple_crater};
pub use derived::{
    DerivedProperties, VolatileSpecies, compute, compute_with_luminosity,
};
pub use descriptor::{Composition, Overrides, ParentBody, PlanetDescriptor};
pub use detail::DetailParams;
pub use pipeline::{
    PipelineConfig, PipelineOutput, SlopeSource, generate, generate_with_derived,
};
pub use sampling::{equirect_lattice, fibonacci_lattice};
pub use seeding::{Rng, sub_seed};
pub use surface::{MaterialId, SurfaceState};
