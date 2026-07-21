//! Celestial sphere — procedural sky model for Thalos.
//!
//! Sources are stored as physical quantities (spectra, flux) rather than
//! RGB pixels, so the same `Universe` feeds both the realtime skybox and
//! (later) a telescope imaging path at arbitrary wavelengths.
//!
//! See `docs/celestial.md` for architecture.

pub mod coords;
pub mod spectrum;

pub mod sources;

pub mod catalog;

pub mod generate;

pub mod render;

pub use catalog::Universe;
pub use coords::{EquatorialCoord, UnitVector3};
pub use sources::{Galaxy, NebulaField, Source, Star};
pub use spectrum::{Passband, Spectrum, passbands};
