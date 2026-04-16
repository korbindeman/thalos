//! Rendering adapters over `Universe`.
//!
//! Two consumers share the same catalog:
//!   * [`cubemap`] — bakes the whole sphere into an HDR cubemap at
//!     startup for the realtime skybox.
//!   * (later) telescope imager — integrates a pointed FOV to a sensor
//!     frame.

pub mod cubemap;
pub mod psf;

pub use cubemap::{BakeParams, HdrCubemap, bake_skybox};
