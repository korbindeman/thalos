//! Procedural planetary cloud cover via curl-noise warp advection.
//!
//! Pure Rust, zero Bevy. Produces a `Cubemap<f32>` of cloud density in
//! `[0, 1]` for a planet, using Wedekind's 2023 technique: a scalar
//! potential built from hemisphere-salted 3D Worley noise drives a curl
//! vector field tangent to the sphere; 50 iterations of Lagrangian
//! advection accumulate the flow into a warp field; the final density
//! is a 3D Worley fBm sampled at the advected direction.
//!
//! Reference:
//!   Wedekind, "Procedural generation of global cloud cover" (2023)
//!   <https://www.wedesoft.de/software/2023/03/20/procedural-global-cloud-cover/>
//!   Prototype source: github.com/wedesoft/sfsim/etc/cover.clj
//!
//! The flow field encodes prevailing winds and counter-rotating band
//! structure by construction, so the resulting cloud pattern carries
//! cyclonic spirals, frontal bands, and hemisphere-dependent rotation
//! sense — the structural cues that distinguish "weather from orbit"
//! from undifferentiated fBm.

mod bake;
mod flow;
mod worley;

pub use bake::{CloudBakeConfig, bake_cloud_cover};
pub use flow::{FlowConfig, curl_on_sphere, flow_field, flow_gradient};
pub use worley::{WorleyVolume, worley_3d_tileable, worley_fbm};
