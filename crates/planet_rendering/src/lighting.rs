//! Shared scene-lighting model.
//!
//! Every planet-surface material (impostor terrain, gas giant, ring, and
//! future real-scale terrain pipeline) reads a single `SceneLighting`
//! struct that captures everything needed to evaluate direct illumination
//! at a fragment:
//!
//! - **Stars.** Up to `MAX_STARS` light sources, each with a world-space
//!   direction toward the star, a scalar flux (lux, already scaled by the
//!   camera exposure gain), and a per-star linear-RGB color tint.
//! - **Ambient.** A scalar floor applied equally by every material.
//! - **Eclipse occluders.** Analytical sphere list tested by shaders as
//!   shadow rays along each star direction. Gives cross-body eclipses for
//!   free to any material that binds this struct.
//! - **Planetshine parent.** A single secondary light source — the
//!   orbital parent of a moon — that reflects star flux back at the
//!   fragment. Terrestrial moons use this; gas giants and rings leave it
//!   zeroed.
//!
//! The WGSL mirror lives at `crates/planet_rendering/src/shaders/lighting.wgsl`
//! and is registered as a shader library via `load_shader_library!` so
//! every material shader can `#import thalos::lighting::*`.

use bevy::math::Vec4;
use bevy::render::render_resource::ShaderType;

/// Maximum number of stars the lighting model supports. Binary/triple
/// systems are a stretch goal — today `solar_system.ron` defines one
/// star, so `star_count == 1` is the only live code path. Bumping this
/// requires changing the matching constant in `lighting.wgsl`.
pub const MAX_STARS: usize = 4;

/// Maximum number of eclipse occluders per fragment. 8 covers solar-
/// system-scale scenes comfortably. Matches `MAX_ECLIPSE_OCCLUDERS` in
/// `lighting.wgsl`.
pub const MAX_ECLIPSE_OCCLUDERS: usize = 8;

/// One star's per-fragment light contract.
///
/// `dir_flux.xyz` is the unit direction from the fragment toward the
/// star in world-render space. `dir_flux.w` is flux in lux, already
/// multiplied by the camera exposure gain the rest of the pipeline
/// uses, so shaders can multiply it into a BRDF response directly.
///
/// `color.xyz` is a per-star linear-RGB tint (defaults to white — stars
/// differing only in luminosity collapse to the scalar flux term).
/// `color.w` is reserved.
#[derive(Clone, Copy, ShaderType)]
pub struct StarLight {
    pub dir_flux: Vec4,
    pub color: Vec4,
}

impl Default for StarLight {
    fn default() -> Self {
        Self {
            dir_flux: Vec4::new(0.0, 1.0, 0.0, 0.0),
            color: Vec4::new(1.0, 1.0, 1.0, 0.0),
        }
    }
}

/// Full scene-lighting description consumed by every planet material.
///
/// Embedded as a sub-struct inside `PlanetParams`, `GasGiantParams`, and
/// `RingParams` so the CPU-side update path can produce one value per
/// body and write it into whichever material the body spawned.
///
/// Field order is load-bearing — the WGSL `SceneLighting` mirror must
/// match. `encase` handles std140 padding automatically for the derived
/// `ShaderType`, so the manual `_pad0` below is only there to keep the
/// 16-byte scalar header aligned cleanly before the `stars` array.
#[derive(Clone, ShaderType)]
pub struct SceneLighting {
    /// Number of valid entries in `stars`.
    pub star_count: u32,
    /// Number of valid entries in `occluders`.
    pub occluder_count: u32,
    /// Ambient illuminance (lux). Applied uniformly by every material.
    pub ambient_intensity: f32,
    pub scene_header_pad: f32,

    pub stars: [StarLight; MAX_STARS],

    /// Eclipse occluder spheres. xyz = world render-space center,
    /// w = render-unit radius. Unused slots zeroed. Shaders loop
    /// `0..occluder_count` and test a shadow ray per star direction.
    pub occluders: [Vec4; MAX_ECLIPSE_OCCLUDERS],

    /// Planetshine parent: xyz = world render-space center,
    /// w = render-unit radius. `w == 0` disables. Used by terrestrial
    /// moons to pick up reflected light from their parent body.
    pub planetshine_pos_radius: Vec4,

    /// Planetshine tint: xyz = Bond albedo × parent color (the effective
    /// per-wavelength reflectance the parent sends back at zero phase),
    /// w = enable flag (1.0 active, 0.0 disabled).
    pub planetshine_tint_flag: Vec4,
}

impl Default for SceneLighting {
    fn default() -> Self {
        Self {
            star_count: 0,
            occluder_count: 0,
            ambient_intensity: 0.0,
            scene_header_pad: 0.0,
            stars: [StarLight::default(); MAX_STARS],
            occluders: [Vec4::ZERO; MAX_ECLIPSE_OCCLUDERS],
            planetshine_pos_radius: Vec4::ZERO,
            planetshine_tint_flag: Vec4::ZERO,
        }
    }
}
