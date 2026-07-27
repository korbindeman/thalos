//! Shared scene-lighting model.
//!
//! Every planet-surface material (impostor, gas giant, ring, ground-LOD
//! terrain) reads a single `SceneLighting` struct that captures
//! everything needed to evaluate direct illumination at a fragment:
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
//! The WGSL mirror lives at `shaders/lighting.wgsl` and is registered as
//! a shader library by [`crate::PlanetLightingPlugin`], so every material
//! shader can `#import thalos::lighting::*`.

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

/// Scene-flux → radiance normalisation. Mirrors `SCENE_FLUX_SCALE` in
/// `lighting.wgsl`; it absorbs the BRDF normalisation (the 1/4π family) so the
/// spine's BRDFs return bare radiance factors and callers multiply flux in
/// directly. Keep in lockstep.
pub const SCENE_FLUX_SCALE: f32 = 0.5;

/// Direct-sun reflectance scale. Mirrors `SURFACE_DIRECT_SCALE` in
/// `lighting.wgsl`. Keep in lockstep.
pub const SURFACE_DIRECT_SCALE: f32 = 0.23;

/// The camera exposure at which **Bevy's stock PBR renders a surface exactly as
/// bright as the spine does** — the bridge between the two lighting universes
/// (`gfx §3`), given the flux→lux constant the Bevy sun is driven with.
///
/// Both paths are fed the same heliocentric scene flux, and each turns one unit
/// of it into display radiance its own way. For a Lambert-ish surface of albedo
/// `a` at incidence `cos θ`:
///
/// - spine: `a · cos θ · flux · SCENE_FLUX_SCALE · SURFACE_DIRECT_SCALE`
///   (its BRDFs return radiance factors — the 1/π lives in `SCENE_FLUX_SCALE`);
/// - Bevy: `a/π · cos θ · (flux · lux_per_spine_flux) · view.exposure`.
///
/// Equating them leaves exposure as the only free term. **This was never set:**
/// nothing inserted an [`Exposure`] on the camera, so the Bevy universe ran at
/// its `EV100_BLENDER` default (9.7) — about 1.5 stops hot — and every
/// StandardMaterial surface, including the NTR-X1 tile terrain, rendered
/// brighter and flatter than the spine ground beside it (backlog NTR-X5).
///
/// Returned as an [`Exposure`] rather than a bare number so the caller cannot
/// mix up Bevy's `exposure = 1 / (2^ev100 · 1.2)` convention. Because only the
/// **product** `lux_per_spine_flux × exposure` reaches a Bevy-lit fragment,
/// retuning either input keeps parity as long as it flows through here.
pub fn spine_parity_exposure(lux_per_spine_flux: f32) -> bevy::camera::Exposure {
    let exposure =
        SCENE_FLUX_SCALE * SURFACE_DIRECT_SCALE * core::f32::consts::PI / lux_per_spine_flux;
    bevy::camera::Exposure {
        ev100: -(exposure * 1.2).log2(),
    }
}

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
#[derive(Clone, Copy, PartialEq, ShaderType)]
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
/// Embedded as a sub-struct inside `PlanetParams`, `GasGiantParams`,
/// `RingParams`, and `BodyTerrainMaterial` so the CPU-side update path
/// can produce one value per body and write it into whichever material
/// the body spawned.
///
/// Field order is load-bearing — the WGSL `SceneLighting` mirror must
/// match. `encase` handles std140 padding automatically for the derived
/// `ShaderType`, so the manual `scene_header_pad` below is only there to
/// keep the 16-byte scalar header aligned cleanly before the `stars`
/// array.
#[derive(Clone, PartialEq, ShaderType)]
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

    /// Moonlight onto this body's surface — the reverse of planetshine: the
    /// brightest child moon, treated as a single soft directional light, so a
    /// full moon overhead lights the night landscape. xyz = unit direction
    /// from the surface toward the moon in world-render space (≈ moon dir from
    /// the body centre — the moon is far enough that per-fragment parallax is
    /// negligible). w = artistic flux already folded with the moon's phase,
    /// size, albedo, and distance (NOT physical lux — physical moonlight is
    /// ~1e-6 of sunlight and would be invisible after tonemapping; this is the
    /// tuned night-lift). Consumed by `shade_surface`, night- and
    /// horizon-gated; `0` flux disables it.
    pub moonlight_dir_flux: Vec4,

    /// Moonlight tint: xyz = the moon's linear-RGB hue (normalised so the flux
    /// carries the brightness, the tint only the colour), w = enable flag
    /// (1.0 active, 0.0 disabled — bodies with no lit child moon leave it 0).
    pub moonlight_color: Vec4,
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
            moonlight_dir_flux: Vec4::ZERO,
            moonlight_color: Vec4::ZERO,
        }
    }
}
