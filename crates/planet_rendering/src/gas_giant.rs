//! Gas / ice giant material.
//!
//! This is the gas-giant counterpart of [`PlanetMaterial`]. Where
//! `PlanetMaterial` composites a baked cubemap (height + albedo + SSBOs)
//! onto an impostor sphere, `GasGiantMaterial` has no baked textures at
//! all — the entire visible disk is synthesised per-fragment from a
//! small set of atmosphere parameters.
//!
//! ## Layer contract
//!
//! The shader reads two uniforms:
//!
//! - [`GasGiantParams`] — per-frame camera/lighting/orientation state.
//!   Rebuilt every frame by the game's rendering layer just like
//!   `PlanetParams`.
//! - [`GasGiantLayers`] — mostly static atmosphere authoring: palette,
//!   cloud-deck band structure, haze, rim halo. Rebuilt only when the
//!   body's `AtmosphereParams` change (typically once at spawn).
//!
//! The split keeps the per-frame update path small while leaving room
//! for the layer set to grow into storms, auroras, cloud-top height
//! displacement, etc. without disturbing the hot uniform.
//!
//! ## Future fidelity
//!
//! The current shader implements only the first three layers (cloud
//! deck, haze, rim halo). Storms and auroras are explicitly stubbed in
//! [`GasGiantLayers`] as zeroed fields, so adding them later is an
//! additive change to the shader and the uniform packing — no layout
//! break for bodies already shipping.

use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::math::UVec4;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::lighting::SceneLighting;

/// Maximum palette stops the shader supports. Ten covers Saturn's
/// full north/south Cassini palette (nine authored stops + a reserve
/// slot) with room to grow. If a body needs more, bump this constant
/// in lockstep with the WGSL `MAX_PALETTE_STOPS` and the
/// `array<vec4<f32>, N>` in the `GasGiantLayers` WGSL struct.
pub const MAX_PALETTE_STOPS: usize = 10;

/// Samples in the signed speed / turbulence latitude profiles. Packed
/// 4-per-Vec4, so the shader reads `PROFILE_N / 4` Vec4s.
pub const PROFILE_N: usize = 16;

/// Maximum analytic long-lived vortices per body. Jupiter's full
/// authored set (GRS + ovals) is well under this.
pub const MAX_VORTICES: usize = 16;

/// Per-frame uniform: camera, lighting, orientation.
///
/// Mirrors the structure of `PlanetParams` so the game's per-frame
/// update system can populate both with the same camera/light data.
#[derive(Clone, ShaderType)]
pub struct GasGiantParams {
    /// Sphere radius in render units (cloud-deck altitude).
    pub radius: f32,
    /// Cumulative rotation phase, radians. Advances at the rate implied
    /// by the body's `rotation_period_s` so bands scroll slowly.
    pub rotation_phase: f32,
    /// Raw sim time used by differential rotation, edge waves, and edge
    /// vortex chain epochs.
    pub elapsed_time: f32,
    pub _pad0: f32,
    /// Quaternion (xyzw) rotating world-space directions into
    /// body-local space. Identity = no rotation. Axial tilt and
    /// `rotation_phase` both fold into this.
    pub orientation: Vec4,
    /// Stars, eclipse occluders, ambient. See `crate::lighting::SceneLighting`.
    /// Gas giants leave the planetshine parent slot zeroed.
    pub scene: SceneLighting,
}

impl Default for GasGiantParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            rotation_phase: 0.0,
            elapsed_time: 0.0,
            _pad0: 0.0,
            orientation: Vec4::new(0.0, 0.0, 0.0, 1.0),
            scene: SceneLighting::default(),
        }
    }
}

/// Per-body authoring uniform: palette, bands, haze, rim halo.
///
/// Populated once at spawn from [`AtmosphereParams`]. The layout is
/// padded by hand so WGSL can decode it as a plain struct without
/// alignment gymnastics.
#[derive(Clone, ShaderType)]
pub struct GasGiantLayers {
    /// Palette stops: xyz = linear RGB, w = signed latitude in [-1, 1].
    /// Only the first `stop_count` entries are meaningful.
    pub palette: [Vec4; MAX_PALETTE_STOPS],

    /// Number of valid entries in `palette`. Shader iterates over this.
    pub stop_count: u32,

    /// Cloud-deck band frequency (bands per hemisphere).
    pub band_frequency: f32,
    /// Horizontal band warp amplitude (fraction of a band's width).
    pub band_warp: f32,
    /// Fine-scale turbulence amplitude.
    pub turbulence: f32,

    /// Overall cloud-deck tint multiplier. w = band_contrast (belt/zone
    /// luminance swing amplitude).
    pub tint: Vec4,

    /// Band shaping. x = band edge transition narrowness (1.0 = full
    /// smoothstep blend across a palette span, →0 = step-function edges
    /// between adjacent bands). y..w = padding.
    pub band_shape_params: Vec4,

    /// Haze: xyz = tint, w = thickness. Thickness 0 disables the layer.
    pub haze_tint_thickness: Vec4,
    /// Haze: x = terminator_bias, yzw = padding.
    pub haze_params: Vec4,

    /// Rim halo: xyz = color, w = intensity. Intensity 0 disables it.
    pub rim_color_intensity: Vec4,
    /// Rim halo: x = scale_height (render units), y = outer altitude
    /// (render units), zw = padding.
    pub rim_shape: Vec4,

    /// Signed per-latitude scroll rate profile, 16 samples packed four
    /// per Vec4. Zero everywhere disables differential rotation.
    pub speed_profile: [Vec4; 4],

    /// Per-latitude turbulence amplitude in [0, 1], 16 samples packed
    /// four per Vec4. Empty profile maps to a constant `1.0` curve.
    pub turbulence_profile: [Vec4; 4],

    /// x = differential_rotation_rate, y = edge_wave_amp,
    /// z = curl_amp, w = parallax_amp.
    pub dynamics: Vec4,

    /// x = vortex_count, y = edge_chain_slots, z = edge_chain_enabled,
    /// w = speed_profile_valid.
    pub counts: UVec4,

    /// x = edge_chain base_radius, y = strength, z = lifetime_s,
    /// w = unused.
    pub edge_chain: Vec4,

    /// xyz = terminator warmth colour, w = strength.
    pub terminator_warmth: Vec4,

    /// xyz = fresnel rim colour, w = strength.
    pub fresnel_rim: Vec4,

    /// Rayleigh blue-gap layer. xyz = scattered-blue colour, w = strength.
    /// Strength 0 disables the contribution.
    pub rayleigh_color: Vec4,
    /// Rayleigh parameters. x = haze_scale, y = clearing_threshold,
    /// z = latitude_bias, w = unused.
    pub rayleigh_params: Vec4,

    /// Per-channel Minnaert limb darkening exponents.
    /// x = red, y = green, z = blue, w = overall strength.
    /// w = 0 disables the effect.
    pub limb_exponents: Vec4,

    /// Ring shadow on the cloud deck.
    /// x = inner radius (render units), y = outer radius (render units),
    /// z = ringlet noise amplitude, w = enabled flag (0 or 1).
    pub ring_shadow: Vec4,

    /// Analytic vortex locations: xy = (lat, lon), z = angular radius,
    /// w = swirl strength.
    pub vortex_pos: [Vec4; MAX_VORTICES],

    /// Analytic vortex tints: xyz = linear RGB, w = unused.
    pub vortex_tint: [Vec4; MAX_VORTICES],

    /// Per-body noise seed, packed into two u32 lanes for WGSL
    /// compatibility (no native u64 uniform).
    pub seed_lo: u32,
    pub seed_hi: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl Default for GasGiantLayers {
    fn default() -> Self {
        Self {
            palette: [Vec4::ZERO; MAX_PALETTE_STOPS],
            stop_count: 0,
            band_frequency: 0.0,
            band_warp: 0.0,
            turbulence: 0.0,
            tint: Vec4::new(1.0, 1.0, 1.0, 0.22),
            band_shape_params: Vec4::new(0.20, 0.0, 0.0, 0.0),
            haze_tint_thickness: Vec4::ZERO,
            haze_params: Vec4::ZERO,
            rim_color_intensity: Vec4::ZERO,
            rim_shape: Vec4::ZERO,
            speed_profile: [Vec4::ZERO; 4],
            turbulence_profile: [Vec4::ONE; 4],
            dynamics: Vec4::ZERO,
            counts: UVec4::ZERO,
            edge_chain: Vec4::ZERO,
            terminator_warmth: Vec4::ZERO,
            fresnel_rim: Vec4::ZERO,
            rayleigh_color: Vec4::ZERO,
            rayleigh_params: Vec4::new(4.0, 0.35, 0.0, 0.0),
            limb_exponents: Vec4::new(0.25, 0.32, 0.40, 0.0),
            ring_shadow: Vec4::ZERO,
            vortex_pos: [Vec4::ZERO; MAX_VORTICES],
            vortex_tint: [Vec4::ZERO; MAX_VORTICES],
            seed_lo: 0,
            seed_hi: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

impl GasGiantLayers {
    /// Build a layer set from `AtmosphereParams` authored in the RON.
    ///
    /// `meters_per_render_unit` converts the real-world meters used in
    /// `AtmosphereParams` (scale heights, outer altitudes) into the
    /// render-unit space the shader works in. Pass the same factor the
    /// rest of the rendering pipeline uses.
    ///
    /// `rings` is the body-level ring system (now sibling to
    /// `atmosphere`, not a child of it). When present, its inner/outer
    /// radii feed the cloud-deck shadow term.
    pub fn from_params(
        atmos: &thalos_atmosphere_gen::AtmosphereParams,
        rings: Option<&thalos_atmosphere_gen::RingSystem>,
        meters_per_render_unit: f32,
    ) -> Self {
        let mut layers = Self::default();

        // ── Cloud deck ────────────────────────────────────────────────
        let stops = &atmos.cloud_deck.palette;
        let n = stops.len().min(MAX_PALETTE_STOPS);
        for (i, stop) in stops.iter().take(n).enumerate() {
            layers.palette[i] = Vec4::new(
                stop.color[0],
                stop.color[1],
                stop.color[2],
                stop.lat.clamp(-1.0, 1.0),
            );
        }
        layers.stop_count = n as u32;
        layers.band_frequency = atmos.cloud_deck.band_frequency;
        layers.band_warp = atmos.cloud_deck.band_warp;
        layers.turbulence = atmos.cloud_deck.turbulence;
        layers.tint = Vec4::new(
            atmos.cloud_deck.tint[0],
            atmos.cloud_deck.tint[1],
            atmos.cloud_deck.tint[2],
            atmos.cloud_deck.band_contrast,
        );
        layers.band_shape_params =
            Vec4::new(atmos.cloud_deck.band_sharpness, 0.0, 0.0, 0.0);

        // ── Haze ──────────────────────────────────────────────────────
        if let Some(haze) = &atmos.haze {
            layers.haze_tint_thickness =
                Vec4::new(haze.tint[0], haze.tint[1], haze.tint[2], haze.thickness);
            layers.haze_params = Vec4::new(haze.terminator_bias, 0.0, 0.0, 0.0);
        }

        // ── Rim halo ──────────────────────────────────────────────────
        if let Some(rim) = &atmos.rim_halo {
            layers.rim_color_intensity =
                Vec4::new(rim.color[0], rim.color[1], rim.color[2], rim.intensity);
            let inv_m = 1.0 / meters_per_render_unit.max(1.0);
            layers.rim_shape = Vec4::new(
                rim.scale_height_m * inv_m,
                rim.outer_altitude_m * inv_m,
                0.0,
                0.0,
            );
        }

        // ── Rayleigh blue-gap ─────────────────────────────────────────
        if let Some(ray) = &atmos.rayleigh {
            layers.rayleigh_color =
                Vec4::new(ray.color[0], ray.color[1], ray.color[2], ray.strength);
            layers.rayleigh_params = Vec4::new(
                ray.haze_scale.max(1e-3),
                ray.clearing_threshold.clamp(0.0, 1.0),
                ray.latitude_bias,
                0.0,
            );
        }

        // ── Per-channel limb darkening ────────────────────────────────
        if let Some(ld) = &atmos.limb_darkening {
            layers.limb_exponents = Vec4::new(
                ld.red.max(0.0),
                ld.green.max(0.0),
                ld.blue.max(0.0),
                ld.strength.clamp(0.0, 1.0),
            );
        }

        // ── Ring shadow cast onto the cloud deck ──────────────────────
        if let Some(rings) = rings {
            let inv_m = 1.0 / meters_per_render_unit.max(1.0);
            layers.ring_shadow = Vec4::new(
                rings.inner_radius_m * inv_m,
                rings.outer_radius_m * inv_m,
                rings.ringlet_noise.clamp(0.0, 1.0),
                1.0,
            );
        }

        // ── Limb lighting tweaks ──────────────────────────────────────
        if let Some(limb) = &atmos.limb {
            layers.terminator_warmth = Vec4::new(
                limb.terminator_warmth[0],
                limb.terminator_warmth[1],
                limb.terminator_warmth[2],
                limb.terminator_strength,
            );
            layers.fresnel_rim = Vec4::new(
                limb.fresnel_color[0],
                limb.fresnel_color[1],
                limb.fresnel_color[2],
                limb.fresnel_strength,
            );
        }

        // ── Latitude profiles ─────────────────────────────────────────
        let speed_valid = !atmos.cloud_deck.speed_profile.is_empty();
        if speed_valid {
            fill_profile(&mut layers.speed_profile, &atmos.cloud_deck.speed_profile);
        }
        // Turbulence profile defaults to 1.0 everywhere (constant scalar
        // `turbulence` from the cloud deck takes over). Author-provided
        // profile overrides that uniform curve.
        if !atmos.cloud_deck.turbulence_profile.is_empty() {
            fill_profile(
                &mut layers.turbulence_profile,
                &atmos.cloud_deck.turbulence_profile,
            );
        }

        // ── Dynamics scalar bag ───────────────────────────────────────
        layers.dynamics = Vec4::new(
            atmos.cloud_deck.differential_rotation_rate,
            atmos.cloud_deck.edge_wave_amp,
            atmos.cloud_deck.curl_amp,
            atmos.cloud_deck.parallax_amp,
        );

        // ── Named vortices ────────────────────────────────────────────
        let n_vort = atmos.cloud_deck.named_vortices.len().min(MAX_VORTICES);
        for (i, v) in atmos.cloud_deck.named_vortices.iter().take(n_vort).enumerate() {
            layers.vortex_pos[i] = Vec4::new(v.lat.clamp(-1.0, 1.0), v.lon, v.radius, v.strength);
            layers.vortex_tint[i] = Vec4::new(v.tint[0], v.tint[1], v.tint[2], 0.0);
        }

        // ── Edge vortex chain ─────────────────────────────────────────
        let (chain_slots, chain_on) = match &atmos.cloud_deck.edge_vortex_chain {
            Some(c) => {
                layers.edge_chain = Vec4::new(c.base_radius, c.strength, c.lifetime_s, 0.0);
                (c.slots_per_band, 1u32)
            }
            None => (0, 0),
        };

        layers.counts = UVec4::new(n_vort as u32, chain_slots, chain_on, speed_valid as u32);

        layers.seed_lo = atmos.seed as u32;
        layers.seed_hi = (atmos.seed >> 32) as u32;
        layers
    }
}

/// Pack a variable-length scalar profile into 4 Vec4s (16 samples).
/// Short profiles are linearly resampled; long profiles are truncated
/// at the 16th sample. A single-entry profile becomes a constant.
fn fill_profile(dst: &mut [Vec4; 4], src: &[f32]) {
    for i in 0..PROFILE_N {
        let t = i as f32 / (PROFILE_N - 1) as f32;
        let sample = sample_profile(src, t);
        dst[i / 4][i % 4] = sample;
    }
}

fn sample_profile(src: &[f32], t: f32) -> f32 {
    match src.len() {
        0 => 0.0,
        1 => src[0],
        n => {
            let x = t * (n - 1) as f32;
            let i0 = x.floor() as usize;
            let i1 = (i0 + 1).min(n - 1);
            let f = x - i0 as f32;
            src[i0] * (1.0 - f) + src[i1] * f
        }
    }
}

/// Gas giant material.
///
/// Binding layout (group 3):
///
/// | Binding | Kind    | WGSL type       | Source     |
/// |---------|---------|-----------------|------------|
/// | 0       | uniform | GasGiantParams  | `params`   |
/// | 1       | uniform | GasGiantLayers  | `layers`   |
///
/// Deliberately tiny for first pass. Future additions (storms SSBO,
/// aurora parameters, optional noise texture) slot in as new bindings
/// without disturbing 0/1.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GasGiantMaterial {
    #[uniform(0)]
    pub params: GasGiantParams,
    #[uniform(1)]
    pub layers: GasGiantLayers,
}

impl Material for GasGiantMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/gas_giant.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/gas_giant.wgsl".into()
    }

    // Must stay Opaque so the gas giant writes depth before rings (a
    // Blend material) draw — otherwise the single ring mesh, which
    // straddles the planet, sorts as one unit against the gas giant and
    // flips between "whole ring in front" and "whole ring behind" as
    // the camera moves, causing visible flashing. The rim-halo
    // compositing bug documented for `PlanetMaterial` exists here too
    // but is currently invisible: the halo only overwrites the black
    // sky. Revisit if a gas giant ever needs to render in front of
    // another bright body.
    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Handle component so the per-frame update system can find and mutate
/// the material from the parent `CelestialBody` entity.
#[derive(Component)]
pub struct GasGiantMaterialHandle(pub Handle<GasGiantMaterial>);
