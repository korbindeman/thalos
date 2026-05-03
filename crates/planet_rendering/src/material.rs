use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::shader::ShaderRef;
use thalos_atmosphere_gen::TerrestrialAtmosphere;
use thalos_terrain_gen::BodyData;

use crate::lighting::SceneLighting;

/// Re-export so existing call sites resolve unchanged.
pub use crate::lighting::MAX_ECLIPSE_OCCLUDERS;

/// Per-planet uniform data sent to the impostor shader.
#[derive(Clone, ShaderType)]
pub struct PlanetParams {
    /// Sphere radius in render units.
    pub radius: f32,
    /// Height range in meters — the maximum absolute displacement stored in the
    /// R16Unorm height cubemap.  The shader needs this to scale gradients correctly.
    pub height_range: f32,
    /// Surface roughness hint for the terminator wrap term (0.0 = smooth gas
    /// giant, 1.0 = very rough regolith). Feeds the Lambert wrap slack used to
    /// fake multiple scattering near the day/night line.
    pub terminator_wrap: f32,
    /// Debug fullbright toggle (0.0 = off, 1.0 = on). When on, the direct-sun
    /// term is flattened so the surface reads as unshaded albedo; atmosphere,
    /// Rayleigh, and cloud compositing still run normally for authoring.
    pub fullbright: f32,
    /// Quaternion (xyzw) rotating world-space directions into body-local space
    /// (where the cubemaps were baked). For tidally-locked moons this aligns
    /// the baked near-side (+Z) with the direction toward the parent body.
    /// Identity quaternion = no rotation.
    pub orientation: Vec4,
    /// Stars, eclipse occluders, ambient, and planetshine parent. See
    /// `crate::lighting::SceneLighting`.
    pub scene: SceneLighting,
    /// Sea-level elevation (meters, in the same encoding as the height
    /// cubemap — 0 m = the post-rebase sea level on water worlds). The
    /// shader triggers the water BRDF where `sample_height_m(dir) <
    /// sea_level_m`. Set to a large negative sentinel for airless bodies
    /// so no fragment ever crosses the threshold.
    pub sea_level_m: f32,
    /// Apparent deep-water color and minimum optical depth. xyz is linear RGB;
    /// w is the minimum water-column depth used for shading, in meters. This
    /// keeps flat ocean placeholders from rendering as 1 m-deep shelf water.
    pub water_color_depth: Vec4,
    /// Amplitude (in radians of arc on the unit sphere) of the canonical
    /// high-frequency *domain warp* applied before the impostor reads
    /// the baked height cubemap. The cubemap-texel staircase visible
    /// from orbit is a function of the texel grid; perturbing the
    /// sample direction by ~1 texel of arc (~7.5e-4 rad on Thalos)
    /// breaks the iso-contour out of the grid without adding any
    /// surface height roughness — the bake's bilinear-interpolated
    /// height field is read at a fractally perturbed location instead.
    ///
    /// Continues the same fractal-warp scheme that
    /// `topography.rs`/`coarse_elevation.rs` apply at lower
    /// frequencies during the bake. Bake + shader warps compose into
    /// a single canonical multi-band warp.
    ///
    /// Set to 0 to disable (e.g. airless bodies whose bake already
    /// captures all visible bands). For Earth-like bodies, ~8e-4 rad
    /// (~1 texel on Thalos) is the design point.
    pub coastline_warp_amp_radians: f32,
    /// Cycles-per-meter of the warp's base octave. `1.0 / 2500.0`
    /// puts the largest warp wavelength at ~2.5 km on the surface;
    /// subsequent octaves (lacunarity 2) extend into sub-km territory
    /// to give fractal texture below the cubemap Nyquist.
    pub coastline_warp_freq_per_m: f32,
    /// Amplitude (in meters) of the canonical high-frequency *height
    /// jitter* added on top of the (warped) baked height. Provides
    /// sub-texel surface detail visible on close approach. Set to 0
    /// to disable. ~30 m is a sensible default — invisible at orbit,
    /// visible up close.
    pub coastline_jitter_amp_m: f32,
    /// Cycles-per-meter of the height-jitter's base octave.
    pub coastline_jitter_freq_per_m: f32,
    /// Octave count, shared by the warp and jitter fbm calls.
    /// Capped at 8 in the shader.
    pub coastline_octaves: u32,
    /// Per-body seed for the canonical high-frequency bands. Folds
    /// the 64-bit body seed (low XOR high halves), then xors a
    /// per-band magic so the warp/jitter fields decorrelate from any
    /// bake-time fbm fields that share the body seed.
    pub coastline_seed: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PlanetWaterParams {
    pub color_depth: Vec4,
}

impl PlanetWaterParams {
    pub fn from_body_data(body: &BodyData) -> Self {
        if body.sea_level_m.is_some() {
            Self {
                color_depth: Vec4::new(
                    body.mean_albedo[0],
                    body.mean_albedo[1],
                    body.mean_albedo[2],
                    120.0,
                ),
            }
        } else {
            Self {
                color_depth: Vec4::new(0.012, 0.040, 0.090, 120.0),
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlanetCoastlineParams {
    pub warp_amp_radians: f32,
    pub jitter_amp_m: f32,
    pub seed: u32,
}

impl PlanetCoastlineParams {
    pub fn from_body_data(body: &BodyData) -> Self {
        let seed = (body.detail_params.seed as u32)
            ^ ((body.detail_params.seed >> 32) as u32)
            ^ 0xC0A5_711E_u32;

        let has_ocean = body.sea_level_m.is_some();
        let flat_ocean_placeholder = has_ocean
            && body.materials.is_empty()
            && body.craters.is_empty()
            && body.volcanoes.is_empty()
            && body.channels.is_empty();

        if has_ocean && !flat_ocean_placeholder {
            Self {
                // ~1 texel of arc on a 2048² cube = 2π/(4·2048) ≈ 7.7e-4 rad.
                warp_amp_radians: 8.0e-4,
                jitter_amp_m: 30.0,
                seed,
            }
        } else {
            Self {
                warp_amp_radians: 0.0,
                jitter_amp_m: 0.0,
                seed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_terrain_gen::{
        OceanTerrainConfig, TerrainCompileContext, TerrainCompileOptions, TerrainConfig,
        compile_terrain_config,
    };

    #[test]
    fn flat_ocean_placeholder_has_no_coastline_height_jitter() {
        let terrain = TerrainConfig::Ocean(OceanTerrainConfig {
            seed: 1003,
            cubemap_resolution: 16,
            seabed_albedo: [0.02, 0.05, 0.10],
            water_roughness: 0.04,
            sea_level_m: 1.0,
        });
        let body = compile_terrain_config(
            &terrain,
            &TerrainCompileContext {
                body_name: "Thalos".to_string(),
                radius_m: 6_000_000.0,
                gravity_m_s2: 9.81,
                rotation_hours: None,
                obliquity_deg: None,
                tidal_axis: None,
                axial_tilt_rad: 0.0,
            },
            TerrainCompileOptions::default(),
        )
        .expect("ocean terrain should compile");

        let coastline = PlanetCoastlineParams::from_body_data(&body);
        assert_eq!(coastline.warp_amp_radians, 0.0);
        assert_eq!(coastline.jitter_amp_m, 0.0);
    }

    #[test]
    fn ocean_water_color_uses_uniform_body_tint() {
        let terrain = TerrainConfig::Ocean(OceanTerrainConfig {
            seed: 1003,
            cubemap_resolution: 16,
            seabed_albedo: [0.02, 0.05, 0.10],
            water_roughness: 0.04,
            sea_level_m: 1.0,
        });
        let body = compile_terrain_config(
            &terrain,
            &TerrainCompileContext {
                body_name: "Thalos".to_string(),
                radius_m: 6_000_000.0,
                gravity_m_s2: 9.81,
                rotation_hours: None,
                obliquity_deg: None,
                tidal_axis: None,
                axial_tilt_rad: 0.0,
            },
            TerrainCompileOptions::default(),
        )
        .expect("ocean terrain should compile");

        let water = PlanetWaterParams::from_body_data(&body);
        assert!((water.color_depth.x - 0.02).abs() < 0.002);
        assert!((water.color_depth.y - 0.05).abs() < 0.002);
        assert!((water.color_depth.z - 0.10).abs() < 0.002);
        assert_eq!(water.color_depth.w, 120.0);
    }
}

impl Default for PlanetParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            height_range: 1.0,
            terminator_wrap: 0.0,
            fullbright: 0.0,
            orientation: Vec4::new(0.0, 0.0, 0.0, 1.0),
            scene: SceneLighting::default(),
            // Large negative sentinel — airless bodies leave this at the
            // default, and the shader's `sample_height_m(dir) < sea_level_m`
            // test never fires.
            sea_level_m: -1.0e9,
            water_color_depth: Vec4::new(0.012, 0.040, 0.090, 120.0),
            // Defaults are 0 so airless / preview bodies pay zero
            // cost. Bodies with a sea level should set warp_amp to
            // ~8e-4 rad (≈1 texel on Thalos) and jitter_amp_m to
            // ~30 m for the design-point Earth-like look.
            coastline_warp_amp_radians: 0.0,
            coastline_warp_freq_per_m: 1.0 / 2500.0,
            coastline_jitter_amp_m: 0.0,
            coastline_jitter_freq_per_m: 1.0 / 1500.0,
            coastline_octaves: 4,
            coastline_seed: 0,
        }
    }
}

/// Per-planet **detail** uniform consumed by the shader's per-fragment
/// small-crater synthesis pass and the SSBO iteration loop.
#[derive(Clone, ShaderType)]
pub struct PlanetDetailParams {
    pub body_radius_m: f32,
    pub d_min_m: f32,
    pub d_max_m: f32,
    pub sfd_alpha: f32,
    pub global_k_per_km2: f32,
    pub d_sc_m: f32,
    pub body_age_gyr: f32,
    /// Radius cutoff (meters) above which craters were baked into the
    /// cubemap. The SSBO iteration loop skips craters at-or-above this
    /// threshold to avoid double-counting their displacement.
    pub cubemap_bake_threshold_m: f32,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

impl Default for PlanetDetailParams {
    fn default() -> Self {
        Self {
            body_radius_m: 1.0,
            d_min_m: 0.0,
            d_max_m: 0.0,
            sfd_alpha: 2.0,
            global_k_per_km2: 0.0,
            d_sc_m: 1.0,
            body_age_gyr: 4.5,
            cubemap_bake_threshold_m: f32::INFINITY,
            seed_lo: 0,
            seed_hi: 0,
        }
    }
}

impl PlanetDetailParams {
    /// Build from terrain_gen's DetailNoiseParams plus the Cratering stage's
    /// cubemap bake threshold. Both come from `BodyData` — pass both.
    pub fn from_body(
        detail: &thalos_terrain_gen::DetailNoiseParams,
        cubemap_bake_threshold_m: f32,
    ) -> Self {
        Self {
            body_radius_m: detail.body_radius_m,
            d_min_m: detail.d_min_m,
            d_max_m: detail.d_max_m,
            sfd_alpha: detail.sfd_alpha,
            global_k_per_km2: detail.global_k_per_km2,
            d_sc_m: detail.d_sc_m,
            body_age_gyr: detail.body_age_gyr,
            cubemap_bake_threshold_m,
            seed_lo: detail.seed as u32,
            seed_hi: (detail.seed >> 32) as u32,
        }
    }
}

/// Atmosphere uniform consumed by the impostor's atmosphere pass.
///
/// Mirrors `AtmosphereBlock` in `shaders/atmosphere.wgsl`. Every field
/// corresponds to one optional layer of the terrestrial atmosphere
/// schema; the shader skips a layer when its intensity/strength scalar
/// is zero, so `AtmosphereBlock::default()` produces an impostor that
/// renders identically to one with no atmosphere at all.
#[derive(Clone, Copy, ShaderType)]
pub struct AtmosphereBlock {
    /// xyz = rim halo colour (linear RGB), w = intensity scalar.
    pub rim_color_intensity: Vec4,
    /// x = scale height (render units), y = outer-shell altitude
    /// (render units), z/w reserved for future extensions.
    pub rim_shape: Vec4,
    /// xyz = terminator warmth tint, w = strength.
    pub terminator_warmth: Vec4,
    /// xyz = Fresnel rim colour, w = strength.
    pub fresnel_rim: Vec4,
    /// xyz = per-channel Minnaert exponents (R, G, B), w = strength.
    pub limb_exponents: Vec4,
    /// xyz = sunlit-cloud albedo, w = coverage fraction in [0, 1].
    pub cloud_albedo_coverage: Vec4,
    /// x = fBm frequency, y = boundary softness, z = octave count (f32),
    /// w = differential-rotation coefficient.
    pub cloud_shape: Vec4,
    /// x = equatorial scroll rate (rad/s), y = sim time seconds
    /// (wrapped; written per frame by `update_planet_light_dirs`),
    /// z = seed lo bits (bitcast u32→f32),
    /// w = seed hi bits.
    pub cloud_dynamics: Vec4,
    /// Per-wavelength Rayleigh scattering. xyz = vertical optical depth
    /// at zenith (R, G, B); w = overall strength multiplier. w = 0
    /// disables Rayleigh entirely — `apply_rayleigh_*` early-out and
    /// the impostor renders with unattenuated white sunlight.
    pub rayleigh: Vec4,
    /// Cloud main-deck band phases 0..=3. 16 total phases packed into
    /// four `Vec4`s carry the per-latitude-strip rotation state for
    /// the banded cloud decomposition. See
    /// `CLOUD_BAND_COUNT` / `CloudBandState` on the CPU side and
    /// `sample_cloud_banded` in `planet_impostor.wgsl` for usage.
    pub cloud_bands_a: Vec4,
    /// Cloud main-deck band phases 4..=7.
    pub cloud_bands_b: Vec4,
    /// Cloud main-deck band phases 8..=11.
    pub cloud_bands_c: Vec4,
    /// Cloud main-deck band phases 12..=15.
    pub cloud_bands_d: Vec4,
}

impl Default for AtmosphereBlock {
    fn default() -> Self {
        Self {
            rim_color_intensity: Vec4::ZERO,
            rim_shape: Vec4::ZERO,
            terminator_warmth: Vec4::ZERO,
            fresnel_rim: Vec4::ZERO,
            limb_exponents: Vec4::ZERO,
            cloud_albedo_coverage: Vec4::ZERO,
            cloud_shape: Vec4::ZERO,
            cloud_dynamics: Vec4::ZERO,
            rayleigh: Vec4::ZERO,
            cloud_bands_a: Vec4::ZERO,
            cloud_bands_b: Vec4::ZERO,
            cloud_bands_c: Vec4::ZERO,
            cloud_bands_d: Vec4::ZERO,
        }
    }
}

/// Number of latitudinal cloud rotation bands. Each band has its own
/// rigid rotation speed `ω_i = scroll_rate × (1 − diff × sin²(lat_i))`
/// where `sin²(lat_i) = i / (CLOUD_BAND_COUNT − 1)`. Per-band phases are
/// accumulated on the CPU (see `CloudBandState` in the game crate), mod
/// `TAU` in f64, uploaded as four `Vec4`s into `AtmosphereBlock`, and
/// consumed by `sample_cloud_banded` in `planet_impostor.wgsl` — which
/// samples the cloud cube at the two bands bracketing a fragment's
/// latitude and blends. Because each per-band phase wraps independently
/// mod TAU, there is no latitude at which rotation seams — rotation is
/// seamless forever. State persists trivially as 16 × f64 per body.
pub const CLOUD_BAND_COUNT: usize = 16;

impl AtmosphereBlock {
    /// Build from a `TerrestrialAtmosphere` and the body's
    /// meters-per-render-unit ratio. Any layer not present in the
    /// source struct is left at zero, which the shader interprets as
    /// "skip this layer entirely." The `cloud_dynamics.y` (sim time)
    /// field is left at zero here; `update_planet_light_dirs` writes
    /// the current sim time every frame.
    pub fn from_terrestrial(atmos: &TerrestrialAtmosphere, meters_per_render_unit: f32) -> Self {
        let mut out = Self::default();
        if let Some(rim) = &atmos.rim_halo {
            out.rim_color_intensity =
                Vec4::new(rim.color[0], rim.color[1], rim.color[2], rim.intensity);
            let inv_m = 1.0 / meters_per_render_unit.max(1.0);
            out.rim_shape = Vec4::new(
                rim.scale_height_m * inv_m,
                rim.outer_altitude_m * inv_m,
                0.0,
                0.0,
            );
        }
        if let Some(limb) = &atmos.limb {
            out.terminator_warmth = Vec4::new(
                limb.terminator_warmth[0],
                limb.terminator_warmth[1],
                limb.terminator_warmth[2],
                limb.terminator_strength,
            );
            out.fresnel_rim = Vec4::new(
                limb.fresnel_color[0],
                limb.fresnel_color[1],
                limb.fresnel_color[2],
                limb.fresnel_strength,
            );
        }
        if let Some(ld) = &atmos.limb_darkening {
            out.limb_exponents = Vec4::new(
                ld.red.max(0.0),
                ld.green.max(0.0),
                ld.blue.max(0.0),
                ld.strength.clamp(0.0, 1.0),
            );
        }
        if let Some(ray) = &atmos.rayleigh {
            out.rayleigh = Vec4::new(
                ray.vertical_optical_depth[0].max(0.0),
                ray.vertical_optical_depth[1].max(0.0),
                ray.vertical_optical_depth[2].max(0.0),
                ray.strength.max(0.0),
            );
        }
        if let Some(clouds) = &atmos.clouds {
            out.cloud_albedo_coverage = Vec4::new(
                clouds.albedo[0],
                clouds.albedo[1],
                clouds.albedo[2],
                clouds.coverage.clamp(0.0, 1.0),
            );
            out.cloud_shape = Vec4::new(
                clouds.frequency.max(0.0),
                clouds.softness.max(0.0),
                0.0,
                clouds.differential_rotation,
            );
            out.cloud_dynamics = Vec4::new(
                clouds.scroll_rate,
                0.0,
                f32::from_bits(clouds.seed as u32),
                f32::from_bits((clouds.seed >> 32) as u32),
            );
        }
        out
    }
}

// Bind group layout (group 3, planet material). `PlanetMaterial` and
// `PlanetHaloMaterial` intentionally share this exact layout; only the
// pipeline shader def / depth-write state differs. This is the contract both
// the shader and `bake_from_body_data` must match.
//
// | Binding | Kind             | WGSL type                 | Source             |
// |---------|------------------|---------------------------|--------------------|
// | 0       | uniform          | PlanetParams              | `params` field     |
// | 1       | texture cube     | texture_cube<f32>         | `albedo` cube      |
// | 2       | sampler          | sampler                   | `albedo` sampler   |
// | 3       | texture cube     | texture_cube<f32>         | `height` cube      |
// | 4       | sampler          | sampler                   | `height` sampler   |
// | 5       | uniform          | PlanetDetailParams        | `detail` field     |
// | 6       | texture cube     | texture_cube<f32>         | `roughness` cube   |
// | 7       | sampler          | sampler                   | `roughness` sampler|
// | 8       | storage (read)   | array<Crater>             | `craters_buffer`   |
// | 9       | storage (read)   | array<CellRange>          | `cell_index_buf`   |
// | 10      | storage (read)   | array<u32>                | `feature_ids_buf`  |
// | 12      | uniform          | AtmosphereBlock           | `atmosphere` field |
// | 13      | texture cube     | texture_cube<f32>         | `cloud_cover` cube |
// | 14      | sampler          | sampler                   | `cloud_cover` sampler |
//
// Storage buffers (8-10) use std430 layout. Struct definitions for
// `Crater`, `CellRange` are mirrored in the shader and must stay in sync
// with `shader_types.rs`.
//
// A `CellRange` is `{ start: u32, count: u32 }`. For each ico cell the
// shader looks up its range, then reads `count` crater indices from
// `feature_ids_buf` starting at `start`, each of which is an index into
// `craters_buffer`.
//
// Note: surface normals are reconstructed per-fragment in the shader via
// finite-differencing the filterable height cube
// (`perturb_normal_from_height` in `planet_impostor.wgsl`). 8-bit object-
// space normal encoding crushed the shallow slope gradients that drive
// terminator depth and crater rim transitions, so the baked
// `normal_cubemap` in `BodyData` is reserved for future ground LOD
// consumers where chunked geometry can't cheaply finite-difference at
// runtime.
//
// Binding 12 is the per-body atmosphere uniform. Zero-initialised means
// "no atmosphere" — the shader gates every layer on its own intensity
// scalar, so bodies without a `terrestrial_atmosphere` block (Mira,
// Ignis, the airless moons) cost only a handful of scalar comparisons.
//
// Bindings 13–14 carry the baked cloud-cover cubemap (R8Unorm; each
// texel's value divided by 255 is the raw Worley-fBm density at the
// curl-warp-advected direction). Bodies without a cloud layer bind a
// 1×1 blank cube; the shader gates its cloud path on
// `AtmosphereBlock::cloud_albedo_coverage.w > 0` so airless bodies pay
// just one texture fetch + a branch.

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct PlanetMaterial {
    #[uniform(0)]
    pub params: PlanetParams,
    #[texture(1, dimension = "cube")]
    #[sampler(2)]
    pub albedo: Handle<Image>,
    #[texture(3, dimension = "cube")]
    #[sampler(4)]
    pub height: Handle<Image>,
    #[uniform(5)]
    pub detail: PlanetDetailParams,
    #[texture(6, dimension = "cube")]
    #[sampler(7)]
    pub roughness: Handle<Image>,
    #[storage(8, read_only)]
    pub craters: Handle<ShaderStorageBuffer>,
    #[storage(9, read_only)]
    pub cell_index: Handle<ShaderStorageBuffer>,
    #[storage(10, read_only)]
    pub feature_ids: Handle<ShaderStorageBuffer>,
    #[uniform(12)]
    pub atmosphere: AtmosphereBlock,
    // Cloud-cover cubemap (R8Unorm). Produced by
    // `bake_cloud_cover_image`, or a 1×1 black fallback via
    // `blank_cloud_cover_image` for bodies with no clouds.
    #[texture(13, dimension = "cube")]
    #[sampler(14)]
    pub cloud_cover: Handle<Image>,
}

impl Material for PlanetMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/planet_impostor.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/planet_impostor.wgsl".into()
    }

    // Body pass only: the shader discards all miss/rim-halo fragments
    // when `HALO_PASS` is absent. Surface hits output `alpha = 1`, so
    // the material belongs in the opaque pass and should write depth
    // like any other solid body.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        // Kept explicit because this material used to be a premultiplied
        // body+halo pass. The body pass must always populate depth so
        // later transparent items (stars, galaxies, rings, halo pass)
        // test correctly against the planet surface.
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = true;
        }
        Ok(())
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct PlanetHaloMaterial {
    #[uniform(0)]
    pub params: PlanetParams,
    #[texture(1, dimension = "cube")]
    #[sampler(2)]
    pub albedo: Handle<Image>,
    #[texture(3, dimension = "cube")]
    #[sampler(4)]
    pub height: Handle<Image>,
    #[uniform(5)]
    pub detail: PlanetDetailParams,
    #[texture(6, dimension = "cube")]
    #[sampler(7)]
    pub roughness: Handle<Image>,
    #[storage(8, read_only)]
    pub craters: Handle<ShaderStorageBuffer>,
    #[storage(9, read_only)]
    pub cell_index: Handle<ShaderStorageBuffer>,
    #[storage(10, read_only)]
    pub feature_ids: Handle<ShaderStorageBuffer>,
    #[uniform(12)]
    pub atmosphere: AtmosphereBlock,
    #[texture(13, dimension = "cube")]
    #[sampler(14)]
    pub cloud_cover: Handle<Image>,
}

impl From<&PlanetMaterial> for PlanetHaloMaterial {
    fn from(material: &PlanetMaterial) -> Self {
        Self {
            params: material.params.clone(),
            albedo: material.albedo.clone(),
            height: material.height.clone(),
            detail: material.detail.clone(),
            roughness: material.roughness.clone(),
            craters: material.craters.clone(),
            cell_index: material.cell_index.clone(),
            feature_ids: material.feature_ids.clone(),
            atmosphere: material.atmosphere,
            cloud_cover: material.cloud_cover.clone(),
        }
    }
}

impl Material for PlanetHaloMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/planet_impostor.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/planet_impostor.wgsl".into()
    }

    // The halo shader returns premultiplied atmospheric in-scatter over
    // whatever passed the depth test behind the rim.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.push("HALO_PASS".into());
        }
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            // The rim must depth-test against opaque foreground objects,
            // but it must not write depth: stars and galaxies draw at the
            // reverse-Z far plane and should remain visible behind the halo.
            depth.depth_write_enabled = false;
        }
        Ok(())
    }
}

/// Stored on the parent `CelestialBody` entity so the per-frame update system
/// can find and mutate the material without traversing children.
#[derive(Component)]
pub struct PlanetMaterialHandle(pub Handle<PlanetMaterial>);

#[derive(Component)]
pub struct PlanetHaloMaterialHandle(pub Handle<PlanetHaloMaterial>);
