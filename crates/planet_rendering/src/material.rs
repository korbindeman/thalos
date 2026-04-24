use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::shader::ShaderRef;
use thalos_atmosphere_gen::TerrestrialAtmosphere;

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
    pub fn from_terrestrial(
        atmos: &TerrestrialAtmosphere,
        meters_per_render_unit: f32,
    ) -> Self {
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

// Bind group layout (group 3, planet material). This is the contract both
// the shader and `bake_from_body_data` must match. Phase-2 agents E + F
// both consume it.
//
// | Binding | Kind             | WGSL type                 | Source             |
// |---------|------------------|---------------------------|--------------------|
// | 0       | uniform          | PlanetParams              | `params` field     |
// | 1       | texture cube     | texture_cube<f32>         | `albedo` cube      |
// | 2       | sampler          | sampler                   | `albedo` sampler   |
// | 3       | texture cube     | texture_cube<f32>         | `height` cube      |
// | 4       | sampler          | sampler                   | `height` sampler   |
// | 5       | uniform          | PlanetDetailParams        | `detail` field     |
// | 6       | texture 2d array | texture_2d_array<u32>     | `material_cubemap` |
// | 7       | sampler (non-flt)| sampler (unused for load) | `material` sampler |
//
// Binding 6 is a 2D array with 6 layers, NOT a cube. WGSL has no
// `textureLoad` overload for `texture_cube<u32>`, so the shader reads this
// as `texture_2d_array<u32>` and does its own direction → (face, x, y)
// lookup. Face order matches `CubemapFace::ALL`.
// | 8       | storage (read)   | array<Crater>             | `craters_buffer`   |
// | 9       | storage (read)   | array<CellRange>          | `cell_index_buf`   |
// | 10      | storage (read)   | array<u32>                | `feature_ids_buf`  |
// | 11      | storage (read)   | array<Material>           | `materials_buffer` |
//
// Storage buffers (8-11) use std430 layout. Struct definitions for
// `Crater`, `CellRange`, `Material` are mirrored in the shader and must
// stay in sync with `shader_types.rs` (produced by Phase 2F).
//
// A `CellRange` is `{ start: u32, count: u32 }`. For each ico cell the
// shader looks up its range, then reads `count` crater indices from
// `feature_ids_buf` starting at `start`, each of which is an index into
// `craters_buffer`.
// | 12      | uniform          | AtmosphereBlock           | `atmosphere` field |
// | 13      | texture cube     | texture_cube<f32>         | `cloud_cover` cube |
// | 14      | sampler          | sampler                   | `cloud_cover` sampler |
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
    // ------- Phase 2F: material + feature SSBOs ---------------------------
    #[texture(6, dimension = "2d_array", sample_type = "u_int")]
    #[sampler(7, sampler_type = "non_filtering")]
    pub material_cube: Handle<Image>,
    #[storage(8, read_only)]
    pub craters: Handle<ShaderStorageBuffer>,
    #[storage(9, read_only)]
    pub cell_index: Handle<ShaderStorageBuffer>,
    #[storage(10, read_only)]
    pub feature_ids: Handle<ShaderStorageBuffer>,
    #[storage(11, read_only)]
    pub materials: Handle<ShaderStorageBuffer>,
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

    // Premultiplied because the fragment shader's miss path outputs
    // `(rim.contribution, rim.opacity)` where `contribution = color *
    // strength` and `opacity = strength`, i.e. the RGB is already
    // premultiplied by alpha. With Opaque, the small dim premultiplied
    // RGB overwrites whatever is behind (e.g. a gas giant) with
    // near-black, producing a visible dark band at the shell. Premult
    // compositing gives `src.rgb + dst * (1 - src.a)` — physically
    // correct in-scatter over the background. Hit-path fragments output
    // `(lit, 1.0)` which resolves to just `lit`, unchanged.
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
        // Bevy disables depth-write for Premultiplied by default (a
        // transparent-phase standard). Re-enable it: the hit path outputs
        // `α=1` and must occlude anything drawn later in the transparent
        // phase (stars/galaxies are `AlphaMode::Add`, depth-test but don't
        // write), otherwise the celestial backdrop leaks through the
        // planet's silhouette. The halo miss-path pixels also write their
        // closest-approach depth — a thin ring-shaped region where stars
        // directly behind the shell get depth-culled instead of shining
        // through. Acceptable because the shell is narrow (~5 % of radius)
        // and star-through-atmosphere is a subtle cue anyway; revisit with
        // a split halo-entity pass if it becomes visually objectionable.
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = true;
        }
        Ok(())
    }
}

/// Stored on the parent `CelestialBody` entity so the per-frame update system
/// can find and mutate the material without traversing children.
#[derive(Component)]
pub struct PlanetMaterialHandle(pub Handle<PlanetMaterial>);
