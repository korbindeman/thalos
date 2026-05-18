use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::shader::ShaderRef;
use thalos_planet_lighting::SceneLighting;
use thalos_terrain::StaticSurfaceData;

/// Re-export so existing call sites resolve unchanged.
pub use thalos_planet_lighting::{AtmosphereBlock, CLOUD_BAND_COUNT, MAX_ECLIPSE_OCCLUDERS};

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
    /// Quaternion (xyzw) mapping body-local directions into the active-dune
    /// overlay texture. Slow dune migration can update this field without
    /// rebuilding static terrain or running per-fragment dune synthesis.
    pub active_dune_texture_from_body: Vec4,
    /// Stars, eclipse occluders, ambient, and planetshine parent. See
    /// `thalos_planet_lighting::SceneLighting`.
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
    pub fn from_static_surface(body: &StaticSurfaceData) -> Self {
        if let Some(water) = body.water_appearance {
            Self {
                color_depth: Vec4::new(
                    water.color_depth[0],
                    water.color_depth[1],
                    water.color_depth[2],
                    water.color_depth[3],
                ),
            }
        } else if body.sea_level_m.is_some() {
            Self {
                color_depth: Vec4::new(0.012, 0.040, 0.090, 120.0),
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
    pub fn from_static_surface(body: &StaticSurfaceData) -> Self {
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
                // Coastline shape is baked into the terrain cubemap. Keep the
                // runtime material from adding high-frequency shoreline fuzz
                // that makes editor/runtime diverge from the bake.
                warp_amp_radians: 0.0,
                jitter_amp_m: 0.0,
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

impl Default for PlanetParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            height_range: 1.0,
            terminator_wrap: 0.0,
            fullbright: 0.0,
            orientation: Vec4::new(0.0, 0.0, 0.0, 1.0),
            active_dune_texture_from_body: Vec4::new(0.0, 0.0, 0.0, 1.0),
            scene: SceneLighting::default(),
            // Large negative sentinel — airless bodies leave this at the
            // default, and the shader's `sample_height_m(dir) < sea_level_m`
            // test never fires.
            sea_level_m: -1.0e9,
            water_color_depth: Vec4::new(0.012, 0.040, 0.090, 120.0),
            // Defaults are 0 so airless / preview bodies pay zero cost.
            // Coastline shape belongs in the bake; shader-side perturbation
            // is reserved for explicit experiments.
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
    /// Build from terrain's DetailNoiseParams plus the Cratering stage's
    /// cubemap bake threshold. Both come from `StaticSurfaceData` — pass both.
    pub fn from_body(
        detail: &thalos_terrain::DetailNoiseParams,
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

// `AtmosphereBlock` and `CLOUD_BAND_COUNT` re-exported above; the canonical
// definitions live in `thalos_planet_lighting`.

// Bind group layout (group 3, planet material). `PlanetMaterial` and
// `PlanetHaloMaterial` intentionally share this exact layout; only the
// pipeline shader def / depth-write state differs. This is the contract both
// the shader and `bake_from_planet_surface` must match.
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
// | 11      | storage (read)   | array<RadialFeature>      | `radial_features`  |
// | 12      | uniform          | AtmosphereBlock           | `atmosphere` field |
// | 13      | texture cube     | texture_cube<f32>         | `cloud_cover` cube |
// | 14      | sampler          | sampler                   | `cloud_cover` sampler |
// | 15      | storage (read)   | array<IceCap>             | `ice_caps`         |
// | 16      | storage (read)   | array<DuneSea>            | `active_dunes`     |
// | 17      | texture cube     | texture_cube<f32>         | `active_dune_height` |
// | 18      | sampler          | sampler                   | `active_dune_height_sampler` |
// | 19      | texture cube     | texture_cube<f32>         | `active_dune_albedo` |
// | 20      | sampler          | sampler                   | `active_dune_albedo_sampler` |
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
// `normal_cubemap` in `StaticSurfaceData` is reserved for future ground LOD
// consumers where chunked geometry can't cheaply finite-difference at
// runtime.
//
// Binding 12 is the per-body atmosphere uniform. Zero-initialised means
// "no atmosphere" — the shader gates every layer on its own intensity
// scalar, so bodies without a `terrestrial_atmosphere` block (Mira,
// Ignis, the airless moons) cost only a handful of scalar comparisons.
//
// Bindings 13–14 carry the reference cloud-cover cubemap (R8Unorm).
// Bodies without a reference overlay bind a 1×1 blank cube; the shader
// gates its cloud path on `AtmosphereBlock::cloud_albedo_coverage.w > 0`
// so airless bodies pay just one texture fetch + a branch.
//
// Bindings 15-16 carry dynamic surface overlays that are not part of the
// static terrain bake. Runtime climate/wind systems can update these buffers
// without rebuilding terrain cubemaps.

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
    #[storage(11, read_only)]
    pub radial_features: Handle<ShaderStorageBuffer>,
    #[uniform(12)]
    pub atmosphere: AtmosphereBlock,
    // Cloud-cover cubemap (R8Unorm). Produced by
    // reference equirectangular overlays, or a 1×1 black fallback via
    // `blank_cloud_cover_image` for bodies without a reference overlay.
    #[texture(13, dimension = "cube")]
    #[sampler(14)]
    pub cloud_cover: Handle<Image>,
    #[storage(15, read_only)]
    pub ice_caps: Handle<ShaderStorageBuffer>,
    #[storage(16, read_only)]
    pub active_dunes: Handle<ShaderStorageBuffer>,
    #[texture(17, dimension = "cube")]
    #[sampler(18)]
    pub active_dune_height: Handle<Image>,
    #[texture(19, dimension = "cube")]
    #[sampler(20)]
    pub active_dune_albedo: Handle<Image>,
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
    #[storage(11, read_only)]
    pub radial_features: Handle<ShaderStorageBuffer>,
    #[uniform(12)]
    pub atmosphere: AtmosphereBlock,
    #[texture(13, dimension = "cube")]
    #[sampler(14)]
    pub cloud_cover: Handle<Image>,
    #[storage(15, read_only)]
    pub ice_caps: Handle<ShaderStorageBuffer>,
    #[storage(16, read_only)]
    pub active_dunes: Handle<ShaderStorageBuffer>,
    #[texture(17, dimension = "cube")]
    #[sampler(18)]
    pub active_dune_height: Handle<Image>,
    #[texture(19, dimension = "cube")]
    #[sampler(20)]
    pub active_dune_albedo: Handle<Image>,
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
            radial_features: material.radial_features.clone(),
            atmosphere: material.atmosphere,
            cloud_cover: material.cloud_cover.clone(),
            ice_caps: material.ice_caps.clone(),
            active_dunes: material.active_dunes.clone(),
            active_dune_height: material.active_dune_height.clone(),
            active_dune_albedo: material.active_dune_albedo.clone(),
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
