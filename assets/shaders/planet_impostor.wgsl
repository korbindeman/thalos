// Planet sphere impostor.
//
// Renders a camera-facing quad (billboard) whose fragment shader ray-traces a
// sphere of the correct radius.  Every pixel gets the mathematically exact
// surface normal, giving a perfectly smooth silhouette at any resolution.
//
// Surface detail comes from two layered sources (the sample.rs LOD contract):
//
//   1. Cubemap textures from `thalos_terrain_gen` — albedo (sRGB RGBA8),
//      height (R16Unorm displacement), and material ID (R8Uint palette index).
//      These hold the low-frequency baked features: primordial topography,
//      basins, mare flooding, and the regional material palette. One fetch
//      per sample; covers features ≥ 5 km.
//
//   2. Feature SSBO (500 m – 5 km) — real craters iterated per fragment via a
//      3D cell-hash spatial index. Each fragment looks up the cell it lives in
//      and its 3×3×3 neighborhood, reads the per-cell (start, count) range
//      from `cell_index`, and evaluates every listed crater's profile. Each
//      crater's contribution is faded in by a screen-space smoothstep so it
//      never pops during zoom.
//
//   3. Dynamic surface overlays — seasonal ice caps and active dune texture
//      layers are rendered on top of the baked terrain without invalidating
//      the static terrain cache.
//
// Sub-500 m craters are intentionally not rendered in the impostor — the
// statistical shader-hash layer was dropped.
//
// Lighting: diffuse Lambertian + tiny ambient + terminator wrap + opposition
// surge on the lit side.
//
// ────────────────────────────────────────────────────────────────────────────
// SSBO SPATIAL-INDEX CONTRACT (must agree with Agent F's baker)
//
//   cell_size_unit:   read from `detail.ssbo_cell_size` uniform. Target 0.06
//                     (unit-sphere coords ⇒ ~52 km on Mira).
//   CELL_TABLE_SIZE:  8192 (power of two, ~1.6× over-provision vs. ~5000
//                     populated cells at cell_size 0.06 on Mira).
//   hash function:    `hash_cell(ix, iy, iz, octave=0u)` — masked with
//                     `& (CELL_TABLE_SIZE - 1u)` to index the dense table.
//   neighborhood:     3×3×3 = 27 cells centered on the fragment's cell. Worley
//                     pattern, correctness-first.
//
// Struct layouts — std430 storage buffers. Keep in sync with
// `crates/planet_rendering/src/material.rs` (bind group comment block) and
// whatever `shader_types.rs` Agent F produces.
// ────────────────────────────────────────────────────────────────────────────

#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_functions::get_world_from_local
#import thalos::lighting::{SceneLighting, StarLight, PlanetShineSample, eclipse_factor, planetshine_sample}
#import thalos::noise::fbm3
#import bevy_erosion_filter::erosion::{
    ErosionFilterParams,
    erosion_filter,
}
#import thalos::atmosphere::{
    AtmosphereBlock,
    ScatterResult,
    integrate_atmosphere,
    atmosphere_jitter,
    atmosphere_scattering_active,
    apply_limb_darkening,
    composite_clouds,
    cloud_band_phase,
    rotate_around_y,
    CLOUD_BAND_COUNT,
}

const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

const CELL_TABLE_SIZE: u32 = 8192u;
const CELL_TABLE_MASK: u32 = 8191u;

// Scene-flux normalisation. Hapke's BRDF returns a radiance factor; the
// prior pipeline used a Lambert `/PI` normalisation that we fold into
// this single scalar so existing flux values don't need re-tuning. The
// atmosphere raymarch consumes the same scaled flux so haze radiance
// stays in unit consistency with the lit surface — without the scale,
// in-scatter reads ~2× too bright relative to the ground.
const SCENE_FLUX_SCALE: f32 = 0.5;

// Cell size for the SSBO spatial index, in unit-sphere coordinates.
// ~0.06 on the unit sphere ≈ 52 km per cell on a 869 km Mira — chosen so
// ~18 features land in each cell on average for a ~90k-crater population.
//
// MUST match Agent F's CPU-side baker. If runtime tuning becomes necessary,
// promote this to a new `ssbo_cell_size: f32` field on `PlanetDetailParams`
// and read from the uniform; the uniform layout is the handoff point.
const SSBO_CELL_SIZE_UNIT: f32 = 0.06;

const FRESH_AGE_GYR: f32 = 0.1;

// ── Material uniforms (binding group 3) ─────────────────────────────────────

struct PlanetParams {
    radius:          f32,
    height_range:    f32,
    // Terminator wrap factor (0 = razor-sharp Lambert, >0 = softened edge
    // for atmospheric/rough bodies). Replaces the old `light_dir.w` slot.
    terminator_wrap: f32,
    // Debug fullbright toggle (0.0 = off, >= 0.5 = on). When on, the direct
    // sun term collapses to a constant so albedo reads uniformly; atmosphere,
    // Rayleigh, and clouds still composite normally.
    fullbright:      f32,
    // Quaternion (xyzw) rotating world-space directions into body-local space
    // where the cubemaps were baked. Identity = no rotation.
    orientation:     vec4<f32>,
    // Quaternion mapping body-local directions into the active-dune texture
    // layer. Dune evolution can update this one uniform instead of running
    // per-fragment procedural dune synthesis.
    active_dune_texture_from_body: vec4<f32>,
    // Shared scene-lighting description: stars, ambient, eclipse occluders,
    // planetshine parent. Mirror of `thalos::lighting::SceneLighting`.
    scene:           SceneLighting,
    // Sea-level elevation (m, same encoding as the height cubemap). The
    // water BRDF fires where `sample_height_m(dir) < sea_level_m`. Airless
    // bodies set this to a large negative sentinel so the threshold is
    // never crossed.
    sea_level_m:     f32,
    // xyz = linear-RGB apparent deep-water colour. w = minimum optical depth
    // used by water shading, in meters. The minimum matters for flat ocean
    // placeholders whose mask depth is only 1 m by convention.
    water_color_depth: vec4<f32>,
    // Canonical high-frequency terrain bands. The impostor applies a
    // domain warp (perturbs the cubemap sample direction) AND adds a
    // height-jitter fbm — both defined in
    // `crates/planet_rendering/src/shaders/noise.wgsl`, mirrored
    // bit-exact by `crates/terrain_gen/src/noise.rs`.
    //
    // The warp is what breaks the cubemap-texel staircase visible
    // from orbit: a few-texel arc displacement on the sphere shifts
    // the iso-contour out of grid alignment without adding height
    // roughness. The height-jitter adds sub-texel detail visible
    // up close. Both feed `sample_height_m`, so water mask,
    // surface normals, and self-shadow all see the same canonical
    // perturbed surface.
    //
    // Future 3D terrain meshing must evaluate the same fbm with the
    // same parameters at vertex/sample time so the LOD handoff is
    // continuous.
    coastline_warp_amp_radians:  f32,
    coastline_warp_freq_per_m:   f32,
    coastline_jitter_amp_m:      f32,
    coastline_jitter_freq_per_m: f32,
    coastline_octaves:           u32,
    coastline_seed:              u32,
}

// Layout matches `PlanetDetailParams` in `crates/planet_rendering/src/material.rs`.
// Kept identical across this edit to avoid disturbing the uniform buffer
// contract — the cell-size value is a WGSL const for now and can be
// promoted to a uniform field by a later reconciliation pass.
struct PlanetDetail {
    body_radius_m:             f32,
    d_min_m:                   f32,
    d_max_m:                   f32,
    sfd_alpha:                 f32,
    global_k_per_km2:          f32,
    d_sc_m:                    f32,
    body_age_gyr:              f32,
    // Craters ≥ this radius were rasterized into the height cubemap by
    // the Cratering stage. SSBO iteration skips them to avoid double-
    // counting the displacement.
    cubemap_bake_threshold_m:  f32,
    seed_lo:                   u32,
    seed_hi:                   u32,
}

// ── SSBO struct layouts (std430, agreed with Agent F) ──────────────────────

// Crater: one explicit feature in the 500 m – 5 km band.
// Layout mirrored from `crates/planet_rendering/src/shader_types.rs::GpuCrater`.
// std430, 32 bytes total. Do not reorder without updating the Rust side.
//
//   center:        unit-sphere direction to the crater center.
//   radius_m:      real crater radius in meters (diameter_m/2).
//   depth_m:       measured depth in meters (the baker already accounts for
//                  simple vs complex morphology; the shader uses it directly
//                  rather than re-deriving from d/d_sc).
//   rim_height_m:  rim uplift height in meters.
//   age_gyr:       formation age for maturity shading.
//   material_id:   reserved for future SSBO crater material overrides. The
//                  shader currently does not branch on it — the rim albedo
//                  delta is expressed via `crater_albedo_delta` against the
//                  primary diffuse cube.
struct Crater {
    center:       vec3<f32>,
    radius_m:     f32,
    depth_m:      f32,
    rim_height_m: f32,
    age_gyr:      f32,
    material_id:  u32,
}

struct CellRange {
    start: u32,
    count: u32,
}

// RadialFeature: feature-local shader detail for broad radial landmarks such
// as shield volcanoes. The cubemaps already carry the macro relief; this
// buffer adds sub-cubemap erosion color, roughness, and normal detail.
struct RadialFeature {
    center:          vec3<f32>,
    radius_m:        f32,
    east:            vec3<f32>,
    height_m:        f32,
    north:           vec3<f32>,
    erosion_scale_m: f32,
    seed:            u32,
    material_id:     u32,
    _pad0:           u32,
    _pad1:           u32,
}

// IceCap: dynamic surface layer. Displacement contributes to canonical
// sampled height where needed, but not to impostor finite-difference normals,
// self-shadow, or disk silhouette.
struct IceCap {
    axis:                 vec3<f32>,
    flags:                u32,
    albedo_linear:        vec3<f32>,
    edge_latitude_deg:    f32,
    dust_albedo_linear:   vec3<f32>,
    solid_latitude_deg:   f32,
    edge_noise_deg:       f32,
    edge_sharpness:       f32,
    noise_frequency:      f32,
    max_thickness_m:      f32,
    albedo_strength:      f32,
    roughness:            f32,
    roughness_strength:   f32,
    obliquity_response:   f32,
    coverage_scale:       f32,
    edge_offset_deg:      f32,
    thickness_scale:      f32,
    dustiness:            f32,
    seed:                 u32,
    _pad0:                u32,
    _pad1:                u32,
    _pad2:                u32,
}

struct DuneSea {
    center:            vec3<f32>,
    radius_rad:        f32,
    axis_tangent:      vec3<f32>,
    feather_rad:       f32,
    albedo_crest_lin:  vec3<f32>,
    crest_strength:    f32,
    lambda_draa_m:     f32,
    amplitude_draa_m:  f32,
    lambda_dune_m:     f32,
    amplitude_dune_m:  f32,
    alpha_skew:        f32,
    warp_amp_unit:     f32,
    warp_freq:         f32,
    coverage_scale:    f32,
    phase_offset_m:    f32,
    amplitude_scale:   f32,
    mobility:          f32,
    seed:              u32,
}

@group(3) @binding(0)  var<uniform> params:          PlanetParams;
@group(3) @binding(1)  var          albedo_tex:      texture_cube<f32>;
@group(3) @binding(2)  var          albedo_sampler:  sampler;
@group(3) @binding(3)  var          height_tex:      texture_cube<f32>;
@group(3) @binding(4)  var          height_sampler:  sampler;
@group(3) @binding(5)  var<uniform> detail:          PlanetDetail;
@group(3) @binding(6)  var          roughness_tex:   texture_cube<f32>;
@group(3) @binding(7)  var          roughness_sampler: sampler;
@group(3) @binding(8)  var<storage, read> craters:     array<Crater>;
@group(3) @binding(9)  var<storage, read> cell_index:  array<CellRange>;
@group(3) @binding(10) var<storage, read> feature_ids: array<u32>;
@group(3) @binding(11) var<storage, read> radial_features: array<RadialFeature>;
// Optional atmosphere layer — see `thalos::atmosphere`. For bodies with
// no atmosphere (Mira, Ignis, …) every layer's intensity scalar is zero
// and the atmosphere path is effectively skipped.
@group(3) @binding(12) var<uniform> atmosphere:      AtmosphereBlock;
// Reference cloud-cover cubemap (R8Unorm). For bodies without a reference
// overlay this is a 1×1 blank cube; the cloud path gates on
// `cloud_albedo_coverage.w > 0` so those bodies pay just the branch cost.
@group(3) @binding(13) var          cloud_cover_tex: texture_cube<f32>;
@group(3) @binding(14) var          cloud_cover_sampler: sampler;
@group(3) @binding(15) var<storage, read> ice_caps:    array<IceCap>;
@group(3) @binding(16) var<storage, read> active_dunes: array<DuneSea>;
@group(3) @binding(17) var          active_dune_height_tex: texture_cube<f32>;
@group(3) @binding(18) var          active_dune_height_sampler: sampler;
@group(3) @binding(19) var          active_dune_albedo_tex: texture_cube<f32>;
@group(3) @binding(20) var          active_dune_albedo_sampler: sampler;

// ── Vertex stage ─────────────────────────────────────────────────────────────

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) sphere_center: vec3<f32>,
    // Pixel footprint on the sphere surface, in meters. Computed once per
    // vertex (same for all three, so `flat` avoids interpolator waste) from
    // the projection matrix + viewport, dodging the `dpdx(hit)` silhouette
    // flicker caused by a 2×2 quad straddling a discarded fragment.
    @location(2) @interpolate(flat) pixel_size_m: f32,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let model        = get_world_from_local(in.instance_index);
    let sphere_center = (model * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    let cam_pos  = view.world_position;
    let to_cam   = normalize(cam_pos - sphere_center);

    let ref_up = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(to_cam.y) > 0.99);
    let right  = normalize(cross(ref_up, to_cam));
    let up     = normalize(cross(to_cam, right));

    // Expand the billboard to the atmosphere shell silhouette, not just
    // the solid sphere. The quad is square with the silhouette inscribed,
    // so along the cardinal edges (up/right) the quad has zero margin
    // beyond the inscribed circle — and the in-scattered halo, which
    // lives at altitudes up to `atmosphere.atmos_geom.x`, gets scissored
    // off there. Sizing from the outer shell radius keeps the halo
    // visible all the way around. Airless bodies have `atmos_geom.x == 0`,
    // so this collapses to the original formula.
    let effective_radius = params.radius + atmosphere.atmos_geom.x;
    let d      = length(cam_pos - sphere_center);
    let d_safe = max(d, effective_radius * 1.0001);
    let billboard_radius = effective_radius * d_safe
        / sqrt(d_safe * d_safe - effective_radius * effective_radius);

    let world_pos = sphere_center
        + in.position.x * right * billboard_radius
        + in.position.y * up   * billboard_radius;

    // Pixel size on the planet's nearest surface, in meters. For a perspective
    // projection, `clip_from_view[1][1] = 1 / tan(fov_y/2)`, so the world-space
    // height of one pixel at view-space distance z is `2*z / (h * f)`. We use
    // the nearest-surface distance `d - params.radius` so LOD cutoffs are set
    // by the sharpest-detail sample — matches the intent of the old dpdx value.
    let f_y          = view.clip_from_view[1][1];
    let viewport_h   = view.viewport.w;
    let near_surface = max(d - params.radius, params.radius * 0.001);
    let pixel_render = 2.0 * near_surface / max(viewport_h * f_y, 1e-6);
    let m_per_render = detail.body_radius_m / max(params.radius, 1e-6);

    var out: VertexOutput;
    out.clip_position  = view.clip_from_world * vec4(world_pos, 1.0);
    out.world_position = world_pos;
    out.sphere_center  = sphere_center;
    out.pixel_size_m   = pixel_render * m_per_render;
    return out;
}

// ── Fragment stage ────────────────────────────────────────────────────────────

struct FragOutput {
    @location(0)          color: vec4<f32>,
    @builtin(frag_depth)  depth: f32,
}

// ── Hash primitives ─────────────────────────────────────────────────────────

fn pcg(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_cell(ix: i32, iy: i32, iz: i32, octave: u32) -> u32 {
    let ux = bitcast<u32>(ix);
    let uy = bitcast<u32>(iy);
    let uz = bitcast<u32>(iz);
    var h = ux * 73856093u;
    h = h ^ (uy * 19349663u);
    h = h ^ (uz * 83492791u);
    h = pcg(h);
    h = h ^ (octave * 2654435769u);
    h = h ^ detail.seed_lo;
    h = pcg(h);
    h = h ^ (detail.seed_hi * 1540483477u);
    return pcg(h);
}

// ── Crater profile (must agree with planet_gen/src/crater.rs) ───────────────

const SIMPLE_DEPTH_RATIO: f32       = 0.2;
const SIMPLE_RIM_RATIO: f32         = 0.04;
const SIMPLE_INTERIOR_EXPONENT: f32 = 2.5;
const EJECTA_EXTENT: f32            = 2.5;
const RIM_FRESHNESS_SIGMA: f32      = 0.22;
const COMPLEX_FLOOR_FRACTION: f32   = 0.55;
const COMPLEX_PEAK_HEIGHT_FRAC: f32 = 0.15;
const COMPLEX_PEAK_BASE_FRAC: f32   = 0.15;
const COMPLEX_MIN_DEPTH_RATIO: f32  = 0.05;

const CRATER_KAPPA_M2_PER_MYR: f32 = 5.5;
const CRATER_C_DIFF: f32 = 14.5;
const CRATER_MIN_RETENTION: f32 = 0.03;
const CRATER_D_RELAX_THRESHOLD_M: f32 = 30000.0;
const CRATER_D_RELAX_REF_M: f32 = 100000.0;
const CRATER_RELAX_TAU_GYR: f32 = 3.0;
const CRATER_INFILL_START_GYR: f32 = 0.35;
const CRATER_INFILL_FULL_GYR: f32 = 3.8;
const CRATER_OLD_INFILL_RETENTION: f32 = 0.38;

fn complex_depth_ratio(d_over_dsc: f32) -> f32 {
    let t = exp(-max(d_over_dsc - 1.0, 0.0) / 3.0);
    return COMPLEX_MIN_DEPTH_RATIO + (SIMPLE_DEPTH_RATIO - COMPLEX_MIN_DEPTH_RATIO) * t;
}

fn crater_degradation_factor(radius_m: f32, age_gyr: f32) -> f32 {
    let d_m = radius_m * 2.0;
    let k = CRATER_KAPPA_M2_PER_MYR * age_gyr * 1000.0;
    let diffusion = exp(-CRATER_C_DIFF * k / max(d_m * d_m, 1.0));

    var relaxation = 1.0;
    if d_m > CRATER_D_RELAX_THRESHOLD_M {
        let excess = (d_m - CRATER_D_RELAX_THRESHOLD_M) / CRATER_D_RELAX_REF_M;
        relaxation = exp(-excess * (age_gyr / CRATER_RELAX_TAU_GYR));
    }

    let infill_age = smoothstep(CRATER_INFILL_START_GYR, CRATER_INFILL_FULL_GYR, age_gyr);
    let infill = 1.0 - infill_age * (1.0 - CRATER_OLD_INFILL_RETENTION);
    return max(diffusion * relaxation * infill, CRATER_MIN_RETENTION);
}

fn simple_profile(r: f32, depth: f32, rim: f32) -> vec2<f32> {
    if r <= 1.0 {
        let n = SIMPLE_INTERIOR_EXPONENT;
        let h = -depth + (depth + rim) * pow(r, n);
        let dh = (depth + rim) * n * pow(r, n - 1.0);
        return vec2(h, dh);
    } else {
        let span = EJECTA_EXTENT - 1.0;
        let t = clamp((r - 1.0) / span, 0.0, 1.0);
        let s_taper = t * t * (3.0 - 2.0 * t);
        let fade = 1.0 - s_taper;
        let dfade_dr = -6.0 * t * (1.0 - t) / span;

        let inv3 = 1.0 / (r * r * r);
        let base = rim * inv3;
        let dbase_dr = -3.0 * rim / (r * r * r * r);

        let h = base * fade;
        let dh = dbase_dr * fade + base * dfade_dr;
        return vec2(h, dh);
    }
}

fn complex_profile(r: f32, depth: f32, rim: f32) -> vec2<f32> {
    var base_h: f32;
    var base_dh: f32;
    if r <= 1.0 {
        if r <= COMPLEX_FLOOR_FRACTION {
            base_h = -depth;
            base_dh = 0.0;
        } else {
            let span = 1.0 - COMPLEX_FLOOR_FRACTION;
            let t = (r - COMPLEX_FLOOR_FRACTION) / span;
            let s = t * t * (3.0 - 2.0 * t);
            let ds_dr = 6.0 * t * (1.0 - t) / span;
            let h_total = depth + rim;
            base_h = -depth + h_total * s;
            base_dh = h_total * ds_dr;
        }
    } else {
        let span = EJECTA_EXTENT - 1.0;
        let t = clamp((r - 1.0) / span, 0.0, 1.0);
        let s_taper = t * t * (3.0 - 2.0 * t);
        let fade = 1.0 - s_taper;
        let dfade_dr = -6.0 * t * (1.0 - t) / span;

        let inv3 = 1.0 / (r * r * r);
        let raw = rim * inv3;
        let draw_dr = -3.0 * rim / (r * r * r * r);
        base_h = raw * fade;
        base_dh = draw_dr * fade + raw * dfade_dr;
    }
    let sigma = COMPLEX_PEAK_BASE_FRAC;
    let g = exp(-(r * r) / (2.0 * sigma * sigma));
    let peak = COMPLEX_PEAK_HEIGHT_FRAC * depth * g;
    let dpeak = -COMPLEX_PEAK_HEIGHT_FRAC * depth * g * (r / (sigma * sigma));
    return vec2(base_h + peak, base_dh + dpeak);
}

fn fresh_crater_maturity(r: f32) -> f32 {
    let dr = r - 1.0;
    let dip = exp(-(dr * dr) / (2.0 * RIM_FRESHNESS_SIGMA * RIM_FRESHNESS_SIGMA));
    var ejecta = 0.0;
    if r > 1.0 && r < EJECTA_EXTENT {
        let t = (r - 1.0) / (EJECTA_EXTENT - 1.0);
        let one_minus_t = 1.0 - t;
        ejecta = one_minus_t * one_minus_t;
    }
    let freshness = max(dip, ejecta);
    return clamp(1.0 - freshness, 0.0, 1.0);
}

// ── Per-cell crater accumulator (SSBO layer) ───────────────────────────────

struct CraterAccum {
    grad_tangent: vec3<f32>,
    height: f32,
    min_maturity: f32,
    // Self-shadow term from per-crater rim-occlusion tests. 1.0 = fully lit,
    // 0.0 = fully shadowed. Computed analytically: for each crater the
    // fragment lies inside, check if the sun-side rim blocks the sun given
    // the current sun elevation. `min` accumulator across craters.
    shadow: f32,
    // Signed albedo modulation. Interpreted at the call site as
    //   final_albedo = baked_albedo * clamp(1.0 + albedo_mod, 0.0, 4.0)
    // Per-crater zones: floor darken (negative), rim brighten + ejecta apron
    // (positive). The SSBO layer iterates craters below the cubemap bake
    // threshold, so its craters carry no CPU-painted albedo and need an
    // analytic equivalent here.
    albedo_mod: f32,
}

// Per-crater albedo signature used by the SSBO layer. Returns a
// signed scalar that should be folded into `albedo_mod`. `t` is the radial
// distance from the crater center in units of crater radius. `freshness`
// is in [0,1] (1 = pristine, 0 = mature) — older craters keep only muted
// contrast, matching the Pass 1.5 CPU path in
// `space_weather.rs`.
fn crater_albedo_delta(t: f32, freshness: f32) -> f32 {
    let strength = 0.22 + 0.78 * freshness;
    var delta: f32 = 0.0;
    if t < 0.55 {
        delta = delta - 0.85 * (1.0 - t / 0.55);
    }
    let rim_half: f32 = 0.28;
    if t > 1.0 - rim_half && t < 1.0 + rim_half {
        let rim_w = 1.0 - abs(t - 1.0) / rim_half;
        delta = delta + 1.15 * rim_w;
    }
    if t > 1.0 && t < 2.5 {
        let apron = 1.0 / (t * t * t);
        let fade = clamp((2.5 - t) / 1.5, 0.0, 1.0);
        delta = delta + 0.75 * apron * fade;
    }
    return delta * strength;
}

// Evaluate a single explicit crater from the SSBO at `p_unit` and fold its
// contribution into `accum`. Returns early if the crater is outside the
// ejecta blanket or its screen-space size is below the smoothstep floor.
fn apply_ssbo_crater(
    accum: ptr<function, CraterAccum>,
    p_unit: vec3<f32>,
    crater: Crater,
    pixel_size_m: f32,
    light_dir_local: vec3<f32>,
) {
    // Skip craters already rasterized into the height cubemap — the Layer 1
    // texel lookup already includes their displacement, so iterating them
    // here would double-count. Cratering publishes its bake threshold via
    // the detail uniform.
    if crater.radius_m >= detail.cubemap_bake_threshold_m {
        return;
    }

    let diameter_m = 2.0 * crater.radius_m;
    let diameter_px = diameter_m / max(pixel_size_m, 1e-6);
    // Fade window 1.5 – 10 px. Sub-pixel craters get fully culled at far
    // zooms (otherwise a population of barely-resolved features adds
    // shimmery noise to the disk); intermediate sizes ramp in smoothly.
    let weight = smoothstep(1.5, 10.0, diameter_px);
    if weight <= 0.0 {
        return;
    }

    let center = normalize(crater.center);
    let cos_theta = clamp(dot(p_unit, center), -1.0, 1.0);
    let theta = acos(cos_theta);
    let s_arc_m = theta * detail.body_radius_m;
    var r = s_arc_m / max(crater.radius_m, 1e-3);
    if r >= EJECTA_EXTENT {
        return;
    }

    // Projection of `center` into p_unit's tangent plane — points FROM the
    // sample point TOWARD the crater center (i.e., direction of decreasing
    // r). Sign is handled in the gradient line below.
    let proj = center - cos_theta * p_unit;
    let proj_len2 = dot(proj, proj);

    // Morphology branch: d/d_sc decides simple vs complex profile. The stored
    // depth/rim are pristine dimensions; apply the same age degradation used
    // by the bake and CPU sample paths before evaluating the SSBO crater.
    let degradation = crater_degradation_factor(crater.radius_m, crater.age_gyr);
    let depth = crater.depth_m * degradation;
    let rim = crater.rim_height_m * degradation;
    let d_over_dsc = diameter_m / max(detail.d_sc_m, 1.0);

    var hd: vec2<f32>;
    if d_over_dsc >= 1.0 {
        hd = complex_profile(r, depth, rim);
    } else {
        hd = simple_profile(r, depth, rim);
    }
    let h_m = hd.x;
    let dh_dr = hd.y;

    let fresh_m = fresh_crater_maturity(r);
    let age_blend = smoothstep(0.0, FRESH_AGE_GYR, crater.age_gyr);
    let aged_m = mix(fresh_m, 1.0, age_blend);
    let weighted_m = mix(1.0, aged_m, weight);

    let freshness = 1.0 - aged_m;
    let albedo_delta_w = crater_albedo_delta(r, freshness) * weight;

    let grad_proj_len = sqrt(proj_len2);
    if grad_proj_len < 1e-8 {
        (*accum).height = (*accum).height + h_m * weight;
        (*accum).min_maturity = min((*accum).min_maturity, weighted_m);
        (*accum).albedo_mod = (*accum).albedo_mod + albedo_delta_w;
        return;
    }
    let t_hat = proj / grad_proj_len;
    let grad = -(dh_dr) / max(crater.radius_m, 1.0) * t_hat;

    (*accum).grad_tangent = (*accum).grad_tangent + grad * weight;
    (*accum).height = (*accum).height + h_m * weight;
    (*accum).min_maturity = min((*accum).min_maturity, weighted_m);

    // Per-crater albedo modulation — analytic version of CPU Pass 1.5.
    // Computed before the early-return branch above so dead-center floor
    // darkening (where grad_proj_len → 0) still gets folded in.
    (*accum).albedo_mod = (*accum).albedo_mod + albedo_delta_w;

    // ── Per-crater analytical shadow ───────────────────────────────────────
    // The crater rim casts a shadow onto whatever fragment lies sun-ward of
    // it — could be the crater floor (r < 1), the ejecta blanket (1 < r <
    // EJECTA_EXTENT), or anywhere in between. Walks the sun direction in
    // the fragment's tangent plane and finds where it crosses the rim
    // circle, then compares the rim's height rise to the sun's elevation.
    let sin_sun = dot(light_dir_local, p_unit);
    if sin_sun > 0.0 {
        let sun_tangent = light_dir_local - sin_sun * p_unit;
        let cos_sun = length(sun_tangent);
        if cos_sun > 1e-4 {
            let sun_hat = sun_tangent / cos_sun;
            // Fragment position relative to crater center, in crater radii.
            // `t_hat` points from fragment toward center, so negate.
            let frag_rel = -t_hat * r;
            // Ray from frag in sun direction → rim circle |p|=1:
            //   (frag_rel + t·sun_hat)·(frag_rel + t·sun_hat) = 1
            //   t² + 2b·t + c = 0, b = frag_rel·sun_hat, c = r²-1
            let b = dot(frag_rel, sun_hat);
            let c = r * r - 1.0;
            let disc = b * b - c;
            if disc > 0.0 {
                let s_disc = sqrt(disc);
                // Interior: ray starts inside circle, take far exit (far rim).
                // Exterior: ray outside circle, take near entry (first rim hit).
                var t_hit: f32 = -1.0;
                if c < 0.0 {
                    t_hit = -b + s_disc;
                } else {
                    let t_near = -b - s_disc;
                    if t_near > 0.0 { t_hit = t_near; }
                }
                if t_hit > 0.0 {
                    let delta_h = rim - h_m;
                    let lhs = delta_h * cos_sun;
                    let rhs = t_hit * crater.radius_m * sin_sun;
                    if lhs > rhs {
                        let margin = (lhs - rhs) / max(rhs, 1.0);
                        let s = 1.0 - clamp(margin * 8.0, 0.0, 1.0);
                        (*accum).shadow = min((*accum).shadow, mix(1.0, s, weight));
                    }
                }
            }
        }
    }
}

// Iterate every explicit crater in the 3×3×3 cell neighborhood of `p_unit`
// via the cell-hash spatial index. Contract:
//   - `detail.ssbo_cell_size` is the cell edge length in unit-sphere coords.
//   - `cell_index[hash & MASK]` → (start, count) into `feature_ids`.
//   - Each `feature_ids[start+i]` is an index into `craters[]`.
fn iterate_ssbo_craters(
    p_unit: vec3<f32>,
    pixel_size_m: f32,
    light_dir_local: vec3<f32>,
) -> CraterAccum {
    var accum: CraterAccum;
    accum.grad_tangent = vec3<f32>(0.0);
    accum.height = 0.0;
    accum.min_maturity = 1.0;
    accum.shadow = 1.0;
    accum.albedo_mod = 0.0;

    let cell_size_unit = SSBO_CELL_SIZE_UNIT;
    if arrayLength(&cell_index) == 0u {
        return accum;
    }

    // Whole-layer LOD cull. Every SSBO crater has `diameter < 2*bake_threshold`,
    // and `apply_ssbo_crater` fades in with `smoothstep(1.5, 10.0, diameter_px)`.
    // If even the largest possible crater is below 1.5 px, no crater in the
    // layer can contribute — skip the 27-cell iteration entirely.
    let max_diameter_m = 2.0 * detail.cubemap_bake_threshold_m;
    if max_diameter_m < 1.5 * pixel_size_m {
        return accum;
    }

    let inv = 1.0 / cell_size_unit;
    let px_cell = p_unit.x * inv;
    let py_cell = p_unit.y * inv;
    let pz_cell = p_unit.z * inv;
    let cx = i32(floor(px_cell));
    let cy = i32(floor(py_cell));
    let cz = i32(floor(pz_cell));

    // Adaptive neighborhood. Baker indexes each crater in exactly one cell
    // (the cell containing its center — see `build_ssbo_cell_table` in
    // crates/planet_rendering/src/bake.rs), so the shader must read every
    // cell whose stored craters can influence `p_unit`. A crater at the far
    // edge of a neighbor cell can reach at most
    //     EJECTA_EXTENT * bake_threshold_m / body_radius_m
    // into the fragment's own cell, in unit-sphere coords. Dividing by the
    // cell size gives the per-axis "must visit this neighbor" margin:
    let max_infl_unit = EJECTA_EXTENT * detail.cubemap_bake_threshold_m / max(detail.body_radius_m, 1.0);
    let infl = min(max_infl_unit / cell_size_unit, 1.0);

    let fx = px_cell - f32(cx);
    let fy = py_cell - f32(cy);
    let fz = pz_cell - f32(cz);
    let dx_lo = select(0, -1, fx < infl);
    let dx_hi = select(0,  1, fx > 1.0 - infl);
    let dy_lo = select(0, -1, fy < infl);
    let dy_hi = select(0,  1, fy > 1.0 - infl);
    let dz_lo = select(0, -1, fz < infl);
    let dz_hi = select(0,  1, fz > 1.0 - infl);

    // On Mira (bake_threshold≈5 km, cell≈52 km) `infl ≈ 0.24`, so the loop
    // visits ~3 cells on average instead of 27. The `min(..,1.0)` clamp
    // falls back to the full 3×3×3 worst case if a body's max crater ever
    // rivals the cell size.
    for (var dx: i32 = dx_lo; dx <= dx_hi; dx = dx + 1) {
        for (var dy: i32 = dy_lo; dy <= dy_hi; dy = dy + 1) {
            for (var dz: i32 = dz_lo; dz <= dz_hi; dz = dz + 1) {
                let h = hash_cell(cx + dx, cy + dy, cz + dz, 0u);
                let slot = h & CELL_TABLE_MASK;
                let range = cell_index[slot];
                for (var i: u32 = 0u; i < range.count; i = i + 1u) {
                    let crater_idx = feature_ids[range.start + i];
                    let crater = craters[crater_idx];
                    apply_ssbo_crater(&accum, p_unit, crater, pixel_size_m, light_dir_local);
                }
            }
        }
    }

    return accum;
}

// ── Regional (large-scale) albedo modulation ───────────────────────────────

fn regional_albedo_mod(_p: vec3<f32>) -> f32 {
    return 1.0;
}

// ── Dynamic ice-cap overlay ────────────────────────────────────────────────

struct IceOverlay {
    coverage: f32,
    albedo: vec3<f32>,
    albedo_strength: f32,
    roughness: f32,
    roughness_strength: f32,
    height_delta_m: f32,
}

struct IceCapMasks {
    coverage: f32,
    interior: f32,
    dirty_fringe: f32,
    detached_frost: f32,
    edge_band: f32,
    troughs: f32,
}

fn ice_cap_texture(dir: vec3<f32>, cap: IceCap) -> f32 {
    let broad = fbm3(
        dir * 2.7,
        cap.seed,
        4u,
        0.55,
        2.01,
    );
    let mottle = fbm3(
        dir * 13.0,
        cap.seed ^ 0x51F16E23u,
        3u,
        0.52,
        2.04,
    );
    return clamp(0.5 + broad * 0.34 + mottle * 0.16, 0.0, 1.0);
}

fn empty_ice_cap_masks() -> IceCapMasks {
    return IceCapMasks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

fn combine_ice_cap_masks(a: IceCapMasks, b: IceCapMasks) -> IceCapMasks {
    return IceCapMasks(
        max(a.coverage, b.coverage),
        max(a.interior, b.interior),
        max(a.dirty_fringe, b.dirty_fringe),
        max(a.detached_frost, b.detached_frost),
        max(a.edge_band, b.edge_band),
        max(a.troughs, b.troughs),
    );
}

fn ice_pole_masks(
    dir: vec3<f32>,
    cap: IceCap,
    pole_latitude_deg: f32,
    edge_latitude: f32,
    solid_latitude: f32,
    lace: f32,
    seed: u32,
) -> IceCapMasks {
    let sharpness = clamp(cap.edge_sharpness, 0.0, 1.0);
    let transition_deg = max(mix(1.25, 0.32, sharpness), 0.25);
    let solid_span_deg = max(solid_latitude - edge_latitude, transition_deg + 0.35);
    let signed_edge_distance = pole_latitude_deg + lace - edge_latitude;

    let coverage = smoothstep(0.0, transition_deg, signed_edge_distance);
    let interior = smoothstep(transition_deg, solid_span_deg, signed_edge_distance);

    let outside_distance = max(-signed_edge_distance, 0.0);
    let fringe_width_deg = max(1.2, cap.edge_noise_deg * 0.55 + 0.4);
    let dirty_fringe =
        (1.0 - coverage) * (1.0 - smoothstep(0.0, fringe_width_deg, outside_distance));

    let patch_shell = smoothstep(0.25, 0.9, outside_distance)
        * (1.0 - smoothstep(fringe_width_deg, fringe_width_deg + 1.2, outside_distance));
    let patch_noise = 0.5 + 0.5 * fbm3(
        dir * max(cap.noise_frequency, 0.001) * 10.5,
        seed ^ 0xD17C0DEu,
        4u,
        0.58,
        2.11,
    );
    let detached_frost = patch_shell * smoothstep(0.66, 0.86, patch_noise);

    let edge_width_deg = max(0.75, cap.edge_noise_deg * 0.26);
    let edge_band = 1.0 - smoothstep(0.0, edge_width_deg, abs(signed_edge_distance));

    let trough_noise = 0.5 + 0.5 * fbm3(
        dir * max(cap.noise_frequency, 0.001) * 3.2,
        seed ^ 0x7A10F205u,
        4u,
        0.52,
        2.03,
    );
    let troughs = coverage * smoothstep(0.70, 0.88, trough_noise) * (0.35 + 0.35 * edge_band);

    return IceCapMasks(
        coverage,
        interior,
        dirty_fringe,
        detached_frost,
        edge_band,
        troughs,
    );
}

fn ice_cap_masks(dir: vec3<f32>, cap: IceCap) -> IceCapMasks {
    if cap.flags == 0u || cap.coverage_scale <= 0.0 || cap.max_thickness_m <= 0.0 {
        return empty_ice_cap_masks();
    }

    let axis = normalize(cap.axis);
    let sample_lat_deg = asin(clamp(dot(dir, axis), -1.0, 1.0)) * (180.0 / PI);
    let reach_latitude = clamp(
        cap.edge_latitude_deg + cap.edge_offset_deg - cap.edge_noise_deg * 1.8 - 3.0,
        0.0,
        89.5,
    );
    let near_included_pole =
        (((cap.flags & 1u) != 0u) && sample_lat_deg >= reach_latitude)
        || (((cap.flags & 2u) != 0u) && -sample_lat_deg >= reach_latitude);
    if !near_included_pole {
        return empty_ice_cap_masks();
    }

    let freq = max(cap.noise_frequency, 0.001);
    let edge_noise = fbm3(dir * freq, cap.seed ^ 0xA71C3E55u, 4u, 0.55, 2.03);
    let scallop_noise = fbm3(dir * freq * 5.8, cap.seed ^ 0x8CEB7A91u, 3u, 0.56, 2.07);
    let lace_noise = fbm3(dir * freq * 11.0, cap.seed ^ 0xC1CEB47Du, 3u, 0.52, 2.07);

    let scallop_strength = 0.25 + 0.35 * clamp(cap.edge_sharpness, 0.0, 1.0);
    let edge_offset = edge_noise * cap.edge_noise_deg + scallop_noise * cap.edge_noise_deg * scallop_strength;
    let edge_latitude = clamp(cap.edge_latitude_deg + cap.edge_offset_deg + edge_offset, 0.0, 89.5);
    let solid_latitude = clamp(
        max(cap.solid_latitude_deg + cap.edge_offset_deg + edge_noise * cap.edge_noise_deg * 0.18, edge_latitude + 0.6),
        edge_latitude + 0.6,
        90.0,
    );
    let lace = lace_noise * cap.edge_noise_deg * 0.20;

    var masks = empty_ice_cap_masks();
    if (cap.flags & 1u) != 0u {
        masks = combine_ice_cap_masks(
            masks,
            ice_pole_masks(dir, cap, sample_lat_deg, edge_latitude, solid_latitude, lace, cap.seed),
        );
    }
    if (cap.flags & 2u) != 0u {
        masks = combine_ice_cap_masks(
            masks,
            ice_pole_masks(
                dir,
                cap,
                -sample_lat_deg,
                edge_latitude,
                solid_latitude,
                -lace,
                cap.seed ^ 0x5A17C4A7u,
            ),
        );
    }

    return masks;
}

fn sample_ice_caps(dir: vec3<f32>) -> IceOverlay {
    var out = IceOverlay(
        0.0,
        vec3<f32>(0.0),
        0.0,
        0.0,
        0.0,
        0.0,
    );

    let count = arrayLength(&ice_caps);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let cap = ice_caps[i];
        let masks = ice_cap_masks(dir, cap);
        let sparse_ice = max(masks.detached_frost * 0.68, masks.dirty_fringe * 0.34);
        let coverage = clamp(max(masks.coverage, sparse_ice) * cap.coverage_scale, 0.0, 1.0);
        if coverage <= out.coverage {
            continue;
        }

        let texture = ice_cap_texture(dir, cap);
        let dirty_edge = clamp(
            masks.edge_band * 0.58 + masks.dirty_fringe * 0.70 + masks.troughs * 0.55,
            0.0,
            1.0,
        );
        let clean_ice = clamp(
            0.74 + texture * 0.16 + masks.interior * 0.20 - dirty_edge * 0.24,
            0.0,
            1.0,
        );
        let clean_color = mix(cap.dust_albedo_linear, cap.albedo_linear, clean_ice);
        let frost_color = mix(clean_color, cap.dust_albedo_linear, clamp(cap.dustiness, 0.0, 1.0));
        let dirty_color = mix(cap.dust_albedo_linear, frost_color, 0.32 + texture * 0.26);
        let dirty_mix = clamp(
            masks.dirty_fringe * 0.85 + masks.edge_band * 0.45 + masks.troughs * 0.55,
            0.0,
            1.0,
        );
        out.coverage = coverage;
        out.albedo = mix(frost_color, dirty_color, dirty_mix);
        out.albedo_strength = cap.albedo_strength;
        out.roughness = cap.roughness;
        out.roughness_strength = cap.roughness_strength;
        out.height_delta_m = cap.max_thickness_m * max(cap.thickness_scale, 0.0) * coverage;
    }

    return out;
}

// ── Dynamic active-dune overlay ────────────────────────────────────────────

struct DuneOverlay {
    coverage: f32,
    albedo: vec3<f32>,
    albedo_strength: f32,
    roughness: f32,
    roughness_strength: f32,
    height_delta_m: f32,
}

fn empty_dune_overlay() -> DuneOverlay {
    return DuneOverlay(0.0, vec3<f32>(0.0), 0.0, 0.76, 0.0, 0.0);
}

fn active_dune_layer_dir(dir: vec3<f32>) -> vec3<f32> {
    return normalize(rotate_quat(params.active_dune_texture_from_body, dir));
}

fn sample_baked_active_dunes(dir: vec3<f32>) -> DuneOverlay {
    if arrayLength(&active_dunes) == 0u {
        return empty_dune_overlay();
    }

    let layer_dir = active_dune_layer_dir(dir);
    let rgba = textureSample(active_dune_albedo_tex, active_dune_albedo_sampler, layer_dir);
    return DuneOverlay(
        rgba.a,
        rgba.rgb,
        1.0,
        0.78,
        clamp(rgba.a * 0.32, 0.0, 0.32),
        0.0,
    );
}

fn dynamic_height_delta_m(dir: vec3<f32>, include_active_dunes: bool) -> f32 {
    var h = sample_ice_caps(dir).height_delta_m;
    if include_active_dunes {
        h = h + textureSample(active_dune_height_tex, active_dune_height_sampler, active_dune_layer_dir(dir)).r * params.height_range;
    }
    return h;
}

// ── Radial feature erosion/color detail ────────────────────────────────────

struct RadialAccum {
    grad_tangent: vec3<f32>,
    dark_mix: f32,
    warm_mix: f32,
    roughness_delta: f32,
}

fn radial_feature_params(feature: RadialFeature) -> ErosionFilterParams {
    return ErosionFilterParams(
        feature.erosion_scale_m,
        0.018,
        0.55,
        1.7,
        vec4<f32>(0.1, 0.0, 0.1, 2.0),
        vec4<f32>(1.25, 1.25, 2.8, 1.5) * 0.22,
        vec2<f32>(0.45, 0.95),
        0.7,
        0.5,
        4,
        2.0,
        0.5,
    );
}

fn radial_seed_offset(seed: u32) -> vec2<f32> {
    let lo = f32(seed & 0xFFFFu) * (1.0 / 65535.0);
    let hi = f32((seed >> 16u) & 0xFFFFu) * (1.0 / 65535.0);
    return vec2<f32>(lo * 9137.0 + 173.0, hi * 7193.0 + 421.0);
}

fn radial_seed_phase(seed: u32, shift: u32) -> f32 {
    let bits = (seed >> shift) & 0xFFu;
    return f32(bits) * (TAU / 255.0);
}

fn radial_angular_lobe(theta: f32, center: f32, half_width: f32) -> f32 {
    let delta = fract((theta - center + PI) / TAU) * TAU - PI;
    return 1.0 - smoothstep(0.0, half_width, abs(delta));
}

fn radial_boundary_scale(local_m: vec2<f32>, rx: f32, ry: f32, seed: u32) -> f32 {
    let ux = local_m.x / rx;
    let uy = local_m.y / ry;
    let theta = atan2(uy, ux);
    let s = sin(theta);
    let c = cos(theta);
    let phase_a = radial_seed_phase(pcg(seed ^ 0x56A3C9B5u), 0u);
    let phase_b = radial_seed_phase(pcg(seed ^ 0xA511E9B3u), 8u);
    let phase_c = radial_seed_phase(pcg(seed ^ 0x3198F52Fu), 16u);
    let phase_d = radial_seed_phase(pcg(seed ^ 0xC73D5B91u), 24u);
    let low_lobes =
        sin(theta * 2.0 + phase_a) * 0.145 +
        sin(theta * 3.0 + phase_b) * 0.120 +
        sin(theta * 5.0 + phase_c) * 0.085 +
        sin(theta * 8.0 + phase_d) * 0.052;
    let directed_lobes =
        radial_angular_lobe(theta, phase_a + 0.70, 0.52) * 0.20 +
        radial_angular_lobe(theta, phase_b + 1.25, 0.38) * 0.14 -
        radial_angular_lobe(theta, phase_c + 0.45, 0.46) * 0.17 -
        radial_angular_lobe(theta, phase_d + 2.10, 0.34) * 0.12;
    let broad = fbm3(
        vec3<f32>(c * 1.45, s * 1.45, 0.37),
        pcg(seed ^ 0x8E2C4F15u),
        4u,
        0.56,
        2.03,
    ) * 0.160;
    let scallop = fbm3(
        vec3<f32>(ux * 5.8 + c * 0.85, uy * 5.8 + s * 0.85, 1.19),
        pcg(seed ^ 0xD4129C7Du),
        3u,
        0.52,
        2.08,
    ) * 0.105;
    return clamp(1.0 + low_lobes + directed_lobes + broad + scallop, 0.58, 1.58);
}

fn radial_profile_radius(local_m: vec2<f32>, rx: f32, ry: f32, seed: u32) -> f32 {
    let raw_r = sqrt((local_m.x / rx) * (local_m.x / rx) + (local_m.y / ry) * (local_m.y / ry));
    let edge_scale = radial_boundary_scale(local_m, rx, ry, seed);
    let edge_weight = smoothstep(0.32, 0.92, raw_r);
    return raw_r / max(1.0 + (edge_scale - 1.0) * edge_weight, 0.55);
}

fn radial_dome_height_and_slope(local_m: vec2<f32>, rx: f32, ry: f32, height_m: f32, seed: u32) -> vec3<f32> {
    let raw_r = sqrt((local_m.x / rx) * (local_m.x / rx) + (local_m.y / ry) * (local_m.y / ry));
    let r = radial_profile_radius(local_m, rx, ry, seed);
    if r >= 1.0 {
        return vec3<f32>(0.0);
    }

    let p = 1.72;
    let q = 1.18;
    let inner = max(1.0 - pow(r, p), 0.0);
    let profile = pow(inner, q);
    let h = profile * height_m;
    if r < 1e-5 || inner <= 1e-5 {
        return vec3<f32>(h, 0.0, 0.0);
    }

    let dprofile_dr = -q * p * pow(r, p - 1.0) * pow(inner, q - 1.0);
    let dh_dr = dprofile_dr * height_m;
    let edge_scale = radial_boundary_scale(local_m, rx, ry, seed);
    let edge_weight = smoothstep(0.32, 0.92, raw_r);
    let profile_scale = 1.0 + (edge_scale - 1.0) * edge_weight;
    let r_safe = max(raw_r, 1e-5);
    let dr_dx = local_m.x / (rx * rx * r_safe * profile_scale);
    let dr_dy = local_m.y / (ry * ry * r_safe * profile_scale);
    return vec3<f32>(h, dh_dr * dr_dx, dh_dr * dr_dy);
}

fn apply_radial_feature(
    accum: ptr<function, RadialAccum>,
    p_unit: vec3<f32>,
    feature: RadialFeature,
    pixel_size_m: f32,
) {
    if feature.radius_m <= 0.0 || feature.height_m <= 0.0 || feature.erosion_scale_m <= 0.0 {
        return;
    }

    let center = normalize(feature.center);
    let cos_center = clamp(dot(p_unit, center), -0.999999, 0.999999);
    let local_m = vec2<f32>(
        atan2(dot(p_unit, feature.east), cos_center),
        atan2(dot(p_unit, feature.north), cos_center),
    ) * detail.body_radius_m;

    let rx = feature.radius_m * 1.06;
    let ry = feature.radius_m * 0.94;
    let r = radial_profile_radius(local_m, rx, ry, feature.seed);
    if r > 1.22 {
        return;
    }

    let apron_mask = 1.0 - smoothstep(0.90, 1.18, r);
    let flank_mask = smoothstep(0.18, 0.36, r) * (1.0 - smoothstep(0.54, 1.12, r));
    let caldera_suppression = 1.0 - smoothstep(0.055, 0.17, r);
    let erodible = flank_mask * (1.0 - caldera_suppression);
    let basal_scarp = exp(-pow((r - 1.0) / 0.052, 2.0)) * (1.0 - smoothstep(0.84, 1.16, r));
    let caldera_rim = exp(-pow((r - 0.142) / 0.028, 2.0));

    let scale_px = feature.erosion_scale_m / max(pixel_size_m, 1.0);
    let color_lod = smoothstep(0.45, 1.75, scale_px);
    let normal_lod = smoothstep(1.8, 6.0, scale_px);
    if color_lod <= 0.0 && normal_lod <= 0.0 {
        return;
    }

    let base = radial_dome_height_and_slope(local_m, rx, ry, feature.height_m, feature.seed);
    let erosion = erosion_filter(
        local_m + radial_seed_offset(feature.seed),
        base,
        clamp(base.x / max(feature.height_m * 0.55, 1.0), -1.0, 1.0),
        radial_feature_params(feature),
    );

    let magnitude = max(erosion.magnitude, 1e-5);
    let erosion_delta = clamp(erosion.delta.x / magnitude, -1.0, 1.0);
    let crease = smoothstep(0.15, 0.85, -erosion.ridge_map) * erodible;
    let ridge = smoothstep(0.30, 0.95, erosion.ridge_map) * erodible;
    let incision = clamp((0.5 - erosion_delta) * 0.70 + crease * 0.45, 0.0, 1.0);

    let flow_phase = atan2(local_m.y / max(ry, 1.0), local_m.x / max(rx, 1.0));
    let lava_lobes =
        (sin(flow_phase * 17.0 + f32(feature.seed & 255u) * 0.017) * 0.5 + 0.5)
        * apron_mask
        * smoothstep(0.24, 0.96, r);

    let dark = color_lod * apron_mask * clamp(
        incision * 0.34 + basal_scarp * 0.34 + caldera_rim * 0.32 + lava_lobes * 0.10,
        0.0,
        0.68,
    );
    let warm = color_lod * apron_mask * clamp(
        ridge * 0.18 + (1.0 - incision) * lava_lobes * 0.06,
        0.0,
        0.24,
    );

    let grad_vec = feature.east * erosion.delta.y + feature.north * erosion.delta.z;
    let grad_tangent = grad_vec - dot(grad_vec, p_unit) * p_unit;
    (*accum).grad_tangent = (*accum).grad_tangent + grad_tangent * erodible * normal_lod * 0.72;
    (*accum).dark_mix = max((*accum).dark_mix, dark);
    (*accum).warm_mix = max((*accum).warm_mix, warm);
    (*accum).roughness_delta = max(
        (*accum).roughness_delta,
        color_lod * apron_mask * (incision * 0.050 + ridge * 0.020 + basal_scarp * 0.030),
    );
}

fn iterate_radial_features(p_unit: vec3<f32>, pixel_size_m: f32) -> RadialAccum {
    var accum: RadialAccum;
    accum.grad_tangent = vec3<f32>(0.0);
    accum.dark_mix = 0.0;
    accum.warm_mix = 0.0;
    accum.roughness_delta = 0.0;

    let count = arrayLength(&radial_features);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        apply_radial_feature(&accum, p_unit, radial_features[i], pixel_size_m);
    }

    return accum;
}

// ── Normal perturbation from height cubemap ────────────────────────────────
//
// Finite-difference normals derived from the filterable height cubemap.
// Per-fragment evaluation gives full f32 precision for the gradient — the
// pre-baked normal cube in `StaticSurfaceData` exists for future ground LOD use, but
// the impostor reconstructs normals here so the shading retains the
// continuous depth that 8-bit object-space encoding can't preserve at
// shallow slope angles (where the terminator and crater rim transitions
// live).

fn perturb_normal_from_height(n: vec3<f32>) -> vec3<f32> {
    let res = f32(textureDimensions(height_tex).x);
    if res < 2.0 {
        return n;
    }

    // Branchless orthonormal tangent frame on the sphere (Duff et al. 2017,
    // "Building an Orthonormal Basis, Revisited"). Continuous everywhere
    // except n.z = -1; a `select` on |n.y| > 0.99 would flip the tangent
    // ~90° at latitude ±82° and bisect features that cross it.
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let tangent   = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bitangent = vec3<f32>(b, s + n.y * n.y * a, -n.y);

    // Offset ~1.5 texels on the cubemap.
    let offset = 1.5 / res;

    let h_e = sample_height_normal_m(n + tangent * offset);
    let h_w = sample_height_normal_m(n - tangent * offset);
    let h_n = sample_height_normal_m(n + bitangent * offset);
    let h_s = sample_height_normal_m(n - bitangent * offset);

    let ds = detail.body_radius_m * offset * 2.0;
    if ds < 1e-6 {
        return n;
    }

    let relief_normal_strength = select(1.0, 0.42, params.sea_level_m > -1.0e8);
    let dh_dt = (h_e - h_w) / ds * relief_normal_strength;
    let dh_db = (h_n - h_s) / ds * relief_normal_strength;

    return normalize(n - tangent * dh_dt - bitangent * dh_db);
}

// ── Rotation helper ────────────────────────────────────────────────────────
//
// Rotate a direction by a quaternion. Used to transform a world-space normal
// into body-local space (where the cubemaps were baked) and back.

fn rotate_quat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

fn conjugate_quat(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

// ── Self-shadow raymarch ───────────────────────────────────────────────────
//
// Casts a ray from the surface point along the sun direction through the
// height cubemap. Captures basin/crater-rim shadows at the frequencies baked
// into the cubemap (≥ 5 km features on Mira). Sub-texel features are not
// shadowed — their normal perturbation already darkens the lit term. Cheap:
// only runs near the terminator, where shadows actually reach across texels.

// Canonical high-frequency *direction* warp. Perturbs the cubemap
// sample direction by a vec3 fbm field; the displacement on the
// sphere is `amp * fbm * R`. With `amp ≈ 1 texel of arc`, this
// breaks the cubemap-texel staircase out of grid alignment without
// adding any height roughness — the bilinear-interpolated baked
// height field is simply read at a fractally perturbed location.
//
// Three independent fbm evaluations (seed-decorrelated by `+1`,
// `+2` on the sub-seed). Each fbm shares the same lattice via the
// `+offset` constants on the input — same trick as
// `topography.rs::nearest_centroid_warped`.
fn coastline_warp_dir(dir: vec3<f32>) -> vec3<f32> {
    if params.coastline_warp_amp_radians <= 0.0 {
        return dir;
    }
    let p = dir * detail.body_radius_m * params.coastline_warp_freq_per_m;
    let oct = params.coastline_octaves;
    let s = params.coastline_seed;
    let wx = fbm3(p,                                   s,                  oct, 0.5, 2.0);
    let wy = fbm3(p + vec3<f32>(17.31, 17.31, 17.31), s + 1u,             oct, 0.5, 2.0);
    let wz = fbm3(p + vec3<f32>(41.17, 41.17, 41.17), s + 2u,             oct, 0.5, 2.0);
    let warp = vec3<f32>(wx, wy, wz) * params.coastline_warp_amp_radians;
    return normalize(dir + warp);
}

// Canonical high-frequency *height* jitter. Adds a scalar fbm in
// meters on top of the (already warp-sampled) baked height. Gives
// sub-texel surface detail visible on close approach.
fn coastline_jitter_m(dir: vec3<f32>) -> f32 {
    if params.coastline_jitter_amp_m <= 0.0 {
        return 0.0;
    }
    let p = dir * detail.body_radius_m * params.coastline_jitter_freq_per_m;
    let n = fbm3(p, params.coastline_seed, params.coastline_octaves, 0.5, 2.0);
    return n * params.coastline_jitter_amp_m;
}

// Bare cubemap height in meters, LOD 0. No warp, no jitter — the
// raw baked field. Used by the self-shadow ray march, where the
// canonical high-frequency band (sub-texel) costs ~21 fbm
// evaluations per fragment to evaluate and contributes shadows
// below the impostor's visible shadow scale. The bake's resolved
// frequencies are what the shadow march needs.
fn sample_height_baked_m(dir: vec3<f32>) -> f32 {
    let stored = textureSampleLevel(height_tex, height_sampler, dir, 0.0).r;
    return (stored - 0.5) * 2.0 * params.height_range;
}

// Bare cubemap height in meters, auto-LOD. Used by the
// normal-perturbation finite-difference pass: at orbital distance
// the GPU mip-blurs the cubemap so normals don't shimmer, and at
// approach the canonical high-freq band would only contribute
// ~5° of normal slope — invisible against the bake's native
// terrain detail. Skipping warp+jitter here keeps the per-fragment
// cost low (4 cubemap reads, no fbm).
fn sample_height_baked_auto_lod_m(dir: vec3<f32>) -> f32 {
    let stored = textureSample(height_tex, height_sampler, dir).r;
    return (stored - 0.5) * 2.0 * params.height_range;
}

fn sample_height_normal_m(dir: vec3<f32>) -> f32 {
    let d = normalize(dir);
    return sample_height_baked_auto_lod_m(d);
}

// Canonical surface height in meters: bake (sampled at the warped
// direction) + height jitter. This is the function the future 3D
// mesher must reproduce at the iso-contour to keep the LOD handoff
// continuous. Currently only consumed by the water-mask test —
// that's where the iso-contour lives, and the only place the cost
// of evaluating warp + jitter (4 fbm calls) is justified.
//
// Iso-contour cull: read the bare cubemap height first; when it sits
// well clear of sea level, optional canonical warp + jitter cannot push
// the result across the smoothstep band at the call site, so skip the
// four fbm evaluations and return the bare value. Current ocean materials
// leave these amplitudes at zero because coastline shape is baked; this
// path remains for explicit material experiments.
fn sample_height_m(dir: vec3<f32>) -> f32 {
    let bare_stored = textureSampleLevel(height_tex, height_sampler, dir, 0.0).r;
    let dynamic_m = dynamic_height_delta_m(dir, true);
    let bare_m = (bare_stored - 0.5) * 2.0 * params.height_range + dynamic_m;
    let bare_above_sea = bare_m - params.sea_level_m;
    let band = params.coastline_jitter_amp_m * 10.0 + 100.0;
    if abs(bare_above_sea) > band {
        return bare_m;
    }

    let warped = coastline_warp_dir(dir);
    let stored = textureSampleLevel(height_tex, height_sampler, warped, 0.0).r;
    let baked_m = (stored - 0.5) * 2.0 * params.height_range;
    return baked_m + coastline_jitter_m(warped) + dynamic_m;
}

fn sample_height_shadow_m(dir: vec3<f32>) -> f32 {
    // Dynamic veneers are too small or too broad to justify evaluating inside
    // the long-range self-shadow ray march. Keep this static-only: the ray
    // loop is 20 taps and sits on the fragment hot path near the terminator.
    return sample_height_baked_m(dir);
}

fn self_shadow(sample_dir: vec3<f32>, light_dir_local: vec3<f32>) -> f32 {
    let radius_m = detail.body_radius_m;
    // Bare baked height — the canonical high-freq band would only
    // contribute sub-texel shadows below the impostor's visible
    // shadow scale, and 21 fbm evaluations per ray is too costly.
    let h0 = sample_height_shadow_m(sample_dir);

    // Start a hair above the local surface so we don't self-intersect.
    let bias_m = max(radius_m * 0.0001, 5.0);
    let origin = sample_dir * (radius_m + h0 + bias_m);

    // Exponentially growing step so the ray covers short-range rim shadows
    // and long-range megabasin shadows in the same loop. 20 steps with
    // growth 1.3 reach ~radius_m * 0.6 of horizontal distance — enough for
    // megabasin-scale shadows at grazing sun.
    var step_m: f32 = radius_m * 0.0003;
    let growth: f32 = 1.3;
    let num_steps: i32 = 20;

    var shadow: f32 = 1.0;
    var t: f32 = 0.0;
    for (var i: i32 = 0; i < num_steps; i = i + 1) {
        t = t + step_m;
        let p = origin + light_dir_local * t;
        let r = length(p);
        let d = p / r;
        let h = sample_height_shadow_m(d);
        let surface_r = radius_m + h;
        if surface_r > r {
            let penetration = (surface_r - r) / (radius_m * 0.0003);
            shadow = min(shadow, 1.0 - clamp(penetration, 0.0, 1.0));
            if shadow <= 0.0 { break; }
        }
        step_m = step_m * growth;
    }
    return shadow;
}

// ── Hapke BRDF for airless regolith ────────────────────────────────────────
//
// Hapke (1981, 2002). Three physical ingredients:
//   - Shadow-hiding opposition effect B(g)
//   - Single-particle phase function P(g) (Henyey-Greenstein, back-scatter)
//   - Multiple-scattering H-functions via Chandrasekhar approximation
//
// Returns a reflectance factor that multiplies incoming flux × albedo to
// get reflected radiance. Parameters are tuned for lunar-type regolith.
//
// `roughness` (0..1) modulates the opposition surge width — smoother
// surfaces have a sharper, narrower surge (h smaller); rougher surfaces
// have a broader, more diffuse surge (h larger). Physical interpretation:
// shadow-hiding requires a coherent opposition direction; macroscopic
// roughness spreads that out angularly.
//
// Inputs are all in the same space; handedness doesn't matter.
fn hapke_brdf(n_dot_l: f32, n_dot_v: f32, cos_phase: f32, roughness: f32) -> f32 {
    let mu0 = max(n_dot_l, 0.0);
    let mu  = max(n_dot_v, 0.0);
    if mu0 <= 0.0 || mu <= 0.0 { return 0.0; }

    // Single-scattering albedo (0..1). Lunar highlands ~0.4, mare ~0.2.
    // Picked to match the prior visual brightness when combined with the
    // 2π global scale below.
    let w: f32 = 0.45;

    let cp = clamp(cos_phase, -1.0, 1.0);
    let g  = acos(cp);

    // Shadow-hiding opposition effect: B(g) = B0 / (1 + tan(g/2)/h).
    // h tuned to lunar regolith (~3.4°) at roughness 0.85; widened or
    // narrowed by surface roughness so e.g. a smooth icy patch keeps a
    // sharper opposition spike than a rubble field.
    let B0: f32 = 1.0;
    let h = mix(0.04, 0.10, clamp(roughness, 0.0, 1.0));
    let B_g = B0 / (1.0 + tan(g * 0.5) / h);

    // Single-particle phase function: Henyey-Greenstein with asymmetry
    // g_hg = -0.3 (back-scatter, typical of rough regolith grains).
    let g_hg: f32 = -0.3;
    let denom = 1.0 + g_hg * g_hg - 2.0 * g_hg * cp;
    let P_g = (1.0 - g_hg * g_hg) / pow(max(denom, 1e-6), 1.5);

    // Chandrasekhar H-function (Hapke 2002 two-stream approximation).
    let gamma = sqrt(max(1.0 - w, 0.0));
    let H_mu0 = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * mu0 * gamma);
    let H_mu  = (1.0 + 2.0 * mu) / (1.0 + 2.0 * mu * gamma);

    // Full radiance factor. The `(1 / (4π))` normalization is folded into
    // a global scale at the call site so brightness matches the prior
    // Lambert pipeline without re-tuning every planet's `light_intensity`.
    let r = w * (mu0 / (mu0 + mu)) * ((1.0 + B_g) * P_g + H_mu0 * H_mu - 1.0);
    return max(r, 0.0);
}

// ── Water BRDF (Cook-Torrance) ─────────────────────────────────────────────
//
// Replaces the Hapke path where the sampled height sits below sea level.
// Hapke is tuned for back-scattering regolith (opposition surge at phase 0);
// water is the opposite — a forward-scattering near-mirror that peaks at
// the specular direction. Ingredients:
//
//   - GGX (Trowbridge-Reitz) D lobe, roughness α = 0.06. Deliberately
//     non-mirror so the sun glint reads as a visible ~5–10 %-of-disk
//     patch from orbit rather than a single-pixel mirror flash.
//   - Smith G with the UE4 Schlick-k remap.
//   - Schlick Fresnel with F0 = 0.02 (water's normal-incidence reflectance
//     at 550 nm). Drives the darker-near-nadir / brighter-at-limb signature.
//   - Subsurface diffuse: water owns its colour below sea level. The terrain
//     albedo under the ocean is intentionally hidden; water colour comes from
//     an optical column that absorbs red fastest and keeps flat-ocean bodies
//     from looking like one-meter-deep shelf water.
//   - Grazing-angle reflection tinted by Rayleigh β so the limb reads
//     as "reflecting sky", not a white sun disk on vacuum.
//
// Drives both direct-star and planetshine through the same BRDF via the
// shared `thalos::lighting` helpers (`planetshine_sample`, `eclipse_factor`)
// so calibration matches the Hapke land path. Uses the geometric sphere
// normal — water is smooth at planetary scale. Self-shadow and crater
// shadows do not apply (flat surface).
fn water_brdf(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    n_dot_v: f32,
    f_nv: f32,
    alpha: f32,
    f0: f32,
    subsurface: vec3<f32>,
) -> vec3<f32> {
    let n_dot_l = max(dot(n, l), 0.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return vec3<f32>(0.0);
    }
    let h = normalize(l + v);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let a2 = alpha * alpha;
    let d_denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let d_ggx = a2 / (PI * d_denom * d_denom);

    let k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
    let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    let g_smith = g_v * g_l;

    let f_h = f0 + (1.0 - f0) * pow(max(1.0 - v_dot_h, 0.0), 5.0);

    let specular = (d_ggx * g_smith * f_h) / max(4.0 * n_dot_v, 1e-4);
    let diffuse = (1.0 - f_nv) * subsurface * n_dot_l / PI;
    return diffuse + vec3<f32>(specular);
}

fn water_column_color(depth_m: f32, n_dot_v: f32) -> vec3<f32> {
    let base = max(params.water_color_depth.xyz, vec3<f32>(0.0));
    let min_depth_m = max(params.water_color_depth.w, 1.0);
    let path_m = max(depth_m, min_depth_m) / max(n_dot_v, 0.18);

    // Clear water removes red quickly, green more slowly, and blue slowest.
    // The small volume-scatter term keeps deep water blue instead of merely
    // black, but caps far below the old shallow-column turquoise.
    let absorption = exp(-vec3<f32>(0.018, 0.010, 0.004) * path_m);
    let scatter_t = 1.0 - exp(-path_m / 180.0);
    let deep_scatter = vec3<f32>(0.002, 0.018, 0.060) * scatter_t;
    let apparent = base * absorption + deep_scatter;
    return clamp(apparent, vec3<f32>(0.0), vec3<f32>(0.08, 0.14, 0.20));
}

fn shade_water(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    depth_m: f32,
    sun_flux: f32,
    ambient: f32,
    sky_tint: vec3<f32>,
    hit: vec3<f32>,
) -> vec3<f32> {
    let f0 = 0.02;
    // Cox-Munk wave-slope σ for moderate wind (~5 m/s) is ~6–8°. GGX
    // roughness here encodes that statistical sub-pixel slope spread,
    // not "how rough each individual facet is" — a wider lobe means
    // many sub-resolution wave facets, mathematically equivalent to
    // explicit wave normals at sub-pixel scale (which would alias).
    // α = 0.10 puts the glint at ~12° across, matching ISS imagery.
    let alpha = 0.10;
    // Matches `hapke_scale` on the land path so both BRDFs calibrate
    // against the same `sun_flux`.
    let brdf_scale = 0.5;

    let n_dot_v = max(dot(n, v), 0.0);

    let f_nv = f0 + (1.0 - f0) * pow(max(1.0 - n_dot_v, 0.0), 5.0);
    let subsurface = water_column_color(depth_m, n_dot_v);

    // Ambient: Fresnel-modulated sky reflection + subsurface diffuse, both on
    // the same `ambient` scale.
    var lit = (f_nv * sky_tint + (1.0 - f_nv) * subsurface) * ambient;

    // Direct star.
    let sun_brdf = water_brdf(n, v, l, n_dot_v, f_nv, alpha, f0, subsurface);
    let sun_shadow = eclipse_factor(params.scene, hit, l);
    lit = lit + sun_brdf * sun_flux * brdf_scale * sun_shadow;

    // Planetshine — parent body acting as a Lambert reflector. Same BRDF
    // with the parent's direction as the incoming light, mirroring how
    // the Hapke land path runs `hapke_brdf` twice.
    let shine = planetshine_sample(params.scene, hit, l, sun_flux);
    if shine.enabled {
        let shine_brdf = water_brdf(n, v, shine.dir, n_dot_v, f_nv, alpha, f0, subsurface);
        lit = lit + shine_brdf * shine.tint * shine.flux * brdf_scale;
    }

    return lit;
}

// ── Fragment ────────────────────────────────────────────────────────────────
//
// Atmosphere integration (terrestrial impostor):
//
// - Surface-hit rays: compute Hapke lighting as before, then apply
//   limb darkening + terminator warmth + Fresnel rim + additive rim
//   halo on top of the lit output.
// - Miss rays (ray doesn't hit the solid sphere): check whether the
//   ray passes through the atmospheric shell. If it does and the rim
//   halo contribution is non-negligible, return the halo as the
//   fragment colour; otherwise discard as before.
//
// Bodies without a `terrestrial_atmosphere` block (Mira, Ignis, the
// airless moons) have every `atmosphere.*` scalar at zero — the helpers
// early-out and the shader's output is bit-identical to the pre-
// atmosphere pipeline.

// Primary star accessor — every caller of the atmosphere helpers needs
// this same triple, so compute it once per fragment.
struct PrimaryLight {
    dir_ws: vec3<f32>,
    flux: f32,
}

fn primary_light() -> PrimaryLight {
    let s = params.scene.stars[0];
    return PrimaryLight(s.dir_flux.xyz, s.dir_flux.w);
}

/// Geometric distances at which the view ray enters and exits the
/// atmosphere shell. Used by both the halo and body passes to bound
/// the scattering raymarch.
struct AtmosShellHit {
    /// True iff the view ray intersects the atmosphere shell (false
    /// means the body has no atmosphere or the ray misses entirely).
    valid: bool,
    /// Entry distance along the ray; clamped to 0 if the camera is
    /// inside the shell.
    t_enter: f32,
    /// Exit distance along the ray (far intersection with the outer
    /// shell). Always >= t_enter when `valid` is true.
    t_exit: f32,
}

fn atmosphere_shell_hit(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
) -> AtmosShellHit {
    let r_outer = params.radius + atmosphere.atmos_geom.x;
    if atmosphere.atmos_geom.x <= 0.0 {
        return AtmosShellHit(false, 0.0, 0.0);
    }
    let oc = cam_pos - center;
    let half_b = dot(oc, ray_dir);
    let c_o = dot(oc, oc) - r_outer * r_outer;
    let disc_o = half_b * half_b - c_o;
    if disc_o < 0.0 {
        return AtmosShellHit(false, 0.0, 0.0);
    }
    let sq = sqrt(disc_o);
    let t_near = -half_b - sq;
    let t_far  = -half_b + sq;
    if t_far <= 0.0 {
        return AtmosShellHit(false, 0.0, 0.0);
    }
    return AtmosShellHit(true, max(t_near, 0.0), t_far);
}

/// Sample the cloud-cover cube with banded differential rotation.
///
/// Each fragment's latitude determines a position in `sin²(lat) ∈
/// [0, 1]`. The K bands partition that interval evenly; two bracketing
/// bands supply their own rigidly-wrapped rotation phases, which we
/// use to rotate the sample direction twice (once per band) around
/// the body-local +Y axis. We fetch the cube at both rotated directions
/// and blend the scalar densities by the fragment's fractional
/// position between the bands. Because each band's phase wraps
/// independently mod TAU on the CPU (see `CloudBandState`), there is
/// no discontinuity anywhere on the sphere — rotation is seamless
/// forever, at every latitude, across save/load boundaries.
fn sample_cloud_banded(dir_local: vec3<f32>) -> f32 {
    let sin2 = clamp(dir_local.y * dir_local.y, 0.0, 1.0);
    let bf = sin2 * f32(CLOUD_BAND_COUNT - 1u);
    let lo = u32(floor(bf));
    let hi = min(lo + 1u, CLOUD_BAND_COUNT - 1u);
    let alpha = bf - floor(bf);
    let phase_lo = cloud_band_phase(lo, atmosphere);
    let phase_hi = cloud_band_phase(hi, atmosphere);
    let dir_lo = rotate_around_y(dir_local, phase_lo);
    let dir_hi = rotate_around_y(dir_local, phase_hi);
    let s_lo = textureSampleLevel(cloud_cover_tex, cloud_cover_sampler, dir_lo, 0.0).r;
    let s_hi = textureSampleLevel(cloud_cover_tex, cloud_cover_sampler, dir_hi, 0.0).r;
    return mix(s_lo, s_hi, alpha);
}

@fragment
fn fragment(in: VertexOutput) -> FragOutput {
    let cam_pos  = view.world_position;
    let ray_dir  = normalize(in.world_position - cam_pos);
    let light = primary_light();

    // Body and halo are split into two pipelines — see PlanetMaterial
    // and PlanetHaloMaterial in material.rs. The body pipeline writes
    // depth and discards rim-halo (miss) fragments; the halo pipeline
    // disables depth-write and discards body-hit fragments. Splitting
    // them is the only way to give the halo a depth-test that doesn't
    // occlude the celestial backdrop (stars/galaxies render at clip.z
    // = 0 and would fail `0 >= halo_depth` if the halo wrote a real
    // silhouette depth) while keeping the body's surface depth correct
    // for opaque objects that sit beyond the halo. WGSL has no
    // per-fragment depth-write toggle, so two pipelines is the answer.
    //
    // The shader-def `HALO_PASS` selects the halo pipeline; without
    // it, the shader compiles as the body pipeline.

    // Ray-sphere intersection against the body radius.
    let oc      = cam_pos - in.sphere_center;
    let half_b  = dot(oc, ray_dir);
    let c       = dot(oc, oc) - params.radius * params.radius;
    let disc    = half_b * half_b - c;
    let t       = -half_b - sqrt(max(disc, 0.0));
    // `disc < 0`     → ray never reaches the sphere.
    // `t   < 0`      → sphere is entirely behind the camera (or the
    //                  near intersection is behind it).
    // Both are halo-only fragments; only the body pipeline cares about
    // the distinction, and it discards either way.
    let is_miss = disc < 0.0 || t < 0.0;

#ifdef HALO_PASS
    if !is_miss {
        discard;
    }
    if !atmosphere_scattering_active(atmosphere) {
        discard;
    }
    let shell = atmosphere_shell_hit(cam_pos, ray_dir, in.sphere_center);
    if !shell.valid {
        discard;
    }
    // Per-pixel jitter breaks the regular sample pattern that would
    // otherwise show as banding at the terminator (where the sun
    // column changes rapidly between adjacent pixels).
    let jitter = atmosphere_jitter(in.clip_position.xy);
    let scatter = integrate_atmosphere(
        cam_pos, ray_dir, in.sphere_center, light.dir_ws,
        light.flux * SCENE_FLUX_SCALE,
        shell.t_enter, shell.t_exit,
        params.radius, atmosphere, jitter,
    );
    // Halo opacity = how much of the background is occluded by the
    // atmospheric column. Rec.709 luminance of the per-channel
    // transmittance gives a coherent single-alpha value that fades
    // smoothly between vacuum (T=1, α=0) and a fully extinguished
    // chord (T=0, α=1). Bias by a tiny floor so the discard below
    // skips numerically-empty fragments without dropping faint glow.
    let alpha = clamp(1.0 - dot(scatter.transmittance, vec3<f32>(0.2126, 0.7152, 0.0722)),
                      0.0, 1.0);
    let lum = dot(scatter.in_scatter, vec3<f32>(0.2126, 0.7152, 0.0722));
    if alpha < 0.002 && lum < 0.0005 {
        discard;
    }
    // Closest-approach depth gives a sensible silhouette depth for
    // halo fragments — used so opaque objects in front of the halo
    // (and other halos sorted closer) depth-test correctly.
    let closest_point = cam_pos + ray_dir * max(-half_b, 0.0);
    let clip = view.clip_from_world * vec4(closest_point, 1.0);
    return FragOutput(vec4(scatter.in_scatter, alpha), clip.z / clip.w);
#else
    if is_miss {
        discard;
    }

    let hit    = cam_pos + t * ray_dir;
    let normal = normalize(hit - in.sphere_center);

    // Apply planet orientation to the sample direction (not the geometry).
    // `orientation` maps world-space → body-local space (the frame the
    // cubemaps, SSBO craters, and shader-synthesized features were baked in).
    let sample_dir = rotate_quat(params.orientation, normal);

    // ── Layer 1a: filterable diffuse + roughness ──────────────────────────
    // Both cubes are bilinearly filtered and carry the field's continuous
    // per-texel values directly — no discrete material-id indirection, so
    // biome boundaries don't read as polygonal regions.
    let regional = clamp(regional_albedo_mod(sample_dir), 0.7, 1.3);
    var baked_albedo = textureSample(albedo_tex, albedo_sampler, sample_dir).rgb * regional;
    var surface_roughness = textureSample(roughness_tex, roughness_sampler, sample_dir).r;

    // ── Pixel size in meters ────────────────────────────────────────────
    // Carried from the vertex stage; see vertex() for the derivation.
    let pixel_size_m = in.pixel_size_m;

    // ── Layer 1b: height-derived normal perturbation ─────────────────────
    // Per-fragment finite-difference of the filterable height cube. f32
    // precision in the gradient preserves shallow slope angles that an
    // 8-bit baked normal cube would crush — terminator depth and crater
    // rim transitions read continuously here. Body-local until after the
    // SSBO crater gradients combine, then rotated to world space.
    var shading_normal = perturb_normal_from_height(sample_dir);

    // Feature-local radial erosion detail. This is intentionally independent
    // of `sample_height_m`: at impostor distances the signal should read
    // mostly as exposed material/roughness and normal detail, not true
    // silhouette displacement.
    let radial = iterate_radial_features(sample_dir, pixel_size_m);
    baked_albedo = mix(baked_albedo, vec3<f32>(0.16, 0.085, 0.055), radial.dark_mix);
    baked_albedo = mix(baked_albedo, vec3<f32>(0.64, 0.29, 0.105), radial.warm_mix);
    surface_roughness = clamp(surface_roughness + radial.roughness_delta, 0.45, 0.98);

    // Primary star — single-star path today. Multi-star support lives in
    // `params.scene.stars[0..star_count]` and the lighting helpers; the
    // crater iteration / SSBO layer is still expressed against a single
    // star direction, so add a loop here when more than one star is live.
    let primary_star = params.scene.stars[0];
    let sun_dir_ws = primary_star.dir_flux.xyz;
    let sun_flux   = primary_star.dir_flux.w;

    // ── Dark-hemisphere early-out ───────────────────────────────────────
    // Crater normal perturbation is clamped to `geo_n_dot_l + 0.05` below,
    // and the terminator wrap adds at most `roughness * 0.08`. Any fragment
    // deeper than that on the dark side is unreachable by both sources, so
    // the crater layers cannot change the final color — skip them.
    let geo_n_dot_l = dot(normal, sun_dir_ws);
    let wrap_slack  = params.terminator_wrap * 0.08;
    let dark_side   = geo_n_dot_l < -(0.05 + wrap_slack);

    // Sun direction in body-local space — shared by crater iteration (for
    // per-crater analytical shadow) and the cubemap raymarch below.
    let light_dir_local = rotate_quat(params.orientation, sun_dir_ws);

    var ssbo_grad  = vec3<f32>(0.0);
    var min_maturity = 1.0;
    var crater_shadow = 1.0;
    var crater_albedo_mod = 0.0;
    if !dark_side {
        // SSBO craters live in body-local space, so sample with the rotated
        // `sample_dir`, not the world-space `normal`.
        // ── Layer 2: SSBO craters (500 m – 5 km) ─────────────────────────
        let ssbo = iterate_ssbo_craters(sample_dir, pixel_size_m, light_dir_local);

        ssbo_grad  = ssbo.grad_tangent;
        min_maturity = ssbo.min_maturity;
        crater_shadow = ssbo.shadow;
        crater_albedo_mod = ssbo.albedo_mod;
    }

    // SSBO crater gradient is in body-local space (its tangent frame was
    // built around `sample_dir`), so the projection and subtraction happen
    // in body-local space; we rotate the final shading normal to world
    // space afterwards.
    let feature_grad = ssbo_grad + radial.grad_tangent;
    if length(feature_grad) > 0.0 {
        let grad_tangent = feature_grad - dot(feature_grad, sample_dir) * sample_dir;
        shading_normal = normalize(shading_normal - grad_tangent);
    }
    // Transform the fully-perturbed shading normal from body-local to world
    // space so the lighting dot product below is consistent with `light_dir`.
    shading_normal = rotate_quat(conjugate_quat(params.orientation), shading_normal);
    // Fresh craters: subtle brightening.  The material palette already encodes
    // the "fresh regolith" bias, so the ad-hoc FRESH_BIAS_COLOR tint is gone.
    let fresh_boost = clamp((1.0 - min_maturity) * 0.3, 0.0, 0.4);
    // Per-crater albedo signature from the SSBO layer. Clamped to a sane
    // range so a few stacked craters can't blow out the surface or drive
    // it negative. The factor parallels the CPU `space_weather.rs` Pass
    // 1.5 strength so cubemap-baked and SSBO craters look consistent
    // across the bake threshold.
    let crater_mod = clamp(crater_albedo_mod, -0.65, 1.20);
    var albedo = baked_albedo * (1.0 + fresh_boost + crater_mod);

    // Dynamic active dunes sit above the static substrate. The orbital
    // impostor applies them as a cheap material layer; dynamic height is
    // reserved for canonical sampled height where needed, not the normal path.
    let dune_overlay = sample_baked_active_dunes(sample_dir);
    if dune_overlay.coverage > 0.001 {
        let dune_albedo_t = clamp(dune_overlay.coverage * dune_overlay.albedo_strength, 0.0, 1.0);
        albedo = mix(albedo, dune_overlay.albedo, dune_albedo_t);
        let dune_roughness_t = clamp(dune_overlay.roughness_strength, 0.0, 1.0);
        surface_roughness = mix(surface_roughness, dune_overlay.roughness, dune_roughness_t);
    }
    // Seasonal ice caps are dynamic surface state, not baked terrain. The
    // orbital impostor keeps the visible veneer material cheap and does not
    // perturb normals or self-shadow from ice height.
    let ice_overlay = sample_ice_caps(sample_dir);
    if ice_overlay.coverage > 0.001 {
        let ice_albedo_t = clamp(ice_overlay.coverage * ice_overlay.albedo_strength, 0.0, 1.0);
        albedo = mix(albedo, ice_overlay.albedo, ice_albedo_t);
        let ice_roughness_t = clamp(ice_overlay.coverage * ice_overlay.roughness_strength, 0.0, 1.0);
        surface_roughness = mix(surface_roughness, ice_overlay.roughness, ice_roughness_t);
    }

    // ── Lighting: Hapke BRDF + planetshine ────────────────────────────────
    //
    // Replaces the previous Lambert + ad-hoc opposition surge. Hapke already
    // contains its own shadow-hiding surge term. Headroom ramp still caps
    // the perturbed shading normal against the geometric normal so crater
    // rims can't out-light body curvature near the terminator.
    let headroom = mix(0.05, 0.30, smoothstep(0.15, 0.40, geo_n_dot_l));
    let view_dir = normalize(cam_pos - hit);
    let n_dot_v = max(dot(shading_normal, view_dir), 0.0);

    // Primary: direct sunlight.
    let sun_n_dot_l_raw = dot(shading_normal, sun_dir_ws);
    let sun_n_dot_l = min(sun_n_dot_l_raw, geo_n_dot_l + headroom);
    let cos_phase_sun = dot(view_dir, sun_dir_ws);
    var sun_r = hapke_brdf(max(sun_n_dot_l, 0.0), n_dot_v, cos_phase_sun, surface_roughness);

    // Apply all shadow terms to the sun contribution only. Planetshine uses
    // a different incident direction so these don't apply to it.
    sun_r = sun_r * crater_shadow;
    // Self-shadow is a 20-tap cubemap ray march. The cast shadows it
    // captures are long near the terminator and short under high sun,
    // so fade the contribution to "no shadow" as the sun climbs above
    // ~50° from local zenith — at that elevation cubemap features
    // shorter than ~feature_height project a sub-pixel shadow that
    // doesn't survive the lit-side BRDF anyway. Saves 20 cubemap reads
    // per fragment on the bright cap of the disk.
    if geo_n_dot_l > 0.0 {
        let shadow_strength = 1.0 - smoothstep(0.5, 0.7, geo_n_dot_l);
        if shadow_strength > 0.001 {
            let sh = self_shadow(sample_dir, light_dir_local);
            sun_r = sun_r * mix(1.0, sh, shadow_strength);
        }
    }
    sun_r = sun_r * eclipse_factor(params.scene, hit, sun_dir_ws);

    // Secondary: planetshine from the orbital parent.
    //
    // Physical model: the parent reflects sunlight back at the moon as a
    // finite-angular-radius disk. `planetshine_sample_uniform` returns the
    // direction and arriving flux; we feed the same Hapke BRDF with that
    // direction so the moon's night side picks up a photographically
    // faithful dim glow when the parent is "full" overhead.
    var shine_rgb = vec3<f32>(0.0);
    let shine = planetshine_sample(params.scene, hit, sun_dir_ws, sun_flux);
    if shine.enabled {
        let shine_n_dot_l = dot(shading_normal, shine.dir);
        let shine_cos_phase = dot(view_dir, shine.dir);
        let shine_r = hapke_brdf(max(shine_n_dot_l, 0.0), n_dot_v, shine_cos_phase, surface_roughness);
        shine_rgb = shine.tint * shine_r * shine.flux;
    }

    // Combine. Hapke's r is a radiance factor; the prior pipeline used a
    // Lambert `/PI` normalization we now fold into a global scale so
    // existing flux values don't need re-tuning.
    let hapke_scale: f32 = SCENE_FLUX_SCALE;
    var sun_rgb = vec3<f32>(sun_r * sun_flux * hapke_scale);
    var ambient_term = vec3<f32>(params.scene.ambient_intensity);
    if params.fullbright >= 0.5 {
        // Collapse direct-light contribution so `lit = albedo` everywhere.
        // Atmosphere/Rayleigh/clouds still shade downstream so surface detail
        // is readable without losing atmosphere authoring cues.
        sun_rgb = vec3<f32>(1.0);
        shine_rgb = vec3<f32>(0.0);
        ambient_term = vec3<f32>(0.0);
    }
    var lit = albedo * (sun_rgb + shine_rgb + ambient_term);

    // ── Water shading branch ────────────────────────────────────────────
    //
    // Where the filtered height sits below sea level (encoded midpoint at
    // 0 m in `sample_height_m`), replace terrain lighting with a
    // Cook-Torrance water BRDF. The smoothstep gives a soft coastline at
    // the height cube's bilinear-filter scale — no separate coastline mask
    // needed.
    if params.sea_level_m > -1.0e8 {
        let height_above_sea_m = sample_height_m(sample_dir) - params.sea_level_m;
        let water_depth_m = -height_above_sea_m;
        let water_t = smoothstep(-1.0, 1.0, water_depth_m) * (1.0 - ice_overlay.coverage);
        if water_t > 0.0 {
            var water_lit = vec3<f32>(0.0);
            if params.fullbright >= 0.5 {
                let water_n_dot_v = max(dot(normal, view_dir), 0.0);
                water_lit = water_column_color(water_depth_m, water_n_dot_v);
            } else {
                // Sky tint for grazing-angle reflection. β_R · H_R is
                // the per-channel vertical optical depth (= Rayleigh τ_v
                // at zenith); the multiplier matches the prior visual
                // calibration. Airless bodies have β = 0 → black sky
                // reflection, which is physically correct for vacuum.
                let sky_tint = atmosphere.rayleigh_beta_h.xyz
                    * atmosphere.rayleigh_beta_h.w
                    * atmosphere.atmos_geom.z
                    * 3.0;
                water_lit = shade_water(
                    normal,
                    view_dir,
                    sun_dir_ws,
                    water_depth_m,
                    sun_flux,
                    params.scene.ambient_intensity,
                    sky_tint,
                    hit,
                );
            }
            lit = mix(lit, water_lit, water_t);
        }
    }

    // ── Atmospheric scattering raymarch ─────────────────────────────────
    //
    // One numeric integration along the view ray produces both the
    // per-channel transmittance from camera to surface (aerial
    // perspective + sun column attenuation) and the in-scattered
    // radiance reaching the camera (the daylight haze, the lit-limb
    // halo, the terminator orange band).
    //
    // The raymarch is run BEFORE cloud compositing so the surface is
    // already darkened by the sun-column transmission when clouds
    // cast their own shadow on it. Clouds themselves are not
    // multiplied by `transmittance` — they sit above most of the
    // atmosphere mass, and applying the surface-path transmittance
    // to them would over-dim them at the limb (see atmosphere.wgsl
    // header for the discussion). The in-scatter is added on top of
    // the surface+cloud composite so it correctly hazes both.
    let shell_body = atmosphere_shell_hit(cam_pos, ray_dir, in.sphere_center);
    var scatter_body: ScatterResult;
    if shell_body.valid {
        let jitter_b = atmosphere_jitter(in.clip_position.xy);
        scatter_body = integrate_atmosphere(
            cam_pos, ray_dir, in.sphere_center, sun_dir_ws,
            sun_flux * SCENE_FLUX_SCALE,
            shell_body.t_enter, t,
            params.radius, atmosphere, jitter_b,
        );
    } else {
        // Vacuum / airless: identity transmittance, zero in-scatter.
        scatter_body = ScatterResult(vec3<f32>(0.0), vec3<f32>(1.0));
    }
    lit = lit * scatter_body.transmittance;

    // ── Cloud layer ─────────────────────────────────────────────────────
    //
    // Main cloud layer is a reference density cubemap. Drift over sim
    // time is reintroduced by the banded sampler below, which rotates
    // each latitude band independently.
    //
    // Cloud-shell intersection. Clouds live on a shell at a slight
    // altitude above the surface (~0.15 % of body radius ≈ 9 km on a
    // 6000 km body). Using THIS intersection point for the cloud
    // sample — rather than the surface sample — introduces visible
    // parallax at grazing viewing angles: the same cloud mass appears
    // displaced outward from the terrain below it, the dominant
    // perceptual cue that clouds float above the surface.
    let cloud_altitude = params.radius * 0.0015;
    let cloud_r = params.radius + cloud_altitude;
    let c_cloud = dot(oc, oc) - cloud_r * cloud_r;
    let disc_cloud = half_b * half_b - c_cloud;
    var cloud_sample_dir = sample_dir;
    if disc_cloud > 0.0 {
        let t_cloud = -half_b - sqrt(disc_cloud);
        if t_cloud > 0.0 {
            let cloud_hit = cam_pos + t_cloud * ray_dir;
            let cloud_normal_ws = normalize(cloud_hit - in.sphere_center);
            cloud_sample_dir = rotate_quat(params.orientation, cloud_normal_ws);
        }
    }

    // Main density + shadow probe both go through the banded
    // rotation sampler: each of the K bands carries its own phase
    // wrapped mod TAU on the CPU, so per-band sampling is always
    // seamless; differential rotation emerges from sampling the two
    // bands bracketing each fragment's latitude and blending by its
    // position in sin²(lat). See `sample_cloud_banded` below.
    let main_cloud_density = sample_cloud_banded(cloud_sample_dir);

    // Shadow probe: offset the SURFACE direction toward the sun
    // (0.018 rad ≈ 100 km on a 6000 km body) then run the same
    // banded sampler. This reads "what cloud sits between this
    // terrain pixel and the sun". `composite_clouds` only consumes
    // the result inside its `raw_ndl > -0.10` branch, so skip the
    // cubemap fetch on night-side fragments — the value isn't read.
    var shadow_cloud_density: f32 = 0.0;
    if geo_n_dot_l > -0.10 {
        let shadow_offset = 0.018;
        let shadow_dir_raw = normalize(sample_dir + light_dir_local * shadow_offset);
        shadow_cloud_density = sample_cloud_banded(shadow_dir_raw);
    }

    lit = composite_clouds(
        lit,
        normal,
        sun_dir_ws,
        sun_flux * hapke_scale,
        params.scene.ambient_intensity,
        atmosphere,
        main_cloud_density,
        shadow_cloud_density,
    );

    // ── Atmospheric in-scatter + artistic limb darkening ────────────────
    //
    // The in-scatter is what carries the daylight blue haze, the
    // terminator orange band, and the Mie forward-scattered haze near
    // the sun direction. Added on top of the surface+cloud composite so
    // it hazes both correctly.
    lit = lit + scatter_body.in_scatter;
    // Pure-artistic limb darkening (Minnaert per-channel). Most
    // terrestrial bodies leave this at zero strength; gas-giant-style
    // bodies use it to round the disk past what scattering already
    // gives them.
    lit = apply_limb_darkening(
        lit,
        n_dot_v,
        atmosphere.limb_exponents.xyz,
        atmosphere.limb_exponents.w,
    );

    // Correct depth.
    let hit_clip = view.clip_from_world * vec4(hit, 1.0);
    let depth    = hit_clip.z / hit_clip.w;

    return FragOutput(vec4(lit, 1.0), depth);
#endif
}
