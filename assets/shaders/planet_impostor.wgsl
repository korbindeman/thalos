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
#import thalos::atmosphere::{
    AtmosphereBlock,
    RimHit,
    rim_halo_contribution,
    apply_terminator_warmth,
    apply_fresnel_rim,
    apply_limb_darkening,
    apply_rayleigh_ground_transmission,
    apply_rayleigh_inscatter,
    atmosphere_is_active,
    composite_clouds,
    rotate_cloud_dir_local,
    cloud_band_phase,
    rotate_around_y,
    CLOUD_BAND_COUNT,
}

const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

const CELL_TABLE_SIZE: u32 = 8192u;
const CELL_TABLE_MASK: u32 = 8191u;

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
    // Shared scene-lighting description: stars, ambient, eclipse occluders,
    // planetshine parent. Mirror of `thalos::lighting::SceneLighting`.
    scene:           SceneLighting,
    // Sea-level elevation (m, same encoding as the height cubemap). The
    // water BRDF fires where `sample_height_m(dir) < sea_level_m`. Airless
    // bodies set this to a large negative sentinel so the threshold is
    // never crossed.
    sea_level_m:     f32,
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
//   material_id:   index into `materials` — reserved for a future rim
//                  material override. The material cube is currently the
//                  primary source for surface material.
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

struct Material {
    albedo:    vec3<f32>,
    roughness: f32,
}

@group(3) @binding(0)  var<uniform> params:          PlanetParams;
@group(3) @binding(1)  var          albedo_tex:      texture_cube<f32>;
@group(3) @binding(2)  var          albedo_sampler:  sampler;
@group(3) @binding(3)  var          height_tex:      texture_cube<f32>;
@group(3) @binding(4)  var          height_sampler:  sampler;
@group(3) @binding(5)  var<uniform> detail:          PlanetDetail;
// NOTE: binding 6 is a `texture_2d_array<u32>` with 6 layers, NOT a
// `texture_cube<u32>`. WGSL/naga has no `textureLoad` overload for cube
// textures, and integer-format textures cannot be filtered (so
// `textureSample` is not an option either). A 2D array with one layer per
// cubemap face is the idiomatic way to expose an R8Uint "cube" to a fragment
// shader, and `textureLoad(tex, xy, layer, lod)` maps cleanly onto it.
//
// Agent F: upload the material cubemap as an R8Uint 2D array, 6 layers, in
// the canonical PosX, NegX, PosY, NegY, PosZ, NegZ order that `CubemapFace`
// already uses. The sampler at binding 7 is still required by the bind group
// layout contract but is unused — declare it as a non-filtering sampler.
@group(3) @binding(6)  var          material_cube:   texture_2d_array<u32>;
@group(3) @binding(7)  var          material_sampler: sampler;
@group(3) @binding(8)  var<storage, read> craters:     array<Crater>;
@group(3) @binding(9)  var<storage, read> cell_index:  array<CellRange>;
@group(3) @binding(10) var<storage, read> feature_ids: array<u32>;
@group(3) @binding(11) var<storage, read> materials:   array<Material>;
// Optional atmosphere layer — see `thalos::atmosphere`. For bodies with
// no atmosphere (Mira, Ignis, …) every layer's intensity scalar is zero
// and the atmosphere path is effectively skipped.
@group(3) @binding(12) var<uniform> atmosphere:      AtmosphereBlock;
// Baked cloud-cover cubemap (R8Unorm). Produced by `thalos_cloud_gen`
// via Wedekind curl-noise warp advection. For airless bodies this is a
// 1×1 blank cube; the cloud path gates on `cloud_albedo_coverage.w > 0`
// so those bodies pay just the branch cost.
@group(3) @binding(13) var          cloud_cover_tex: texture_cube<f32>;
@group(3) @binding(14) var          cloud_cover_sampler: sampler;

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
    // beyond the inscribed circle — and the rim-halo shell, which lives
    // at altitudes up to `atmosphere.rim_shape.y`, gets scissored off
    // there. Sizing from the outer shell radius keeps the halo visible
    // all the way around. Airless bodies have `rim_shape.y == 0`, so
    // this collapses to the original formula.
    let effective_radius = params.radius + atmosphere.rim_shape.y;
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

fn complex_depth_ratio(d_over_dsc: f32) -> f32 {
    let t = exp(-max(d_over_dsc - 1.0, 0.0) / 3.0);
    return COMPLEX_MIN_DEPTH_RATIO + (SIMPLE_DEPTH_RATIO - COMPLEX_MIN_DEPTH_RATIO) * t;
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
// is in [0,1] (1 = pristine, 0 = mature) — older craters keep about half
// of the contrast a fresh crater has, matching the Pass 1.5 CPU path in
// `space_weather.rs`.
fn crater_albedo_delta(t: f32, freshness: f32) -> f32 {
    let strength = 0.55 + 0.45 * freshness;
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
    // Fade window 0.5 – 8 px so sub-pixel features still contribute
    // statistically to surface shading — real-Moon density comes from the
    // *population* of barely-resolved craters, not just those big enough to
    // cast individual shadows.
    let weight = smoothstep(0.5, 8.0, diameter_px);
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

    // Morphology branch: d/d_sc decides simple vs complex profile. The baker
    // gave us `depth_m` and `rim_height_m` already; we feed them in directly
    // rather than re-deriving from ratios.
    let depth = crater.depth_m;
    let rim = crater.rim_height_m;
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
                    let delta_h = crater.rim_height_m - h_m;
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
    // and `apply_ssbo_crater` fades in with `smoothstep(0.5, 8.0, diameter_px)`.
    // If even the largest possible crater is below 0.5 px, no crater in the
    // layer can contribute — skip the 27-cell iteration entirely.
    let max_diameter_m = 2.0 * detail.cubemap_bake_threshold_m;
    if max_diameter_m < 0.5 * pixel_size_m {
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

fn regional_albedo_mod(p: vec3<f32>) -> f32 {
    let s = f32(detail.seed_lo & 0xFFFFu) * (1.0 / 65535.0);
    let phase = vec3<f32>(s * 6.283, s * 4.193, s * 2.711);
    let low =
          sin(p.x * 2.3 + phase.x + 0.7)
        * sin(p.y * 2.1 + phase.y + 1.3)
        * sin(p.z * 2.7 + phase.z + 2.1);
    let mid =
          sin(p.x * 5.1 + phase.y + 3.0)
        * sin(p.y * 4.7 + phase.z + 0.5)
        * sin(p.z * 5.9 + phase.x + 1.7);
    return 1.0 + 0.12 * low + 0.06 * mid;
}

// ── Normal perturbation from height cubemap ────────────────────────────────
//
// Finite-difference normals derived from the height cubemap.  Samples 4
// neighbors offset by a small angle in the tangent plane.  No pre-baked
// slope texture needed — this automatically reflects whichever frequency
// bands are stored in the cubemap.

fn perturb_normal_from_height(n: vec3<f32>) -> vec3<f32> {
    let res = f32(textureDimensions(height_tex).x);
    if res < 2.0 {
        return n;
    }

    // Build a continuous orthonormal tangent frame on the sphere.
    // Duff et al. 2017 ("Building an Orthonormal Basis, Revisited") — branchless,
    // continuous everywhere except n.z = -1. A `select` on |n.y| > 0.99 (the
    // previous approach) flips the tangent ~90° at latitude ±82°, which swaps
    // which axis the finite-difference gradient reads from and bisects any
    // feature spanning that latitude.
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let tangent   = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bitangent = vec3<f32>(b, s + n.y * n.y * a, -n.y);

    // Offset ~1.5 texels on the cubemap.
    let offset = 1.5 / res;

    // Sample the canonical surface height (baked cubemap + fbm jitter)
    // at four neighbours. Auto-LOD on the cubemap part keeps far
    // bodies' normals stable (preserves the original mip-blur on the
    // baked height); the fbm contribution rides on top so all three
    // consumers — water mask, self-shadow, normals — agree on the
    // perturbed surface even though the cubemap LOD selection
    // differs between them.
    let h_e = sample_height_baked_auto_lod_m(n + tangent * offset);
    let h_w = sample_height_baked_auto_lod_m(n - tangent * offset);
    let h_n = sample_height_baked_auto_lod_m(n + bitangent * offset);
    let h_s = sample_height_baked_auto_lod_m(n - bitangent * offset);

    // Convert texel offset to world-space distance on the body surface.
    let ds = detail.body_radius_m * offset * 2.0;
    if ds < 1e-6 {
        return n;
    }

    // h_* are already in meters; central-difference slope is m / m.
    let dh_dt = (h_e - h_w) / ds;
    let dh_db = (h_n - h_s) / ds;

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

// ── Material cubemap lookup ────────────────────────────────────────────────
//
// The material cube is exposed as a 6-layer `texture_2d_array<u32>` so that
// `textureLoad` (nearest-neighbor, integer payload) works. We translate a
// unit direction to (face, x, y) ourselves — the face layout matches
// `terrain_gen::cubemap::dir_to_face_uv`:
//
//   layer 0: +X    layer 1: -X    layer 2: +Y
//   layer 3: -Y    layer 4: +Z    layer 5: -Z
//
// with u ranging right, v ranging down within each face.
fn sample_material_id(dir: vec3<f32>) -> u32 {
    let ax = abs(dir.x);
    let ay = abs(dir.y);
    let az = abs(dir.z);

    var face: i32;
    var sc: f32;
    var tc: f32;
    var ma: f32;

    if ax >= ay && ax >= az {
        ma = ax;
        if dir.x > 0.0 {
            face = 0;
            sc = -dir.z;
            tc = -dir.y;
        } else {
            face = 1;
            sc = dir.z;
            tc = -dir.y;
        }
    } else if ay >= az {
        ma = ay;
        if dir.y > 0.0 {
            face = 2;
            sc = dir.x;
            tc = dir.z;
        } else {
            face = 3;
            sc = dir.x;
            tc = -dir.z;
        }
    } else {
        ma = az;
        if dir.z > 0.0 {
            face = 4;
            sc = dir.x;
            tc = -dir.y;
        } else {
            face = 5;
            sc = -dir.x;
            tc = -dir.y;
        }
    }

    let u = 0.5 * (sc / ma + 1.0);
    let v = 0.5 * (tc / ma + 1.0);

    let dims = textureDimensions(material_cube);
    let res = i32(dims.x);
    let x = clamp(i32(u * f32(res)), 0, res - 1);
    let y = clamp(i32(v * f32(res)), 0, res - 1);

    return textureLoad(material_cube, vec2<i32>(x, y), face, 0).r;
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

// Canonical surface height in meters: bake (sampled at the warped
// direction) + height jitter. This is the function the future 3D
// mesher must reproduce at the iso-contour to keep the LOD handoff
// continuous. Currently only consumed by the water-mask test —
// that's where the iso-contour lives, and the only place the cost
// of evaluating warp + jitter (4 fbm calls) is justified.
//
// Iso-contour cull: read the bare cubemap height first; when it sits
// well clear of sea level, the canonical warp + jitter cannot push
// the result across the smoothstep band at the call site, so skip
// the four fbm evaluations and return the bare value. Authored
// `coastline_jitter_amp_m` is ~30 m on water bodies; the warp arc is
// ~5 km on a 6 Mm body — a band of `10 × jitter_amp + 100 m` covers
// the worst plausible perturbation. Far from a coast (most of the
// disk), this is the hot path. Airless bodies already short-circuit
// inside `coastline_warp_dir` / `coastline_jitter_m`, so the explicit
// cull below is a no-op for them.
fn sample_height_m(dir: vec3<f32>) -> f32 {
    let bare_stored = textureSampleLevel(height_tex, height_sampler, dir, 0.0).r;
    let bare_m = (bare_stored - 0.5) * 2.0 * params.height_range;
    let bare_above_sea = bare_m - params.sea_level_m;
    let band = params.coastline_jitter_amp_m * 10.0 + 100.0;
    if abs(bare_above_sea) > band {
        return bare_m;
    }

    let warped = coastline_warp_dir(dir);
    let stored = textureSampleLevel(height_tex, height_sampler, warped, 0.0).r;
    let baked_m = (stored - 0.5) * 2.0 * params.height_range;
    return baked_m + coastline_jitter_m(warped);
}

fn self_shadow(sample_dir: vec3<f32>, light_dir_local: vec3<f32>) -> f32 {
    let radius_m = detail.body_radius_m;
    // Bare baked height — the canonical high-freq band would only
    // contribute sub-texel shadows below the impostor's visible
    // shadow scale, and 21 fbm evaluations per ray is too costly.
    let h0 = sample_height_baked_m(sample_dir);

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
        let h = sample_height_baked_m(d);
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
// Inputs are all in the same space; handedness doesn't matter.
fn hapke_brdf(n_dot_l: f32, n_dot_v: f32, cos_phase: f32) -> f32 {
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
    // Matches the Moon's ~40% surge at α=0 with h ≈ 0.06 (~3.4°).
    let B0: f32 = 1.0;
    let h:  f32 = 0.06;
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
//   - Subsurface diffuse: shallow water shows the seabed through a clear
//     column; deeper water replaces it with the absorption-tinted column
//     colour over a 30 m e-folding scale. Below ~150 m the seabed is
//     fully obscured.
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

fn shade_water(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    depth_m: f32,
    sun_flux: f32,
    ambient: f32,
    sky_tint: vec3<f32>,
    seabed: vec3<f32>,
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

    // Subsurface colour: blend seabed visibility with absorbed water
    // column. Shallow water shows the baked seabed (so dark-side coastal
    // shelves stay visible at the same ambient brightness as the land);
    // deep water replaces it with the column colour.
    let shallow = vec3<f32>(0.10, 0.35, 0.42);
    let deep = vec3<f32>(0.005, 0.02, 0.06);
    let column = mix(shallow, deep, 1.0 - exp(-max(depth_m, 0.0) / 200.0));
    let seabed_visibility = exp(-max(depth_m, 0.0) / 30.0);
    let subsurface = mix(column, seabed, seabed_visibility);

    let f_nv = f0 + (1.0 - f0) * pow(max(1.0 - n_dot_v, 0.0), 5.0);

    // Ambient: Fresnel-modulated sky reflection + subsurface diffuse,
    // both on the same `ambient` scale so dark-side water sits at the
    // same brightness as the dark-side seabed it replaces.
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

fn sample_rim_halo(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    light: PrimaryLight,
) -> RimHit {
    return rim_halo_contribution(
        cam_pos, ray_dir, center, light.dir_ws,
        params.radius,
        atmosphere.rim_shape.y,
        atmosphere.rim_shape.x,
        atmosphere.rim_color_intensity.xyz,
        atmosphere.rim_color_intensity.w,
    );
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
    let rim = sample_rim_halo(cam_pos, ray_dir, in.sphere_center, light);
    if rim.opacity <= 0.001 {
        discard;
    }
    let color = rim.contribution * light.flux * (1.0 / (4.0 * PI));
    // Closest-approach depth gives a sensible silhouette depth for
    // halo fragments — used so opaque objects in front of the halo
    // (and other halos sorted closer) depth-test correctly.
    let closest_point = cam_pos + ray_dir * max(-half_b, 0.0);
    let clip = view.clip_from_world * vec4(closest_point, 1.0);
    return FragOutput(vec4(color, rim.opacity), clip.z / clip.w);
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

    // ── Layer 1a: material cube → primary albedo/roughness ─────────────────
    let mat_id = sample_material_id(sample_dir);
    let mat_count = arrayLength(&materials);
    var mat_albedo = vec3<f32>(0.5);
    var mat_roughness = 1.0;
    if mat_count > 0u {
        let idx = min(mat_id, mat_count - 1u);
        let m = materials[idx];
        mat_albedo = m.albedo;
        mat_roughness = m.roughness;
    }

    // Keep the RGBA8 albedo cube as a tint fallback.  The material palette is
    // the primary source, but the baked albedo still carries any per-texel
    // variation a stage painted (splotches, basin rays, etc.).
    let baked_tint = textureSample(albedo_tex, albedo_sampler, sample_dir).rgb;
    let regional = clamp(regional_albedo_mod(sample_dir), 0.7, 1.3);
    // Multiplicative tint composition.
    //   baked_tint = 0.5 → tint_mult = 1.0  (legacy neutral, identical to old formula)
    //   baked_tint = 0.0 → tint_mult = 0.0  (was 0.5 × mat under old `mix(1, …, 0.5)`)
    //   baked_tint = 1.0 → tint_mult = 2.0  (was 1.5 × mat — gains chromatic headroom)
    // The full 0..2 range lets stages that bake real chromatic colours into
    // the albedo cube (PaintBiomes) drive the surface colour through the
    // FILTERABLE texture path, which the discrete material-id lookup can
    // never do — biome polygons go away because the GPU's bilinear filter
    // smooths boundary transitions automatically.
    let tint_mod = baked_tint * 2.0;
    let baked_albedo = mat_albedo * tint_mod * regional;

    // ── Layer 1b: height-derived normal perturbation ─────────────────────
    // Kept in body-local space until after the SSBO/synthesis crater
    // gradients are combined, then rotated back to world space for lighting.
    var shading_normal = perturb_normal_from_height(sample_dir);

    // ── Pixel size in meters ────────────────────────────────────────────
    // Carried from the vertex stage; see vertex() for the derivation.
    let pixel_size_m = in.pixel_size_m;

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
    if length(ssbo_grad) > 0.0 {
        let grad_tangent = ssbo_grad - dot(ssbo_grad, sample_dir) * sample_dir;
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
    let albedo = baked_albedo * (1.0 + fresh_boost + crater_mod);
    // mat_roughness: reserved for a future BRDF upgrade; Lambertian for now.

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
    var sun_r = hapke_brdf(max(sun_n_dot_l, 0.0), n_dot_v, cos_phase_sun);

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
        let shine_r = hapke_brdf(max(shine_n_dot_l, 0.0), n_dot_v, shine_cos_phase);
        shine_rgb = shine.tint * shine_r * shine.flux;
    }

    // Combine. Hapke's r is a radiance factor; the prior pipeline used a
    // Lambert `/PI` normalization we now fold into a global scale so
    // existing flux values don't need re-tuning.
    let hapke_scale: f32 = 0.5;
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
    // 0 m in `sample_height_m`), overlay a Cook-Torrance water BRDF on
    // top of the Hapke-shaded seabed. The smoothstep gives a soft
    // coastline at the height cube's bilinear-filter scale — no separate
    // coastline mask needed.
    let height_above_sea_m = sample_height_m(sample_dir) - params.sea_level_m;
    let water_depth_m = -height_above_sea_m;
    let water_t = smoothstep(-1.0, 1.0, water_depth_m);
    if water_t > 0.0 {
        // Sky tint for grazing-angle reflection. Rayleigh τ carries the
        // per-channel atmosphere tint cue (blue > green > red on Earth-
        // like). Airless bodies have rayleigh = 0 → black sky reflection,
        // which is physically correct for vacuum.
        let sky_tint = atmosphere.rayleigh.xyz * atmosphere.rayleigh.w * 3.0;
        let water_lit = shade_water(
            normal,
            view_dir,
            sun_dir_ws,
            water_depth_m,
            sun_flux,
            params.scene.ambient_intensity,
            sky_tint,
            albedo,
            hit,
        );
        lit = mix(lit, water_lit, water_t);
    }

    // ── Rayleigh ground transmission ────────────────────────────────────
    //
    // Attenuate the lit surface by the per-channel optical depth of the
    // sun's column to the ground. At the terminator the sun's path is
    // an order of magnitude longer than at noon — blue is scattered
    // out, so the surface is lit by progressively redder light. This is
    // the physical cause of sunset colouring on the ground; previously
    // approximated with a flat "terminator warmth" tint.
    //
    // Applied BEFORE cloud compositing so the surface already carries
    // the transmitted colour when clouds darken it with cast shadows.
    lit = apply_rayleigh_ground_transmission(
        lit,
        geo_n_dot_l,
        atmosphere.rayleigh.xyz,
        atmosphere.rayleigh.w,
    );

    // ── Cloud layer ─────────────────────────────────────────────────────
    //
    // Main cumulus layer is a baked cubemap produced by
    // `thalos_cloud_gen` (Wedekind curl-warp advection) — density
    // lives on the sphere, no live fBm evaluation. Drift over sim
    // time is reintroduced here by rotating the sample direction via
    // `rotate_cloud_dir_local` (equator-fastest, decision 1.B).
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
        cloud_sample_dir,
        normal,
        sun_dir_ws,
        sun_flux * hapke_scale,
        params.scene.ambient_intensity,
        atmosphere,
        main_cloud_density,
        shadow_cloud_density,
    );

    // ── Atmosphere composition ──────────────────────────────────────────
    //
    // Applied on top of the Hapke-lit surface. Each helper early-outs on
    // a zero scalar so bodies without a `terrestrial_atmosphere` block
    // cost nothing here.
    //
    // Order: limb darkening first (it multiplies the lit colour before
    // any rim tints fold in), then terminator warmth, then Fresnel rim.
    //
    // The rim halo is NOT added on surface-hit fragments — only on miss
    // rays (outside the silhouette). Adding it on hit was the cause of
    // two visible bugs:
    //   1. Night side never went dark. The halo's `sun_factor` bottoms
    //      out at 0.5, so every hit fragment picked up a uniform blue
    //      tint no matter which hemisphere the ray terminated on.
    //   2. A bright singular point at the screen-space centre of the
    //      planet. A ray aimed exactly at the body centre has
    //      `closest = oc + ray_dir * -half_b = 0`, and `normalize(0)`
    //      is undefined — the NaN propagated into `sun_factor` and
    //      lit a pinprick pixel.
    //
    // A proper in-scatter integral (Rayleigh scattering along the
    // column from camera to surface, sun-gated per sample) would
    // restore a physical "atmosphere over surface" contribution; that
    // is a follow-up. Limb shading below already carries the dominant
    // "blue atmosphere on the lit disk" cue.
    lit = apply_limb_darkening(
        lit,
        n_dot_v,
        atmosphere.limb_exponents.xyz,
        atmosphere.limb_exponents.w,
    );
    // Rayleigh view-path in-scatter. At the sub-solar point this adds
    // the familiar blue daylight haze (β_blue >> β_red, short sun
    // column so T_sun is still mostly white, blue wins the β · T
    // product). At the terminator the long sun column wipes blue
    // entirely and the orange/red residue dominates — the in-scatter
    // band visible at the limb in every orbital sunset photograph.
    lit = apply_rayleigh_inscatter(
        lit,
        geo_n_dot_l,
        n_dot_v,
        atmosphere.rayleigh.xyz,
        atmosphere.rayleigh.w,
        sun_flux * hapke_scale,
    );
    // Legacy tint helpers — retained for artistic control on bodies
    // without a full Rayleigh setup. With Rayleigh authored, both
    // `terminator_warmth` and `fresnel_rim` are zero-strength on the
    // relevant bodies so they cost only the early-return.
    lit = apply_terminator_warmth(
        lit,
        geo_n_dot_l,
        atmosphere.terminator_warmth.xyz,
        atmosphere.terminator_warmth.w,
    );
    lit = apply_fresnel_rim(
        lit,
        geo_n_dot_l,
        n_dot_v,
        atmosphere.fresnel_rim.xyz,
        atmosphere.fresnel_rim.w,
    );

    // Correct depth.
    let hit_clip = view.clip_from_world * vec4(hit, 1.0);
    let depth    = hit_clip.z / hit_clip.w;

    return FragOutput(vec4(lit, 1.0), depth);
#endif
}
