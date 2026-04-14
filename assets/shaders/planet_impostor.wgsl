// Planet sphere impostor.
//
// Renders a camera-facing quad (billboard) whose fragment shader ray-traces a
// sphere of the correct radius.  Every pixel gets the mathematically exact
// surface normal, giving a perfectly smooth silhouette at any resolution.
//
// Surface detail comes from three layered sources (the sample.rs LOD contract):
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
//   3. Shader-hash detail (< 500 m) — statistical small-crater synthesis via a
//      hashed 3D cell grid on the unit sphere. No identity; pure statistical
//      tail handed off at 500 m from layer 2.
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
//   hash function:    `hash_cell(ix, iy, iz, octave=0u)` — same primes as the
//                     shader-hash layer, so any future SFD continuity across
//                     layers uses one code path. The result is masked with
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
    radius:            f32,
    light_intensity:   f32,
    ambient_intensity: f32,
    height_range:      f32,
    light_dir:         vec4<f32>,
    // Quaternion (xyzw) rotating world-space directions into body-local space
    // where the cubemaps were baked. Identity = no rotation.
    orientation:       vec4<f32>,
    // Eclipse occluders: analytical sphere-shadow test against other bodies.
    occluder_count:    u32,
    _pad0:             u32,
    _pad1:             u32,
    _pad2:             u32,
    // xyz = world render-space center, w = render-unit radius.
    occluders:         array<vec4<f32>, 8>,
    // Planetshine: parent body for secondary illumination.
    parent_pos:        vec4<f32>,  // xyz = render pos, w = render radius
    parent_tint:       vec4<f32>,  // xyz = bond_albedo × tint, w = enable flag
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

    let d      = length(cam_pos - sphere_center);
    let d_safe = max(d, params.radius * 1.0001);
    let billboard_radius = params.radius * d_safe
        / sqrt(d_safe * d_safe - params.radius * params.radius);

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

fn u32_to_unit(x: u32) -> f32 {
    return f32(x) / 4294967296.0;
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

fn sample_diameter(u: f32, d_lo: f32, d_hi: f32, alpha: f32) -> f32 {
    let lo = pow(d_lo, -alpha);
    let hi = pow(d_hi, -alpha);
    let y = lo + (hi - lo) * u;
    return pow(y, -1.0 / alpha);
}

// ── Per-cell crater accumulator (shared by SSBO and hash layers) ───────────

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
    // (positive). The SSBO and hash layers iterate craters below the cubemap
    // bake threshold, so their craters carry no CPU-painted albedo and need
    // an analytic equivalent here.
    albedo_mod: f32,
}

// Per-crater albedo signature shared by the SSBO and hash layers. Returns a
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
    // Fade window matches the hash layer (0.5 – 8 px) so sub-pixel features
    // still contribute statistically to surface shading — real-Moon density
    // comes from the *population* of barely-resolved craters, not just
    // those big enough to cast individual shadows.
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
    // r). Matches the hash layer; sign is handled in the gradient line below.
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
    // layer can contribute — skip the 27-cell iteration entirely. This is the
    // SSBO analog of the per-octave cull in `synthesize_small_craters`.
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

// ── Shader-hash layer: statistical small-crater synthesis (< 500 m) ────────

fn visit_cell(
    accum: ptr<function, CraterAccum>,
    p_unit: vec3<f32>,
    ix: i32,
    iy: i32,
    iz: i32,
    octave: u32,
    cell_size_unit: f32,
    d_lo: f32,
    d_hi: f32,
    expected_per_cell: f32,
    pixel_size_m: f32,
    light_dir_local: vec3<f32>,
) {
    let h = hash_cell(ix, iy, iz, octave);
    let u_exists = u32_to_unit(h);
    if u_exists >= expected_per_cell {
        return;
    }

    let u_diam = u32_to_unit(pcg(h ^ 0x68E31DA4u));
    let u_px   = u32_to_unit(pcg(h ^ 0xB5297A4Du));
    let u_py   = u32_to_unit(pcg(h ^ 0xBE5466CFu));
    let u_pz   = u32_to_unit(pcg(h ^ 0x1B873593u));
    let u_age  = u32_to_unit(pcg(h ^ 0xD2A9C4B1u));
    let u_ellip   = u32_to_unit(pcg(h ^ 0xA1C4E9F2u));
    let u_orient  = u32_to_unit(pcg(h ^ 0x3F7B8C21u));
    let u_rim_ph  = u32_to_unit(pcg(h ^ 0x9D2E5A73u));
    let u_rim_lob = u32_to_unit(pcg(h ^ 0x54F1D8B6u));

    let cell_origin = vec3<f32>(f32(ix), f32(iy), f32(iz)) * cell_size_unit;
    let cand = cell_origin + vec3<f32>(u_px, u_py, u_pz) * cell_size_unit;
    let cand_len = length(cand);
    if cand_len < 1e-6 {
        return;
    }
    let center = cand / cand_len;

    let diameter_m = sample_diameter(u_diam, d_lo, d_hi, detail.sfd_alpha);

    let diameter_px = diameter_m / max(pixel_size_m, 1e-6);
    // Same smoothstep window as the SSBO layer — shared continuity contract.
    let lod_weight = smoothstep(0.5, 4.0, diameter_px);
    if lod_weight <= 0.0 {
        return;
    }

    let radius_m = 0.5 * diameter_m;

    let cos_theta = clamp(dot(p_unit, center), -1.0, 1.0);
    let theta = acos(cos_theta);
    let s_arc_m = theta * detail.body_radius_m;
    var r = s_arc_m / radius_m;
    if r >= EJECTA_EXTENT {
        return;
    }

    let proj = center - cos_theta * p_unit;
    let proj_len2 = dot(proj, proj);
    var azimuth = 0.0;
    if proj_len2 > 1e-12 {
        let px = proj.x;
        let py = proj.y + proj.z * 0.7;
        azimuth = atan2(px, py);
    }

    let ellipticity = u_ellip * 0.2;
    let ellip_angle = u_orient * TAU;
    let ellip_factor = 1.0 + ellipticity * cos(2.0 * (azimuth - ellip_angle));
    r = r / ellip_factor;

    let rim_lobes = floor(u_rim_lob * 4.0 + 3.0);
    let rim_phase = u_rim_ph * TAU;
    var rim_irregular = 0.0;
    if r > 0.85 && r < 1.15 {
        let wave = sin(rim_lobes * azimuth + rim_phase);
        let band = max(0.0, 1.0 - 4.0 * (r - 1.0) * (r - 1.0));
        rim_irregular = 0.35 * wave * band;
    }

    let d_over_dsc = diameter_m / detail.d_sc_m;
    let depth_ratio = select(SIMPLE_DEPTH_RATIO, complex_depth_ratio(d_over_dsc), d_over_dsc >= 1.0);
    let depth = diameter_m * depth_ratio;
    let rim = diameter_m * SIMPLE_RIM_RATIO;

    var hd: vec2<f32>;
    if d_over_dsc >= 1.0 {
        hd = complex_profile(r, depth, rim);
    } else {
        hd = simple_profile(r, depth, rim);
    }
    let h_m = hd.x + rim_irregular * rim;
    let dh_dr = hd.y;

    let age_gyr = u_age * detail.body_age_gyr;
    let age_blend = smoothstep(0.0, FRESH_AGE_GYR, age_gyr);
    let fresh_m = fresh_crater_maturity(r);
    let aged_m = mix(fresh_m, 1.0, age_blend);
    let weighted_m = mix(1.0, aged_m, lod_weight);

    let freshness_h = 1.0 - aged_m;
    let albedo_delta_lw = crater_albedo_delta(r, freshness_h) * lod_weight;

    let grad_proj_len = sqrt(proj_len2);
    if grad_proj_len < 1e-8 {
        (*accum).height = (*accum).height + h_m * lod_weight;
        (*accum).min_maturity = min((*accum).min_maturity, weighted_m);
        (*accum).albedo_mod = (*accum).albedo_mod + albedo_delta_lw;
        return;
    }
    let t_hat = proj / grad_proj_len;

    let grad = -(dh_dr) / radius_m * t_hat;

    (*accum).grad_tangent = (*accum).grad_tangent + grad * lod_weight;
    (*accum).height = (*accum).height + h_m * lod_weight;
    (*accum).min_maturity = min((*accum).min_maturity, weighted_m);
    (*accum).albedo_mod = (*accum).albedo_mod + albedo_delta_lw;

    // Per-crater analytical shadow — covers interior and ejecta blanket.
    // See `apply_ssbo_crater` for derivation. Synthesis rim peak is `rim`.
    let sin_sun = dot(light_dir_local, p_unit);
    if sin_sun > 0.0 {
        let sun_tangent = light_dir_local - sin_sun * p_unit;
        let cos_sun = length(sun_tangent);
        if cos_sun > 1e-4 {
            let sun_hat = sun_tangent / cos_sun;
            let frag_rel = -t_hat * r;
            let b = dot(frag_rel, sun_hat);
            let c = r * r - 1.0;
            let disc = b * b - c;
            if disc > 0.0 {
                let s_disc = sqrt(disc);
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
                    let rhs = t_hit * radius_m * sin_sun;
                    if lhs > rhs {
                        let margin = (lhs - rhs) / max(rhs, 1.0);
                        let s = 1.0 - clamp(margin * 8.0, 0.0, 1.0);
                        (*accum).shadow = min((*accum).shadow, mix(1.0, s, lod_weight));
                    }
                }
            }
        }
    }
}

// High-frequency statistical crater tail.  Caps `d_max_m` at 500 m so the
// shader-hash band never overlaps the SSBO band. Lower floor still obeys
// `detail.d_min_m` (production value ~80 m).
fn synthesize_small_craters(
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

    if detail.global_k_per_km2 <= 0.0 || detail.d_min_m <= 0.0 {
        return accum;
    }

    // LOD contract: shader-hash layer covers d < 500 m only.  The SSBO layer
    // owns anything larger.
    let hash_d_max = min(detail.d_max_m, 500.0);
    if hash_d_max <= detail.d_min_m {
        return accum;
    }

    let body_r = detail.body_radius_m;
    for (var oi: u32 = 0u; oi < 11u; oi = oi + 1u) {
        let d_lo = detail.d_min_m * pow(2.0, f32(oi));
        let d_hi = min(detail.d_min_m * pow(2.0, f32(oi + 1u)), hash_d_max);
        if d_hi <= d_lo {
            break;
        }

        // Whole-octave LOD cull.  If the largest diameter in this octave is
        // below half a pixel, every cell in the 3×3×3 neighborhood would
        // contribute zero weight — skip the 27 hashes entirely.
        if d_hi < 0.5 * pixel_size_m {
            continue;
        }

        let d_lo_km = d_lo * 1e-3;
        let d_hi_km = d_hi * 1e-3;
        let per_km2 = detail.global_k_per_km2
            * (pow(d_lo_km, -detail.sfd_alpha) - pow(d_hi_km, -detail.sfd_alpha));

        let cell_size_m = 2.0 * d_hi;
        let cell_area_km2 = (cell_size_m * 1e-3) * (cell_size_m * 1e-3);

        let expected_per_cell = per_km2 * cell_area_km2 / 3.0;
        if expected_per_cell <= 0.0 {
            continue;
        }

        let cell_size_unit = cell_size_m / body_r;
        let inv_cell = 1.0 / cell_size_unit;
        let px_cell = p_unit.x * inv_cell;
        let py_cell = p_unit.y * inv_cell;
        let pz_cell = p_unit.z * inv_cell;
        let cx = i32(floor(px_cell));
        let cy = i32(floor(py_cell));
        let cz = i32(floor(pz_cell));

        // Adaptive neighborhood (replaces fixed 3×3×3). Max crater radius in
        // this octave is 0.5*d_hi and ejecta extends EJECTA_EXTENT×radius, so
        // a crater's influence reaches at most `HASH_INFLUENCE_CELLS` of the
        // cell edge into any neighbor. A neighbor can therefore be skipped
        // when the fragment's intra-cell position is far enough from that
        // neighbor's boundary.
        //
        //   infl = EJECTA_EXTENT * 0.5 * d_hi / cell_size_m
        //        = 2.5 * 0.5 / 2.0 = 0.625
        //
        // Average visited cells drops from 27 → ~11 (≈2.4× speedup) without
        // losing any crater visible in a 3×3×3 sweep.
        let infl: f32 = 0.625;
        let fx = px_cell - f32(cx);
        let fy = py_cell - f32(cy);
        let fz = pz_cell - f32(cz);
        let dx_lo = select(0, -1, fx < infl);
        let dx_hi = select(0,  1, fx > 1.0 - infl);
        let dy_lo = select(0, -1, fy < infl);
        let dy_hi = select(0,  1, fy > 1.0 - infl);
        let dz_lo = select(0, -1, fz < infl);
        let dz_hi = select(0,  1, fz > 1.0 - infl);

        for (var dx: i32 = dx_lo; dx <= dx_hi; dx = dx + 1) {
            for (var dy: i32 = dy_lo; dy <= dy_hi; dy = dy + 1) {
                for (var dz: i32 = dz_lo; dz <= dz_hi; dz = dz + 1) {
                    visit_cell(
                        &accum,
                        p_unit,
                        cx + dx, cy + dy, cz + dz,
                        oi + 1u, // octave 0 is reserved for the SSBO hash
                        cell_size_unit,
                        d_lo, d_hi,
                        expected_per_cell,
                        pixel_size_m,
                        light_dir_local,
                    );
                }
            }
        }
    }

    let grad_len = length(accum.grad_tangent);
    if grad_len > 2.0 {
        accum.grad_tangent = accum.grad_tangent * (2.0 / grad_len);
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

    let h_e = textureSample(height_tex, height_sampler, n + tangent * offset).r;
    let h_w = textureSample(height_tex, height_sampler, n - tangent * offset).r;
    let h_n = textureSample(height_tex, height_sampler, n + bitangent * offset).r;
    let h_s = textureSample(height_tex, height_sampler, n - bitangent * offset).r;

    // Convert texel offset to world-space distance on the body surface.
    let ds = detail.body_radius_m * offset * 2.0;
    if ds < 1e-6 {
        return n;
    }

    // Height is R16Unorm in [0, 1].  The stored encoding is
    //   stored = 0.5 + real_meters / (2 * height_range)
    // so a texture-space difference of Δstored corresponds to
    //   Δreal = Δstored · 2 · height_range meters.
    // The factor of 2 was missing previously, which halved the normal
    // perturbation and produced a noticeably softer terminator.
    let scale = (2.0 * params.height_range) / ds;
    let dh_dt = (h_e - h_w) * scale;
    let dh_db = (h_n - h_s) * scale;

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

fn sample_height_m(dir: vec3<f32>) -> f32 {
    let stored = textureSampleLevel(height_tex, height_sampler, dir, 0.0).r;
    return (stored - 0.5) * 2.0 * params.height_range;
}

fn self_shadow(sample_dir: vec3<f32>, light_dir_local: vec3<f32>) -> f32 {
    let radius_m = detail.body_radius_m;
    let h0 = sample_height_m(sample_dir);

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
        let h = sample_height_m(d);
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

// ── Eclipse (cross-body shadow) ────────────────────────────────────────────
//
// Analytical sphere-shadow test: for each occluder (other body in the scene),
// check whether the ray from the fragment toward the sun passes through the
// occluder sphere. A simple soft penumbra scales with the occluder's apparent
// radius over a small fraction — not a full solar-disk angular model, but
// enough to avoid a hard on/off eclipse edge.
//
// Inputs: `hit_ws` fragment world position (render space), `light_dir_ws`
// sun direction in world space.

fn eclipse_term(hit_ws: vec3<f32>, light_dir_ws: vec3<f32>) -> f32 {
    var factor: f32 = 1.0;
    let count = params.occluder_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let oc = params.occluders[i];
        let center = oc.xyz;
        let r = oc.w;
        if r <= 0.0 { continue; }
        let delta = center - hit_ws;
        let t = dot(delta, light_dir_ws);
        // Occluder must be on the sun-ward side of the fragment.
        if t <= 0.0 { continue; }
        // Perpendicular distance from sun ray to occluder center.
        let perp2 = dot(delta, delta) - t * t;
        let perp = sqrt(max(perp2, 0.0));
        // Soft edge: full umbra inside r, fades out to ~1.1 r.
        let penumbra = max(r * 0.1, 1.0);
        let s = smoothstep(r, r + penumbra, perp);
        factor = min(factor, s);
        if factor <= 0.0 { break; }
    }
    return factor;
}

// ── Fragment ────────────────────────────────────────────────────────────────

@fragment
fn fragment(in: VertexOutput) -> FragOutput {
    let cam_pos  = view.world_position;
    let ray_dir  = normalize(in.world_position - cam_pos);

    // Ray-sphere intersection.
    let oc      = cam_pos - in.sphere_center;
    let half_b  = dot(oc, ray_dir);
    let c       = dot(oc, oc) - params.radius * params.radius;
    let disc    = half_b * half_b - c;
    if disc < 0.0 { discard; }
    let t = -half_b - sqrt(max(disc, 0.0));
    if t < 0.0 { discard; }

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
    // Modulate the palette albedo by the baked tint (treated as a multiplier
    // around grey) and the regional low-frequency modulation.
    let tint_mod = mix(vec3<f32>(1.0), baked_tint * 2.0, 0.5);
    let baked_albedo = mat_albedo * tint_mod * regional;

    // ── Layer 1b: height-derived normal perturbation ─────────────────────
    // Kept in body-local space until after the SSBO/synthesis crater
    // gradients are combined, then rotated back to world space for lighting.
    var shading_normal = perturb_normal_from_height(sample_dir);

    // ── Pixel size in meters ────────────────────────────────────────────
    // Carried from the vertex stage; see vertex() for the derivation.
    let pixel_size_m = in.pixel_size_m;

    // ── Dark-hemisphere early-out ───────────────────────────────────────
    // Crater normal perturbation is clamped to `geo_n_dot_l + 0.05` below,
    // and the terminator wrap adds at most `roughness * 0.08`. Any fragment
    // deeper than that on the dark side is unreachable by both sources, so
    // the crater layers cannot change the final color — skip them.
    let geo_n_dot_l = dot(normal, params.light_dir.xyz);
    let wrap_slack  = params.light_dir.w * 0.08;
    let dark_side   = geo_n_dot_l < -(0.05 + wrap_slack);

    // Sun direction in body-local space — shared by crater iteration (for
    // per-crater analytical shadow) and the cubemap raymarch below.
    let light_dir_local = rotate_quat(params.orientation, params.light_dir.xyz);

    var ssbo_grad  = vec3<f32>(0.0);
    var small_grad = vec3<f32>(0.0);
    var min_maturity = 1.0;
    var crater_shadow = 1.0;
    var crater_albedo_mod = 0.0;
    if !dark_side {
        // Both SSBO craters and the procedural small-crater layer live in
        // body-local space, so sample with the rotated `sample_dir`, not the
        // world-space `normal`.
        // ── Layer 2: SSBO craters (500 m – 5 km) ─────────────────────────
        let ssbo = iterate_ssbo_craters(sample_dir, pixel_size_m, light_dir_local);

        // ── Layer 3: statistical small-crater synthesis (< 500 m) ──────
        let small = synthesize_small_craters(sample_dir, pixel_size_m, light_dir_local);

        ssbo_grad  = ssbo.grad_tangent;
        small_grad = small.grad_tangent;
        min_maturity = min(ssbo.min_maturity, small.min_maturity);
        crater_shadow = min(ssbo.shadow, small.shadow);
        crater_albedo_mod = ssbo.albedo_mod + small.albedo_mod;
    }

    // Combine mid + high frequency crater gradients. Both gradients are in
    // body-local space (their tangent frames were built around `sample_dir`),
    // so the projection and subtraction happen in body-local space; we
    // rotate the final shading normal to world space afterwards.
    var combined_grad = ssbo_grad + small_grad;
    if length(combined_grad) > 0.0 {
        let grad_tangent = combined_grad - dot(combined_grad, sample_dir) * sample_dir;
        shading_normal = normalize(shading_normal - grad_tangent);
    }
    // Transform the fully-perturbed shading normal from body-local to world
    // space so the lighting dot product below is consistent with `light_dir`.
    shading_normal = rotate_quat(conjugate_quat(params.orientation), shading_normal);
    // Fresh craters: subtle brightening.  The material palette already encodes
    // the "fresh regolith" bias, so the ad-hoc FRESH_BIAS_COLOR tint is gone.
    let fresh_boost = clamp((1.0 - min_maturity) * 0.3, 0.0, 0.4);
    // Per-crater albedo signature from the SSBO + hash layers. Clamped to
    // a sane range so a few stacked craters can't blow out the surface or
    // drive it negative. The factor parallels the CPU `space_weather.rs`
    // Pass 1.5 strength so cubemap-baked and shader-only craters look
    // consistent across the bake threshold.
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
    let sun_dir = params.light_dir.xyz;
    let sun_n_dot_l_raw = dot(shading_normal, sun_dir);
    let sun_n_dot_l = min(sun_n_dot_l_raw, geo_n_dot_l + headroom);
    let cos_phase_sun = dot(view_dir, sun_dir);
    var sun_r = hapke_brdf(max(sun_n_dot_l, 0.0), n_dot_v, cos_phase_sun);

    // Apply all shadow terms to the sun contribution only. Planetshine uses
    // a different incident direction so these don't apply to it.
    sun_r = sun_r * crater_shadow;
    if geo_n_dot_l > 0.0 {
        sun_r = sun_r * self_shadow(sample_dir, light_dir_local);
    }
    sun_r = sun_r * eclipse_term(hit, sun_dir);

    // Secondary: planetshine from the orbital parent.
    //
    // Physical model: the parent reflects sunlight back at the moon as a
    // finite-angular-radius disk. Irradiance arriving at the fragment is
    //     E_shine = E_sun · A_parent · (R/D)² · f(α)
    // where α is the sun-parent-moon angle seen from the fragment and
    //     f(α) = (sin α + (π-α) cos α) / π
    // is the Lambert-sphere integrated phase function (f(0) = 1, f(π) = 0).
    //
    // Applied through the same Hapke BRDF with the parent direction as the
    // light vector, so the moon's night side picks up a photographically
    // faithful dim glow when the parent is "full" overhead.
    var shine_rgb = vec3<f32>(0.0);
    if params.parent_tint.w > 0.5 {
        let to_parent = params.parent_pos.xyz - hit;
        let parent_dist = length(to_parent);
        let parent_radius = params.parent_pos.w;
        if parent_dist > parent_radius && parent_radius > 0.0 {
            let parent_dir = to_parent / parent_dist;
            let shine_n_dot_l = dot(shading_normal, parent_dir);
            let shine_cos_phase = dot(view_dir, parent_dir);
            let shine_r = hapke_brdf(max(shine_n_dot_l, 0.0), n_dot_v, shine_cos_phase);
            // Parent's illuminated-fraction phase function (Lambert sphere).
            let cos_alpha = clamp(dot(sun_dir, parent_dir), -1.0, 1.0);
            let alpha     = acos(cos_alpha);
            let lambert_phase = (sin(alpha) + (PI - alpha) * cos_alpha) / PI;
            let angular_ratio = parent_radius / parent_dist;
            let angular_sq    = angular_ratio * angular_ratio;
            let flux_factor   = params.light_intensity * angular_sq * lambert_phase;
            shine_rgb = params.parent_tint.xyz * shine_r * flux_factor;
        }
    }

    // Combine. Hapke's r is a radiance factor; the prior pipeline used a
    // Lambert `/PI` normalization we now fold into a global scale so
    // existing `light_intensity` values don't need re-tuning.
    let hapke_scale: f32 = 0.5;
    let sun_rgb = vec3<f32>(sun_r * params.light_intensity * hapke_scale);
    let lit = albedo * (sun_rgb + shine_rgb + vec3<f32>(params.ambient_intensity));

    // Correct depth.
    let hit_clip = view.clip_from_world * vec4(hit, 1.0);
    let depth    = hit_clip.z / hit_clip.w;

    return FragOutput(vec4(lit, 1.0), depth);
}
