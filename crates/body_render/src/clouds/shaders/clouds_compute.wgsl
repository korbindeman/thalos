#import bevy_open_world::common

// Camera-change threshold for the temporal-history gate. Coarse on purpose:
// while the sim runs, the rotating body-fixed frame puts ~1e-7/column f32
// rounding jitter into an otherwise-static camera matrix (and the game scales
// the translation column to match — see `drive_clouds`); real camera motion
// (≳ metres or ≳ 0.01°/frame) still trips it.
const CAM_EPSILON = 0.0001;
// The game stores the camera's body-fixed position in the view matrix's
// translation column scaled DOWN by 1e-4 (see `drive_clouds`); invert to
// recover metres for reprojection.
const CAM_POS_COLUMN_SCALE = 1.0e4;
// History blend under camera motion — slightly weaker than the steady-view
// `reprojection_strength`, since the reprojected sample is approximate.
const MOVING_REPROJECTION_STRENGTH = 0.9;
const MAX_DISTANCE = 1.0e9;
const WORLEY_RESOLUTION = 32;
const WORLEY_RESOLUTION_F32 = 32.0;

// ── Thalos fork: body-fixed spherical raymarch ───────────────────────────────
// Upstream is a Y-up flat-plane demo: the camera sits on the +Y axis and the
// noise fields are sampled in camera-relative tangent coordinates, so clouds
// follow the camera horizontally and the horizon shifts with view angle. This
// fork works entirely in the **body-fixed frame** of the planet: the camera's
// true planet-centred position arrives in `config.camera_translation`, view
// rays arrive pre-rotated into body-fixed space (`inverse_camera_view` =
// body_from_world × world_from_view), and every noise field is sampled at the
// body-fixed sample position — so clouds are glued to the ground, co-rotate
// with the planet, and the shell geometry is exact from the surface to orbit.
//
// f32 caveat at planet scale: body-fixed positions are ~3.2e6 m, where f32 has
// a ~0.25 m lattice. That is harmless for noise features ≥ tens of metres,
// but naive `u32(pos * scale) % N` texel math overflows f32 integer precision,
// so all field sampling below wraps positions into one world-space tile period
// FIRST (`fract(pos / period)`) and scales the small remainder up.
//
// ── Bounded-step raymarch ────────────────────────────────────────────────────
// Upstream marches a *fixed step count* across the whole [start, end] segment.
// At planet scale that breaks near the horizon: a near-tangent view ray grazes
// the cloud shell, its segment stretches to tens of km, and the few steps alias
// into radial "fountain" streaks. We march a world-space step bounded to
// [MIN_STEP_M, CLOUD_STEP_M] instead — sized so a short band crossing (a
// steep ray through the deck) gets ~TARGET_BAND_STEPS samples; a coarser step
// there leaves so few samples per band that the start dither shows as a
// moiré/crosshatch pattern. Long near-tangent segments stay at CLOUD_STEP_M,
// capped at MAX_RAY_STEPS, with a distance haze-out so the bounded reach reads
// as natural aerial perspective rather than a hard edge.
const CLOUD_STEP_M = 280.0;       // coarsest step (long horizon segments)
const MIN_STEP_M = 64.0;          // finest step (short band crossings)
const TARGET_BAND_STEPS = 20.0;   // target samples across one band crossing
const MAX_RAY_STEPS_CAP = 96u;    // compile-time safety cap; config selects ≤ this
const MAX_CLOUD_DIST = 25000.0;   // metres; clouds whose entry is past this are dropped
const CLOUD_FADE_START = 13000.0; // metres; begin hazing clouds toward the cap
// Per-column base-height undulation amount, in fractions of the band thickness
// (× the centered atlas-alpha field, ~±0.5). Breaks the flat deck base.
const BASE_UNDULATION = 0.55;

struct Config {
    clouds_base_scale: f32,
    clouds_raymarch_steps_count: u32,
    clouds_bottom_height: f32,
    clouds_top_height: f32,
    clouds_coverage: f32,
    clouds_density: f32,
    clouds_detail_scale: f32,
    clouds_detail_strength: f32,
    clouds_base_edge_softness: f32,
    clouds_bottom_softness: f32,
    clouds_shadow_raymarch_steps_count: u32,
    clouds_shadow_raymarch_step_size: f32,
    clouds_shadow_raymarch_step_multiply: f32,
    clouds_ambient_color_top: vec4f,
    clouds_ambient_color_bottom: vec4f,
    clouds_min_transmittance: f32,
    planet_radius: f32,
    forward_scattering_g: f32,
    backward_scattering_g: f32,
    scattering_lerp: f32,
    sun_dir: vec4f,
    sun_color: vec4f,
    camera_translation: vec3f,
    time: f32,
    reprojection_strength: f32,
    render_resolution: vec2f,
    inverse_camera_view: mat4x4f,
    inverse_camera_projection: mat4x4f,
    wind_displacement: vec3f,
};

@group(0) @binding(0) var<uniform> config: Config;

@group(1) @binding(0) var clouds_render_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(1) var clouds_atlas_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(2) var clouds_worley_texture: texture_storage_3d<rgba32float, read_write>;
// Per-pixel nearest cloud-hit distance (metres from the camera; MAX_DISTANCE
// where the ray hit no cloud). The game's body_sky composite reads it for
// true depth occlusion against terrain / the ship hull; the raymarch's own
// history reads use `history_distance_texture` below.
@group(1) @binding(3) var cloud_distance_texture: texture_storage_2d<r32float, write>;
// Planet-fixed cubemap weather field: coverage, cloud type, normalized base,
// normalized top. CLOUD-1 consumes coverage; CLOUD-3 consumes the remaining
// channels for type-specific vertical structure.
@group(1) @binding(4) var weather_texture: texture_cube<f32>;
@group(1) @binding(5) var weather_sampler: sampler;
// Previous frame's render + distance textures, snapshotted by the render node
// after each update dispatch. ALL temporal-history reads (same-pixel
// accumulation, motion reprojection, the saved camera rows) come from these —
// reading the in-flight storage textures instead races across workgroups and
// paints coherent streak artifacts.
@group(1) @binding(6) var history_texture: texture_2d<f32>;
@group(1) @binding(7) var history_distance_texture: texture_2d<f32>;

struct Ray {
    step_distance: f32,
    dir_length: f32,
    start: f32,
    end: f32,
}

struct RaymarchResult {
    dist: f32,
    color: vec4f,
}

// ── Body-fixed field sampling ────────────────────────────────────────────────
//
// Zonal wind is NOT handled here: the game folds the accumulated drift into
// the body-fixed frame it feeds (a slow rotation about the spin axis baked
// into `inverse_camera_view` / `camera_translation` / `sun_dir`), so the
// raymarch samples an already-advected field at zero per-sample cost.
// `wind_displacement.yz` still drifts the detail-erosion noise ("boiling").
//
// PERF INVARIANT: nothing inside the per-sample density path may call
// transcendentals (sin/cos/atan2/acos/normalize/pow). Coverage and the
// triplanar weights are planet-scale smooth (≥ tens of km), so they are
// computed ONCE PER RAY in `raymarch` and passed down; a previous version
// evaluated them per density sample and dropped the frame rate severalfold.

// Local overcast fraction from the canonical planet-fixed weather cubemap.
fn sample_coverage(n: vec3f) -> f32 {
    return textureSampleLevel(weather_texture, weather_sampler, n, 0.0).r;
}

// World-space tile period of the base atlas, metres (identical on both texel
// axes: upstream's `u32(p * k * res) % res` is `fract(p * k) * res`, so the
// period is 1/k regardless of the 1920×1080 texel rectangle).
fn atlas_period() -> f32 {
    return 1.0 / (0.00005 * config.clouds_base_scale);
}

// Wrap-first atlas fetch: reduce the planar coordinate to one tile period in
// f32-safe small numbers, then scale up to texels.
fn atlas_load(plane: vec2f) -> vec4f {
    let w = fract(plane / atlas_period());
    let res = config.render_resolution;
    let tx = min(u32(w.x * res.x), u32(res.x) - 1u);
    let ty = min(u32(w.y * res.y), u32(res.y) - 1u);
    return textureLoad(clouds_atlas_texture, vec2u(tx, ty));
}

// Triplanar projection of the tiling 2-D atlas onto the sphere. `tri_w` is
// the squared body-fixed surface normal (weights summing to 1), computed once
// per ray in `raymarch` — within one ray's ≤ 25 km reach on a ~3000 km body
// it is effectively constant. The blend only matters across the planet, where
// it removes the polar pinch a lat/lon mapping would have.
fn atlas_triplanar(pos: vec3f, tri_w: vec3f) -> vec4f {
    return atlas_load(pos.yz) * tri_w.x + atlas_load(pos.xz) * tri_w.y
        + atlas_load(pos.xy) * tri_w.z;
}

fn worley_corner(c: vec3i) -> f32 {
    let w = (c + vec3i(WORLEY_RESOLUTION)) % vec3i(WORLEY_RESOLUTION);
    return textureLoad(clouds_worley_texture, vec3u(w)).r;
}

// Wrap-first, trilinearly-filtered 3-D Worley detail fetch (period = 32 cells
// in world space). Nearest-neighbour sampling here reads as a pixel-scale
// crosshatch on the cloud surfaces once the raymarch dither is time-stable.
fn cloud_map_detail(position: vec3f) -> f32 {
    let s = 0.0016 * config.clouds_base_scale * config.clouds_detail_scale;
    let period = WORLEY_RESOLUTION_F32 / s;
    let p = fract(position / period) * WORLEY_RESOLUTION_F32;

    let pf = p - 0.5; // texel centres
    let base = floor(pf);
    let f = pf - base;
    let b = vec3i(base);

    let c000 = worley_corner(b + vec3i(0, 0, 0));
    let c100 = worley_corner(b + vec3i(1, 0, 0));
    let c010 = worley_corner(b + vec3i(0, 1, 0));
    let c110 = worley_corner(b + vec3i(1, 1, 0));
    let c001 = worley_corner(b + vec3i(0, 0, 1));
    let c101 = worley_corner(b + vec3i(1, 0, 1));
    let c011 = worley_corner(b + vec3i(0, 1, 1));
    let c111 = worley_corner(b + vec3i(1, 1, 1));

    let x00 = mix(c000, c100, f.x);
    let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x);
    let x11 = mix(c011, c111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Erode a bit from the clouds_bottom_height and clouds_top_height of the cloud layer
fn cloud_gradient(normalized_height: f32) -> f32 {
    return (
        common::linearstep(0.0, 0.1, normalized_height) -
        common::linearstep(0.8, 1.2, normalized_height)
    );
}

// Per-sample density. `cov` (local overcast fraction × global knob) and
// `tri_w` (triplanar weights) are the per-ray context from `raymarch` — see
// the PERF INVARIANT note above.
fn get_cloud_map_density(pos: vec3f, normalized_height: f32, cov: f32, tri_w: vec3f) -> f32 {
    // One triplanar fetch serves base shape (r), remap threshold (g), height
    // gradient (b), and the per-column base-height undulation field (a) that
    // breaks the flat deck base.
    let atlas = atlas_triplanar(pos, tri_w);
    let nh = clamp(normalized_height - (atlas.a - 0.5) * BASE_UNDULATION, 0.0, 1.0);

    // (1 - nh)^16 as a multiply chain — pow() hits the SFU path.
    let t1 = 1.0 - nh;
    let t2 = t1 * t1;
    let t4 = t2 * t2;
    let t8 = t4 * t4;
    let height_shape = nh * nh * atlas.b + t8 * t8;
    var m = common::remap(atlas.r - height_shape, atlas.g, 1.0) * cloud_gradient(nh);

    let clouds_detail_strength = smoothstep(1.0, 0.5, m);

    // Erode with detail; the y/z wind components drift the erosion field for
    // slow "boiling" independent of the zonal advection.
    if (clouds_detail_strength > 0.0) {
        let boil = vec3f(0.0, config.wind_displacement.y, config.wind_displacement.z);
        m -= cloud_map_detail(pos + boil) * clouds_detail_strength * config.clouds_detail_strength;
    }

    m = smoothstep(0.0, config.clouds_base_edge_softness, m + cov - 1.0);
    m *= common::linearstep0(config.clouds_bottom_softness, nh);

    return clamp(m * config.clouds_density, 0.0, 1.0);
}

fn get_normalized_height(pos: vec3f) -> f32 {
    let clouds_height = config.clouds_top_height - config.clouds_bottom_height;
    return (length(pos) - (config.planet_radius + config.clouds_bottom_height)) / clouds_height;
}

// `jitter` perturbs the march's start offset: unjittered, the exponential
// step pattern is a deterministic bias that draws sun-aligned bands across
// translucent cloud — a settled view never averages it away. Jittered, it is
// noise the temporal history converges out of.
fn volumetric_shadow(origin: vec3f, cov: f32, tri_w: vec3f, jitter: f32) -> f32 {
    var ray_step_size = config.clouds_shadow_raymarch_step_size;
    var distance_along_ray = ray_step_size * (0.25 + 0.5 * jitter);
    var transmittance = 1.0;

    for (var step: u32 = 0; step < config.clouds_shadow_raymarch_steps_count; step++) {
        let pos = origin + config.sun_dir.xyz * distance_along_ray;
        let normalized_height = get_normalized_height(pos);

        if (normalized_height > 1.0) { return transmittance; };

        let clouds_density = get_cloud_map_density(pos, normalized_height, cov, tri_w);
        transmittance *= exp(-clouds_density * ray_step_size);

        ray_step_size *= config.clouds_shadow_raymarch_step_multiply;
        distance_along_ray += ray_step_size;
    }

    return transmittance;
}

fn henyey_greenstein(ray_dot_sun: f32, g: f32) -> f32 {
    let g_squared = g * g;
    return (1.0 - g_squared) / pow(1.0 + g_squared - 2.0 * g * ray_dot_sun, 1.5);
}

fn get_ray(ray_origin: vec3f, ray_dir: vec3f, max_dist: f32, jitter: f32) -> Ray {
    // True ray-sphere intersection against the two cloud shells, centred at
    // the planet origin (`length(pos)` == radius), from the camera's ACTUAL
    // body-fixed position (radius = planet_radius + altitude), so the deck
    // sits at a fixed absolute altitude and the horizon curvature is exact
    // regardless of camera height.
    let r_base = config.planet_radius + config.clouds_bottom_height;
    let r_top = config.planet_radius + config.clouds_top_height;
    let cam_r = length(ray_origin);
    let b = dot(ray_origin, ray_dir);
    let oc2 = dot(ray_origin, ray_origin);

    // Outer (top) shell. A miss means the ray never reaches cloud altitude.
    let disc_top = b * b - (oc2 - r_top * r_top);
    if (disc_top <= 0.0) {
        return Ray(CLOUD_STEP_M, max_dist + 1.0, max_dist + 1.0, max_dist);
    }
    let sq_top = sqrt(disc_top);
    let tt0 = -b - sq_top;
    let tt1 = -b + sq_top;

    // Inner (base) shell — may be missed when the ray grazes above the base.
    let disc_base = b * b - (oc2 - r_base * r_base);
    let hit_base = disc_base > 0.0;
    var tb0 = 0.0;
    var tb1 = 0.0;
    if (hit_base) {
        let sq_base = sqrt(disc_base);
        tb0 = -b - sq_base;
        tb1 = -b + sq_base;
    }

    // First forward slab segment, for the three camera regimes.
    var seg_start = 0.0;
    var seg_end = 0.0;
    if (cam_r > r_top) {
        // Above the deck: enter at the top shell, exit at the near base hit (if
        // the ray dips through the layer) or the far top hit otherwise.
        seg_start = tt0;
        if (hit_base && tb0 > tt0) { seg_end = tb0; } else { seg_end = tt1; }
    } else if (cam_r < r_base) {
        // Below the deck (near-surface): begins at the forward base crossing,
        // ends at the far top crossing.
        if (hit_base && tb1 > 0.0) { seg_start = tb1; } else { seg_start = max(tt0, 0.0); }
        seg_end = tt1;
    } else {
        // Inside the deck: from the camera to the nearest shell.
        seg_start = 0.0;
        if (hit_base && tb0 > 0.0) { seg_end = tb0; } else { seg_end = tt1; }
    }

    seg_start = max(seg_start, 0.0);
    // Cap the marched segment so near-tangent rays can't stretch the layer
    // across tens of km and alias (see the header comment).
    seg_end = min(min(seg_end, max_dist), seg_start + MAX_CLOUD_DIST);

    // Adaptive step: ~TARGET_BAND_STEPS samples across this segment, bounded
    // (see the header comment).
    let step = clamp((seg_end - seg_start) / TARGET_BAND_STEPS, MIN_STEP_M, CLOUD_STEP_M);
    let dir_length = seg_start - step * jitter;

    return Ray(step, dir_length, seg_start, seg_end);
}

fn raymarch(ray_origin: vec3f, ray_dir: vec3f, max_dist: f32, jitter: f32) -> RaymarchResult {
    let ray = get_ray(ray_origin, ray_dir, max_dist, jitter);

    if (ray.start > max_dist) {
        return RaymarchResult(max_dist, vec4f(0.0, 0.0, 0.0, 1.0));
    }

    // Per-ray field context (see the PERF INVARIANT note): coverage and the
    // triplanar weights vary over tens-of-km scales, so one evaluation at the
    // segment midpoint serves every sample on this ≤ 25 km ray — and a fully
    // clear column skips the march outright.
    let mid = ray_origin + (0.5 * (ray.start + ray.end)) * ray_dir;
    let n_mid = normalize(mid);
    let cov = clamp(sample_coverage(n_mid) * config.clouds_coverage, 0.0, 1.0);
    if (cov <= 1.0e-3) {
        return RaymarchResult(max_dist, vec4f(0.0, 0.0, 0.0, 1.0));
    }
    let tri_w = n_mid * n_mid; // unit normal → weights sum to 1

    // Frostbite: dual-lobe phase function
    let ray_dot_sun = dot(ray_dir, -config.sun_dir.xyz);
    let scattering = mix(
        henyey_greenstein(ray_dot_sun, config.forward_scattering_g),
        henyey_greenstein(ray_dot_sun, config.backward_scattering_g),
        config.scattering_lerp
    );

    var dir_length = ray.dir_length;
    var dist = max_dist;
    var scattered_light = vec3f(0.0, 0.0, 0.0);
    var transmittance = 1.0;

    // March-exhaustion fade: a ray travelling far *inside* the deck uses all
    // its configured step cap and stops mid-cloud; the entry-distance haze below never
    // fires for it (entry was close), so without this the stop front reads
    // as a sharp tonal seam arc across the deck. Dissolve density over the
    // last third of the marched reach instead.
    let ray_step_limit = clamp(config.clouds_raymarch_steps_count, 1u, MAX_RAY_STEPS_CAP);
    let march_span = f32(ray_step_limit) * ray.step_distance;
    let reach_fade_begin = ray.start + 0.65 * march_span;
    let reach_end = ray.start + march_span;

    for (var step: u32 = 0u; step < ray_step_limit; step++) {
        if (dir_length > ray.end) { break; }
        let world_position = ray_origin + dir_length * ray_dir;

        let normalized_height = clamp(get_normalized_height(world_position), 0.0, 1.0);
        let clouds_density_sampled =
            get_cloud_map_density(world_position, normalized_height, cov, tri_w)
            * (1.0 - smoothstep(reach_fade_begin, reach_end, dir_length));

        if (clouds_density_sampled > 0.0) {
            dist = min(dist, dir_length);

            let ambient_light = mix(
                config.clouds_ambient_color_bottom,
                config.clouds_ambient_color_top,
                normalized_height
            );

            // Shadow jitter decorrelated PER SAMPLE (golden-ratio sequence
            // over the step index): one shared value would wobble the whole
            // pixel's sun term coherently every frame — visible lighting
            // flicker no history fully hides. Per-sample, the noise averages
            // across the ~20 samples of the ray before temporal blending.
            let shadow_jitter = fract(jitter + f32(step) * 0.6180339887);

            // Frostbite energy-conversing integration
            let S = clouds_density_sampled * (
                ambient_light.rgb +
                config.sun_color.rgb * scattering
                    * volumetric_shadow(world_position, cov, tri_w, shadow_jitter)
            );
            let delta_transmittance = exp(-clouds_density_sampled * ray.step_distance);
            let integrated_scattering = S * (1.0 - delta_transmittance) / clouds_density_sampled;

            scattered_light += transmittance * integrated_scattering;
            transmittance *= delta_transmittance;
        }

        if transmittance <= config.clouds_min_transmittance { break; }

        dir_length += ray.step_distance;
    }

    // Distance haze-out: clouds whose entry is past CLOUD_FADE_START dissolve
    // toward fully transparent by the cap, so the bounded reach reads as aerial
    // perspective instead of a hard cut (and the streak-prone near-tangent rays
    // at the very horizon contribute nothing).
    let dist_fade = 1.0 - smoothstep(CLOUD_FADE_START, MAX_CLOUD_DIST, ray.start);
    transmittance = mix(1.0, transmittance, dist_fade);
    scattered_light *= dist_fade;

    return RaymarchResult(dist, vec4f(scattered_light, transmittance));
}

fn render_clouds_atlas(frag_coord: vec2f) -> vec4f {
    let v_uv = frag_coord / config.render_resolution.xy;
    let coord = vec3f(v_uv, 0.5);

    let mfbm = 0.9;
    let mvor = 0.7;

    return vec4f(
        mix(1.0, common::tilable_perlin_fbm(coord, 7, 4), mfbm) *
            mix(1.0, common::tilable_voronoi(coord, 8, 9.0), mvor),
        0.625 * common::tilable_voronoi(coord, 3, 15.0) +
            0.250 * common::tilable_voronoi(coord, 3, 19.0) +
            0.125 * common::tilable_voronoi(coord, 3, 23.0) -
            1.0,
        1.0 - common::tilable_voronoi(coord + 0.5, 6, 9.0),
        // Thalos fork: alpha carries a low-frequency, seamlessly-tiling
        // base-height field (large-scale undulation, ~[0,1]) sampled per column
        // in `get_cloud_map_density` to break the flat deck base.
        common::tilable_perlin_fbm(coord + 0.27, 3, 2)
    );
}

fn render_clouds_worley(coord: vec3f) -> vec4f {
    let r = common::tilable_voronoi(coord, 16, 3.0);
    let g = common::tilable_voronoi(coord, 4, 8.0);
    let b = common::tilable_voronoi(coord, 4, 16.0);

    let c = max(0.0, 1.0 - (r + g * 0.5 + b * 0.25) / 1.75);

    return vec4f(c);
}

struct CloudsOutput {
    color: vec4f,
    dist: f32,
}

// Bilinear history fetch in texel space (rr ∈ [0,1]² of the render target).
// Nearest-texel history resampling re-quantises the reprojected position
// every frame, which itself injects noise; bilinear keeps the accumulation
// converging under sub-texel camera motion. Clamped two rows short of the
// top texture rows, which hold the camera-save payload.
fn sample_history_bilinear(rr: vec2f) -> vec4f {
    let res = config.render_resolution.xy;
    let p = rr * res - 0.5;
    let base = floor(p);
    let f = p - base;
    let b = vec2i(base);
    let cmax = vec2i(i32(res.x) - 1, i32(res.y) - 3);
    let c00 = clamp(b, vec2i(0, 0), cmax);
    let c11 = clamp(b + vec2i(1, 1), vec2i(0, 0), cmax);
    let h00 = textureLoad(history_texture, vec2u(c00), 0);
    let h10 = textureLoad(history_texture, vec2u(vec2i(c11.x, c00.y)), 0);
    let h01 = textureLoad(history_texture, vec2u(vec2i(c00.x, c11.y)), 0);
    let h11 = textureLoad(history_texture, vec2u(c11), 0);
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

fn get_clouds_color(frag_coord: vec2f, camera: mat4x4f, old_cam: mat4x4f, ray_dir: vec3f, ray_origin: vec3f) -> CloudsOutput {
    if (frag_coord.y < 1.5) {
        if frag_coord.x < 1.0 {
            return CloudsOutput(vec4f(config.render_resolution.xy, 0.0, 0.0), MAX_DISTANCE);
        }
        return CloudsOutput(common::save_camera(camera, frag_coord, ray_origin), MAX_DISTANCE);
    }

    // Camera-change metric: a steady view (in the body-fixed frame:
    // landed/parked, even with the sim running) accumulates same-pixel
    // history; a moving view reprojects history through the previous frame's
    // camera below.
    let cam_delta = length(
        abs(old_cam[0] - camera[0]) +
        abs(old_cam[1] - camera[1]) +
        abs(old_cam[2] - camera[2]) +
        abs(old_cam[3] - camera[3])
    );
    let cam_static = cam_delta <= CAM_EPSILON;

    // Interleaved-gradient spatial dither for the raymarch start offset, with
    // a ~golden-ratio phase advance per frame (assuming ~60 fps; any large
    // irrational-ish step decorrelates frames). Always temporal: both history
    // paths below average it away, which is what keeps the dither from
    // reading as a static moiré/halftone pattern on the cloud surfaces.
    var jitter = fract(52.9829189 * fract(dot(frag_coord, vec2f(0.06711056, 0.00583715))));
    jitter = fract(jitter + config.time * 37.08204);

    let result = raymarch(ray_origin, ray_dir, MAX_DISTANCE, jitter);

    // Thalos fork: store the *clean* raymarch result — rgb = premultiplied
    // in-scatter, a = transmittance — with NO built-in sky/fog mix. The game
    // composites this layer over its own scene (atmosphere, terrain, stars) in
    // a separate fullscreen pass, so baking a sky color and distance fog in
    // here would double-paint a sky we don't want.
    let col = result.color;

    if (cam_static) {
        // Steady view: same-pixel accumulation from the history snapshot.
        let original_color = textureLoad(
            history_texture,
            vec2u(u32(frag_coord.x),
            u32(config.render_resolution.y - 1.0) - u32(frag_coord.y)),
            0
        );
        return CloudsOutput(mix(col, original_color, config.reprojection_strength), result.dist);
    }

    // Moving view: reproject this ray's nearest cloud point through the
    // previous frame's camera (stored in the history's save rows) and blend
    // the history texel there. Cloud points are body-fixed, so the point
    // itself is frame-invariant.
    if (result.dist < 1.0e8) {
        let p = ray_origin + result.dist * ray_dir;
        let cam_old = old_cam[3].xyz * CAM_POS_COLUMN_SCALE;
        let rel = p - cam_old;
        // old_cam[0..2].xyz are the old view-space basis vectors expressed in
        // body-fixed coords; project into old view space (looks down -Z).
        let dv = vec3f(
            dot(old_cam[0].xyz, rel),
            dot(old_cam[1].xyz, rel),
            dot(old_cam[2].xyz, rel),
        );
        if (dv.z < 0.0) {
            // Symmetric perspective: the inverse projection's diagonal holds
            // the frustum tangents (same assumption as get_ray_direction).
            let tan_x = config.inverse_camera_projection[0][0];
            let tan_y = config.inverse_camera_projection[1][1];
            let ndc = vec2f(dv.x / (-dv.z * tan_x), dv.y / (-dv.z * tan_y));
            // Invert get_ray_direction's frag→NDC mapping (y flip included);
            // rr is in invocation-texel space. Exclude the top two texture
            // rows (rr.y ≳ 0.998) — they hold the camera-save payload.
            let rr = (vec2f(ndc.x, -ndc.y) + 1.0) * 0.5;
            if (rr.x > 0.0 && rr.x < 1.0 && rr.y > 0.0 && rr.y < 0.998) {
                let texel = vec2u(rr * config.render_resolution.xy);
                let hist_dist = textureLoad(history_distance_texture, texel, 0).r;
                // Soft disocclusion: weight history by cloud-depth agreement
                // instead of a binary reject — hard accept/reject boundaries
                // themselves pattern as fresh-noise speckle at every edge.
                // The hit distance carries ~one-step jitter (≈ 5-10 %), so
                // the ramp starts well above that.
                let rel_err = abs(hist_dist - result.dist) / max(result.dist, 1.0);
                let agree = 1.0 - smoothstep(0.15, 0.5, rel_err);
                // The public temporal-strength control gates *both* the
                // steady same-pixel path and this moving-view path. Previously
                // setting it to zero still kept 90% motion history, so a
                // temporal-disabled diagnostic capture was impossible.
                let w = MOVING_REPROJECTION_STRENGTH * agree
                    * clamp(config.reprojection_strength, 0.0, 1.0);
                if (w > 0.01) {
                    let hist = sample_history_bilinear(rr);
                    return CloudsOutput(mix(col, hist, w), result.dist);
                }
            }
        }
    }
    return CloudsOutput(col, result.dist);
}

fn get_ray_direction(frag_coord: vec2f) -> vec3f {
    // inverse_camera_projection is also called view_from_clip
    // inverse_camera_view is also called world_from_view; here it is
    // body_from_world × world_from_view, so rays come out body-fixed.
    let rect_relative = frag_coord / config.render_resolution;

    // Flip the Y co-ordinate from the clouds_top_height to the clouds_bottom_height to enter NDC.
    let ndc_xy = (rect_relative * 2.0 - vec2f(1.0, 1.0)) * vec2f(1.0, -1.0);

    let ray_clip = vec4f(ndc_xy.xy, -1.0, 1.0);
    let ray_eye = config.inverse_camera_projection * ray_clip;
    let ray_world = config.inverse_camera_view * vec4f(ray_eye.xy, -1.0, 0.0);

    return normalize(ray_world.xyz);
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let index = vec2f(f32(invocation_id.x), f32(invocation_id.y)) + vec2f(0.5);
    let inverted_y_coord = config.render_resolution.y - index.y;

    let worley_coord = vec2f(index.x, inverted_y_coord);

    let z = floor(worley_coord.x / WORLEY_RESOLUTION_F32) + 8.0 * floor(worley_coord.y / WORLEY_RESOLUTION_F32);
    let xy = vec2f(index.x, inverted_y_coord) % WORLEY_RESOLUTION_F32;
    let xyz = vec3f(xy, z);

    let worley_col = render_clouds_worley(xyz / WORLEY_RESOLUTION_F32);
    let atlas_col = render_clouds_atlas(vec2f(index.x, inverted_y_coord));

    storageBarrier();

    textureStore(clouds_atlas_texture, invocation_id.xy, atlas_col);
    textureStore(clouds_worley_texture, vec3u(u32(xyz.x), u32(xyz.y), u32(xyz.z)), worley_col);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let index = vec2f(f32(invocation_id.x), f32(invocation_id.y)) + vec2f(0.5);

    // Previous frame's camera, from the history snapshot's save rows.
    let sample_y = u32(config.render_resolution.y) - 1;
    let old_cam = mat4x4f(
        textureLoad(history_texture, vec2u(1, sample_y), 0),
        textureLoad(history_texture, vec2u(2, sample_y), 0),
        textureLoad(history_texture, vec2u(3, sample_y), 0),
        textureLoad(history_texture, vec2u(4, sample_y), 0),
    );
    var frag_coord = vec2f(index.x, config.render_resolution.y - index.y);

    // Body-fixed camera position, planet-centred (the consumer feeds it via
    // CameraMatrices.translation).
    let ray_origin = config.camera_translation;
    let ray_dir = get_ray_direction(index);
    let out = get_clouds_color(frag_coord, config.inverse_camera_view, old_cam, ray_dir, ray_origin);

    storageBarrier();

    textureStore(clouds_render_texture, invocation_id.xy, out.color);
    textureStore(cloud_distance_texture, invocation_id.xy, vec4f(out.dist, 0.0, 0.0, 0.0));
}
