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
const WORLEY_RESOLUTION = 64;
const WORLEY_RESOLUTION_F32 = 64.0;

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
const CLOUD_STEP_M = 500.0;       // coarsest step (long horizon segments)
const MIN_STEP_M = 65.0;          // finest step (short band crossings)
const TARGET_BAND_STEPS = 42.0;   // target samples across one band crossing
const MAX_RAY_STEPS_CAP = 128u;   // compile-time safety cap; config selects ≤ this
const MAX_CLOUD_DIST = 50000.0;   // metres; orbital projection owns farther clouds
const CLOUD_FADE_START = 36000.0; // metres; begin atmospheric horizon fade

struct Config {
    clouds_base_shape_scale_m: f32,
    clouds_raymarch_steps_count: u32,
    clouds_bottom_height: f32,
    clouds_top_height: f32,
    clouds_coverage: f32,
    clouds_density: f32,
    clouds_detail_scale_m: f32,
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
@group(1) @binding(1) var clouds_worley_texture: texture_storage_3d<rgba32float, read_write>;
// Per-pixel nearest cloud-hit distance (metres from the camera; MAX_DISTANCE
// where the ray hit no cloud). The game's body_sky composite reads it for
// true depth occlusion against terrain / the ship hull; the raymarch's own
// history reads use `history_distance_texture` below.
@group(1) @binding(2) var cloud_distance_texture: texture_storage_2d<r32float, write>;
// Planet-fixed cubemap weather field: coverage, cloud type, normalized base,
// normalized top. CLOUD-3 consumes all channels for type-specific vertical
// structure.
@group(1) @binding(3) var weather_texture: texture_cube<f32>;
@group(1) @binding(4) var weather_sampler: sampler;
// Previous frame's render + distance textures, snapshotted by the render node
// after each update dispatch. ALL temporal-history reads (same-pixel
// accumulation, motion reprojection, the saved camera rows) come from these —
// reading the in-flight storage textures instead races across workgroups and
// paints coherent streak artifacts.
@group(1) @binding(5) var history_texture: texture_2d<f32>;
@group(1) @binding(6) var history_distance_texture: texture_2d<f32>;

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

// Coverage, cloud kind, normalized local base and normalized local top from
// the canonical planet-fixed weather cubemap.
fn sample_weather(n: vec3f) -> vec4f {
    return textureSampleLevel(weather_texture, weather_sampler, n, 0.0);
}

fn volume_corner(c: vec3i) -> vec4f {
    let w = (c + vec3i(WORLEY_RESOLUTION)) % vec3i(WORLEY_RESOLUTION);
    return textureLoad(clouds_worley_texture, vec3u(w));
}

// Wrap-first, trilinearly-filtered 3-D shape fetch. The same periodic volume
// is sampled through decorrelated domains/scales below; because every domain
// is genuinely 3-D, cells grow and erode in altitude instead of extruding a
// 2-D atlas into the vertical curtains visible before CLOUD-3.
fn cloud_volume(position: vec3f, period: f32) -> vec4f {
    let p = fract(position / period) * WORLEY_RESOLUTION_F32;

    let pf = p - 0.5; // texel centres
    let base = floor(pf);
    var f = pf - base;
    f = f * f * (3.0 - 2.0 * f);
    let b = vec3i(base);

    let c000 = volume_corner(b + vec3i(0, 0, 0));
    let c100 = volume_corner(b + vec3i(1, 0, 0));
    let c010 = volume_corner(b + vec3i(0, 1, 0));
    let c110 = volume_corner(b + vec3i(1, 1, 0));
    let c001 = volume_corner(b + vec3i(0, 0, 1));
    let c101 = volume_corner(b + vec3i(1, 0, 1));
    let c011 = volume_corner(b + vec3i(0, 1, 1));
    let c111 = volume_corner(b + vec3i(1, 1, 1));

    let x00 = mix(c000, c100, f.x);
    let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x);
    let x11 = mix(c011, c111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

fn rotated_domain(p: vec3f) -> vec3f {
    // Cheap orthonormal-ish axis mixing. It decorrelates repeated volume
    // samples without trig or a second texture.
    return vec3f(
        0.72 * p.x + 0.41 * p.y - 0.56 * p.z,
        -0.35 * p.x + 0.91 * p.y + 0.22 * p.z,
        0.60 * p.x + 0.05 * p.y + 0.80 * p.z,
    );
}

// Typed volumetric density. `weather` is constant over a short view segment,
// while every base/detail sample is full 3-D world-space structure.
fn get_cloud_map_density(pos: vec3f, shell_height: f32, weather: vec4f) -> f32 {
    let cov = clamp(weather.r * config.clouds_coverage, 0.0, 1.0);
    let local_base = clamp(weather.b, 0.0, 0.92);
    let local_top = max(clamp(weather.a, 0.02, 1.0), local_base + 0.02);
    let h = (shell_height - local_base) / (local_top - local_base);
    if (h <= 0.0 || h >= 1.0 || cov <= 1.0e-3) {
        return 0.0;
    }

    let stratus_w = 1.0 - smoothstep(0.18, 0.38, weather.g);
    let storm_w = smoothstep(0.72, 0.88, weather.g);
    let cumulus_w = max(0.0, 1.0 - stratus_w - storm_w);

    let shape_scale = max(config.clouds_base_shape_scale_m, 500.0);
    let broad = cloud_volume(
        rotated_domain(pos) + vec3f(1800.0, -4200.0, 900.0),
        shape_scale,
    );
    let macro_shape = cloud_volume(
        pos + vec3f(-7300.0, 2100.0, 4900.0),
        shape_scale * 2.7,
    );
    // Broad Perlin/Worley masses. The erosion channel stays out of the solid
    // body: promoting its small cells into `shape` fragmented the volume into
    // screen-space stipple instead of adding readable billows.
    let shape = broad.r * 0.52 + broad.g * 0.24 + broad.a * 0.24;

    // Coverage is a formation threshold, not an opacity multiplier: lowering
    // it opens clear sky between otherwise equally dense cloud masses.
    // The second, much larger period groups otherwise repeating 8 km bodies
    // into broad systems without contributing its own low-resolution blobs to
    // the boundary. Keeping it in the threshold preserves solid cloud cores.
    let threshold = mix(0.58, 0.30, cov) + (0.5 - macro_shape.a) * 0.07;
    let vertical_narrow = h * (0.04 * stratus_w + 0.19 * cumulus_w + 0.09 * storm_w);
    var mass = shape - threshold - vertical_narrow;

    // Cumulonimbus anvils broaden again near the tropopause, but only where
    // the storm weather channel permits them.
    let anvil_profile = smoothstep(0.62, 0.76, h) * (1.0 - smoothstep(0.90, 1.0, h));
    let anvil_shape = broad.r * 0.72 + broad.a * 0.28 - (threshold - 0.06);
    mass = max(mass, anvil_shape * anvil_profile * storm_w);

    let bottom_softness = max(config.clouds_bottom_softness, 0.01);
    let stratus_profile = smoothstep(0.0, bottom_softness * 0.45, h)
        * (1.0 - smoothstep(0.72, 1.0, h));
    let cumulus_profile = smoothstep(0.0, bottom_softness * 0.75, h)
        * (1.0 - smoothstep(0.70, 1.0, h));
    let storm_profile = smoothstep(0.0, bottom_softness * 0.35, h)
        * (1.0 - smoothstep(0.88, 1.0, h));
    let vertical_profile = stratus_profile * stratus_w
        + cumulus_profile * cumulus_w
        + storm_profile * storm_w;

    // Fine 3-D Worley erosion is strongest only near the boundary, preserving
    // solid cores for deep self-shadow while cutting cauliflower detail into
    // silhouettes. Detail moves slowly in a decorrelated domain.
    let boil = vec3f(0.0, config.wind_displacement.y, config.wind_displacement.z);
    let edge = 1.0 - smoothstep(0.04, 0.18, mass);
    if (edge > 1.0e-3) {
        let detail = cloud_volume(
            rotated_domain(pos + boil) + vec3f(270.0, -610.0, 130.0),
            // Channel B contains eight primary Worley cells across the stored
            // tile. `clouds_detail_scale_m` describes one authored physical
            // erosion feature, not the whole tile period.
            max(config.clouds_detail_scale_m, 50.0) * 8.0,
        );
        mass -= (1.0 - detail.b) * edge * config.clouds_detail_strength * 0.55;
    }

    let shaped = smoothstep(0.0, max(config.clouds_base_edge_softness, 0.015), mass);
    return max(shaped * vertical_profile * config.clouds_density, 0.0);
}

fn get_normalized_height(pos: vec3f) -> f32 {
    let clouds_height = config.clouds_top_height - config.clouds_bottom_height;
    return (length(pos) - (config.planet_radius + config.clouds_bottom_height)) / clouds_height;
}

// The sparse exponential shadow integral is deliberately centred and stable.
// Per-pixel jitter here used to turn three widely spaced density taps into a
// high-contrast stipple that remained visible after temporal reconstruction.
// The smooth 3-D field plus increasing step reach already avoids coherent
// slab banding without injecting a second stochastic signal per view sample.
fn volumetric_shadow(origin: vec3f, weather: vec4f) -> f32 {
    var ray_step_size = config.clouds_shadow_raymarch_step_size;
    var distance_along_ray = ray_step_size * 0.5;
    var transmittance = 1.0;

    for (var step: u32 = 0; step < config.clouds_shadow_raymarch_steps_count; step++) {
        let pos = origin + config.sun_dir.xyz * distance_along_ray;
        let normalized_height = get_normalized_height(pos);

        if (normalized_height > 1.0) { return transmittance; };

        let clouds_density = get_cloud_map_density(pos, normalized_height, weather);
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
    // The full-step temporal phase is safe now that the minimum density
    // feature is larger than one march step. It prevents near-horizontal rays
    // from stacking their samples into coherent bands; history averages the
    // phase instead of trying to hide under-resolved density.
    let dir_length = seg_start - step * jitter;

    return Ray(step, dir_length, seg_start, seg_end);
}

fn raymarch(ray_origin: vec3f, ray_dir: vec3f, max_dist: f32, jitter: f32) -> RaymarchResult {
    let ray = get_ray(ray_origin, ray_dir, max_dist, jitter);

    if (ray.start > max_dist) {
        return RaymarchResult(max_dist, vec4f(0.0, 0.0, 0.0, 1.0));
    }

    // Per-ray weather context varies over tens-of-km scales, so one evaluation
    // at the segment midpoint serves the short view segment. Base/detail shape
    // remains full 3-D per sample.
    let mid = ray_origin + (0.5 * (ray.start + ray.end)) * ray_dir;
    let n_mid = normalize(mid);
    let weather = sample_weather(n_mid);
    if (weather.r * config.clouds_coverage <= 1.0e-3) {
        return RaymarchResult(max_dist, vec4f(0.0, 0.0, 0.0, 1.0));
    }

    // Frostbite: dual-lobe phase function
    let ray_dot_sun = dot(ray_dir, -config.sun_dir.xyz);
    let raw_scattering = mix(
        henyey_greenstein(ray_dot_sun, config.forward_scattering_g),
        henyey_greenstein(ray_dot_sun, config.backward_scattering_g),
        config.scattering_lerp
    );
    // HG above omits 1/(4π). Normalize and retain a broad diffuse multiple-
    // scattering floor so side-lit clouds stay bright without a white forward
    // spike consuming the whole frame.
    let scattering = 0.22 + 0.78 * min(raw_scattering * 0.07957747, 1.8);

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
            get_cloud_map_density(world_position, normalized_height, weather)
            * (1.0 - smoothstep(reach_fade_begin, reach_end, dir_length));

        if (clouds_density_sampled > 0.0) {
            dist = min(dist, dir_length);

            let ambient_light = mix(
                config.clouds_ambient_color_bottom,
                config.clouds_ambient_color_top,
                normalized_height
            );

            // Frostbite energy-conversing integration
            let sun_visibility = volumetric_shadow(world_position, weather);
            // A small multiple-scattering floor keeps shaded vapor legible;
            // the old 12% floor plus bright ambient erased nearly all optical
            // depth and made kilometre-thick clouds read as pale fog.
            let multiple_scatter_fill = 0.035 + 0.965 * sun_visibility;
            let S = clouds_density_sampled * (
                ambient_light.rgb +
                config.sun_color.rgb * scattering
                    * multiple_scatter_fill
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

fn render_clouds_volume(coord: vec3f) -> vec4f {
    // Four distinct, seamlessly tileable 3-D bases. The runtime samples them
    // through different physical periods/domains to form broad organization,
    // cumulus bodies, boundary erosion, and a second macro octave.
    // Mass formation is deliberately low-bandwidth: at the authored 8 km
    // period these octaves are 4 km / 2 km / 1 km. Higher frequencies belong
    // in boundary erosion; putting them through the coverage threshold turns
    // one cloud into a field of detached micro-cloudlets.
    let perlin = common::tilable_perlin_fbm(coord, 3, 2.0);
    let cellular = common::tilable_voronoi(coord + vec3f(0.17, 0.31, 0.07), 2, 3.0);
    // A single smooth cell scale serves two physical roles: ~1 km lobes when
    // sampled with the base period and ~55 m boundary cuts with the detail
    // period. Multi-octave content here polluted solid interiors with a fine
    // cellular texture before the runtime erosion stage.
    let erosion = common::tilable_voronoi(coord + vec3f(0.53, 0.11, 0.43), 1, 8.0);
    let macro_noise = common::tilable_perlin_fbm(coord + vec3f(0.29, 0.47, 0.61), 3, 1.0);
    return clamp(vec4f(perlin, cellular, erosion, macro_noise), vec4f(0.0), vec4f(1.0));
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

    // Interleaved-gradient phase for the view march; temporal history removes
    // the residual while the offset prevents coherent horizon bands.
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
    let xyz = vec3f(invocation_id) + vec3f(0.5);
    let volume = render_clouds_volume(xyz / WORLEY_RESOLUTION_F32);
    textureStore(clouds_worley_texture, invocation_id, volume);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    if (invocation_id.x >= u32(config.render_resolution.x)
        || invocation_id.y >= u32(config.render_resolution.y)) {
        return;
    }
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
