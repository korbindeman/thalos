#import bevy_open_world::common
#import thalos::atmosphere::{
    cloud_surface_shape,
    cloud_march_step_m,
    cloud_cell_field,
    cloud_cell_style,
    cloud_far_ownership,
    cloud_march_stop_m,
    cloud_strata_warp,
    CLOUD_MARCH_REACH_M,
    CLOUD_MARCH_MIN_STEP_M,
    CLOUD_MARCH_FADE_FRACTION,
}

/// Radians subtended by one cloud-target pixel. `inverse_camera_projection` is
/// the inverse of the projection, so element [0][0] is 1/P00 — the half-width
/// tangent at unit depth; two of those over the target width is the per-pixel
/// angle. Derived rather than uniform-fed so it cannot drift from the
/// projection the same march is tracing. This is the LOD driver for the whole
/// cloud system: distance is not footprint (see the thalos::atmosphere march
/// contract).
fn cloud_pixel_angle() -> f32 {
    return 2.0 * config.inverse_camera_projection[0][0]
        / max(config.render_resolution.x, 1.0);
}

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
// Moving history is only a stabilizer. A cloud ray integrates a translucent
// interval, so one nearest-hit motion vector cannot justify history-dominant
// blending the way a single opaque surface can.
const MOVING_REPROJECTION_STRENGTH = 0.45;
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
// ── Cheap/full-density adaptive march, distance-banded ───────────────────────
// Clear air advances at the per-band broad cadence (`cloud_march_step_m`,
// 600 m near → 4.8 km at range) while evaluating the smooth broad mass only.
// A meaningful hit backs up one interval and switches to the full-density
// cadence at 1/5 of the local broad step; four empty fine samples return to
// coarse mode. Footprint-stretched steps are safe ONLY because the density
// field is band-limited ahead of every step increase (erosion retires as the
// refine step grows past the detail scale; the shape spectrum narrows; past
// ~90 km density is the derived homogenized field) — see the contract block
// in thalos::atmosphere and BL-20260724T003705Z. Steep rays at range clamp
// the step to a radial rate that resolves thin layers (`MAX_RADIAL_STEP_M`);
// such rays have geometrically short in-shell segments, so the clamp is
// bounded work.
const REFINE_STEP_FRACTION = 0.2; // fine cadence = broad cadence × this
const REFINE_EMPTY_LIMIT = 4u;
const BROAD_HIT_DENSITY_FRACTION = 0.01;
// Compile-time safety cap; config selects ≤ this. Sized for the planetary
// reach budgets (2026-07-29): with the broad-only budget accounting and the
// footprint-relaxed refine cadence, ~660 broad probes carry a grazing ray
// from a ~40 km shell entry to ~300 km — beyond which cells are marginal at
// the compute resolution and the far tier's derived sharp response owns the
// tail. The former 224 was sized for the 600 m step floor era and halved in
// distance when the floor halved (INC-20260729T012803Z context note).
const MAX_RAY_STEPS_CAP = 672u;
const MAX_RADIAL_STEP_M = 350.0;  // vertical resolution floor for steep rays
// Rays ENTERING the shell beyond the entry-fade window dissolve out of the
// near estimator entirely; the composite's weather-column band march owns
// them (partition of unity via the shared thalos::atmosphere march contract).
// Unlike the pre-CLOUD-6 haze-out these clouds are replaced, not deleted.
const DETAIL_FILTER_BEGIN = 0.25;
const DETAIL_FILTER_END = 0.50;
const CAMERA_CUT_DELTA = 0.05;
// Airlight restoration: in-scattered air between the camera and each cloud
// sample. The composite draws clouds OVER the already-integrated sky, which
// attenuates the foreground airlight by cloud opacity; re-adding it here keeps
// distant clouds behind a natural blue/warm veil instead of a soot-brown
// extinction-only tint (the "dirty distant deck" failure).
const AIRLIGHT_GAIN = 1.35;
// Convective cell aspect (round-9). The authored `clouds_base_shape_scale_m`
// describes the SHEET scale; a cumulus lobe is taller than it is wide, so
// convective columns stretch the domain's RADIAL axis — a metre of climb moves
// the sample only 1/stretch of a metre through the noise, so one lobe stays
// coherent from base to top instead of decorrelating into a stack of unrelated
// blobs. `march_column` (fill_lut.rs) mirrors both; keep them in lockstep.
//
// The horizontal period stays at the authored scale, and the reason is a
// measured failure: narrowing it to 0.42× (≈3.4 km cells) made the near field
// beautifully solid AND painted the cruise deck as a regular lattice of
// identical puffs in rows — the 64³ volume's Cartesian repeat, which becomes
// planet-visible once many periods fit on screen (the ADR-20260722T141000Z
// "rows" family, from the tile side this time). Apparent cell size must come
// from the FORMATION THRESHOLD picking peaks out of a wide period — one lobe
// per period is what broken cumulus actually looks like — not from shortening
// the period. If narrower cells are ever wanted, the repeat has to be broken
// first (a second octave at a non-commensurate period), not shortened.
const CONVECTIVE_WIDTH_SCALE = 1.0;
const CONVECTIVE_STRETCH = 1.9;
// How much the periodic Cartesian volume perturbs the aperiodic cell field.
// It is a SUB-CELL sculptor now: it adds billow and vertical structure inside
// a cell that `cloud_cell_field` placed, and it retires at range. Raising it
// toward 1.0 hands cell-scale organization back to an 8 km repeat, which is
// the "rows" failure above. Mirrored in fill_lut.rs — keep in lockstep.
const SUBCELL_SHAPE_WEIGHT = 0.55;

struct Config {
    clouds_base_shape_scale_m: f32,
    clouds_raymarch_steps_count: u32,
    clouds_bottom_height: f32,
    clouds_top_height: f32,
    clouds_coverage: f32,
    clouds_density: f32,
    clouds_detail_scale_m: f32,
    surface_density_coupling: f32,
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
    cloud_albedo: vec4f,
    camera_translation: vec3f,
    time: f32,
    reprojection_strength: f32,
    render_resolution: vec2f,
    frame_index: u32,
    history_epoch: u32,
    sparse_march: u32,
    inverse_camera_view: mat4x4f,
    inverse_camera_projection: mat4x4f,
    wind_displacement: vec3f,
    // Formation-threshold curve vs strata density: 8 piecewise-linear nodes
    // (node i at env i/7), derived per body by
    // `fill_lut::derive_fill_calibration` so the near tier's areal fill
    // tracks the strata density (the far tier's authority).
    fill_threshold0: vec4f,
    fill_threshold1: vec4f,
    // Cloud sun-transmittance cascade placement, body-fixed. Mirror of
    // `CloudShadowFrame` (shadow_frame.rs) — the receivers' `CloudShadowBlock`
    // carries the identical numbers, so producer and consumer cannot drift.
    // xyz = map centre on the reference sphere, w = half extent (m).
    shadow_origin: vec4f,
    // xyz = +u tangent, w = metres per texel.
    shadow_axis_u: vec4f,
    // xyz = +v tangent, w = 1 when the cascade is live.
    shadow_axis_v: vec4f,
    // xyz = reference-plane normal, w = sun elevation cosine at the centre.
    shadow_up: vec4f,
};

@group(0) @binding(0) var<uniform> config: Config;

@group(1) @binding(0) var clouds_render_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(1) var clouds_worley_texture: texture_storage_3d<rgba32float, read_write>;
// Per-pixel cloud-hit span (metres from the camera; MAX_DISTANCE where the ray
// hit no cloud). `r` = nearest hit, `g` = the far end of the ray's
// optical-depth-weighted slab (`slab_far`, below). The game's body_sky
// composite reads BOTH for true depth occlusion against terrain / the ship
// hull — the near hit alone cannot say how much of the integrated cloud lies in
// front of a depth sample, which is what made distant cloud go see-through
// wherever terrain sat behind it. The raymarch's own history reads use
// `history_distance_texture` below and key on `r` only.
@group(1) @binding(2) var cloud_distance_texture: texture_storage_2d<rg32float, write>;
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
// Canonical surface-space broad shape at four normalized-height strata. It is
// generated/versioned with weather and shares weather_sampler's filter state.
@group(1) @binding(7) var surface_density_texture: texture_cube<f32>;
// View-anchored cloud sun-transmittance cascade (CLOUD-5 / W2 near tier).
// r = fraction of the sun beam that survives the whole deck to reach the
// reference plane at this texel. Written by `cloud_shadow` below; sampled by
// every surface receiver through `thalos::cloud_shadow`.
@group(1) @binding(8) var cloud_shadow_texture: texture_storage_2d<rgba16float, write>;
struct Ray {
    step_distance: f32,
    dir_length: f32,
    start: f32,
    end: f32,
}

struct RaymarchResult {
    dist: f32,
    // Far end of the equivalent uniform slab this ray's extinction occupies.
    // See `slab_far_distance`.
    slab_far: f32,
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

// Derived formation threshold vs strata density (see the Config fields).
fn formation_threshold(env: f32) -> f32 {
    // `var`: naga requires a reference (not a value) for dynamic indexing.
    var nodes = array<f32, 8>(
        config.fill_threshold0.x, config.fill_threshold0.y,
        config.fill_threshold0.z, config.fill_threshold0.w,
        config.fill_threshold1.x, config.fill_threshold1.y,
        config.fill_threshold1.z, config.fill_threshold1.w,
    );
    let t = clamp(env, 0.0, 1.0) * 7.0;
    let i = u32(min(t, 6.0));
    let f = t - f32(i);
    return mix(nodes[i], nodes[i + 1u], f);
}

// `layer_height` is LAYER-RELATIVE (0 = local base, 1 = local top, from the
// same weather texel's base/top channels); outside the layer the shared shape
// returns a hard zero. See cloud_surface_shape in thalos::atmosphere.
fn sample_surface_density(n: vec3f, layer_height: f32) -> f32 {
    let strata = textureSampleLevel(surface_density_texture, weather_sampler, n, 0.0);
    return cloud_surface_shape(strata, layer_height);
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

fn weather_phase_offset(weather: vec4f, period: f32) -> vec3f {
    // Continuous coverage/base/top channels bend periodic 3-D source domains
    // into the non-periodic planet weather field. The categorical type channel
    // is deliberately excluded so a type boundary cannot pop the volume.
    return period * vec3f(
        1.35 * (weather.r - 0.5) + 0.65 * (weather.b - 0.5),
        -1.10 * (weather.a - 0.5) + 0.55 * (weather.r - 0.5),
        1.20 * (weather.r - 0.5) - 0.75 * (weather.b - 0.5)
            + 0.45 * (weather.a - 0.5),
    );
}

// Typed volumetric density. Weather changes continuously along the view ray;
// every base/detail sample is full 3-D world-space structure. Empty-space
// acceleration must use a truly conservative volume in a future optimization:
// heuristically resuming at base/top or macro thresholds produced stable
// horizontal isosurfaces at grazing angles (INC-0011).
// `filter_m` is the caller's sampling scale in metres — the march step, the
// sun-tap spacing, or the shadow-texel footprint, whichever applies. It is the
// ONLY LOD input: camera distance does not appear in this function at all.
// Everything that has to retire with scale retires against it — the cell
// field's octaves, the periodic Cartesian sub-cell sculptor, erosion, and the
// formation-edge width — so the field band-limits BEFORE the sampler that needs
// it coarsens (ADR-20260721T033055Z's conservative-bounds requirement), and
// range degrades cells into COARSER CELLS rather than into a mean.
fn get_cloud_map_density(
    pos: vec3f,
    shell_height: f32,
    weather: vec4f,
    surface_density: f32,
    detail_weight: f32,
    filter_m: f32,
    macro_noise: f32,
    formation: f32,
) -> f32 {
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

    // Sub-cell sculptor LOD, derived from the ONE sampling scale rather than
    // from camera distance. The periodic Cartesian volume adds lobes well under
    // its authored period, so it retires as `filter_m` climbs toward that
    // period; the aperiodic cell field carries morphology from there on and
    // band-limits itself the same way. There is no longer a `coarse` ladder —
    // distance never enters the density function.
    let sculpt_fade = smoothstep(
        0.10 * max(config.clouds_base_shape_scale_m, 500.0),
        0.55 * max(config.clouds_base_shape_scale_m, 500.0),
        filter_m,
    );
    let coupling = clamp(config.surface_density_coupling, 0.0, 1.0);
    let env = clamp(surface_density, 0.0, 1.0);
    // Occupancy gates FORMATION, never extinction (round-9; user verdict
    // 2026-07-25 "clouds are super ghost-ish"). `env` is the strata cube's
    // AREAL FRACTION — the share of a ~5 km weather texel that is cloudy, see
    // `cloud_surface_density_cpu` — so scaling extinction by it thinned every
    // cell in proportion to how sparse its NEIGHBOURHOOD is: a 30 %-coverage
    // region rendered as one 30 %-opaque film rather than solid cells with
    // clear air between them. It also double-counted, because the derived
    // `formation_threshold(env)` above already spends the same occupancy on
    // deciding where cloud exists. This is the grey-veil error
    // `weather_column_from_texel` warns about, reached from the near tier's
    // side. What remains is a hard clear-air cut, so the Cartesian shape noise
    // cannot grow cloud where the producer authored none; the fill contract is
    // preserved because the spawn-time Monte-Carlo re-fits the threshold curve
    // against this exact math (E[column alpha] still tracks strata mean — it
    // now reaches it with FEW OPAQUE columns instead of many translucent ones).
    // The toe matters as much as the removal: a gate that ramps from zero
    // turns a 3 %-occupancy region into half-density fog across the whole
    // region, which is the ghost read arriving by a different door (the
    // massif framing is exactly such a low-occupancy lane). Stay at hard zero
    // through genuinely clear air and reach full extinction by the time a
    // texel holds any real cloud.
    let formation_gate = smoothstep(0.02, 0.08, env);

    let bottom_softness = max(config.clouds_bottom_softness, 0.01);
    // Round-7 morphology: convective tops are SCULPTED, not faded. The former
    // cumulus/storm profiles bled density out over the top 30% of every
    // column, which capped each weather texel with the same soft flat lid —
    // the "flat sheets" verdict. Tops are now carved by the height-rising
    // dome threshold below (a noise isosurface: strong lobes tower, weak
    // lobes stay squat) and the profile keeps only a thin condensation skin.
    // Stratus stays a genuine sheet. Mirrored in `march_column` (fill_lut.rs)
    // and `cloud_surface_density_cpu` — keep the three in lockstep.
    let stratus_profile = smoothstep(0.0, bottom_softness * 0.45, h)
        * (1.0 - smoothstep(0.72, 1.0, h));
    let cumulus_profile = smoothstep(0.0, bottom_softness * 0.75, h)
        * (1.0 - smoothstep(0.93, 1.0, h));
    let storm_profile = smoothstep(0.0, bottom_softness * 0.35, h)
        * (1.0 - smoothstep(0.94, 1.0, h));
    let vertical_profile = stratus_profile * stratus_w
        + cumulus_profile * cumulus_w
        + storm_profile * storm_w;

    let column_tall = smoothstep(0.30, 0.65, local_top - local_base);
    let radius = length(pos);
    let up_shape = pos / max(radius, 1.0);

    // ── Cell-scale morphology: the aperiodic column field ────────────────────
    // This is the formation carrier at EVERY range. It replaces the old
    // arrangement where cell structure lived only in the periodic Cartesian
    // volume and was therefore deleted past ~90 km, leaving a flat sheet
    // (2026-07-25 verdict: "incoherent soupy blobby mess"). Nothing the
    // weather cube can store reaches this scale — 4.9 km texels, ~15–25 km of
    // authored content — so the cells have to be analytic and aperiodic.
    // Morphology is a property of the PLACE, not a global constant: rolls,
    // round cells, coarse storm clusters and lane-cut sheets are all the same
    // field under a different style (`cloud_cell_style`). Resolved per sample
    // rather than per ray because a 300 km grazing ray genuinely crosses style
    // regions; it costs one extra low-frequency noise fetch.
    let cell_style = cloud_cell_style(up_shape, weather.g);
    let cell = cloud_cell_field(up_shape, config.planet_radius, filter_m, cell_style);

    // ── Sub-cell sculpting: the periodic Cartesian volume ────────────────────
    // Demoted to what it is actually good at — intra-cell billow and vertical
    // structure inside a cell the field above already placed. It retires with
    // `c2` because its 8 km repeat is the term that cuts planet-visible rows
    // once many periods fit on screen; retiring it costs morphology only at
    // ranges where its lobes are sub-footprint anyway. Skipping the fetch
    // entirely also keeps the long-range broad probes cheaper than near ones.
    var shape = cell;
    let sub_weight = SUBCELL_SHAPE_WEIGHT * (1.0 - sculpt_fade);
    var anvil_base = cell;
    if (sub_weight > 1.0e-3) {
        // Convective morphology is ANISOTROPIC (round-9; user ask 2026-07-25,
        // "vertical development", MSFS/cumulonimbus reference). A radially
        // STRETCHED domain keeps one lobe coherent over its whole depth
        // instead of decorrelating into a stack of unrelated blobs. Sheets
        // keep the isotropic period — a stratus deck genuinely is wide and
        // flat. `convective_w` is continuous in the weather type channel, so
        // no type boundary can pop the volume. The horizontal period stays
        // WIDE: narrowing it is the rejected round-9 experiment that painted
        // the cruise deck as a lattice of identical puffs in rows.
        // Mirrored in `march_column` (fill_lut.rs) — keep the two in lockstep.
        let convective_w = clamp(cumulus_w + storm_w, 0.0, 1.0);
        let shape_scale = max(config.clouds_base_shape_scale_m, 500.0)
            * mix(1.0, CONVECTIVE_WIDTH_SCALE, convective_w);
        let alt_local = radius - (config.planet_radius + config.clouds_bottom_height);
        // Pull the domain back along the local vertical in proportion to
        // altitude: a metre of climb then moves the sample only 1/stretch of a
        // metre through the noise, elongating every feature vertically.
        let shape_domain = pos
            - up_shape * (alt_local * (1.0 - 1.0 / CONVECTIVE_STRETCH) * convective_w);
        let broad = cloud_volume(
            rotated_domain(shape_domain)
                + weather_phase_offset(weather, shape_scale)
                + vec3f(1800.0, -4200.0, 900.0),
            shape_scale,
        );
        // The erosion channel stays out of the solid body: promoting its small
        // cells into `shape` fragmented the volume into screen-space stipple
        // instead of adding readable billows. Spectrum follows column height:
        // tall (congestus/storm) columns weight the low-frequency channels so a
        // tower reads as ONE coherent mass with large billows; squat
        // fair-weather columns keep the small-lobe mix.
        let shape_squat = broad.r * 0.52 + broad.g * 0.24 + broad.a * 0.24;
        let shape_tall = broad.r * 0.64 + broad.g * 0.06 + broad.a * 0.30;
        let sub = mix(shape_squat, shape_tall, max(column_tall, sculpt_fade));
        // Additive and mean-preserving: as `sub_weight` retires, the field's
        // distribution keeps its mean AND its variance, so the derived fill
        // calibration stays valid and contrast does not step at the band edge.
        shape = cell + (sub - 0.5) * sub_weight;
        anvil_base = cell + ((broad.r * 0.72 + broad.a * 0.28) - 0.5) * sub_weight;
    }

    // Formation authority is NON-PERIODIC at every scale: the strata cube owns
    // synoptic/mesoscale occupancy (≥ ~15 km), `cloud_cell_field` owns cell
    // scale (~1–5 km), and only sub-cell sculpting comes from the periodic
    // Cartesian volume — which is why no repeat can organize clouds at a scale
    // the eye reads as a pattern (ADR-20260722T141000Z; 2026-07-23 user
    // verdict). `macro_noise` remains only as a faint sub-dominant variety
    // term.
    // The threshold makes the near tier's COLUMN areal fill (seen from above)
    // track the strata density — the contract the far tier reads directly.
    // The curve is DERIVED at body spawn by a CPU Monte-Carlo mirror of this
    // exact density math over the real weather cube
    // (`fill_lut::derive_fill_calibration` — keep the two in lockstep); the
    // same derivation emits the far tier's opacity response, so the two tiers
    // pair by construction. Hand-fitted constants here failed twice: the
    // exceedance distribution is too steep to tune from captures, and the
    // last fit was against a corrupted cube (INC-20260723T221126Z).
    let threshold_surface = formation_threshold(env) + (0.5 - macro_noise) * 0.05;
    // Capture-only legacy branch (surface_density_coupling = 0): the old
    // Cartesian-organized threshold, kept for A/B attribution.
    let threshold_legacy = mix(0.58, 0.30, cov)
        + (0.5 - macro_noise) * 0.07
        + (0.5 - formation) * 0.17
        + (0.35 - surface_density) * 0.08;
    let threshold = mix(threshold_legacy, threshold_surface, coupling);
    // Dome sculpting: the threshold rises QUADRATICALLY with height for the
    // convective types, so each lobe's top is where its own shape noise dips
    // under the rising bar — a per-lobe carved dome, cheap (no
    // transcendentals) and calibration-safe (the spawn-time fit re-derives
    // the formation threshold against this exact math, and the term is near
    // zero at the base where areal fill is decided). Tall congestus/storm
    // columns keep more mass with height so towers stay coherent. Mirrored
    // in `march_column` (fill_lut.rs) and `cloud_surface_density_cpu` — keep
    // the three in lockstep. (`column_tall` is declared above the cell field.)
    let dome = h * h;
    // Dome coefficients are scaled against the spread of `shape` — see the
    // derivation on `cloud_surface_density_traced` (solar_system_state.rs).
    // The former 0.42 was ~4.7σ and cut every column's top half off, which
    // rendered the deck as flat pancakes. Do not retune one mirror alone.
    let vertical_narrow = h * 0.012 * stratus_w
        + dome * (0.130 * cumulus_w + 0.093 * storm_w) * (1.0 - 0.45 * column_tall);
    var mass = shape - threshold - vertical_narrow;

    // Cumulonimbus anvils broaden again near the tropopause, but only where
    // the storm weather channel permits them.
    // Blend by the gate; never `max` against a gate-scaled value. That form
    // collapses to `max(mass, 0.0)` wherever the gate is zero — a mass floor,
    // not an anvil. Harmless here (this tier realizes with
    // `smoothstep(0.0, edge_softness, mass)`, so a zero floor is zero density)
    // but the CPU producer's realization is centred on zero, where the same
    // line emitted a planet-wide 0.5 cloud floor. Fixed in all three mirrors
    // together — see `cloud_surface_density_traced` (solar_system_state.rs).
    let anvil_profile = smoothstep(0.62, 0.76, h) * (1.0 - smoothstep(0.90, 1.0, h));
    let anvil_shape = anvil_base - (threshold - 0.06);
    mass = mix(mass, max(mass, anvil_shape), anvil_profile * storm_w);

    // Fine 3-D Worley erosion is strongest only near the boundary, preserving
    // solid cores for deep self-shadow while cutting cauliflower detail into
    // silhouettes. Detail moves slowly in a decorrelated domain.
    let boil = vec3f(0.0, config.wind_displacement.y, config.wind_displacement.z);
    // Wide, gentle erosion falloff: a narrow 0.04–0.18 window drew its outer
    // iso-contour as visible "fingerprint" rings inside big lobes once the
    // softer CLOUD-4 lighting stopped hiding them.
    let edge = 1.0 - smoothstep(0.02, 0.34, mass);
    // Erosion retires with the sculptor (the caller's footprint-keyed
    // `detail_weight` also fades it as the refine step outgrows the authored
    // detail scale — either gate suffices).
    let detail_weight_lod = detail_weight * (1.0 - sculpt_fade);
    if (edge * detail_weight_lod > 1.0e-3) {
        let detail = cloud_volume(
            rotated_domain(pos + boil) + vec3f(270.0, -610.0, 130.0),
            // Channel B contains eight primary Worley cells across the stored
            // tile. `clouds_detail_scale_m` describes one authored physical
            // erosion feature, not the whole tile period.
            max(config.clouds_detail_scale_m, 50.0) * 8.0,
        );
        // Height-typed erosion character: near the base the Worley field is
        // FLIPPED so undersides shred into wisps; on domes it cuts
        // cauliflower billows, slightly stronger up high so tops crisp.
        // Mirrored in fill_lut's `SampleRecord::erode` — keep in lockstep.
        let erode_src = mix(detail.b, 1.0 - detail.b, smoothstep(0.10, 0.32, h));
        mass -= erode_src * (0.80 + 0.55 * h) * edge * detail_weight_lod
            * config.clouds_detail_strength * 0.55;
    }

    // Formation-edge softening with range. This was `0.30` at c1 — keyed to
    // the BROAD band step even though density is only ever integrated at the
    // 0.2× refine cadence, so it over-softened by ~5× and turned every distant
    // cell into a low-alpha skirt. That was the round-9 `massif-ridge` milky
    // veil's prime remaining suspect, and softening is the one thing that
    // cannot survive here: the whole point of the cell field is that contrast
    // is preserved with range, and a widening edge erases it just as
    // effectively as rendering the mean did.
    let edge_softness = mix(max(config.clouds_base_edge_softness, 0.015), 0.075, sculpt_fade);
    let shaped = smoothstep(0.0, edge_softness, mass);
    // The strata field owns planet-scale occupancy in every projection — but it
    // spends that authority through the DERIVED formation threshold, not by
    // scaling extinction (see `formation_gate`). Far consumers sample the same
    // strata payload at footprint mips AND the same cell field, so range
    // removes sub-cell sculpting and nothing else.
    return max(
        shaped
            * vertical_profile
            * mix(1.0, formation_gate, coupling)
            * config.clouds_density,
        0.0,
    );
}

fn get_normalized_height(pos: vec3f) -> f32 {
    let clouds_height = config.clouds_top_height - config.clouds_bottom_height;
    return (length(pos) - (config.planet_radius + config.clouds_bottom_height)) / clouds_height;
}

// Directional sun optical depth through the FILTERED density field.
// Two properties are load-bearing:
// - `detail_weight = 0`: the probe samples only the smooth typed broad mass.
//   Probing the fine erosion field keyed every sample's whole direct term on
//   ~55 m noise, which rendered as dirty cellular charcoal patches across
//   sunlit lobes (the acceptance-bar "soot" failure).
// - The tap ladder is jittered per pixel: fixed exponential tap distances
//   through the typed vertical profile previously banded into nested strata
//   (see `init_cloud_appearance`); decorrelating the ladder start converts
//   that into benign noise the temporal accumulation removes.
// Returns optical depth τ ≥ 0 so multi-scatter octaves can each apply their
// own attenuation `exp(-τ · c_i)` without a per-sample log.
fn volumetric_sun_depth(
    origin: vec3f,
    weather: vec4f,
    surface_density: f32,
    macro_noise: f32,
    formation: f32,
    jitter: f32,
    filter_m: f32,
) -> f32 {
    var ray_step_size = config.clouds_shadow_raymarch_step_size;
    var distance_along_ray = ray_step_size * (0.25 + 0.5 * jitter);
    var optical_depth = 0.0;

    for (var step: u32 = 0; step < config.clouds_shadow_raymarch_steps_count; step++) {
        let pos = origin + config.sun_dir.xyz * distance_along_ray;
        let normalized_height = get_normalized_height(pos);
        if (normalized_height > 1.0) { return optical_depth; };

        let density =
            get_cloud_map_density(
                pos,
                normalized_height,
                weather,
                surface_density,
                0.0,
                filter_m,
                macro_noise,
                formation,
            );
        optical_depth += density * ray_step_size;
        ray_step_size *= config.clouds_shadow_raymarch_step_multiply;
        distance_along_ray += ray_step_size;
    }

    return optical_depth;
}

fn henyey_greenstein(ray_dot_sun: f32, g: f32) -> f32 {
    let g_squared = g * g;
    return (1.0 - g_squared) / pow(1.0 + g_squared - 2.0 * g * ray_dot_sun, 1.5);
}

// ── CLOUD-4 atmosphere coupling ──────────────────────────────────────────────
// Normal rendering samples the SAME Bevy transmittance and sky-view LUTs that
// render the rocky-body atmosphere. The analytic branch remains solely as the
// deterministic fallback for the legacy custom-atmosphere A/B.

const ATMOS_SCALE_HEIGHT_M = 8000.0;
// Optical depth per metre at sea level × scale height ≈ column optical depth.
// Tuned so noon overhead is nearly white and horizon sun goes amber/red.
const ATMOS_BETA_SEA = vec3f(5.5e-6, 1.3e-5, 3.2e-5);

/// Approximate air-mass for a ray from radius `r` toward direction `mu =
/// cos(zenith)`. Avoids trig-heavy Chapman; matches low-sun reddening well.
fn atmosphere_air_mass(r: f32, mu: f32) -> f32 {
    let h = max(r - config.planet_radius, 0.0);
    let dens = exp(-h / ATMOS_SCALE_HEIGHT_M);
    // Schueler-style: 1/(mu + k) with a soft floor past the terminator.
    let m = 1.0 / max(mu + 0.15, 0.055);
    return dens * m;
}

/// Sun → sample transmittance (RGB). Applied on top of the CPU-fed sun colour
/// so low solar elevation and low-altitude samples both lose blue.
fn atmosphere_sun_transmittance(sample_pos: vec3f) -> vec3f {
    let r = max(length(sample_pos), config.planet_radius);
    let up = sample_pos / r;
    let mu = dot(up, config.sun_dir.xyz);
    // The planet itself occludes the sun below the geometric horizon.
    let horizon_mu = -sqrt(max(1.0 - (config.planet_radius * config.planet_radius) / (r * r), 0.0));
    if mu <= horizon_mu {
        return vec3f(0.0);
    }
    let am = atmosphere_air_mass(r, mu);
    // Column τ ≈ β * H * air_mass_factor; β_sea already per-metre, so × H.
    let tau = ATMOS_BETA_SEA * ATMOS_SCALE_HEIGHT_M * am;
    return exp(-tau);
}

/// Sample → camera transmittance. Foreground air between a lit cloud sample
/// and the viewer dims/reddens the in-scatter so the BodySky composite does
/// not treat cloud light as if it originated at the camera.
fn atmosphere_view_transmittance(sample_pos: vec3f, camera_pos: vec3f) -> vec3f {
    let delta = sample_pos - camera_pos;
    let dist = length(delta);
    if (dist < 1.0) {
        return vec3f(1.0);
    }
    // Midpoint density proxy — good enough for ≤50 km cloud reach.
    let mid = 0.5 * (sample_pos + camera_pos);
    let h = max(length(mid) - config.planet_radius, 0.0);
    let dens = exp(-h / ATMOS_SCALE_HEIGHT_M);
    let tau = ATMOS_BETA_SEA * dens * dist;
    return exp(-tau);
}

const INV_PI = 0.31830989;
const INV_FOUR_PI = 0.07957747;

/// Per-octave dual-lobe phase values (Nubis/Frostbite multi-scatter octaves).
/// Evaluated ONCE PER RAY; per sample each octave is attenuated by its own
/// `exp(-τ_sun · c_i)`, so deep shade retains soft wide-lobe fill from the
/// later octaves instead of multiplying the whole direct term toward black
/// (the old `0.04 + 0.96 · shadow` fill collapsed shaded cores to charcoal).
fn multi_scatter_lobes(cos_theta: f32, g_fwd: f32, g_bwd: f32, lerp_g: f32) -> vec3f {
    var lobes = vec3f(0.0);
    var gf = g_fwd;
    var gb = g_bwd;
    for (var i = 0; i < 3; i++) {
        let lobe = mix(
            henyey_greenstein(cos_theta, gf),
            henyey_greenstein(cos_theta, gb),
            lerp_g,
        );
        // HG omits 1/(4π); normalize into scene units and bound the forward peak.
        lobes[i] = min(lobe * INV_FOUR_PI, 2.2);
        gf *= 0.5;
        gb *= 0.5;
    }
    return lobes;
}

// Octave energy weights and shadow-attenuation exponents. Energy drops per
// octave; attenuation drops faster so multiple scattering "leaks around"
// occluders (Wrenninge/Nubis approximation). Keep the higher octaves modest:
// at (1.0, 0.52, 0.26) deep shade retained ~40% of lit energy and lobes went
// flat cotton with no readable sun side.
const MS_OCTAVE_WEIGHTS = vec3f(1.0, 0.34, 0.13);
const MS_OCTAVE_EXTINCTION = vec3f(1.0, 0.25, 0.06);

// ── Why clouds are white ─────────────────────────────────────────────────────
// A water cloud scatters conservatively (ϖ ≈ 0.9999) through a strongly
// forward phase, so an optically thick sunlit cell reaches the diffusion
// limit: it returns ~0.75–0.85 of the incident flux, and that light leaves
// close to isotropically. Its radiance is therefore ≈ A·E/π, roughly SIX
// times the single-scattering side lobe (p(90°)·E ≈ 0.043·E here).
//
// Single scattering alone renders a cloud no brighter than the sky ambient
// filling it — grey-blue mud that takes its chroma from the ambient rather
// than the sun. That was the defect: the octave sum was divided by Σw, which
// makes it a weighted AVERAGE of phase values, so the "multiple-scattering
// octaves" only reshaped the lobe and added none of the energy they name.
// Measured on `cloud_cruise` (2026-07-24 capture): brightest near-tier cloud
// pixel 0.30 display luminance against 0.49–0.73 for the sky behind it and
// 0.73 for the far tier's rendering of the same field.
//
// The replacement splits the source term the way the physics does:
//   single  — exact normalized phase against the unattenuated beam; owns the
//             forward glare and the silver lining;
//   multi   — an isotropic-equivalent reservoir at the cloud's diffusion
//             albedo, whose DEPTH response is the surviving job of the wider
//             octaves (they attenuate far more slowly, so light leaks around
//             occluders instead of multiplying cores to charcoal).
// `CLOUD_MS_ALBEDO` is a physical property of the medium, not a brightness
// knob: it is what makes a lit cell as bright as a white Lambertian surface
// facing the same sun, which is the anchor the far tier must be matched to.
const CLOUD_MS_ALBEDO = 0.80;
// Residual anisotropy of the multiply-scattered reservoir: it is not perfectly
// isotropic — the sun side of a cell stays brighter. 0 = flat, 1 = the widest
// octave's full lobe.
const MS_ANISO = 0.7;

/// Silver-lining / powder: thin edges facing the light brighten; the same thin
/// path looking *away* from the light darkens (HZD powder). Restrained: the
/// former 0.85 away-darkening painted lobes near-black and read as dirt rather
/// than shading. Caveat: the 0.35/0.35 constants (and MS_ANISO above) were
/// tuned while the phase argument was negated — that "dirt" was landing on the
/// SUNWARD side — so they are retune candidates against sunset captures now
/// that the geometry is correct.
fn powder_term(density_fraction: f32, cos_theta: f32) -> f32 {
    let d = clamp(density_fraction, 0.0, 1.0);
    let powder = 1.0 - exp(-d * 2.0);
    // cos_theta = ray·sun: +1 looking toward the sun (silver lining).
    let toward_sun = clamp(cos_theta, 0.0, 1.0);
    let away = clamp(-cos_theta, 0.0, 1.0);
    return mix(1.0, powder, away * 0.35) * (1.0 + toward_sun * d * 0.35);
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
        return Ray(600.0, max_dist + 1.0, max_dist + 1.0, max_dist);
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
    // Cap the marched segment at the banded reach; the far estimator owns
    // everything beyond (shared thalos::atmosphere march contract).
    seg_end = min(min(seg_end, max_dist), seg_start + CLOUD_MARCH_REACH_M);

    // Jitter only the first fine sample. Coarse probes back up before they hand
    // a hit to the full-density cadence, so they cannot skip the entry edge.
    // `MIN_STEP` is the right scale here: the caller re-derives the real
    // footprint/budget step per sample, and this only phases the first one.
    let step = CLOUD_MARCH_MIN_STEP_M * REFINE_STEP_FRACTION;
    // The full-step temporal phase is safe now that the minimum density
    // feature is larger than one march step. It prevents near-horizontal rays
    // from stacking their samples into coherent bands; history averages the
    // phase instead of trying to hide under-resolved density.
    let dir_length = seg_start - step * jitter;

    return Ray(step, dir_length, seg_start, seg_end);
}

/// Far end of the equivalent uniform slab for a ray whose extinction has total
/// optical depth `tau_total` with distance-moment `tau_moment`, entering at
/// `near`.
///
/// The composite has to answer "how much of this ray's cloud is in front of the
/// depth buffer?" from what the march can afford to store. One distance cannot
/// answer it, and the predecessor's stand-in — a CONSTANT 5.4 km denominator —
/// is what made every cloud with terrain behind it render see-through: a 900 m
/// deep cumulus with the ground a kilometre behind it was drawn at ~1/5 of its
/// opacity even though ALL of it is in front of that ground
/// (INC-20260729T051500Z).
///
/// Matching the first moment of the real extinction profile is the cheapest
/// summary that gets the limits right: terrain beyond the cloud partitions to
/// full opacity, terrain in front to none. Weighting by optical depth rather
/// than taking the last hit also keeps a single wispy tail sample — or a second
/// cloud 20 km further down the same ray — from stretching the slab across
/// empty air.
fn slab_far_distance(near: f32, tau_total: f32, tau_moment: f32) -> f32 {
    if (tau_total <= 1.0e-6) {
        return near;
    }
    let centroid = tau_moment / tau_total;
    return near + 2.0 * max(centroid - near, 0.0);
}

fn raymarch(ray_origin: vec3f, ray_dir: vec3f, max_dist: f32, jitter: f32) -> RaymarchResult {
    let ray = get_ray(ray_origin, ray_dir, max_dist, jitter);
    let pixel_angle = cloud_pixel_angle();

    // The far/orbital projection owns this ray only once cell-scale morphology
    // is genuinely sub-pixel (whole-disc / map framings). Keying this to the
    // shell ENTRY DISTANCE instead — the old 240–300 km window — deleted the
    // volumetrics in the middle of every ascent, which is exactly where the
    // deck is best resolved (345 m/pixel at 300 km).
    let entry_footprint = ray.start * pixel_angle;
    let far_own = cloud_far_ownership(entry_footprint);
    if (far_own >= 1.0 || ray.start > max_dist) {
        return RaymarchResult(max_dist, max_dist, vec4f(0.0, 0.0, 0.0, 1.0));
    }

    // Per-ray weather context varies over tens-of-km scales, so one evaluation
    // at the segment midpoint serves the short view segment. Base/detail shape
    // remains full 3-D per sample.
    // The per-ray weather/formation context must remain local when range grows:
    // preserve the old midpoint for short segments, but never anchor more than
    // 25 km beyond shell entry (finding G in the BL-33 fidelity pass).
    let context_t = ray.start + min(0.5 * (ray.end - ray.start), 25000.0);
    let mid = ray_origin + context_t * ray_dir;
    let n_mid = normalize(mid);
    var weather = sample_weather(n_mid);
    // The regime producer authors REAL zero-coverage regions, so a clear
    // context point no longer implies a clear ray: also probe a coarse mip
    // (~80 km footprint) before culling, or clouds vanish whenever the 25 km
    // anchor lands in the clear lane ahead of a system.
    let weather_region = textureSampleLevel(weather_texture, weather_sampler, n_mid, 4.0);
    // The banded march reaches ~300 km, far past the 25 km context anchor —
    // a second, coarser region probe (~320 km footprint) far along the ray
    // keeps the cull from erasing systems the long tail would reach.
    let far_anchor = ray.start + min(0.75 * (ray.end - ray.start), 150000.0);
    let far_region = textureSampleLevel(
        weather_texture,
        weather_sampler,
        normalize(ray_origin + far_anchor * ray_dir),
        5.0,
    );
    let region_coverage = max(max(weather.r, weather_region.r), far_region.r);
    if (region_coverage * config.clouds_coverage <= 1.0e-3) {
        return RaymarchResult(max_dist, max_dist, vec4f(0.0, 0.0, 0.0, 1.0));
    }
    // The anti-tiling modulation has a ~21.6 km period. Sampling it once per
    // view segment preserves a smoothly varying per-pixel system bias without
    // paying a second trilinear volume fetch at every 200–500 m step. The same
    // fetch's perlin channel doubles as the weather-system `formation` field
    // (cluster gate in `get_cloud_map_density`) at zero extra cost.
    let macro_period = max(config.clouds_base_shape_scale_m, 500.0) * 2.7;
    let macro_sample = cloud_volume(
        mid
            + weather_phase_offset(weather, macro_period)
            + vec3f(-7300.0, 2100.0, 4900.0),
        macro_period,
    );
    let macro_noise = macro_sample.a;
    let formation = macro_sample.r;

    // CLOUD-4: multi-scatter dual-lobe phase octaves (shared for the whole
    // ray). The scattering cosine is between propagation directions: photons
    // travel along -sun_dir and scatter toward the camera along -ray_dir, so
    // cosθ = dot(-sun_dir, -ray_dir) = dot(ray_dir, sun_dir) — +1 looking
    // sunward, where the g>0 forward lobe must peak. Mixing a propagation
    // direction with a view direction (ray·-sun) negates this and renders the
    // glare/silver-lining 180° from the sun.
    let ray_dot_sun = dot(ray_dir, config.sun_dir.xyz);
    let ms_lobes = multi_scatter_lobes(
        ray_dot_sun,
        config.forward_scattering_g,
        config.backward_scattering_g,
        config.scattering_lerp,
    );
    var ambient_bottom = config.clouds_ambient_color_bottom.rgb;
    var ambient_top = config.clouds_ambient_color_top.rgb;
    // Airlight radiance estimate for the veil between camera and cloud —
    // chroma comes from (1 − T_view), magnitude from the sky-ambient scale.
    let airlight_radiance = (ambient_bottom + ambient_top) * (0.5 * AIRLIGHT_GAIN);
    var dir_length = ray.dir_length;
    var dist = max_dist;
    var scattered_light = vec3f(0.0, 0.0, 0.0);
    var transmittance = 1.0;
    // Zeroth and first distance moments of this ray's extinction, for the
    // composite's depth partition (`slab_far_distance`).
    var tau_total = 0.0;
    var tau_moment = 0.0;
    var refining = false;
    var consecutive_empty = 0u;
    // Never backtrack outside the physical shell. Later coarse hits may rewind
    // one broad interval, but never before the last completed fine frontier.
    var refined_until = ray.start;

    let ray_step_limit = clamp(config.clouds_raymarch_steps_count, 1u, MAX_RAY_STEPS_CAP);
    // Where this ray's probe budget runs out, under the same footprint step
    // law. Only grazing rays ever reach it — a steep one finishes its short
    // in-shell segment long before. The marcher dissolves over the last
    // stretch and the far tier fades in complementarily.
    let reach_end = cloud_march_stop_m(f32(ray_step_limit), ray.start, pixel_angle);
    let reach_fade_begin = mix(ray.start, reach_end, CLOUD_MARCH_FADE_FRACTION);

    // The reach budget counts BROAD probes only. `cloud_march_stop_m` above is
    // the closed-form integral of the broad ladder alone, and the far tier
    // places its complementary fade-in at that frontier — so if refinement
    // (fine cadence, plus the rewind) drains the same counter, the marcher
    // truncates SHORT of the frontier with no fade and no far cover
    // (INC-20260729T012803Z: the silhouette lace). Fine work is bounded per
    // broad hit, so a separate iteration cap keeps the loop TDR-safe —
    // uniform-derived, NOT a compile-time constant, so no downstream compiler
    // is invited to unroll the march body thousands of times.
    var broad_spent = 0u;
    let iteration_cap = ray_step_limit * 3u;

    for (var iteration: u32 = 0u; iteration < iteration_cap; iteration++) {
        if (dir_length > ray.end) { break; }
        if (dir_length > reach_end) { break; }
        let world_position = ray_origin + dir_length * ray_dir;
        let up = normalize(world_position);
        weather = sample_weather(up);
        let normalized_height = get_normalized_height(world_position);
        let layer_h = (normalized_height - weather.b) / max(weather.a - weather.b, 0.02);

        // Banded cadence: broad step from the shared distance table, clamped
        // so steep rays still resolve thin layers vertically (such rays have
        // geometrically short in-shell segments, so the clamp stays cheap).
        let radial_rate = abs(dot(up, ray_dir));
        let broad_step = cloud_march_step_m(dir_length, pixel_angle, radial_rate);
        // Refinement resolves sub-broad-step density structure, so its cadence
        // is bounded below by what a pixel can SHOW: never finer than the
        // projected footprint. Near-field this is the unchanged 0.2× fraction
        // (footprint ≪ step); at range fine → broad and refinement stops
        // costing anything — which is what lets the reach budget extend to
        // planetary distances instead of burning steps on invisible detail.
        let fine_step = clamp(
            dir_length * pixel_angle,
            broad_step * REFINE_STEP_FRACTION,
            broad_step,
        );
        // Sampling scale everything band-limits against: the REFINE cadence
        // (density is only ever integrated there) or the projected pixel
        // footprint, whichever is coarser. This single value now drives the
        // cell field's octave fade, the sub-cell sculptor's retirement, and the
        // formation-edge width — one driver, so they cannot disagree.
        let cell_filter_m = max(fine_step, dir_length * pixel_angle);
        // Warp the strata lookup through the SHARED contract so the ~5 km texel
        // lattice reads as organic cells; the far tier warps identically, so
        // the fields stay registered. This used to be gated by range, on the
        // premise that the Cartesian sculptor owned near-field
        // morphology and hid the lattice. That premise died with this change:
        // the cell field is thresholded against `env` directly, so a texel
        // edge now cuts a VERTICAL WALL through a cloud, which the runway
        // capture showed as boxy right-angled silhouettes. The warp is a
        // measure-preserving direction remap, so applying it everywhere leaves
        // the derived calibration untouched.
        let up_strata = cloud_strata_warp(up, 1.0);
        let surface_density = sample_surface_density(up_strata, layer_h);
        let density_threshold = max(config.clouds_density, 1.0e-5)
            * BROAD_HIT_DENSITY_FRACTION;

        if (!refining) {
            if (broad_spent >= ray_step_limit) { break; }
            broad_spent += 1u;
            let broad_density = get_cloud_map_density(
                world_position,
                clamp(normalized_height, 0.0, 1.0),
                weather,
                surface_density,
                0.0,
                cell_filter_m,
                macro_noise,
                formation,
            );
            if (broad_density > density_threshold) {
                refining = true;
                consecutive_empty = 0u;
                dir_length = max(
                    refined_until,
                    dir_length - broad_step,
                );
            } else {
                dir_length += broad_step;
            }
            continue;
        }

        // Erosion detail fades as the local refine cadence outgrows the
        // authored feature scale — the footprint-matched octave fade.
        let detail_feature_m = max(config.clouds_detail_scale_m, 50.0);
        let detail_weight = 1.0 - smoothstep(
            detail_feature_m * DETAIL_FILTER_BEGIN,
            detail_feature_m * DETAIL_FILTER_END,
            fine_step,
        );

        let clouds_density_sampled =
            get_cloud_map_density(
                world_position,
                clamp(normalized_height, 0.0, 1.0),
                weather,
                surface_density,
                detail_weight,
                cell_filter_m,
                macro_noise,
                formation,
            )
            * (1.0 - smoothstep(reach_fade_begin, reach_end, dir_length));

        if (clouds_density_sampled > density_threshold) {
            consecutive_empty = 0u;
        } else {
            consecutive_empty += 1u;
        }

        if (clouds_density_sampled > 0.0) {
            dist = min(dist, dir_length);

            let h_clamped = clamp(normalized_height, 0.0, 1.0);
            let ambient_light = mix(
                ambient_bottom,
                ambient_top,
                h_clamped
            );

            let density_fraction = clamp(
                clouds_density_sampled / max(config.clouds_density, 1.0e-5),
                0.0,
                1.0,
            );
            // Filtered sun optical depth (smooth broad mass only, jittered tap
            // ladder) drives per-octave multi-scatter attenuation.
            let tau_sun =
                volumetric_sun_depth(
                    world_position,
                    weather,
                    surface_density,
                    macro_noise,
                    formation,
                    jitter,
                    cell_filter_m,
                );
            let sun_T = atmosphere_sun_transmittance(world_position);
            let powder = powder_term(density_fraction, ray_dot_sun);
            let octave_shadow = vec3f(
                exp(-tau_sun * MS_OCTAVE_EXTINCTION.x),
                exp(-tau_sun * MS_OCTAVE_EXTINCTION.y),
                exp(-tau_sun * MS_OCTAVE_EXTINCTION.z),
            );
            // Single scattering: exact normalized phase against the beam that
            // actually survives to this sample.
            let single = ms_lobes.x * octave_shadow.x;
            // Multiple scattering: the diffusion reservoir (see
            // CLOUD_MS_ALBEDO). The wider octaves no longer carry phase
            // energy — they supply the reservoir's depth response, which is
            // the part of them that was ever physical.
            let ms_depth = dot(MS_OCTAVE_WEIGHTS.yz, octave_shadow.yz)
                / (MS_OCTAVE_WEIGHTS.y + MS_OCTAVE_WEIGHTS.z);
            let ms_aniso = mix(1.0, ms_lobes.z / INV_FOUR_PI, MS_ANISO);
            let multi = CLOUD_MS_ALBEDO * INV_PI * ms_depth * ms_aniso;
            let scattering = single + multi;
            let direct = config.sun_color.rgb
                * sun_T
                * scattering
                * powder;
            // Ambient self-occlusion: a deep interior sample sees far less
            // sky than a fringe. Without this, the physical-magnitude sky
            // ambient (SkyAmbient binding) flattened every lobe into one pale
            // sheet. Driven by the same filtered sun depth the multi-scatter
            // octaves use — a correlated stand-in for sky visibility that
            // costs no extra probe (a directional sky march à la Blackrack is
            // the eventual CLOUD-5 upgrade).
            let ambient_occlusion = 0.30 + 0.70 * exp(-tau_sun * 0.45);
            let amb = ambient_light * ambient_occlusion;

            // Frostbite energy-conserving step, then sample→camera air so
            // in-scatter is pre-attenuated before the BodySky composite, plus
            // the airlight the cloud's opacity occludes out of the composite.
            let S = clouds_density_sampled * (amb + direct);
            let delta_transmittance = exp(-clouds_density_sampled * fine_step);
            let view_T = atmosphere_view_transmittance(world_position, ray_origin);
            var integrated_scattering = S * (1.0 - delta_transmittance) / clouds_density_sampled;
            integrated_scattering = integrated_scattering * view_T
                + airlight_radiance * (vec3f(1.0) - view_T) * (1.0 - delta_transmittance);

            scattered_light += transmittance * integrated_scattering;
            transmittance *= delta_transmittance;

            let d_tau = clouds_density_sampled * fine_step;
            tau_total += d_tau;
            tau_moment += d_tau * dir_length;
        }

        if transmittance <= config.clouds_min_transmittance { break; }

        dir_length += fine_step;
        refined_until = max(refined_until, dir_length);
        if (consecutive_empty >= REFINE_EMPTY_LIMIT) {
            refining = false;
            consecutive_empty = 0u;
        }
    }

    // Sub-pixel dissolve: the volumetric tier hands over only once cell-scale
    // morphology is genuinely smaller than a pixel, where the far projection's
    // smooth answer is the correct one and no transition can be visible. The
    // predecessor keyed this to shell-ENTRY DISTANCE (240–300 km), which for a
    // nadir view is just altitude — so the volumetrics were switched off in the
    // middle of every ascent, at framings where the deck is best resolved.
    let near_own = 1.0 - far_own;
    transmittance = mix(1.0, transmittance, near_own);
    scattered_light *= near_own;

    // Soft energy peak: per-channel Reinhard so the forward-scatter spike
    // (a lit cell viewed toward the sun reaches ~8× an ordinary lit face)
    // stays bounded. The white point must sit well ABOVE ordinary sunlit
    // cloud, or the limiter becomes a dimmer: at the old 2.2 it was tuned
    // against single-scatter-only radiance (~0.2) and would now eat a third
    // of every white cloud top.
    let peak_limit = 10.0;
    scattered_light = scattered_light / (vec3f(1.0) + scattered_light / peak_limit);

    return RaymarchResult(
        dist,
        slab_far_distance(dist, tau_total, tau_moment),
        vec4f(scattered_light, transmittance),
    );
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
    slab_far: f32,
}

// Select one coherent colour/depth history sample from the reprojected 2×2
// footprint. Distances cannot be bilinearly filtered: mixing a finite hit with
// the clear sentinel creates a fictitious range, and even two finite depths can
// straddle different cloud lobes. The expected old-camera range chooses the
// geometrically closest tap; all-clear footprints return canonical clear.
fn sample_history_matching(rr: vec2f, expected_dist: f32) -> CloudsOutput {
    let res = config.render_resolution.xy;
    let p = rr * res - 0.5;
    let base = floor(p);
    let b = vec2i(base);
    let cmax = vec2i(i32(res.x) - 1, i32(res.y) - 3);
    let c00 = clamp(b, vec2i(0, 0), cmax);
    let c11 = clamp(b + vec2i(1, 1), vec2i(0, 0), cmax);
    let coords = array<vec2i, 4>(
        c00,
        vec2i(c11.x, c00.y),
        vec2i(c00.x, c11.y),
        c11,
    );
    var best = CloudsOutput(vec4f(0.0, 0.0, 0.0, 1.0), MAX_DISTANCE, MAX_DISTANCE);
    var best_error = MAX_DISTANCE;
    for (var i = 0u; i < 4u; i += 1u) {
        let coord = vec2u(coords[i]);
        let span = textureLoad(history_distance_texture, coord, 0).rg;
        if span.x < 1.0e8 {
            let error = abs(span.x - expected_dist);
            if error < best_error {
                best_error = error;
                best = CloudsOutput(textureLoad(history_texture, coord, 0), span.x, span.y);
            }
        }
    }
    return best;
}

struct HistoryNeighborhood {
    minimum: vec4f,
    maximum: vec4f,
    lum_mean: f32,
    lum_variance: f32,
}

fn history_neighborhood(rr: vec2f) -> HistoryNeighborhood {
    let res = vec2i(config.render_resolution.xy);
    let centre = vec2i(rr * config.render_resolution.xy);
    let cmax = res - vec2i(1, 3);
    var minimum = vec4f(1.0e20);
    var maximum = vec4f(-1.0e20);
    var lum_sum = 0.0;
    var lum_sq_sum = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let tap = textureLoad(
                history_texture,
                vec2u(clamp(centre + vec2i(x, y), vec2i(0), cmax)),
                0,
            );
            minimum = min(minimum, tap);
            maximum = max(maximum, tap);
            let lum = dot(tap.rgb, vec3f(0.2126, 0.7152, 0.0722));
            lum_sum += lum;
            lum_sq_sum += lum * lum;
        }
    }
    let mean = lum_sum / 9.0;
    return HistoryNeighborhood(minimum, maximum, mean, max(lum_sq_sum / 9.0 - mean * mean, 0.0));
}

// Variance clip plus component neighborhood clamp. This prevents a bright rim
// or dark core from being reprojected indefinitely across a newly exposed
// edge, while retaining enough range for HDR cloud lighting.
fn clamp_history(history: vec4f, current: vec4f, rr: vec2f) -> vec4f {
    let neighborhood = history_neighborhood(rr);
    let margin = vec4f(0.025, 0.025, 0.025, 0.015);
    let lo = min(neighborhood.minimum, current) - margin;
    let hi = max(neighborhood.maximum, current) + margin;
    var clipped = clamp(history, lo, hi);
    let sigma = sqrt(neighborhood.lum_variance);
    let lum_lo = neighborhood.lum_mean - 1.75 * sigma - 0.01;
    let lum_hi = neighborhood.lum_mean + 1.75 * sigma + 0.01;
    let lum = dot(clipped.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let clipped_lum = clamp(lum, lum_lo, lum_hi);
    if (lum > 1.0e-5) {
        clipped = vec4f(clipped.rgb * (clipped_lum / lum), clipped.a);
    }
    return clipped;
}

fn camera_position(camera: mat4x4f) -> vec3f {
    return camera[3].xyz * CAM_POS_COLUMN_SCALE;
}

fn camera_ray_direction(rr: vec2f, camera: mat4x4f) -> vec3f {
    let ndc_xy = (rr * 2.0 - vec2f(1.0)) * vec2f(1.0, -1.0);
    let ray_eye = config.inverse_camera_projection * vec4f(ndc_xy, -1.0, 1.0);
    return normalize((camera * vec4f(ray_eye.xy, -1.0, 0.0)).xyz);
}

fn reproject_uv(point: vec3f, old_cam: mat4x4f) -> vec3f {
    let cam_old = camera_position(old_cam);
    let rel = point - cam_old;
    let dv = vec3f(
        dot(old_cam[0].xyz, rel),
        dot(old_cam[1].xyz, rel),
        dot(old_cam[2].xyz, rel),
    );
    if (dv.z >= 0.0) {
        return vec3f(0.0, 0.0, 0.0);
    }
    let tan_x = config.inverse_camera_projection[0][0];
    let tan_y = config.inverse_camera_projection[1][1];
    let ndc = vec2f(dv.x / (-dv.z * tan_x), dv.y / (-dv.z * tan_y));
    let rr = (vec2f(ndc.x, -ndc.y) + 1.0) * 0.5;
    let valid = select(0.0, 1.0, rr.x > 0.0 && rr.x < 1.0 && rr.y > 0.0 && rr.y < 0.998);
    return vec3f(rr, valid);
}

fn get_clouds_color(frag_coord: vec2f, camera: mat4x4f, old_cam: mat4x4f, ray_dir: vec3f, ray_origin: vec3f) -> CloudsOutput {
    if (frag_coord.y < 1.5) {
        if frag_coord.x < 1.0 {
            return CloudsOutput(vec4f(
                config.render_resolution.xy,
                config.inverse_camera_projection[0][0],
                config.inverse_camera_projection[1][1],
            ), MAX_DISTANCE, MAX_DISTANCE);
        }
        if frag_coord.x < 6.0 && frag_coord.x >= 5.0 {
            return CloudsOutput(vec4f(f32(config.history_epoch), config.time, 0.0, 0.0), MAX_DISTANCE, MAX_DISTANCE);
        }
        return CloudsOutput(common::save_camera(camera, frag_coord, ray_origin), MAX_DISTANCE, MAX_DISTANCE);
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
    let history_meta = textureLoad(
        history_texture,
        vec2u(0u, u32(config.render_resolution.y) - 1u),
        0,
    );
    let history_epoch = textureLoad(
        history_texture,
        vec2u(5u, u32(config.render_resolution.y) - 1u),
        0,
    ).x;
    let history_valid = all(abs(history_meta.xy - config.render_resolution.xy) < vec2f(0.5))
        && abs(history_meta.z - config.inverse_camera_projection[0][0]) < 1.0e-4
        && abs(history_meta.w - config.inverse_camera_projection[1][1]) < 1.0e-4
        && abs(history_epoch - f32(config.history_epoch)) < 0.5
        && cam_delta < CAMERA_CUT_DELTA;
    let current_texel = vec2u(
        u32(frag_coord.x),
        u32(config.render_resolution.y - 1.0) - u32(frag_coord.y),
    );

    // A steady view uses the cheap rotating 3×3 topology: history is already
    // screen-aligned, so one ninth of pixels can refresh each frame. A moving
    // translucent volume has no single motion depth that can reconstruct its
    // integrated radiance, even for one frame, so camera motion always traces
    // every current ray. History may stabilize that fresh result below, but it
    // never substitutes for current radiance. Invalid history also forces a
    // coherent full frame after cuts/resizes/body changes.
    let sparse_slot = (current_texel.x % 3u) + 3u * (current_texel.y % 3u);
    let trace_pixel = config.sparse_march == 0u
        || !history_valid
        || !cam_static
        || sparse_slot == (config.frame_index % 9u);
    if (!trace_pixel) {
        let old_span = textureLoad(history_distance_texture, current_texel, 0).rg;
        return CloudsOutput(textureLoad(history_texture, current_texel, 0), old_span.x, old_span.y);
    }

    // Interleaved-gradient phase for the view march; temporal history removes
    // the residual while the offset prevents coherent horizon bands.
    var jitter = fract(52.9829189 * fract(dot(frag_coord, vec2f(0.06711056, 0.00583715))));
    // Decorrelation must vary in both pixel and frame. A frame-wide phase
    // shifts every ray in lockstep and survives sparse accumulation as nested
    // march contours inside dark cores.
    jitter = fract(jitter + common::hash13(vec3f(frag_coord, f32(config.frame_index))));

    let result = raymarch(ray_origin, ray_dir, MAX_DISTANCE, jitter);

    // Thalos fork: store the *clean* raymarch result — rgb = premultiplied
    // in-scatter, a = transmittance — with NO built-in sky/fog mix. The game
    // composites this layer over its own scene (atmosphere, terrain, stars) in
    // a separate fullscreen pass, so baking a sky color and distance fog in
    // here would double-paint a sky we don't want.
    let col = result.color;

    if (history_valid && cam_static) {
        // Steady view: same-pixel accumulation from the history snapshot.
        let original_color = textureLoad(
            history_texture,
            vec2u(u32(frag_coord.x),
            u32(config.render_resolution.y - 1.0) - u32(frag_coord.y)),
            0
        );
        let rr = (vec2f(current_texel) + 0.5) / config.render_resolution.xy;
        let history = clamp_history(original_color, col, rr);
        let steady_weight = select(
            config.reprojection_strength,
            min(config.reprojection_strength, 0.82),
            config.sparse_march != 0u,
        );
        return CloudsOutput(mix(col, history, steady_weight), result.dist, result.slab_far);
    }

    // Moving view: reproject this ray's nearest cloud point through the
    // previous frame's camera (stored in the history's save rows) and blend
    // the history texel there. Cloud points are body-fixed, so the point
    // itself is frame-invariant.
    if (history_valid && result.dist < 1.0e8) {
        let p = ray_origin + result.dist * ray_dir;
        let projected = reproject_uv(p, old_cam);
        if (projected.z > 0.5) {
                let expected_old_dist = length(p - camera_position(old_cam));
                let history = sample_history_matching(projected.xy, expected_old_dist);
                // Soft disocclusion: weight history by cloud-depth agreement
                // instead of a binary reject — hard accept/reject boundaries
                // themselves pattern as fresh-noise speckle at every edge.
                // Compare in the OLD camera's metric. Comparing its stored
                // range to the current-camera range accepted false matches
                // whenever the camera translated.
                let rel_err = abs(history.dist - expected_old_dist)
                    / max(expected_old_dist, 1.0);
                let depth_agree = 1.0 - smoothstep(0.08, 0.25, rel_err);
                let opacity_agree = 1.0
                    - smoothstep(0.08, 0.28, abs(history.color.a - col.a));
                // The public temporal-strength control gates *both* the
                // steady same-pixel path and this moving-view path. Previously
                // setting it to zero still kept 90% motion history, so a
                // temporal-disabled diagnostic capture was impossible.
                let w = MOVING_REPROJECTION_STRENGTH
                    * depth_agree * opacity_agree
                    * clamp(config.reprojection_strength, 0.0, 1.0);
                if (w > 0.01) {
                    let hist = clamp_history(history.color, col, projected.xy);
                    return CloudsOutput(mix(col, hist, w), result.dist, result.slab_far);
                }
        }
    }
    return CloudsOutput(col, result.dist, result.slab_far);
}

fn get_ray_direction(frag_coord: vec2f) -> vec3f {
    // inverse_camera_projection is also called view_from_clip
    // inverse_camera_view is also called world_from_view; here it is
    // body_from_world × world_from_view, so rays come out body-fixed.
    return camera_ray_direction(
        frag_coord / config.render_resolution,
        config.inverse_camera_view,
    );
}

// ── Cloud sun-transmittance cascade (CLOUD-5 / W2 near tier) ────────────────
//
// One texel = one point on the reference plane under the view anchor; its value
// is the transmittance of the sun beam arriving *at that point* after crossing
// the whole deck. Receivers project their own sun ray onto the same plane, so
// the deck's parallax — a low sun laying shadows kilometres downwind of the
// cloud that casts them — falls out of the geometry instead of being faked.
//
// Three properties keep this honest against the visible volume:
//
// - It integrates `get_cloud_map_density` — the SAME field, config, weather
//   cube and strata cube the view march samples. There is no second cloud
//   distribution to disagree with (docs/rendering/clouds.md §2, principle 4).
// - Extinction matches the primary beam exactly (`exp(-τ)`, the view march's
//   `MS_OCTAVE_EXTINCTION.x` octave), so a cell that reads opaque from the
//   air lays an equally opaque shadow.
// - Erosion detail rides the same footprint fade the view march uses, keyed on
//   the map's texel instead of the ray's step. The field is band-limited
//   before the sampler outgrows it — the alias-free contract from
//   BL-20260724T003705Z, in a second geometry.
//
// Only the DIRECT beam is gated. Ground under solid overcast keeps its sky
// ambient, which is why an overcast landscape reads flat-lit rather than black.

// Steps across the deck. The vertical span is a ~1–3 km slab; at 20 steps a
// zenith sun samples every ~100 m, comfortably under the authored base-shape
// scale, and the slant clamp below stops a low sun from stretching that past
// the structure it is supposed to resolve.
const CLOUD_SHADOW_STEPS: u32 = 20u;
const CLOUD_SHADOW_MAX_STEP_M: f32 = 400.0;

// Distance along `dir` from `origin` (inside radius `r`) to the sphere of
// radius `r`. Positive root only — the caller is always inside the shell.
fn shell_exit_distance(origin: vec3f, dir: vec3f, r: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - r * r;
    let disc = max(b * b - c, 0.0);
    return -b + sqrt(disc);
}

// Transmittance plus the two intermediates that identify a failure: the local
// coverage the march saw, and the optical depth it accumulated. They ride the
// map's spare channels (it is RGBA16F for the filtering, and one channel is all
// the term needs), so `THALOS_CLOUD_SHADOW=show` can paint a *diagnosis*, not
// just an answer — clear ground with zero coverage and clear ground with a
// march that never entered the deck look identical in the r channel alone.
struct CloudShadowSample {
    transmittance: f32,
    coverage: f32,
    optical_depth: f32,
}

fn cloud_shadow_transmittance(base: vec3f, jitter: f32) -> CloudShadowSample {
    var out: CloudShadowSample;
    out.transmittance = 1.0;
    out.coverage = 0.0;
    out.optical_depth = 0.0;
    let sun = config.sun_dir.xyz;
    let r_bottom = config.planet_radius + config.clouds_bottom_height;
    let r_top = config.planet_radius + config.clouds_top_height;
    let entry = shell_exit_distance(base, sun, r_bottom);
    let exit = shell_exit_distance(base, sun, r_top);
    let span = exit - entry;
    if (span <= 0.0) {
        return out;
    }

    // Per-texel weather/formation context, taken at the slab midpoint exactly
    // as the view march takes it at its segment midpoint: these fields vary
    // over tens of km, and re-sampling them per step buys nothing but cost.
    let mid = base + sun * (entry + 0.5 * span);
    let weather_mid = sample_weather(normalize(mid));
    out.coverage = weather_mid.r * config.clouds_coverage;
    if (out.coverage <= 1.0e-3) {
        return out;
    }
    let macro_period = max(config.clouds_base_shape_scale_m, 500.0) * 2.7;
    let macro_sample = cloud_volume(
        mid
            + weather_phase_offset(weather_mid, macro_period)
            + vec3f(-7300.0, 2100.0, 4900.0),
        macro_period,
    );
    let macro_noise = macro_sample.a;
    let formation = macro_sample.r;

    let step_m = min(span / f32(CLOUD_SHADOW_STEPS), CLOUD_SHADOW_MAX_STEP_M);
    // Footprint-matched erosion fade: the map's texel is the sampler here, and
    // a slanted beam smears each texel further along the ground.
    let footprint_m = max(
        config.shadow_axis_u.w / max(config.shadow_up.w, 0.15),
        step_m,
    );
    let detail_feature_m = max(config.clouds_detail_scale_m, 50.0);
    let detail_weight = 1.0 - smoothstep(
        detail_feature_m * DETAIL_FILTER_BEGIN,
        detail_feature_m * DETAIL_FILTER_END,
        footprint_m,
    );
    // The cascade already worked in footprint, which is now the ONE LOD driver
    // everywhere: it just passes `footprint_m` to the density function as its
    // `filter_m` and the field band-limits itself.
    var optical_depth = 0.0;
    for (var i = 0u; i < CLOUD_SHADOW_STEPS; i++) {
        let t = entry + (f32(i) + jitter) * step_m;
        if (t > exit) { break; }
        let pos = base + sun * t;
        let up = normalize(pos);
        let weather = sample_weather(up);
        let normalized_height = get_normalized_height(pos);
        let layer_h = (normalized_height - weather.b) / max(weather.a - weather.b, 0.02);
        let up_strata = cloud_strata_warp(up, 1.0);
        let surface_density = sample_surface_density(up_strata, layer_h);
        optical_depth += step_m * get_cloud_map_density(
            pos,
            clamp(normalized_height, 0.0, 1.0),
            weather,
            surface_density,
            detail_weight,
            footprint_m,
            macro_noise,
            formation,
        );
    }
    out.optical_depth = optical_depth;
    out.transmittance = exp(-optical_depth);
    return out;
}

@compute @workgroup_size(8, 8, 1)
fn cloud_shadow(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let size = textureDimensions(cloud_shadow_texture);
    if (invocation_id.x >= size.x || invocation_id.y >= size.y) {
        return;
    }
    var probe: CloudShadowSample;
    probe.transmittance = 1.0;
    probe.coverage = 0.0;
    probe.optical_depth = 0.0;
    // `shadow_axis_v.w` is the live flag: no cloud body, clouds off, or a sun
    // at/below the anchor's horizon all leave the map fully lit rather than
    // stale — a receiver can never read last frame's shadows for a sun that
    // has since set.
    if (config.shadow_axis_v.w > 0.5) {
        let uv = (vec2f(invocation_id.xy) + 0.5) / vec2f(size);
        let offset = (uv * 2.0 - 1.0) * config.shadow_origin.w;
        let base = config.shadow_origin.xyz
            + config.shadow_axis_u.xyz * offset.x
            + config.shadow_axis_v.xyz * offset.y;
        // Jitter the tap ladder per texel: a fixed phase across the whole map
        // resolves the deck's vertical profile into banding, the same failure
        // `volumetric_sun_depth` documents for its own ladder. Keyed on the
        // TEXEL ONLY, never the frame — this map carries no temporal
        // accumulation to absorb per-frame noise, so a time-varying ladder
        // would shimmer every shadow edge. Static dither disappears under the
        // receivers' bilinear fetch instead.
        let jitter = common::hash13(vec3f(vec2f(invocation_id.xy), 0.0));
        probe = cloud_shadow_transmittance(base, jitter);
    }
    textureStore(
        cloud_shadow_texture,
        invocation_id.xy,
        vec4f(probe.transmittance, probe.coverage, probe.optical_depth, 1.0),
    );
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
    textureStore(
        cloud_distance_texture,
        invocation_id.xy,
        vec4f(out.dist, out.slab_far, 0.0, 0.0),
    );
}
