#import bevy_open_world::common
#import thalos::atmosphere::cloud_surface_shape

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
// ── Cheap/full-density adaptive march ────────────────────────────────────────
// Clear air advances in 600 m intervals while evaluating the smooth broad mass
// only. A meaningful hit backs up one interval and switches to the proven
// 120 m full-density cadence; four empty fine samples return to coarse mode.
// This samples every interval at the frequency it can represent — it is not an
// empty-space leap and does not use weather/profile hints as a resume gate.
const CLOUD_STEP_M = 600.0;       // broad-mass cadence through clear air
const MIN_STEP_M = 120.0;         // full-density cadence after a broad hit
const REFINE_EMPTY_LIMIT = 4u;
const BROAD_HIT_DENSITY_FRACTION = 0.01;
const MAX_RAY_STEPS_CAP = 128u;   // compile-time safety cap; config selects ≤ this
const MAX_CLOUD_DIST = 75000.0;   // metres; the march's cost-bounded shell segment
// The march owns only this near region: with ≤128 steps it cannot sample the
// 1–4 km base features across a 100+ km limb chord without moiré (verified
// 2026-07-22 — a distance-stretched step aliased the whole disc into a dot
// grid), and from orbit the 8 km noise-tile period itself reads as a repeating
// dot lattice. Rays ENTERING the shell beyond ENTRY_FADE dissolve out of the
// near estimator entirely; the composite's reduced weather-column band march
// owns them (partition of unity, see cloud_composite.wgsl — keep these
// windows in lockstep with `march_reach` there). Unlike the pre-CLOUD-6
// haze-out these clouds are replaced, not deleted.
const ENTRY_FADE_START = 56000.0;
const ENTRY_FADE_END = 75000.0;
const DETAIL_FILTER_BEGIN = 0.25;
const DETAIL_FILTER_END = 0.50;
const CAMERA_CUT_DELTA = 0.05;
// Airlight restoration: in-scattered air between the camera and each cloud
// sample. The composite draws clouds OVER the already-integrated sky, which
// attenuates the foreground airlight by cloud opacity; re-adding it here keeps
// distant clouds behind a natural blue/warm veil instead of a soot-brown
// extinction-only tint (the "dirty distant deck" failure).
const AIRLIGHT_GAIN = 1.35;

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
// Canonical surface-space broad shape at four normalized-height strata. It is
// generated/versioned with weather and shares weather_sampler's filter state.
@group(1) @binding(7) var surface_density_texture: texture_cube<f32>;
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
fn get_cloud_map_density(
    pos: vec3f,
    shell_height: f32,
    weather: vec4f,
    surface_density: f32,
    detail_weight: f32,
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

    let shape_scale = max(config.clouds_base_shape_scale_m, 500.0);
    let broad = cloud_volume(
        rotated_domain(pos)
            + weather_phase_offset(weather, shape_scale)
            + vec3f(1800.0, -4200.0, 900.0),
        shape_scale,
    );
    // Broad Perlin/Worley masses. The erosion channel stays out of the solid
    // body: promoting its small cells into `shape` fragmented the volume into
    // screen-space stipple instead of adding readable billows.
    // Spectrum follows column height: tall (congestus/storm) columns weight
    // the low-frequency channels so a tower reads as ONE coherent mass with
    // large billows; squat fair-weather columns keep the small-lobe mix. This
    // is a sub-strata morphology choice, so it has no CPU mirror.
    let column_tall = smoothstep(0.30, 0.65, local_top - local_base);
    let shape_squat = broad.r * 0.52 + broad.g * 0.24 + broad.a * 0.24;
    let shape_tall = broad.r * 0.64 + broad.g * 0.06 + broad.a * 0.30;
    let shape = mix(shape_squat, shape_tall, column_tall);

    // Formation authority is the NON-PERIODIC surface field. The strata
    // density (coverage threshold + typed vertical response, applied by the
    // CPU producer on the body-direction sphere) decides where cloud bodies
    // live; the periodic Cartesian volume only sculpts sub-texel lobes inside
    // that envelope. The previous threshold let the 21.6 km macro octave and
    // the 8 km tile's low frequencies organize clouds over tens of km, and the
    // spherical shell cut that Cartesian repeat into planet-visible rows
    // (ADR-20260722T141000Z; 2026-07-23 user verdict). `macro_noise` remains
    // only as a faint sub-dominant variety term.
    let coupling = clamp(config.surface_density_coupling, 0.0, 1.0);
    // The threshold makes the near tier's COLUMN areal fill (seen from above)
    // equal the strata density — the contract the far tier reads directly. A
    // vertical ray takes several decorrelated 3-D samples through the layer,
    // so column fill is the UNION of per-sample exceedance; per-sample
    // quantile mapping alone still rendered a 0.1–0.3 field as a ~0.69 deck
    // (measured, tier A/B 2026-07-23). Curve fitted empirically against the
    // pixel-measured near-only fill at the spaceport framing.
    let env = clamp(surface_density, 0.0, 1.0);
    let threshold_surface = mix(0.81, 0.44, env) + (0.5 - macro_noise) * 0.05;
    // Capture-only legacy branch (surface_density_coupling = 0): the old
    // Cartesian-organized threshold, kept for A/B attribution.
    let threshold_legacy = mix(0.58, 0.30, cov)
        + (0.5 - macro_noise) * 0.07
        + (0.5 - formation) * 0.17
        + (0.35 - surface_density) * 0.08;
    let threshold = mix(threshold_legacy, threshold_surface, coupling);
    // Tall columns keep mass with height (towers); squat puffs round off.
    // Mirrored in `cloud_surface_density_cpu` — keep in lockstep.
    // (`column_tall` is declared at the shape-spectrum blend above.)
    let vertical_narrow = h
        * (0.04 * stratus_w + 0.19 * cumulus_w + 0.09 * storm_w)
        * (1.0 - 0.55 * column_tall);
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
    // Wide, gentle erosion falloff: a narrow 0.04–0.18 window drew its outer
    // iso-contour as visible "fingerprint" rings inside big lobes once the
    // softer CLOUD-4 lighting stopped hiding them.
    let edge = 1.0 - smoothstep(0.02, 0.34, mass);
    if (edge * detail_weight > 1.0e-3) {
        let detail = cloud_volume(
            rotated_domain(pos + boil) + vec3f(270.0, -610.0, 130.0),
            // Channel B contains eight primary Worley cells across the stored
            // tile. `clouds_detail_scale_m` describes one authored physical
            // erosion feature, not the whole tile period.
            max(config.clouds_detail_scale_m, 50.0) * 8.0,
        );
        mass -= (1.0 - detail.b) * edge * detail_weight * config.clouds_detail_strength * 0.55;
    }

    let shaped = smoothstep(0.0, max(config.clouds_base_edge_softness, 0.015), mass);
    // The surface field owns planet-scale occupancy in every projection. At
    // mip 0 this is a coherent body-fixed envelope; the Cartesian volume only
    // sculpts sub-cell shape inside it. Far consumers sample the same payload
    // at footprint mips, so refinement adds morphology instead of replacing
    // one cloud distribution with another.
    let shared_envelope = smoothstep(0.04, 0.42, surface_density);
    return max(
        shaped
            * vertical_profile
            * mix(1.0, shared_envelope, coupling)
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
        lobes[i] = min(lobe * 0.07957747, 2.2);
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

/// Silver-lining / powder: thin edges facing the light brighten; the same thin
/// path looking *away* from the light darkens (HZD powder). Restrained: the
/// former 0.85 away-darkening painted every anti-sun lobe near-black and read
/// as dirt rather than shading.
fn powder_term(density_fraction: f32, cos_theta: f32) -> f32 {
    let d = clamp(density_fraction, 0.0, 1.0);
    let powder = 1.0 - exp(-d * 2.0);
    // cos_theta = ray·(-sun): +1 looking toward the sun (silver lining).
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

    // Jitter only the first fine sample. Coarse probes back up before they hand
    // a hit to the full-density cadence, so they cannot skip the entry edge.
    let step = MIN_STEP_M;
    // The full-step temporal phase is safe now that the minimum density
    // feature is larger than one march step. It prevents near-horizontal rays
    // from stacking their samples into coherent bands; history averages the
    // phase instead of trying to hide under-resolved density.
    let dir_length = seg_start - step * jitter;

    return Ray(step, dir_length, seg_start, seg_end);
}

fn raymarch(ray_origin: vec3f, ray_dir: vec3f, max_dist: f32, jitter: f32) -> RaymarchResult {
    let ray = get_ray(ray_origin, ray_dir, max_dist, jitter);

    // Shell entered beyond the near estimator's ownership: the composite's
    // band march owns the whole ray; skip the volume work entirely.
    if (ray.start > min(max_dist, ENTRY_FADE_END)) {
        return RaymarchResult(max_dist, vec4f(0.0, 0.0, 0.0, 1.0));
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
    if (max(weather.r, weather_region.r) * config.clouds_coverage <= 1.0e-3) {
        return RaymarchResult(max_dist, vec4f(0.0, 0.0, 0.0, 1.0));
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
    // ray). cosθ = view·sun_incoming with sun_incoming = -sun_dir.
    let ray_dot_sun = dot(ray_dir, -config.sun_dir.xyz);
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
    var refining = false;
    var consecutive_empty = 0u;
    // Never backtrack outside the physical shell. Later coarse hits may rewind
    // one broad interval, but never before the last completed fine frontier.
    var refined_until = ray.start;

    // March-exhaustion fade: a ray travelling far *inside* the deck uses all
    // its configured step cap and stops mid-cloud; the entry-distance haze below never
    // fires for it (entry was close), so without this the stop front reads
    // as a sharp tonal seam arc across the deck. Dissolve density over the
    // last third of the marched reach instead.
    let ray_step_limit = clamp(config.clouds_raymarch_steps_count, 1u, MAX_RAY_STEPS_CAP);
    let march_span = f32(ray_step_limit) * CLOUD_STEP_M;
    // Wide dissolve (half the span) so the far estimator's smoother texture
    // fades in over kilometres instead of appearing at a visible seam line.
    let reach_fade_begin = ray.start + 0.50 * march_span;
    let reach_end = ray.start + march_span;

    for (var step: u32 = 0u; step < ray_step_limit; step++) {
        if (dir_length > ray.end) { break; }
        let world_position = ray_origin + dir_length * ray_dir;
        weather = sample_weather(normalize(world_position));
        let normalized_height = get_normalized_height(world_position);
        let layer_h = (normalized_height - weather.b) / max(weather.a - weather.b, 0.02);
        let surface_density = sample_surface_density(
            normalize(world_position),
            layer_h,
        );
        let density_threshold = max(config.clouds_density, 1.0e-5)
            * BROAD_HIT_DENSITY_FRACTION;

        if (!refining) {
            let broad_density = get_cloud_map_density(
                world_position,
                clamp(normalized_height, 0.0, 1.0),
                weather,
                surface_density,
                0.0,
                macro_noise,
                formation,
            );
            if (broad_density > density_threshold) {
                refining = true;
                consecutive_empty = 0u;
                dir_length = max(
                    refined_until,
                    dir_length - CLOUD_STEP_M,
                );
            } else {
                dir_length += CLOUD_STEP_M;
            }
            continue;
        }

        let detail_feature_m = max(config.clouds_detail_scale_m, 50.0);
        let detail_weight = 1.0 - smoothstep(
            detail_feature_m * DETAIL_FILTER_BEGIN,
            detail_feature_m * DETAIL_FILTER_END,
            MIN_STEP_M,
        );

        let clouds_density_sampled =
            get_cloud_map_density(
                world_position,
                clamp(normalized_height, 0.0, 1.0),
                weather,
                surface_density,
                detail_weight,
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
                );
            let sun_T = atmosphere_sun_transmittance(world_position);
            let powder = powder_term(density_fraction, ray_dot_sun);
            let octave_shadow = vec3f(
                exp(-tau_sun * MS_OCTAVE_EXTINCTION.x),
                exp(-tau_sun * MS_OCTAVE_EXTINCTION.y),
                exp(-tau_sun * MS_OCTAVE_EXTINCTION.z),
            );
            let scattering = dot(MS_OCTAVE_WEIGHTS * ms_lobes, octave_shadow)
                / (MS_OCTAVE_WEIGHTS.x + MS_OCTAVE_WEIGHTS.y + MS_OCTAVE_WEIGHTS.z);
            let direct = config.sun_color.rgb
                * sun_T
                * scattering
                * powder;
            let amb = ambient_light;

            // Frostbite energy-conserving step, then sample→camera air so
            // in-scatter is pre-attenuated before the BodySky composite, plus
            // the airlight the cloud's opacity occludes out of the composite.
            let S = clouds_density_sampled * (amb + direct);
            let delta_transmittance = exp(-clouds_density_sampled * MIN_STEP_M);
            let view_T = atmosphere_view_transmittance(world_position, ray_origin);
            var integrated_scattering = S * (1.0 - delta_transmittance) / clouds_density_sampled;
            integrated_scattering = integrated_scattering * view_T
                + airlight_radiance * (vec3f(1.0) - view_T) * (1.0 - delta_transmittance);

            scattered_light += transmittance * integrated_scattering;
            transmittance *= delta_transmittance;
        }

        if transmittance <= config.clouds_min_transmittance { break; }

        dir_length += MIN_STEP_M;
        refined_until = max(refined_until, dir_length);
        if (consecutive_empty >= REFINE_EMPTY_LIMIT) {
            refining = false;
            consecutive_empty = 0u;
        }
    }

    // Entry-distance dissolve: rays that only meet the shell far away hand
    // the whole interval to the composite's band march (which fades in over
    // the same window). Unlike the pre-CLOUD-6 haze-out, this replaces the
    // clouds with the far estimator rather than deleting them.
    let entry_fade = 1.0 - smoothstep(ENTRY_FADE_START, ENTRY_FADE_END, ray.start);
    transmittance = mix(1.0, transmittance, entry_fade);
    scattered_light *= entry_fade;

    // Soft energy peak: per-channel Reinhard against a modest white point so
    // stacked phase peaks no longer blow the composite to a white sheet.
    let peak_limit = 2.2;
    scattered_light = scattered_light / (vec3f(1.0) + scattered_light / peak_limit);

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
    var best = CloudsOutput(vec4f(0.0, 0.0, 0.0, 1.0), MAX_DISTANCE);
    var best_error = MAX_DISTANCE;
    for (var i = 0u; i < 4u; i += 1u) {
        let coord = vec2u(coords[i]);
        let dist = textureLoad(history_distance_texture, coord, 0).r;
        if dist < 1.0e8 {
            let error = abs(dist - expected_dist);
            if error < best_error {
                best_error = error;
                best = CloudsOutput(textureLoad(history_texture, coord, 0), dist);
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
            ), MAX_DISTANCE);
        }
        if frag_coord.x < 6.0 && frag_coord.x >= 5.0 {
            return CloudsOutput(vec4f(f32(config.history_epoch), config.time, 0.0, 0.0), MAX_DISTANCE);
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
        let old_dist = textureLoad(history_distance_texture, current_texel, 0).r;
        return CloudsOutput(textureLoad(history_texture, current_texel, 0), old_dist);
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
        return CloudsOutput(mix(col, history, steady_weight), result.dist);
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
                    return CloudsOutput(mix(col, hist, w), result.dist);
                }
        }
    }
    return CloudsOutput(col, result.dist);
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
