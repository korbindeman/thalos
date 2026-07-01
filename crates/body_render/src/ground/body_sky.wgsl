// Unified atmosphere fullscreen pass per body.
//
// Renders one fullscreen quad per body (per camera) that integrates
// single-scattering Rayleigh + Mie atmospheric scattering for every view
// ray. The integration interval is clipped by both the body's atmosphere
// shell and the scene depth from `scene_depth_texture` (a per-frame copy
// of the main pass's depth attachment maintained by `CopySceneDepthNode`
// on the game crate's side). It also raymarches a volumetric cloud layer as
// a thin slab between two concentric shells, using the same reference cloud
// cover as the impostor for the large-scale weather pattern. The game keeps
// this pass visible while real terrain LOD is active:
//
//   * Mid  — camera outside the shell but terrain is visible. The pass
//     produces in-front haze/clouds on terrain pixels and halo on rim pixels.
//   * Near — camera inside the shell. The integral runs from cam to terrain
//     depth (aerial perspective) or to the shell exit on sky pixels.
//
// Depth-compare is disabled (`Always` in `sky_material.rs::specialize`), so
// the quad rasterizes on every pixel, including terrain. The integration
// length comes from scene_depth, not from the depth attachment.

#import bevy_pbr::mesh_view_bindings::view
#import thalos::atmosphere::{
    AtmosphereBlock,
    CLOUD_BAND_COUNT,
    atmosphere_jitter,
    cloud_band_phase,
    integrate_atmosphere_multiscatter,
    rotate_around_y,
}
#import thalos::lighting::SCENE_FLUX_SCALE
#import thalos::water::shade_ocean

// Standard MaterialPlugin bind group in Bevy 0.18: group 3 (group 2 is the
// material-indices storage buffer used by the bindless material allocator).
@group(3) @binding(0) var<uniform> sky_atmos: AtmosphereBlock;

struct SkyAtmosExtra {
    sun_dir_flux:              vec4<f32>,  // xyz = sun dir (normalized), w = flux
    planet_center_radius:      vec4<f32>,  // xyz = planet center (render-space), w = radius
    world_to_body_orientation: vec4<f32>,  // render-space direction -> body-local cubemap direction
    cloud_band_radii:          vec4<f32>,  // x = cloud base radius, y = cloud top radius (render units), z = airlight ratio, w = cloud-composite-enable flag (1 = bind live clouds)
    ocean:                     vec4<f32>,  // x = ocean sphere radius (render units = m), y = enable (>=0.5), z = wave-scroll time (s), w = camera height above sea (m, CPU f64-precise)
    ocean_color_depth:         vec4<f32>,  // xyz = deep-water linear-RGB tint, w = min optical-depth scale
}
@group(3) @binding(1) var<uniform> sky_atmos_extra: SkyAtmosExtra;

// Scene-depth copy: contains the main pass's depth attachment at the
// moment the copy node runs (between `Opaque3d` and `Transparent3d`).
// `texture_depth_2d` is sampled with `textureLoad` (no sampler) for
// unfiltered exact texel reads at fragment coordinates.
@group(3) @binding(2) var scene_depth_texture: texture_depth_2d;

// Reference cloud-cover cubemap. Matches `PlanetMaterial`'s cloud binding
// semantically, but lives on this material because the terrain path needs the
// cloud shell in the fullscreen pass rather than in the terrain material.
@group(3) @binding(3) var cloud_cover_tex: texture_cube<f32>;
@group(3) @binding(4) var cloud_cover_sampler: sampler;

// Precomputed multi-scatter LUT (Rgba16Float, 32×32). Each cell stores the
// average single-scattered radiance arriving at a point from every direction
// (per unit sun flux × strength), indexed by (u = (sun·zenith + 1) / 2,
// v = altitude / atmos_top). `integrate_atmosphere_multiscatter` samples it at
// every view step and adds the second bounce, which is what gives the daytime
// sky its blue luminance and lets the alpha boost below crush stars at noon.
@group(3) @binding(5) var ms_lut_tex: texture_2d<f32>;
@group(3) @binding(6) var ms_lut_sampler: sampler;

// High-fidelity volumetric cloud layer (thalos_volumetric_clouds raymarch
// output): rgb = premultiplied in-scatter, a = transmittance. Sampled in
// screen space with textureLoad and composited over the atmosphere in-scatter
// below — but only when `cloud_band_radii.w >= 0.5` (the body whose live texture
// is bound). Bodies with no active cloud layer carry a 1×1 blank here, which the
// screen-space loads would read out of bounds (→ opaque black sky); the w flag
// gates the composite so the blank is never sampled.
@group(3) @binding(7) var cloud_layer_tex: texture_2d<f32>;

// Per-pixel nearest cloud-hit distance from the same raymarch (R32F, metres
// from the camera; ≥ 1e8 sentinel = no cloud on this ray). Drives the
// geometry-occlusion ramp below. 1×1 far-sentinel fallback on bodies without
// an active cloud layer.
@group(3) @binding(8) var cloud_distance_tex: texture_2d<f32>;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    // Mesh = Rectangle::new(2.0, 2.0), corners at ±1 in local x/y. Pass
    // through unchanged to cover the entire viewport in NDC regardless of
    // where the entity is parented. `z = 1.0` is the near plane in
    // reverse-Z; since depth_compare = Always the value doesn't matter
    // beyond keeping clip-space valid.
    var out: VertexOutput;
    out.clip_position = vec4(in.position.x, in.position.y, 1.0, 1.0);
    return out;
}

fn rotate_quat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

fn sample_cloud_banded(dir_local: vec3<f32>) -> f32 {
    let sin2 = clamp(dir_local.y * dir_local.y, 0.0, 1.0);
    let bf = sin2 * f32(CLOUD_BAND_COUNT - 1u);
    let lo = u32(floor(bf));
    let hi = min(lo + 1u, CLOUD_BAND_COUNT - 1u);
    let alpha = bf - floor(bf);
    let phase_lo = cloud_band_phase(lo, sky_atmos);
    let phase_hi = cloud_band_phase(hi, sky_atmos);
    let dir_lo = rotate_around_y(dir_local, phase_lo);
    let dir_hi = rotate_around_y(dir_local, phase_hi);
    let s_lo = textureSampleLevel(cloud_cover_tex, cloud_cover_sampler, dir_lo, 0.0).r;
    let s_hi = textureSampleLevel(cloud_cover_tex, cloud_cover_sampler, dir_hi, 0.0).r;
    return mix(s_lo, s_hi, alpha);
}

struct CloudOverlay {
    premul_rgb: vec3<f32>,
    opacity: f32,
}

fn no_cloud_overlay() -> CloudOverlay {
    return CloudOverlay(vec3<f32>(0.0), 0.0);
}

// ── Volumetric cloud layer ──────────────────────────────────────────────
//
// A thin-slab raymarch between two concentric spheres: cloud base
// `r_base = radius + cloud_shape.x` and cloud top `r_base + cloud_shape.y`.
// Density at each sample is the banded reference-coverage cube (the large-
// scale weather map, co-rotating via `sample_cloud_banded`) shaped by a
// vertical profile and eroded by a few octaves of animated 3-D value noise.
// Each lit sample takes a short march toward the sun for self-shadowing; a
// forward-biased phase plus a powder term give the silver-lining read. The
// marched segment is clipped by `t_max` (atmosphere exit / scene depth) so
// terrain and the ship hull correctly occlude clouds. The result is
// premultiplied and composited exactly like the old flat-shell overlay.

const CLOUD_VIEW_STEPS_MAX: u32 = 32u;
const CLOUD_VIEW_STEPS_MIN: u32 = 8u;
const CLOUD_SUN_STEPS: u32 = 4u;
// Per-render-unit (== per-meter at SHIP_SCALE) extinction at density 1. Tuned
// so a ~5 km slab at full coverage builds near-opaque cores while thin edges
// stay translucent. The authored `density` (`cloud_shape.z`) scales this.
const CLOUD_BASE_EXTINCTION: f32 = 0.0016;
// Feature size of the largest detail-noise octave, in meters.
const CLOUD_NOISE_FEATURE_M: f32 = 9000.0;

// 3-D → scalar hash in [0, 1) (Hoskins hash13). Integer-mix, no trig — a
// sin-based hash bands visibly at planet-scale coordinates.
fn cloud_hash(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.zyx + 31.32);
    return fract((q.x + q.y) * q.z);
}

// Trilinearly-interpolated value noise.
fn cloud_value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = cloud_hash(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = cloud_hash(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = cloud_hash(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = cloud_hash(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = cloud_hash(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = cloud_hash(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = cloud_hash(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = cloud_hash(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// 3-octave fbm normalized to [0, 1].
fn cloud_fbm(p: vec3<f32>) -> f32 {
    var f = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    var norm = 0.0;
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        f += amp * cloud_value_noise(p * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.6;
    }
    return f / norm;
}

// Cheap forward-scatter bias in ~[0.5, 1.1]; not physical, tuned so sun-facing
// clouds pick up a silver lining without the back side going dark.
fn cloud_phase(cos_theta: f32) -> f32 {
    return 0.5 + 0.6 * pow(max(cos_theta, 0.0), 2.0);
}

// Interleaved-gradient noise for the per-pixel raymarch start offset. Clouds
// are high-frequency so this dither is invisible (unlike on smooth sky) and
// lets a modest step count avoid slab banding.
fn cloud_jitter(coord: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(coord, vec2<f32>(0.06711056, 0.00583715))));
}

// Slow body-local drift so the detail noise evolves over time. The coverage
// cube already carries the large-scale rotation; this is gentle "boiling".
fn cloud_wind_offset() -> vec3<f32> {
    let t = sky_atmos.cloud_dynamics.y;
    return vec3<f32>(t * 2.0e-6, t * 0.6e-6, t * 1.3e-6);
}

// Normalized cloud density in [0, 1] at a world-space point.
fn cloud_density_at(
    pos: vec3<f32>,
    planet_center: vec3<f32>,
    r_base: f32,
    r_top: f32,
    coverage: f32,
    wind: vec3<f32>,
) -> f32 {
    let rel = pos - planet_center;
    let r = length(rel);
    let h = (r - r_base) / max(r_top - r_base, 1.0);
    if h < 0.0 || h > 1.0 {
        return 0.0;
    }
    let dir_world = rel / max(r, 1.0e-3);
    let dir_local = rotate_quat(sky_atmos_extra.world_to_body_orientation, dir_world);
    // Large-scale coverage from the reference weather cube. The ×2×coverage
    // map makes the authored `coverage` approximate the overcast fraction,
    // matching the impostor convention.
    let cov = clamp(sample_cloud_banded(dir_local) * 2.0 * coverage, 0.0, 1.0);
    if cov <= 1.0e-3 {
        return 0.0;
    }
    // Rounded base, eroded top.
    let vprofile = smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.55, 1.0, h));
    // Detail noise in body-local space so billows co-rotate with the surface.
    let np = rotate_quat(sky_atmos_extra.world_to_body_orientation, rel)
        / CLOUD_NOISE_FEATURE_M + wind;
    let n = cloud_fbm(np);
    // Coverage-threshold erosion: high coverage keeps clouds even where noise
    // is low; low coverage carves them away (the classic `n - (1 - cov)`).
    let d = clamp((n + cov - 1.0) * 1.6, 0.0, 1.0);
    return d * vprofile;
}

// Beer's-law transmittance from `pos` toward the sun across ~one slab depth.
fn cloud_sun_transmittance(
    pos: vec3<f32>,
    planet_center: vec3<f32>,
    r_base: f32,
    r_top: f32,
    coverage: f32,
    wind: vec3<f32>,
    sun_dir: vec3<f32>,
    extinction: f32,
) -> f32 {
    let ds = (r_top - r_base) / f32(CLOUD_SUN_STEPS);
    var tau = 0.0;
    var p = pos;
    for (var i: u32 = 0u; i < CLOUD_SUN_STEPS; i = i + 1u) {
        p += sun_dir * ds;
        let d = cloud_density_at(p, planet_center, r_base, r_top, coverage, wind);
        tau += d * extinction * ds;
    }
    return exp(-tau);
}

fn cloud_volume_overlay(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    planet_center: vec3<f32>,
    planet_radius: f32,
    pixel_coord: vec2<f32>,
    t_min: f32,
    t_max: f32,
    surface_fade: f32,
) -> CloudOverlay {
    let coverage = sky_atmos.cloud_albedo_coverage.w;
    let base_alt = sky_atmos.cloud_shape.x;
    let thickness = sky_atmos.cloud_shape.y;
    let density_mult = max(sky_atmos.cloud_shape.z, 0.0);
    if coverage <= 0.0 || thickness <= 0.0 || density_mult <= 0.0 || t_max <= t_min {
        return no_cloud_overlay();
    }

    let r_base = planet_radius + base_alt;
    let r_top = r_base + thickness;

    let oc = cam_pos - planet_center;
    let cam_r = length(oc);
    let b = dot(oc, ray_dir);

    // Top-shell intersection. A miss means the ray never reaches cloud
    // altitude at all.
    let c_top = dot(oc, oc) - r_top * r_top;
    let disc_top = b * b - c_top;
    if disc_top <= 0.0 {
        return no_cloud_overlay();
    }
    let sq_top = sqrt(disc_top);
    let tt0 = -b - sq_top;
    let tt1 = -b + sq_top;

    // Base-shell intersection (may miss when the ray grazes above the base).
    let c_base = dot(oc, oc) - r_base * r_base;
    let disc_base = b * b - c_base;
    let hit_base = disc_base > 0.0;
    var tb0 = 0.0;
    var tb1 = 0.0;
    if hit_base {
        let sq_base = sqrt(disc_base);
        tb0 = -b - sq_base;
        tb1 = -b + sq_base;
    }

    // Resolve the first forward slab segment for the three camera regimes.
    var seg_start = 0.0;
    var seg_end = 0.0;
    var fade = 1.0;
    if cam_r > r_top {
        // Above the layer (orbit / space): clouds only over the disk, faded
        // across the geometric horizon to avoid a hard limb tangent band.
        if surface_fade <= 0.0 {
            return no_cloud_overlay();
        }
        fade = surface_fade;
        seg_start = tt0;
        if hit_base && tb0 > tt0 {
            seg_end = tb0;
        } else {
            seg_end = tt1;
        }
    } else if cam_r < r_base {
        // Below the layer (on / near the surface): show the underside even on
        // sky pixels. Reaching the base from below = base-shell exit `tb1`.
        if hit_base && tb1 > 0.0 {
            seg_start = tb1;
        } else {
            seg_start = max(tt0, 0.0);
        }
        seg_end = tt1;
    } else {
        // Inside the layer: march from the camera to the nearest shell.
        seg_start = 0.0;
        if hit_base && tb0 > 0.0 {
            seg_end = tb0;
        } else {
            seg_end = tt1;
        }
    }

    seg_start = max(seg_start, max(t_min, 0.0));
    seg_end = min(seg_end, t_max);
    if seg_end <= seg_start {
        return no_cloud_overlay();
    }

    // Adaptive step count: target one sample per ~thickness/16, clamped.
    let seg_len = seg_end - seg_start;
    let target_step = thickness / 16.0;
    let want = u32(ceil(seg_len / max(target_step, 1.0)));
    let n_steps = clamp(want, CLOUD_VIEW_STEPS_MIN, CLOUD_VIEW_STEPS_MAX);
    let ds = seg_len / f32(n_steps);

    let extinction = density_mult * CLOUD_BASE_EXTINCTION;
    let sun_dir = sky_atmos_extra.sun_dir_flux.xyz;
    let sun_flux = sky_atmos_extra.sun_dir_flux.w * SCENE_FLUX_SCALE;
    let albedo = sky_atmos.cloud_albedo_coverage.xyz;
    let phase = cloud_phase(dot(ray_dir, sun_dir));
    let wind = cloud_wind_offset();
    let jitter = cloud_jitter(pixel_coord);

    var transmittance = 1.0;
    var scattered = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < n_steps; i = i + 1u) {
        let t = seg_start + (f32(i) + jitter) * ds;
        let pos = cam_pos + t * ray_dir;
        let d = cloud_density_at(pos, planet_center, r_base, r_top, coverage, wind);
        if d > 1.0e-3 {
            let sigma = d * extinction;
            let normal = normalize(pos - planet_center);
            // Per-sample terminator fade so dark-side clouds dim smoothly.
            let night = smoothstep(-0.20, 0.05, dot(normal, sun_dir));
            let sun_t = cloud_sun_transmittance(
                pos, planet_center, r_base, r_top, coverage, wind, sun_dir, extinction,
            );
            let light = albedo * sun_flux * phase * sun_t * night;
            // Small ambient floor so shadowed undersides aren't pure black.
            let ambient = albedo * sun_flux * 0.04 * night;
            let step_trans = exp(-sigma * ds);
            scattered += transmittance * (light + ambient) * (1.0 - step_trans);
            transmittance *= step_trans;
            if transmittance < 0.01 {
                break;
            }
        }
    }

    // Fade the opacity to transparent across the terminator (using the near
    // point of the segment) so night-side clouds don't punch black holes into
    // the starfield.
    let near_normal = normalize((cam_pos + seg_start * ray_dir) - planet_center);
    let seg_night = smoothstep(-0.25, 0.05, dot(near_normal, sun_dir));
    let opacity = (1.0 - transmittance) * fade * seg_night;
    return CloudOverlay(scattered * fade, opacity);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct world-space view ray from fragment screen position.
    // Camera-basis form (vs. `world_from_clip * ndc`) keeps everything in
    // small numbers — the matrix-inverse form loses precision at orbital
    // distances when subtracted with `view.world_position`.
    let cam_right = view.world_from_view[0].xyz;
    let cam_up    = view.world_from_view[1].xyz;
    let cam_fwd   = -view.world_from_view[2].xyz;
    let ndc_x = (in.clip_position.x / view.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (in.clip_position.y / view.viewport.w) * 2.0;
    let tan_fov_y = 1.0 / view.clip_from_view[1][1];
    let tan_fov_x = 1.0 / view.clip_from_view[0][0];
    let ray_dir = normalize(
        cam_right * (ndc_x * tan_fov_x)
        + cam_up * (ndc_y * tan_fov_y)
        + cam_fwd
    );

    let cam_pos       = view.world_position;
    let planet_center = sky_atmos_extra.planet_center_radius.xyz;
    let planet_radius = sky_atmos_extra.planet_center_radius.w;
    let atmos_top_r   = planet_radius + sky_atmos.atmos_geom.x;

    // Atmosphere-shell intersection: t_enter to t_exit defines the segment
    // of the view ray that lies inside the atmosphere. When the camera is
    // already inside, t_enter clamps to 0.
    let oc        = cam_pos - planet_center;
    let oc_len_sq = dot(oc, oc);
    let b         = dot(oc, ray_dir);
    let c_atmos   = oc_len_sq - atmos_top_r * atmos_top_r;
    let disc      = b * b - c_atmos;
    if disc < 0.0 {
        discard;
    }
    let sqrt_disc = sqrt(disc);
    var t_enter   = max(-b - sqrt_disc, 0.0);
    var t_exit    = -b + sqrt_disc;
    if t_exit <= 0.0 {
        discard;
    }

    // Fallback solid-sphere hit. Scene depth is authoritative when present:
    // ground LOD terrain can sit above the mean-radius sphere, especially at
    // low grazing angles where mountains peek over the geometric horizon. If
    // the fallback sphere clips first, those depth-visible terrain pixels get
    // composited as though the ray ended at the hidden reference sphere, which
    // crushes the horizon into a dark band.
    let c_planet    = oc_len_sq - planet_radius * planet_radius;
    let disc_planet = b * b - c_planet;
    var fallback_t_surface: f32 = 1.0e30;
    var fallback_surface_fade: f32 = 0.0;
    var surface_fade: f32 = 0.0;
    // Camera→surface distance on a geometry/fallback hit, for aerial perspective.
    var surface_dist: f32 = 0.0;
    // Distance to opaque geometry at this pixel (ship hull, terrain), or a large
    // sentinel when the pixel is sky. Used below to keep the cloud layer from
    // painting over geometry that sits in front of the cloud band.
    var scene_t: f32 = 1.0e30;
    if disc_planet > 0.0 {
        let sqrt_disc_planet = sqrt(disc_planet);
        let t_planet = -b - sqrt_disc_planet;
        if t_planet > 0.0 {
            fallback_t_surface = t_planet;
            // Fade cloud compositing in across the geometric horizon.
            // Without this, an observer above the fixed cloud deck sees a
            // hard tangent band where sky-only rays begin hitting the cloud
            // shell. The fade is in metres because ship space is 1 unit = 1 m.
            fallback_surface_fade = smoothstep(0.0, 20000.0, sqrt_disc_planet);
        }
    }

    // Clip at scene depth too: if there is opaque geometry in this pixel
    // (terrain, ship hull, impostor body), terminate the raymarch there.
    // `depth_sample == 0` means "cleared / no geometry at this pixel" in
    // reverse-Z; skip the clip in that case.
    let depth_sample = textureLoad(scene_depth_texture, vec2<i32>(in.clip_position.xy), 0);
    if depth_sample > 0.0 {
        // Reconstruct view-space position at the sampled depth, then take
        // its length as the world-space distance from the camera (the
        // view-from-world basis preserves distances).
        let view_pos_h = view.view_from_clip * vec4<f32>(ndc_x, ndc_y, depth_sample, 1.0);
        let view_pos   = view_pos_h.xyz / view_pos_h.w;
        let t_scene    = length(view_pos);
        scene_t = t_scene;
        // Distinguish NEAR geometry (terrain, ship hull — always within the
        // atmosphere shell plus a small margin) from a FAR background celestial
        // body (a moon/planet impostor seen through this atmosphere, millions of
        // metres away). For near geometry, clip the raymarch at it and run the
        // surface aerial-perspective path. For a far body, leave `t_exit` at the
        // shell exit and `surface_fade = 0` so this pixel is treated exactly like
        // a SKY pixel: the in-scatter integrates the full air column and the
        // perceptual sky-luminance opacity boost below CRUSHES the body by day
        // (the same way it crushes stars), while a dim night sky lets it show.
        // Without this the impostor was veiled as if it were distant *terrain*
        // 190,000 km away, leaving Mira far too prominent in daylight.
        if t_scene <= atmos_top_r * 4.0 {
            t_exit = min(t_exit, t_scene);
            surface_fade = 1.0;
            surface_dist = t_scene;
        }
    } else if fallback_t_surface < 1.0e29 {
        t_exit = min(t_exit, fallback_t_surface);
        surface_fade = fallback_surface_fade;
        surface_dist = fallback_t_surface;
    }

    // ── Analytic ocean ────────────────────────────────────────────────────
    // Ray-trace a math sphere at sea level and treat its surface as water
    // wherever it sits in FRONT of the opaque seabed/terrain (`scene_t`). This
    // is the smooth replacement for the meshed water shell: no facets, no sag,
    // identical from orbit to sea level.
    //
    // Numerical stability at planet radius is the whole ballgame here. The naive
    // `oc·oc − r_sea²` (two ~R² terms) and the near root `−b − √disc` (two ~R
    // terms) both catastrophically cancel in f32, so the surface jitters by
    // metres as the camera moves. Instead we take the camera's EXACT height
    // above the sea `h` from the CPU (f64-computed), form `c_sea = h·(2r+h)`
    // with no cancellation, and recover the near root from `t_near·t_far = c_sea`
    // (Vieta) using the well-conditioned far root `t_far = −b + √disc`.
    var water_here = false;
    var t_ocean = 0.0;
    var ocean_column_m = 0.0;
    if sky_atmos_extra.ocean.y >= 0.5 {
        let r_sea = sky_atmos_extra.ocean.x;
        let h = sky_atmos_extra.ocean.w;              // camera height above sea (m)
        let up = normalize(oc);                        // planet-centre → camera, unit
        let mu = dot(up, ray_dir);
        let cam_r = r_sea + h;
        let c_sea = h * (2.0 * r_sea + h);             // = cam_r² − r_sea², cancellation-free
        let b_sea = cam_r * mu;
        let disc_sea = b_sea * b_sea - c_sea;
        if disc_sea > 0.0 {
            let sq_sea = sqrt(disc_sea);
            let t_far = -b_sea + sq_sea;                // sum of positives when looking down (mu<0)
            if t_far > 0.0 {
                // Vieta near root: same sign as t_far, no large-cancellation.
                let t_near = c_sea / t_far;
                t_ocean = select(t_far, t_near, t_near > 0.0);
                if t_ocean > 0.0 && t_ocean <= scene_t {
                    water_here = true;
                    ocean_column_m = max(scene_t - t_ocean, 0.0);
                    // Integrate the air column to the WATER surface, not the
                    // seabed behind it, so aerial perspective lands on the water.
                    t_exit = min(t_exit, t_ocean);
                    surface_fade = 1.0;
                    surface_dist = t_ocean;
                }
            }
        }
    }

    if t_exit <= t_enter {
        discard;
    }

    let jitter = atmosphere_jitter(in.clip_position.xy);
    var scatter = integrate_atmosphere_multiscatter(
        cam_pos, ray_dir, planet_center,
        sky_atmos_extra.sun_dir_flux.xyz,
        sky_atmos_extra.sun_dir_flux.w * SCENE_FLUX_SCALE,
        t_enter, t_exit, planet_radius, sky_atmos, jitter,
        ms_lut_tex, ms_lut_sampler,
    );

    // Aerial-perspective decoupling. The authored in-scatter strength is tuned
    // so the SKY DOME reads bright and crushes stars, but the same in-scatter is
    // also the airlight added on top of terrain — at that strength it over-fogs
    // the ground at any altitude. `cloud_band_radii.z` is the CPU-computed
    // ratio `aerial_perspective_strength / sky_strength` (see `AtmosphereTuning`):
    // it scales the in-scatter on surface/geometry-hit pixels only down to an
    // absolute clear-weather airlight, blended by `surface_fade` so it eases to
    // full sky-dome strength across the horizon (sky pixels are untouched).
    // Extinction (transmittance) is left physical — it already matches
    // Earth-clear-day visibility — so this dims the additive haze veil without
    // changing how distance fades contrast. `0` is unset (airless / pre-first-
    // update); treat it as full strength so we never blank the in-scatter.
    let airlight_ratio = sky_atmos_extra.cloud_band_radii.z;

    // Aerial perspective. The clear-weather airlight above keeps NEAR ground
    // crisp, but real distant terrain desaturates and tints toward the sky as
    // the air column between camera and surface grows. We drive an artistic
    // veil from camera→surface distance and fade the surface toward the
    // atmospheric in-scatter (haze) colour: it is folded into BOTH the additive
    // in-scatter strength (here) and the dst-attenuation opacity (below), so the
    // composite reduces to a clean mix(terrain, haze, veil) at range while
    // leaving near ground at its tuned look. Physical extinction is untouched.
    // Gated to bodies with an atmosphere (airlight_ratio > 0) so airless
    // surfaces (Mira) are never veiled toward a black/near-zero in-scatter.
    // Air-mass driver. The veil must scale with how much air the view ray
    // actually traverses to reach the surface, NOT the Euclidean camera→surface
    // distance. `view_tau` is the mean optical depth the integrator already
    // accumulated over `[t_enter, t_exit]` (recovered from its transmittance):
    // a thin vertical column at nadir-from-orbit, a long slant column at the
    // limb or along a low horizontal flight path. Keying on distance instead
    // saturated the ramp for the WHOLE disc the moment the camera left the
    // atmosphere — veiling even the crisp nadir uniformly (the "washed out from
    // orbit" bug). The tau thresholds are calibrated to the old distance ramp at
    // sea level (`view_tau ≈ 0.30` at the 8 km onset, `≈ 2.40` at the 70 km full
    // veil), so the on-surface look is unchanged and only altitude re-grades it.
    let mean_trans = (scatter.transmittance.x + scatter.transmittance.y + scatter.transmittance.z) / 3.0;
    let view_tau = -log(clamp(mean_trans, 1.0e-4, 1.0));
    let aerial_tau_near = 0.30;
    let aerial_tau_far = 2.40;
    let aerial_max = 0.72;
    let aerial = select(
        0.0,
        smoothstep(aerial_tau_near, aerial_tau_far, view_tau)
            * aerial_max
            * clamp(surface_fade, 0.0, 1.0),
        airlight_ratio > 0.0,
    );
    // `surface_dist` is retained for readability of the hit-classification branches
    // above but no longer drives the veil (air mass does). Phony-assign so naga
    // doesn't flag it unused.
    _ = surface_dist;

    let base_surface_airlight = mix(1.0, airlight_ratio, clamp(surface_fade, 0.0, 1.0));
    let surface_airlight = max(base_surface_airlight, aerial);
    let airlight_scale = select(surface_airlight, 1.0, airlight_ratio <= 0.0);
    scatter.in_scatter = scatter.in_scatter * airlight_scale;

    // Composite the high-fidelity volumetric cloud layer: a screen-space sample
    // of the `thalos_volumetric_clouds` raymarch output, rather than the legacy
    // in-shader slab march (`cloud_volume_overlay`, kept for reference). Doing it
    // inside this fullscreen pass — after the atmosphere in-scatter — lands the
    // clouds deterministically on top of the sky; a separate transparent quad
    // sorted unreliably against this pass under big_space.
    //
    // Manual bilinear: the layer is RGBA32F (not filterable), and the cloud
    // texture is a fixed 1920×1080 while the viewport may be larger (4K) —
    // nearest sampling turns the raymarch's per-texel dither into visible
    // 2×2-block checkering. Four loads + lerp smooth both the upscale and
    // most of the dither. Clamped to row 1077: the top two texture rows
    // (screen-bottom) hold the compute pass's camera-save payload.
    let cloud_res = vec2<f32>(1920.0, 1080.0);
    let cloud_uv = in.clip_position.xy / view.viewport.zw;
    let cloud_p = cloud_uv * cloud_res - 0.5;
    let cloud_base = floor(cloud_p);
    let cloud_f = cloud_p - cloud_base;
    let cb = clamp(vec2<i32>(cloud_base), vec2<i32>(0, 0), vec2<i32>(1918, 1076));
    let cs00 = textureLoad(cloud_layer_tex, cb, 0);
    let cs10 = textureLoad(cloud_layer_tex, cb + vec2<i32>(1, 0), 0);
    let cs01 = textureLoad(cloud_layer_tex, cb + vec2<i32>(0, 1), 0);
    let cs11 = textureLoad(cloud_layer_tex, cb + vec2<i32>(1, 1), 0);
    let cloud_sample = mix(mix(cs00, cs10, cloud_f.x), mix(cs01, cs11, cloud_f.x), cloud_f.y);
    let cloud_texel = vec2<i32>(cloud_uv * cloud_res);

    // Suppress the cloud where opaque geometry sits in front of it, so close
    // geometry (the ship hull) isn't painted over by clouds behind it.
    // `cloud_near` is the per-pixel distance at which the raymarch first hit
    // actual cloud density on this ray (exported alongside the cloud layer);
    // if the scene depth is nearer, the cloud is entirely behind the geometry.
    // `cloud_vis` then ramps from 0 (geometry before the first cloud) to 1
    // (geometry past the band exit `band_far`, from a ray-shell intersection)
    // — an approximation of the in-front fraction that makes the ship cross
    // the cloud boundary smoothly instead of popping. Using the true per-pixel
    // hit distance (rather than the geometric base-shell crossing) keeps
    // geometry under a sparse deck from dimming clouds that are actually far
    // behind it.
    let r_cloud_base = sky_atmos_extra.cloud_band_radii.x;
    let r_cloud_top = sky_atmos_extra.cloud_band_radii.y;
    let cam_r_len = sqrt(oc_len_sq);
    let disc_cb = b * b - (oc_len_sq - r_cloud_base * r_cloud_base);
    let disc_ct = b * b - (oc_len_sq - r_cloud_top * r_cloud_top);
    let sqrt_cb = sqrt(max(disc_cb, 0.0));
    let sqrt_ct = sqrt(max(disc_ct, 0.0));
    var band_far = 1.0e30;
    if cam_r_len < r_cloud_base {
        // Below the deck: the band ends at the far top-shell exit.
        band_far = max(-b + sqrt_ct, 0.0);
    } else if cam_r_len <= r_cloud_top {
        // Inside the deck: nearest forward shell crossing (top exit, or the
        // downward base crossing).
        var bf = -b + sqrt_ct;
        let base_down = -b - sqrt_cb;
        if disc_cb > 0.0 && base_down > 0.0 {
            bf = min(bf, base_down);
        }
        band_far = max(bf, 1.0);
    } else {
        // Above the deck: exit at the base near root (ray dips through) or
        // the top far root otherwise.
        if disc_cb > 0.0 {
            band_far = max(-b - sqrt_cb, 0.0);
        } else {
            band_far = max(-b + sqrt_ct, 0.0);
        }
    }
    let cloud_near = textureLoad(cloud_distance_tex, cloud_texel, 0).r;
    var cloud_vis = 1.0;
    if scene_t < 1.0e29 && r_cloud_top > r_cloud_base {
        let near = min(cloud_near, band_far);
        cloud_vis = clamp((scene_t - near) / max(band_far - near, 1.0), 0.0, 1.0);
    }
    // `cloud_band_radii.w` is the composite-enable flag, set to 1.0 by
    // `update_body_terrain_atmosphere` only on the body whose live cloud texture
    // is bound. On every other body — and when clouds are disabled in graphics
    // settings (the active cloud body is then cleared) — it is 0.0 and the cloud
    // layer is skipped. This guard is load-bearing: those bodies carry the 1×1
    // blank `cloud_layer_tex`, but the screen-space `textureLoad`s above read
    // texels up to 1919×1077 — out of bounds for a 1×1 texture, which returns
    // (0,0,0,0) → transmittance 0 → an opaque black sky. Never composite the
    // blank.
    var cloud = CloudOverlay(vec3<f32>(0.0), 0.0);
    if (sky_atmos_extra.cloud_band_radii.w >= 0.5) {
        cloud = CloudOverlay(cloud_sample.rgb * cloud_vis, (1.0 - cloud_sample.a) * cloud_vis);
    }

    // Premultiplied: `rgb` is already weighted by sun flux and β coefficients
    // inside `integrate_atmosphere`. Alpha is the mean opacity over the three
    // channels — the standard `Premultiplied` blend dims what was drawn
    // behind (terrain albedo, impostor surface, stars).
    // `mean_trans` computed above (drives the air-mass aerial veil).
    let physical_opacity = clamp(1.0 - mean_trans, 0.0, 1.0);

    // Perceptual sky-luminance opacity boost. The dst-attenuation factor of
    // a premultiplied blend is `1 − α`, so to make a bright daytime sky drown
    // out stars the alpha has to approach 1.0 even when extinction (and
    // therefore physical opacity) is small — Earth's midday sky has τ_v ≈ 0.2
    // for blue, so the physically correct factor only dims background stars
    // by ~20%, far too little against star peak values in the hundreds. Stars
    // are calibrated to be visible against a black sky, so the perceptual fix
    // is to crush them whenever the local in-scatter radiance is high enough
    // that a real observer's eye would adapt away from them. Restricted to
    // sky pixels (no opaque hit) so terrain aerial perspective stays driven
    // by physical transmittance only. The analytic planet-sphere fallback
    // also counts as a surface hit; otherwise the horizon flips between
    // boosted sky opacity and physical surface transmittance as a hard band.
    var opacity = physical_opacity;
    if surface_fade <= 0.0 {
        let sky_lum = max(scatter.in_scatter.r,
                          max(scatter.in_scatter.g, scatter.in_scatter.b));
        let lum_opacity = smoothstep(0.03, 0.20, sky_lum);
        opacity = max(opacity, lum_opacity);
    } else {
        // Surface aerial veil: lift the dst-attenuation to match the `aerial`
        // in-scatter strength added above, so distant terrain's own colour is
        // replaced by the haze colour (desaturate + tint) rather than merely
        // brightened. Near ground keeps `physical_opacity` (aerial ≈ 0).
        opacity = max(opacity, aerial);
    }
    let combined_opacity = clamp(
        1.0 - (1.0 - opacity) * (1.0 - cloud.opacity),
        0.0,
        1.0,
    );
    // Attenuate the atmosphere in-scatter by the cloud transmittance so opaque
    // cloud cores hide the sky behind them, then add the cloud's own premult
    // in-scatter on top. `1 - cloud.opacity` already folds in the depth
    // suppression above.
    let sky_rgb = scatter.in_scatter * (1.0 - cloud.opacity) + cloud.premul_rgb;

    // Analytic ocean composite. Water occludes the seabed already in the
    // framebuffer, so we supply its radiance here and output fully opaque
    // (alpha = 1, the framebuffer seabed contributes 0). The surface is dimmed
    // by air transmittance `(1 − opacity)` (physical + aerial veil, mirroring
    // how terrain in the framebuffer is attenuated) and by clouds in front of
    // it. The in-scatter / clouds were already integrated to the water surface
    // (`t_exit = t_ocean`).
    if water_here {
        let hit_ws = cam_pos + t_ocean * ray_dir;
        let geo_n = normalize(hit_ws - planet_center);
        let sun_flux_scaled = sky_atmos_extra.sun_dir_flux.w * SCENE_FLUX_SCALE;
        let water = shade_ocean(
            hit_ws,
            geo_n,
            -ray_dir,
            t_ocean,
            sky_atmos_extra.ocean.z,
            sky_atmos_extra.sun_dir_flux.xyz,
            sun_flux_scaled,
            sky_atmos_extra.ocean_color_depth,
            ocean_column_m,
        );
        let surf_trans = (1.0 - opacity) * (1.0 - cloud.opacity);
        // Feather the shoreline. Right at the waterline the seabed/terrain sits
        // within a metre of sea level and the grass blades straddle it, so a hard
        // water/land test dithers on the blade-height band. Ramp water coverage
        // over the first few metres of depth: the seabed framebuffer shows through
        // (partial alpha) in the shallowest sliver, giving a soft wet edge and
        // letting clear shallows read their bed. Deeper than the band it is fully
        // opaque water.
        let shore_cov = clamp(ocean_column_m / 3.0, 0.0, 1.0);
        let out_rgb = sky_rgb + water * surf_trans * shore_cov;
        let out_a = mix(combined_opacity, 1.0, shore_cov);
        return vec4(out_rgb, out_a);
    }

    return vec4(sky_rgb, combined_opacity);
}
