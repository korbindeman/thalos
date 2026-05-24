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

// Standard MaterialPlugin bind group in Bevy 0.18: group 3 (group 2 is the
// material-indices storage buffer used by the bindless material allocator).
@group(3) @binding(0) var<uniform> sky_atmos: AtmosphereBlock;

struct SkyAtmosExtra {
    sun_dir_flux:              vec4<f32>,  // xyz = sun dir (normalized), w = flux
    planet_center_radius:      vec4<f32>,  // xyz = planet center (render-space), w = radius
    world_to_body_orientation: vec4<f32>,  // render-space direction -> body-local cubemap direction
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
        t_exit = min(t_exit, t_scene);
        surface_fade = 1.0;
    } else if fallback_t_surface < 1.0e29 {
        t_exit = min(t_exit, fallback_t_surface);
        surface_fade = fallback_surface_fade;
    }

    if t_exit <= t_enter {
        discard;
    }

    let jitter = atmosphere_jitter(in.clip_position.xy);
    let scatter = integrate_atmosphere_multiscatter(
        cam_pos, ray_dir, planet_center,
        sky_atmos_extra.sun_dir_flux.xyz,
        sky_atmos_extra.sun_dir_flux.w * SCENE_FLUX_SCALE,
        t_enter, t_exit, planet_radius, sky_atmos, jitter,
        ms_lut_tex, ms_lut_sampler,
    );

    let cloud = cloud_volume_overlay(
        cam_pos,
        ray_dir,
        planet_center,
        planet_radius,
        in.clip_position.xy,
        t_enter,
        t_exit,
        surface_fade,
    );

    // Premultiplied: `rgb` is already weighted by sun flux and β coefficients
    // inside `integrate_atmosphere`. Alpha is the mean opacity over the three
    // channels — the standard `Premultiplied` blend dims what was drawn
    // behind (terrain albedo, impostor surface, stars).
    let mean_trans = (scatter.transmittance.x + scatter.transmittance.y + scatter.transmittance.z) / 3.0;
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
    }
    let combined_opacity = clamp(
        1.0 - (1.0 - opacity) * (1.0 - cloud.opacity),
        0.0,
        1.0,
    );
    return vec4(scatter.in_scatter + cloud.premul_rgb, combined_opacity);
}
