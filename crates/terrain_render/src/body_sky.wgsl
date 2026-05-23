// Unified atmosphere fullscreen pass per body.
//
// Renders one fullscreen quad per body (per camera) that integrates
// single-scattering Rayleigh + Mie atmospheric scattering for every view
// ray. The integration interval is clipped by both the body's atmosphere
// shell and the scene depth from `scene_depth_texture` (a per-frame copy
// of the main pass's depth attachment maintained by `CopySceneDepthNode`
// on the game crate's side). It also draws the same reference cloud cover
// used by the impostor on a fixed-altitude shell. The game keeps this pass
// visible while real terrain LOD is active:
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

fn cloud_shell_overlay(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    planet_center: vec3<f32>,
    planet_radius: f32,
    t_min: f32,
    t_max: f32,
    surface_fade: f32,
) -> CloudOverlay {
    let coverage = sky_atmos.cloud_albedo_coverage.w;
    if coverage <= 0.0 || t_max <= t_min || surface_fade <= 0.0 {
        return no_cloud_overlay();
    }

    // Same fixed shell as the impostor: ~0.15% of body radius, about 9 km
    // on a 6000 km world. The terrain path uses this as a detached overlay
    // rather than painting clouds into the surface material.
    let cloud_altitude = planet_radius * 0.0015;
    let cloud_r = planet_radius + cloud_altitude;
    let oc = cam_pos - planet_center;
    let half_b = dot(oc, ray_dir);
    let c_cloud = dot(oc, oc) - cloud_r * cloud_r;
    let disc_cloud = half_b * half_b - c_cloud;
    if disc_cloud <= 0.0 {
        return no_cloud_overlay();
    }

    let sq = sqrt(disc_cloud);
    let t0 = -half_b - sq;
    let t1 = -half_b + sq;
    var t_cloud = 1.0e30;
    if t0 > t_min && t0 < t_max {
        t_cloud = t0;
    } else if t1 > t_min && t1 < t_max {
        t_cloud = t1;
    } else {
        return no_cloud_overlay();
    }

    let cloud_hit = cam_pos + t_cloud * ray_dir;
    let cloud_normal_ws = normalize(cloud_hit - planet_center);
    let cloud_dir_local = rotate_quat(
        sky_atmos_extra.world_to_body_orientation,
        cloud_normal_ws,
    );
    let main_density = sample_cloud_banded(cloud_dir_local);

    let density = clamp(main_density * 2.0 * coverage, 0.0, 1.0);
    if density < 1.0e-3 {
        return no_cloud_overlay();
    }

    let sun_dir = sky_atmos_extra.sun_dir_flux.xyz;
    let raw_ndl = dot(cloud_normal_ws, sun_dir);
    let night_suppress = smoothstep(-0.15, 0.10, raw_ndl);
    if night_suppress <= 0.0 {
        return no_cloud_overlay();
    }

    let wrap = 0.15;
    let n_dot_l = clamp((raw_ndl + wrap) / (1.0 + wrap), 0.0, 1.0);
    let core = smoothstep(0.75, 1.00, density);
    let self_shadow = mix(1.0, 0.80, core);
    let cloud_lit = sky_atmos.cloud_albedo_coverage.xyz
        * n_dot_l
        * self_shadow
        * sky_atmos_extra.sun_dir_flux.w
        * SCENE_FLUX_SCALE;

    let tau = density * density * 3.0;
    let opacity = clamp(1.0 - exp(-tau), 0.0, 1.0) * night_suppress * surface_fade;
    return CloudOverlay(cloud_lit * night_suppress * opacity, opacity);
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

    let cloud = cloud_shell_overlay(
        cam_pos,
        ray_dir,
        planet_center,
        planet_radius,
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
