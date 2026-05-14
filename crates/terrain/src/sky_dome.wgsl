// Unified atmosphere fullscreen pass per body.
//
// Renders one fullscreen quad per body (per camera) that integrates
// single-scattering Rayleigh + Mie atmospheric scattering for every view
// ray. The integration interval is clipped by both the body's atmosphere
// shell and the scene depth from `scene_depth_texture` (a per-frame copy
// of the main pass's depth attachment maintained by `CopySceneDepthNode`
// on the game crate's side). This gives one shader three regimes for free:
//
//   * Far  — body occupies a small disc, most rays miss the shell entirely
//     (`disc < 0` → discard). Rim pixels graze the shell and produce halo.
//   * Mid  — same as Far, but with larger silhouette. Halo still comes from
//     the fullscreen pass; no separate impostor halo pass needed.
//   * Near — camera inside the shell. The integral runs from cam to the
//     terrain depth (aerial perspective) or to the shell exit on sky
//     pixels.
//
// Depth-compare is disabled (`Always` in `sky_material.rs::specialize`), so
// the quad rasterizes on every pixel, including terrain. The integration
// length comes from scene_depth, not from the depth attachment.

#import bevy_pbr::mesh_view_bindings::view
#import thalos::atmosphere::{AtmosphereBlock, integrate_atmosphere_multiscatter}

// Standard MaterialPlugin bind group in Bevy 0.18: group 3 (group 2 is the
// material-indices storage buffer used by the bindless material allocator).
@group(3) @binding(0) var<uniform> sky_atmos: AtmosphereBlock;

struct SkyAtmosExtra {
    sun_dir_flux:         vec4<f32>,  // xyz = sun dir (normalized), w = flux
    planet_center_radius: vec4<f32>,  // xyz = planet center (render-space), w = radius
}
@group(3) @binding(1) var<uniform> sky_atmos_extra: SkyAtmosExtra;

// Scene-depth copy: contains the main pass's depth attachment at the
// moment the copy node runs (between `Opaque3d` and `Transparent3d`).
// `texture_depth_2d` is sampled with `textureLoad` (no sampler) for
// unfiltered exact texel reads at fragment coordinates.
@group(3) @binding(2) var scene_depth_texture: texture_depth_2d;

// Multi-scatter LUT for this body's atmosphere. Indexed by
// `(u = (μ_s + 1) / 2, v = h / atmos_top)`; baked once on body spawn by
// `thalos_planet_lighting::bake_multi_scatter_lut`.
@group(3) @binding(3) var multi_scatter_lut: texture_2d<f32>;
@group(3) @binding(4) var multi_scatter_sampler: sampler;

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

    // Clip the raymarch at the body's solid surface (planet sphere) so the
    // ground absorbs the integral on rays that hit it without going through
    // depth sampling. Necessary on the impostor body and as a fallback when
    // scene depth is unavailable.
    let c_planet    = oc_len_sq - planet_radius * planet_radius;
    let disc_planet = b * b - c_planet;
    if disc_planet > 0.0 {
        let t_planet = -b - sqrt(disc_planet);
        if t_planet > 0.0 {
            t_exit = min(t_exit, t_planet);
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
    }

    if t_exit <= t_enter {
        discard;
    }

    let scatter = integrate_atmosphere_multiscatter(
        cam_pos, ray_dir, planet_center,
        sky_atmos_extra.sun_dir_flux.xyz,
        sky_atmos_extra.sun_dir_flux.w,
        t_enter, t_exit, planet_radius, sky_atmos, 0.5,
        multi_scatter_lut, multi_scatter_sampler,
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
    // by physical transmittance only.
    var opacity = physical_opacity;
    if depth_sample <= 0.0 {
        let sky_lum = max(scatter.in_scatter.r,
                          max(scatter.in_scatter.g, scatter.in_scatter.b));
        let lum_opacity = smoothstep(0.03, 0.20, sky_lum);
        opacity = max(opacity, lum_opacity);
    }
    return vec4(scatter.in_scatter, opacity);
}
