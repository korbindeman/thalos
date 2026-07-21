// Shared analytic-ocean shading.
//
// The planet ocean is rendered as a RAY-TRACED MATH SPHERE inside the
// `body_sky.wgsl` fullscreen pass — not a tessellated mesh. A meshed icosphere
// at planet scale sags tens of metres below the true sphere (flat triangle
// chords), so the seabed punches through everywhere but the vertices; a math
// sphere is perfectly smooth from orbit to sea level.
//
// This library supplies the surface shading the sky pass calls once it has the
// ocean hit point: a filtered body-fixed slope field, a Cook-Torrance GGX
// water BRDF (sun glint + Fresnel), and a depth-graded subsurface colour driven
// by the water-column thickness read from the scene-depth buffer (shallows
// cyan, deep blue). Calibrated to `planet_impostor.wgsl::shade_water` so the
// ground↔impostor LOD handoff does not pop.

#define_import_path thalos::water

#import thalos::lighting::{
    SCENE_FLUX_SCALE,
    compute_surface_sky,
    env_brdf_approx,
    sky_ambient_irradiance,
}

const PI_WATER: f32 = 3.14159265358979;

// ── Filtered slope-field input ────────────────────────────────────────────
//
// The detailed body-sky path supplies resolved slopes from its shared
// mipmapped broadband texture. Keeping the BRDF downstream of that seam is
// deliberate: the eventual FFT simulation can replace the producer without a
// second water material or a lighting rewrite. Far/map callers pass a neutral
// slope plus the statistical roughness of the unresolved spectrum.

struct OceanWaveSample {
    normal_body: vec3<f32>,
    alpha_ggx: f32,
    whitecap: f32,
    breakup: f32,
}

fn rotate_by_quat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    return 2.0 * dot(u, v) * u
        + (q.w * q.w - dot(u, u)) * v
        + 2.0 * q.w * cross(u, v);
}

// ── Water BRDF (Cook-Torrance, GGX + Smith-Schlick + Schlick Fresnel).
// Mirror of `planet_impostor.wgsl::water_brdf`. ──────────────────────────────

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
    let d_ggx = a2 / (PI_WATER * d_denom * d_denom);

    let k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
    let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    let g_smith = g_v * g_l;

    let f_h = f0 + (1.0 - f0) * pow(max(1.0 - v_dot_h, 0.0), 5.0);

    let specular = (d_ggx * g_smith * f_h) / max(4.0 * n_dot_v, 1e-4);
    let diffuse = (1.0 - f_nv) * subsurface * n_dot_l / PI_WATER;
    return diffuse + vec3<f32>(specular);
}

// Depth-graded subsurface colour. `column_m` is the camera-ray thickness
// between the sea surface and the seabed (scene depth near, baked bathymetry
// far — ADR-20260720T185957Z-coastline-as-authored-data); near zero at the waterline, large in open ocean. Shallows
// lighten toward cyan, deep water saturates to the per-body deep tint.
//
// The 8 m e-folding keeps the pale-shallow band a *fringe*: only the first
// couple of tens of metres of depth read shallow. The previous 14 m ramp
// painted every < ~30 m shelf pale, which turned whole archipelago seas into
// one vast translucent bank instead of water with bright island fringes.
fn water_subsurface(color_depth: vec4<f32>, column_m: f32) -> vec3<f32> {
    let deep = max(color_depth.xyz, vec3<f32>(0.0));
    // Sand-floored shallow water is a warm saturated turquoise (BL-10) — the
    // old grey-green (0.10, 0.20, 0.22) blended into the coastal grass and
    // read as marsh instead of beach water.
    let shallow = vec3<f32>(0.055, 0.215, 0.255);
    let depth_t = 1.0 - exp(-max(column_m, 0.0) / 8.0);
    return mix(shallow, deep, depth_t);
}

// Shore-interaction ranges (BL-10, tier 1 — MSFS-class: normals + albedo only,
// no displaced geometry, ADR-20260720T185954Z-analytic-planet-water-never-meshed intact). All shore effects are keyed on the
// signed sea field's depth / shore distance the sky pass supplies, so they are
// as LOD-stable as the coastline itself.
const SHORE_FX_VIEW_LO_M: f32 = 3500.0;  // shore effects full inside this…
const SHORE_FX_VIEW_HI_M: f32 = 9000.0;  // …gone past this (orbital ocean unchanged)
const SHORE_FX_DIST_M: f32 = 4000.0;     // max distance-to-shore that gets shore FX
const SHOAL_DEPTH_M: f32 = 30.0;         // chop calms from here up to the waterline
const BREAKER_WAVELENGTH_M: f32 = 70.0;  // crest-to-crest spacing of shore waves
const BREAKER_SPEED_M_S: f32 = 5.0;      // shoreward crest speed

// Shade the ocean surface at a ray-traced hit point.
//
// `sun_flux` is expected pre-scaled (× SCENE_FLUX_SCALE) by the caller so the
// glint matches the rest of the scene's flux calibration. The reflected sky and
// aerial perspective are added by the atmosphere integral the sky pass already
// runs over the camera→water-surface segment, so here the sky term is only the
// base Fresnel-blue ambient.
//
// Shore inputs (pass far sentinels — depth 1e6, shore_dist 1e9, dir anything —
// to disable, e.g. from the map/impostor paths):
//   depth_m      — vertical water depth at the hit (from the signed sea field)
//   shore_dist_m — horizontal distance to the waterline
//   shore_dir    — unit tangent pointing toward the shore (uphill on the field)
//   footprint_m  — surface metres one screen pixel spans at the hit (the sky
//                  pass's analytic footprint). Band-limits every procedural
//                  shore pattern: a stripe or noise whose wavelength
//                  approaches the footprint fades to its mean instead of
//                  aliasing into pixel dither at grazing angles. (Analytic on
//                  purpose — naga rejects fwidth() in non-uniform control
//                  flow, and this whole function runs inside a water branch.)
//
// The shore-wave phase is a function of SHORE DISTANCE, so crest lines are
// parallel to the beach by construction — refraction for free — and march
// shoreward with time; they steepen (normal ridges) and break into foam in a
// narrow surf window (~0.2–3 m depth — a couple of lines, not a ruled field),
// and a thin churned swash edge rides the last half metre.
fn shade_ocean_detailed(
    geo_n_body: vec3<f32>,
    body_to_world: vec4<f32>,
    geo_n: vec3<f32>,
    view_dir: vec3<f32>,
    view_dist: f32,
    time: f32,
    sun_dir: vec3<f32>,
    sun_flux: f32,
    color_depth: vec4<f32>,
    column_m: f32,
    depth_m: f32,
    shore_dist_m: f32,
    shore_dir: vec3<f32>,
    footprint_m: f32,
    wave_slope: vec2<f32>,
    wave_alpha_ggx: f32,
    wave_breakup: f32,
    foam_slope_onset: f32,
    wind_basis: vec3<f32>,
    crosswind_basis: vec3<f32>,
    sky_tau_zenith: vec3<f32>,
    atmosphere_strength: f32,
) -> vec3<f32> {
    // Shore effects fade with view distance (they are a near-field treatment;
    // at range the GGX slope lobe + subsurface colour carry the ocean).
    let shore_view_t = 1.0 - smoothstep(SHORE_FX_VIEW_LO_M, SHORE_FX_VIEW_HI_M, view_dist);

    // Shoaling: open-sea chop calms as the bottom rises, so inshore water goes
    // glassier and the breaker lines read against it.
    let shoal = (1.0 - smoothstep(1.0, SHOAL_DEPTH_M, depth_m)) * shore_view_t;
    let resolved_slope = wave_slope * (1.0 - 0.7 * shoal);
    let gradient_body =
        wind_basis * resolved_slope.x + crosswind_basis * resolved_slope.y;
    var waves: OceanWaveSample;
    waves.normal_body = rotate_by_quat(
        body_to_world,
        normalize(geo_n_body - gradient_body),
    );
    waves.alpha_ggx = mix(wave_alpha_ggx, 0.045, shoal);
    waves.breakup = wave_breakup;
    waves.whitecap = smoothstep(
        foam_slope_onset,
        foam_slope_onset + 0.12,
        length(resolved_slope),
    )
        * smoothstep(0.62, 0.90, wave_breakup);
    var n = waves.normal_body;

    // Open-water whitecaps are sparse compression events from the same wave
    // phases that shape the normal, so bright streaks sit on crests rather than
    // crawling as independent noise. The production persistent/advection field
    // will replace this source term without changing the foam shading below.
    var foam = waves.whitecap * 0.55;
    if (shore_dist_m < SHORE_FX_DIST_M && shore_view_t > 0.003) {
        // Resolution guards: fade each pattern out as one screen pixel grows
        // toward its feature size (its mean is ~0, so fading to zero is the
        // band-limited result, not a brightness shift).
        let stripe_res = 1.0 - smoothstep(0.08, 0.30, footprint_m / BREAKER_WAVELENGTH_M);
        let noise_res = 1.0 - smoothstep(4.0, 14.0, footprint_m);

        // Crest phase keyed on shore distance: `fract` gives periodic wave
        // fronts, and `+ time·v` marches them down the gradient (shoreward).
        let phase = (shore_dist_m + time * BREAKER_SPEED_M_S) / BREAKER_WAVELENGTH_M;
        let crest = fract(phase);
        // Narrow surf window: waves break over roughly the last couple of
        // crest spacings of a natural foreshore, not across the whole shelf
        // (a 0.25–7 m window on a ~1 % beach was ~700 m of ruled stripes).
        let breaker_zone = smoothstep(3.2, 1.9, depth_m) * smoothstep(0.18, 0.45, depth_m);
        // A narrow foam stripe rides just behind each crest, broken up
        // along-shore (~30 m clumps) so the lines are ragged, not ruled.
        let stripe = exp(-pow((crest - 0.2) * 5.5, 2.0));
        let foam_breaker =
            breaker_zone * stripe * waves.breakup * stripe_res;
        // Swash foam: a thin always-on bright line at the water's edge (the
        // last ~15 cm) plus intermittent noisy patches in the last half
        // metre — not the solid white strip a high constant base painted.
        let foam_edge = (1.0 - smoothstep(0.02, 0.15, depth_m)) * 0.85
            + (1.0 - smoothstep(0.05, 0.6, depth_m))
                * (0.10 + 0.60 * waves.breakup);
        foam = max(
            foam,
            clamp(foam_breaker + foam_edge * noise_res, 0.0, 1.0) * shore_view_t,
        );
        // Swell ridges: tilt the normal along the shore direction on the crest
        // profile so the incoming wave fronts catch light before they break.
        // Same resolution guard, or the ridge speculars dither at range.
        let ridge = sin(phase * 6.2831853);
        n = normalize(
            n + shore_dir * (ridge * 0.14 * breaker_zone * shore_view_t * stripe_res));
    }

    let n_dot_v = max(dot(n, view_dir), 0.0);
    let f0 = 0.02;
    let f_nv = f0 + (1.0 - f0) * pow(max(1.0 - n_dot_v, 0.0), 5.0);
    let alpha_ggx = waves.alpha_ggx;

    let subsurface = water_subsurface(color_depth, column_m);

    // One-world lighting: derive water's direct beam and hemispheric sky from
    // the same atmosphere/flux function as terrain and foliage. The incoming
    // `sun_flux` is already SCENE_FLUX_SCALE'd for historical callers; undo it
    // before the shared helper applies that scale once.
    let sky = compute_surface_sky(
        sky_tau_zenith,
        atmosphere_strength,
        geo_n,
        sun_dir,
        sun_flux / max(SCENE_FLUX_SCALE, 1.0e-4),
    );
    let sun_brdf = water_brdf(n, view_dir, sun_dir, n_dot_v, f_nv, alpha_ggx, f0, subsurface);
    let direct = sun_brdf * sky.sun_color * sky.sun_scale;

    let ambient_irradiance = sky_ambient_irradiance(sky, n, geo_n);
    let subsurface_ambient =
        (1.0 - f_nv) * subsurface * ambient_irradiance * 0.54;
    let reflected = reflect(-view_dir, n);
    let reflected_sky = mix(
        sky.ground_radiance,
        sky.sky_radiance,
        smoothstep(-0.12, 0.20, dot(reflected, geo_n)),
    );
    let dfg = env_brdf_approx(alpha_ggx, n_dot_v);
    // Keep the clear-sky hemisphere subordinate to the direct sun road. The
    // atmosphere helper is an irradiance approximation rather than a
    // prefiltered radiance probe, so an energy calibration is still required
    // until F7/F9 supply the shared prefiltered environment.
    let environment_specular = reflected_sky * (dfg.x * f0 + dfg.y) * 0.60;
    var lit = direct + subsurface_ambient + environment_specular;

    // Foam: bright rough diffuse replacing the water response (it also kills
    // the glint — foam is matte). Lit by the same sun + ambient calibration.
    if (foam > 0.001) {
        let foam_albedo = vec3<f32>(0.68, 0.70, 0.70);
        let foam_lit = foam_albedo
            * (max(dot(geo_n, sun_dir), 0.0) * sky.sun_color * sky.sun_scale / PI_WATER
                + ambient_irradiance * 0.90);
        lit = mix(lit, foam_lit, clamp(foam, 0.0, 0.85));
    }
    return lit;
}

// Stable far/map compatibility entry point. Those paths have no resolved slope
// texture, so they pass a neutral normal and the calibrated statistical
// roughness of the unresolved sea. Keeping this wrapper avoids a second
// distant-ocean BRDF while the detailed body-sky path carries the filtered
// body-fixed field and authored atmosphere inputs.
fn shade_ocean(
    hit_ws: vec3<f32>,
    geo_n: vec3<f32>,
    view_dir: vec3<f32>,
    view_dist: f32,
    time: f32,
    sun_dir: vec3<f32>,
    sun_flux: f32,
    color_depth: vec4<f32>,
    column_m: f32,
    depth_m: f32,
    shore_dist_m: f32,
    shore_dir: vec3<f32>,
    footprint_m: f32,
) -> vec3<f32> {
    return shade_ocean_detailed(
        geo_n,
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
        geo_n,
        view_dir,
        view_dist,
        time,
        sun_dir,
        sun_flux,
        color_depth,
        column_m,
        depth_m,
        shore_dist_m,
        shore_dir,
        footprint_m,
        vec2<f32>(0.0),
        0.14,
        0.5,
        0.22,
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.046, 0.108, 0.264),
        1.0,
    );
}
