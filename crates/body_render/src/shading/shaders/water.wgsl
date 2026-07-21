// Shared analytic-ocean shading.
//
// The planet ocean is rendered as a RAY-TRACED MATH SPHERE inside the
// `body_sky.wgsl` fullscreen pass — not a tessellated mesh. A meshed icosphere
// at planet scale sags tens of metres below the true sphere (flat triangle
// chords), so the seabed punches through everywhere but the vertices; a math
// sphere is perfectly smooth from orbit to sea level.
//
// This library supplies the surface shading the sky pass calls once it has the
// ocean hit point: two-octave scrolled wave normals, a Cook-Torrance GGX water
// BRDF (sun glint + Fresnel), and a depth-graded subsurface colour driven by
// the water-column thickness read from the scene-depth buffer (shallows cyan,
// deep blue). Calibrated to `planet_impostor.wgsl::shade_water` so the
// ground↔impostor LOD handoff does not pop.

#define_import_path thalos::water

const PI_WATER: f32 = 3.14159265358979;

// ── 3-octave value noise for wave normals. Inline (not the impostor's fbm,
// whose octave counts/frequencies are tuned for cubemap evaluation). ─────────

fn h2o_hash3(p: vec3<f32>) -> f32 {
    let q = sin(vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6)),
    ));
    return fract(sin(dot(q, vec3<f32>(43.81, 17.23, 95.71))) * 43758.5453);
}

fn h2o_vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = h2o_hash3(i);
    let c100 = h2o_hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = h2o_hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = h2o_hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = h2o_hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = h2o_hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = h2o_hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = h2o_hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z);
}

fn h2o_vfbm3(p: vec3<f32>) -> f32 {
    var sum: f32 = 0.0;
    var amp: f32 = 0.5;
    var pos = p;
    for (var i: i32 = 0; i < 3; i = i + 1) {
        sum = sum + amp * h2o_vnoise(pos);
        pos = pos * 2.04;
        amp = amp * 0.5;
    }
    return sum;
}

// Stable tangent frame from the geometric sphere normal. Column 0 = tangent,
// column 1 = bitangent.
fn h2o_tangent_frame(n: vec3<f32>) -> mat2x3<f32> {
    let ref_up = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(n.y) > 0.99,
    );
    let t = normalize(cross(ref_up, n));
    let b = cross(n, t);
    return mat2x3<f32>(t, b);
}

// Two-octave scrolled noise normal perturbation. `view_dist_m` is the
// camera-to-fragment distance in metres; the perturbation fades to zero past
// ~6 km so distant/orbital water reads as a smooth sphere modulated only by the
// GGX statistical-slope lobe — the same model the impostor uses past the swap.
// `amp_scale` scales the whole perturbation: the shore-shoaling term passes
// < 1 so chop calms as the water shallows (BL-10).
fn water_wave_normal(
    world_pos: vec3<f32>,
    geo_n: vec3<f32>,
    t: f32,
    view_dist_m: f32,
    amp_scale: f32,
) -> vec3<f32> {
    let near_m = 80.0;
    let far_m = 6000.0;
    let fade =
        clamp(1.0 - (view_dist_m - near_m) / (far_m - near_m), 0.0, 1.0) * amp_scale;
    if fade <= 0.005 {
        return geo_n;
    }

    let tf = h2o_tangent_frame(geo_n);
    let tan_v = tf[0];
    let bit_v = tf[1];

    // Wavelengths λ_lo ≈ 67 m, λ_hi ≈ 14 m. Surface scroll velocity in m/s,
    // applied along the tangent axes so waves travel across the sphere rather
    // than translating in world space (which slips at the frame poles).
    let f_lo = 0.015;
    let f_hi = 0.07;
    let v_lo = 0.45;
    let v_hi = 1.10;

    let scroll_lo = tan_v * (t * v_lo) + bit_v * (t * v_lo * 0.5);
    let scroll_hi = tan_v * (-t * v_hi * 0.6) + bit_v * (t * v_hi);

    let p_lo = world_pos * f_lo + scroll_lo;
    let p_hi = world_pos * f_hi + scroll_hi;

    let eps = 0.5;
    let h0_lo = h2o_vfbm3(p_lo);
    let h0_hi = h2o_vfbm3(p_hi);
    let dt_lo = h2o_vfbm3(p_lo + tan_v * eps) - h0_lo;
    let db_lo = h2o_vfbm3(p_lo + bit_v * eps) - h0_lo;
    let dt_hi = h2o_vfbm3(p_hi + tan_v * eps) - h0_hi;
    let db_hi = h2o_vfbm3(p_hi + bit_v * eps) - h0_hi;

    let amp_lo = 1.4 * fade;
    let amp_hi = 0.7 * fade;
    let slope_t = amp_lo * dt_lo + amp_hi * dt_hi;
    let slope_b = amp_lo * db_lo + amp_hi * db_hi;

    return normalize(geo_n - tan_v * slope_t - bit_v * slope_b);
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
    // Shore effects fade with view distance (they are a near-field treatment;
    // at range the GGX slope lobe + subsurface colour carry the ocean).
    let shore_view_t = 1.0 - smoothstep(SHORE_FX_VIEW_LO_M, SHORE_FX_VIEW_HI_M, view_dist);

    // Shoaling: open-sea chop calms as the bottom rises, so inshore water goes
    // glassier and the breaker lines read against it.
    let shoal = (1.0 - smoothstep(1.0, SHOAL_DEPTH_M, depth_m)) * shore_view_t;
    var n = water_wave_normal(hit_ws, geo_n, time, view_dist, 1.0 - 0.7 * shoal);

    var foam = 0.0;
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
        let breakup = h2o_vfbm3(hit_ws * 0.033 + vec3<f32>(0.0, time * 0.05, 0.0));
        let foam_breaker =
            breaker_zone * stripe * smoothstep(0.35, 0.60, breakup) * stripe_res;
        // Swash foam: a thin always-on bright line at the water's edge (the
        // last ~15 cm) plus intermittent noisy patches in the last half
        // metre — not the solid white strip a high constant base painted.
        let swash_n = h2o_vfbm3(hit_ws * 0.09 + vec3<f32>(time * -0.10, 0.0, 0.0));
        let foam_edge = (1.0 - smoothstep(0.02, 0.15, depth_m)) * 0.85
            + (1.0 - smoothstep(0.05, 0.6, depth_m))
                * (0.10 + 0.60 * smoothstep(0.48, 0.72, swash_n));
        foam = clamp(foam_breaker + foam_edge * noise_res, 0.0, 1.0) * shore_view_t;
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
    // GGX α = 0.10 — Cox-Munk wind-driven slope σ ≈ 6–8°.
    let alpha_ggx = 0.10;
    // Same BRDF scale as the Hapke land path so flux calibration matches.
    let brdf_scale = 0.5;

    let subsurface = water_subsurface(color_depth, column_m);

    // Base reflected-sky blue at grazing angles. The sky pass paints physically
    // correct in-scatter on top (aerial perspective), so this is just the dim
    // ambient reflection, not the full sky radiance.
    let sky_tint = vec3<f32>(0.35, 0.55, 0.95);
    // Fade the ambient sky-fill across the terminator (sun elevation vs the
    // GEOMETRIC normal, so waves don't flicker it). Without this the flat 0.15
    // floor lights the whole planet on the distant impostor — the night-side
    // ocean glows. The sun BRDF already vanishes at night via its own n·l;
    // moonlight, when present, is a separate additive term (not here).
    let day = smoothstep(-0.15, 0.15, dot(geo_n, sun_dir));
    let ambient = 0.15 * day;

    var lit = (f_nv * sky_tint + (1.0 - f_nv) * subsurface) * ambient;

    let sun_brdf = water_brdf(n, view_dir, sun_dir, n_dot_v, f_nv, alpha_ggx, f0, subsurface);
    lit = lit + sun_brdf * sun_flux * brdf_scale;

    // Foam: bright rough diffuse replacing the water response (it also kills
    // the glint — foam is matte). Lit by the same sun + ambient calibration.
    if (foam > 0.001) {
        let foam_albedo = vec3<f32>(0.60, 0.62, 0.63);
        let foam_lit = foam_albedo
            * (max(dot(geo_n, sun_dir), 0.0) * sun_flux * brdf_scale / PI_WATER
                + ambient * 2.5);
        lit = mix(lit, foam_lit, foam);
    }
    return lit;
}
