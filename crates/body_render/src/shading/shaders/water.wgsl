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
fn water_wave_normal(world_pos: vec3<f32>, geo_n: vec3<f32>, t: f32, view_dist_m: f32) -> vec3<f32> {
    let near_m = 80.0;
    let far_m = 6000.0;
    let fade = clamp(1.0 - (view_dist_m - near_m) / (far_m - near_m), 0.0, 1.0);
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
// between the sea surface and the seabed (from the scene-depth buffer); near
// zero at the waterline, large in open ocean. Shallows lighten toward cyan,
// deep water saturates to the per-body deep tint.
fn water_subsurface(color_depth: vec4<f32>, column_m: f32) -> vec3<f32> {
    let deep = max(color_depth.xyz, vec3<f32>(0.0));
    let shallow = vec3<f32>(0.10, 0.20, 0.22);
    let depth_t = 1.0 - exp(-max(column_m, 0.0) / 14.0);
    return mix(shallow, deep, depth_t);
}

// Shade the ocean surface at a ray-traced hit point.
//
// `sun_flux` is expected pre-scaled (× SCENE_FLUX_SCALE) by the caller so the
// glint matches the rest of the scene's flux calibration. The reflected sky and
// aerial perspective are added by the atmosphere integral the sky pass already
// runs over the camera→water-surface segment, so here the sky term is only the
// base Fresnel-blue ambient.
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
) -> vec3<f32> {
    let n = water_wave_normal(hit_ws, geo_n, time, view_dist);
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
    return lit;
}
