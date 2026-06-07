// Ground-LOD water shader.
//
// Opaque icosphere at `body_radius + sea_level_m + ε`. Cook-Torrance GGX
// (α = 0.10, F0 = 0.02) supplies sun glint and Fresnel-modulated sky/
// subsurface blend; subsurface colour comes from the per-body
// `water_color_depth` (linear-RGB tint × min-optical-depth absorption).
// Wave normals are two-octave scrolled value noise in the tangent plane
// of the geometric sphere normal, fading to flat at view distances where
// they would alias — distant water reads as a statistical-slope GGX
// surface, the same model the impostor uses past the LOD swap.
//
// Calibration matches `planet_impostor.wgsl::shade_water` so the
// ground-LOD ↔ impostor handoff at 4 × radius does not pop.

#import bevy_pbr::{
    mesh_view_bindings::view,
    forward_io::VertexOutput,
}
#import thalos::lighting::{
    SceneLighting,
    SCENE_FLUX_SCALE,
    eclipse_factor,
    planetshine_sample,
}

struct BodyWaterParams {
    color_depth: vec4<f32>,
    planet_center_radius: vec4<f32>,
    time: vec4<f32>,
}

@group(3) @binding(0) var<uniform> water_scene: SceneLighting;
@group(3) @binding(1) var<uniform> water_params: BodyWaterParams;

const PI_W: f32 = 3.14159265358979;

// ── 3-octave value noise. Kept inline so the water material doesn't drag
// in the impostor's `fbm3` (those octave counts and frequencies are tuned
// for cubemap evaluation, not wave normals). ──────────────────────────────

fn hash3(p: vec3<f32>) -> f32 {
    let q = sin(vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6)),
    ));
    return fract(sin(dot(q, vec3<f32>(43.81, 17.23, 95.71))) * 43758.5453);
}

fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = hash3(i);
    let c100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z);
}

fn vfbm3(p: vec3<f32>) -> f32 {
    var sum: f32 = 0.0;
    var amp: f32 = 0.5;
    var pos = p;
    for (var i: i32 = 0; i < 3; i = i + 1) {
        sum = sum + amp * vnoise(pos);
        pos = pos * 2.04;
        amp = amp * 0.5;
    }
    return sum;
}

// Stable tangent frame from the geometric sphere normal.
// Returns mat2x3 with column 0 = tangent, column 1 = bitangent.
fn tangent_frame(n: vec3<f32>) -> mat2x3<f32> {
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
// camera-to-fragment distance in metres; the perturbation fades to zero
// past ~6 km so distant water reads as a smooth sphere modulated by the
// GGX α statistical-slope lobe.
fn wave_normal(world_pos: vec3<f32>, geo_n: vec3<f32>, t: f32, view_dist_m: f32) -> vec3<f32> {
    let near_m = 80.0;
    let far_m = 6000.0;
    let fade = clamp(1.0 - (view_dist_m - near_m) / (far_m - near_m), 0.0, 1.0);
    if fade <= 0.005 {
        return geo_n;
    }

    let tf = tangent_frame(geo_n);
    let tan_v = tf[0];
    let bit_v = tf[1];

    // Wavelengths λ_lo ≈ 1/f_lo = 67 m, λ_hi ≈ 14 m. Surface scroll velocity
    // in m/s, applied along the tangent axes so waves travel "across" the
    // sphere rather than translating in world space (which would look like
    // a slipping skin at the poles of the reference frame).
    let f_lo = 0.015;
    let f_hi = 0.07;
    let v_lo = 0.45;
    let v_hi = 1.10;

    let scroll_lo = tan_v * (t * v_lo) + bit_v * (t * v_lo * 0.5);
    let scroll_hi = tan_v * (-t * v_hi * 0.6) + bit_v * (t * v_hi);

    let p_lo = world_pos * f_lo + scroll_lo;
    let p_hi = world_pos * f_hi + scroll_hi;

    // Finite-difference gradient on each octave. `eps` is in noise-domain
    // units (already multiplied by frequency), so the same scalar covers
    // both scales.
    let eps = 0.5;
    let h0_lo = vfbm3(p_lo);
    let h0_hi = vfbm3(p_hi);
    let dt_lo = vfbm3(p_lo + tan_v * eps) - h0_lo;
    let db_lo = vfbm3(p_lo + bit_v * eps) - h0_lo;
    let dt_hi = vfbm3(p_hi + tan_v * eps) - h0_hi;
    let db_hi = vfbm3(p_hi + bit_v * eps) - h0_hi;

    let amp_lo = 1.4 * fade;
    let amp_hi = 0.7 * fade;
    let slope_t = amp_lo * dt_lo + amp_hi * dt_hi;
    let slope_b = amp_lo * db_lo + amp_hi * db_hi;

    return normalize(geo_n - tan_v * slope_t - bit_v * slope_b);
}

// ── Water BRDF (Cook-Torrance, GGX + Smith-Schlick + Schlick Fresnel).
// Mirror of `planet_impostor.wgsl::water_brdf`; do not diverge or the LOD
// swap will pop. ─────────────────────────────────────────────────────────

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
    let d_ggx = a2 / (PI_W * d_denom * d_denom);

    let k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
    let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    let g_smith = g_v * g_l;

    let f_h = f0 + (1.0 - f0) * pow(max(1.0 - v_dot_h, 0.0), 5.0);

    let specular = (d_ggx * g_smith * f_h) / max(4.0 * n_dot_v, 1e-4);
    let diffuse = (1.0 - f_nv) * subsurface * n_dot_l / PI_W;
    return diffuse + vec3<f32>(specular);
}

// Apparent deep-water colour from the per-body `water_color_depth`.
// Uses the per-body minimum optical depth as a fixed reference column
// thickness because the opaque ground-LOD water path does not sample
// scene depth (see module header for the rationale).
fn water_column_color(n_dot_v: f32) -> vec3<f32> {
    let base = max(water_params.color_depth.xyz, vec3<f32>(0.0));
    let min_depth_m = max(water_params.color_depth.w, 1.0);
    let path_m = min_depth_m / max(n_dot_v, 0.18);
    let absorption = exp(-vec3<f32>(0.018, 0.010, 0.004) * path_m);
    let scatter_t = 1.0 - exp(-path_m / 180.0);
    let deep_scatter = vec3<f32>(0.002, 0.018, 0.060) * scatter_t;
    let apparent = base * absorption + deep_scatter;
    return clamp(apparent, vec3<f32>(0.0), vec3<f32>(0.08, 0.14, 0.20));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let planet_center = water_params.planet_center_radius.xyz;
    let hit_ws = in.world_position.xyz;
    let geo_n = normalize(hit_ws - planet_center);
    let view_dir = normalize(view.world_position - hit_ws);
    let view_dist = distance(view.world_position, hit_ws);

    let t = water_params.time.w;
    let n = wave_normal(hit_ws, geo_n, t, view_dist);

    let n_dot_v = max(dot(n, view_dir), 0.0);
    let f0 = 0.02;
    let f_nv = f0 + (1.0 - f0) * pow(max(1.0 - n_dot_v, 0.0), 5.0);
    // GGX α = 0.10 — Cox-Munk wind-driven slope σ ≈ 6–8°.
    let alpha_ggx = 0.10;
    // Same BRDF scale as the Hapke land path so flux calibration matches.
    let brdf_scale = 0.5;

    let subsurface = water_column_color(n_dot_v);

    let primary = water_scene.stars[0];
    let sun_dir = primary.dir_flux.xyz;
    let sun_flux = primary.dir_flux.w;

    // Sky tint approximates Rayleigh-blue ambient. Drives the limb colour
    // where Fresnel approaches 1. The atmosphere fullscreen pass paints
    // physically correct in-scatter on top, so this is the dim-side blue
    // visible at high sun elevation (vertical view) rather than the full
    // sky radiance.
    let sky_tint = vec3<f32>(0.35, 0.55, 0.95);
    let ambient = water_scene.ambient_intensity;

    var lit = (f_nv * sky_tint + (1.0 - f_nv) * subsurface) * ambient;

    let sun_brdf = water_brdf(n, view_dir, sun_dir, n_dot_v, f_nv, alpha_ggx, f0, subsurface);
    let sun_shadow = eclipse_factor(water_scene, hit_ws, sun_dir);
    lit = lit + sun_brdf * sun_flux * SCENE_FLUX_SCALE * brdf_scale * sun_shadow;

    let shine = planetshine_sample(water_scene, hit_ws, sun_dir, sun_flux);
    if shine.enabled {
        let shine_brdf = water_brdf(n, view_dir, shine.dir, n_dot_v, f_nv, alpha_ggx, f0, subsurface);
        lit = lit + shine_brdf * shine.tint * shine.flux * brdf_scale;
    }

    return vec4<f32>(lit, 1.0);
}
