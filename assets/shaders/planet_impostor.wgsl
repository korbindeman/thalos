// Planet sphere impostor.
//
// Renders a camera-facing quad (billboard) whose fragment shader ray-traces a
// sphere of the correct radius.  Every pixel gets the mathematically exact
// surface normal, giving a perfectly smooth silhouette at any resolution.
//
// Surface detail comes from two sources, layered:
//
//   1. Cubemap textures from `thalos_terrain_gen` — albedo (sRGB RGBA8) and
//      height (R16Unorm displacement).  These hold the low-frequency baked
//      features: primordial topography, basins, mare flooding, etc.
//
//   2. Per-fragment small-crater synthesis (the "detail hook").  Below the
//      bake/shader split diameter, craters are procedurally generated per
//      fragment via a 3D-hashed cell grid on the unit sphere.
//
// Lighting: diffuse Lambertian + tiny ambient + terminator wrap.

#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_functions::get_world_from_local

const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

// ── Material uniforms (binding group 3) ─────────────────────────────────────

struct PlanetParams {
    radius:            f32,
    rotation_phase:    f32,
    light_intensity:   f32,
    ambient_intensity: f32,
    light_dir: vec4<f32>,
}

struct PlanetDetail {
    body_radius_m:    f32,
    d_min_m:          f32,
    d_max_m:          f32,
    sfd_alpha:        f32,
    global_k_per_km2: f32,
    d_sc_m:           f32,
    body_age_gyr:     f32,
    _pad:             f32,
    seed_lo:          u32,
    seed_hi:          u32,
}

const FRESH_AGE_GYR: f32 = 0.1;

@group(3) @binding(0) var<uniform> params:         PlanetParams;
@group(3) @binding(1) var          albedo_tex:     texture_cube<f32>;
@group(3) @binding(2) var          albedo_sampler: sampler;
@group(3) @binding(3) var          height_tex:     texture_cube<f32>;
@group(3) @binding(4) var          height_sampler: sampler;
@group(3) @binding(5) var<uniform> detail:         PlanetDetail;

// ── Vertex stage ─────────────────────────────────────────────────────────────

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) sphere_center: vec3<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let model        = get_world_from_local(in.instance_index);
    let sphere_center = (model * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    let cam_pos  = view.world_position;
    let to_cam   = normalize(cam_pos - sphere_center);

    let ref_up = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(to_cam.y) > 0.99);
    let right  = normalize(cross(ref_up, to_cam));
    let up     = normalize(cross(to_cam, right));

    let d      = length(cam_pos - sphere_center);
    let d_safe = max(d, params.radius * 1.0001);
    let billboard_radius = params.radius * d_safe
        / sqrt(d_safe * d_safe - params.radius * params.radius);

    let world_pos = sphere_center
        + in.position.x * right * billboard_radius
        + in.position.y * up   * billboard_radius;

    var out: VertexOutput;
    out.clip_position  = view.clip_from_world * vec4(world_pos, 1.0);
    out.world_position = world_pos;
    out.sphere_center  = sphere_center;
    return out;
}

// ── Fragment stage ────────────────────────────────────────────────────────────

struct FragOutput {
    @location(0)          color: vec4<f32>,
    @builtin(frag_depth)  depth: f32,
}

// ── Hash primitives ─────────────────────────────────────────────────────────

fn pcg(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_cell(ix: i32, iy: i32, iz: i32, octave: u32) -> u32 {
    let ux = bitcast<u32>(ix);
    let uy = bitcast<u32>(iy);
    let uz = bitcast<u32>(iz);
    var h = ux * 73856093u;
    h = h ^ (uy * 19349663u);
    h = h ^ (uz * 83492791u);
    h = pcg(h);
    h = h ^ (octave * 2654435769u);
    h = h ^ detail.seed_lo;
    h = pcg(h);
    h = h ^ (detail.seed_hi * 1540483477u);
    return pcg(h);
}

fn u32_to_unit(x: u32) -> f32 {
    return f32(x) / 4294967296.0;
}

// ── Crater profile (must agree with planet_gen/src/crater.rs) ───────────────

const SIMPLE_DEPTH_RATIO: f32       = 0.2;
const SIMPLE_RIM_RATIO: f32         = 0.04;
const SIMPLE_INTERIOR_EXPONENT: f32 = 2.5;
const EJECTA_EXTENT: f32            = 2.5;
const RIM_FRESHNESS_SIGMA: f32      = 0.22;
const COMPLEX_FLOOR_FRACTION: f32   = 0.55;
const COMPLEX_PEAK_HEIGHT_FRAC: f32 = 0.15;
const COMPLEX_PEAK_BASE_FRAC: f32   = 0.15;
const COMPLEX_MIN_DEPTH_RATIO: f32  = 0.05;

fn complex_depth_ratio(d_over_dsc: f32) -> f32 {
    let t = exp(-max(d_over_dsc - 1.0, 0.0) / 3.0);
    return COMPLEX_MIN_DEPTH_RATIO + (SIMPLE_DEPTH_RATIO - COMPLEX_MIN_DEPTH_RATIO) * t;
}

fn simple_profile(r: f32, depth: f32, rim: f32) -> vec2<f32> {
    if r <= 1.0 {
        let n = SIMPLE_INTERIOR_EXPONENT;
        let h = -depth + (depth + rim) * pow(r, n);
        let dh = (depth + rim) * n * pow(r, n - 1.0);
        return vec2(h, dh);
    } else {
        let span = EJECTA_EXTENT - 1.0;
        let t = clamp((r - 1.0) / span, 0.0, 1.0);
        let s_taper = t * t * (3.0 - 2.0 * t);
        let fade = 1.0 - s_taper;
        let dfade_dr = -6.0 * t * (1.0 - t) / span;

        let inv3 = 1.0 / (r * r * r);
        let base = rim * inv3;
        let dbase_dr = -3.0 * rim / (r * r * r * r);

        let h = base * fade;
        let dh = dbase_dr * fade + base * dfade_dr;
        return vec2(h, dh);
    }
}

fn complex_profile(r: f32, depth: f32, rim: f32) -> vec2<f32> {
    var base_h: f32;
    var base_dh: f32;
    if r <= 1.0 {
        if r <= COMPLEX_FLOOR_FRACTION {
            base_h = -depth;
            base_dh = 0.0;
        } else {
            let span = 1.0 - COMPLEX_FLOOR_FRACTION;
            let t = (r - COMPLEX_FLOOR_FRACTION) / span;
            let s = t * t * (3.0 - 2.0 * t);
            let ds_dr = 6.0 * t * (1.0 - t) / span;
            let h_total = depth + rim;
            base_h = -depth + h_total * s;
            base_dh = h_total * ds_dr;
        }
    } else {
        let span = EJECTA_EXTENT - 1.0;
        let t = clamp((r - 1.0) / span, 0.0, 1.0);
        let s_taper = t * t * (3.0 - 2.0 * t);
        let fade = 1.0 - s_taper;
        let dfade_dr = -6.0 * t * (1.0 - t) / span;

        let inv3 = 1.0 / (r * r * r);
        let raw = rim * inv3;
        let draw_dr = -3.0 * rim / (r * r * r * r);
        base_h = raw * fade;
        base_dh = draw_dr * fade + raw * dfade_dr;
    }
    let sigma = COMPLEX_PEAK_BASE_FRAC;
    let g = exp(-(r * r) / (2.0 * sigma * sigma));
    let peak = COMPLEX_PEAK_HEIGHT_FRAC * depth * g;
    let dpeak = -COMPLEX_PEAK_HEIGHT_FRAC * depth * g * (r / (sigma * sigma));
    return vec2(base_h + peak, base_dh + dpeak);
}

fn fresh_crater_maturity(r: f32) -> f32 {
    let dr = r - 1.0;
    let dip = exp(-(dr * dr) / (2.0 * RIM_FRESHNESS_SIGMA * RIM_FRESHNESS_SIGMA));
    var ejecta = 0.0;
    if r > 1.0 && r < EJECTA_EXTENT {
        let t = (r - 1.0) / (EJECTA_EXTENT - 1.0);
        let one_minus_t = 1.0 - t;
        ejecta = one_minus_t * one_minus_t;
    }
    let freshness = max(dip, ejecta);
    return clamp(1.0 - freshness, 0.0, 1.0);
}

fn sample_diameter(u: f32, d_lo: f32, d_hi: f32, alpha: f32) -> f32 {
    let lo = pow(d_lo, -alpha);
    let hi = pow(d_hi, -alpha);
    let y = lo + (hi - lo) * u;
    return pow(y, -1.0 / alpha);
}

// ── Per-cell crater accumulator ─────────────────────────────────────────────

struct CraterAccum {
    grad_tangent: vec3<f32>,
    height: f32,
    min_maturity: f32,
}

fn visit_cell(
    accum: ptr<function, CraterAccum>,
    p_unit: vec3<f32>,
    ix: i32,
    iy: i32,
    iz: i32,
    octave: u32,
    cell_size_unit: f32,
    d_lo: f32,
    d_hi: f32,
    expected_per_cell: f32,
    pixel_size_m: f32,
) {
    let h = hash_cell(ix, iy, iz, octave);
    let u_exists = u32_to_unit(h);
    if u_exists >= expected_per_cell {
        return;
    }

    let u_diam = u32_to_unit(pcg(h ^ 0x68E31DA4u));
    let u_px   = u32_to_unit(pcg(h ^ 0xB5297A4Du));
    let u_py   = u32_to_unit(pcg(h ^ 0xBE5466CFu));
    let u_pz   = u32_to_unit(pcg(h ^ 0x1B873593u));
    let u_age  = u32_to_unit(pcg(h ^ 0xD2A9C4B1u));
    let u_ellip   = u32_to_unit(pcg(h ^ 0xA1C4E9F2u));
    let u_orient  = u32_to_unit(pcg(h ^ 0x3F7B8C21u));
    let u_rim_ph  = u32_to_unit(pcg(h ^ 0x9D2E5A73u));
    let u_rim_lob = u32_to_unit(pcg(h ^ 0x54F1D8B6u));

    let cell_origin = vec3<f32>(f32(ix), f32(iy), f32(iz)) * cell_size_unit;
    let cand = cell_origin + vec3<f32>(u_px, u_py, u_pz) * cell_size_unit;
    let cand_len = length(cand);
    if cand_len < 1e-6 {
        return;
    }
    let center = cand / cand_len;

    let diameter_m = sample_diameter(u_diam, d_lo, d_hi, detail.sfd_alpha);

    let diameter_px = diameter_m / max(pixel_size_m, 1e-6);
    let lod_weight = smoothstep(3.0, 8.0, diameter_px);
    if lod_weight <= 0.0 {
        return;
    }

    let radius_m = 0.5 * diameter_m;

    let cos_theta = clamp(dot(p_unit, center), -1.0, 1.0);
    let theta = acos(cos_theta);
    let s_arc_m = theta * detail.body_radius_m;
    var r = s_arc_m / radius_m;
    if r >= EJECTA_EXTENT {
        return;
    }

    let proj = center - cos_theta * p_unit;
    let proj_len2 = dot(proj, proj);
    var azimuth = 0.0;
    if proj_len2 > 1e-12 {
        let px = proj.x;
        let py = proj.y + proj.z * 0.7;
        azimuth = atan2(px, py);
    }

    let ellipticity = u_ellip * 0.2;
    let ellip_angle = u_orient * TAU;
    let ellip_factor = 1.0 + ellipticity * cos(2.0 * (azimuth - ellip_angle));
    r = r / ellip_factor;

    let rim_lobes = floor(u_rim_lob * 4.0 + 3.0);
    let rim_phase = u_rim_ph * TAU;
    var rim_irregular = 0.0;
    if r > 0.85 && r < 1.15 {
        let wave = sin(rim_lobes * azimuth + rim_phase);
        let band = max(0.0, 1.0 - 4.0 * (r - 1.0) * (r - 1.0));
        rim_irregular = 0.35 * wave * band;
    }

    let d_over_dsc = diameter_m / detail.d_sc_m;
    let depth_ratio = select(SIMPLE_DEPTH_RATIO, complex_depth_ratio(d_over_dsc), d_over_dsc >= 1.0);
    let depth = diameter_m * depth_ratio;
    let rim = diameter_m * SIMPLE_RIM_RATIO;

    var hd: vec2<f32>;
    if d_over_dsc >= 1.0 {
        hd = complex_profile(r, depth, rim);
    } else {
        hd = simple_profile(r, depth, rim);
    }
    let h_m = hd.x + rim_irregular * rim;
    let dh_dr = hd.y;

    let age_gyr = u_age * detail.body_age_gyr;
    let age_blend = smoothstep(0.0, FRESH_AGE_GYR, age_gyr);
    let fresh_m = fresh_crater_maturity(r);
    let aged_m = mix(fresh_m, 1.0, age_blend);
    let weighted_m = mix(1.0, aged_m, lod_weight);

    let grad_proj_len = sqrt(proj_len2);
    if grad_proj_len < 1e-8 {
        (*accum).height = (*accum).height + h_m * lod_weight;
        (*accum).min_maturity = min((*accum).min_maturity, weighted_m);
        return;
    }
    let t_hat = proj / grad_proj_len;

    let grad = -(dh_dr) / radius_m * t_hat;

    (*accum).grad_tangent = (*accum).grad_tangent + grad * lod_weight;
    (*accum).height = (*accum).height + h_m * lod_weight;
    (*accum).min_maturity = min((*accum).min_maturity, weighted_m);
}

fn synthesize_small_craters(p_unit: vec3<f32>, pixel_size_m: f32) -> CraterAccum {
    var accum: CraterAccum;
    accum.grad_tangent = vec3<f32>(0.0);
    accum.height = 0.0;
    accum.min_maturity = 1.0;

    if detail.global_k_per_km2 <= 0.0 || detail.d_min_m <= 0.0 {
        return accum;
    }

    let body_r = detail.body_radius_m;
    for (var oi: u32 = 0u; oi < 8u; oi = oi + 1u) {
        let d_lo = detail.d_min_m * pow(2.0, f32(oi));
        let d_hi = min(detail.d_min_m * pow(2.0, f32(oi + 1u)), detail.d_max_m);
        if d_hi <= d_lo {
            break;
        }

        if d_hi < pixel_size_m * 2.0 {
            continue;
        }

        let d_lo_km = d_lo * 1e-3;
        let d_hi_km = d_hi * 1e-3;
        let per_km2 = detail.global_k_per_km2
            * (pow(d_lo_km, -detail.sfd_alpha) - pow(d_hi_km, -detail.sfd_alpha));

        let cell_size_m = 2.0 * d_hi;
        let cell_area_km2 = (cell_size_m * 1e-3) * (cell_size_m * 1e-3);

        let expected_per_cell = per_km2 * cell_area_km2 / 3.0;
        if expected_per_cell <= 0.0 {
            continue;
        }

        let cell_size_unit = cell_size_m / body_r;
        let inv_cell = 1.0 / cell_size_unit;
        let cx = i32(floor(p_unit.x * inv_cell));
        let cy = i32(floor(p_unit.y * inv_cell));
        let cz = i32(floor(p_unit.z * inv_cell));

        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
                    visit_cell(
                        &accum,
                        p_unit,
                        cx + dx, cy + dy, cz + dz,
                        oi,
                        cell_size_unit,
                        d_lo, d_hi,
                        expected_per_cell,
                        pixel_size_m,
                    );
                }
            }
        }
    }

    let grad_len = length(accum.grad_tangent);
    if grad_len > 2.0 {
        accum.grad_tangent = accum.grad_tangent * (2.0 / grad_len);
    }

    return accum;
}

// ── Regional (large-scale) albedo modulation ───────────────────────────────

fn regional_albedo_mod(p: vec3<f32>) -> f32 {
    let s = f32(detail.seed_lo & 0xFFFFu) * (1.0 / 65535.0);
    let phase = vec3<f32>(s * 6.283, s * 4.193, s * 2.711);
    let low =
          sin(p.x * 2.3 + phase.x + 0.7)
        * sin(p.y * 2.1 + phase.y + 1.3)
        * sin(p.z * 2.7 + phase.z + 2.1);
    let mid =
          sin(p.x * 5.1 + phase.y + 3.0)
        * sin(p.y * 4.7 + phase.z + 0.5)
        * sin(p.z * 5.9 + phase.x + 1.7);
    return 1.0 + 0.12 * low + 0.06 * mid;
}

// ── Normal perturbation from height cubemap ────────────────────────────────
//
// Finite-difference normals derived from the height cubemap.  Samples 4
// neighbors offset by a small angle in the tangent plane.  No pre-baked
// slope texture needed — this automatically reflects whichever frequency
// bands are stored in the cubemap.

fn perturb_normal_from_height(n: vec3<f32>) -> vec3<f32> {
    let res = f32(textureDimensions(height_tex).x);
    if res < 2.0 {
        return n;
    }

    // Build tangent frame on the sphere.
    let ref_up = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let tangent = normalize(cross(ref_up, n));
    let bitangent = cross(n, tangent);

    // Offset ~1.5 texels on the cubemap.
    let offset = 1.5 / res;

    let h_e = textureSample(height_tex, height_sampler, n + tangent * offset).r;
    let h_w = textureSample(height_tex, height_sampler, n - tangent * offset).r;
    let h_n = textureSample(height_tex, height_sampler, n + bitangent * offset).r;
    let h_s = textureSample(height_tex, height_sampler, n - bitangent * offset).r;

    // Convert texel offset to world-space distance on the body surface.
    let ds = detail.body_radius_m * offset * 2.0;
    if ds < 1e-6 {
        return n;
    }

    // Height is R16Unorm — the raw sample is in [0, 1].  The shader doesn't
    // need absolute meters, only the *gradient*; since all four samples share
    // the same encoding, the differences are correct as-is (proportional to
    // real meters).
    let dh_dt = (h_e - h_w) / ds;
    let dh_db = (h_n - h_s) / ds;

    return normalize(n - tangent * dh_dt - bitangent * dh_db);
}

// ── Rotation helper ────────────────────────────────────────────────────────
//
// Rotate a direction around the Y axis by `angle` radians.
// Used to apply planet rotation to the cubemap sample direction.

fn rotate_y(dir: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(
        dir.x * c + dir.z * s,
        dir.y,
        -dir.x * s + dir.z * c,
    );
}

// ── Fragment ────────────────────────────────────────────────────────────────

const FRESH_BIAS_COLOR: vec3<f32> = vec3<f32>(185.0, 178.0, 168.0) / 255.0;

@fragment
fn fragment(in: VertexOutput) -> FragOutput {
    let cam_pos  = view.world_position;
    let ray_dir  = normalize(in.world_position - cam_pos);

    // Ray-sphere intersection.
    let oc      = cam_pos - in.sphere_center;
    let half_b  = dot(oc, ray_dir);
    let c       = dot(oc, oc) - params.radius * params.radius;
    let disc    = half_b * half_b - c;
    if disc < 0.0 { discard; }
    let t = -half_b - sqrt(max(disc, 0.0));
    if t < 0.0 { discard; }

    let hit    = cam_pos + t * ray_dir;
    let normal = normalize(hit - in.sphere_center);

    // Apply planet rotation to the sample direction (not the geometry).
    let sample_dir = rotate_y(normal, params.rotation_phase * TAU);

    // Sample cubemap albedo and apply regional modulation.
    let raw_baked = textureSample(albedo_tex, albedo_sampler, sample_dir).rgb;
    let regional = clamp(regional_albedo_mod(sample_dir), 0.7, 1.3);
    let baked_albedo = raw_baked * regional;

    // ── Layer 1: height-derived normal perturbation ──────────────────────────
    var shading_normal = perturb_normal_from_height(sample_dir);
    // Transform back from rotated space to world space for lighting.
    shading_normal = rotate_y(shading_normal, -params.rotation_phase * TAU);

    // ── Layer 2: per-fragment small-crater synthesis ─────────────────────────
    let hit_ddx = dpdx(hit);
    let hit_ddy = dpdy(hit);
    let pixel_render = max(length(hit_ddx), length(hit_ddy));
    let m_per_render_unit = detail.body_radius_m / max(params.radius, 1e-6);
    let pixel_size_m = pixel_render * m_per_render_unit;

    let small = synthesize_small_craters(normal, pixel_size_m);
    if length(small.grad_tangent) > 0.0 {
        let grad_tangent = small.grad_tangent - dot(small.grad_tangent, normal) * normal;
        shading_normal = normalize(shading_normal - grad_tangent);
    }

    let fresh_boost = clamp((1.0 - small.min_maturity) * 0.6, 0.0, 0.8);
    let albedo = mix(baked_albedo, FRESH_BIAS_COLOR, fresh_boost);

    // ── Lighting ───────────────────────────────────────────────────────────
    let roughness   = params.light_dir.w;
    let geo_n_dot_l = dot(normal, params.light_dir.xyz);
    let n_dot_l     = min(dot(shading_normal, params.light_dir.xyz), geo_n_dot_l + 0.05);
    let wrap        = roughness * 0.08;
    let diffuse     = max(0.0, (n_dot_l + wrap) / (1.0 + wrap));
    let irradiance  = params.light_intensity * diffuse + params.ambient_intensity;
    let lit         = albedo * irradiance * (1.0 / PI);

    // Correct depth.
    let hit_clip = view.clip_from_world * vec4(hit, 1.0);
    let depth    = hit_clip.z / hit_clip.w;

    return FragOutput(vec4(lit, 1.0), depth);
}
