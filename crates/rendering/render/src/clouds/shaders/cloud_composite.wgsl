// Canonical fullscreen cloud composite. The custom BodySky atmosphere renders
// earlier in the frame; this pass owns both the near-volume layer and the
// weather-derived orbital projection.

#import bevy_pbr::mesh_view_bindings::view
#import thalos::atmosphere::{
    AtmosphereBlock,
    weather_column_from_texel,
    orbital_cloud_altitude,
    orbital_cloud_shade,
    sample_weather_soft,
    orbital_cloud_normal_body,
    WEATHER_TEXEL_ANGLE,
}
#import thalos::lighting::SCENE_FLUX_SCALE

@group(3) @binding(0) var<uniform> cloud_atmosphere: AtmosphereBlock;

struct CloudCompositeParams {
    sun_dir_flux:              vec4<f32>,
    planet_center_radius:      vec4<f32>,
    world_to_body_orientation: vec4<f32>,
    cloud_band_radii:          vec4<f32>,
    ocean:                     vec4<f32>,
    ocean_color_depth:         vec4<f32>,
    ocean_camera_phase:        vec4<f32>,
    ocean_low_phase:           vec4<f32>,
    ocean_high_phase:          vec4<f32>,
    ocean_slope_amplitudes:    vec4<f32>,
    ocean_spectrum:            vec4<f32>,
    ocean_wind_basis:          vec4<f32>,
    ocean_crosswind_basis:     vec4<f32>,
    tile_lookup:               vec4<f32>,
    tile_atlas_uv:             vec4<f32>,
    // x = near-march view step count; the composite mirrors the marcher's
    // per-ray reach from it (see march_reach) for the near/far partition.
    cloud_march:               vec4<f32>,
}
@group(3) @binding(1) var<uniform> cloud_params: CloudCompositeParams;
@group(3) @binding(2) var scene_depth_texture: texture_depth_2d;
@group(3) @binding(3) var weather_texture: texture_cube<f32>;
@group(3) @binding(4) var weather_sampler: sampler;
@group(3) @binding(5) var cloud_layer_texture: texture_2d<f32>;
@group(3) @binding(6) var cloud_distance_texture: texture_2d<f32>;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4(in.position.xy, 1.0, 1.0);
    return out;
}

fn rotate_quat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

struct CloudOverlay {
    premul_rgb: vec3<f32>,
    opacity: f32,
}

fn reconstruct_ray(pixel: vec2<f32>) -> vec3<f32> {
    let ndc_x = (pixel.x / view.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y / view.viewport.w) * 2.0;
    let cam_right = view.world_from_view[0].xyz;
    let cam_up = view.world_from_view[1].xyz;
    let cam_fwd = -view.world_from_view[2].xyz;
    return normalize(
        cam_right * (ndc_x / view.clip_from_view[0][0])
        + cam_up * (ndc_y / view.clip_from_view[1][1])
        + cam_fwd
    );
}

fn scene_distance(pixel: vec2<f32>) -> f32 {
    let depth = textureLoad(scene_depth_texture, vec2<i32>(pixel), 0);
    if depth <= 0.0 {
        return 1.0e30;
    }
    let ndc_x = (pixel.x / view.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y / view.viewport.w) * 2.0;
    let view_h = view.view_from_clip * vec4<f32>(ndc_x, ndc_y, depth, 1.0);
    return length(view_h.xyz / view_h.w);
}

fn sample_near_cloud(pixel: vec2<f32>) -> vec4<f32> {
    let dims = textureDimensions(cloud_layer_texture);
    let cloud_res = vec2<f32>(dims);
    let uv = pixel / view.viewport.zw;
    let p = uv * cloud_res - 0.5;
    let base = floor(p);
    let f = p - base;
    // Last rows/columns store the temporal camera metadata.
    let bilinear_max = vec2<i32>(i32(dims.x) - 2, i32(dims.y) - 4);
    let cb = clamp(vec2<i32>(base), vec2<i32>(0), bilinear_max);
    let cs00 = textureLoad(cloud_layer_texture, cb, 0);
    let cs10 = textureLoad(cloud_layer_texture, cb + vec2<i32>(1, 0), 0);
    let cs01 = textureLoad(cloud_layer_texture, cb + vec2<i32>(0, 1), 0);
    let cs11 = textureLoad(cloud_layer_texture, cb + vec2<i32>(1, 1), 0);
    let cd00 = textureLoad(cloud_distance_texture, cb, 0).r;
    let cd10 = textureLoad(cloud_distance_texture, cb + vec2<i32>(1, 0), 0).r;
    let cd01 = textureLoad(cloud_distance_texture, cb + vec2<i32>(0, 1), 0).r;
    let cd11 = textureLoad(cloud_distance_texture, cb + vec2<i32>(1, 1), 0).r;
    let ref_coord = cb + vec2<i32>(select(0, 1, f.x >= 0.5), select(0, 1, f.y >= 0.5));
    let cloud_near = textureLoad(cloud_distance_texture, ref_coord, 0).r;
    let depth_scale = max(cloud_near, 2000.0);
    let hit_ref = cloud_near < 1.0e8;
    let dw00 = select(0.0, select(1.0, exp(-abs(cd00 - cloud_near) / depth_scale), hit_ref && cd00 < 1.0e8), hit_ref == (cd00 < 1.0e8));
    let dw10 = select(0.0, select(1.0, exp(-abs(cd10 - cloud_near) / depth_scale), hit_ref && cd10 < 1.0e8), hit_ref == (cd10 < 1.0e8));
    let dw01 = select(0.0, select(1.0, exp(-abs(cd01 - cloud_near) / depth_scale), hit_ref && cd01 < 1.0e8), hit_ref == (cd01 < 1.0e8));
    let dw11 = select(0.0, select(1.0, exp(-abs(cd11 - cloud_near) / depth_scale), hit_ref && cd11 < 1.0e8), hit_ref == (cd11 < 1.0e8));
    let w00 = (1.0 - f.x) * (1.0 - f.y) * dw00;
    let w10 = f.x * (1.0 - f.y) * dw10;
    let w01 = (1.0 - f.x) * f.y * dw01;
    let w11 = f.x * f.y * dw11;
    let weight = max(w00 + w10 + w01 + w11, 1.0e-5);
    return (cs00 * w00 + cs10 * w10 + cs01 * w01 + cs11 * w11) / weight;
}

fn near_visibility(
    cloud_near: f32,
    scene_t: f32,
    oc_len_sq: f32,
    b: f32,
) -> f32 {
    let r_base = cloud_params.cloud_band_radii.x;
    let r_top = cloud_params.cloud_band_radii.y;
    if scene_t >= 1.0e29 || r_top <= r_base {
        return 1.0;
    }
    let cam_r = sqrt(oc_len_sq);
    let disc_base = b * b - (oc_len_sq - r_base * r_base);
    let disc_top = b * b - (oc_len_sq - r_top * r_top);
    let sqrt_base = sqrt(max(disc_base, 0.0));
    let sqrt_top = sqrt(max(disc_top, 0.0));
    var band_far = 1.0e30;
    if cam_r < r_base {
        band_far = max(-b + sqrt_top, 0.0);
    } else if cam_r <= r_top {
        var exit = -b + sqrt_top;
        let base_down = -b - sqrt_base;
        if disc_base > 0.0 && base_down > 0.0 {
            exit = min(exit, base_down);
        }
        band_far = max(exit, 1.0);
    } else if disc_base > 0.0 {
        band_far = max(-b - sqrt_base, 0.0);
    } else {
        band_far = max(-b + sqrt_top, 0.0);
    }
    let near = min(cloud_near, band_far);
    return clamp((scene_t - near) / max(band_far - near, 1.0), 0.0, 1.0);
}

// Mirror of the near marcher's per-ray reach (`get_ray` in
// clouds_compute.wgsl — keep the two in lockstep). Returns
// vec2(fade_begin, t_stop): the marcher dissolves its density over
// [fade_begin, t_stop], so the orbital estimator fades in complementarily and
// owns everything beyond t_stop. This is the partition of unity that replaced
// the old camera-altitude branch (which left a cloudless trough between the
// 50 km march cap and the 22 km altitude gate).
fn march_reach(oc_len_sq: f32, b: f32) -> vec3<f32> {
    let r_base = cloud_params.cloud_band_radii.x;
    let r_top = cloud_params.cloud_band_radii.y;
    let steps = max(cloud_params.cloud_march.x, 1.0);
    let cam_r = sqrt(oc_len_sq);
    let disc_top = b * b - (oc_len_sq - r_top * r_top);
    if disc_top <= 0.0 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let sq_top = sqrt(disc_top);
    let tt0 = -b - sq_top;
    let tt1 = -b + sq_top;
    let disc_base = b * b - (oc_len_sq - r_base * r_base);
    let hit_base = disc_base > 0.0;
    var tb0 = 0.0;
    var tb1 = 0.0;
    if hit_base {
        let sq_base = sqrt(disc_base);
        tb0 = -b - sq_base;
        tb1 = -b + sq_base;
    }
    var seg_start = 0.0;
    var seg_end = 0.0;
    if cam_r > r_top {
        seg_start = tt0;
        if hit_base && tb0 > tt0 { seg_end = tb0; } else { seg_end = tt1; }
    } else if cam_r < r_base {
        if hit_base && tb1 > 0.0 { seg_start = tb1; } else { seg_start = max(tt0, 0.0); }
        seg_end = tt1;
    } else {
        seg_start = 0.0;
        if hit_base && tb0 > 0.0 { seg_end = tb0; } else { seg_end = tt1; }
    }
    seg_start = max(seg_start, 0.0);
    seg_end = min(seg_end, seg_start + 50000.0);
    let step = clamp((seg_end - seg_start) / 42.0, 65.0, 500.0);
    let span = steps * step;
    let t_stop = min(seg_start + span, seg_end);
    let fade_begin = min(seg_start + 0.50 * span, t_stop);
    // z = the marcher's entry-distance ownership (its ENTRY_FADE window):
    // shells entered far away belong wholly to this band march.
    let near_scale = 1.0 - smoothstep(36000.0, 50000.0, seg_start);
    return vec3<f32>(fade_begin, t_stop, near_scale);
}

// Reduced weather-column band march: the far estimator's answer to grazing
// geometry. K stratified samples traverse the ray's shell band accumulating
// column occupancy × local vertical profile, so the limb gets true vertical
// thickness and distant horizon decks break where columns are clear — instead
// of one representative sphere whose tangent painted a hard-edged solid wall.
// Weather varies at ≥ ~20 km/texel, so K samples across even a 500 km chord
// stay well-sampled (no 3-D noise is touched here).
const ORBITAL_MARCH_SAMPLES = 6u;

fn sample_orbital_cloud(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    planet_center: vec3<f32>,
    planet_radius: f32,
    oc_len_sq: f32,
    b: f32,
    scene_t: f32,
) -> CloudOverlay {
    if cloud_atmosphere.cloud_albedo_coverage.w <= 0.0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }
    let r_base = cloud_params.cloud_band_radii.x;
    let r_top = cloud_params.cloud_band_radii.y;
    if r_top <= r_base {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }

    // First forward slab of the shell band, same regime logic as the marcher.
    let disc_top = b * b - (oc_len_sq - r_top * r_top);
    if disc_top <= 0.0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }
    let sq_top = sqrt(disc_top);
    let tt0 = -b - sq_top;
    let tt1 = -b + sq_top;
    let disc_base = b * b - (oc_len_sq - r_base * r_base);
    let hit_base = disc_base > 0.0;
    var tb0 = 0.0;
    var tb1 = 0.0;
    if hit_base {
        let sq_base = sqrt(disc_base);
        tb0 = -b - sq_base;
        tb1 = -b + sq_base;
    }
    let cam_r = sqrt(oc_len_sq);
    var t0 = 0.0;
    var t1 = 0.0;
    if cam_r > r_top {
        t0 = tt0;
        if hit_base && tb0 > tt0 { t1 = tb0; } else { t1 = tt1; }
    } else if cam_r < r_base {
        if hit_base && tb1 > 0.0 { t0 = tb1; } else { t0 = max(tt0, 0.0); }
        t1 = tt1;
    } else {
        t0 = 0.0;
        if hit_base && tb0 > 0.0 { t1 = tb0; } else { t1 = tt1; }
    }
    t0 = max(t0, 0.0);
    t1 = min(t1, scene_t);
    if t1 <= t0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }

    let reach = march_reach(oc_len_sq, b);
    let base_alt = max(cloud_atmosphere.cloud_shape.x, 0.0);
    let thickness = max(cloud_atmosphere.cloud_shape.y, 1.0);

    let seg = (t1 - t0) / f32(ORBITAL_MARCH_SAMPLES);
    let inv_thickness = 1.0 / thickness;
    // Footprint mip: one march segment's angular span over the weather cube.
    let lod_chord = clamp(
        log2(max((seg / planet_radius) / WEATHER_TEXEL_ANGLE, 1.0)),
        0.0,
        5.0,
    );
    // `col.opacity` is an AREAL FRACTION (see weather_cloud_opacity), so the
    // chord takes the strongest single column plus a damped stacking term —
    // compounding fractions as independent opaque slabs saturated a
    // 46%-coverage planet into solid overcast.
    var best_c = 0.0;
    var sum_c = 0.0;
    var best_t = -1.0;
    // Height at the running segment boundary, as a fraction of the shell.
    var h_prev = (length(cam_pos + t0 * ray_dir - planet_center) - planet_radius - base_alt)
        * inv_thickness;
    for (var i = 0u; i < ORBITAL_MARCH_SAMPLES; i++) {
        let t_a = t0 + f32(i) * seg;
        let t_b = t_a + seg;
        let t_m = 0.5 * (t_a + t_b);
        let h_a = h_prev;
        let h_b = (length(cam_pos + t_b * ray_dir - planet_center) - planet_radius - base_alt)
            * inv_thickness;
        h_prev = h_b;
        // Partition of unity with the near march: segments the marcher already
        // integrated contribute nothing here.
        var far_w = 1.0;
        if reach.y > 0.0 && reach.z > 0.0 {
            var near_own = 0.0;
            if reach.y - reach.x > 1.0 {
                near_own = 1.0 - smoothstep(reach.x, reach.y, t_m);
            } else {
                near_own = select(1.0, 0.0, t_m > reach.y);
            }
            far_w = 1.0 - reach.z * near_own;
        }
        if far_w <= 0.0 {
            continue;
        }
        let p = cam_pos + t_m * ray_dir - planet_center;
        let n_l = rotate_quat(cloud_params.world_to_body_orientation, normalize(p));
        let col = weather_column_from_texel(
            textureSampleLevel(weather_texture, weather_sampler, n_l, lod_chord),
        );
        if col.opacity <= 1.0e-4 {
            continue;
        }
        // Analytic box-profile occupancy: the fraction of this segment's
        // height span that lies inside the column's deck. Point-sampling the
        // profile needed a per-pixel jitter against arc stripes, and that
        // jitter — with no temporal pass over this material — rendered as
        // white-noise speckle holes across the far disc; the segment integral
        // is deterministic and smooth in both t and screen space.
        let h_lo = min(h_a, h_b);
        let h_hi = max(h_a, h_b);
        let overlap = max(min(h_hi, col.top_frac) - max(h_lo, col.base_frac), 0.0);
        let profile = clamp(overlap / max(h_hi - h_lo, 1.0e-4), 0.0, 1.0);
        let a_col = 1.0 - exp(-col.optical_depth * 1.2);
        let candidate = col.opacity * profile * a_col * far_w;
        if candidate > best_c {
            best_c = candidate;
            best_t = t_m;
        }
        sum_c += candidate;
    }
    let march_opacity = clamp(
        best_c + 0.35 * (sum_c - best_c) * (1.0 - best_c),
        0.0,
        0.92,
    );
    if march_opacity <= 1.0e-3 || best_t < 0.0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }

    // Light the dominant sample through the shared soft column + height-moment
    // normal so this projection shades exactly like SolidPlanet's. The moment
    // normal's ~15 km relief detail is only meaningful while a weather texel
    // still spans several pixels; at long range it flickered per-texel through
    // the terminator (amber/white confetti at sunset), so it relaxes toward
    // the smooth sphere normal with distance.
    let p_best = cam_pos + best_t * ray_dir - planet_center;
    let cloud_alt = length(p_best) - planet_radius;
    let cloud_n_w = normalize(p_best);
    let cloud_n_l = rotate_quat(cloud_params.world_to_body_orientation, cloud_n_w);
    let column = weather_column_from_texel(
        sample_weather_soft(weather_texture, weather_sampler, cloud_n_l, lod_chord),
    );
    let n_body_lit_raw =
        orbital_cloud_normal_body(weather_texture, weather_sampler, cloud_n_l, lod_chord);
    let normal_detail = smoothstep(220000.0, 60000.0, best_t);
    let n_body_lit = normalize(mix(cloud_n_l, n_body_lit_raw, 0.25 + 0.75 * normal_detail));
    let q = cloud_params.world_to_body_orientation;
    let cloud_n_lit_w = rotate_quat(vec4(-q.xyz, q.w), n_body_lit);
    let n_dot_l = dot(cloud_n_lit_w, cloud_params.sun_dir_flux.xyz);
    let view_mu = max(dot(cloud_n_lit_w, -ray_dir), 0.0);
    let shade = orbital_cloud_shade(column, n_dot_l, view_mu);
    let night = smoothstep(-0.12, 0.08, n_dot_l);
    let sun_chroma = mix(
        vec3<f32>(1.0, 0.45, 0.18),
        vec3<f32>(1.0, 0.97, 0.92),
        smoothstep(0.05, 0.45, clamp(n_dot_l, 0.0, 1.0)),
    );
    // Occupancy comes from the band march; `shade.y` would double-count it.
    let opacity = march_opacity * night;
    var radiance = cloud_atmosphere.cloud_albedo_coverage.rgb
        * vec3<f32>(0.90, 0.93, 0.97)
        * sun_chroma
        * cloud_params.sun_dir_flux.w
        * SCENE_FLUX_SCALE
        * 0.55
        * shade.x
        * night;

    // Foreground airlight: this overlay draws OVER the already-integrated sky,
    // so its opacity would otherwise delete the air in front of the cloud —
    // distant decks then read as extinction-only brown. Approximate the veil
    // with the same analytic β the near march uses and restore it here.
    let dens = exp(-max(cloud_alt, 0.0) / 8000.0);
    let mu_view = max(abs(dot(cloud_n_w, ray_dir)), 0.2);
    let d_air = min(best_t, 60000.0 / mu_view);
    let tau = vec3<f32>(5.5e-6, 1.3e-5, 3.2e-5) * dens * d_air;
    let view_T = exp(-tau);
    let day = smoothstep(-0.05, 0.30, n_dot_l);
    let e_air = vec3<f32>(0.40, 0.55, 0.85)
        * cloud_params.sun_dir_flux.w
        * SCENE_FLUX_SCALE
        * 0.12
        * day;
    radiance = radiance * view_T + e_air * (vec3<f32>(1.0) - view_T);

    return CloudOverlay(radiance * opacity, opacity);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Only the body currently projected by the one ship-view cloud compute
    // pass enables this material. The guard also prevents out-of-bounds reads
    // from the inactive 1x1 fallback textures.
    if cloud_params.cloud_band_radii.w < 0.5 {
        discard;
    }

    let ray_dir = reconstruct_ray(in.clip_position.xy);
    let cam_pos = view.world_position;
    let planet_center = cloud_params.planet_center_radius.xyz;
    let planet_radius = cloud_params.planet_center_radius.w;
    let oc = cam_pos - planet_center;
    let oc_len_sq = dot(oc, oc);
    let b = dot(oc, ray_dir);
    let scene_t = scene_distance(in.clip_position.xy);

    let near_sample = sample_near_cloud(in.clip_position.xy);
    let dims = textureDimensions(cloud_layer_texture);
    let uv = in.clip_position.xy / view.viewport.zw;
    let ref_coord = clamp(
        vec2<i32>(uv * vec2<f32>(dims)),
        vec2<i32>(0),
        vec2<i32>(dims) - vec2<i32>(1),
    );
    let cloud_near = textureLoad(cloud_distance_texture, ref_coord, 0).r;
    let near_vis = near_visibility(cloud_near, scene_t, oc_len_sq, b);
    let near_cloud = CloudOverlay(
        near_sample.rgb * near_vis,
        (1.0 - near_sample.a) * near_vis,
    );
    let orbital = sample_orbital_cloud(
        cam_pos, ray_dir, planet_center, planet_radius, oc_len_sq, b, scene_t,
    );
    let cloud = CloudOverlay(
        near_cloud.premul_rgb + orbital.premul_rgb * (1.0 - near_cloud.opacity),
        1.0 - (1.0 - near_cloud.opacity) * (1.0 - orbital.opacity),
    );
    if cloud.opacity <= 1.0e-5 {
        discard;
    }
    return vec4(cloud.premul_rgb, clamp(cloud.opacity, 0.0, 1.0));
}
