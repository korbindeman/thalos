// Canonical fullscreen cloud composite. Atmosphere rendering happens earlier
// in the frame (Bevy raymarch normally, legacy BodySky only for an explicit
// A/B); this pass owns both the near-volume layer and the weather-derived
// orbital projection, so changing atmosphere backends never hides clouds.

#import bevy_pbr::mesh_view_bindings::view
#import thalos::atmosphere::{
    AtmosphereBlock,
    weather_column_from_texel,
    orbital_cloud_altitude,
    orbital_cloud_shade,
    sample_weather_soft,
    orbital_cloud_normal_body,
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

fn sample_orbital_cloud(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    planet_center: vec3<f32>,
    planet_radius: f32,
    oc_len_sq: f32,
    b: f32,
) -> CloudOverlay {
    let cam_r = sqrt(oc_len_sq);
    let orbital_blend = smoothstep(22000.0, 85000.0, cam_r - planet_radius);
    if orbital_blend <= 0.0 || cloud_atmosphere.cloud_albedo_coverage.w <= 0.0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }

    let r_seed = 0.5 * (cloud_params.cloud_band_radii.x + cloud_params.cloud_band_radii.y);
    let disc_seed = b * b - (oc_len_sq - r_seed * r_seed);
    if disc_seed <= 0.0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }
    let sq_seed = sqrt(disc_seed);
    var t_seed = -b - sq_seed;
    if t_seed <= 0.0 { t_seed = -b + sq_seed; }
    if t_seed <= 0.0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }

    let seed_n_w = normalize(cam_pos + t_seed * ray_dir - planet_center);
    let seed_n_l = rotate_quat(cloud_params.world_to_body_orientation, seed_n_w);
    let seed_col = weather_column_from_texel(
        sample_weather_soft(weather_texture, weather_sampler, seed_n_l),
    );
    var cloud_n_w = seed_n_w;
    var cloud_n_l = seed_n_l;
    if seed_col.opacity > 1.0e-3 {
        let r_orbit = planet_radius + orbital_cloud_altitude(seed_col, cloud_atmosphere);
        let disc = b * b - (oc_len_sq - r_orbit * r_orbit);
        if disc > 0.0 {
            let sq = sqrt(disc);
            var t = -b - sq;
            if t <= 0.0 { t = -b + sq; }
            if t > 0.0 {
                cloud_n_w = normalize(cam_pos + t * ray_dir - planet_center);
                cloud_n_l = rotate_quat(cloud_params.world_to_body_orientation, cloud_n_w);
            }
        }
    }

    let column = weather_column_from_texel(
        sample_weather_soft(weather_texture, weather_sampler, cloud_n_l),
    );
    if column.opacity <= 1.0e-3 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }
    let n_body_lit = orbital_cloud_normal_body(weather_texture, weather_sampler, cloud_n_l);
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
    let opacity = shade.y * night * orbital_blend;
    let radiance = cloud_atmosphere.cloud_albedo_coverage.rgb
        * vec3<f32>(0.90, 0.93, 0.97)
        * sun_chroma
        * cloud_params.sun_dir_flux.w
        * SCENE_FLUX_SCALE
        * 0.55
        * shade.x
        * night;
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
        cam_pos, ray_dir, planet_center, planet_radius, oc_len_sq, b,
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
