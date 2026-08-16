#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    mesh_bindings::mesh,
    mesh_functions,
    prepass_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}
#else
#import bevy_pbr::{
    mesh_bindings::mesh,
    mesh_functions,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{Vertex, VertexOutput, FragmentOutput},
    view_transformations::position_world_to_clip,
}
#endif

#import thalos::ocean_waves::{
    OceanSlopeSample,
    OceanSurfaceWave,
    ocean_coastal_wave_scale,
    ocean_sample_slope_field,
    ocean_sample_surface_wave,
}

struct OceanMaterial {
    deep_color: vec4<f32>,
    shelf_color: vec4<f32>,
    shallow_color: vec4<f32>,
    slope_amplitudes: vec4<f32>,
    low_phase: vec4<f32>,
    high_phase: vec4<f32>,
    surface_wavelengths_m: vec4<f32>,
    surface_amplitudes_m: vec4<f32>,
    surface_phases_rad: vec4<f32>,
    previous_surface_phases_rad: vec4<f32>,
    // xy = wave direction, z = whitecap slope onset, w = coarse coast range.
    wind_and_foam: vec4<f32>,
    // xy = minimum local XZ, zw = local width/depth.
    coast_bounds: vec4<f32>,
    shore_bounds: vec4<f32>,
    // x = signed shore range, yz = wave geometry fade range.
    shore_params: vec4<f32>,
    // x = current elapsed time, y = frame delta.
    time: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> ocean: OceanMaterial;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var ocean_slope_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var ocean_slope_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var coast_distance_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104)
var coast_distance_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(105)
var shore_properties_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106)
var shore_properties_sampler: sampler;

struct CoastSample {
    distance_to_land_m: f32,
    shore_distance_m: f32,
    cliff: f32,
    exposure: f32,
}

fn inside_texture(uv: vec2<f32>) -> f32 {
    return f32(all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)));
}

fn sample_coast(world_xz: vec2<f32>) -> CoastSample {
    let coast_uv = (world_xz - ocean.coast_bounds.xy) / ocean.coast_bounds.zw;
    let coast_inside = inside_texture(coast_uv);
    let coarse = textureSampleLevel(
        coast_distance_texture,
        coast_distance_sampler,
        clamp(coast_uv, vec2<f32>(0.0), vec2<f32>(1.0)),
        0.0,
    ).r;

    let shore_uv = (world_xz - ocean.shore_bounds.xy) / ocean.shore_bounds.zw;
    let shore_inside = inside_texture(shore_uv);
    let properties = textureSampleLevel(
        shore_properties_texture,
        shore_properties_sampler,
        clamp(shore_uv, vec2<f32>(0.0), vec2<f32>(1.0)),
        0.0,
    );

    var result: CoastSample;
    result.distance_to_land_m = mix(ocean.wind_and_foam.w, coarse * ocean.wind_and_foam.w, coast_inside);
    result.shore_distance_m = mix(
        -ocean.shore_params.x,
        (properties.r * 2.0 - 1.0) * ocean.shore_params.x,
        shore_inside,
    );
    result.cliff = properties.g * shore_inside;
    result.exposure = mix(1.0, properties.b, shore_inside);
    return result;
}

fn coastal_wave_scale(coast: CoastSample) -> f32 {
    let water_distance_m = max(-coast.shore_distance_m, 0.0);
    return ocean_coastal_wave_scale(
        water_distance_m,
        ocean.shore_params.x,
        coast.exposure,
    );
}

fn sample_surface_wave(
    world_xz: vec2<f32>,
    sample_spacing_m: f32,
    phases_rad: vec4<f32>,
) -> OceanSurfaceWave {
    let wind = normalize(ocean.wind_and_foam.xy);
    let coast = sample_coast(world_xz);
    return ocean_sample_surface_wave(
        world_xz,
        sample_spacing_m,
        wind,
        ocean.surface_wavelengths_m,
        ocean.surface_amplitudes_m,
        phases_rad,
        coastal_wave_scale(coast),
    );
}

fn vertex_sample_spacing(local_xz: vec2<f32>) -> f32 {
    return max(2.0, length(local_xz) / 40.0);
}

fn displace_ocean_vertex(
    world_position: vec4<f32>,
    local_xz: vec2<f32>,
    phases_rad: vec4<f32>,
) -> vec4<f32> {
    var displaced = world_position;
    let geometry_fade = 1.0 - smoothstep(
        ocean.shore_params.y,
        ocean.shore_params.z,
        length(local_xz),
    );
    displaced.y += sample_surface_wave(
        world_position.xz,
        vertex_sample_spacing(local_xz),
        phases_rad,
    ).height * geometry_fade;
    return displaced;
}

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    let base_world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex_in.position, 1.0),
    );
    let surface = sample_surface_wave(
        base_world_position.xz,
        vertex_sample_spacing(vertex_in.position.xz),
        ocean.surface_phases_rad,
    );
    out.world_position = displace_ocean_vertex(
        base_world_position,
        vertex_in.position.xz,
        ocean.surface_phases_rad,
    );
    out.position = position_world_to_clip(out.world_position.xyz);

#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif

#ifdef VERTEX_NORMALS
#ifdef PREPASS_PIPELINE
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    out.world_normal = normalize(vec3<f32>(-surface.slope.x, 1.0, -surface.slope.y));
#endif
#else
    out.world_normal = normalize(vec3<f32>(-surface.slope.x, 1.0, -surface.slope.y));
#endif
#endif

#ifdef PREPASS_PIPELINE
#ifdef MOTION_VECTOR_PREPASS
    let previous_world_from_local = mesh_functions::get_previous_world_from_local(vertex_in.instance_index);
    let previous_base_world_position = mesh_functions::mesh_position_local_to_world(
        previous_world_from_local,
        vec4<f32>(vertex_in.position, 1.0),
    );
    out.previous_world_position = displace_ocean_vertex(
        previous_base_world_position,
        vertex_in.position.xz,
        ocean.previous_surface_phases_rad,
    );
#endif
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex_in.instance_index;
#endif

#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex_in.instance_index,
        world_from_local[3],
    );
#endif

    return out;
}

#ifndef PREPASS_PIPELINE
fn sample_slope_field(world_xz: vec2<f32>) -> OceanSlopeSample {
    let wind = normalize(ocean.wind_and_foam.xy);
    let crosswind = vec2<f32>(-wind.y, wind.x);
    let local_m = vec2<f32>(dot(world_xz, wind), dot(world_xz, crosswind));
    let world_dx = dpdx(world_xz);
    let world_dy = dpdy(world_xz);
    let local_dx = vec2<f32>(dot(world_dx, wind), dot(world_dx, crosswind));
    let local_dy = vec2<f32>(dot(world_dy, wind), dot(world_dy, crosswind));
    return ocean_sample_slope_field(
        ocean_slope_texture,
        ocean_slope_sampler,
        local_m,
        local_dx,
        local_dy,
        ocean.low_phase,
        ocean.high_phase,
        ocean.slope_amplitudes,
        0.0,
        0.34,
    );
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let coast = sample_coast(pbr_input.world_position.xz);
    let world_dx = dpdx(pbr_input.world_position.xz);
    let world_dy = dpdy(pbr_input.world_position.xz);
    let footprint_m = sqrt(max(length(world_dx) * length(world_dy), 1.0e-6));
    let surface = sample_surface_wave(
        pbr_input.world_position.xz,
        footprint_m,
        ocean.surface_phases_rad,
    );
    let detail = sample_slope_field(pbr_input.world_position.xz);
    let total_slope = surface.slope + detail.slope;
    pbr_input.N = normalize(vec3<f32>(-total_slope.x, 1.0, -total_slope.y));

    let water_distance_m = max(-coast.shore_distance_m, 0.0);
    // Curaçao's wave-facing north coast is predominantly a limestone edge;
    // exposure supplements the coarse DEM when the 30 m height field rounds
    // the actual cliff lip away.
    let cliff = max(coast.cliff, coast.exposure * 0.72);
    let beach = 1.0 - cliff;
    let shelf = (1.0 - smoothstep(220.0, 1900.0, coast.distance_to_land_m))
        * mix(0.28, 1.0, beach);
    let near_shoal = (1.0 - smoothstep(14.0, 115.0, water_distance_m)) * beach;
    let protected_shoal = (1.0 - smoothstep(120.0, 850.0, coast.distance_to_land_m))
        * beach * 0.42;
    let shoal = max(near_shoal, protected_shoal);
    var water_color = mix(ocean.deep_color.rgb, ocean.shelf_color.rgb, shelf * 0.82);
    water_color = mix(water_color, ocean.shallow_color.rgb, shoal * 0.78);
    pbr_input.material.base_color = vec4<f32>(water_color, 1.0);
    let alpha_ggx = clamp(
        sqrt(detail.alpha_ggx * detail.alpha_ggx + 2.0 * surface.omitted_variance),
        0.06,
        0.22,
    );
    pbr_input.material.perceptual_roughness = sqrt(alpha_ggx);

    let whitecap = smoothstep(
        ocean.wind_and_foam.z,
        ocean.wind_and_foam.z + 0.12,
        length(total_slope),
    ) * smoothstep(0.62, 0.90, detail.breakup);
    let breaker_band = smoothstep(18.0, 42.0, water_distance_m)
        * (1.0 - smoothstep(78.0, 116.0, water_distance_m));
    let breaker = breaker_band * coast.exposure
        * mix(1.0, 0.66, cliff)
        * smoothstep(0.59, 0.84, surface.crest)
        * smoothstep(0.24, 0.68, detail.breakup);
    let runup_phase = 0.5 + 0.5 * sin(
        water_distance_m * 0.23
        - ocean.time.x * 1.55
        + sin(dot(pbr_input.world_position.xz, vec2<f32>(0.035, 0.027))) * 1.4,
    );
    let runup_coverage = 0.16 + 0.84 * smoothstep(0.58, 0.82, runup_phase);
    let runup = beach
        * (1.0 - smoothstep(2.0, 13.0, water_distance_m))
        * runup_coverage;
    let cliff_impact = cliff * coast.exposure
        * (1.0 - smoothstep(3.0, 20.0, water_distance_m))
        * smoothstep(0.48, 0.76, surface.crest + detail.breakup * 0.22);
    let foam = clamp(whitecap * 0.52 + breaker * 0.48 + runup * 0.34 + cliff_impact * 0.62, 0.0, 0.86);
    let foamed_color = mix(
        pbr_input.material.base_color.rgb,
        vec3<f32>(0.82, 0.86, 0.84),
        foam,
    );
    pbr_input.material.base_color = vec4<f32>(foamed_color, 1.0);
    pbr_input.material.perceptual_roughness = mix(
        pbr_input.material.perceptual_roughness,
        0.88,
        foam,
    );

    pbr_input.material.base_color = alpha_discard(
        pbr_input.material,
        pbr_input.material.base_color,
    );
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
#endif
