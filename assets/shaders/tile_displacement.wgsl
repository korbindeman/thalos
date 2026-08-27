// Shared terrain vertex stage for visible, prepass, deferred, and shadow
// pipelines. Keep displacement in this one entry point: the main opaque pass
// uses Equal depth after the prepass, so even a numerically equivalent second
// implementation can self-reject.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::{Vertex, VertexOutput}
#else
#import bevy_pbr::forward_io::{Vertex, VertexOutput}
#endif

// Reading the MATERIAL bind group from a vertex stage that also serves the
// prepass and the shadow pass is what forces `DISPLACED_PREPASS_ALPHA_MODE`
// (tiles/material.rs): a depth-only opaque pass would otherwise bind an empty
// group 3 here and wgpu would fail pipeline creation outright.
@group(#{MATERIAL_BIND_GROUP}) @binding(111)
var tile_position_atlas: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(112)
var tile_surface_atlas: texture_2d_array<f32>;

const TILE_ATLAS_WIDTH: u32 = 131u;
const TILE_ATLAS_SLOT_MASK: u32 = 2047u;
const TILE_ATLAS_SLOT_BITS: u32 = 11u;
const TILE_ORIGIN_BODY_TEXEL: u32 = 17673u;
const TILE_ORIGIN_WRAPPED_TEXEL: u32 = 17674u;

fn atlas_coord(linear_index: u32) -> vec2<i32> {
    return vec2<i32>(
        i32(linear_index % TILE_ATLAS_WIDTH),
        i32(linear_index / TILE_ATLAS_WIDTH),
    );
}

fn position_texel(coord: vec2<i32>, layer: i32) -> vec4<f32> {
    return textureLoad(tile_position_atlas, coord, layer, 0);
}

fn coarser_sample_coord(sample_coord: vec2<i32>, source_step: i32) -> vec4<i32> {
    let coarse_step = source_step * 2;
    let local = sample_coord - vec2<i32>(1);
    let lo = vec2<i32>(1) + (local / vec2<i32>(coarse_step)) * vec2<i32>(coarse_step);
    let hi = min(lo + vec2<i32>(coarse_step), vec2<i32>(129));
    return vec4<i32>(lo, hi);
}

fn morph_position_sample(
    sample_coord: vec2<i32>,
    layer: i32,
    source_step: i32,
    morph: f32,
) -> vec4<f32> {
    let bounds = coarser_sample_coord(sample_coord, source_step);
    let lo = bounds.xy;
    let hi = bounds.zw;
    let extent = max(vec2<f32>(hi - lo), vec2<f32>(1.0));
    let t = vec2<f32>(sample_coord - lo) / extent;
    let a = position_texel(vec2<i32>(lo.x, lo.y), layer);
    let b = position_texel(vec2<i32>(hi.x, lo.y), layer);
    let c = position_texel(vec2<i32>(lo.x, hi.y), layer);
    let d = position_texel(vec2<i32>(hi.x, hi.y), layer);
    let coarse = mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
    return mix(coarse, position_texel(sample_coord, layer), morph);
}

fn surface_texel(coord: vec2<i32>, layer: i32) -> vec4<f32> {
    return textureLoad(tile_surface_atlas, coord, layer, 0);
}

fn morph_surface_sample(
    sample_coord: vec2<i32>,
    layer: i32,
    source_step: i32,
    morph: f32,
) -> vec4<f32> {
    let bounds = coarser_sample_coord(sample_coord, source_step);
    let lo = bounds.xy;
    let hi = bounds.zw;
    let extent = max(vec2<f32>(hi - lo), vec2<f32>(1.0));
    let t = vec2<f32>(sample_coord - lo) / extent;
    let a = surface_texel(vec2<i32>(lo.x, lo.y), layer);
    let b = surface_texel(vec2<i32>(hi.x, lo.y), layer);
    let c = surface_texel(vec2<i32>(lo.x, hi.y), layer);
    let d = surface_texel(vec2<i32>(hi.x, hi.y), layer);
    let coarse = mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
    return mix(coarse, surface_texel(sample_coord, layer), morph);
}

fn surface_normal_local(
    sample_coord: vec2<i32>,
    layer: i32,
    origin_body: vec3<f32>,
    source_step: i32,
    morph: f32,
) -> vec3<f32> {
    let center = morph_position_sample(sample_coord, layer, source_step, morph).xyz;
    let du = morph_position_sample(
        sample_coord + vec2<i32>(1, 0), layer, source_step, morph,
    ).xyz - morph_position_sample(
        sample_coord - vec2<i32>(1, 0), layer, source_step, morph,
    ).xyz;
    let dv = morph_position_sample(
        sample_coord + vec2<i32>(0, 1), layer, source_step, morph,
    ).xyz - morph_position_sample(
        sample_coord - vec2<i32>(0, 1), layer, source_step, morph,
    ).xyz;
    var normal = normalize(cross(du, dv));
    let outward = normalize(origin_body + center);
    if dot(normal, outward) < 0.0 {
        normal = -normal;
    }
    return normal;
}

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let instance_index = vertex_in.instance_index;
    let tag = mesh_functions::get_tag(instance_index);
    let layer = i32(tag & TILE_ATLAS_SLOT_MASK);
    let morph = f32((tag >> TILE_ATLAS_SLOT_BITS) & 255u) / 255.0;
    let previous_morph = f32((tag >> (TILE_ATLAS_SLOT_BITS + 8u)) & 255u) / 255.0;
    let source_step = i32(vertex_in.uv.x);
    let is_skirt = vertex_in.uv.y > 0.5;

    // Shared-patch POSITION is an integer address, not geometry:
    // xy = displaced-position texel, z = top-surface sample linear index.
    // Skirt vertices address their packed bottom position through xy while z
    // keeps their top sample for normals/material coordinates.
    let position_coord = vec2<i32>(i32(vertex_in.position.x), i32(vertex_in.position.y));
    let sample_coord = atlas_coord(u32(vertex_in.position.z));
    let exact_displaced = position_texel(position_coord, layer);
    let top = morph_position_sample(sample_coord, layer, source_step, morph);
    let displaced = select(top, exact_displaced, is_skirt);
    let origin_body = position_texel(atlas_coord(TILE_ORIGIN_BODY_TEXEL), layer).xyz;
    let origin_wrapped = position_texel(
        atlas_coord(TILE_ORIGIN_WRAPPED_TEXEL),
        layer,
    ).xyz;
    let surface = morph_surface_sample(sample_coord, layer, source_step, morph);

    let world_from_local = mesh_functions::get_world_from_local(instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(displaced.xyz, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);

#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif

    let normal_local = surface_normal_local(
        sample_coord, layer, origin_body, source_step, morph,
    );
#ifdef PREPASS_PIPELINE
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(normal_local, instance_index);
#endif
#else
    out.world_normal = mesh_functions::mesh_normal_local_to_world(normal_local, instance_index);
#endif

    let wrapped = origin_wrapped + top.xyz;
#ifdef VERTEX_UVS_A
    out.uv = wrapped.xy;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vec2<f32>(wrapped.z, top.w);
#endif
#ifdef VERTEX_COLORS
    out.color = surface;
#endif

#ifdef MOTION_VECTOR_PREPASS
    let previous_world_from_local = mesh_functions::get_previous_world_from_local(instance_index);
    let previous_top = morph_position_sample(
        sample_coord, layer, source_step, previous_morph,
    );
    let previous_displaced = select(previous_top, exact_displaced, is_skirt);
    out.previous_world_position = mesh_functions::mesh_position_local_to_world(
        previous_world_from_local,
        vec4<f32>(previous_displaced.xyz, 1.0),
    );
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = instance_index;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        instance_index,
        world_from_local[3],
    );
#endif
    return out;
}
