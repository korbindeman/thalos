// Flat, sky-model-lit ground patch shader (previews / dioramas).
//
// A non-UDLOD ground plane shaded through the SAME shared `thalos::lighting`
// surface model the in-game ground LOD uses — the rough-dielectric BRDF
// (Oren–Nayar + GGX) over an analytic hemisphere sky fill from
// `compute_surface_sky` — and receiving the SAME cascaded sun-shadows that
// scattered trees cast (the `tree.wgsl` / `body_terrain.wgsl` cascade sampler,
// verbatim). So a previewed plant sits on ground that lights like the terrain it
// grows from, with the plant's own shadow falling across it. Opaque.
//
// Deliberately simple: a flat patch with a low-frequency value-noise albedo
// breakup, not the full ecological-band terrain stack. The grass decoration on
// top carries the fine texture; this is the shadow receiver + the colour the
// gaps between blades read as.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import thalos::lighting::{
    SceneLighting, ThalosSurface, SurfaceSky, shade_surface, compute_surface_sky,
    SURFACE_DIELECTRIC,
}

// Reuses the `GrassParams` field layout (only the lighting fields are read).
struct GroundParams {
    // xyz = unit direction toward the star (world render space), w = sun flux.
    sun_dir: vec4<f32>,
    // unused (wind) — kept so the uniform mirrors `GrassParams` byte-for-byte.
    wind: vec4<f32>,
    // x = time (s); unused here.
    time_fade: vec4<f32>,
    // xyz = local radial up for the sky hemisphere split.
    sky_up: vec4<f32>,
    // xyz = Rayleigh vertical optical depth τ_v, w = atmosphere strength.
    sky_tau: vec4<f32>,
    // unused (vegetation focus offset).
    anchor: vec4<f32>,
}

// Mirrors `ShadowCascadeBlock` (body_material.rs); array sizes == CASCADE_COUNT.
struct ShadowCascadeBlock {
    view_proj: array<mat4x4<f32>, 3>,
    params: array<vec4<f32>, 3>,
    // x = strength (0 ⇒ skip), y = active cascade count, zw reserved.
    gate: vec4<f32>,
}

// Standard MaterialPlugin bind group in Bevy 0.18: group 3.
@group(3) @binding(0) var<uniform> ground: GroundParams;
@group(3) @binding(1) var<uniform> ground_shadow: ShadowCascadeBlock;
@group(3) @binding(2) var sun_shadow_map_0: texture_depth_2d;
@group(3) @binding(3) var sun_shadow_map_1: texture_depth_2d;
@group(3) @binding(4) var sun_shadow_map_2: texture_depth_2d;

// Ecological ground anchors, matching `body_terrain.wgsl`'s palette so the patch
// reads as the same ground the in-game terrain shades.
const C_FOREST: vec3<f32> = vec3<f32>(0.034, 0.084, 0.028);
const C_GRASS: vec3<f32>  = vec3<f32>(0.072, 0.152, 0.050);
const C_SOIL: vec3<f32>   = vec3<f32>(0.112, 0.074, 0.042);

// One cascade's shadow factor, or a negative sentinel if outside its box (the
// caller falls through to the next cascade). Identical to `tree.wgsl`.
fn cascade_factor(
    world_pos: vec3<f32>,
    vp: mat4x4<f32>,
    bias: f32,
    strength: f32,
    tex: texture_depth_2d,
    inset: f32,
    fade: bool,
) -> f32 {
    let clip = vp * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return -1.0;
    }
    let ndc = clip.xyz / clip.w;
    if (any(ndc.xy < vec2<f32>(-inset)) || any(ndc.xy > vec2<f32>(inset)) ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return -1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let dims = vec2<f32>(textureDimensions(tex));
    var lit = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let texel = vec2<i32>(uv * dims) + vec2<i32>(dx, dy);
            let stored = textureLoad(tex, texel, 0);
            lit = lit + select(1.0, 0.0, stored > ndc.z + bias);
        }
    }
    lit = lit / 9.0;
    var edge_fade = 1.0;
    if (fade) {
        let edge = max(abs(ndc.x), abs(ndc.y));
        edge_fade = 1.0 - smoothstep(0.85, 1.0, edge);
    }
    return 1.0 - strength * (1.0 - lit) * edge_fade;
}

// Walk cascades near→far, use the tightest hit. `gate.x == 0` ⇒ fully lit.
fn sun_shadow_factor(world_pos: vec3<f32>) -> f32 {
    let s = ground_shadow.gate.x;
    if (s <= 0.0) {
        return 1.0;
    }
    var f = cascade_factor(
        world_pos, ground_shadow.view_proj[0], ground_shadow.params[0].x,
        s, sun_shadow_map_0, 0.98, false,
    );
    if (f < 0.0) {
        f = cascade_factor(
            world_pos, ground_shadow.view_proj[1], ground_shadow.params[1].x,
            s, sun_shadow_map_1, 0.98, false,
        );
    }
    if (f < 0.0) {
        f = cascade_factor(
            world_pos, ground_shadow.view_proj[2], ground_shadow.params[2].x,
            s, sun_shadow_map_2, 1.0, true,
        );
    }
    if (f < 0.0) {
        return 1.0;
    }
    return f;
}

// Cheap 2-D value noise + 3-octave fBm for the albedo breakup.
fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p: vec2<f32>) -> f32 {
    return 0.6 * value_noise(p) + 0.3 * value_noise(p * 2.3) + 0.1 * value_noise(p * 5.1);
}

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let world_pos =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0)).xyz;
    var out: VertexOutput;
    out.world_position = world_pos;
    out.clip_position = position_world_to_clip(world_pos);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let up = normalize(ground.sky_up.xyz);
    let sun_dir = ground.sun_dir.xyz;
    let sun_flux = ground.sun_dir.w;
    let view_dir = normalize(view.world_position - in.world_position);

    // Same atmosphere-derived sky/sun environment the grass + ground build.
    let sky = compute_surface_sky(ground.sky_tau.xyz, ground.sky_tau.w, up, sun_dir, sun_flux);
    // Tree-cast (and self-) sun shadows.
    let shadow = sun_shadow_factor(in.world_position);

    // Low-frequency ecological breakup so the gaps between grass blades don't
    // read as one flat tint: forest-floor green ↔ grass green, with a faint
    // soil tint in the lightest patches and a gentle value mottle.
    let n = fbm2(in.world_position.xz * 0.06);
    var base = mix(C_FOREST, C_GRASS, smoothstep(0.30, 0.72, n));
    base = mix(base, C_SOIL, smoothstep(0.82, 0.98, n) * 0.4);
    let albedo = base * (0.80 + 0.42 * n);

    var scene: SceneLighting;  // zeroed; the dielectric path never reads it.
    var s: ThalosSurface;
    s.albedo = albedo;
    s.roughness = 0.93;
    s.normal_ws = normalize(in.world_normal);
    s.geo_normal_ws = up;
    s.emissive = vec3<f32>(0.0);
    s.occlusion = 1.0;
    s.metallic = 0.0;
    s.translucency = 0.0;
    s.style = SURFACE_DIELECTRIC;

    let lit = shade_surface(
        s, view_dir, in.world_position, sun_dir, sun_flux, scene, sky, shadow, shadow,
    );
    return vec4<f32>(lit, 1.0);
}
