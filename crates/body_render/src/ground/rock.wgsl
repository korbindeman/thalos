// Scattered pebble / rock instanced shader.
//
// Vertex: standard mesh transform plus the per-rock clipmap **scale-grow fade**
// (each stone grows from zero across its ring's near/far edges, scaled about its
// own base — the same seamless, pop-free handoff the grass and trees use). No
// wind; rocks are rigid.
//
// Fragment: lights the vertex-coloured stone through the SAME shared
// `thalos::lighting` rough-dielectric surface model the in-game ground LOD and
// the diorama ground patch use (Oren–Nayar + GGX over an analytic hemisphere sky
// fill), and receives the SAME cascaded sun-shadows the trees cast — so a pebble
// sits in the meadow lit exactly like the ground it rests on, with the trees'
// and its own shadow falling across it. Opaque.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import thalos::lighting::{
    SceneLighting, ThalosSurface, SurfaceSky, shade_surface, compute_surface_sky,
    SURFACE_DIELECTRIC, object_aerial_recession,
}
#import thalos::shadow::{ShadowCascadeBlock, sun_shadow_factor}

// Reuses the `GrassParams` field layout (sun / sky / fade / anchor); only wind
// is unused for rocks.
struct RockParams {
    // xyz = unit direction toward the star (world render space), w = sun flux.
    sun_dir: vec4<f32>,
    // unused (wind) — kept so the uniform mirrors `GrassParams` byte-for-byte.
    wind: vec4<f32>,
    // x = time (s, unused), y = near-edge fade (m), z = far-edge fade (m),
    // w = fade band half-width (m). Drives the per-rock scale-grow handoff.
    time_fade: vec4<f32>,
    // xyz = local radial up for the sky hemisphere split.
    sky_up: vec4<f32>,
    // xyz = Rayleigh vertical optical depth τ_v, w = atmosphere strength.
    sky_tau: vec4<f32>,
    // xyz = vegetation focus OFFSET = (player craft − camera), render space;
    // w = 1 valid / 0 = camera. The fade reference is `view.world_position +
    // offset` — origin-invariant across big_space recentres (see `tree.wgsl`).
    anchor: vec4<f32>,
}

// Standard MaterialPlugin bind group in Bevy 0.18: group 3. Same layout as
// `ground_patch.wgsl` (params, shadow block, three cascade depth maps).
@group(3) @binding(0) var<uniform> rock: RockParams;
@group(3) @binding(1) var<uniform> rock_shadow: ShadowCascadeBlock;
@group(3) @binding(2) var sun_shadow_map_0: texture_depth_2d;
@group(3) @binding(3) var sun_shadow_map_1: texture_depth_2d;
@group(3) @binding(4) var sun_shadow_map_2: texture_depth_2d;

// Stone is fairly rough but not chalk — a touch of broad sheen reads as a worn,
// slightly-damp pebble rather than dust.
const ROCK_ROUGHNESS: f32 = 0.82;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Rock base (this stone's root, tile-centre-relative), baked by the per-tile
    // combiner: uv0 = base.xy, uv1.x = base.z. Drives the per-rock scale-fade.
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    // rgb = stone albedo × baked cavity-AO / top-bleach (linear); a unused.
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);

    // Each stone's base (tile-centre-relative root) is baked into the UVs; the
    // mesh is ONE batched mesh per tile, vertices relative to the tile centre.
    let base = vec3<f32>(in.uv0.x, in.uv0.y, in.uv1.x);

    // Per-rock clipmap scale-grow fade: the stone grows from zero across its
    // ring's near/far edges, scaled about its own base, so the near band fades
    // in/out with no dither and no pop (a collapsed stone is a degenerate,
    // invisible mesh). Distance is from the craft anchor (zoom-independent),
    // rebuilt in the current render origin so big_space recentres cancel.
    let base_world =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(base, 1.0)).xyz;
    let ref_pos = view.world_position + rock.anchor.xyz;
    let inst_dist = distance(ref_pos, base_world);
    let near_edge = rock.time_fade.y;
    let far_edge = rock.time_fade.z;
    let band = max(rock.time_fade.w, 1.0);
    let fade_in = smoothstep(near_edge - band, near_edge + band, inst_dist);
    let fade_out = 1.0 - smoothstep(far_edge - band, far_edge + band, inst_dist);
    let grow = fade_in * fade_out;
    let local = base + (in.position - base) * grow;
    let world_pos =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0)).xyz;

    var out: VertexOutput;
    out.world_position = world_pos;
    out.clip_position = position_world_to_clip(world_pos);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    out.color = in.color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let up = normalize(rock.sky_up.xyz);
    let sun_dir = rock.sun_dir.xyz;
    let sun_flux = rock.sun_dir.w;
    let view_dir = normalize(view.world_position - in.world_position);

    // Same atmosphere-derived sky/sun environment the grass + ground build.
    let sky = compute_surface_sky(rock.sky_tau.xyz, rock.sky_tau.w, up, sun_dir, sun_flux);
    // Tree-cast (and self-) sun shadows — the same cascade the ground samples.
    let shadow = sun_shadow_factor(
        in.world_position, rock_shadow, sun_shadow_map_0, sun_shadow_map_1, sun_shadow_map_2,
    );

    var scene: SceneLighting;  // zeroed; the dielectric path never reads it.
    var s: ThalosSurface;
    s.albedo = in.color.rgb;
    s.roughness = ROCK_ROUGHNESS;
    s.normal_ws = normalize(in.world_normal);
    s.geo_normal_ws = up;
    s.emissive = vec3<f32>(0.0);
    s.occlusion = 1.0;
    s.metallic = 0.0;
    s.translucency = 0.0;
    s.style = SURFACE_DIELECTRIC;

    var lit = shade_surface(
        s, view_dir, in.world_position, sun_dir, sun_flux, scene, sky, shadow, shadow,
    );
    // Recede toward the air with distance (matches the trees), so far stones
    // don't stay crisp against hazed terrain.
    lit = object_aerial_recession(lit, sky, in.world_position, view.world_position);
    return vec4<f32>(lit, 1.0);
}
