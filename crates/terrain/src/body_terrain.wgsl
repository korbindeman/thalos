// Ground-LOD terrain shader for procedural bodies.
//
// Reads height, albedo, and roughness from the bevy_terrain attachment
// atlases (group 1), ray-tests a local craft-shadow proxy, and shades
// with the shared Hapke BRDF helper
// (`thalos::lighting::shade_hapke_surface`). The same routine drives
// `planet_impostor.wgsl`, so the LOD swap is visually continuous.
//
// Atmospheric scattering is NOT applied here — it is composited on top
// by `BodySky` (the fullscreen pass in `sky_dome.wgsl`) whenever this
// ground terrain LOD is visible. Outside the terrain handoff distance,
// the impostor handles the body and its own inline atmosphere/cloud path.

#import bevy_terrain::types::AtlasTile
#import bevy_terrain::bindings::{config, atlas_sampler, attachments, attachment2_atlas}
#import bevy_terrain::attachments::{sample_attachment1, sample_normal, attachment_uv}
#import bevy_terrain::fragment::{FragmentInput, FragmentOutput, fragment_info}
#import bevy_terrain::functions::lookup_tile
#import bevy_pbr::mesh_view_bindings as view_bindings
#import thalos::atmosphere::AtmosphereBlock
#import thalos::lighting::{SceneLighting, shade_hapke_surface}

struct BodyTerrainShadow {
    // xyz = craft proxy center in render-space metres, w = capsule radius.
    caster_pos_radius: vec4<f32>,
    // xyz = craft long axis, w = capsule half-length.
    caster_axis_half_len: vec4<f32>,
    // x = strength, y = penumbra width in metres, z = max receiver distance,
    // w = enabled flag.
    params: vec4<f32>,
}

// Material bind group (group 3 in bevy_terrain's pipeline layout:
//   0 = view, 1 = terrain, 2 = terrain-view, 3 = material).
@group(3) @binding(0) var<uniform> terrain_atmos: AtmosphereBlock;
@group(3) @binding(1) var<uniform> terrain_scene: SceneLighting;
@group(3) @binding(2) var<uniform> terrain_shadow: BodyTerrainShadow;

// Bilinear roughness sample. Mirrors the `sample_attachment1` helper but
// returns the red channel (the only one populated by the tile provider's
// R16-upscaled u8 roughness path).
fn sample_roughness(tile: AtlasTile) -> f32 {
    let uv = attachment_uv(tile.coordinate.uv, 2u);
#ifdef FRAGMENT
#ifdef SAMPLE_GRAD
    return textureSampleGrad(
        attachment2_atlas, atlas_sampler, uv, tile.index,
        tile.coordinate.uv_dx, tile.coordinate.uv_dy,
    ).x;
#else
    return textureSampleLevel(attachment2_atlas, atlas_sampler, uv, tile.index, 0.0).x;
#endif
#else
    return textureSampleLevel(attachment2_atlas, atlas_sampler, uv, tile.index, 0.0).x;
#endif
}

fn distance_to_segment(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let ba = b - a;
    let len2 = dot(ba, ba);
    if len2 <= 1.0e-6 {
        return distance(p, a);
    }
    let h = clamp(dot(p - a, ba) / len2, 0.0, 1.0);
    return distance(p, a + ba * h);
}

fn local_craft_shadow(hit_ws: vec3<f32>, sun_dir_ws: vec3<f32>) -> f32 {
    if terrain_shadow.params.w <= 0.5 {
        return 1.0;
    }

    let radius = terrain_shadow.caster_pos_radius.w;
    let half_len = terrain_shadow.caster_axis_half_len.w;
    if radius <= 0.0 || half_len <= 0.0 {
        return 1.0;
    }

    // Cast a ray from the terrain fragment toward the star. If the craft
    // capsule lies sunward of this fragment, project its capsule silhouette
    // onto the plane perpendicular to the sun ray and test whether the ray
    // passes through it. This is anchored in world space, so zoom/cascade
    // changes cannot move the shadow.
    let delta = terrain_shadow.caster_pos_radius.xyz - hit_ws;
    let ray_t = dot(delta, sun_dir_ws);
    if ray_t <= 0.0 || ray_t > terrain_shadow.params.z {
        return 1.0;
    }

    let receiver_to_shadow_center = delta - sun_dir_ws * ray_t;
    let axis = normalize(terrain_shadow.caster_axis_half_len.xyz);
    let axis_projection = axis - sun_dir_ws * dot(axis, sun_dir_ws);
    let half_axis_projection = axis_projection * half_len;
    let silhouette_distance = distance_to_segment(
        vec3<f32>(0.0),
        receiver_to_shadow_center - half_axis_projection,
        receiver_to_shadow_center + half_axis_projection,
    );

    let penumbra = max(terrain_shadow.params.y, 0.01);
    let coverage = 1.0 - smoothstep(radius, radius + penumbra, silhouette_distance);
    let fade = 1.0 - smoothstep(terrain_shadow.params.z * 0.75, terrain_shadow.params.z, ray_t);
    return 1.0 - terrain_shadow.params.x * coverage * fade;
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var info = fragment_info(input);
    let tile = lookup_tile(info.coordinate, info.blend, 0u);

    var albedo    = sample_attachment1(tile);
    var normal    = sample_normal(tile, info.world_normal);
    var roughness = sample_roughness(tile);

    if (info.blend.ratio > 0.0) {
        let tile2 = lookup_tile(info.coordinate, info.blend, 1u);
        albedo    = mix(albedo,    sample_attachment1(tile2),                info.blend.ratio);
        normal    = mix(normal,    sample_normal(tile2, info.world_normal),  info.blend.ratio);
        roughness = mix(roughness, sample_roughness(tile2),                  info.blend.ratio);
    }
    normal = normalize(normal);

    // Primary star — single-star path matches the impostor.
    let primary = terrain_scene.stars[0];
    let sun_dir_ws = primary.dir_flux.xyz;
    let sun_flux   = primary.dir_flux.w;

    let geo_normal = normalize(info.world_normal);
    let view_dir   = normalize(view_bindings::view.world_position - info.world_position.xyz);
    let hit_ws     = info.world_position.xyz;

    // No crater-shadow or terrain self-shadow yet on the ground LOD path —
    // those are impostor-only today. The local craft proxy supplies a
    // stable sun-ray shadow term for direct sunlight only.
    let external_shadow = local_craft_shadow(hit_ws, sun_dir_ws);
    let lit = shade_hapke_surface(
        albedo.rgb,
        clamp(roughness, 0.0, 1.0),
        normal,
        geo_normal,
        view_dir,
        hit_ws,
        sun_dir_ws,
        sun_flux,
        terrain_scene,
        external_shadow,
    );

    var output: FragmentOutput;
    output.color = vec4<f32>(lit, albedo.a);
    return output;
}
