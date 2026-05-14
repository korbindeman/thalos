// Ground-LOD terrain shader for procedural bodies.
//
// Reads height, albedo, and roughness from the bevy_terrain attachment
// atlases (group 1) and shades them with the shared Hapke BRDF helper
// (`thalos::lighting::shade_hapke_surface`). The same routine drives
// `planet_impostor.wgsl`, so the LOD swap is visually continuous.
//
// Atmospheric scattering is NOT applied here — it is composited on top
// by `BodySky` (the fullscreen pass in `sky_dome.wgsl`) whenever the
// camera is inside the body's atmosphere shell. Outside the shell, the
// impostor handles the body and this terrain pass is hidden by the
// impostor↔terrain LOD swap.

#import bevy_terrain::types::AtlasTile
#import bevy_terrain::bindings::{config, atlas_sampler, attachments, attachment2_atlas}
#import bevy_terrain::attachments::{sample_attachment1, sample_normal, attachment_uv}
#import bevy_terrain::fragment::{FragmentInput, FragmentOutput, fragment_info}
#import bevy_terrain::functions::lookup_tile
#import bevy_pbr::mesh_view_bindings::view
#import thalos::atmosphere::AtmosphereBlock
#import thalos::lighting::{SceneLighting, shade_hapke_surface}

// Material bind group (group 3 in bevy_terrain's pipeline layout:
//   0 = view, 1 = terrain, 2 = terrain-view, 3 = material).
@group(3) @binding(0) var<uniform> terrain_atmos: AtmosphereBlock;
@group(3) @binding(1) var<uniform> terrain_scene: SceneLighting;

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
    let view_dir   = normalize(view.world_position - info.world_position.xyz);
    let hit_ws     = info.world_position.xyz;

    // No crater-shadow or self-shadow yet on the terrain path — those are
    // impostor-only today. Pass 1.0 so the helper degrades to plain Hapke +
    // eclipse + planetshine + ambient.
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
        1.0,
    );

    var output: FragmentOutput;
    output.color = vec4<f32>(lit, albedo.a);
    return output;
}
