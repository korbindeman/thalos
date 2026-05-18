// Ground-LOD terrain shader for procedural bodies.
//
// Reads height, albedo, and roughness from the thalos_udlod attachment
// atlases (group 1), ray-tests a local craft-shadow proxy, and shades
// with the shared Hapke BRDF helper
// (`thalos::lighting::shade_hapke_surface`). The same routine drives
// `planet_impostor.wgsl`, so the LOD swap is visually continuous.
//
// Atmospheric scattering is NOT applied here — it is composited on top
// by `BodySky` (the fullscreen pass in `sky_dome.wgsl`) whenever this
// ground terrain LOD is visible. Outside the terrain handoff distance,
// the impostor handles the body and its own inline atmosphere/cloud path.

#import thalos_udlod::types::AtlasTile
#import thalos_udlod::bindings::{config, atlas_sampler, attachments, attachment2_atlas}
#import thalos_udlod::attachments::{sample_attachment1, sample_normal, attachment_uv}
#import thalos_udlod::fragment::{FragmentInput, FragmentOutput, fragment_info}
#import thalos_udlod::functions::lookup_tile
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

// Debug overlay: see `BodyTerrainDebug` in `body_material.rs` for the
// layout. `view_phase.xyz` carries the camera's body-fixed position
// taken `mod (2 × cell_size)` per axis (computed on the CPU in f64 then
// downcast), so the only body-scale magnitude in the parity chain stays
// in CPU precision. `world_to_body_rot` is the inverse of the body
// grid's render-space rotation.
struct BodyTerrainDebug {
    params: vec4<f32>,
    view_phase: vec4<f32>,
    world_to_body_rot: vec4<f32>,
}

// Material bind group (group 3 in thalos_udlod's pipeline layout:
//   0 = view, 1 = terrain, 2 = terrain-view, 3 = material).
@group(3) @binding(0) var<uniform> terrain_atmos: AtmosphereBlock;
@group(3) @binding(1) var<uniform> terrain_scene: SceneLighting;
@group(3) @binding(2) var<uniform> terrain_shadow: BodyTerrainShadow;
@group(3) @binding(3) var<uniform> terrain_debug: BodyTerrainDebug;

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

// Standard quaternion rotation `v' = q * v * q⁻¹`, expanded into the
// `v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)` identity that avoids
// the explicit quat product and stays in vec3 land.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

// Analytic anti-aliased 3D checkerboard. Adapted from Iñigo Quílez's
// "Filtering the checkerboard" (https://iquilezles.org/articles/checkerfiltering/);
// the 2D version is extended to 3D by carrying the signed parity through
// a third axis. Each axis's `i` is the screen-footprint average of a
// period-2 square wave, in [-1, +1]; their triple product is +1 on cells
// where `floor(x)+floor(y)+floor(z)` is even and -1 where it is odd, so
// `0.5 - 0.5*i.x*i.y*i.z` reads as a clean 0/1 checker that softens to
// 0.5 wherever the per-fragment footprint exceeds the cell size.
fn checker_3d_aa(p: vec3<f32>, w: vec3<f32>) -> f32 {
    let w_safe = max(w, vec3<f32>(0.01));
    let i = 2.0 * (abs(fract((p - 0.5 * w_safe) * 0.5) - 0.5)
                 - abs(fract((p + 0.5 * w_safe) * 0.5) - 0.5)) / w_safe;
    return 0.5 - 0.5 * i.x * i.y * i.z;
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

    // Debug checkerboard overlay. The body-fixed position of this
    // fragment is recovered as:
    //
    //   body_local = view_phase + R(world_to_body_rot) · (world_pos − view.world_pos)
    //
    // - `world_pos − view.world_pos` is the vertex-interpolated offset
    //   from the camera, in render space. With UDLOD's vertex
    //   high-precision branch active (see `TERRAIN_PRECISION_THRESHOLD_M`
    //   in `ground_terrain.rs`) this is the Taylor relative-position
    //   path's sub-mm output near the camera; the rasterizer smooths it
    //   across the triangle for free.
    // - `R(world_to_body_rot)` rotates that delta into the body-fixed
    //   frame so the cell grid stays put as the body spins underneath.
    // - `view_phase` is the camera's body-fixed position mod-(2·cell)
    //   per axis, taken on the CPU in f64 — the only term whose source
    //   carried body-scale magnitude, kept small before reaching the GPU.
    //
    // None of the terms ever has to span the planet radius in f32, so
    // 1 m cell edges resolve cleanly. Derivatives are evaluated
    // unconditionally to keep them outside divergent control flow.
    let debug_cell = max(terrain_debug.params.z, 1e-3);
    let frag_relative_position = info.world_position.xyz - view_bindings::view.world_position;
    let debug_rel = quat_rotate(terrain_debug.world_to_body_rot, frag_relative_position);
    let debug_p = (terrain_debug.view_phase.xyz + debug_rel) / debug_cell;
    let debug_w = max(abs(dpdx(debug_p)), abs(dpdy(debug_p)));
    let debug_checker = checker_3d_aa(debug_p, debug_w);
    if (terrain_debug.params.x >= 0.5) {
        let dark = vec3<f32>(0.05, 0.05, 0.05);
        let light = vec3<f32>(0.80, 0.80, 0.80);
        albedo = vec4<f32>(mix(dark, light, debug_checker), 1.0);
    }

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
