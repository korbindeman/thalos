// Ground-LOD terrain shader for procedural bodies.
//
// Reads height, albedo, and roughness from the thalos_udlod attachment
// atlases (group 1), ray-tests a local craft-shadow proxy, and shades
// with the shared Hapke BRDF helper
// (`thalos::lighting::shade_hapke_surface`). The same routine drives
// `planet_impostor.wgsl`, so the LOD swap is visually continuous.
//
// Atmospheric scattering is NOT applied here — it is composited on top
// by `BodySky` (the fullscreen pass in `body_sky.wgsl`) whenever this
// ground terrain LOD is visible. Outside the terrain handoff distance,
// the impostor handles the body and its own inline atmosphere/cloud path.

#import thalos_udlod::types::AtlasTile
#import thalos_udlod::bindings::{config, atlas_sampler, attachments, attachment2_atlas}
#import thalos_udlod::attachments::{sample_attachment1, sample_normal, attachment_uv}
#import thalos_udlod::fragment::{FragmentInput, FragmentOutput, fragment_info}
#import thalos_udlod::functions::lookup_tile
#import thalos::atmosphere::AtmosphereBlock
#import thalos::lighting::{SceneLighting, SCENE_FLUX_SCALE, shade_hapke_surface}

const MAX_TERRAIN_SHADOW_CASTERS: u32 = 16u;

struct BodyTerrainShadow {
    // x = strength, y = minimum penumbra width in metres,
    // z = max receiver distance, w = valid caster count.
    params: vec4<f32>,
    // xyz = part top/near endpoint in render-space metres, w = endpoint radius.
    caster_a_radius: array<vec4<f32>, 16>,
    // xyz = part bottom/far endpoint in render-space metres, w = endpoint radius.
    caster_b_radius: array<vec4<f32>, 16>,
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

// Temporary visual smoothing for the current Thalos ground-LOD path.
//
// The pre-rewrite terrain source still has large macro texels; sampling
// height-derived normals at full strength turns those broad height bands into
// dark contour lines. Keep a little local slope detail, but let the sphere's
// geometric normal carry most of the lighting until the terrain generator can
// provide genuinely continuous close-up fields.
const HEIGHT_NORMAL_WEIGHT: f32 = 0.10;

fn atmospheric_surface_fill(geo_n_dot_l: f32, sun_flux: f32) -> vec3<f32> {
    let atmosphere_strength = max(terrain_atmos.atmos_geom.z, 0.0);
    if (atmosphere_strength <= 0.0 || sun_flux <= 0.0) {
        return vec3<f32>(0.0);
    }

    let rayleigh_tau = max(
        terrain_atmos.rayleigh_beta_h.xyz * terrain_atmos.rayleigh_beta_h.w,
        vec3<f32>(0.0),
    );
    let mie_tau = max(
        terrain_atmos.mie_beta_g.xyz * terrain_atmos.atmos_geom.y,
        vec3<f32>(0.0),
    );
    let tau_rgb = (rayleigh_tau + mie_tau) * atmosphere_strength;
    let tau_mean = max(dot(tau_rgb, vec3<f32>(0.3333333)), 1.0e-5);

    // This is not the camera-path haze; BodySky handles that fullscreen.
    // It is the missing diffuse sky irradiance on terrain faces that are
    // visible under a lit atmosphere but not directly sun-facing.
    let daylight = smoothstep(-0.45, 0.12, geo_n_dot_l);
    let fill_strength = clamp(tau_mean * 0.28, 0.0, 0.08) * daylight;
    let spectral_tint = clamp(
        mix(vec3<f32>(1.0), tau_rgb / tau_mean, 0.25),
        vec3<f32>(0.55),
        vec3<f32>(1.35),
    );

    return spectral_tint * sun_flux * SCENE_FLUX_SCALE * fill_strength;
}

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

fn tapered_segment_shadow(
    hit_ws: vec3<f32>,
    sun_dir_ws: vec3<f32>,
    a_radius: vec4<f32>,
    b_radius: vec4<f32>,
) -> f32 {
    let radius_a = max(a_radius.w, 0.0);
    let radius_b = max(b_radius.w, 0.0);
    if max(radius_a, radius_b) <= 0.0 {
        return 1.0;
    }

    // Cast a ray from the terrain fragment toward the star. Each procedural
    // ship part is represented as the projected silhouette of its visual
    // frustum/cylinder endpoints on the plane perpendicular to that sun ray.
    // This keeps the shadow anchored in world space and gives tanks, adapters,
    // pods, and engines separate tapered silhouettes instead of one whole-ship
    // capsule blob.
    let delta_a = a_radius.xyz - hit_ws;
    let delta_b = b_radius.xyz - hit_ws;
    let ray_t_a = dot(delta_a, sun_dir_ws);
    let ray_t_b = dot(delta_b, sun_dir_ws);
    let proj_a = delta_a - sun_dir_ws * ray_t_a;
    let proj_b = delta_b - sun_dir_ws * ray_t_b;
    let segment = proj_b - proj_a;
    let segment_len2 = dot(segment, segment);

    var h = 0.0;
    var closest = proj_a;
    var radius = max(radius_a, radius_b);
    var ray_t = min(ray_t_a, ray_t_b);
    if segment_len2 > 1.0e-6 {
        h = clamp(dot(-proj_a, segment) / segment_len2, 0.0, 1.0);
        closest = proj_a + segment * h;
        radius = mix(radius_a, radius_b, h);
        ray_t = mix(ray_t_a, ray_t_b, h);
    } else {
        closest = 0.5 * (proj_a + proj_b);
        ray_t = 0.5 * (ray_t_a + ray_t_b);
    }

    if ray_t <= 0.0 || ray_t > terrain_shadow.params.z {
        return 1.0;
    }

    let silhouette_distance = length(closest);
    let penumbra = max(terrain_shadow.params.y, max(radius * 0.18, 0.03));
    let coverage = 1.0 - smoothstep(radius, radius + penumbra, silhouette_distance);
    let fade = 1.0 - smoothstep(terrain_shadow.params.z * 0.75, terrain_shadow.params.z, ray_t);
    return 1.0 - terrain_shadow.params.x * coverage * fade;
}

fn local_craft_shadow(hit_ws: vec3<f32>, sun_dir_ws: vec3<f32>) -> f32 {
    let caster_count = min(
        u32(max(terrain_shadow.params.w, 0.0)),
        MAX_TERRAIN_SHADOW_CASTERS,
    );
    if caster_count == 0u {
        return 1.0;
    }

    var shadow = 1.0;
    for (var i = 0u; i < MAX_TERRAIN_SHADOW_CASTERS; i = i + 1u) {
        if i >= caster_count {
            break;
        }
        shadow = min(
            shadow,
            tapered_segment_shadow(
                hit_ws,
                sun_dir_ws,
                terrain_shadow.caster_a_radius[i],
                terrain_shadow.caster_b_radius[i],
            ),
        );
    }
    return shadow;
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

    var albedo        = sample_attachment1(tile);
    var height_normal = sample_normal(tile, info.world_normal);
    var roughness     = sample_roughness(tile);

    if (info.blend.ratio > 0.0) {
        let tile2 = lookup_tile(info.coordinate, info.blend, 1u);
        albedo = mix(albedo, sample_attachment1(tile2), info.blend.ratio);
        height_normal = mix(
            height_normal,
            sample_normal(tile2, info.world_normal),
            info.blend.ratio,
        );
        roughness = mix(roughness, sample_roughness(tile2), info.blend.ratio);
    }

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
    let frag_relative_position = -info.view_vector;
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
    let normal = normalize(mix(geo_normal, normalize(height_normal), HEIGHT_NORMAL_WEIGHT));
    let view_dir = normalize(info.view_vector);
    let hit_ws = info.world_position.xyz;

    // No crater-shadow or terrain self-shadow yet on the ground LOD path —
    // those are impostor-only today. The local craft proxy supplies a
    // stable sun-ray shadow term for direct sunlight only.
    let external_shadow = local_craft_shadow(hit_ws, sun_dir_ws);
    var lit = shade_hapke_surface(
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
    lit = lit + albedo.rgb * atmospheric_surface_fill(dot(geo_normal, sun_dir_ws), sun_flux);

    var output: FragmentOutput;
    output.color = vec4<f32>(lit, albedo.a);
    return output;
}
