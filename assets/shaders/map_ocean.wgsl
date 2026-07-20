// Analytic ocean sphere for the orbital map view.
//
// Fullscreen ray-traced sphere at sea level (the same billboard approach as
// `solid_planet.wgsl`): the vertex shader emits a fullscreen clip-space quad,
// the fragment shader reconstructs each pixel's world ray, intersects the ocean
// sphere, shades it as water, and writes the true hit depth via
// `@builtin(frag_depth)`. Being an opaque depth-writing pass, it depth-tests
// against the map terrain — land occludes it, the seabed sits behind it — so the
// waterline is exact and there is no mesh, no facets, no z-fighting.

#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_functions::get_world_from_local
#import thalos::lighting::{SceneLighting, SCENE_FLUX_SCALE}
#import thalos::water::shade_ocean

struct MapOceanParams {
    radius:      f32,
    color_depth: vec4<f32>,
    scene:       SceneLighting,
}

@group(3) @binding(0) var<uniform> params: MapOceanParams;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) sphere_center: vec3<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let model = get_world_from_local(in.instance_index);
    let sphere_center = (model * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    // Fullscreen clip-space quad. The mesh is `Rectangle::new(2.0, 2.0)` with
    // corners at ±1, so the raw position covers the whole viewport in NDC
    // regardless of where the body sits; the model transform only carries the
    // sphere centre. z = 1.0 = near plane (reverse-Z) so the raster accepts every
    // fragment; the fragment shader writes the real per-pixel depth on a hit.
    var out: VertexOutput;
    out.clip_position = vec4(in.position.x, in.position.y, 1.0, 1.0);
    out.sphere_center = sphere_center;
    return out;
}

struct FragOutput {
    @location(0)         color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fragment(in: VertexOutput) -> FragOutput {
    let cam_pos = view.world_position;

    // Reconstruct the world ray in the camera basis (small numbers — precision-
    // safe), matching `solid_planet.wgsl`.
    let cam_right = view.world_from_view[0].xyz;
    let cam_up    = view.world_from_view[1].xyz;
    let cam_fwd   = -view.world_from_view[2].xyz;
    let ndc_x = (in.clip_position.x / view.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (in.clip_position.y / view.viewport.w) * 2.0;
    let tan_fov_y = 1.0 / view.clip_from_view[1][1];
    let tan_fov_x = 1.0 / view.clip_from_view[0][0];
    let ray_dir = normalize(
        cam_right * (ndc_x * tan_fov_x)
        + cam_up * (ndc_y * tan_fov_y)
        + cam_fwd
    );

    let center = in.sphere_center;

    // Ray-sphere against the sea-level radius. At MAP_SCALE the magnitudes are
    // O(units), so the naive quadratic is f32-stable.
    let oc     = cam_pos - center;
    let half_b = dot(oc, ray_dir);
    let c      = dot(oc, oc) - params.radius * params.radius;
    let disc   = half_b * half_b - c;
    if disc < 0.0 {
        discard;
    }
    let t = -half_b - sqrt(disc);
    if t < 0.0 {
        discard;
    }

    let hit      = cam_pos + t * ray_dir;
    let geo_n    = normalize(hit - center);
    let view_dir = normalize(cam_pos - hit);

    let star     = params.scene.stars[0];
    let sun_dir  = star.dir_flux.xyz;
    let sun_flux = star.dir_flux.w * SCENE_FLUX_SCALE;

    // Waves off at map scale (view_dist huge → the wave-normal fade returns the
    // geometric normal); deep-water colour (column huge — no seabed sample here).
    let water = shade_ocean(
        hit,
        geo_n,
        view_dir,
        1.0e9,
        0.0,
        sun_dir,
        sun_flux,
        params.color_depth,
        1.0e6,
        // Far shore sentinels: no shore interaction at map scale.
        1.0e6,
        1.0e9,
        vec3<f32>(0.0, 0.0, 1.0),
        1.0e9,
    );

    // True hit depth so the hardware depth buffer sorts this against the terrain.
    let clip = view.clip_from_world * vec4(hit, 1.0);
    return FragOutput(vec4(water, 1.0), clip.z / clip.w);
}
