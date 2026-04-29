// Solid-color planet placeholder.
//
// Fullscreen-quad ray-traced sphere. The vertex shader emits a fullscreen
// clip-space quad, and the fragment shader reconstructs each pixel's
// world-space ray direction from screen position, intersects the sphere,
// and shades it with simple Lambertian + ambient + planetshine. Used for
// bodies that don't have a terrain pipeline yet.
//
// Why fullscreen instead of a tightly-fit billboard: any 3D billboard
// sized to enclose the sphere's silhouette has failure modes when the
// camera is close and the body off-axis (corners behind near plane, or
// quad center far enough off-axis that the in-plane offsets don't move
// the corners onto the visible silhouette). Fullscreen sidesteps the
// whole class of bugs — every screen pixel gets a fragment, ray-sphere
// decides what to shade vs. discard.

#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_functions::get_world_from_local
#import thalos::lighting::{SceneLighting, eclipse_factor, planetshine_sample}

const PI: f32 = 3.14159265358979323846;

struct SolidPlanetParams {
    radius:  f32,
    albedo:  vec4<f32>,
    scene:   SceneLighting,
}

@group(3) @binding(0) var<uniform> params: SolidPlanetParams;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) sphere_center:  vec3<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let model = get_world_from_local(in.instance_index);
    let sphere_center = (model * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    // Fullscreen clip-space quad. The mesh is `Rectangle::new(2.0, 2.0)`
    // with corners at ±1 in local x/y, so passing the raw in.position
    // through covers the entire viewport in NDC, regardless of where the
    // body is. The model transform is intentionally ignored.
    //
    // z = 1.0 is the near plane in Bevy's reverse-Z, so the rasterizer
    // accepts every fragment by the early depth test. The fragment
    // shader then writes the real per-pixel hit depth via
    // @builtin(frag_depth) on hit, or discards on miss (no depth or
    // color write).
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

    // Reconstruct the world-space ray through this fragment from screen
    // position. Avoid `world_from_clip` and large-magnitude world-space
    // arithmetic — at orbital distances both `world_from_clip * ndc`
    // and `cam_pos` are millions of meters and the subtraction loses
    // f32 precision, which collapses `dpdx/dpdy` of any downstream
    // texture coords across adjacent pixels. The camera-basis form
    // works in small numbers (basis vectors are unit, ndc and tan_fov
    // are O(1)) so adjacent-pixel `ray_dir` differs by a real angular
    // delta rather than precision noise.
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

    // Ray-sphere intersection.
    let oc     = cam_pos - in.sphere_center;
    let half_b = dot(oc, ray_dir);
    let c      = dot(oc, oc) - params.radius * params.radius;
    let disc   = half_b * half_b - c;
    if disc < 0.0 { discard; }
    let t = -half_b - sqrt(max(disc, 0.0));
    if t < 0.0 { discard; }

    let hit    = cam_pos + t * ray_dir;
    let normal = normalize(hit - in.sphere_center);

    let star     = params.scene.stars[0];
    let sun_dir  = star.dir_flux.xyz;
    let sun_flux = star.dir_flux.w;

    let n_dot_l = max(dot(normal, sun_dir), 0.0);
    let eclipse = eclipse_factor(params.scene, hit, sun_dir);
    var lit = params.albedo.xyz * sun_flux * n_dot_l * eclipse * (1.0 / PI);

    let shine = planetshine_sample(params.scene, hit, sun_dir, sun_flux);
    if shine.enabled {
        let n_dot_p = max(dot(normal, shine.dir), 0.0);
        lit = lit + params.albedo.xyz * shine.tint * shine.flux * n_dot_p * (1.0 / PI);
    }

    lit = lit + params.albedo.xyz * params.scene.ambient_intensity * (1.0 / PI);

    let clip = view.clip_from_world * vec4(hit, 1.0);
    return FragOutput(vec4(lit, 1.0), clip.z / clip.w);
}
