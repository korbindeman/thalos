// Transonic vapour cone — the condensation collar that forms around an airframe
// near Mach 1, rendered as a thin scattering shell integrated along the view ray.
//
// **This is not a shock, and it is not emissive.** Both are easy mistakes. The
// visible cloud is condensed water: air accelerating over the airframe expands,
// its static temperature falls below the local dew point, and the vapour it was
// carrying comes out as droplets. Those droplets *scatter sunlight* — so unlike
// the plume and the reentry shell, which share this module's geometry approach,
// the source term here is illumination, not radiance from hot gas. Shading it as
// an emitter produces a self-lit white cone that reads as a decal and looks
// identical at midnight.
//
// The collar is a FILLED body of revolution about the freestream axis — not a
// shell on the shock surface. Reference photographs settle this: the classic
// transonic bell is opaque, you cannot see the airframe through it. The shock
// surface is the *boundary*; the condensation fills the low-pressure region
// inside it.
//
//   s          distance downstream of the collar's origin plane
//   R(s)       bell radius, flaring fast off the apex and easing toward its max
//   rho        filled interior, falling to exactly zero at R(s)
//   L          multi-scatter radiance from `thalos::volumetrics`
//
// Radiance comes from the SHARED volumetric model, not a local phase function.
// A single Henyey-Greenstein lobe renders any cloud "no brighter than the sky
// ambient filling it — grey-blue mud" (measured; see that library). The bell's
// near-opaque white in the reference photos IS the diffusion limit, so it falls
// out of the physics rather than needing to be dialled in.
//
// Everything about *whether* the collar exists lives CPU-side in
// `rendering::vapor_cone` (the Mach window, humidity and dynamic-pressure
// gates); this renders the resolved body.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import thalos::volumetrics::{
    ambient_occlusion,
    multi_scatter_lobes,
    powder_term,
    volumetric_scattering,
    water_cloud_albedo,
}

// Resolved collar profile. Mirrors `VaporConeParams` (Rust, ShaderType). Lanes are
// positional — repurposing one is a rename, not an edit.
struct VaporConeParams {
    // rgb = sun colour reaching the collar, a = extinction per metre at full
    // condensation.
    tint: vec4<f32>,
    // xyz = freestream arrival direction in craft-local axes (unit),
    // w = maximum bell radius (m).
    flow: vec4<f32>,
    // x = downstream station where the collar starts (m, craft-local),
    // y = collar length (m), z = bell flare exponent, w = bound radius (m).
    shape: vec4<f32>,
    // xyz = sun direction in craft-local axes (unit), w = condensation 0..1.
    sun: vec4<f32>,
}

@group(3) @binding(0) var<uniform> params: VaporConeParams;

const MAX_STEPS: i32 = 48;
const SUN_TAPS: i32 = 4;
const TRANS_EPS: f32 = 0.004;

// Droplet phase asymmetry. Water droplets are strongly forward-scattering; these
// are the same dual-lobe inputs the cloud march feeds the shared model.
const G_FORWARD: f32 = 0.75;
const G_BACKWARD: f32 = -0.35;
const G_LERP: f32 = 0.35;

// Bell radius at axial fraction `a` (0 at the apex, 1 at the tail). Flares fast
// off the apex and eases toward the maximum — a straight cone reads as a party
// hat, and the reference bells are visibly ogival.
fn bell_radius(a: f32) -> f32 {
    return params.flow.w * pow(clamp(a, 0.0, 1.0), params.shape.z);
}

// Normalized density at a point, given its axial fraction and radius.
//
// FILLED, not a shell: 1 through the interior, falling to exactly zero at the
// bell surface so the silhouette feathers with no hard edge. The axial window
// ramps in at the apex (condensation begins where the expansion does) and closes
// at the tail (it re-evaporates as pressure recovers) — square ends read as a
// solid object rather than a cloud.
fn collar_density(a: f32, r: f32) -> f32 {
    if (a <= 0.0 || a >= 1.0) {
        return 0.0;
    }
    let radius = bell_radius(a);
    if (radius <= 1e-4 || r >= radius) {
        return 0.0;
    }
    let x = r / radius;
    // Compact support at the surface, flat through the middle.
    let radial = 1.0 - x * x;
    let axial = smoothstep(0.0, 0.22, a) * (1.0 - smoothstep(0.72, 1.0, a));
    return radial * axial;
}

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    // Shared flow-effect prism template: xy = unit circle, z = axial fraction.
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) origin: vec3<f32>,
    @location(2) @interpolate(flat) axis_x: vec3<f32>,
    @location(3) @interpolate(flat) axis_y: vec3<f32>,
    @location(4) @interpolate(flat) axis_z: vec3<f32>,
}

// The proxy hull is a sphere bounding the whole collar, in craft-local axes.
// Closed and convex, so culling back faces leaves exactly one fragment per ray.
@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);

    let u = mix(-0.999, 0.999, in.position.z);
    let ring = sqrt(max(1.0 - u * u, 0.0));
    let unit = vec3<f32>(in.position.x * ring, in.position.y * ring, u);
    let world_pos = world_from_local * vec4<f32>(unit * params.shape.w, 1.0);

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world_pos.xyz);
    out.world_pos = world_pos.xyz;
    out.origin = world_from_local[3].xyz;
    out.axis_x = world_from_local[0].xyz;
    out.axis_y = world_from_local[1].xyz;
    out.axis_z = world_from_local[2].xyz;
    return out;
}

// Ray/sphere overlap about the local origin, as [near, far]; far <= near = miss.
fn sphere_hit(o: vec3<f32>, d: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = 2.0 * dot(o, d);
    let c = dot(o, o) - radius * radius;
    let disc = b * b - 4.0 * c;
    if (disc < 0.0) {
        return vec2<f32>(1.0, 0.0);
    }
    let sq = sqrt(disc);
    return vec2<f32>((-b - sq) * 0.5, (-b + sq) * 0.5);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let condensation = params.sun.w;
    if (condensation <= 0.0) {
        discard;
    }

    let flow_local = params.flow.xyz;
    let collar_start = params.shape.x;
    let collar_len = max(params.shape.y, 1e-3);
    let bound = params.shape.w;
    let extinction = params.tint.a;

    let to_frag = in.world_pos - view.world_position;
    if (dot(to_frag, to_frag) < 1e-12) {
        discard;
    }
    let ray_world = normalize(to_frag);

    // Into craft-local axes: the transform is rigid, so the inverse rotation is
    // three dots against the basis columns.
    let rel = view.world_position - in.origin;
    let o = vec3<f32>(dot(rel, in.axis_x), dot(rel, in.axis_y), dot(rel, in.axis_z));
    let d = vec3<f32>(
        dot(ray_world, in.axis_x),
        dot(ray_world, in.axis_y),
        dot(ray_world, in.axis_z),
    );

    let hit = sphere_hit(o, d, bound);
    if (hit.y <= hit.x) {
        discard;
    }
    let t_start = max(hit.x, 0.0);
    let t_end = hit.y;
    if (t_end <= t_start) {
        discard;
    }

    let downstream = -flow_local;
    let sun_local = params.sun.xyz;
    // cos between the view ray and the sun: +1 looking toward the sun, where the
    // forward lobe peaks. Same convention as the cloud march, which had this
    // negated once and rendered the silver lining 180° from the sun.
    let ray_dot_sun = dot(d, sun_local);
    let lobes = multi_scatter_lobes(ray_dot_sun, G_FORWARD, G_BACKWARD, G_LERP);

    let span = t_end - t_start;
    let ds = span / f32(MAX_STEPS);
    // Sun-tap spacing: across the bell rather than along the ray, so the shadow
    // estimate resolves the body it is shading.
    let sun_step = max(params.flow.w, 0.1) / f32(SUN_TAPS);

    var trans = 1.0;
    var radiance = vec3<f32>(0.0);

    for (var i = 0; i < MAX_STEPS; i = i + 1) {
        let t = t_start + (f32(i) + 0.5) * ds;
        let p = o + d * t;

        let s = dot(p, downstream) - collar_start;
        let a = s / collar_len;
        let radial = p - downstream * dot(p, downstream);
        let r = length(radial);

        let density = collar_density(a, r) * condensation;
        if (density <= 0.001) {
            continue;
        }

        // Optical depth toward the sun, by short march through the same body.
        // Without it the collar has no interior shading at all and reads as a
        // flat cut-out — the multi-scatter model's depth response is exactly
        // what this feeds.
        var tau_sun = 0.0;
        for (var k = 1; k <= SUN_TAPS; k = k + 1) {
            let q = p + sun_local * (f32(k) * sun_step);
            let qs = dot(q, downstream) - collar_start;
            let qr = length(q - downstream * dot(q, downstream));
            tau_sun += collar_density(qs / collar_len, qr) * condensation * sun_step * extinction;
        }

        let scattering = volumetric_scattering(lobes, tau_sun, water_cloud_albedo());
        let powder = powder_term(density, ray_dot_sun);
        let lit = params.tint.rgb * scattering * powder;
        // Sky fill, self-occluded by the same depth the octaves use.
        let amb = params.tint.rgb * 0.35 * ambient_occlusion(tau_sun);

        let dtau = density * extinction * ds;
        let alpha = 1.0 - exp(-dtau);
        radiance += trans * (lit + amb) * alpha;
        trans *= 1.0 - alpha;

        if (trans < TRANS_EPS) {
            break;
        }
    }

    let alpha = clamp(1.0 - trans, 0.0, 1.0);
    // Premultiplied: the collar both scatters its own light and OCCLUDES what is
    // behind it. Unlike the plume and the shock layer this is not additive — a
    // cloud is not transparent to what it covers, which is the whole reason the
    // reference bells hide the airframe.
    return vec4<f32>(radiance, alpha);
}
