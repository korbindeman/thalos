// Shared scene-lighting helpers for every planet-surface material.
//
// Mirror of `crates/planet_rendering/src/lighting.rs`. The Rust side
// derives `ShaderType` (encase std140), so the field order here is
// load-bearing: every field, every padding slot, in the same sequence.
//
// Materials `#import` these symbols and embed `SceneLighting` as a
// sub-struct of their own params uniform. The helpers take the scene
// by value (not by pointer) because WGSL forbids passing pointers of
// storage class `uniform` into functions — naga rejects it.

#define_import_path thalos::lighting

const MAX_STARS: u32 = 4u;
const MAX_ECLIPSE_OCCLUDERS: u32 = 8u;
const PI_LIGHTING: f32 = 3.14159265358979323846;

struct StarLight {
    // xyz = unit direction from fragment toward the star in world render space.
    // w   = flux (lux), already scaled by camera exposure gain.
    dir_flux: vec4<f32>,
    // xyz = linear-RGB per-star tint. w = reserved.
    color: vec4<f32>,
}

struct SceneLighting {
    star_count:        u32,
    occluder_count:    u32,
    ambient_intensity: f32,
    scene_header_pad:  f32,

    stars:             array<StarLight, 4>,

    // xyz = world render-space center, w = render-unit radius.
    occluders:         array<vec4<f32>, 8>,

    // Planetshine parent: xyz = center, w = radius. radius == 0 disables.
    planetshine_pos_radius: vec4<f32>,
    // xyz = Bond albedo × tint, w = enable flag.
    planetshine_tint_flag:  vec4<f32>,
}

// Analytical sphere-shadow test along a star ray.
//
// For each occluder, check whether the ray from `hit_ws` toward the star
// passes through the occluder sphere. Soft-edged so the terminator
// doesn't pop. Returns 1.0 = fully lit, 0.0 = fully occluded.
fn eclipse_factor(
    scene: SceneLighting,
    hit_ws: vec3<f32>,
    star_dir: vec3<f32>,
) -> f32 {
    var factor: f32 = 1.0;
    let count = scene.occluder_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let oc = scene.occluders[i];
        let center = oc.xyz;
        let r = oc.w;
        if r <= 0.0 { continue; }
        let delta = center - hit_ws;
        let t = dot(delta, star_dir);
        if t <= 0.0 { continue; }
        let perp2 = dot(delta, delta) - t * t;
        let perp = sqrt(max(perp2, 0.0));
        let penumbra = max(r * 0.1, 1.0);
        let s = smoothstep(r, r + penumbra, perp);
        factor = min(factor, s);
        if factor <= 0.0 { break; }
    }
    return factor;
}

// Planetshine irradiance sample.
//
// Describes the parent body as a Lambert-sphere reflector illuminated by
// the primary star. Returns the direction from the fragment toward the
// parent, the scalar flux arriving at the fragment from that direction,
// and an enable flag (false = no planetshine active at this fragment).
struct PlanetShineSample {
    dir:      vec3<f32>,
    flux:     f32,
    tint:     vec3<f32>,
    enabled:  bool,
}

fn planetshine_sample(
    scene: SceneLighting,
    hit_ws: vec3<f32>,
    star_dir: vec3<f32>,
    star_flux: f32,
) -> PlanetShineSample {
    var out: PlanetShineSample;
    out.dir     = vec3(0.0, 1.0, 0.0);
    out.flux    = 0.0;
    out.tint    = vec3(0.0);
    out.enabled = false;

    let tint_flag = scene.planetshine_tint_flag;
    if tint_flag.w < 0.5 { return out; }

    let pos_rad = scene.planetshine_pos_radius;
    let parent_center = pos_rad.xyz;
    let parent_radius = pos_rad.w;
    if parent_radius <= 0.0 { return out; }

    let to_parent = parent_center - hit_ws;
    let dist = length(to_parent);
    if dist <= parent_radius { return out; }

    let parent_dir = to_parent / dist;
    // Lambert-sphere phase function: f(0) = 1, f(π) = 0.
    let cos_alpha = clamp(dot(star_dir, parent_dir), -1.0, 1.0);
    let alpha     = acos(cos_alpha);
    let phase     = (sin(alpha) + (PI_LIGHTING - alpha) * cos_alpha) / PI_LIGHTING;
    let angular_ratio = parent_radius / dist;
    let angular_sq    = angular_ratio * angular_ratio;

    out.dir     = parent_dir;
    out.flux    = star_flux * angular_sq * phase;
    out.tint    = tint_flag.xyz;
    out.enabled = true;
    return out;
}
