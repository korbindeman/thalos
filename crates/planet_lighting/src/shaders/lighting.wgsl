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

// ── Hapke surface BRDF and unified direct-lighting helper ─────────────────
//
// `shade_hapke_surface` is the one surface-shading routine every planet
// material consumes — the impostor billboard and the bevy_terrain ground
// LOD both call into this same function, so the two render paths shade
// identically at the LOD swap. Material-specific effects (atmospheric
// scattering, cloud compositing, water BRDF, limb darkening) live in
// their own callers and apply on top of the value returned here.

// Scene-flux normalisation. Hapke's BRDF returns a radiance factor; the
// prior pipeline used a Lambert `/PI` normalisation that we fold into
// this single scalar so existing flux values don't need re-tuning. The
// atmosphere raymarch consumes the same scaled flux so haze radiance
// stays in unit consistency with the lit surface.
const SCENE_FLUX_SCALE: f32 = 0.5;

// Hapke (2002) radiance factor. Inputs are cosines and the
// surface-roughness parameter that drives the opposition-surge width.
// Tuned empirically to match the lunar regolith comparison shots; the
// global `SCENE_FLUX_SCALE` constant absorbs the (1 / 4π) normalisation
// so callers can multiply flux in directly.
fn hapke_brdf(n_dot_l: f32, n_dot_v: f32, cos_phase: f32, roughness: f32) -> f32 {
    let mu0 = max(n_dot_l, 0.0);
    let mu  = max(n_dot_v, 0.0);
    if mu0 <= 0.0 || mu <= 0.0 { return 0.0; }

    // Single-scattering albedo (0..1). Lunar highlands ~0.4, mare ~0.2.
    // Picked to match the prior visual brightness when combined with the
    // 2π global scale below.
    let w: f32 = 0.45;

    let cp = clamp(cos_phase, -1.0, 1.0);
    let g  = acos(cp);

    // Shadow-hiding opposition effect: B(g) = B0 / (1 + tan(g/2)/h).
    // h tuned to lunar regolith (~3.4°) at roughness 0.85; widened or
    // narrowed by surface roughness so e.g. a smooth icy patch keeps a
    // sharper opposition spike than a rubble field.
    let B0: f32 = 1.0;
    let h = mix(0.04, 0.10, clamp(roughness, 0.0, 1.0));
    let B_g = B0 / (1.0 + tan(g * 0.5) / h);

    // Single-particle phase function: Henyey-Greenstein with asymmetry
    // g_hg = -0.3 (back-scatter, typical of rough regolith grains).
    let g_hg: f32 = -0.3;
    let denom = 1.0 + g_hg * g_hg - 2.0 * g_hg * cp;
    let P_g = (1.0 - g_hg * g_hg) / pow(max(denom, 1e-6), 1.5);

    // Chandrasekhar H-function (Hapke 2002 two-stream approximation).
    let gamma = sqrt(max(1.0 - w, 0.0));
    let H_mu0 = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * mu0 * gamma);
    let H_mu  = (1.0 + 2.0 * mu) / (1.0 + 2.0 * mu * gamma);

    // Full radiance factor. The `(1 / (4π))` normalization is folded into
    // a global scale at the call site so brightness matches the prior
    // Lambert pipeline without re-tuning every planet's `light_intensity`.
    let r = w * (mu0 / (mu0 + mu)) * ((1.0 + B_g) * P_g + H_mu0 * H_mu - 1.0);
    return max(r, 0.0);
}

// Compute direct-light radiance for one fragment on a planet surface.
//
// Combines the Hapke BRDF (sun + planetshine), eclipse-occluder shadow
// terms, a per-fragment ambient floor, and a perturbed-normal headroom
// clamp that prevents micro-detail normals from out-lighting the body's
// large-scale curvature near the terminator. Returns the *surface-side*
// radiance the camera observes before any atmosphere transmittance,
// cloud composite, or in-scatter is applied on top.
//
// Parameters:
//
//   `albedo`            base-colour radiance reflectance (linear).
//   `roughness`         Hapke opposition-surge / H-function input (0..1).
//   `shading_normal`    perturbed normal, world-space, unit length.
//   `geo_normal`        unperturbed geometric normal (sphere outward).
//   `view_dir`          surface → camera, unit length.
//   `hit_ws`            fragment world position (used for eclipse +
//                       planetshine geometry).
//   `sun_dir_ws`        unit vector from fragment toward primary star.
//   `sun_flux`          per-fragment flux already scaled by camera gain.
//   `scene`             eclipse occluders + planetshine parent.
//   `external_shadow`   product of caller-side shadow factors (crater
//                       shadow × self-shadow on the impostor; 1.0 on
//                       paths that don't carry those terms yet).
//
// Atmospheric scattering, cloud composite, water shading, and limb
// darkening intentionally live outside this function so the impostor
// can keep its inline atmosphere pass for the outside-shell regime and
// the unified `BodySky` fullscreen pass can handle the inside-shell
// regime without duplication.
fn shade_hapke_surface(
    albedo: vec3<f32>,
    roughness: f32,
    shading_normal: vec3<f32>,
    geo_normal: vec3<f32>,
    view_dir: vec3<f32>,
    hit_ws: vec3<f32>,
    sun_dir_ws: vec3<f32>,
    sun_flux: f32,
    scene: SceneLighting,
    external_shadow: f32,
) -> vec3<f32> {
    let geo_n_dot_l = dot(geo_normal, sun_dir_ws);
    let headroom = mix(0.05, 0.30, smoothstep(0.15, 0.40, geo_n_dot_l));
    let n_dot_v = max(dot(shading_normal, view_dir), 0.0);

    // Primary: direct sunlight. Clamp the perturbed n·l against the
    // geometric n·l + headroom so a steep micro-normal can't out-light
    // body curvature near the terminator (an artefact when the cubemap-
    // height normal is sharper than the LOD that produced it).
    let sun_n_dot_l_raw = dot(shading_normal, sun_dir_ws);
    let sun_n_dot_l = min(sun_n_dot_l_raw, geo_n_dot_l + headroom);
    let cos_phase_sun = dot(view_dir, sun_dir_ws);
    var sun_r = hapke_brdf(max(sun_n_dot_l, 0.0), n_dot_v, cos_phase_sun, roughness);
    sun_r = sun_r * external_shadow;
    sun_r = sun_r * eclipse_factor(scene, hit_ws, sun_dir_ws);

    // Secondary: planetshine. A separate Hapke evaluation against the
    // parent body's direction; eclipse / external-shadow terms don't
    // apply because the parent is the light, not the occluder.
    var shine_rgb = vec3<f32>(0.0);
    let shine = planetshine_sample(scene, hit_ws, sun_dir_ws, sun_flux);
    if shine.enabled {
        let shine_n_dot_l = dot(shading_normal, shine.dir);
        let shine_cos_phase = dot(view_dir, shine.dir);
        let shine_r = hapke_brdf(max(shine_n_dot_l, 0.0), n_dot_v, shine_cos_phase, roughness);
        shine_rgb = shine.tint * shine_r * shine.flux;
    }

    let sun_rgb = vec3<f32>(sun_r * sun_flux * SCENE_FLUX_SCALE);
    let ambient_term = vec3<f32>(scene.ambient_intensity);
    return albedo * (sun_rgb + shine_rgb + ambient_term);
}
