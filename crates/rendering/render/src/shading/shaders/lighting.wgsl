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

    // Moonlight onto this surface (reverse of planetshine — the brightest child
    // moon as one soft directional light). xyz = unit dir toward the moon
    // (world render space), w = artistic flux (phase × size × albedo × distance,
    // already night-lift-tuned). 0 flux disables.
    moonlight_dir_flux: vec4<f32>,
    // xyz = moon hue (normalised), w = enable flag (1 active).
    moonlight_color: vec4<f32>,
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
// material consumes — the impostor billboard and the thalos_udlod ground
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
    let geo_n_dot_v = dot(geo_normal, view_dir);
    let headroom = mix(0.05, 0.30, smoothstep(0.15, 0.40, geo_n_dot_l));
    // A height-derived micro-normal can point away from the camera at
    // grazing angles even though the geometric surface is visible. Keep
    // visibility anchored to the geometric normal so terrain does not fall
    // into black contour bands at low altitude.
    let view_visible_floor = select(0.0, max(geo_n_dot_v, 1.0e-3), geo_n_dot_v > 0.0);
    let n_dot_v = max(dot(shading_normal, view_dir), view_visible_floor);

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

// ── Surface incident sky lighting (vegetated dielectric path) ─────────────────
//
// A cheap, analytic hemisphere-skylight model for the wet / vegetated ground
// LOD (`body_terrain.wgsl`) and the grass that grows on it (`grass.wgsl`).
// Factoring it here is what keeps the two from drifting: both shaders call
// exactly these functions on exactly the same inputs.
//
// SCOPE: this is the *surface incident* lighting only — the direct sun reaching
// the ground (reddened by its own slant path through the air) plus the diffuse
// skylight and warm ground-bounce that fill faces the sun doesn't see. It does
// NOT produce camera-path haze / aerial perspective: that is the fullscreen
// `BodySky` pass (`body_sky.wgsl`), so the two never double-count. The sun→
// surface reddening here is a *different* optical path from BodySky's camera→
// surface haze, so reddening the beam is legitimate, not duplicated.
//
// Everything is expressed in the same flux units as `shade_hapke_surface`
// (`sun_flux * SCENE_FLUX_SCALE`), so the dielectric ground, the Hapke regolith,
// and the sky pass share one exposure scale and track `CameraExposure` together.

// Direct-sun reflectance scale. Converts one unit of scene radiance
// (`sun_flux * SCENE_FLUX_SCALE`) into surface direct lighting. At Thalos
// (flux ~10, focus gain ~1) the resulting `sun_scale` = 10 · 0.5 · 0.20 = 1.0;
// the old placeholder ignored flux and used a flat 0.62, so midday direct now
// lifts ~1.6× — the previous ground read too dark. This is the single knob for
// overall ground brightness; flux now carries exposure + sun-distance.
const SURFACE_DIRECT_SCALE: f32 = 0.23;
// Diffuse sky-dome (skylight) scale. Drives how strongly the blue sky fills
// shadowed / up-facing ground at midday. Second-loudest knob after direct.
const SURFACE_SKY_SCALE: f32 = 0.15;
// Sky chroma gain: how saturated the blue sky tint reads. The tint is
// `1 - exp(-tau_v · gain)`, so higher = more saturated blue (more Rayleigh
// out-scatter). Brightness is separate (`SURFACE_SKY_SCALE`).
const SURFACE_SKY_CHROMA_GAIN: f32 = 8.0;
// Warm ground-bounce: a representative sunlit-land albedo and its scale. Lights
// down-facing facets with a warm complement to the cool sky.
const SURFACE_GROUND_ALBEDO: vec3<f32> = vec3<f32>(0.10, 0.085, 0.055);
const SURFACE_GROUND_SCALE: f32 = 0.10;
// Sunset reddening of the direct beam: how hard the lengthening slant path eats
// blue as the sun drops. 1.0 = physical (per the column optical depth).
const SURFACE_SUN_REDDEN_GAIN: f32 = 1.0;
// Faint night floor so the dark side reads as *night*, not pure black. Cool,
// roughly matching the old `night_fill = 0.012` luminance.
const SURFACE_NIGHT_AMBIENT: vec3<f32> = vec3<f32>(0.008, 0.010, 0.014);

// Daylight gate over the *geometric* horizon, from the sun's elevation
// `dot(up, sun_dir)`. 1 = sun well up, 0 = sun below the horizon, with a soft
// twilight band on the dawn side so direct light fades instead of snapping to
// black. This is the single definition of the terminator band: `compute_surface_sky`
// gates the sky/ground fill with it, and `shade_foliage` gates the wrap-diffuse
// direct/transmit with it (the wrap intentionally bleeds past the *normal's*
// terminator, so it needs this *sun's* terminator to vanish at night).
fn sun_daylight(sun_elev: f32) -> f32 {
    return smoothstep(-0.15, 0.12, sun_elev);
}

// Resolved per-fragment lighting environment. Built once by
// `compute_surface_sky`, then consumed by the BRDF (direct) and
// `sky_ambient_irradiance` (ambient) so both shaders shade from one source.
struct SurfaceSky {
    // Direct-beam tint after its slant path through the atmosphere (≈ white at
    // noon, orange near sunset). Multiply the direct BRDF response by this.
    sun_color: vec3<f32>,
    // Direct-beam radiance scale in scene-flux units. The caller multiplies the
    // BRDF, the n·l cosine, and any shadow term by `sun_color * sun_scale`.
    sun_scale: f32,
    // Diffuse sky-dome radiance (blue), already elevation-scaled, plus the night
    // floor. Lights up-facing normals; also reflected as ambient specular.
    sky_radiance: vec3<f32>,
    // Warm ground-bounce radiance, plus the night floor. Lights down-facing
    // normals.
    ground_radiance: vec3<f32>,
}

// Derive the surface lighting environment from the *vertical* Rayleigh optical
// depth `tau_zenith` (= β_R · H_R = the authored τ_v; note this product is
// independent of the meters-per-render-unit scale), the artistic atmosphere
// `strength`, the local radial up, the sun direction, and the per-fragment sun
// flux. Pure analytic — no raymarch.
fn compute_surface_sky(
    tau_zenith: vec3<f32>,
    strength: f32,
    up: vec3<f32>,
    sun_dir: vec3<f32>,
    sun_flux: f32,
) -> SurfaceSky {
    var out: SurfaceSky;
    let scene_radiance = max(sun_flux, 0.0) * SCENE_FLUX_SCALE;
    let sun_elev = dot(up, sun_dir);
    // Daylight gate over the *geometric* horizon (soft twilight band — see
    // `sun_daylight`). Slightly wide on the dawn side so twilight keeps a dim band
    // rather than snapping to black.
    let day = sun_daylight(sun_elev);
    let sun_up = clamp(sun_elev, 0.0, 1.0);
    let tau_eff = max(tau_zenith, vec3<f32>(0.0)) * max(strength, 0.0);

    // Direct-beam reddening. The relative air mass along the sun ray grows from
    // ~1 at the zenith toward the horizon; subtracting 1 keeps a high sun white.
    // exp(-τ · (airmass−1)) eats blue first, leaving the orange sunset beam.
    let airmass = clamp(1.0 / (sun_up + 0.10), 1.0, 8.0);
    out.sun_color = exp(-tau_eff * (airmass - 1.0) * SURFACE_SUN_REDDEN_GAIN);
    out.sun_scale = scene_radiance * SURFACE_DIRECT_SCALE;

    // Blue sky-dome chroma from how much each wavelength scatters out of the
    // vertical column (more blue → bluer sky). Brightness tracks sun elevation
    // so shadows read blue by day, dim at sunset, and fall to the night floor.
    let sky_chroma = vec3<f32>(1.0) - exp(-tau_eff * SURFACE_SKY_CHROMA_GAIN);
    let sky_strength = scene_radiance * SURFACE_SKY_SCALE * day * (0.35 + 0.65 * sun_up);
    out.sky_radiance = sky_chroma * sky_strength + SURFACE_NIGHT_AMBIENT;

    // Warm ground bounce: representative sunlit land albedo, only while the
    // ground itself is lit.
    out.ground_radiance =
        SURFACE_GROUND_ALBEDO * scene_radiance * SURFACE_GROUND_SCALE * sun_up
        + SURFACE_NIGHT_AMBIENT;
    return out;
}

// Hemispheric ambient irradiance for a surface normal: blue sky on up-facing
// faces, warm ground-bounce on down-facing faces, blended by the normal's
// elevation over the local horizon. Multiply by albedo (and AO) for the ambient
// diffuse term.
fn sky_ambient_irradiance(sky: SurfaceSky, normal: vec3<f32>, up: vec3<f32>) -> vec3<f32> {
    let w_up = clamp(0.5 + 0.5 * dot(normal, up), 0.0, 1.0);
    return mix(sky.ground_radiance, sky.sky_radiance, w_up);
}

// Karis' mobile split-sum environment-BRDF approximation ("Physically Based
// Shading on Mobile", 2014). Returns the `(scale, bias)` pair such that the
// specular environment response for reflectance `F0` is `F0 * scale + bias`,
// and the white-furnace directional albedo (used for Kulla–Conty energy
// compensation) is `scale + bias`. One MAD-heavy evaluation, no LUT binding.
fn env_brdf_approx(roughness: f32, n_dot_v: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// ── Geometric specular antialiasing ──────────────────────────────────────────
// Sub-pixel variation in the shading normal makes a sharp GGX highlight sparkle
// and crawl as the camera moves — the dominant aliasing on relief-normal
// terrain, grass blades, and foliage, the kind an edge-AA pass (SMAA/MSAA) can't
// touch. Following Kaplanyan "Stable Specular Highlights" (2016) / the filament
// implementation, widen the roughness by the screen-space normal variance so the
// specular lobe covers the pixel's normal cone, trading a hair of gloss for a
// stable highlight.
//
// Split in two so the derivative builtins stay in uniform control flow: call
// `specular_aa_variance` once on the shading normal *before* any non-uniform
// branch, then feed its result to `specular_aa_apply` inside each BRDF path.

const SPECULAR_AA_VARIANCE: f32 = 0.25;   // screen-space normal-variance gain
const SPECULAR_AA_THRESHOLD: f32 = 0.18;  // clamp on the added kernel roughness

// Screen-space normal variance for world-space normal `n`. Uses `dpdx`/`dpdy`,
// so it MUST be evaluated in uniform control flow.
fn specular_aa_variance(n: vec3<f32>) -> f32 {
    let dndx = dpdx(n);
    let dndy = dpdy(n);
    return SPECULAR_AA_VARIANCE * (dot(dndx, dndx) + dot(dndy, dndy));
}

// Widen `perceptual_roughness` to cover the normal cone described by `variance`
// (from `specular_aa_variance`). Works in GGX α = roughness² space; pure, so it
// is safe to call inside any branch.
fn specular_aa_apply(perceptual_roughness: f32, variance: f32) -> f32 {
    let alpha = perceptual_roughness * perceptual_roughness;
    let kernel = min(2.0 * variance, SPECULAR_AA_THRESHOLD);
    let filtered_alpha = sqrt(clamp(alpha * alpha + kernel, 0.0, 1.0));
    return sqrt(filtered_alpha);
}

// ── Rough-dielectric surface BRDF ─────────────────────────────────────────────
// The shared physically-based BRDF for every non-regolith surface (vegetated
// ground, water, rock, ship hull). Two lobes: an Oren–Nayar rough-diffuse term
// that degrades gracefully at grazing angles (no opposition-surge contour bands)
// plus a Cook–Torrance GGX microfacet specular with a dielectric F0, so wet
// ground and snow pick up a tight highlight Hapke cannot express. Moved here from
// `body_terrain.wgsl` so every dielectric surface shades through one BRDF.

const PI_BRDF: f32 = 3.14159265358979323846;
// Non-metallic surface normal reflectance at normal incidence (~4%).
const DIELECTRIC_F0: f32 = 0.04;

// GGX / Trowbridge–Reitz normal distribution.
fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI_BRDF * d * d, 1.0e-7);
}

// Smith height-correlated visibility term for GGX, with the specular
// denominator 1/(4·n·l·n·v) folded in.
fn smith_visibility(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, 1.0e-5);
}

// Schlick Fresnel for a scalar (achromatic dielectric) F0.
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m2 = m * m;
    return f0 + (1.0 - f0) * (m2 * m2 * m);
}

// Oren–Nayar rough-diffuse BRDF scalar (sans albedo, sans cosine). Uses the
// trig-free `s/t` formulation so there is no `acos`/`tan`/`normalize`-of-zero
// hazard; `s = L·V − (N·L)(N·V)` reconstructs cos(Δφ)·sinθᵢ·sinθᵣ directly.
fn oren_nayar_term(
    n_dot_l: f32,
    n_dot_v: f32,
    l: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
) -> f32 {
    let sigma2 = roughness * roughness;
    let a = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
    let b = 0.45 * sigma2 / (sigma2 + 0.09);
    let s = dot(l, v) - n_dot_l * n_dot_v;
    let t = select(max(n_dot_l, n_dot_v), 1.0, s <= 0.0);
    return a + b * s / max(t, 1.0e-4);
}

// Combined rough-dielectric reflectance for one light direction. Returns the
// reflected radiance factor (diffuse albedo-tinted + white dielectric
// specular), excluding the irradiance cosine and incident flux, which the
// caller applies.
fn surface_brdf(
    albedo: vec3<f32>,
    roughness: f32,
    n: vec3<f32>,
    l: vec3<f32>,
    v: vec3<f32>,
    n_dot_l: f32,
    n_dot_v: f32,
) -> vec3<f32> {
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(l + v);
    let n_dot_h = max(dot(n, h), 0.0);
    let l_dot_h = max(dot(l, h), 0.0);

    let f = fresnel_schlick(l_dot_h, DIELECTRIC_F0);
    let d = ggx_distribution(n_dot_h, roughness);
    let vis = smith_visibility(n_dot_l, max(n_dot_v, 1.0e-4), roughness);

    // Kulla–Conty multiple-scattering energy compensation. Single-scattering
    // GGX drops the energy from microfacet rays that bounce more than once, so a
    // rough surface loses reflectance and reads muddy. The directional albedo
    // `E_ss = scale + bias` (white-furnace split-sum) tells us how much survives;
    // scaling the lobe by `1 + F0·(1/E_ss − 1)` puts the lost energy back. For a
    // dielectric (F0 ≈ 0.04) the factor is small, but it keeps wet ground / snow
    // from dimming at grazing angles and is the correct thing to do.
    let dfg = env_brdf_approx(roughness, max(n_dot_v, 1.0e-4));
    let e_ss = max(dfg.x + dfg.y, 1.0e-3);
    let ms = 1.0 + DIELECTRIC_F0 * (1.0 / e_ss - 1.0);
    let spec = d * vis * f * ms;

    let diff = oren_nayar_term(n_dot_l, n_dot_v, l, v, roughness);
    let diffuse = albedo * diff * (1.0 - f);

    return diffuse + vec3<f32>(spec);
}

// ── The canonical surface-shading entry point ─────────────────────────────────
// Every ship-view surface fills a `ThalosSurface` and calls `shade_surface`,
// which dispatches on `style` to the matching BRDF and returns the surface-side
// radiance the camera observes BEFORE atmosphere / aerial-perspective is
// composited on top (that lives in the `BodySky` fullscreen pass, keyed on scene
// depth — same split `shade_hapke_surface` documents). Stylization is dialed in
// on the inputs the caller writes into `ThalosSurface`, never in the BRDF.

const SURFACE_DIELECTRIC: u32 = 0u; // vegetated ground, rock, ship hull, water-ish
const SURFACE_REGOLITH: u32 = 1u;   // airless ground + impostor (Hapke)
const SURFACE_FOLIAGE: u32 = 2u;    // grass, leaves (wrap-diffuse + translucency)
const SURFACE_WATER: u32 = 3u;      // low-roughness dielectric + sky reflection

// Per-fragment material description shared by every surface material.
struct ThalosSurface {
    albedo: vec3<f32>,        // linear diffuse reflectance
    roughness: f32,           // perceptual GGX roughness (caller pre-clamps / AA-widens)
    normal_ws: vec3<f32>,     // shading normal, world render space, unit
    geo_normal_ws: vec3<f32>, // geometric (sphere-outward) normal — terminator anchor
    emissive: vec3<f32>,      // self-emission, pre-exposure (engines etc.)
    occlusion: f32,           // [0,1] ambient occlusion applied to ambient terms
    metallic: f32,            // 0 dielectric … 1 metal (reserved; ships)
    translucency: f32,        // foliage two-sided lobe weight (reserved; 0 opaque)
    style: u32,               // SURFACE_DIELECTRIC | _REGOLITH | _FOLIAGE | _WATER
}

// Soft directional moonlight onto a surface fragment — the reverse of
// planetshine (a child moon reflecting the star back onto its parent). A plain
// Lambert lobe (moonlight is soft, no need for a microfacet model), gated so it
// only appears at NIGHT (sun below the local horizon) and only where the moon is
// itself above the local horizon. `scene.moonlight_dir_flux.w` already carries
// the phase/size/albedo/distance-weighted artistic flux, so this is just the
// cosine term × the two gates. Returns zero when disabled.
fn moonlight_radiance(
    scene: SceneLighting,
    albedo: vec3<f32>,
    normal_ws: vec3<f32>,
    geo_normal_ws: vec3<f32>,
    sun_dir_ws: vec3<f32>,
) -> vec3<f32> {
    if (scene.moonlight_color.w < 0.5) {
        return vec3<f32>(0.0);
    }
    let moon_flux = scene.moonlight_dir_flux.w;
    if (moon_flux <= 0.0) {
        return vec3<f32>(0.0);
    }
    let moon_dir = scene.moonlight_dir_flux.xyz;
    // Night gate: fade out wherever the sun lights this facet's horizon, so
    // moonlight never brightens the day side (where it is invisible anyway).
    let night = 1.0 - smoothstep(-0.10, 0.06, dot(geo_normal_ws, sun_dir_ws));
    // The moon must be above the local horizon to cast light here.
    let moon_up = smoothstep(-0.03, 0.06, dot(geo_normal_ws, moon_dir));
    let n_dot_m = max(dot(normal_ws, moon_dir), 0.0);
    return albedo * scene.moonlight_color.xyz * (moon_flux * n_dot_m * night * moon_up);
}

// Surface-side radiance only. `direct_shadow` gates the direct sun term (craft ×
// self × cascade); `ambient_shadow` gates the ambient terms (canopy bleed). The
// resolved `SurfaceSky` carries the per-fragment direct/sky/ground environment
// for the dielectric path; the regolith path folds its ambient into the Hapke
// helper and ignores `sky` / `ambient_shadow`.
fn shade_surface(
    s: ThalosSurface,
    view_dir_ws: vec3<f32>,
    hit_ws: vec3<f32>,
    sun_dir_ws: vec3<f32>,
    sun_flux: f32,
    scene: SceneLighting,
    sky: SurfaceSky,
    direct_shadow: f32,
    ambient_shadow: f32,
) -> vec3<f32> {
    if (s.style == SURFACE_REGOLITH) {
        return shade_hapke_surface(
            s.albedo,
            s.roughness,
            s.normal_ws,
            s.geo_normal_ws,
            view_dir_ws,
            hit_ws,
            sun_dir_ws,
            sun_flux,
            scene,
            direct_shadow,
        ) + s.emissive
            + moonlight_radiance(scene, s.albedo, s.normal_ws, s.geo_normal_ws, sun_dir_ws);
    }

    // Rough-dielectric (foliage/water fall through here for now): direct sun
    // (BRDF × cosine × shadow, tinted by the reddened beam, scaled into the
    // shared scene-flux exposure) + hemisphere sky IBL (blue sky-dome + warm
    // ground bounce) + a subtle split-sum sky specular.
    let n_dot_l = max(dot(s.normal_ws, sun_dir_ws), 0.0);
    let n_dot_v = max(dot(s.normal_ws, view_dir_ws), 1.0e-4);

    let brdf = surface_brdf(
        s.albedo,
        s.roughness,
        s.normal_ws,
        sun_dir_ws,
        view_dir_ws,
        n_dot_l,
        n_dot_v,
    );
    let direct = brdf * (n_dot_l * direct_shadow) * sky.sun_color * sky.sun_scale;

    // Specular occlusion: smooth (wet/snow) surfaces keep more of their sky
    // reflection out of creases than rough matte ground does.
    let spec_occ = clamp(s.occlusion + (1.0 - s.roughness) * 0.4, 0.0, 1.0);

    // Hemispheric sky ambient (diffuse): blue sky on up-facing normals, warm
    // ground bounce on down-facing, gated by AO and the canopy ambient shadow.
    let ambient_irr = sky_ambient_irradiance(sky, s.normal_ws, s.geo_normal_ws);
    let ambient_diffuse = s.albedo * ambient_irr * s.occlusion * ambient_shadow;

    // Ambient sky specular: split-sum environment reflection of the sky dome.
    let dfg = env_brdf_approx(s.roughness, n_dot_v);
    let env_spec = dfg.x * DIELECTRIC_F0 + dfg.y;
    let ambient_spec = sky.sky_radiance * env_spec * spec_occ * ambient_shadow;

    // Statistical foliage-canopy transmit: a surface standing in for a grass /
    // leaf layer (terrain grassland past the blade clipmap sets `translucency`
    // per fragment from its grass mask) gets the same warm backlit lobe
    // `shade_foliage` gives real blades, so the geometry→shading handoff keeps
    // its low-sun rim. Zero-cost for every opaque caller (all pass 0).
    var transmit = vec3<f32>(0.0);
    if (s.translucency > 0.0) {
        let daylight = sun_daylight(dot(s.geo_normal_ws, sun_dir_ws));
        let lt_dir = normalize(sun_dir_ws + s.normal_ws * 0.30);
        let back = pow(clamp(dot(view_dir_ws, -lt_dir), 0.0, 1.0), 2.5);
        let warm = vec3<f32>(1.30, 1.05, 0.50); // green → yellow/orange shift (shade_foliage's)
        transmit = s.albedo * warm
            * (back * s.translucency * sky.sun_scale * direct_shadow * daylight)
            * sky.sun_color;
    }

    let moon = moonlight_radiance(scene, s.albedo, s.normal_ws, s.geo_normal_ws, sun_dir_ws);
    return direct + ambient_diffuse + ambient_spec + transmit + s.emissive + moon;
}

// ── Foliage wrap-diffuse shading ──────────────────────────────────────────────
// The one lighting routine for grass blades, mesh trees, and tree impostors, so
// they read consistently and the mesh→impostor handoff stays photometrically
// continuous. Wrap diffuse (foliage scatters past the terminator) + the shared
// hemisphere sky IBL + an optional two-sided translucency for backlit leaves. No
// specular — foliage is matte. `translucency` / `ambient_scale` / `ambient_bleed`
// carry the few legitimate per-type differences: grass matches the ground fill;
// canopies dim their ambient and bleed shadow into it; only leaves transmit.

const FOLIAGE_WRAP_BIAS: f32 = 0.40;

struct FoliageSurface {
    albedo: vec3<f32>,
    normal_ws: vec3<f32>,
    translucency: f32,  // 0 = opaque blade/bark, 1 = leaf (two-sided transmit)
    ambient_scale: f32, // hemisphere fill scale (grass 1.0 ground-match; canopy 0.8)
    ambient_bleed: f32, // shadow → ambient bleed (0 grass, ~0.5 canopy)
}

fn shade_foliage(
    s: FoliageSurface,
    view_dir_ws: vec3<f32>,
    up: vec3<f32>,
    sun_dir_ws: vec3<f32>,
    sky: SurfaceSky,
    shadow: f32,
) -> vec3<f32> {
    let n = s.normal_ws;
    let n_dot_l = dot(n, sun_dir_ws);

    // Daylight gate on the DIRECT terms. The wrap-diffuse below bleeds light past
    // the leaf-normal terminator (any n·l > −FOLIAGE_WRAP_BIAS), which is right
    // while the sun is up but would keep leaves facing the buried sun lit at night
    // — the ground avoids this because its `max(n·l, 0)` cosine zeroes once the sun
    // drops below the horizon, but the wrap has no such floor. `sun_scale` itself
    // isn't elevation-gated, so gate the wrap by the sun's own horizon here (the
    // ambient terms already fall to `SURFACE_NIGHT_AMBIENT` via `compute_surface_sky`).
    let daylight = sun_daylight(dot(up, sun_dir_ws));

    // Direct: wrap diffuse so the shaded side stays leafy rather than black.
    let wrap = clamp((n_dot_l + FOLIAGE_WRAP_BIAS) / (1.0 + FOLIAGE_WRAP_BIAS), 0.0, 1.0);
    let direct = s.albedo * (wrap * sky.sun_scale * shadow * daylight) * sky.sun_color;

    // Ambient: hemisphere sky-dome + warm ground bounce, scaled per type and bled
    // by the shadow (a shaded canopy sees less sky too).
    let ambient = s.albedo * sky_ambient_irradiance(sky, n, up)
        * (s.ambient_scale * mix(1.0, shadow, s.ambient_bleed));

    // Two-sided translucency: backlit leaves transmit a warm forward-scattered
    // glow — a view-dependent lobe (looking toward the sun through the leaf) plus
    // a softer isotropic through-scatter.
    var transmit = vec3<f32>(0.0);
    if (s.translucency > 0.0) {
        let lt_dir = normalize(sun_dir_ws + n * 0.30);
        let back = pow(clamp(dot(view_dir_ws, -lt_dir), 0.0, 1.0), 2.5);
        let warm = vec3<f32>(1.30, 1.05, 0.50); // green → yellow/orange shift
        let thru = (back + 0.16 * clamp(-n_dot_l, 0.0, 1.0)) * s.translucency;
        transmit = s.albedo * warm * (thru * sky.sun_scale * shadow * daylight) * sky.sun_color;
    }

    return direct + ambient + transmit;
}

// ── Object aerial recession ───────────────────────────────────────────────────
// Surface objects (foliage today; buildings next) read more saturated and
// higher-contrast than the ground, so at the SAME camera distance they pop
// against terrain that the fullscreen `BodySky` pass (`body_sky.wgsl`) has
// already hazed. BodySky can't fix this — it is keyed on scene depth alone and
// can't tell an object pixel from a terrain pixel — so each object material
// recedes its OWN lit colour toward the local air colour HERE, starting CLOSER
// than BodySky's ~8 km terrain-aerial onset. The target is the sky-dome radiance
// the same `SurfaceSky` already carries, so the recession tracks time-of-day and
// converges toward the same air the terrain fades into. It is deliberately
// earlier/stronger than the terrain veil: a distant forest should read as a flat
// blue-grey mass ("blue ridge"), not crisp green cut-outs. Below
// `OBJECT_AERIAL_NEAR_M` it is a no-op, so the foreground stays crisp.
//
// These are the tuning dials — adjust from a `just game` surface screenshot. (The
// headless `just preview` camera sits well inside NEAR, so it shows no effect —
// that's the regression check: close-up objects must look unchanged.)
// The ramp is spread over a LONG distance (1 → 35 km) on purpose: a short ramp
// concentrates the whole fade into a narrow transition band (the smoothstep's
// steep middle) that reads as an abrupt "haze line". Stretching it makes the
// recession build up gradually from near to far, with no band, and keeps the
// far trees near the terrain's own gentle veil instead of racing past it.
const OBJECT_AERIAL_NEAR_M: f32 = 1000.0;  // recession begins (crisp below this)
const OBJECT_AERIAL_FAR_M: f32  = 35000.0; // strongest veil reached by here (well past
                                           // the tree band, so it never plateaus into
                                           // a saturated horizon band)
const OBJECT_AERIAL_MAX: f32    = 0.32;    // max blend toward air (low: only a touch
                                           // more than the terrain veil)
// The analytic sky-dome radiance is several × brighter than lit foliage, so
// fading straight toward it BRIGHTENS distant canopies to white instead of
// hazing them. Cap the haze target to a small multiple of the object's own
// luminance so the fade reads as HAZE (desaturate + gentle bluish lift), matched
// to how the terrain recedes — not a white blow-out.
const OBJECT_AERIAL_BRIGHTEN_CAP: f32 = 1.5;

fn object_aerial_recession(
    color: vec3<f32>,
    sky: SurfaceSky,
    world_pos: vec3<f32>,
    cam_pos: vec3<f32>,
) -> vec3<f32> {
    let dist = distance(cam_pos, world_pos);
    let t = smoothstep(OBJECT_AERIAL_NEAR_M, OBJECT_AERIAL_FAR_M, dist) * OBJECT_AERIAL_MAX;
    if (t <= 0.0) {
        return color;
    }
    let lum_w = vec3<f32>(0.2126, 0.7152, 0.0722);
    let obj_lum = dot(color, lum_w);
    // Air the object recedes into: the bluish sky-dome radiance the terrain also
    // fades toward, but clamped so it can't get much brighter than the object —
    // the analytic radiance outruns foliage brightness, and an over-bright target
    // is exactly the white-out. At night `sky_radiance` falls to the cool floor,
    // so distant objects correctly dim rather than fade toward a daytime blue.
    let haze = sky.sky_radiance;
    let haze_lum = max(dot(haze, lum_w), 1.0e-4);
    let cap = (obj_lum + 0.02) * OBJECT_AERIAL_BRIGHTEN_CAP;
    let haze_capped = haze * min(1.0, cap / haze_lum);
    return mix(color, haze_capped, t);
}
