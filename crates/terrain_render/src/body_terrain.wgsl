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

// ── Naturalistic surface grading (Step 0 + Step 1) ────────────────────────
// Applied to the baked macro albedo in-shader, so it iterates without a
// re-bake and stays resolution-independent. The bake stays the low-frequency
// layer (which biome / broad climate); these knobs add the slope response and
// take the harsh edge off the vegetation palette. Higher-frequency albedo
// breakup and detail normals (Steps 2-3) land on top of this later.

// Step 0 — de-harsh the palette. The baked vegetation anchors read as a
// grating, over-bright chartreuse; pull them toward luminance and trim value.
// Saturation is back up from the first pass (which over-bleached into a pale,
// "ghostly" wash): the per-pixel variation now comes from the Step 2/3 detail
// layer below, not from flattening the colour, so the macro tint can keep its
// hue. Value is only lightly trimmed — the detail normal supplies the dark
// micro-contrast that grounds the surface.
const SURFACE_SATURATION: f32 = 0.78; // <1 desaturates toward grey
const SURFACE_VALUE_GAIN: f32  = 0.82; // slight overall value trim

// Step 1 — slope-driven substrate. Local steepness is the deviation of the
// terrain normal from the radial (geometric) normal: ~0 on ground that lies
// flat against the sphere, rising toward 1 on cliffs. Gentle slopes expose
// soil, steep faces expose rock. Thresholds are kept high and strengths
// modest so thin risers between coarse height texels don't trace grey contour
// ribbons across otherwise flat ground.
const SUBSTRATE_SOIL: vec3<f32> = vec3<f32>(0.160, 0.120, 0.080);
const SUBSTRATE_ROCK: vec3<f32> = vec3<f32>(0.190, 0.180, 0.160);
const SLOPE_SOIL_LO: f32 = 0.05;
const SLOPE_SOIL_HI: f32 = 0.30;
const SLOPE_ROCK_LO: f32 = 0.32;
const SLOPE_ROCK_HI: f32 = 0.70;
const SOIL_STRENGTH: f32 = 0.55;
const ROCK_STRENGTH: f32 = 0.65;

// Step 2 — albedo breakup. Multi-octave value noise modulating the macro
// colour's value (plus a faint warm/cool hue drift) at the tens-of-metres
// scale, so the ground stops reading as one flat tint.
const BREAKUP_SCALE: f32 = 0.05;     // 1/period_m → ~20 m base patches
const BREAKUP_VALUE_AMT: f32 = 0.16; // ± fractional value variation
const BREAKUP_HUE_AMT: f32 = 0.06;   // ± warm/cool drift

// Step 3 — micro-relief normal. A detail height field whose gradient tilts the
// lighting normal, giving the surface the light/dark micro-contrast under a
// grazing sun that separates "solid ground" from "luminous fog".
const DETAIL_SCALE: f32 = 0.8;            // 1/period_m → ~1.25 m base relief
const DETAIL_OCTAVES: i32 = 3;
const DETAIL_EPS: f32 = 0.25;             // finite-difference step, metres
const DETAIL_NORMAL_STRENGTH: f32 = 0.5;  // facet-tilt amount

// Both detail layers fade out with camera distance: their period goes
// sub-pixel on the far field and would shimmer. The macro/slope colour carries
// the distance instead.
const DETAIL_FADE_NEAR: f32 = 80.0;  // full detail within this range (m)
const DETAIL_FADE_FAR: f32  = 600.0; // no detail beyond this range (m)

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

// Pull a linear-RGB colour toward its luminance. `sat` < 1 desaturates,
// `sat` == 1 is a no-op. Rec.709 luminance weights.
fn desaturate(c: vec3<f32>, sat: f32) -> vec3<f32> {
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(vec3<f32>(luma), c, sat);
}

// Local steepness in [0,1]: 0 where the terrain normal aligns with the
// radial (sphere-outward) normal, rising as the surface tilts toward
// vertical. Coarse because the height normal is coarse — fine for *colouring*
// by slope, which is why it's kept separate from the lighting normal.
fn surface_slope(terrain_normal: vec3<f32>, radial_normal: vec3<f32>) -> f32 {
    return clamp(1.0 - dot(normalize(terrain_normal), radial_normal), 0.0, 1.0);
}

// Step 0 + Step 1: de-harsh the baked vegetation colour, then blend exposed
// soil/rock by local steepness. Operates on the (already tile-blended) macro
// albedo; higher-frequency detail composites on top downstream.
fn grade_surface(base: vec3<f32>, slope_t: f32) -> vec3<f32> {
    var c = desaturate(base, SURFACE_SATURATION) * SURFACE_VALUE_GAIN;
    let soil_t = smoothstep(SLOPE_SOIL_LO, SLOPE_SOIL_HI, slope_t) * SOIL_STRENGTH;
    c = mix(c, SUBSTRATE_SOIL, soil_t);
    let rock_t = smoothstep(SLOPE_ROCK_LO, SLOPE_ROCK_HI, slope_t) * ROCK_STRENGTH;
    c = mix(c, SUBSTRATE_ROCK, rock_t);
    return c;
}

// ── Procedural surface detail (Step 2 + Step 3) ───────────────────────────
// Cheap hash-based value noise / fBm, evaluated on the camera-relative world
// position (sharpest underfoot thanks to floating-origin precision). The bake
// supplies macro colour; this synthesises the metre-scale breakup and
// micro-relief the ~1 km/texel atlas cannot carry up close.

fn hash13(p_in: vec3<f32>) -> f32 {
    var p3 = fract(p_in * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

fn value_noise_3d(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

fn fbm3(p_in: vec3<f32>, octaves: i32) -> f32 {
    var p = p_in;
    var amp = 0.5;
    var sum = 0.0;
    var norm = 0.0;
    for (var o = 0; o < octaves; o = o + 1) {
        sum = sum + amp * value_noise_3d(p);
        norm = norm + amp;
        p = p * 2.0;
        amp = amp * 0.5;
    }
    return sum / max(norm, 1.0e-5);
}

fn detail_height(p_ws: vec3<f32>) -> f32 {
    return fbm3(p_ws * DETAIL_SCALE, DETAIL_OCTAVES);
}

struct SurfaceDetail {
    tint: vec3<f32>,          // multiplicative albedo breakup, ~1.0
    normal_offset: vec3<f32>, // tangential perturbation for the lighting normal
}

fn surface_detail(p_ws: vec3<f32>, geo_normal: vec3<f32>, cam_dist: f32) -> SurfaceDetail {
    var out: SurfaceDetail;
    out.tint = vec3<f32>(1.0);
    out.normal_offset = vec3<f32>(0.0);

    // Far field pays nothing: no derivatives here, so the early-out is safe.
    let fade = 1.0 - smoothstep(DETAIL_FADE_NEAR, DETAIL_FADE_FAR, cam_dist);
    if (fade <= 0.0) {
        return out;
    }

    // Step 2 — albedo breakup: patchy value variation + a subtle warm/cool
    // hue drift, so the macro colour stops reading as one flat tint.
    let v = fbm3(p_ws * BREAKUP_SCALE, 3);
    let dv = (v - 0.5) * 2.0;
    let value_mul = 1.0 + dv * BREAKUP_VALUE_AMT;
    let hue = vec3<f32>(1.0 + dv * BREAKUP_HUE_AMT, 1.0, 1.0 - dv * BREAKUP_HUE_AMT);
    out.tint = mix(vec3<f32>(1.0), vec3<f32>(value_mul) * hue, fade);

    // Step 3 — micro-relief normal from the gradient of a detail height field,
    // projected tangent to the sphere so it tilts facets without changing the
    // macro surface orientation. Magnitude is capped so a steep noise gradient
    // can't fold the normal past the horizon.
    let e = DETAIL_EPS;
    let h  = detail_height(p_ws);
    let hx = detail_height(p_ws + vec3<f32>(e, 0.0, 0.0));
    let hy = detail_height(p_ws + vec3<f32>(0.0, e, 0.0));
    let hz = detail_height(p_ws + vec3<f32>(0.0, 0.0, e));
    let grad = (vec3<f32>(hx, hy, hz) - vec3<f32>(h)) / e;
    let grad_t = grad - geo_normal * dot(grad, geo_normal);
    var off = -grad_t * (DETAIL_NORMAL_STRENGTH * fade);
    let off_len = length(off);
    out.normal_offset = off * (0.8 / max(0.8, off_len));
    return out;
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

    // Geometry shared by grading and lighting.
    let geo_normal = normalize(info.world_normal);
    let hit_ws = info.world_position.xyz;
    let cam_dist = length(info.view_vector);
    let debug_on = terrain_debug.params.x >= 0.5;

    // Procedural surface detail (Step 2 breakup + Step 3 micro-relief normal),
    // synthesised from the camera-relative world position.
    let detail = surface_detail(hit_ws, geo_normal, cam_dist);

    // Naturalistic in-shader grading of the baked macro albedo. The coarse
    // height normal feeds slope colour only (never lighting), so the macro
    // texel kinks that forced the low HEIGHT_NORMAL_WEIGHT don't reappear as
    // shading artefacts; Step 2's breakup multiplies on top. All of this runs
    // before the debug overlay so the checkerboard still overrides it cleanly.
    let slope_t = surface_slope(height_normal, geo_normal);
    var surface_rgb = grade_surface(albedo.rgb, slope_t);
    if (!debug_on) {
        surface_rgb = surface_rgb * detail.tint;
    }
    albedo = vec4<f32>(surface_rgb, albedo.a);

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

    // Lighting normal: weak coarse height relief + Step 3 procedural micro-
    // relief. The detail offset is tangent to the sphere, so it tilts facets
    // toward/away from the low sun and supplies the light/dark micro-contrast
    // that reads as solid ground rather than luminous fog.
    let height_n = normalize(mix(geo_normal, normalize(height_normal), HEIGHT_NORMAL_WEIGHT));
    var normal = height_n;
    if (!debug_on) {
        normal = normalize(height_n + detail.normal_offset);
    }
    let view_dir = normalize(info.view_vector);

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
