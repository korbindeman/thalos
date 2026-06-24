// Ground-LOD terrain shader for procedural bodies.
//
// Reads height, albedo, and roughness from the thalos_udlod attachment
// atlases (group 1), ray-tests a local craft-shadow proxy, and shades by a
// per-body surface style (see `TerrainShadingStyle` in body_material.rs,
// carried in `terrain_extras.inspection.y`):
//
//   * Vegetated (Thalos): rough-dielectric BRDF (Oren–Nayar diffuse +
//     Cook–Torrance GGX specular; see the BRDF block below) over the
//     ecological-band albedo model — a wet, vegetated terrestrial body.
//   * Regolith (Mira and other airless impact moons): the baked gray albedo
//     shaded by the Hapke radiative-transfer model via `shade_hapke_surface`,
//     the SAME routine `planet_impostor.wgsl` uses, so the ground LOD and the
//     orbital impostor reconverge across the LOD swap.
//
// Atmospheric scattering is NOT applied here — it is composited on top
// by `BodySky` (the fullscreen pass in `body_sky.wgsl`) whenever this
// ground terrain LOD is visible. Outside the terrain handoff distance,
// the impostor handles the body and its own inline atmosphere/cloud path.

#import thalos_udlod::types::AtlasTile
#import thalos_udlod::bindings::{config, atlas_sampler, attachments, attachment2_atlas, attachment3_atlas}
#import thalos_udlod::attachments::{sample_attachment1, sample_normal, attachment_uv, sample_height_atlas_uv_m, sample_height}
#import thalos_udlod::fragment::{FragmentInput, FragmentOutput, fragment_info}
#import thalos_udlod::functions::{lookup_tile, tile_count}
#import thalos::atmosphere::AtmosphereBlock
#import thalos::lighting::{SceneLighting, SCENE_FLUX_SCALE, shade_hapke_surface}

// Must match `MAX_TERRAIN_SHADOW_CASTERS` / `MAX_TERRAIN_SHADOW_QUADS` in
// `body_material.rs`.
const MAX_TERRAIN_SHADOW_CASTERS: u32 = 24u;
const MAX_TERRAIN_SHADOW_QUADS: u32 = 8u;

// Penumbra growth per metre of caster height: the star's angular diameter
// (~0.6 deg), so contact shadows are crisp and high shadows soften out the
// way a real sun shadow does.
const SHADOW_PENUMBRA_PER_M: f32 = 0.011;

struct BodyTerrainShadow {
    // x = strength, y = minimum penumbra width in metres,
    // z = max receiver distance, w = valid capsule caster count.
    params: vec4<f32>,
    // x = valid quad caster count, yzw reserved.
    quad_params: vec4<f32>,
    // xyz = part top/near endpoint in render-space metres, w = endpoint radius.
    caster_a_radius: array<vec4<f32>, 24>,
    // xyz = part bottom/far endpoint in render-space metres, w = endpoint radius.
    caster_b_radius: array<vec4<f32>, 24>,
    // Thin planform quads (lifting surfaces), corners in render-space metres
    // wound root-LE -> tip-LE -> tip-TE -> root-TE; w unused.
    quad_a: array<vec4<f32>, 8>,
    quad_b: array<vec4<f32>, 8>,
    quad_c: array<vec4<f32>, 8>,
    quad_d: array<vec4<f32>, 8>,
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
//
// Slot 2 packs craft shadow + debug + inspection into one buffer; see
// `BodyTerrainExtras` in `body_material.rs` for the slot-budget rationale
// (Metal vertex stage caps at 16 buffers and AsBindGroup forces vertex
// visibility on every `#[uniform(N)]`).
struct BodyTerrainExtras {
    craft_shadow: BodyTerrainShadow,
    debug: BodyTerrainDebug,
    inspection: vec4<f32>,
}

@group(3) @binding(0) var<uniform> terrain_atmos: AtmosphereBlock;
@group(3) @binding(1) var<uniform> terrain_scene: SceneLighting;
@group(3) @binding(2) var<uniform> terrain_extras: BodyTerrainExtras;

// Blend the atlas-derived macro height normal into the smooth geometric normal.
// Height is sampled through decode-then-filter RG16 interpolation in
// thalos_udlod::attachments, so this no longer turns packed-height residual
// wrap points into contour bands.
const HEIGHT_NORMAL_WEIGHT: f32 = 0.35;

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
const SURFACE_SATURATION: f32 = 0.96; // keep the grass/peat palette earthy, not neon
const SURFACE_VALUE_GAIN: f32  = 0.86; // bring ground luminance back under the sky

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
const SOIL_STRENGTH: f32 = 0.0;
const ROCK_STRENGTH: f32 = 0.0;

// Step 2 — albedo breakup. Multi-octave value noise modulating the macro
// colour's value (plus a faint warm/cool hue drift) at the tens-of-metres
// scale, so the ground stops reading as one flat tint.
const BREAKUP_SCALE: f32 = 0.05;     // 1/period_m → ~20 m base patches
const BREAKUP_VALUE_AMT: f32 = 0.18; // ± fractional value variation
const BREAKUP_HUE_AMT: f32 = 0.04;   // ± warm/cool drift

// Regolith fine detail. Airless particulate regolith reads as a fairly uniform
// gray at the macro level (the baked albedo is the body's mare/highland tone),
// but up close the Moon is densely textured: broad mare/highland mottle, a
// dense speckle of small craters / micro-ejecta (bright fresh spots, dark
// hollows), and a pocked micro-relief that catches the low sun. Without that
// the ground reads as a flat, over-bright plain. All value-only (regolith has
// negligible hue variation). Strengths are kept moderate so the value-noise
// micro-normal doesn't reintroduce the axis-aligned "weave".
const REGOLITH_MOTTLE_SCALE: f32 = 0.02;     // 1/period_m → ~50 m broad patches
const REGOLITH_MOTTLE_AMT: f32 = 0.16;       // ± broad value mottle
const REGOLITH_SPECKLE_SCALE: f32 = 0.55;    // 1/period_m → ~1.8 m dust/ejecta
const REGOLITH_SPECKLE_AMT: f32 = 0.20;      // ± fine value speckle
const REGOLITH_NORMAL_STRENGTH: f32 = 0.45;  // micro-relief facet tilt
// Slight overall darkening: the baked highland albedo plus Hapke backscatter
// at near-opposition reads a touch bright on flat ground; trimming value gives
// the dusty mid-gray the reference Apollo surface shots have.
const REGOLITH_VALUE_TRIM: f32 = 0.90;

// ── Altitude + slope material model (snow, treeline, scree) ───────────────
// Realistic terrain reads as ecological bands stacked by elevation: lush
// lowland → temperate grass → dry alpine grass above the treeline → bare scree
// → snow on the gentle summits, with steep faces staying bare rock at every
// altitude. We synthesise that here in the fragment so it iterates without a
// re-bake. Heights are absolute metres above the body reference sphere (the
// decoded height attachment); these thresholds are tuned to Thalos's land
// envelope (basins ~1.4 km, plains ~2.4 km, peaks well past 5 km) and are the
// primary knobs to nudge after a preview. A low-frequency noise jitters the
// treeline/snowline so the bands don't read as clean contour rings, and the
// snow gate excludes steep faces (snow sloughs off cliffs) — both standard
// altitude/slope/noise snow-line practice (see sources in the summary).
const LUSH_LO_M: f32 = 1800.0;       // full lowland lushness at/below here
const LUSH_HI_M: f32 = 2900.0;       // lushness gone above here
const TREELINE_LO_M: f32 = 3100.0;   // grass starts giving way to dry alpine
const TREELINE_HI_M: f32 = 4000.0;   // fully dry alpine grass / tundra
const SNOW_LINE_LO_M: f32 = 3700.0;  // snow begins (on gentle ground)
const SNOW_LINE_HI_M: f32 = 4600.0;  // permanent snow cap
const SNOW_LINE_NOISE_M: f32 = 400.0; // ± snow/treeline jitter from macro noise
const SNOW_SLOPE_LO: f32 = 0.32;     // snow holds on slopes up to ~47°…
const SNOW_SLOPE_HI: f32 = 0.62;     // …gone by ~68° (cliffs stay bare rock)
const MACRO_VAR_SCALE: f32 = 0.004;  // 1/period_m → ~250 m biome-mottle patches
const MACRO_VAR_AMT: f32 = 0.10;     // ± low-frequency value mottle

// Earthy linear-space material anchors. Deliberately lower-value and less
// saturated than neon grass; direct sun and sky fill lift them in the lighting
// pass. Snow is kept well below 1.0 so a sunlit cap doesn't blow out.
const C_FOREST: vec3<f32>   = vec3<f32>(0.040, 0.066, 0.030); // lowland lush (dark green)
const C_GRASS: vec3<f32>    = vec3<f32>(0.078, 0.112, 0.052); // temperate grass (olive)
const C_DRYGRASS: vec3<f32> = vec3<f32>(0.130, 0.132, 0.074); // alpine dry grass / tundra
const C_SOIL: vec3<f32>     = vec3<f32>(0.092, 0.068, 0.044); // earthy soil / peat
const C_ROCK_LO: vec3<f32>  = vec3<f32>(0.106, 0.094, 0.080); // lower rock (warm grey-brown)
const C_ROCK_HI: vec3<f32>  = vec3<f32>(0.140, 0.138, 0.132); // alpine scree (cool grey)
const C_SNOW: vec3<f32>     = vec3<f32>(0.600, 0.640, 0.720); // snow (faint blue)
const C_WET: vec3<f32>      = vec3<f32>(0.030, 0.050, 0.028); // wet hollow (dark)

// Step 3 — micro-relief normal. A detail height field whose gradient tilts the
// lighting normal, giving the surface the light/dark micro-contrast under a
// grazing sun that separates "solid ground" from "luminous fog".
const DETAIL_SCALE: f32 = 0.8;            // 1/period_m → ~1.25 m base relief
const DETAIL_OCTAVES: i32 = 3;
const DETAIL_EPS: f32 = 0.25;             // finite-difference step, metres
const DETAIL_NORMAL_STRENGTH: f32 = 0.0;  // facet-tilt amount — TEMP 0.0 to confirm
                                          // this value-noise detail normal is the
                                          // grid "weave" artifact (restore after).

// Both detail layers fade out with camera distance: their period goes
// sub-pixel on the far field and would shimmer. The macro/slope colour carries
// the distance instead.
const DETAIL_FADE_NEAR: f32 = 180.0;  // full detail within this range (m)
const DETAIL_FADE_FAR: f32  = 1800.0; // no detail beyond this range (m)
const DETAIL_COORD_PERIOD_M: f32 = 4000.0;

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

// Material mask attachment packed by `PipelineTileProvider`:
// R = vegetation/grass, G = soil/peat/sediment, B = exposed rock,
// A = wetness/cavity darkening. Values are weights, not final albedo.
fn sample_material_masks(tile: AtlasTile) -> vec4<f32> {
    let uv = attachment_uv(tile.coordinate.uv, 3u);
#ifdef FRAGMENT
#ifdef SAMPLE_GRAD
    return textureSampleGrad(
        attachment3_atlas, atlas_sampler, uv, tile.index,
        tile.coordinate.uv_dx, tile.coordinate.uv_dy,
    );
#else
    return textureSampleLevel(attachment3_atlas, atlas_sampler, uv, tile.index, 0.0);
#endif
#else
    return textureSampleLevel(attachment3_atlas, atlas_sampler, uv, tile.index, 0.0);
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

    if ray_t <= 0.0 || ray_t > terrain_extras.craft_shadow.params.z {
        return 1.0;
    }

    let silhouette_distance = length(closest);
    // Penumbra widens with caster height (solar angular diameter) so contact
    // shadows stay crisp while a high overflight softens out.
    let penumbra = max(
        terrain_extras.craft_shadow.params.y,
        max(radius * 0.18, ray_t * SHADOW_PENUMBRA_PER_M),
    );
    let coverage = 1.0 - smoothstep(radius, radius + penumbra, silhouette_distance);
    let fade = 1.0 - smoothstep(terrain_extras.craft_shadow.params.z * 0.75, terrain_extras.craft_shadow.params.z, ray_t);
    return 1.0 - terrain_extras.craft_shadow.params.x * coverage * fade;
}

// Distance from `p` to the segment `a`-`b`, all in the 2-D silhouette plane.
fn segment_distance_2d(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1.0e-8), 0.0, 1.0);
    return length(p - a - ab * h);
}

fn cross_2d(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

// Shadow factor for one thin planform quad (a lifting surface). The four
// corners are projected along the sun ray onto the plane through the
// terrain fragment; coverage is a signed-distance test against the
// projected outline, so the silhouette is the true planform at any sun
// angle — a wing edge-on to the sun casts (almost) nothing instead of the
// chord-thick slab a capsule proxy would throw.
fn planform_quad_shadow(
    hit_ws: vec3<f32>,
    sun_dir_ws: vec3<f32>,
    qa: vec4<f32>,
    qb: vec4<f32>,
    qc: vec4<f32>,
    qd: vec4<f32>,
) -> f32 {
    let delta_a = qa.xyz - hit_ws;
    let delta_b = qb.xyz - hit_ws;
    let delta_c = qc.xyz - hit_ws;
    let delta_d = qd.xyz - hit_ws;
    let t_a = dot(delta_a, sun_dir_ws);
    let t_b = dot(delta_b, sun_dir_ws);
    let t_c = dot(delta_c, sun_dir_ws);
    let t_d = dot(delta_d, sun_dir_ws);
    let ray_t = 0.25 * (t_a + t_b + t_c + t_d);
    if ray_t <= 0.0 || ray_t > terrain_extras.craft_shadow.params.z {
        return 1.0;
    }

    // 2-D basis on the plane perpendicular to the sun ray.
    var basis_x = cross(sun_dir_ws, vec3(0.0, 1.0, 0.0));
    if dot(basis_x, basis_x) < 1.0e-6 {
        basis_x = cross(sun_dir_ws, vec3(1.0, 0.0, 0.0));
    }
    basis_x = normalize(basis_x);
    let basis_y = cross(sun_dir_ws, basis_x);

    let pa = vec2(dot(delta_a, basis_x), dot(delta_a, basis_y));
    let pb = vec2(dot(delta_b, basis_x), dot(delta_b, basis_y));
    let pc = vec2(dot(delta_c, basis_x), dot(delta_c, basis_y));
    let pd = vec2(dot(delta_d, basis_x), dot(delta_d, basis_y));

    // The fragment projects to the origin. Signed distance to the quad
    // outline: winding-independent inside test via the two triangles, edge
    // distance for the magnitude.
    let origin = vec2(0.0, 0.0);
    let edge_distance = min(
        min(segment_distance_2d(origin, pa, pb), segment_distance_2d(origin, pb, pc)),
        min(segment_distance_2d(origin, pc, pd), segment_distance_2d(origin, pd, pa)),
    );
    let s1 = cross_2d(pb - pa, origin - pa);
    let s2 = cross_2d(pc - pb, origin - pb);
    let s3 = cross_2d(pd - pc, origin - pc);
    let s4 = cross_2d(pa - pd, origin - pd);
    let all_neg = s1 <= 0.0 && s2 <= 0.0 && s3 <= 0.0 && s4 <= 0.0;
    let all_pos = s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0 && s4 >= 0.0;
    var signed_distance = edge_distance;
    if all_neg || all_pos {
        signed_distance = -edge_distance;
    }

    let penumbra = max(
        terrain_extras.craft_shadow.params.y,
        ray_t * SHADOW_PENUMBRA_PER_M,
    );
    let coverage = 1.0 - smoothstep(-0.5 * penumbra, 0.5 * penumbra, signed_distance);
    let fade = 1.0 - smoothstep(terrain_extras.craft_shadow.params.z * 0.75, terrain_extras.craft_shadow.params.z, ray_t);
    return 1.0 - terrain_extras.craft_shadow.params.x * coverage * fade;
}

fn local_craft_shadow(hit_ws: vec3<f32>, sun_dir_ws: vec3<f32>) -> f32 {
    let caster_count = min(
        u32(max(terrain_extras.craft_shadow.params.w, 0.0)),
        MAX_TERRAIN_SHADOW_CASTERS,
    );
    let quad_count = min(
        u32(max(terrain_extras.craft_shadow.quad_params.x, 0.0)),
        MAX_TERRAIN_SHADOW_QUADS,
    );
    if caster_count == 0u && quad_count == 0u {
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
                terrain_extras.craft_shadow.caster_a_radius[i],
                terrain_extras.craft_shadow.caster_b_radius[i],
            ),
        );
    }
    for (var i = 0u; i < MAX_TERRAIN_SHADOW_QUADS; i = i + 1u) {
        if i >= quad_count {
            break;
        }
        shadow = min(
            shadow,
            planform_quad_shadow(
                hit_ws,
                sun_dir_ws,
                terrain_extras.craft_shadow.quad_a[i],
                terrain_extras.craft_shadow.quad_b[i],
                terrain_extras.craft_shadow.quad_c[i],
                terrain_extras.craft_shadow.quad_d[i],
            ),
        );
    }
    return shadow;
}

// ── Terrain self-shadowing (render-time horizon march) ─────────────────────
// Marches the resident height attachment along the sun's horizontal direction
// and compares the steepest occluder slope against the sun's elevation. No
// bake: the shadow always reflects whatever surface the Query API currently
// materialises into the atlas (any backing, any LOD, dynamic layers included),
// which is the architecturally honest fit for on-demand terrain. Range is
// bounded by the resident tile footprint — long shadows from off-tile relief
// are out of scope here (they would be a cached per-tile horizon channel
// later, not a global bake).

const TERRAIN_SHADOW_STEPS: u32 = 16u;
const TERRAIN_SHADOW_STRENGTH: f32 = 1.0;
// Initial step + geometric growth, in height texels. Dense near the receiver
// (catches rims/dunes), sparse far out (catches large relief) for few taps.
const TERRAIN_SHADOW_STEP0_TEXELS: f32 = 1.0;
const TERRAIN_SHADOW_STEP_GROWTH: f32 = 1.4;
// The height texture's v axis runs opposite the tile bitangent (see
// `sample_normal`'s `down - up` gradient convention). Flip this sign if the
// shadows visibly fall on the sun-facing side instead of away from the sun.
const TERRAIN_SHADOW_V_SIGN: f32 = -1.0;

// Rebuild the tile's tangent frame exactly as `sample_normal` does, so the
// march direction in height-texture uv lines up with the atlas axes. Columns
// are (tangent, bitangent, normal); +u ↔ tangent, +v ↔ -bitangent.
fn tile_tangent_frame(side: u32, geo_normal: vec3<f32>) -> mat3x3<f32> {
#ifdef SPHERICAL
    var FACE_UP = array<vec3<f32>, 6>(
        vec3<f32>( 0.0, 1.0,  0.0),
        vec3<f32>( 0.0, 1.0,  0.0),
        vec3<f32>( 0.0, 0.0, -1.0),
        vec3<f32>( 0.0, 0.0, -1.0),
        vec3<f32>(-1.0, 0.0,  0.0),
        vec3<f32>(-1.0, 0.0,  0.0),
    );
    let normal = normalize(geo_normal);
    var tangent = cross(FACE_UP[side], normal);
    if (dot(tangent, tangent) < 1.0e-8) {
        var fallback_axis = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(normal.y) > 0.9) { fallback_axis = vec3<f32>(1.0, 0.0, 0.0); }
        tangent = cross(fallback_axis, normal);
    }
    tangent = normalize(tangent);
    let bitangent = normalize(cross(normal, tangent));
    return mat3x3<f32>(tangent, bitangent, normal);
#else
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
    );
#endif
}

fn terrain_self_shadow(tile: AtlasTile, geo_normal: vec3<f32>, sun_dir_ws: vec3<f32>) -> f32 {
    let frame = tile_tangent_frame(tile.coordinate.side, geo_normal);
    // Sun in the tile tangent basis: x → tangent, y → bitangent, z → normal.
    let sun_ts = transpose(frame) * sun_dir_ws;
    let sin_elev = sun_ts.z;
    if (sin_elev <= 1.0e-3) {
        return 1.0; // sun on/below the local horizon: the terminator owns it
    }
    let horiz_len = length(sun_ts.xy);
    if (horiz_len < 1.0e-4) {
        return 1.0; // sun ~ straight up: nothing self-shadows
    }
    let tan_elev = sin_elev / horiz_len; // rise per horizontal metre

    // March direction in height-texture uv (see the +u/+v note above).
    let uv_dir = normalize(vec2<f32>(sun_ts.x, TERRAIN_SHADOW_V_SIGN * sun_ts.y));

    let tex_size = attachments[0u].size;
    let inv_size = 1.0 / tex_size;
    let side_length = 3.14159265359 / 4.0 * config.scale;
    let m_per_texel = side_length / (tex_size * tile_count(tile.coordinate.lod));

    let base_uv = attachment_uv(tile.coordinate.uv, 0u);
    let h0 = sample_height_atlas_uv_m(tile, base_uv);

    // Track the steepest slope from the receiver to any occluder along the ray.
    // If it exceeds the sun's elevation slope, the receiver is shadowed.
    var max_slope = 0.0;
    var dist_texels = 0.0;
    var step = TERRAIN_SHADOW_STEP0_TEXELS;
    for (var k = 0u; k < TERRAIN_SHADOW_STEPS; k = k + 1u) {
        dist_texels = dist_texels + step;
        let suv = base_uv + uv_dir * (dist_texels * inv_size);
        let h = sample_height_atlas_uv_m(tile, suv);
        let d_m = dist_texels * m_per_texel;
        max_slope = max(max_slope, (h - h0) / max(d_m, 1.0e-3));
        step = step * TERRAIN_SHADOW_STEP_GROWTH;
    }

    // Soft penumbra around the horizon crossing; widen a little as the sun
    // climbs so high-noon micro-relief does not read as hard-edged.
    let soft = max(0.03, 0.20 * tan_elev);
    let occ = smoothstep(tan_elev - soft, tan_elev + soft, max_slope);
    return 1.0 - occ * TERRAIN_SHADOW_STRENGTH;
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

struct TerrainMaterialSample {
    albedo: vec3<f32>,
    normal_strength: f32,
    occlusion: f32,
};

// Build the local surface colour by stacking ecological bands by altitude and
// slope on top of the tile provider's grass/soil/rock intent masks.
//   altitude_m — metres above the body reference sphere (drives treeline / snow)
//   slope_t    — geometric steepness in [0,1] (0 flat, 1 vertical)
//   variation  — low-frequency value noise in [-1,1] (jitters bands, mottles)
fn eval_material_stack(
    masks_in: vec4<f32>,
    macro_albedo: vec3<f32>,
    altitude_m: f32,
    slope_t: f32,
    variation: f32,
) -> TerrainMaterialSample {
    var masks = max(masks_in, vec4<f32>(0.0));
    let weight_sum = max(masks.r + masks.g + masks.b, 1.0e-4);
    let grass_w = masks.r / weight_sum;
    let soil_w = masks.g / weight_sum;
    let rock_w = masks.b / weight_sum;
    let wet = clamp(masks.a, 0.0, 1.0);

    // Steepness used to expose rock and keep snow off cliffs: the coarse
    // geometric slope OR the provider's rock intent, whichever reads steeper, so
    // a face stays bare even where the macro normal is too coarse to register.
    let steep = clamp(max(slope_t, rock_w), 0.0, 1.0);
    let jitter = variation * SNOW_LINE_NOISE_M;

    // Vegetation column: dark lush lowland → temperate grass → pale dry alpine
    // grass approaching the treeline.
    let lush = smoothstep(LUSH_HI_M, LUSH_LO_M, altitude_m + jitter); // 1 low, 0 high
    let alpine = smoothstep(TREELINE_LO_M, TREELINE_HI_M, altitude_m + jitter);
    let grass_c = mix(C_GRASS, C_FOREST, lush * 0.6);
    let veg = mix(grass_c, C_DRYGRASS, alpine);

    // Rock cools and greys with altitude (warm soil-stained rock low down,
    // lichen-free scree up high).
    let rock_col = mix(C_ROCK_LO, C_ROCK_HI, alpine);

    // Earthy substrate from the material masks, darkened in wet hollows.
    var ground = veg * grass_w + C_SOIL * soil_w + rock_col * rock_w;
    ground = mix(ground, C_WET, wet * (1.0 - rock_w * 0.55));

    // Snow: above a noise-jittered snowline, only where the ground is not too
    // steep. Permanent cap once well clear of the line.
    let snow_alt = smoothstep(SNOW_LINE_LO_M + jitter, SNOW_LINE_HI_M + jitter, altitude_m);
    let snow_hold = 1.0 - smoothstep(SNOW_SLOPE_LO, SNOW_SLOPE_HI, steep);
    let snow = clamp(snow_alt * snow_hold, 0.0, 1.0);
    ground = mix(ground, C_SNOW, snow);

    // Low-frequency value mottle so even a single biome breaks into patches
    // (snow stays clean).
    ground = ground * (1.0 + variation * MACRO_VAR_AMT * (1.0 - snow));

    // Keep a little of the baked macro albedo as broad climate/body identity,
    // but the altitude/slope model is the local truth. Drop it under snow.
    let macro_tint = desaturate(macro_albedo, 0.86);
    var out: TerrainMaterialSample;
    out.albedo = mix(ground, macro_tint, 0.16 * (1.0 - snow));
    out.normal_strength = mix(mix(0.45, 0.85, rock_w) + soil_w * 0.12, 0.25, snow);
    out.occlusion = 1.0 - wet * 0.18 - soil_w * 0.05 - rock_w * 0.04;
    return out;
}

// ── Procedural surface detail (Step 2 + Step 3) ───────────────────────────
// Cheap hash-based value noise / fBm, evaluated in body-fixed metres so the
// surface detail stays glued to the planet under time warp and floating-origin
// shifts. The CPU supplies the camera's body-fixed phase modulo this period;
// the noise itself wraps on the same period so crossing a period boundary is
// seamless. Keep this a multiple of every base detail wavelength below
// (20 m breakup, 1.25 m micro-relief).

fn hash13(p_in: vec3<f32>) -> f32 {
    var p3 = fract(p_in * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

fn wrap_lattice(p: vec3<f32>, period: f32) -> vec3<f32> {
    return p - floor(p / period) * period;
}

fn value_noise_3d_periodic(x: vec3<f32>, period: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash13(wrap_lattice(i + vec3<f32>(0.0, 0.0, 0.0), period));
    let n100 = hash13(wrap_lattice(i + vec3<f32>(1.0, 0.0, 0.0), period));
    let n010 = hash13(wrap_lattice(i + vec3<f32>(0.0, 1.0, 0.0), period));
    let n110 = hash13(wrap_lattice(i + vec3<f32>(1.0, 1.0, 0.0), period));
    let n001 = hash13(wrap_lattice(i + vec3<f32>(0.0, 0.0, 1.0), period));
    let n101 = hash13(wrap_lattice(i + vec3<f32>(1.0, 0.0, 1.0), period));
    let n011 = hash13(wrap_lattice(i + vec3<f32>(0.0, 1.0, 1.0), period));
    let n111 = hash13(wrap_lattice(i + vec3<f32>(1.0, 1.0, 1.0), period));
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

fn fbm3_periodic(p_in: vec3<f32>, octaves: i32, period_in: f32) -> f32 {
    var p = p_in;
    var period = period_in;
    var amp = 0.5;
    var sum = 0.0;
    var norm = 0.0;
    for (var o = 0; o < octaves; o = o + 1) {
        sum = sum + amp * value_noise_3d_periodic(p, period);
        norm = norm + amp;
        p = p * 2.0;
        period = period * 2.0;
        amp = amp * 0.5;
    }
    return sum / max(norm, 1.0e-5);
}

fn detail_height(p_body: vec3<f32>) -> f32 {
    return fbm3_periodic(
        p_body * DETAIL_SCALE,
        DETAIL_OCTAVES,
        DETAIL_COORD_PERIOD_M * DETAIL_SCALE,
    );
}

// Value noise plus its analytic gradient. The smoothstep weights
// u = f²(3−2f) have derivative du = 6f(1−f); combined with the trilinear
// corner deltas this is the exact gradient of `value_noise_3d_periodic` in a
// single evaluation. Returns vec4(value, ∂value/∂x, ∂value/∂y, ∂value/∂z).
fn value_noise_3d_periodic_grad(x: vec3<f32>, period: f32) -> vec4<f32> {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);

    let n000 = hash13(wrap_lattice(i + vec3<f32>(0.0, 0.0, 0.0), period));
    let n100 = hash13(wrap_lattice(i + vec3<f32>(1.0, 0.0, 0.0), period));
    let n010 = hash13(wrap_lattice(i + vec3<f32>(0.0, 1.0, 0.0), period));
    let n110 = hash13(wrap_lattice(i + vec3<f32>(1.0, 1.0, 0.0), period));
    let n001 = hash13(wrap_lattice(i + vec3<f32>(0.0, 0.0, 1.0), period));
    let n101 = hash13(wrap_lattice(i + vec3<f32>(1.0, 0.0, 1.0), period));
    let n011 = hash13(wrap_lattice(i + vec3<f32>(0.0, 1.0, 1.0), period));
    let n111 = hash13(wrap_lattice(i + vec3<f32>(1.0, 1.0, 1.0), period));

    let k0 = n000;
    let k1 = n100 - n000;
    let k2 = n010 - n000;
    let k3 = n001 - n000;
    let k4 = n000 - n100 - n010 + n110;
    let k5 = n000 - n010 - n001 + n011;
    let k6 = n000 - n100 - n001 + n101;
    let k7 = -n000 + n100 + n010 - n110 + n001 - n101 - n011 + n111;

    let value = k0 + k1 * u.x + k2 * u.y + k3 * u.z
        + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x
        + k7 * u.x * u.y * u.z;
    let grad = vec3<f32>(
        du.x * (k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z),
        du.y * (k2 + k4 * u.x + k5 * u.z + k7 * u.x * u.z),
        du.z * (k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y),
    );
    return vec4<f32>(value, grad);
}

// fBm value + analytic gradient (∂value/∂p_in), matching `fbm3_periodic`. Each
// octave doubles frequency and halves amplitude; the chain rule scales each
// octave's gradient by its frequency, tracked in `freq` (= 2^octave).
fn fbm3_grad(p_in: vec3<f32>, octaves: i32, period_in: f32) -> vec4<f32> {
    var p = p_in;
    var period = period_in;
    var amp = 0.5;
    var freq = 1.0;
    var sum = 0.0;
    var grad = vec3<f32>(0.0);
    var norm = 0.0;
    for (var o = 0; o < octaves; o = o + 1) {
        let vg = value_noise_3d_periodic_grad(p, period);
        sum = sum + amp * vg.x;
        grad = grad + amp * freq * vg.yzw;
        norm = norm + amp;
        p = p * 2.0;
        period = period * 2.0;
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    let inv = 1.0 / max(norm, 1.0e-5);
    return vec4<f32>(sum * inv, grad * inv);
}

// `detail_height` value and gradient w.r.t. body-space metres. The fBm runs in
// scaled coordinates (`p_body * DETAIL_SCALE`), so one more `DETAIL_SCALE`
// factor folds into the gradient by the chain rule.
fn detail_height_grad(p_body: vec3<f32>) -> vec4<f32> {
    let g = fbm3_grad(
        p_body * DETAIL_SCALE,
        DETAIL_OCTAVES,
        DETAIL_COORD_PERIOD_M * DETAIL_SCALE,
    );
    return vec4<f32>(g.x, g.yzw * DETAIL_SCALE);
}

struct SurfaceDetail {
    tint: vec3<f32>,          // multiplicative albedo breakup, ~1.0
    normal_offset: vec3<f32>, // tangential perturbation for the lighting normal
}

fn surface_detail(p_body: vec3<f32>, geo_normal: vec3<f32>, cam_dist: f32) -> SurfaceDetail {
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
    let v = fbm3_periodic(
        p_body * BREAKUP_SCALE,
        3,
        DETAIL_COORD_PERIOD_M * BREAKUP_SCALE,
    );
    let dv = (v - 0.5) * 2.0;
    let value_mul = 1.0 + dv * BREAKUP_VALUE_AMT;
    let hue = vec3<f32>(1.0 + dv * BREAKUP_HUE_AMT, 1.0, 1.0 - dv * BREAKUP_HUE_AMT);
    out.tint = mix(vec3<f32>(1.0), vec3<f32>(value_mul) * hue, fade);

    // Step 3 — micro-relief normal from the gradient of a detail height field,
    // projected tangent to the sphere so it tilts facets without changing the
    // macro surface orientation. Magnitude is capped so a steep noise gradient
    // can't fold the normal past the horizon. The gradient is evaluated
    // analytically in a single fBm pass (`detail_height_grad`) — the previous
    // four finite-difference taps were the dominant per-fragment cost here.
    let grad = detail_height_grad(p_body).yzw;
    let grad_t = grad - geo_normal * dot(grad, geo_normal);
    var off = -grad_t * (DETAIL_NORMAL_STRENGTH * fade);
    let off_len = length(off);
    out.normal_offset = off * (0.8 / max(0.8, off_len));
    return out;
}

// Airless regolith fine detail: value-only albedo mottle + speckle and a
// micro-relief normal. Same body-fixed coordinate basis as `surface_detail`, so
// it stays glued to the surface under time warp / floating-origin shifts, and
// it fades with distance to avoid sub-pixel shimmer (the macro albedo carries
// the far field).
fn regolith_detail(p_body: vec3<f32>, geo_normal: vec3<f32>, cam_dist: f32) -> SurfaceDetail {
    var out: SurfaceDetail;
    out.tint = vec3<f32>(REGOLITH_VALUE_TRIM);
    out.normal_offset = vec3<f32>(0.0);

    let fade = 1.0 - smoothstep(DETAIL_FADE_NEAR, DETAIL_FADE_FAR, cam_dist);
    if (fade <= 0.0) {
        return out;
    }

    // Broad mare/highland mottle + fine dust/ejecta speckle. Both centred on 0
    // so the mean stays at the trimmed baked albedo; the spread is what reads as
    // dusty texture and dark small-crater hollows.
    let mottle = (fbm3_periodic(
        p_body * REGOLITH_MOTTLE_SCALE,
        4,
        DETAIL_COORD_PERIOD_M * REGOLITH_MOTTLE_SCALE,
    ) - 0.5) * 2.0;
    let speckle = (fbm3_periodic(
        p_body * REGOLITH_SPECKLE_SCALE,
        3,
        DETAIL_COORD_PERIOD_M * REGOLITH_SPECKLE_SCALE,
    ) - 0.5) * 2.0;
    let value_mul = 1.0 + (mottle * REGOLITH_MOTTLE_AMT + speckle * REGOLITH_SPECKLE_AMT) * fade;
    out.tint = vec3<f32>(REGOLITH_VALUE_TRIM * clamp(value_mul, 0.45, 1.8));

    // Micro-relief normal from the fine detail height gradient (~1.25 m), tangent
    // to the sphere so it tilts facets toward/away from the low sun — the pocked
    // light/dark micro-contrast that makes regolith read as solid cratered ground.
    let grad = detail_height_grad(p_body).yzw;
    let grad_t = grad - geo_normal * dot(grad, geo_normal);
    var off = -grad_t * (REGOLITH_NORMAL_STRENGTH * fade);
    let off_len = length(off);
    out.normal_offset = off * (0.7 / max(0.7, off_len));
    return out;
}

// ── Rough-dielectric surface BRDF ─────────────────────────────────────────
// Thalos is a wet, vegetated terrestrial body — soil, grass, rock, snow,
// water — not airless particulate regolith, so its ground LOD shades with a
// standard rough-dielectric BRDF rather than the Hapke radiative-transfer
// model (the impostor still uses Hapke for genuinely airless bodies). Two
// lobes: an Oren–Nayar rough-diffuse term that degrades gracefully at grazing
// angles — the harsh opposition-surge contour bands near the terminator were
// the Hapke artefact that motivated the prior P2A placeholder — plus a
// Cook–Torrance GGX microfacet specular with a dielectric F0, so wet ground
// and snow pick up a tight highlight that Hapke cannot express.

const PI_BRDF: f32 = 3.14159265358979323846;
// Non-metallic surface normal reflectance at normal incidence (~4%).
const DIELECTRIC_F0: f32 = 0.04;
// Weight of the direct-sun lobe relative to the sky fill. Carried over from
// the placeholder so overall ground brightness stays in the tonemapper's
// range; the primary knob to nudge after a preview.
const DIRECT_SUN_STRENGTH: f32 = 0.62;

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
    let spec = d * vis * f;

    let diff = oren_nayar_term(n_dot_l, n_dot_v, l, v, roughness);
    let diffuse = albedo * diff * (1.0 - f);

    return diffuse + vec3<f32>(spec);
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var info = fragment_info(input);
    let tile = lookup_tile(info.coordinate, info.blend, 0u);

    var albedo        = sample_attachment1(tile);
    var height_normal = sample_normal(tile, info.world_normal);
    var roughness     = sample_roughness(tile);
    var material_masks = sample_material_masks(tile);

    if (info.blend.ratio > 0.0) {
        let tile2 = lookup_tile(info.coordinate, info.blend, 1u);
        albedo = mix(albedo, sample_attachment1(tile2), info.blend.ratio);
        height_normal = mix(
            height_normal,
            sample_normal(tile2, info.world_normal),
            info.blend.ratio,
        );
        roughness = mix(roughness, sample_roughness(tile2), info.blend.ratio);
        material_masks = mix(material_masks, sample_material_masks(tile2), info.blend.ratio);
    }

    // Geometry shared by grading and lighting.
    let geo_normal = normalize(info.world_normal);
    let hit_ws = info.world_position.xyz;
    let cam_dist = length(info.view_vector);
    let debug_on = terrain_extras.debug.params.x >= 0.5;
    // Surface shading style (see `TerrainShadingStyle` in body_material.rs).
    // Vegetated terrestrial (Thalos) keeps the ecological-band + dielectric-BRDF
    // path; airless regolith (Mira) uses the baked gray albedo + Hapke, matching
    // its orbital impostor across the LOD swap.
    let style_regolith = terrain_extras.inspection.y >= 0.5;

    // Procedural surface detail (Step 2 breakup + Step 3 micro-relief normal),
    // synthesised from body-fixed metres so it remains static under time warp.
    let frag_relative_position = -info.view_vector;
    let body_relative_position = quat_rotate(terrain_extras.debug.world_to_body_rot, frag_relative_position);
    let detail_p_body = terrain_extras.debug.view_phase.xyz + body_relative_position;
    let detail = surface_detail(detail_p_body, geo_normal, cam_dist);

    let altitude_m = sample_height(tile);
    let geo_slope_t = surface_slope(height_normal, geo_normal);

    // `material.normal_strength` / `material.occlusion` feed the vegetated
    // lighting path; regolith ignores them, so default to neutral values.
    var material: TerrainMaterialSample;
    material.albedo = albedo.rgb;
    material.normal_strength = 0.45;
    material.occlusion = 1.0;

    // Regolith micro-relief normal, set in the regolith branch and consumed by
    // the lighting-normal section below (the vegetated path uses `detail`).
    var regolith_normal_offset = vec3<f32>(0.0);

    var surface_rgb: vec3<f32>;
    if (style_regolith) {
        // Airless regolith: baked gray albedo (the body's authored mare/highland
        // tone) textured by value-only mottle + speckle; no ecological bands, no
        // hue drift. The micro-relief normal is fed to the lighting below.
        if (debug_on) {
            surface_rgb = albedo.rgb * REGOLITH_VALUE_TRIM;
        } else {
            let rd = regolith_detail(detail_p_body, geo_normal, cam_dist);
            surface_rgb = albedo.rgb * rd.tint;
            regolith_normal_offset = rd.normal_offset;
        }
    } else {
        // Naturalistic material blending. The tile provider publishes continuous
        // material intent masks, so grass, soil, rock, and wet hollows separate
        // by terrain form; on top of that we stack altitude/slope ecological
        // bands (treeline, alpine scree, snow caps) computed here from the
        // height attachment and the geometric slope.
        //
        // Low-frequency value variation in [-1,1]; jitters the treeline/snowline
        // and mottles vegetation so the altitude bands don't read as clean
        // contour rings.
        let macro_var = (fbm3_periodic(
            detail_p_body * MACRO_VAR_SCALE,
            3,
            DETAIL_COORD_PERIOD_M * MACRO_VAR_SCALE,
        ) - 0.5) * 2.0;
        material = eval_material_stack(
            material_masks,
            grade_surface(albedo.rgb, material_masks.b),
            altitude_m,
            geo_slope_t,
            macro_var,
        );
        surface_rgb = material.albedo;
        if (!debug_on) {
            surface_rgb = surface_rgb * detail.tint;
        }
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
    let debug_cell = max(terrain_extras.debug.params.z, 1e-3);
    let debug_rel = body_relative_position;
    let debug_p = (terrain_extras.debug.view_phase.xyz + debug_rel) / debug_cell;
    let debug_w = max(abs(dpdx(debug_p)), abs(dpdy(debug_p)));
    let debug_checker = checker_3d_aa(debug_p, debug_w);
    if (terrain_extras.debug.params.x >= 0.5) {
        let dark = vec3<f32>(0.05, 0.05, 0.05);
        let light = vec3<f32>(0.80, 0.80, 0.80);
        albedo = vec4<f32>(mix(dark, light, debug_checker), 1.0);
    }

    if (terrain_extras.inspection.x >= 0.5) {
        var output_fullbright: FragmentOutput;
        output_fullbright.color = vec4<f32>(albedo.rgb, albedo.a);
        return output_fullbright;
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
        if (style_regolith) {
            normal = normalize(height_n + regolith_normal_offset);
        } else {
            normal = normalize(height_n + detail.normal_offset * material.normal_strength);
        }
    }
    let view_dir = normalize(info.view_vector);

    // Render-time shadowing: the local craft proxy (analytic ship-part
    // silhouettes) combined with a terrain self-shadow horizon march over the
    // resident height atlas. Both fade only the direct sun term; the sky fill
    // below stands in for skylight reaching shadowed ground.
    let craft_shadow = local_craft_shadow(hit_ws, sun_dir_ws);
    var self_shadow = 1.0;
    if (!debug_on) {
        self_shadow = terrain_self_shadow(tile, geo_normal, sun_dir_ws);
    }
    let external_shadow = craft_shadow * self_shadow;

    // Surface lighting. The shading normal is pulled most of the way toward the
    // relief normal but kept anchored to the geometric normal so steep micro-
    // facets near the terminator can't out-light the body curvature.
    let stable_normal = normalize(mix(geo_normal, normal, 0.85));

    var lit: vec3<f32>;
    if (style_regolith) {
        // Airless regolith: Hapke radiative-transfer BRDF — the exact routine
        // the orbital impostor uses (`shade_hapke_surface`), so the two render
        // paths shade identically at the impostor↔ground LOD swap. No
        // atmospheric sky fill (airless); ambient comes from the scene floor
        // inside the Hapke helper. Roughness drives the opposition-surge width;
        // dry regolith has no wet-hollow tightening.
        let surf_roughness = clamp(roughness, 0.06, 1.0);
        lit = shade_hapke_surface(
            albedo.rgb,
            surf_roughness,
            stable_normal,
            geo_normal,
            view_dir,
            hit_ws,
            sun_dir_ws,
            sun_flux,
            terrain_scene,
            external_shadow,
        );
    } else {
        // Rough-dielectric surface lighting (see the BRDF block above).
        let n_dot_l = max(dot(stable_normal, sun_dir_ws), 0.0);
        let n_dot_v = max(dot(stable_normal, view_dir), 1.0e-4);

        // Specular roughness from the sampled attachment, tightened in wet
        // hollows (material mask .a) so puddles and wet rock get a sharper
        // highlight while dry, rough ground stays matte. Clamped away from 0 to
        // keep the GGX lobe from collapsing to a firefly.
        let wetness = clamp(material_masks.a, 0.0, 1.0);
        let surf_roughness = clamp(mix(roughness, roughness * 0.45, wetness), 0.06, 1.0);

        let brdf = surface_brdf(
            albedo.rgb,
            surf_roughness,
            stable_normal,
            sun_dir_ws,
            view_dir,
            n_dot_l,
            n_dot_v,
        );
        // Irradiance cosine + direct-sun shadowing applied here; flux is folded
        // into DIRECT_SUN_STRENGTH below to stay in the placeholder's brightness
        // range until the scene exposure path is unified.
        let direct_rgb = brdf * n_dot_l * external_shadow;

        // Sky fill (diffuse skylight). A constant ambient floor used to light the
        // night side as brightly as a hazy daytime shadow. Real skylight only
        // exists while the sun illuminates the atmosphere, so drive the fill by
        // the sun's elevation over the *macro* horizon (geometric normal, not
        // the relief normal) and let it fade to a faint starlight floor at night.
        // The daytime level is well under the old 0.28 floor so the overhead-sun
        // case stops washing the flat ground out into the tonemapper's grey
        // shoulder; a gentle cool tint stands in for blue-sky scatter and keeps
        // daylight shadows from reading as flat grey.
        let sun_elevation = dot(geo_normal, sun_dir_ws);
        let daylight = smoothstep(-0.06, 0.12, sun_elevation);
        let night_fill = 0.012;
        let day_fill = 0.15;
        let fill = mix(night_fill, day_fill, daylight) * material.occlusion;
        let sky_tint = mix(vec3<f32>(1.0), vec3<f32>(0.62, 0.74, 1.0), 0.25 * daylight);

        lit = direct_rgb * DIRECT_SUN_STRENGTH + albedo.rgb * sky_tint * fill;
    }

    var output: FragmentOutput;
    output.color = vec4<f32>(lit, albedo.a);
    return output;
}
