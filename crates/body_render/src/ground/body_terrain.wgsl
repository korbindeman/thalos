// Ground-LOD terrain shader for procedural bodies.
//
// Reads height, albedo, and roughness from the thalos_udlod attachment
// atlases (group 1), receives the shared cascaded sun-shadow rig
// (`thalos::shadow`), and shades by a per-body surface style (see
// `TerrainShadingStyle` in body_material.rs, carried in
// `terrain_extras.inspection.y`):
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

#import thalos_udlod::types::{AtlasTile, Coordinate}
#import thalos_udlod::bindings::{config, atlas_sampler, attachments, attachment2_atlas, attachment3_atlas}
#import thalos_udlod::attachments::{sample_attachment1, sample_normal, attachment_uv, sample_height_atlas_uv_m, sample_height}
#import thalos_udlod::fragment::{FragmentInput, FragmentOutput, fragment_info}
#import thalos_udlod::vertex::{VertexInput, VertexOutput, vertex_info, vertex_output}
#import thalos_udlod::functions::{lookup_tile, tile_count, compute_local_position}
#import thalos::atmosphere::AtmosphereBlock
#import thalos::lighting::{
    SceneLighting, ThalosSurface, shade_surface, SURFACE_DIELECTRIC, SURFACE_REGOLITH,
    SurfaceSky, compute_surface_sky,
    specular_aa_variance, specular_aa_apply,
}
// Shared vegetated-ground colour + moisture field, also mirrored CPU-side by
// `ground/landcover.rs` so the grass blades read the exact same green.
#import thalos::landcover::{
    moisture_detail, macro_variation, vegetation_color, climate_cold_lift, climate_warmth,
}
#import thalos::shadow::{ShadowCascadeBlock, sun_shadow_factor_nrm}
#import bevy_pbr::mesh_view_bindings::view

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
// Slot 2 packs debug + inspection + the sun-shadow cascade into one buffer; see
// `BodyTerrainExtras` in `body_material.rs` for the slot-budget rationale
// (Metal vertex stage caps at 16 buffers and AsBindGroup forces vertex
// visibility on every `#[uniform(N)]`).
// `ShadowCascadeBlock` + `sun_shadow_factor` are shared from `thalos::shadow`.
// The depth maps are SEPARATE `texture_depth_2d` bindings (one per cascade).

// One analytic flatten pad, mirrored from `thalos_terrain::TerrainFlatten` by
// `FlattenRegionGpu` in `body_material.rs` (see there for field packing).
struct TerrainFlattenRegion {
    // xyz = unit body-fixed direction to the pad centre, w = plane elevation at
    // the centre (m above the reference radius).
    center_elev: vec4<f32>,
    // xyz = unit body-fixed tangent along the pad, w = rect half-length (m).
    along: vec4<f32>,
    // xyz = unit body-fixed tangent across the pad, w = rect half-width (m).
    across: vec4<f32>,
    // x = rect centre offset along (m), y = rect centre offset across (m),
    // z = cos of the angular reject radius, w = interior feather width (m).
    rect: vec4<f32>,
}

// Mirrors `FlattenBlock` in `body_material.rs`; keep MAX_FLATTEN_REGIONS in sync.
struct TerrainFlattenBlock {
    // x = active region count, y = body reference radius (m), zw reserved.
    // (Named `header` because `meta` is a naga reserved word — see the
    // wgsl-bevy skill. Uniform layout matches the Rust side by declaration
    // order, not name.)
    header: vec4<f32>,
    regions: array<TerrainFlattenRegion, 4>,
}

struct BodyTerrainExtras {
    debug: BodyTerrainDebug,
    inspection: vec4<f32>,
    shadow: ShadowCascadeBlock,
    // WGSL field is named `pads` (not `flatten`) to stay clear of any
    // current/future reserved word; the Rust field is `flatten` — uniform
    // layout matches by declaration order, not name.
    pads: TerrainFlattenBlock,
}

@group(3) @binding(0) var<uniform> terrain_atmos: AtmosphereBlock;
@group(3) @binding(1) var<uniform> terrain_scene: SceneLighting;
@group(3) @binding(2) var<uniform> terrain_extras: BodyTerrainExtras;
// Per-cascade sun-shadow depth maps (near→far), rendered by the game's
// `rendering::sun_shadow` rig. Each a plain `texture_depth_2d` (no depth array).
@group(3) @binding(3) var sun_shadow_map_0: texture_depth_2d;
@group(3) @binding(4) var sun_shadow_map_1: texture_depth_2d;
@group(3) @binding(5) var sun_shadow_map_2: texture_depth_2d;
// Half-res screen-space AO (`rendering::ssao`), multiplied into the AMBIENT
// occlusion only (graphics F5). Bound to the white fallback when unset (no AO);
// `terrain_extras.inspection.w == 0` skips sampling entirely (map terrain / airless).
@group(3) @binding(6) var ao_tex: texture_2d<f32>;
@group(3) @binding(7) var ao_samp: sampler;

// Sample the screen-space AO for this fragment. `frag_coord` is the framebuffer
// pixel coordinate (`FragmentInput.clip_position.xy`); dividing by the actual
// viewport size gives the [0,1] screen UV the half-res AO target spans. (An
// earlier version derived the viewport as `2 × textureDimensions(ao_tex)`, which
// mis-registers by up to a texel on odd-sized windows — a subtle banding source.)
// Returns 1.0 (unoccluded) when SSAO is disabled for this material.
fn screen_space_ao(frag_coord: vec2<f32>) -> f32 {
    if terrain_extras.inspection.w < 0.5 {
        return 1.0;
    }
    let uv = frag_coord / max(view.viewport.zw, vec2<f32>(1.0));
    return textureSampleLevel(ao_tex, ao_samp, uv, 0.0).r;
}

// ── Analytic pad flatten (vertex stage) ────────────────────────────────────
//
// The height sampled from the tile atlas is only *approximately* the flatten
// plane structures are built against: the vertex-stage LOD height blend and
// morph mix in coarse ancestor tiles whose kilometre-scale texels average
// natural terrain into the pad, and tiles baked before a runtime flatten
// installed carry no flatten at all. Either error is decimetres — more than
// the few-cm asphalt lift — so at LOD-transition view distances the rendered
// ground used to rise through the paving and z-fight the whole base.
//
// This override makes the flatten plane a render-time authority instead of a
// bake-time hope: inside a pad rectangle the vertex height is pinned to the
// exact tangent-plane elevation (the same `plane_elevation_m` maths the CPU
// bake, collider, and structure placement use), independent of tile LOD, morph
// state, or bake timing. At fine LOD the baked interior already equals the
// plane, so this is a no-op there; at coarse LOD it removes the error exactly
// where structures live. A feather band just inside the rectangle edge blends
// the correction in so no crease appears against the (uncorrected) ramp.
//
// Precision: everything here is f32, but formulated cancellation-free. The
// plane elevation is NOT computed as `(R+E)/cosθ − R` (which subtracts two
// ~R-magnitude values and quantises to f32 ulp(R) ≈ 0.25 m); instead
// `1 − cosθ = |dir − c|²/2` keeps every term at pad scale, leaving sub-mm
// error. The vertex direction from `compute_local_position` carries ~1e-7 rad
// noise → ~0.2 m lateral wobble of the rectangle test, invisible against the
// feather.
fn flattened_height(coordinate: Coordinate, height: f32) -> f32 {
    let count = u32(terrain_extras.pads.header.x);
    if (count == 0u) {
        return height;
    }
    let radius = terrain_extras.pads.header.y;
    // Unit body-fixed direction of this vertex (the terrain's local frame is
    // the body-fixed frame). Must use the post-morph coordinate so the test
    // matches the position the vertex actually renders at.
    let dir = compute_local_position(coordinate);
    var out_height = height;
    for (var i = 0u; i < count; i = i + 1u) {
        let region = terrain_extras.pads.regions[i];
        let c = region.center_elev.xyz;
        // Cheap angular reject: almost every vertex on the planet is nowhere
        // near a pad. The CPU packs generous slack into `rect.z` because a
        // near-1 cosine compare is noisy at f32.
        if (dot(dir, c) < region.rect.z) {
            continue;
        }
        let d = dir - c;
        // Tangent-plane offset from the pad centre, metres (chord ≈ arc at pad
        // scale — same approximation as `TerrainFlatten::weight`).
        let offset = d * radius;
        let along = abs(dot(offset, region.along.xyz) - region.rect.x) - region.along.w;
        let across = abs(dot(offset, region.across.xyz) - region.rect.y) - region.across.w;
        // Signed interior distance to the rectangle edge (< 0 inside). The
        // override applies only INSIDE the flat rectangle — the smoothstep
        // ramp outside is already baked into the tiles, and re-applying it
        // here would double the blend and pull the rendered ramp metres off
        // the collider.
        let dist = max(along, across);
        if (dist >= 0.0) {
            continue;
        }
        // Flat tangent-plane elevation at `dir` (see `TerrainFlatten::
        // plane_elevation_m`), cancellation-free: 1 − cosθ = |d|²/2 exactly
        // for unit vectors.
        let one_minus_cos = 0.5 * dot(d, d);
        let cos_theta = 1.0 - one_minus_cos;
        let elev = region.center_elev.w
            + (radius + region.center_elev.w) * one_minus_cos / cos_theta;
        let feather = max(region.rect.w, 1.0e-3);
        let w = smoothstep(0.0, feather, -dist);
        out_height = mix(out_height, elev, w);
    }
    return out_height;
}

// Custom vertex entry: identical to `thalos_udlod::vertex::vertex` (the
// default the pipeline would otherwise use) plus the analytic pad flatten
// above. Keep the sample/blend sequence in lockstep with the udlod default.
//
// Gated out of the FRAGMENT compile (and the fragment entry out of the vertex
// compile, below): udlod's `Coordinate` struct gains `uv_dx`/`uv_dy` fields
// under the FRAGMENT def, so each stage's `Coordinate(...)` constructors are
// only valid under that stage's defs — an entry point is always kept by
// naga_oil, dragging the wrong-def constructors into the final module unless
// the off-stage entry is preprocessed away.
#ifndef FRAGMENT
@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var info = vertex_info(input);

    let tile = lookup_tile(info.coordinate, info.blend, 0u);
    var height = sample_height(tile);

    if (info.blend.ratio > 0.0) {
        let tile2 = lookup_tile(info.coordinate, info.blend, 1u);
        height = mix(height, sample_height(tile2), info.blend.ratio);
    }

    height = flattened_height(info.coordinate, height);

    return vertex_output(&info, height);
}
#endif

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
const BREAKUP_VALUE_AMT: f32 = 0.15; // ± fractional value variation (trimmed
                                     // 0.20 → 0.15 2026-07-03: now that the
                                     // footprint fade keeps it visible from the
                                     // air, full strength read as dark stains)
// Warm/cool drift. Kept low: a large value pushed bright patches toward
// red/brown, which read as muddy smears over the green. Stylized-vivid wants
// clean value patchiness, not a brown wash.
const BREAKUP_HUE_AMT: f32 = 0.035;  // ± warm/cool drift

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
// The shared camera exposure is calibrated for the one-world lighting spine;
// lunar albedos otherwise land close to display white after Hapke's opposition
// and backscatter response. Keep this regolith-only so mature dust sits in the
// middle-gray range without changing terrestrial radiance.
const REGOLITH_VALUE_TRIM: f32 = 0.55;

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
const LUSH_LO_M: f32 = 1500.0;       // full lowland/flank forest at/below here
const LUSH_HI_M: f32 = 2400.0;       // forest gone above here
const TREELINE_LO_M: f32 = 2400.0;   // alpine zone begins (cover greys out)
const TREELINE_HI_M: f32 = 3000.0;   // fully alpine: bare scree + patchy tundra
const SNOW_LINE_LO_M: f32 = 3100.0;  // snow begins (on gentle ground)
const SNOW_LINE_HI_M: f32 = 4000.0;  // permanent snow cap
const SNOW_LINE_NOISE_M: f32 = 400.0; // ± snow/treeline jitter from macro noise
const SNOW_SLOPE_LO: f32 = 0.32;     // snow holds on slopes up to ~47°…
const SNOW_SLOPE_HI: f32 = 0.62;     // …gone by ~68° (cliffs stay bare rock)
// How strongly the alpine zone greys out to bare scree above the treeline even
// on moderate ground (0 = pure tundra colour, 1 = full scree). The temperate
// look wants the upper mountain reading grey rock + snow, not green/tan.
const BARREN_STRENGTH: f32 = 0.65;
const MACRO_VAR_SCALE: f32 = 0.004;  // 1/period_m → ~250 m biome-mottle patches
const MACRO_VAR_AMT: f32 = 0.14;     // ± low-frequency value mottle
// Mid-scale moisture: lush ↔ dry grass (and soil in the driest spots) so flat
// lowland reads as a varied carpet, not one flat green. The MACRO moisture
// (regions / mosaics / stands, ≥ ~500 m) is NOT synthesised here — it is the
// f64 `ProceduralSurface::macro_moisture` field baked into the albedo
// attachment's alpha channel (docs/terrain_macro.md); the fragment decodes it
// and adds only `thalos::landcover::moisture_detail` (the wrapped ~125 m tier).
// Do not re-add coarse wrapped tiers: anything at km scale visibly tiles on
// the 4 km coordinate wrap — that was the "repeating leopard-print planet".
const MOISTURE_SCALE: f32 = 0.008;          // 1/period_m → ~125 m medium patches

// Earthy linear-space material anchors. Deliberately lower-value and less
// saturated than neon grass; direct sun and sky fill lift them in the lighting
// pass. Snow is kept well below 1.0 so a sunlit cap doesn't blow out.
// Stylized-vivid pass: chroma pushed back up from the prior desaturated anchors
// (which read as drab felt) — greens get a wider green-vs-red/blue gap so the
// land reads alive, without returning to the old "grating chartreuse". Still
// linear-space and below the lighting blow-out ceiling.
const C_FOREST: vec3<f32>   = vec3<f32>(0.034, 0.084, 0.028); // lowland lush (deep saturated green)
const C_GRASS: vec3<f32>    = vec3<f32>(0.072, 0.152, 0.050); // temperate grass (vivid olive-green)
const C_DRYGRASS: vec3<f32> = vec3<f32>(0.138, 0.150, 0.074); // dry grass (drier regions only)
const C_SOIL: vec3<f32>     = vec3<f32>(0.112, 0.074, 0.042); // earthy soil / peat (warmer)
const C_ROCK_LO: vec3<f32>  = vec3<f32>(0.108, 0.104, 0.098); // lower rock (near-neutral grey)
const C_ROCK_HI: vec3<f32>  = vec3<f32>(0.140, 0.143, 0.150); // alpine scree (cool grey)
const C_SNOW: vec3<f32>     = vec3<f32>(0.600, 0.640, 0.720); // snow (faint blue)
const C_WET: vec3<f32>      = vec3<f32>(0.028, 0.058, 0.026); // wet hollow (dark green)

// Beach strand (BL-10): the coastline is bare sand from the waterline up to
// the generator's +4 m berm top, painted from the RAW altitude (the physical
// coast — never the climate-shifted eco altitude) and gated off steep ground
// so cliff coasts stay rock. The band tracks the generator's `BEACH_RISE_M`;
// vegetation placement clears the same strip (`scatter::VEG_BEACH_CLEAR_M`).
const BEACH_TOP_LO_M: f32 = 3.0;   // sand starts fading out above this
const BEACH_TOP_HI_M: f32 = 5.0;   // fully vegetated ground above this
const BEACH_SLOPE_LO: f32 = 0.35;  // sand holds on gentle strands…
const BEACH_SLOPE_HI: f32 = 0.60;  // …cliff faces keep their rock
const WET_SAND_TOP_M: f32 = 0.8;   // swash-darkened sand below this
const C_BEACH_DRY: vec3<f32> = vec3<f32>(0.280, 0.243, 0.158); // dry strand sand
const C_BEACH_WET: vec3<f32> = vec3<f32>(0.152, 0.128, 0.086); // water-saturated sand

// Step 3 — micro-relief normal. A detail height field whose gradient tilts the
// lighting normal, giving the surface the light/dark micro-contrast under a
// grazing sun that separates "solid ground" from "luminous fog".
const DETAIL_SCALE: f32 = 0.8;            // 1/period_m → ~1.25 m base relief
const DETAIL_OCTAVES: i32 = 3;
const DETAIL_EPS: f32 = 0.25;             // finite-difference step, metres
const DETAIL_NORMAL_STRENGTH: f32 = 0.22; // facet-tilt amount. Derives from
                                          // gradient (Perlin) noise (no weave);
                                          // kept modest so a grazing (sunset) sun
                                          // doesn't churn the ground into sandpaper.
                                          // Trimmed: this only shows in extreme
                                          // close-ups (it fades out by ~1.8 km),
                                          // which isn't the play distance, so it
                                          // was just adding close-up speckle.

// Both detail layers fade out with camera distance: their period goes
// sub-pixel on the far field and would shimmer. The macro/slope colour carries
// the distance instead.
const DETAIL_FADE_NEAR: f32 = 180.0;  // full detail within this range (m)
const DETAIL_FADE_FAR: f32  = 1800.0; // no detail beyond this range (m)
const DETAIL_COORD_PERIOD_M: f32 = 4000.0;

// ── Grass-field detail (the "fluffy grass at any distance" layer) ───────────
// A TWO-SCALE value mottle + soft fluffy normal GATED on the grass mask,
// carrying the grassland texture from the blade clipmap's edge (~340 m) out to
// wherever each scale still resolves on screen. Scales fade INDIVIDUALLY by
// pixel footprint (Nyquist guard — see `OCTAVE_FADE_*` / `fbm3_*_faded`), not
// by camera distance, so an aerial camera keeps whichever scale it can still
// resolve — tufts up close, tussock clusters on a low pass, the 20 m breakup +
// 125 m moisture patches from cruise — and nothing shimmers (an octave
// dissolves to its mean exactly as it stops resolving). The NORMAL is what
// sells it under a grazing sun (light/dark tuft micro-contrast); the BRDF term
// (`GRASS_FIELD_TRANSLUCENCY`) adds the backlit sheen a real field shows.
// Amplitudes trimmed 2026-07-03 after the first screenshot pass: the initial
// values (0.20/0.13 clump/tussock, 0.20 breakup) read as camouflage blotches
// from the air rather than growth variation.
const GRASS_CLUMP_SCALE: f32 = 0.6;             // 1/period_m → ~1.7 m tufts
const GRASS_CLUMP_VALUE_AMT: f32 = 0.15;        // ± value mottle (tuft light/dark)
const GRASS_CLUMP_NORMAL_STRENGTH: f32 = 0.30;  // fluffy facet tilt (grazing read)
const GRASS_TUSSOCK_SCALE: f32 = 0.13;          // 1/period_m → ~7.7 m tussock clusters
const GRASS_TUSSOCK_VALUE_AMT: f32 = 0.09;      // ± value mottle (aerial patchiness)
const GRASS_TUSSOCK_NORMAL_STRENGTH: f32 = 0.20;
// Statistical backlit sheen of a grass canopy: per-fragment translucency weight
// fed into `shade_surface`'s foliage transmit lobe, scaled by the grass gate,
// so a low sun rims distant fields the way it rims the near blades.
const GRASS_FIELD_TRANSLUCENCY: f32 = 0.12;
// Grass mottle → ambient occlusion: the dark between-tuft hollows see less sky
// (root AO), which is what makes a field read dense rather than painted.
const GRASS_AO_FROM_DETAIL: f32 = 0.45;

// Per-octave screen-footprint (Nyquist) fade for the detail layers above and
// the Step-2 breakup: an octave is fully present while the fragment's
// body-space pixel footprint is below `OCTAVE_FADE_LO ×` its wavelength and
// gone by `OCTAVE_FADE_HI ×` it. This replaces fixed camera-distance fades for
// the *albedo* layers (which killed all mid-scale texture seen from the air —
// a cruise-altitude camera resolves 20 m patches just fine); normal-perturbing
// micro-relief keeps its distance fade (grazing-sun sparkle is footprint-safe
// but still reads busy).
const OCTAVE_FADE_LO: f32 = 0.12;
const OCTAVE_FADE_HI: f32 = 0.45;

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

// Large-scale valley ambient occlusion from the height atlas.
// Marches in 4 cardinal UV directions, finds the maximum horizon elevation
// angle in each, and derives a sky-visibility factor. A valley floor enclosed
// by high terrain on all sides loses ambient; an open hilltop stays bright.
// Applied only to the ambient term (surf.occlusion), not the direct sun.
const VALLEY_AO_STEPS: u32 = 8u;
const VALLEY_AO_STEP0_TEXELS: f32 = 2.0;
const VALLEY_AO_STEP_GROWTH: f32 = 1.7;
// At LOD where m_per_texel ≈ 15 m the march reaches ~2.5 km; at coarser LOD it
// scales proportionally, so the AO always captures mountain-valley distances.
const VALLEY_AO_STRENGTH: f32 = 0.55; // 0 = off, 1 = full black in enclosed valleys
const VALLEY_AO_MAX_ANGLE: f32 = 0.9; // horizon angle (rad, ~52°) that maps to full AO

fn terrain_valley_ao(tile: AtlasTile) -> f32 {
    let tex_size = attachments[0u].size;
    let inv_size = 1.0 / tex_size;
    let side_length = 3.14159265359 / 4.0 * config.scale;
    let m_per_texel = side_length / (tex_size * tile_count(tile.coordinate.lod));

    let base_uv = attachment_uv(tile.coordinate.uv, 0u);
    let h0 = sample_height_atlas_uv_m(tile, base_uv);

    let dirs = array<vec2<f32>, 4>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(-1.0, 0.0),
        vec2<f32>(0.0, -1.0),
    );

    var total_horizon = 0.0;
    for (var d = 0u; d < 4u; d = d + 1u) {
        var max_slope = 0.0;
        var dist_texels = 0.0;
        var step_sz = VALLEY_AO_STEP0_TEXELS;
        for (var k = 0u; k < VALLEY_AO_STEPS; k = k + 1u) {
            dist_texels = dist_texels + step_sz;
            let suv = base_uv + dirs[d] * (dist_texels * inv_size);
            let h = sample_height_atlas_uv_m(tile, suv);
            let d_m = dist_texels * m_per_texel;
            max_slope = max(max_slope, (h - h0) / max(d_m, 1.0e-3));
            step_sz = step_sz * VALLEY_AO_STEP_GROWTH;
        }
        total_horizon = total_horizon + atan(max(max_slope, 0.0));
    }

    let avg_horizon = total_horizon * 0.25;
    let occ = clamp(avg_horizon / VALLEY_AO_MAX_ANGLE, 0.0, 1.0);
    return 1.0 - occ * VALLEY_AO_STRENGTH;
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
//   eco_altitude_m — CLIMATE-SHIFTED altitude (`height + climate_cold_lift`),
//                    so the treeline/snow bands descend with latitude (polar
//                    caps at sea level) — docs/terrain_macro.md Phase 2
//   slope_t    — geometric steepness in [0,1] (0 flat, 1 vertical)
//   variation  — low-frequency value noise in [-1,1] (jitters bands, mottles)
//   warmth     — hot-climate weight (1 tropics → 0 cool): dry ground reads as
//                sand only while warm
fn eval_material_stack(
    masks_in: vec4<f32>,
    macro_albedo: vec3<f32>,
    eco_altitude_m: f32,
    altitude_m: f32,
    slope_t: f32,
    variation: f32,
    moisture: f32,
    warmth: f32,
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

    // Landcover classes from the moisture field, kept fairly contrasty so they
    // read as distinct patches from the air (as in real aerials): wet/low →
    // forest (dark canopy), mid → grassland, dry → tan dry grass, driest → bare
    // soil. Forest is the default cover up to the treeline; above it the cover
    // cools to alpine tundra and greys out to bare scree (the `barren` term).
    let alpine = smoothstep(TREELINE_LO_M, TREELINE_HI_M, eco_altitude_m + jitter);
    // The vegetated ground colour comes from the shared `thalos::landcover`
    // library — the SAME function the grass blades' CPU mirror
    // (`ground/landcover.rs`) reads — so the ground and the blades growing from
    // it are literally the same green. The macro-value mottle is applied to the
    // whole `ground` below (so it can be exempted under snow).
    let veg = vegetation_color(eco_altitude_m, moisture, variation, warmth);

    // Rock cools and greys with altitude (warm soil-stained rock low down,
    // lichen-free scree up high).
    let rock_col = mix(C_ROCK_LO, C_ROCK_HI, alpine);

    // Earthy substrate from the material masks, darkened in wet hollows.
    var ground = veg * grass_w + C_SOIL * soil_w + rock_col * rock_w;
    ground = mix(ground, C_WET, wet * (1.0 - rock_w * 0.55));

    // Alpine barren: above the treeline the alpine zone is mostly bare scree /
    // boulder field with only patchy tundra, so exposed grey rock takes over
    // even on MODERATE ground — not just on the steep faces `rock_w` already
    // catches. Without this, gentle high slopes read as the (now cool) alpine
    // tundra green; with it they grey out toward scree the way a real temperate
    // range does above the trees. Held off wet hollows and where rock is already
    // dominant. Snow then caps it above the snowline.
    let barren = alpine * (1.0 - wet) * (1.0 - 0.6 * rock_w);
    ground = mix(ground, C_ROCK_HI, barren * BARREN_STRENGTH);

    // Beach strand (BL-10): bare sand from the waterline to the berm top,
    // wet-darkened in the swash band. Overrides vegetation/soil (the strand IS
    // the substrate there) but sits under snow so polar shores can still ice
    // over; steep coastal faces keep their rock (cliff coasts).
    let beach = (1.0 - smoothstep(BEACH_TOP_LO_M, BEACH_TOP_HI_M, altitude_m))
        * (1.0 - smoothstep(BEACH_SLOPE_LO, BEACH_SLOPE_HI, steep));
    let wet_sand = 1.0 - smoothstep(0.15, WET_SAND_TOP_M, altitude_m);
    let sand_col = mix(C_BEACH_DRY, C_BEACH_WET, wet_sand);
    ground = mix(ground, sand_col, beach);

    // Snow: above a noise-jittered snowline, only where the ground is not too
    // steep. Permanent cap once well clear of the line.
    let snow_alt = smoothstep(SNOW_LINE_LO_M + jitter, SNOW_LINE_HI_M + jitter, eco_altitude_m);
    let snow_hold = 1.0 - smoothstep(SNOW_SLOPE_LO, SNOW_SLOPE_HI, steep);
    let snow = clamp(snow_alt * snow_hold, 0.0, 1.0);
    ground = mix(ground, C_SNOW, snow);

    // Low-frequency value mottle so even a single biome breaks into patches
    // (snow stays clean).
    ground = ground * (1.0 + variation * MACRO_VAR_AMT * (1.0 - snow));

    // Keep a little of the baked macro albedo as broad climate/body identity,
    // but the altitude/slope model is the local truth. Drop it under snow, and
    // keep the blend light so the (cooled, but still climate-warm) macro band
    // can't reintroduce a brown wash over the green/grey local model.
    let macro_tint = desaturate(macro_albedo, 0.86);
    var out: TerrainMaterialSample;
    out.albedo = mix(ground, macro_tint, 0.10 * (1.0 - snow));
    out.normal_strength = mix(mix(0.45, 0.85, rock_w) + soil_w * 0.12, 0.25, snow);
    // Wave-worked sand is smooth; damp the micro-relief on the strand.
    out.normal_strength = mix(out.normal_strength, 0.30, beach);
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

// ── Periodic gradient (Perlin) noise with analytic derivative ──────────────
// Value noise (above) interpolates random *scalars* on the integer lattice, so
// its gradient is strongly axis-aligned — the cubic grid shows through as a
// "weave" in any normal derived from it (which is why the detail normal was
// switched off). Gradient noise puts the randomness in per-corner *gradient
// vectors*, so its derivative is far more isotropic — the right basis for a
// detail normal. Periodic (corners wrapped on `period`) so it stays seamless
// across the floating-origin phase fold, and analytic-derivative so the normal
// costs a single evaluation. Derivative form from Inigo Quilez's
// gradient-noise-with-derivatives (https://iquilezles.org/articles/gradientnoise/).

// Hash a (wrapped) integer lattice corner to a pseudo-random gradient in
// [-1,1]^3. Unnormalised (gradient noise tolerates varied magnitudes and it
// avoids a per-corner normalize); the fBm renormalises by amplitude sum.
fn hash33(p_in: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p_in * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx) * 2.0 - 1.0;
}

// Returns vec4(value, d/dx, d/dy, d/dz). value ~ roughly [-1, 1].
fn perlin3_periodic_grad(x: vec3<f32>, period: f32) -> vec4<f32> {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let ga = hash33(wrap_lattice(i + vec3<f32>(0.0, 0.0, 0.0), period));
    let gb = hash33(wrap_lattice(i + vec3<f32>(1.0, 0.0, 0.0), period));
    let gc = hash33(wrap_lattice(i + vec3<f32>(0.0, 1.0, 0.0), period));
    let gd = hash33(wrap_lattice(i + vec3<f32>(1.0, 1.0, 0.0), period));
    let ge = hash33(wrap_lattice(i + vec3<f32>(0.0, 0.0, 1.0), period));
    let gf = hash33(wrap_lattice(i + vec3<f32>(1.0, 0.0, 1.0), period));
    let gg = hash33(wrap_lattice(i + vec3<f32>(0.0, 1.0, 1.0), period));
    let gh = hash33(wrap_lattice(i + vec3<f32>(1.0, 1.0, 1.0), period));

    let va = dot(ga, f - vec3<f32>(0.0, 0.0, 0.0));
    let vb = dot(gb, f - vec3<f32>(1.0, 0.0, 0.0));
    let vc = dot(gc, f - vec3<f32>(0.0, 1.0, 0.0));
    let vd = dot(gd, f - vec3<f32>(1.0, 1.0, 0.0));
    let ve = dot(ge, f - vec3<f32>(0.0, 0.0, 1.0));
    let vf = dot(gf, f - vec3<f32>(1.0, 0.0, 1.0));
    let vg = dot(gg, f - vec3<f32>(0.0, 1.0, 1.0));
    let vh = dot(gh, f - vec3<f32>(1.0, 1.0, 1.0));

    let value = va
        + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
        + u.x * u.y * (va - vb - vc + vd)
        + u.y * u.z * (va - vc - ve + vg)
        + u.z * u.x * (va - vb - ve + vf)
        + u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);

    let derivative = ga
        + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga)
        + u.x * u.y * (ga - gb - gc + gd)
        + u.y * u.z * (ga - gc - ge + gg)
        + u.z * u.x * (ga - gb - ge + gf)
        + u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh)
        + du * vec3<f32>(
            (vb - va) + u.y * (va - vb - vc + vd) + u.z * (va - vb - ve + vf)
                + u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh),
            (vc - va) + u.z * (va - vc - ve + vg) + u.x * (va - vb - vc + vd)
                + u.z * u.x * (-va + vb + vc - vd + ve - vf - vg + vh),
            (ve - va) + u.x * (va - vb - ve + vf) + u.y * (va - vc - ve + vg)
                + u.x * u.y * (-va + vb + vc - vd + ve - vf - vg + vh),
        );

    return vec4<f32>(value, derivative);
}

// fBm value + analytic gradient over gradient (Perlin) noise, matching the
// octave schedule of `fbm3_grad` (frequency doubles, amplitude halves; each
// octave's gradient is chain-rule-scaled by its frequency).
fn fbm3_perlin_grad(p_in: vec3<f32>, octaves: i32, period_in: f32) -> vec4<f32> {
    var p = p_in;
    var period = period_in;
    var amp = 0.5;
    var freq = 1.0;
    var sum = 0.0;
    var grad = vec3<f32>(0.0);
    var norm = 0.0;
    for (var o = 0; o < octaves; o = o + 1) {
        let vg = perlin3_periodic_grad(p, period);
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

// ── Footprint-faded fBm variants ────────────────────────────────────────────
// Same octave schedules as above, but each octave carries a visibility factor
// from the fragment's body-space pixel footprint vs. that octave's wavelength
// (the `OCTAVE_FADE_*` Nyquist window). Octaves are CENTRED (mean 0) and the
// normalisation is fixed, so as octaves stop resolving the pattern smoothly
// dissolves toward 0 — its true screen-space average — with no fade line and
// no shimmer, at any view geometry (grazing ground view or aerial nadir).

// Centred value-noise fBm with per-octave footprint fade. Returns ~[-1, 1],
// → 0 as the footprint swallows the octaves.
fn fbm3_value_faded(
    p_in: vec3<f32>,
    octaves: i32,
    period_in: f32,
    base_wavelength_m: f32,
    footprint_m: f32,
) -> f32 {
    var p = p_in;
    var period = period_in;
    var wl = base_wavelength_m;
    var amp = 0.5;
    var sum = 0.0;
    var norm = 0.0;
    for (var o = 0; o < octaves; o = o + 1) {
        let vis = 1.0 - smoothstep(wl * OCTAVE_FADE_LO, wl * OCTAVE_FADE_HI, footprint_m);
        sum = sum + amp * vis * (value_noise_3d_periodic(p, period) - 0.5);
        norm = norm + amp;
        p = p * 2.0;
        period = period * 2.0;
        wl = wl * 0.5;
        amp = amp * 0.5;
    }
    return 2.0 * sum / max(norm, 1.0e-5);
}

// Gradient (Perlin) fBm + analytic derivative with per-octave footprint fade.
// Perlin octaves are already centred. Returns vec4(value, gradient w.r.t p_in).
fn fbm3_perlin_grad_faded(
    p_in: vec3<f32>,
    octaves: i32,
    period_in: f32,
    base_wavelength_m: f32,
    footprint_m: f32,
) -> vec4<f32> {
    var p = p_in;
    var period = period_in;
    var wl = base_wavelength_m;
    var amp = 0.5;
    var freq = 1.0;
    var sum = 0.0;
    var grad = vec3<f32>(0.0);
    var norm = 0.0;
    for (var o = 0; o < octaves; o = o + 1) {
        let vis = 1.0 - smoothstep(wl * OCTAVE_FADE_LO, wl * OCTAVE_FADE_HI, footprint_m);
        let vg = perlin3_periodic_grad(p, period);
        sum = sum + amp * vis * vg.x;
        grad = grad + amp * vis * freq * vg.yzw;
        norm = norm + amp;
        p = p * 2.0;
        period = period * 2.0;
        wl = wl * 0.5;
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    let inv = 1.0 / max(norm, 1.0e-5);
    return vec4<f32>(sum * inv, grad * inv);
}

// `detail_height` value and gradient w.r.t. body-space metres. The fBm runs in
// scaled coordinates (`p_body * DETAIL_SCALE`), so one more `DETAIL_SCALE`
// factor folds into the gradient by the chain rule. Uses gradient (Perlin)
// noise so the resulting normal is weave-free.
fn detail_height_grad(p_body: vec3<f32>) -> vec4<f32> {
    let g = fbm3_perlin_grad(
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

fn surface_detail(
    p_body: vec3<f32>,
    geo_normal: vec3<f32>,
    cam_dist: f32,
    footprint_m: f32,
) -> SurfaceDetail {
    var out: SurfaceDetail;
    out.tint = vec3<f32>(1.0);
    out.normal_offset = vec3<f32>(0.0);

    // Step 2 — albedo breakup: patchy value variation + a subtle warm/cool hue
    // drift, so the macro colour stops reading as one flat tint. Faded per
    // octave by pixel footprint (not camera distance), so the ~20 m patchiness
    // survives aerial views for as long as it actually resolves on screen.
    let breakup_wl = 1.0 / BREAKUP_SCALE;
    if (footprint_m < breakup_wl * OCTAVE_FADE_HI) {
        let dv = fbm3_value_faded(
            p_body * BREAKUP_SCALE,
            3,
            DETAIL_COORD_PERIOD_M * BREAKUP_SCALE,
            breakup_wl,
            footprint_m,
        );
        let value_mul = 1.0 + dv * BREAKUP_VALUE_AMT;
        let hue = vec3<f32>(1.0 + dv * BREAKUP_HUE_AMT, 1.0, 1.0 - dv * BREAKUP_HUE_AMT);
        out.tint = vec3<f32>(value_mul) * hue;
    }

    // Step 3 — micro-relief normal from the gradient of a detail height field,
    // projected tangent to the sphere so it tilts facets without changing the
    // macro surface orientation. Magnitude is capped so a steep noise gradient
    // can't fold the normal past the horizon. The gradient is evaluated
    // analytically in a single fBm pass (`detail_height_grad`) — the previous
    // four finite-difference taps were the dominant per-fragment cost here.
    // Keeps its camera-distance fade: a ~1.25 m relief normal is a close-up
    // term, and normal perturbation reads busier at range than albedo mottle.
    let fade = 1.0 - smoothstep(DETAIL_FADE_NEAR, DETAIL_FADE_FAR, cam_dist);
    if (fade > 0.0) {
        let grad = detail_height_grad(p_body).yzw;
        let grad_t = grad - geo_normal * dot(grad, geo_normal);
        var off = -grad_t * (DETAIL_NORMAL_STRENGTH * fade);
        let off_len = length(off);
        out.normal_offset = off * (0.8 / max(0.8, off_len));
    }
    return out;
}

// Grass-field detail for the VEGETATED ground (see the GRASS_CLUMP_* /
// GRASS_TUSSOCK_* constants): the band-2 "grass as terrain shading" layer of
// the vegetation cascade (docs/vegetation.md §5.0). Two gradient-noise scales
// give BOTH the value mottle (`.x`) and the fluffy normal (`.yzw`); each scale
// fades by pixel footprint, so the grassland stays textured from any altitude
// at whichever scale still resolves. `grass_w` is the grass coverage at this
// fragment (0 off grass / above the treeline → early-out, so non-grass ground
// pays nothing; the footprint guard skips it once even the tussock scale is
// sub-pixel).
fn grass_field_detail(
    p_body: vec3<f32>,
    geo_normal: vec3<f32>,
    grass_w: f32,
    footprint_m: f32,
) -> SurfaceDetail {
    var out: SurfaceDetail;
    out.tint = vec3<f32>(1.0);
    out.normal_offset = vec3<f32>(0.0);

    let gate = clamp(grass_w, 0.0, 1.0);
    let tussock_wl = 1.0 / GRASS_TUSSOCK_SCALE;
    if (gate <= 1.0e-3 || footprint_m >= tussock_wl * OCTAVE_FADE_HI) {
        return out;
    }

    let clump = fbm3_perlin_grad_faded(
        p_body * GRASS_CLUMP_SCALE,
        2,
        DETAIL_COORD_PERIOD_M * GRASS_CLUMP_SCALE,
        1.0 / GRASS_CLUMP_SCALE,
        footprint_m,
    );
    let tussock = fbm3_perlin_grad_faded(
        p_body * GRASS_TUSSOCK_SCALE,
        2,
        DETAIL_COORD_PERIOD_M * GRASS_TUSSOCK_SCALE,
        tussock_wl,
        footprint_m,
    );

    // Value mottle: brighter sunlit tuft tops / darker shaded between-tuft
    // hollows (clump scale), grouped into larger light/dark growth patches
    // (tussock scale) — the aerial "lush field" texture.
    let dv = clamp(clump.x, -1.0, 1.0) * GRASS_CLUMP_VALUE_AMT
        + clamp(tussock.x, -1.0, 1.0) * GRASS_TUSSOCK_VALUE_AMT;
    out.tint = mix(vec3<f32>(1.0), vec3<f32>(1.0 + dv), gate);

    // Soft fluffy normal: tilt facets so a low sun rakes the grass texture (the
    // grazing-angle "fluffy" read). Chain rule folds each scale back into
    // body-metre gradients; tangent to the sphere; magnitude capped so a steep
    // gradient can't fold the normal past the horizon.
    let grad = clump.yzw * (GRASS_CLUMP_SCALE * GRASS_CLUMP_NORMAL_STRENGTH)
        + tussock.yzw * (GRASS_TUSSOCK_SCALE * GRASS_TUSSOCK_NORMAL_STRENGTH);
    let grad_t = grad - geo_normal * dot(grad, geo_normal);
    var off = -grad_t * gate;
    let off_len = length(off);
    out.normal_offset = off * (0.8 / max(0.8, off_len));
    return out;
}

// Airless regolith fine detail: value-only albedo mottle + speckle and a
// micro-relief normal. Same body-fixed coordinate basis as `surface_detail`, so
// it stays glued to the surface under time warp / floating-origin shifts, and
// it fades with distance to avoid sub-pixel shimmer (the macro albedo carries
// the far field).
fn regolith_detail(
    p_body: vec3<f32>,
    geo_normal: vec3<f32>,
    cam_dist: f32,
    footprint_m: f32,
) -> SurfaceDetail {
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
    let mottle = fbm3_value_faded(
        p_body * REGOLITH_MOTTLE_SCALE,
        4,
        DETAIL_COORD_PERIOD_M * REGOLITH_MOTTLE_SCALE,
        1.0 / REGOLITH_MOTTLE_SCALE,
        footprint_m,
    );
    let speckle = fbm3_value_faded(
        p_body * REGOLITH_SPECKLE_SCALE,
        3,
        DETAIL_COORD_PERIOD_M * REGOLITH_SPECKLE_SCALE,
        1.0 / REGOLITH_SPECKLE_SCALE,
        footprint_m,
    );
    let value_mul = 1.0 + (mottle * REGOLITH_MOTTLE_AMT + speckle * REGOLITH_SPECKLE_AMT) * fade;
    out.tint = vec3<f32>(REGOLITH_VALUE_TRIM * clamp(value_mul, 0.45, 1.8));

    // Micro-relief normal from the fine detail height gradient (~1.25 m), tangent
    // to the sphere so it tilts facets toward/away from the low sun — the pocked
    // light/dark micro-contrast that makes regolith read as solid cratered ground.
    let detail_grad = fbm3_perlin_grad_faded(
        p_body * DETAIL_SCALE,
        DETAIL_OCTAVES,
        DETAIL_COORD_PERIOD_M * DETAIL_SCALE,
        1.0 / DETAIL_SCALE,
        footprint_m,
    );
    let grad = detail_grad.yzw * DETAIL_SCALE;
    let grad_t = grad - geo_normal * dot(grad, geo_normal);
    var off = -grad_t * (REGOLITH_NORMAL_STRENGTH * fade);
    let off_len = length(off);
    out.normal_offset = off * (0.7 / max(0.7, off_len));
    return out;
}

// Ambient occlusion from the procedural cavity field. The Step-2 albedo breakup
// (`detail.tint`) doubles as a cheap cavity signal — darker breakup ≈ a hollow —
// so we fold its luminance into an AO factor on the sky/ground ambient (never on
// the direct sun, which has its own shadow terms). `AO_FROM_DETAIL` is the blend
// amount, `AO_MIN` the floor so creases darken without going fully black.
const AO_FROM_DETAIL: f32 = 0.65;
const AO_MIN: f32 = 0.15;

// Canopy AO: a tree/object that occludes the sun also blocks a chunk of the sky
// dome, so its sun-shadow footprint should darken the *ambient* term too — not
// just the direct beam. Without this the only thing a shadow removes is direct
// sun, which the bright hemisphere fill then washes back out (shadows read as
// barely-there). Bled from the sun-shadow factor ONLY (terrain self-shadow is
// excluded — a self-shadowed slope still sees open sky). 0 = no bleed, 1 = a
// fully tree-shadowed pixel loses all ambient.
const AMBIENT_SHADOW_BLEED: f32 = 0.6;

// Gated on FRAGMENT for the same reason the vertex entry is gated off it —
// see the note above `fn vertex`.
#ifdef FRAGMENT
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

    // Distant schematic (orbital map terrain). The map view draws the whole body
    // from far outside at MAP_SCALE, where one screen pixel integrates many km of
    // terrain. The baked normal atlas has no mip chain and the screen-space
    // specular-AA widening is clamped (tuned for ground viewing), so the sharp
    // GGX highlight aliases into a crawling gleam as the camera moves. The map is
    // a stand-in for the smooth impostor across the LOD swap, so shade its
    // specular fully matte — the relief/diffuse topography stays.
    let distant_schematic = terrain_extras.inspection.z >= 0.5;

    // Procedural surface detail (Step 2 breakup + Step 3 micro-relief normal),
    // synthesised from body-fixed metres so it remains static under time warp.
    let frag_relative_position = -info.view_vector;
    let body_relative_position = quat_rotate(terrain_extras.debug.world_to_body_rot, frag_relative_position);
    let detail_p_body = terrain_extras.debug.view_phase.xyz + body_relative_position;
    // Body-space size of one screen pixel at this fragment — the Nyquist input
    // for the per-octave detail fades. Computed here, in uniform control flow,
    // before any branch (derivative-builtin requirement).
    let footprint_m = max(
        length(dpdx(detail_p_body)),
        length(dpdy(detail_p_body)),
    );
    let detail = surface_detail(detail_p_body, geo_normal, cam_dist, footprint_m);

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
    // Grass-field outputs, set in the vegetated branch (see `grass_field_detail`)
    // and consumed by the dielectric lighting below: the fluffy normal, the
    // normalized grass gate (drives the field translucency), and the between-tuft
    // root-AO factor folded into the ambient cavity term.
    var grass_normal_offset = vec3<f32>(0.0);
    var grass_gate = 0.0;
    var grass_cavity = 1.0;

    var surface_rgb: vec3<f32>;
    if (style_regolith) {
        // Airless regolith: baked gray albedo (the body's authored mare/highland
        // tone) textured by value-only mottle + speckle; no ecological bands, no
        // hue drift. The micro-relief normal is fed to the lighting below.
        if (debug_on) {
            surface_rgb = albedo.rgb * REGOLITH_VALUE_TRIM;
        } else {
            let regolith_footprint = select(
                footprint_m,
                0.0,
                terrain_extras.inspection.x >= 2.5,
            );
            let rd = regolith_detail(
                detail_p_body,
                geo_normal,
                cam_dist,
                regolith_footprint,
            );
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
        // Moisture = baked macro (albedo attachment alpha, the planet-scale
        // f64 field — no 4 km tiling) + the wrapped fine detail tier from the
        // shared `thalos::landcover` library (mirrored CPU-side by
        // `ground/landcover.rs`), so the grass blades read the exact same field.
        let macro_var = macro_variation(detail_p_body);
        let moisture =
            clamp(albedo.a * 2.0 - 1.0 + moisture_detail(detail_p_body), -1.0, 1.0);
        // Climate: the cube-sphere coordinate gives the body-fixed unit
        // direction (terrain model space is body-fixed), so latitude — and the
        // cold lift that descends the ecological bands toward the poles — is
        // exact per fragment. docs/terrain_macro.md Phase 2.
        let body_dir = compute_local_position(info.coordinate);
        let cold_lift = climate_cold_lift(body_dir.y);
        let warmth = climate_warmth(cold_lift);
        let eco_altitude_m = altitude_m + cold_lift;
        material = eval_material_stack(
            material_masks,
            grade_surface(albedo.rgb, material_masks.b),
            eco_altitude_m,
            altitude_m,
            geo_slope_t,
            macro_var,
            moisture,
            warmth,
        );
        surface_rgb = material.albedo;
        if (!debug_on) {
            surface_rgb = surface_rgb * detail.tint;
            // Grass-field detail (band 2 of the vegetation cascade): textures the
            // mid/far grassland so it reads as a fluffy grass surface (not flat
            // green) and hides the blade→albedo line. Gated on the normalized
            // grass mask, faded out above the treeline so alpine scree isn't
            // grass-textured.
            let grass_w_frag = material_masks.r
                / max(material_masks.r + material_masks.g + material_masks.b, 1e-3);
            // Gated off the beach strand too (BL-10) — the baked grass mask
            // doesn't know sea level, and grass-textured sand reads as algae.
            grass_gate = grass_w_frag
                * (1.0 - smoothstep(TREELINE_LO_M, TREELINE_HI_M, eco_altitude_m))
                * smoothstep(BEACH_TOP_LO_M, BEACH_TOP_HI_M, altitude_m);
            let gd = grass_field_detail(detail_p_body, geo_normal, grass_gate, footprint_m);
            surface_rgb = surface_rgb * gd.tint;
            grass_normal_offset = gd.normal_offset;
            // Root AO: the dark between-tuft mottle also sees less sky. Kept off
            // the direct sun (cavity feeds the ambient terms only, below).
            let gd_luma = dot(gd.tint, vec3<f32>(0.2126, 0.7152, 0.0722));
            grass_cavity = mix(1.0, min(gd_luma, 1.0), GRASS_AO_FROM_DETAIL);
        }
    }
    // The attachment alpha carries baked macro moisture, not opacity — force
    // the output alpha opaque from here on.
    albedo = vec4<f32>(surface_rgb, 1.0);

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

    if (
        terrain_extras.inspection.x >= 0.5
        && terrain_extras.inspection.x < 1.5
    ) {
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
            normal = normalize(
                height_n + detail.normal_offset * material.normal_strength + grass_normal_offset,
            );
        }
    }
    let view_dir = normalize(info.view_vector);

    // Render-time shadowing: the shared cascaded sun-shadow rig (craft, trees,
    // rocks, structures — one shadow world) combined with a terrain self-shadow
    // horizon march over the resident height atlas. Both fade only the direct
    // sun term; the sky fill below stands in for skylight reaching shadowed
    // ground.
    var self_shadow = 1.0;
    // Skip the height-atlas self-shadow march on the orbital map: at planetary
    // distance it samples the height atlas sub-pixel and its result shifts as the
    // LOD tiles change under camera motion, adding to the crawling shimmer.
    if (!debug_on && !distant_schematic) {
        self_shadow = terrain_self_shadow(tile, geo_normal, sun_dir_ws);
    }
    // Tree/craft/structure directional shadows from the sun-shadow rig fold into
    // the same direct-sun gate as the self-shadow march. `tree_shadow` is kept
    // separate so it can also bleed into the ambient term (canopy AO) below —
    // terrain self-shadow must not, so it's excluded there. Normal-aware sampler
    // (stable-CSM): the coarse relief normal drives the receiver offset +
    // slope-scaled bias; the detail-mapped normal would wobble the offset.
    let tree_shadow = sun_shadow_factor_nrm(
        hit_ws, height_n, terrain_extras.shadow,
        sun_shadow_map_0, sun_shadow_map_1, sun_shadow_map_2,
    );
    let external_shadow = self_shadow * tree_shadow;

    // Surface lighting. The shading normal is pulled most of the way toward the
    // relief normal but kept anchored to the geometric normal so steep micro-
    // facets near the terminator can't out-light the body curvature.
    //
    // On the orbital map (`distant_schematic`) the shading normal is forced to the
    // pure geometric sphere normal. At planetary distance the baked terrain normal
    // is sub-pixel — its high-frequency tilt aliases the *diffuse* shading into a
    // crawling gleam as the camera moves (matte specular alone can't fix it). The
    // per-pixel-averaged normal at that distance IS the sphere normal, which is
    // also why the impostor this stands in for doesn't shimmer. Visible relief on
    // the map then comes from the baked albedo bands, not normal shading.
    let stable_normal = select(
        normalize(mix(geo_normal, normal, 0.85)),
        geo_normal,
        distant_schematic || (
            terrain_extras.inspection.x >= 1.5
            && terrain_extras.inspection.x < 2.5
        ),
    );

    // Geometric specular AA: measure the shading normal's screen-space variance
    // once here — in uniform control flow, before the regolith/dielectric branch
    // (the derivative builtins require it) — then widen each path's specular
    // roughness to cover the sub-pixel normal cone, killing highlight sparkle.
    let spec_aa_var = specular_aa_variance(stable_normal);

    // Build the canonical surface description and shade through the shared
    // `thalos::lighting::shade_surface`. Everything terrain-specific (the
    // ecological albedo, the AA-widened roughness, the wet-hollow tightening, the
    // atmosphere-derived sky environment, the cavity/canopy occlusion) is still
    // resolved here per-fragment; `shade_surface` owns only the BRDF composition,
    // so the regolith impostor, the vegetated ground, and (later) foliage/ships
    // all reconverge on one lighting path. Atmosphere/aerial-perspective is
    // composited on top by the `BodySky` pass, not here.
    var lit: vec3<f32>;
    var surf: ThalosSurface;
    surf.albedo = albedo.rgb;
    surf.normal_ws = stable_normal;
    surf.geo_normal_ws = geo_normal;
    surf.emissive = vec3<f32>(0.0);
    surf.metallic = 0.0;
    surf.translucency = 0.0;
    if (style_regolith) {
        // Airless regolith: Hapke radiative-transfer BRDF — the exact routine the
        // orbital impostor uses, so the two render paths shade identically at the
        // impostor↔ground LOD swap. No atmospheric sky fill (airless); ambient
        // comes from the scene floor inside the Hapke helper. Roughness drives the
        // opposition-surge width; dry regolith has no wet-hollow tightening.
        surf.style = SURFACE_REGOLITH;
        surf.roughness = specular_aa_apply(clamp(roughness, 0.06, 1.0), spec_aa_var);
        surf.occlusion = 1.0;
        // Regolith ignores the sky environment (airless); pass a zeroed one.
        let no_sky = SurfaceSky(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0));
        lit = shade_surface(
            surf, view_dir, hit_ws, sun_dir_ws, sun_flux,
            terrain_scene, no_sky, external_shadow, 1.0,
        );
    } else {
        // Rough-dielectric surface: direct sun + an atmosphere-derived hemisphere
        // sky IBL (blue sky-dome + warm ground bounce) + a subtle ambient sky
        // specular. The sky model is shared with the grass shader through
        // `thalos::lighting` so the two can't drift.
        surf.style = SURFACE_DIELECTRIC;

        // Grass-field backlit sheen: a statistical translucency for the blade
        // canopy the ground stands in for at distance, so a low sun rims far
        // fields the way it rims the near blades (shade_surface's transmit
        // lobe; 0 off grass, so bare soil/rock/snow are untouched).
        surf.translucency = GRASS_FIELD_TRANSLUCENCY * grass_gate;

        // Specular roughness from the sampled attachment, tightened in wet
        // hollows (material mask .a) so puddles and wet rock get a sharper
        // highlight while dry, rough ground stays matte. Clamped away from 0 to
        // keep the GGX lobe from collapsing to a firefly. Forced fully matte on
        // the orbital map (see `distant_schematic`) so the highlight can't alias.
        let wetness = clamp(material_masks.a, 0.0, 1.0);
        surf.roughness = select(
            specular_aa_apply(
                clamp(mix(roughness, roughness * 0.45, wetness), 0.06, 1.0),
                spec_aa_var,
            ),
            1.0,
            distant_schematic,
        );

        // Lighting environment from the bound atmosphere. The vertical Rayleigh
        // optical depth τ_v = β_R · H_R is independent of the render-unit scale,
        // and the strength gate matches the live `BodySky` value, so the surface
        // ambient tracks the sky dome through sunrise → noon → sunset.
        let tau_zenith = terrain_atmos.rayleigh_beta_h.xyz * terrain_atmos.rayleigh_beta_h.w;
        let sky = compute_surface_sky(
            tau_zenith,
            terrain_atmos.atmos_geom.z,
            geo_normal,
            sun_dir_ws,
            sun_flux,
        );

        // Cavity AO from the procedural detail breakup × the material mask's
        // occlusion, applied (inside `shade_surface`) to the ambient terms only.
        let detail_luma = dot(detail.tint, vec3<f32>(0.2126, 0.7152, 0.0722));
        let cavity = clamp(
            mix(1.0, detail_luma, AO_FROM_DETAIL) * grass_cavity,
            AO_MIN,
            1.0,
        );
        // Large-scale valley AO: enclosed valley floors lose ambient light because
        // high surrounding terrain blocks the sky hemisphere — mul onto cavity so
        // both small-scale hollows and large-scale valleys compound correctly.
        let valley_ao = select(terrain_valley_ao(tile), 1.0, distant_schematic);
        // Screen-space AO (F5) compounds with the analytic cavity/valley terms —
        // it adds the object-vs-terrain contact occlusion (ship in a valley, base
        // of a building) the analytic terms can't see. Ambient-only, like the rest.
        let ssao = screen_space_ao(input.clip_position.xy);
        surf.occlusion = clamp(material.occlusion * cavity * valley_ao * ssao, 0.0, 1.0);

        // Canopy AO: a tree/object shadow bleeds into the ambient too (a canopy
        // overhead blocks the sky, not only the sun), so shadowed ground reads.
        let canopy_ambient = mix(1.0, tree_shadow, AMBIENT_SHADOW_BLEED);

        lit = shade_surface(
            surf, view_dir, hit_ws, sun_dir_ws, sun_flux,
            terrain_scene, sky, external_shadow, canopy_ambient,
        );
    }

    var output: FragmentOutput;
    output.color = vec4<f32>(lit, albedo.a);
    // F5 diagnostic (`THALOS_SSAO=show`, inspection.w = 2): paint the raw AO
    // value so artifacts can be attributed to the AO pass vs its application.
    if terrain_extras.inspection.w >= 1.5 {
        let ao_raw = screen_space_ao(input.clip_position.xy);
        output.color = vec4<f32>(vec3<f32>(ao_raw), albedo.a);
    }
    return output;
}

#endif
