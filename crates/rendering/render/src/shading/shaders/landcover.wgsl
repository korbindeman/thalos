// Shared large-scale landcover field — the SINGLE definition of the vegetated
// ground colour, so the terrain ground and the grass blades read the *same*
// green (physically they are the same material). The terrain calls this for its
// vegetated albedo; the CPU mirror `ground/landcover.rs` reproduces it
// bit-for-bit for grass placement / blade tint. Keep the two in lockstep.
//
// Pure analytic function of the body-fixed position (metres) + altitude. The
// noise is the same value-noise fbm the terrain detail uses, periodic at
// `DETAIL_COORD_PERIOD_M`; callers pass a position already reduced near the
// origin (the terrain's `detail_p_body`), and because every `PERIOD × SCALE`
// is an integer the wrapped lattice is phase-invariant.

#define_import_path thalos::landcover

// SCALE OWNERSHIP (docs/world/terrain_macro.md): this wrapped f32 field carries only
// FINE detail (≤ ~250 m). The macro moisture (≥ ~500 m — regions, mosaics,
// stands) is the f64 `ProceduralSurface::macro_moisture` field, baked into the
// tile albedo attachment's alpha channel; callers decode it there and add
// `moisture_detail` on top. Do not re-add coarse tiers here — anything at km
// scale visibly tiles on the 4 km coordinate wrap.

const DETAIL_COORD_PERIOD_M: f32 = 4000.0;
const MOISTURE_SCALE: f32 = 0.008;          // ~125 m medium patches
// Signed amplitude of the fine moisture deviation added onto the baked macro
// (matches the old medium tier's share of the full contrasted field).
const MOISTURE_DETAIL_AMT: f32 = 0.30;
const MACRO_VAR_SCALE: f32 = 0.004;         // ~250 m mottle
// Post-slim amplitude of the fine variation tier (the wrapped ~1 km region
// tier moved into the baked albedo's tone mottle).
const MACRO_VAR_FINE_AMT: f32 = 0.6;
const SNOW_LINE_NOISE_M: f32 = 400.0;

// Temperate altitude bands: forest dominant on the lower flanks up to a
// treeline ~2.4 km, then cool alpine tundra (grey scree is exposed on top of
// this in the terrain's `eval_material_stack`), then snow. Forest is the
// DEFAULT cover below the treeline (not gated to valley floors), so mountains
// read green → grey → white the way a wet temperate range does, rather than
// brown dry-grass across the whole mid-flank.
const LUSH_LO_M: f32 = 1500.0;
const LUSH_HI_M: f32 = 2400.0;
const TREELINE_LO_M: f32 = 2400.0;
const TREELINE_HI_M: f32 = 3000.0;

const C_FOREST: vec3<f32>   = vec3<f32>(0.034, 0.084, 0.028);
const C_GRASS: vec3<f32>    = vec3<f32>(0.072, 0.152, 0.050);
const C_DRYGRASS: vec3<f32> = vec3<f32>(0.138, 0.150, 0.074);
const C_SOIL: vec3<f32>     = vec3<f32>(0.112, 0.074, 0.042);
// Hot-desert sand: the driest ground in a WARM climate (see `climate_warmth`);
// cold dry ground stays tan steppe / soil instead.
const C_SAND: vec3<f32>     = vec3<f32>(0.225, 0.190, 0.125);
// Alpine tundra / sparse meadow above the treeline: cool, desaturated grey-green
// — the living cover between bare scree patches. Replaces the old tan dry-grass
// alpine tint that read as brown across the upper mountain.
const C_ALPINE: vec3<f32>   = vec3<f32>(0.082, 0.094, 0.074);

// ── Climate (latitude → cold lift / warmth) ────────────────────────────────
// Mirror of `thalos_terrain::procedural::{climate_cold_lift_m, climate_warmth}`
// (docs/world/terrain_macro.md Phase 2) — keep the constants in lockstep. The cold
// lift is how many metres the ecological altitude bands (lush belt, treeline,
// snowline) descend at a latitude; consumers pass `altitude_m + cold_lift`
// wherever a band threshold is compared.
const CLIMATE_COLD_LIFT_MAX_M: f32 = 3600.0;
const CLIMATE_LAT_LO: f32 = 0.45;
const CLIMATE_LAT_SPAN: f32 = 0.55;
const CLIMATE_LAT_POW: f32 = 2.6;
const CLIMATE_WARM_LO_M: f32 = 500.0;
const CLIMATE_WARM_HI_M: f32 = 1600.0;

// Metres the ecological bands descend at `sin_lat = |body_dir.y|`. The power
// curve keeps mid-latitudes mild (green at 45°) and makes ice a polar feature.
fn climate_cold_lift(sin_lat: f32) -> f32 {
    let t = clamp((abs(sin_lat) - CLIMATE_LAT_LO) / CLIMATE_LAT_SPAN, 0.0, 1.0);
    return CLIMATE_COLD_LIFT_MAX_M * pow(t, CLIMATE_LAT_POW);
}

// Hot-climate weight in [0, 1] from the cold lift (1 = tropics, 0 = cool).
fn climate_warmth(cold_lift_m: f32) -> f32 {
    return 1.0 - smoothstep(CLIMATE_WARM_LO_M, CLIMATE_WARM_HI_M, cold_lift_m);
}

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

// Fine moisture deviation (~125 m patches), signed. ADD to the baked macro
// moisture (albedo attachment alpha, decoded `a * 2 - 1`) and clamp — never a
// standalone moisture value.
fn moisture_detail(p: vec3<f32>) -> f32 {
    let lc_med = fbm3_periodic(p * MOISTURE_SCALE, 3, DETAIL_COORD_PERIOD_M * MOISTURE_SCALE);
    return (lc_med - 0.5) * 2.0 * MOISTURE_DETAIL_AMT;
}

// Fine value variation in [-1, 1] (jitters the bands, mottles tone) — the
// ~250 m tier only; regional tone drift lives in the baked albedo.
fn macro_variation(p: vec3<f32>) -> f32 {
    let macro_fine = (fbm3_periodic(p * MACRO_VAR_SCALE, 3, DETAIL_COORD_PERIOD_M * MACRO_VAR_SCALE) - 0.5) * 2.0;
    return clamp(macro_fine * MACRO_VAR_FINE_AMT, -1.0, 1.0);
}

// The vegetated ground colour (linear, NO macro-value mottle — the caller
// applies that to the whole ground so it can exempt snow). This is the exact
// `veg` branch of the terrain's `eval_material_stack`; the grass blade tint is
// this × the mottle, so blade and ground are the same material.
//
// `eco_altitude_m` is the CLIMATE-SHIFTED altitude (`altitude_m +
// climate_cold_lift(sin_lat)`), so the lush belt / treeline descend with
// latitude. `warmth` (from `climate_warmth`) turns the driest ground into
// hot-desert sand in warm climates and leaves it steppe/soil in cold ones.
fn vegetation_color(eco_altitude_m: f32, moisture: f32, macro_var: f32, warmth: f32) -> vec3<f32> {
    let jitter = macro_var * SNOW_LINE_NOISE_M;
    let lush = smoothstep(LUSH_HI_M, LUSH_LO_M, eco_altitude_m + jitter);
    let alpine = smoothstep(TREELINE_LO_M, TREELINE_HI_M, eco_altitude_m + jitter);
    let dryness = clamp(0.5 - 0.5 * moisture, 0.0, 1.0);
    // Forest is the default cover below the treeline across a wide moisture
    // range; only genuinely dry ground reads as grass, and only the driest as
    // tan dry-grass / bare soil / (warm) sand.
    let forest_amt = smoothstep(0.58, 0.28, dryness) * lush;
    var grass_c = mix(C_GRASS, C_DRYGRASS, smoothstep(0.55, 0.88, dryness));
    grass_c = mix(grass_c, C_SOIL, smoothstep(0.88, 0.98, dryness));
    grass_c = mix(grass_c, C_SAND, smoothstep(0.80, 0.95, dryness) * warmth);
    var veg = mix(grass_c, C_FOREST, forest_amt);
    // Above the treeline the cover cools to alpine tundra (grey scree is
    // exposed on top in `eval_material_stack`), not tan dry-grass.
    veg = mix(veg, C_ALPINE, alpine);
    return veg;
}
