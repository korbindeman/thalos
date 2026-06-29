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

const DETAIL_COORD_PERIOD_M: f32 = 4000.0;
const MOISTURE_SCALE: f32 = 0.008;          // ~125 m medium patches
const LANDCOVER_COARSE_SCALE: f32 = 0.002;  // ~500 m stands
const LANDCOVER_REGION_SCALE: f32 = 0.001;  // ~1 km lush/dry regions
const MACRO_VAR_SCALE: f32 = 0.004;         // ~250 m mottle
const MACRO_REGION_SCALE: f32 = 0.001;      // ~1 km tone drift
const MOISTURE_CONTRAST: f32 = 1.35;
const SNOW_LINE_NOISE_M: f32 = 400.0;

const LUSH_LO_M: f32 = 1800.0;
const LUSH_HI_M: f32 = 2900.0;
const TREELINE_LO_M: f32 = 3100.0;
const TREELINE_HI_M: f32 = 4000.0;

const C_FOREST: vec3<f32>   = vec3<f32>(0.034, 0.084, 0.028);
const C_GRASS: vec3<f32>    = vec3<f32>(0.072, 0.152, 0.050);
const C_DRYGRASS: vec3<f32> = vec3<f32>(0.142, 0.158, 0.072);
const C_SOIL: vec3<f32>     = vec3<f32>(0.112, 0.074, 0.042);

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

// Moisture in [-1, 1] (+ wet, − dry) — the 3-scale landcover field.
fn moisture_at(p: vec3<f32>) -> f32 {
    let lc_region = fbm3_periodic(p * LANDCOVER_REGION_SCALE, 2, DETAIL_COORD_PERIOD_M * LANDCOVER_REGION_SCALE);
    let lc_coarse = fbm3_periodic(p * LANDCOVER_COARSE_SCALE, 3, DETAIL_COORD_PERIOD_M * LANDCOVER_COARSE_SCALE);
    let lc_med = fbm3_periodic(p * MOISTURE_SCALE, 3, DETAIL_COORD_PERIOD_M * MOISTURE_SCALE);
    let raw = mix(mix(lc_region, lc_coarse, 0.45), lc_med, 0.22);
    return clamp((raw - 0.5) * 2.0 * MOISTURE_CONTRAST, -1.0, 1.0);
}

// Low-frequency value variation in [-1, 1] (jitters the bands, mottles tone).
fn macro_variation(p: vec3<f32>) -> f32 {
    let macro_fine = (fbm3_periodic(p * MACRO_VAR_SCALE, 3, DETAIL_COORD_PERIOD_M * MACRO_VAR_SCALE) - 0.5) * 2.0;
    let macro_region = (fbm3_periodic(p * MACRO_REGION_SCALE, 2, DETAIL_COORD_PERIOD_M * MACRO_REGION_SCALE) - 0.5) * 2.0;
    return clamp(mix(macro_fine, macro_region, 0.55), -1.0, 1.0);
}

// The vegetated ground colour (linear, NO macro-value mottle — the caller
// applies that to the whole ground so it can exempt snow). This is the exact
// `veg` branch of the terrain's `eval_material_stack`; the grass blade tint is
// this × the mottle, so blade and ground are the same material.
fn vegetation_color(altitude_m: f32, moisture: f32, macro_var: f32) -> vec3<f32> {
    let jitter = macro_var * SNOW_LINE_NOISE_M;
    let lush = smoothstep(LUSH_HI_M, LUSH_LO_M, altitude_m + jitter);
    let alpine = smoothstep(TREELINE_LO_M, TREELINE_HI_M, altitude_m + jitter);
    let dryness = clamp(0.5 - 0.5 * moisture, 0.0, 1.0);
    let forest_amt = smoothstep(0.46, 0.20, dryness) * lush;
    var grass_c = mix(C_GRASS, C_DRYGRASS, smoothstep(0.40, 0.78, dryness));
    grass_c = mix(grass_c, C_SOIL, smoothstep(0.80, 0.96, dryness));
    var veg = mix(grass_c, C_FOREST, forest_amt);
    veg = mix(veg, C_DRYGRASS, alpine);
    return veg;
}
