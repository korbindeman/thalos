// NTR-X1 tile-terrain surface material (ExtendedMaterial fragment).
//
// The keystone rule ("one lighting universe = Bevy's") shapes this shader:
// both branches below are lit by Bevy's own directional light + per-camera
// ambient + view exposure — no SceneLighting binding, no custom flux units.
// Only the *BRDF* changes per body style:
//
//   style 0 — stock PBR (`apply_pbr_lighting`) with the NTR-X4 material
//             layer stack, for vegetated / generic bodies on the tile path.
//   style 1 — Hapke regolith: the shared `thalos::lighting` Hapke lobe
//             (opposition surge + backscatter + Chandrasekhar H), driven by
//             Bevy's sun. This is what keeps airless ground from reading as
//             waxy plastic under the standard path, and lets tiles
//             reconverge with the impostor's Hapke look across the swap.
//
// Both branches receive the shared `thalos::shadow` cascade (bindings mirror
// `ShadowReceiveExtension` / `shadowed_standard.wgsl`) so craft / structure
// shadows land on tile ground exactly as they do on udlod ground.
//
// ── NTR-X4 material layers (style 0) ───────────────────────────────────────
//
// The vegetated branch re-details the canonical macro albedo with per-class
// material layers selected the way the Alpine reference reads
// (docs/reference/showcase_patch_prompt.md):
//
//   rock   — exposed on two independent grounds: *steep* ground at any
//            altitude (cliff bands in the forested flanks) and *alpine*
//            ground above the canonical treeline, where only benches keep
//            debris. Carries strata banding along a stable pseudo-bedding
//            frame, fall-line gully striation, and a Perlin detail normal.
//   scree  — the shoulder below rock faces plus the alpine benches: lighter,
//            granular debris blending into vegetation.
//   snow   — above the canonical snowline, sharpened by a noise-broken line
//            and shed from steep faces so rock ribs poke through; low
//            roughness.
//   forest — canonical CANOPY COVERAGE (vertex-carried): per-canopy cell
//            stippling (luminance + normal dimples) so aerial forest has
//            per-tree grain long before real scatter trees load. Coverage is
//            climate envelope × stand structure from ONE authority
//            (`thalos_terrain::canopy`), the same number the scatter places
//            trees from — so the grain appears where trees actually are. It
//            used to be the bare climate envelope, which painted canopy-green
//            over open plains while trees clustered elsewhere.
//   meadow — everything else: the canonical albedo with fine grass mottle.
//
// Selection inputs come through the standard mesh pipeline's spare vertex
// channels (see `tiles::build_tile_mesh`):
//   uv      = wrapped body-fixed position .xy   (metres, mod TILE_WRAP_M)
//   uv_b.x  = wrapped body-fixed position .z
//   uv_b.y  = canonical ECOLOGICAL altitude (m) — geometric height plus the
//             latitude cold lift; `thalos::landcover`'s `alpine_weight` /
//             `snowline_weight` cut the treeline and snowline from it at
//             exactly the altitudes the macro palette paints them
//   color   = canonical macro albedo (rgb) + canonical canopy coverage (a)
//
// The wrapped position is continuous within a tile (per-tile anchor snapped
// to the wrap period) and agrees across tiles mod TILE_WRAP_M, so every
// texture wavelength below must divide TILE_WRAP_M exactly — same discipline
// as udlod's 4 km-wrapped detail noise. Slope classification and the detail
// normal are built in the *body-fixed* frame (`tile_params.orient`), so they
// hold still under planet spin and floating-origin moves.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}
#import bevy_pbr::mesh_view_bindings::{view, lights}
#import thalos::shadow::{ShadowCascadeBlock, sun_shadow_factor_nrm}
#import thalos::cloud_shadow::{
    CloudShadowBlock, cloud_sun_transmittance, cloud_shadow_debug, cloud_shadow_payload,
}
#import thalos::lighting::{hapke_brdf_rgb, hapke_w_from_albedo}
#import thalos::landcover::{
    hash13, wrap_lattice, value_noise_3d_periodic, fbm3_periodic,
    alpine_weight, snowline_weight, macro_variation,
    substrate_rock_color, substrate_soil_color, substrate_wet_color,
    macro_forest_anchor, understory_forest_residual,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

struct TileShadingParams {
    // 1 = Hapke regolith (airless), 0 = stock PBR. u32 for tight mirroring
    // with the Rust `TileShadingParams`.
    style: u32,
    // Capture-only inspection mode (mirror of udlod's
    // `THALOS_TERRAIN_INSPECTION`): 0 = lit, 1 = fullbright, 2 = geometric
    // normal. See the Rust `TileShadingParams::inspect`.
    inspect: u32,
    _pad1: u32,
    _pad2: u32,
    // Body→world rotation (unit quaternion, xyzw), written per frame by the
    // game's tile driver (`update_tile_material_params`).
    orient: vec4<f32>,
    // Radial "up" at the view anchor, body-fixed (xyz; w unused). Up varies
    // ~1° per 50 km, so one uniform serves every fragment in frame; by the
    // time the error matters the ground is at the limb and sub-pixel.
    //
    // That reasoning holds for the *material layers* (slope classification is
    // a near-field concern) but NOT for the day/night gate below, which must
    // stay right across a whole globe — hence `center_ws`.
    up_body: vec4<f32>,
    // xyz = body centre in world render space — the per-fragment radial up is
    // `normalize(world_position - center_ws.xyz)`.
    // w = the day/night gate already folded into `lights.ambient_color` at the
    // craft, so the gate below can divide it out (see there).
    center_ws: vec4<f32>,
    // xyz = unit direction toward the star (world render space).
    // w = night-floor fraction of `lights.ambient_color` (see below).
    sun_night: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> recv_shadow: ShadowCascadeBlock;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var recv_shadow_map_0: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var recv_shadow_map_1: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var recv_shadow_map_2: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(104)
var<uniform> tile_params: TileShadingParams;
// Cloud sun-transmittance cascade (CLOUD-5 / W2) — the deck's own shadow,
// marched from the same density field the visible clouds are rendered from and
// fanned in by `tiles::apply_cloud_shadow`. `cloud_shadow.axis_v.w == 0` (no
// cloud body, clouds off, sun below the anchor's horizon) reads fully lit.
@group(#{MATERIAL_BIND_GROUP}) @binding(105)
var cloud_shadow_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106)
var cloud_shadow_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(107)
var<uniform> cloud_shadow: CloudShadowBlock;

// Shadows gate ONLY the direct term (split via a second occlusion-zeroed
// `apply_pbr_lighting` evaluation, exactly like `shadowed_standard.wgsl` /
// `ship_part.wgsl`). The old whole-colour multiply needed a 0.4 floor to keep
// ambient readable, which made every shadow a pale grey wash; now shadow kills
// the sun outright and the ground keeps its full sky/ambient fill.

// ── Per-fragment ambient day/night gate ────────────────────────────────────
//
// `GlobalAmbientLight` is ONE per-camera value, and the game derives it at the
// *craft* — sun elevation there picks the sky/space fill, and it falls to the
// night floor when the craft is on the night side (`rendering::lighting::
// update_sun_light`). That contract is exactly right for the near field, where
// every fragment shares the craft's horizon. It is wrong the moment the tile
// renderer draws a whole globe: from orbit over the day side the day-side fill
// is smeared across the night hemisphere too, and continents that should be
// black stay lit by flat ambient — no `n·l` to zero it, because ambient has no
// cosine.
//
// So the ambient is redistributed per fragment here. The CPU keeps sole
// authority over its *magnitude*; the shader only re-spreads it in space:
//
//   CPU gave us   floor + fill · daylight_craft
//   we want       floor + fill · daylight_fragment
//
// which is a pure multiply once you know `sun_night.w` = floor / ambient and
// `center_ws.w` = daylight_craft. Dividing the craft's gate back out is what
// makes this an *exact identity* in the near field, where every fragment
// shares the craft's horizon — otherwise a surface sunset would ramp through
// the terminator twice, once on the CPU and once here.
//
// The terminator mirrors `body_terrain.wgsl`'s `smoothstep(-0.06, 0.12,
// dot(up, sun_dir))` and its CPU twin `rendering::lighting::surface_daylight`
// at `altitude_ratio = 1`, so tile ground crosses day/night in lockstep with
// spine ground and with the Bevy-lit hull beside it.
//
// `up` is radial, NOT the surface normal: sky/space fill is a horizon
// question, not a facing one — a cliff turned away from a high sun still sees
// the sky, and gating on the normal would punch it black.
fn ambient_daylight_gate(world_pos: vec3<f32>) -> f32 {
    let radial = world_pos - tile_params.center_ws.xyz;
    let len = length(radial);
    if len <= 0.0 {
        return 1.0;
    }
    let sun_elev = dot(radial / len, tile_params.sun_night.xyz);
    let daylight = smoothstep(-0.06, 0.12, sun_elev);
    // Fragment's share of the fill, relative to the craft's. Clamped at 1 so a
    // fragment nearer the sub-solar point than the craft cannot *amplify* the
    // fill past what the CPU authorised.
    let fill = clamp(daylight / max(tile_params.center_ws.w, 1.0e-3), 0.0, 1.0);
    let night_frac = clamp(tile_params.sun_night.w, 0.0, 1.0);
    return night_frac + (1.0 - night_frac) * fill;
}

// Hapke → photometric bridge. `hapke_brdf_rgb` returns a radiance factor with
// the 1/(4π)-family normalisation deliberately left to the call site (see the
// library note); dividing by π puts a nadir-lit, w≈albedo surface in the same
// brightness family as Bevy's Lambert diffuse (albedo/π · n·l), so the sun's
// authored illuminance + camera exposure keep meaning what they mean. The
// remaining deviation from Lambert IS the regolith look (opposition surge,
// limb flattening), not a brightness error. Calibrate against the impostor by
// capture, not by re-tuning the light.
const HAPKE_PHOTOMETRIC_SCALE: f32 = 0.3183098862; // 1/π

// ── NTR-X4 layer constants ─────────────────────────────────────────────────

// Master wrap period of the vertex-carried body-fixed position (metres).
// Mirror of `tiles::TILE_WRAP_M`; every wavelength below divides it exactly.
const TILE_WRAP_M: f32 = 8192.0;

// Slope thresholds as cos(grade): 1 = flat. Rock above ~28.5°, scree on the
// ~22–28.5° shoulder — the reference's slope-driven exposure rule. Tuned LOW
// relative to real-world angles: the 90 m mesh understates true face slopes
// (a 4 km face over 12 km of 90 m verts averages well under 40°), so the
// thresholds must meet the geometry the mesh actually carries. Measured over
// the showcase massif's 36 km window: median 17.6°, p90 32.9°.
const ROCK_COS: f32 = 0.878;  // cos 28.5°
const SCREE_COS: f32 = 0.927; // cos 22°
// Noise jitter applied to the thresholds so transitions follow local texture,
// not contour lines.
const SLOPE_JITTER: f32 = 0.09;

// Above the treeline the slope rule alone leaves the alpine zone bare-looking
// but *untextured*: measured over the massif, the 3.4–4.6 km alpine band has
// a median slope of only 15°, so almost none of it passes ROCK_COS. Alpine
// ground is rock almost everywhere; debris collects only on the true flats.
const ALPINE_ROCK_COS: f32 = 0.990;  // cos 8° — steeper than this = rock
const ALPINE_BENCH_COS: f32 = 0.9976; // cos 4° — flatter than this = talus flat
/// How much of an alpine bench the talus claims (the rest stays parent rock).
/// Full coverage turned the whole upper massif into one pale debris sheet.
const ALPINE_BENCH_AMT: f32 = 0.45;

// Snow shedding, as cos(grade): snow lies on the summit fields and sloughs
// off the ribs between them. The measured slope distribution above the
// snowline is gentle (median 12°, p75 19°), so the window has to open there
// or nothing sheds and the summit is one white dome again.
const SNOW_HOLD_COS: f32 = 0.961; // cos 16° — full cover at/below this grade
const SNOW_SHED_COS: f32 = 0.829; // cos 34° — bare rock above it

// Layer palettes (linear RGB), anchored to the canonical macro band anchors
// in `procedural.rs::albedo_from_bands` so layers and palette agree.
// Darkened from the palette anchors (0.118 rock / 0.168 scree): with the
// alpine zone now claiming the whole upper massif rather than a few cliff
// bands, the anchors read as one pale sheet under the noon-ish sun. Real
// alpine rock sits *below* meadow in value; keeping it there is what gives
// the reference its dark-face / bright-snowfield separation.
const ROCK_ALBEDO: vec3<f32> = vec3<f32>(0.084, 0.079, 0.071);
const SCREE_ALBEDO: vec3<f32> = vec3<f32>(0.118, 0.112, 0.100);
const SNOW_ALBEDO: vec3<f32> = vec3<f32>(0.62, 0.64, 0.68);

// Strata bedding lattice vector: integer components so `dot(p, K)/TILE_WRAP_M`
// changes by an integer when `p` jumps by the wrap period along any axis —
// the band phase stays continuous across wrap boundaries. Bed thickness
// ≈ TILE_WRAP_M / |K| ≈ 128 m along the dip normal — thick enough to read
// from the aerial framings (finer beds vanish below the 90 m relief).
// Direction ≈ the local radial at the showcase window (lat 8.5, lon 178.4),
// so beds stack near-horizontally there like real sedimentary strata; the
// low-frequency warp supplies the dip/undulation variation.
const DIP_K: vec3<f32> = vec3<f32>(-63.0, 9.0, 2.0);

// Fall-line gully striation (see `material_layers`'s rock layer). Couloir
// scale: ~32 m across the slope, elongated over ~96 m down the fall line.
//
// **Never build this by projecting `p` onto the slope frame.**
// `noise(dot(p, across) / L, …)` looks like the obvious formulation and is the
// one that shipped; it is a moiré generator (INC-20260727T004856Z). `p` runs
// to TILE_WRAP_M from its anchor, so the pattern's phase is `|p| · slope
// angle`: tilting the surface by 32/8192 rad ≈ **0.2°** slides it a full
// stripe. Terrain turns far faster than that, so the striation stopped
// tracking the ground and started tracking the *normal field*, drawing
// contour-following whorls — an agate / topographic-map look over every rock
// face, worst up close where nothing fades it out.
//
// The directional low-pass below has the same anisotropy with bounded
// sensitivity: every tap is a body-space sample of an isotropic field, offset
// from `p` by at most half the span, so a tilt change displaces a tap by
// `span/2 · angle` — ~170× less than the projection moved it. The slope
// direction now *orients* the filter instead of setting the phase.
const GULLY_ACROSS_M: f32 = 32.0;   // divides TILE_WRAP_M exactly (256 periods)
const GULLY_TAPS: i32 = 5;
// Tap spacing must stay under the feature size, or the taps decorrelate and
// the sum averages the field flat instead of streaking it.
const GULLY_STEP_M: f32 = 24.0;
// Restores the contrast the 5-tap average removes, so the layer keeps the
// amplitude the 0.22 albedo / 0.9 normal weights below were tuned against
// (partly-correlated taps: σ drops ~0.53×, against the old 2-octave fBm's
// 0.745× — hence ~1.4).
const GULLY_GAIN: f32 = 1.4;

// ── Perlin (gradient) noise with analytic derivative ───────────────────────
// Copied from udlod's `body_terrain.wgsl` prior art (EOL stack — read, not
// imported): value-noise derivatives are strongly axis-aligned and show the
// cubic lattice as a "weave" in any normal built from them; gradient noise
// randomises per-corner *vectors*, so its derivative is isotropic — the right
// basis for a detail normal (see the wgsl-bevy skill note).

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

// fBm value + analytic gradient over gradient (Perlin) noise; frequency
// doubles, amplitude halves, each octave's gradient chain-rule-scaled by its
// frequency.
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

// Rotate `v` by unit quaternion `q` (xyzw).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

// Footprint fade: 1 below `on_m` (fully resolved), 0 above `off_m` — retires
// detail wavelengths the fragment's ground footprint can't resolve so distant
// ground dissolves instead of sparkling (udlod's faded-fBm discipline).
fn footprint_fade(footprint_m: f32, on_m: f32, off_m: f32) -> f32 {
    return 1.0 - smoothstep(on_m, off_m, footprint_m);
}

// Near end of the stand-in band, in **on-screen pixels per cycle**. A stand-in
// is at full strength while its wavelength projects to `STANDIN_ON_PX` or
// fewer, and is fully retired by `STANDIN_OFF_PX`.
//
// The near cut is stated in screen wavelength, not in metres per pixel, and
// that is load-bearing (INC-20260726T040430Z). It used to be a single global
// `1 m/px → 4 m/px` ramp applied to every stand-in regardless of the scale it
// stood in for. But "the real feature has come into reach" is a property of
// *that term's own wavelength* — an 8 m clutter bump and a 64 m rock grain
// stop being texture at completely different camera distances. One global
// cut in metres per pixel cannot express both, and the value it was tuned to
// closed every stand-in below 4 m/px at once. Since the GPU grass layer hides
// above 550 m AGL and 4 m/px is not reached until ~11 km of slant range, that
// left open ground with NO detail source at all through the whole low-flight
// band — measured at 1.09/255 luminance σ from 800 m AGL, i.e. flat paint.
//
// Expressed per-wavelength the same rule covers every term correctly: below
// ~64 px/cycle a stand-in reads as texture, above ~192 px/cycle it has grown
// into terrain and must dissolve. `on_m` (the far-fade knee) is a term's
// wavelength/2 by construction — the far fade starts when the footprint can
// no longer resolve it — so the wavelength is recovered as `2 · on_m` and no
// caller has to restate it.
//
// This leaves the accepted showcase calibration untouched: at the framings
// those were tuned from (≥2 m/px, `artifacts/visual/runs/ntr-x4-nearfield/`)
// every stand-in projects to ≤15 px/cycle, so the near term is exactly 1.
const STANDIN_ON_PX: f32 = 64.0;
const STANDIN_OFF_PX: f32 = 192.0;

// Band-pass footprint window for detail that *stands in for geometry we do not
// have yet* — the canopy stipple (real scatter trees), the meadow mottle and
// the rock/scree grain (real micro-relief).
//
// `footprint_fade` alone is a low-pass: it retires a wavelength once the
// footprint grows past it, but returns 1 all the way down, so a stand-in is at
// FULL strength exactly where the thing it stands in for would be resolved.
// That is what made the low-altitude near field read as deep-fried: an 18 m
// canopy dimple is legitimate grain at 2 px/cell from 5 km up, and house-sized
// rolling bumps at 90 px/cell from 150 m out. The near roll-off is the missing
// half — the stand-in dissolves as the real feature scale comes into reach,
// instead of growing to fill the screen.
//
// Structure that is real at every scale (rock strata, gully striation) keeps
// the plain far-only `footprint_fade`; only stand-ins get the band.
fn footprint_band(footprint_m: f32, on_m: f32, off_m: f32) -> f32 {
    // Screen wavelength of this term, in pixels per cycle. Guarded: a
    // degenerate footprint must retire the stand-in, not produce a NaN.
    let px_per_cycle = 2.0 * on_m / max(footprint_m, 1.0e-4);
    let near = 1.0 - smoothstep(STANDIN_ON_PX, STANDIN_OFF_PX, px_per_cycle);
    return near * footprint_fade(footprint_m, on_m, off_m);
}

// Dielectric specular reflectance (Bevy's `StandardMaterial::reflectance`
// convention: F0 = 0.16 · r²; the stock 0.5 gives F0 = 0.04, a generic
// dielectric). Vegetation is not a generic dielectric — grass and canopy
// scatter far more than they mirror, and at a low sun the stock F0 turns every
// perturbed normal into glitter. Rock and snow keep the stock value.
const VEG_REFLECTANCE: f32 = 0.32;   // F0 ≈ 0.016
const ROCK_REFLECTANCE: f32 = 0.5;   // F0 = 0.04
const SNOW_REFLECTANCE: f32 = 0.5;

// ── Landcover parity (NTR-X7 P1b) ──────────────────────────────────────────
//
// The tile path painted the canonical macro albedo directly and layered
// rock/scree/snow on top. udlod does something materially different, and
// better: it builds ground colour out of the **shared `thalos::landcover`
// model** — the same `vegetation_color` / `macro_variation` the grass blades'
// CPU mirror (`ground/landcover.rs`) reads — and keeps only ~10% of the macro
// albedo as broad body identity.
//
// That fork is not just a look difference, it is a broken invariant: on the
// tile path the ground and the grass growing out of it were computed from
// different definitions, so they could not agree by construction. The
// shared-library rule exists to stop exactly this.
//
// So the hand-rolled macro term that used to live here is gone in favour of
// the library's `macro_variation` — the ~250 m tier, in [-1, 1], with regional
// drift deliberately left to the baked albedo (which this path already
// carries). Same function, same wavelengths, same amount as udlod applies.
//
// `MACRO_VAR_AMT` mirrors `body_terrain.wgsl`'s constant of the same name.
const MACRO_VAR_AMT: f32 = 0.16;

// Understory-recovery footprint window: fully recovered while a ground texel
// is `UNDERSTORY_ON_M` or finer (individual scatter trees clearly resolve, so
// the ground between them must read as ground), fully back to the canonical
// canopy-dark bake by `UNDERSTORY_OFF_M` (stands have collapsed into paint,
// and the dark mix IS the forest again).
const UNDERSTORY_ON_M: f32 = 6.0;
const UNDERSTORY_OFF_M: f32 = 40.0;

// Slope-driven substrate exposure, mirroring udlod's `grade_surface`: soil
// shows on moderate ground, rock takes over as it steepens. This is what gives
// open ground its earthy break-up instead of one flat green — and it is
// derived from the same shared palette constants, not from new colours.
const SLOPE_SOIL_LO: f32 = 0.10;
const SLOPE_SOIL_HI: f32 = 0.34;
const SOIL_STRENGTH: f32 = 0.34;
const SLOPE_ROCK_LO: f32 = 0.30;
const SLOPE_ROCK_HI: f32 = 0.62;
const ROCK_STRENGTH: f32 = 0.70;
/// Damp hollows darken toward `C_WET`. Keyed off the macro variation's low
/// tail, which is the only moisture proxy this path has until the provider
/// carries `landcover_moisture` per vertex (the remaining parity gap).
const WET_STRENGTH: f32 = 0.22;

// ── Meadow / open-ground relief (NTR-X7 P1) ────────────────────────────────
//
// Open vegetated ground contributed ZERO normal offset — only rock and scree
// perturbed the normal — so kilometres of plain shared one shading normal.
// That is what produced the satin sheen: with a single coherent N, GGX's
// Fresnel term `F0 + (1-F0)(1-V·H)^5` climbs toward 1 at grazing no matter how
// low F0 is, and the whole plain lights as one mirror. Real ground never does
// this because its micro-normals are randomised; the fix is to randomise ours.
//
// **Amplitude and wavelength are not interchangeable here**, and getting that
// wrong is how this term goes bad. A first attempt added a 32 m "roll" at 0.13
// — and a 32 m normal wave IS topography at these framings, so the whole plain
// came out corrugated. This file already carried the warning, from the canopy
// stipple that was deleted for the same reason: a stand-in normal whose
// wavelength sits inside the footprints we actually ship reads as crumpled
// foil. The roll is gone.
//
// What survives is deliberately *sub-topographic*: one short wavelength at low
// amplitude, banded so it is absent from aerial framings entirely. It exists
// only to break the specular coherence at contact range. The far field's share
// of the fix is NOT a normal at all — it is `MEADOW_MICRO_VARIANCE` below,
// which is invisible geometry and pure roughness.
const MEADOW_CLUTTER_M: f32 = 8.0;
const MEADOW_CLUTTER_AMP: f32 = 0.045;

// Contact-scale ground grain — tussock, bare soil, litter and track scars.
//
// With the near cut stated per-wavelength, the stand-in ladder's *finest* rung
// became the limiting one: below the 16 m meadow mottle this shader modelled
// nothing at all, so from a few hundred metres up the whole plain still came
// down to one macro albedo. This is the missing rung, at the scale open ground
// actually varies on.
//
// Albedo only, deliberately. The file's two hard-won lessons about stand-in
// normals (the 32 m "roll" that corrugated the plain, the canopy dimple that
// read as crumpled foil) both say the same thing: a stand-in normal whose
// wavelength lands inside a shipped footprint is how this term goes bad. The
// normal's share of the near field is already carried by `MEADOW_CLUTTER_M`,
// which is tuned for exactly this range and which the per-wavelength cut hands
// back. So this rung modulates tone and nothing else.
// Power-of-two wavelength so `TILE_WRAP_M / MEADOW_GRAIN_M` is an exact
// lattice period and the grain wraps seamlessly across the position wrap.
const MEADOW_GRAIN_M: f32 = 4.0;
const MEADOW_GRAIN_AMP: f32 = 0.13;
/// How much soil the grain's positive tail exposes through thin cover. Applied
/// in the substrate stage, alongside the slope-driven term it complements.
const GRAIN_SOIL_STRENGTH: f32 = 0.40;
/// How much of the grain's negative tail counts as damp, feeding the same wet
/// term the macro field's low tail drives.
const GRAIN_WET_TAIL: f32 = 0.55;

// Intrinsic sub-texel roughness of open ground, as normal variance.
//
// Grass, soil and litter are violently rough below the scale any mesh or
// detail normal will ever reach: at one metre per pixel a texel of meadow
// contains thousands of blades pointing everywhere. That variance is REAL and
// it is always there — it is not something a footprint fade threw away — so it
// is stated as a constant rather than derived from a fade, and it is what
// stops open ground from lighting as one coherent mirror at grazing sun.
//
// This is the honest form of the fix: the sheen is a *roughness* error, so it
// is corrected in roughness. Correcting it with visible bumps was the mistake.
const MEADOW_MICRO_VARIANCE: f32 = 0.055;

// ── Specular antialiasing (NTR-X7 P1) ──────────────────────────────────────
//
// Kaplanyan/Toksvig: a detail normal that fades out with distance did not stop
// existing, it stopped being *resolvable*. Its variance has to reappear as
// roughness, or the far field recovers a mirror-smooth normal and with it
// exactly the grazing sheen the detail normals were added to break — the
// distant half of the same defect, and the reason "just add a normal map"
// alone never fixes shiny terrain.
//
// `alpha'^2 = alpha^2 + 2σ^2` on the GGX alpha, where σ^2 is the mean-square
// tangential normal deviation we retired. Every layer reports the amplitude it
// dropped; `LayerResult::normal_variance` accumulates it.
fn specular_aa_roughness(perceptual: f32, variance: f32) -> f32 {
    let alpha = perceptual * perceptual;
    let widened = clamp(alpha * alpha + 2.0 * variance, 0.0, 1.0);
    // sqrt twice: alpha^2 -> alpha -> perceptual. Never *lowers* roughness.
    return clamp(sqrt(sqrt(widened)), perceptual, 1.0);
}

// The NTR-X4 vegetated material stack's output.
struct LayerResult {
    albedo: vec3<f32>,
    roughness: f32,
    // Dielectric specular reflectance, per material class.
    reflectance: f32,
    // Detail-normal offset in the body frame (added to n_body, renormalised).
    normal_offset: vec3<f32>,
    // Mean-square tangential normal deviation this fragment's footprint
    // CANNOT resolve — the perturbation the footprint fades above threw away.
    // Fed to `specular_aa_roughness` so it comes back as roughness.
    normal_variance: f32,
}

fn material_layers(
    p: vec3<f32>,          // wrapped body-fixed position, metres
    n_body: vec3<f32>,     // geometric normal, body frame
    base_albedo: vec3<f32>,
    eco_altitude_m: f32,   // canonical climate-shifted altitude
    canopy_w: f32,
    footprint_m: f32,
) -> LayerResult {
    let up = tile_params.up_body.xyz;
    let cosg = clamp(dot(n_body, up), -1.0, 1.0);

    // ── Selection fields ──────────────────────────────────────────────────
    let jitter = (fbm3_periodic(p / 256.0, 3, TILE_WRAP_M / 256.0) - 0.5) * 2.0;
    let cosg_j = cosg + jitter * SLOPE_JITTER;

    // Canonical altitude lines, jittered by a low-frequency field so they
    // follow local terrain instead of drawing contour rings. Both come from
    // `thalos::landcover`, i.e. the exact altitudes the macro palette cuts.
    let line_jitter = (fbm3_periodic(p / 384.0, 3, TILE_WRAP_M / 384.0) - 0.5) * 2.0;
    let alt_j = eco_altitude_m + line_jitter * 260.0;
    let alpine = alpine_weight(alt_j);
    let snow_band = snowline_weight(alt_j);

    // Rock on two independent grounds: steep ground anywhere (cliff bands in
    // the forested flanks) and alpine ground that is not a bench. The alpine
    // term is what keeps the 3.4–4.6 km zone reading as rock — its median
    // slope is far below the cliff threshold.
    let rock_steep = 1.0 - smoothstep(ROCK_COS - 0.04, ROCK_COS + 0.04, cosg_j);
    let rock_alpine = alpine * (1.0 - smoothstep(ALPINE_ROCK_COS - 0.012, ALPINE_ROCK_COS + 0.012, cosg_j));
    let rock_m = max(rock_steep, rock_alpine);

    // Scree: the slope shoulder below rock faces, plus the alpine benches the
    // rock term deliberately left out (talus collects on the flats). Gated to
    // the alpine zone — ungated, the 22° shoulder swallowed every forested
    // flank (the massif's sub-treeline slopes have a 24.7° median) and laid a
    // pale debris wash over the whole mountain. Below the treeline soil holds
    // to the point where rock takes over, so there is no debris band there;
    // real talus fans *under cliffs* need a curvature term (NTR-X4 P3).
    let scree_slope = (1.0 - smoothstep(SCREE_COS - 0.05, SCREE_COS + 0.05, cosg_j)) * alpine;
    let scree_bench = alpine
        * smoothstep(ALPINE_BENCH_COS - 0.004, ALPINE_BENCH_COS + 0.002, cosg_j)
        * ALPINE_BENCH_AMT;
    let scree_m = max(scree_slope, scree_bench) * (1.0 - rock_m);

    // Snow: above the canonical snowline, sharpened by a noise-broken line
    // and shed from steep ground so rock ribs poke through (reference pt. 5).
    let snow_break = (fbm3_periodic(p / 128.0, 3, TILE_WRAP_M / 128.0) - 0.5) * 2.0;
    let shed = smoothstep(SNOW_SHED_COS, SNOW_HOLD_COS, cosg + 0.04 * jitter);
    let snow_m = smoothstep(0.30, 0.60, snow_band + 0.28 * snow_break) * shed;

    // Forest: canonical canopy coverage, gated off rock/snow and steep ground —
    // and off the alpine zone, which is above the treeline by definition.
    let forest_gate = smoothstep(0.55, 0.75, cosg) * (1.0 - rock_m) * (1.0 - snow_m) * (1.0 - alpine);
    let forest_density = smoothstep(0.15, 0.55, canopy_w + 0.20 * jitter) * forest_gate;

    // ── Layers ────────────────────────────────────────────────────────────
    var albedo = base_albedo;
    var roughness = 0.95;
    var reflectance = VEG_REFLECTANCE;
    var n_off = vec3<f32>(0.0);
    var variance = 0.0;

    // ── Understory recovery (near field) ──────────────────────────────────
    //
    // The baked macro albedo paints forested ground as *closed canopy seen
    // from above* — the `albedo_from_bands` forest anchor, mixed in at the
    // absolute canopy coverage (`canopy_w`, this vertex's alpha). That is
    // the right colour for the far field, where no individual trees render
    // and the dark green IS the forest. Near-field it is a category error:
    // real scatter trees stand on this ground, so the fragment the camera
    // actually sees between and under them is understory — grass, litter,
    // soil — not a canopy viewed from orbit. Painting it canopy-dark is what
    // made bright canopies sit on teal felt (the tree/ground disconnect).
    //
    // The nested band mix is convex and `canopy_w` is the anchor's absolute
    // weight — the *same* product the bake used, which is what keeps this
    // inversion exact now that coverage (not the climate envelope) drives the
    // mix. So the pre-forest ground is recoverable right here:
    //   baked = (1 - w) · rest + w · anchor   ⇒   rest = (baked - w·anchor)/(1 - w)
    // (exact up to the ~30 km tone mottle, which multiplies both terms and
    // survives the subtraction as a small residual). A residual share of the
    // canopy mix is kept (`understory_forest_residual`) — under real trees the
    // ground does hold litter and shade — and the whole recovery rides
    // `footprint_fade`, dissolving back to the canonical canopy colour as the
    // footprint grows past the scale where scatter trees resolve.
    let understory_fade = footprint_fade(footprint_m, UNDERSTORY_ON_M, UNDERSTORY_OFF_M);
    if understory_fade > 0.0 && canopy_w > 0.01 {
        // Cap the inversion: at w → 1 the denominator loses all information
        // (the bake kept none of the pre-forest colour), so stop brightening
        // past the cap instead of amplifying noise.
        let w = min(canopy_w, 0.88);
        let rest = max(
            (base_albedo - macro_forest_anchor() * w) / (1.0 - w),
            vec3<f32>(0.0),
        );
        let understory = mix(rest, macro_forest_anchor(), w * understory_forest_residual());
        albedo = mix(albedo, understory, understory_fade);
    }

    // Open ground: a whisper of contact-scale clutter, and the micro-roughness
    // that does the actual work. Weighted off rock/snow so it does not fight
    // their own detail, and applied BEFORE them so those still win where they
    // claim the fragment.
    let open_w = (1.0 - rock_m) * (1.0 - snow_m);
    if open_w > 0.0 {
        // `footprint_band`, not `footprint_fade`: this is a stand-in for grass
        // and soil texture we do not model, so it must dissolve as the real
        // thing comes into reach AND be absent when the footprint is metres.
        let clutter_fade = footprint_band(footprint_m, 4.0, 16.0);
        if clutter_fade > 0.0 {
            let dg = fbm3_perlin_grad(p / MEADOW_CLUTTER_M, 2, TILE_WRAP_M / MEADOW_CLUTTER_M);
            n_off += dg.yzw * open_w * clutter_fade * MEADOW_CLUTTER_AMP;
        }
        // Always present, at every distance — see `MEADOW_MICRO_VARIANCE`.
        variance += MEADOW_MICRO_VARIANCE * open_w;
    }

    // Contact-scale ground grain — the ladder's finest rung, and the one that
    // carries open ground through the low-flight band the 16 m mottle is too
    // coarse for. Zero-mean over the canonical colour, so it varies tone
    // without introducing a palette of its own (shared-library rule).
    // Kept for the substrate stage below, which breaks thin cover to soil on
    // the grain's positive tail.
    var contact_grain = 0.0;
    let grain_fade_open = footprint_band(footprint_m, 2.0, 8.0);
    if open_w > 0.0 && grain_fade_open > 0.0 {
        let g = fbm3_perlin_grad(p / MEADOW_GRAIN_M, 2, TILE_WRAP_M / MEADOW_GRAIN_M);
        contact_grain = g.x * open_w * grain_fade_open;
        albedo = albedo * (1.0 + MEADOW_GRAIN_AMP * contact_grain);
    }

    // Meadow mottle: fine grass-tone variation on open vegetated ground. A
    // stand-in for real grass/ground-cover variety, so it rides the band —
    // at a metre per pixel the 16 m mottle is a paint blotch, not grass.
    let meadow_fade = footprint_band(footprint_m, 8.0, 32.0);
    if meadow_fade > 0.0 {
        let mottle = fbm3_periodic(p / 16.0, 3, TILE_WRAP_M / 16.0) - 0.5;
        albedo = albedo * (1.0 + meadow_fade * 0.20 * mottle);
    }

    // Forest canopy grain: per-canopy tone variation, and nothing else.
    //
    // This layer used to *also* carry a forest colour — `base_albedo ×
    // (0.52, 0.62, 0.50)` — which double-counted the canopy: the canonical
    // macro palette has already mixed the very same weight toward its own
    // closed-canopy anchor (`procedural.rs::albedo_from_bands`, the `forest`
    // Vec3(0.040, 0.095, 0.032) step) before the vertex colour ever reaches
    // this shader. On the equatorial wet belt, where the canonical forest
    // weight measures ~0.95 (probe: `relief_spectrum`), that second darkening
    // multiplied a ground already painted dark canopy green by ~0.55 again.
    // Palette lives in one place — the shared-library rule — so the tint is
    // gone and only the grain the palette *can't* carry stays here.
    //
    // The grain is a stand-in for scatter trees, so it is zero-mean (it varies
    // the canonical colour rather than shifting it) and rides `footprint_band`,
    // dissolving as the real trees come into reach. It no longer dimples the
    // NORMAL at all: the tile path now draws actual scatter trees (NTR-X2b), so
    // a fake per-tree bump under the real ones is both double detail and the
    // thing that made open ground read as crumpled foil at grazing framings —
    // where the footprint sits in the stipple's own 12–48 m window and nothing
    // retired it.
    let canopy_stipple = footprint_band(footprint_m, 12.0, 48.0);
    if forest_density > 0.0 {
        let cell = value_noise_3d_periodic(p / 24.0, TILE_WRAP_M / 24.0);
        let cell2 = value_noise_3d_periodic(p / 12.0, TILE_WRAP_M / 12.0);
        let grain = (0.6 * cell + 0.4 * cell2) - 0.5;
        albedo = albedo * (1.0 + forest_density * canopy_stipple * 0.55 * grain);
        roughness = mix(roughness, 1.0, forest_density);
    }

    // Rock: strata banding + fall-line gullies + detail normal (ref pts 2, 6).
    if rock_m > 0.0 {
        // Bedding coordinate: integer lattice vector keeps the band phase
        // continuous across the position wrap; low-frequency warp makes the
        // beds undulate like folded strata.
        let warp = fbm3_periodic(p / 512.0, 2, TILE_WRAP_M / 512.0) - 0.5;
        let band_coord = dot(p, DIP_K) / TILE_WRAP_M + warp * 1.6;
        let band = abs(fract(band_coord) * 2.0 - 1.0); // triangle wave
        let strata = smoothstep(0.25, 0.75, band) - 0.5;

        // Fall-line gully striation: an isotropic body-space field, low-passed
        // ALONG the fall line so couloir-scale features elongate downhill.
        // Read GULLY_ACROSS_M's note before touching the coordinates — the
        // slope frame may orient this filter, never set its phase.
        let fall = normalize(up - n_body * cosg + vec3<f32>(1.0e-5, 0.0, 0.0));
        let across = normalize(cross(n_body, fall));
        let rock_fade = footprint_fade(footprint_m, 30.0, 140.0);
        let detail_fade = footprint_fade(footprint_m, 15.0, 90.0);
        var gully = 0.0;
        // Both consumers are weighted by `rock_steep` and one of the two
        // fades, so distant / gentle rock skips the taps entirely.
        if rock_steep * max(rock_fade, detail_fade) > 0.0 {
            var gully_sum = 0.0;
            for (var t = 0; t < GULLY_TAPS; t = t + 1) {
                // Symmetric taps about `p`, walking down the fall line. Offsets
                // are continuous across the position wrap (the normal is), so
                // the periodic lattice still tiles seamlessly tile to tile.
                let s = (f32(t) - 0.5 * f32(GULLY_TAPS - 1)) * GULLY_STEP_M;
                gully_sum += value_noise_3d_periodic(
                    (p + fall * s) / GULLY_ACROSS_M,
                    TILE_WRAP_M / GULLY_ACROSS_M,
                );
            }
            gully = (gully_sum / f32(GULLY_TAPS) - 0.5) * GULLY_GAIN;
        }

        var rock_albedo = mix(ROCK_ALBEDO, base_albedo, 0.22);
        let macro_tone = fbm3_periodic(p / 1024.0, 3, TILE_WRAP_M / 1024.0) - 0.5;
        rock_albedo = rock_albedo * (1.0 + 0.30 * macro_tone);
        // Structure leads, grain follows: the banding + striation carry most
        // of the albedo modulation so faces read as bedded rock, not static.
        // Bedding is a FACE feature — weighted by `rock_steep`, not by the
        // alpine term. Applied across the flat alpine zone as well, the
        // near-radial bedding frame draws its bands along the contours and
        // the upper massif terraces like a topographic map (round-3 finding).
        rock_albedo = rock_albedo
            * (1.0 + rock_fade * rock_steep * (0.26 * strata + 0.22 * gully));

        albedo = mix(albedo, rock_albedo, rock_m);
        roughness = mix(roughness, 0.88, rock_m);
        reflectance = mix(reflectance, ROCK_REFLECTANCE, rock_m);

        // Detail normal: strata ledges along the bedding axis + gully
        // striation across the fall line, with only a whisper of isotropic
        // grain — grain-dominant perturbation reads as speckle from the
        // showcase distances (round-2 finding).
        //
        // Ledges and striation are real lithological structure at any scale,
        // so they keep the far-only fade; the isotropic grain is a stand-in
        // for micro-relief and rides the band, or a cliff walked up to turns
        // into the same 64 m blobs the canopy used to paint on the meadow.
        let grain_fade = footprint_band(footprint_m, 15.0, 90.0);
        if detail_fade > 0.0 {
            let dg = fbm3_perlin_grad(p / 64.0, 2, TILE_WRAP_M / 64.0);
            let ledge = smoothstep(0.25, 0.75, band) - smoothstep(0.75, 0.98, band);
            // Isotropic grain everywhere rock shows; ledges + striation only
            // on faces, for the same reason the bedding albedo is.
            n_off += dg.yzw * rock_m * grain_fade * 0.20
                + (normalize(DIP_K) * ledge * 0.7 + across * gully * 0.9)
                    * rock_steep * detail_fade * 0.35;
        }
        // Rock faces are the roughest thing on the body at micro scale, and
        // the distances they are viewed from retire nearly all of it — the
        // grain band in particular is fully off beyond ~90 m/px. Without the
        // variance handed back, a distant cliff is a polished facet.
        let lost_grain = 0.20 * rock_m * (1.0 - grain_fade);
        let lost_struct = 0.35 * rock_steep * (1.0 - detail_fade);
        variance += lost_grain * lost_grain + lost_struct * lost_struct;
    }

    // Scree: lighter granular debris apron between rock and vegetation. All of
    // its texture is grain — a stand-in for individual blocks — so the whole
    // layer's detail rides the band; only the debris *colour* survives close up.
    if scree_m > 0.0 {
        let scree_fade = footprint_band(footprint_m, 10.0, 50.0);
        let grain = fbm3_periodic(p / 8.0, 2, TILE_WRAP_M / 8.0) - 0.5;
        var scree_albedo = mix(SCREE_ALBEDO, base_albedo, 0.30);
        scree_albedo = scree_albedo * (1.0 + scree_fade * 0.35 * grain);
        albedo = mix(albedo, scree_albedo, scree_m * 0.85);
        roughness = mix(roughness, 0.97, scree_m);
        reflectance = mix(reflectance, ROCK_REFLECTANCE, scree_m);
        if scree_fade > 0.0 {
            let dg = fbm3_perlin_grad(p / 16.0, 2, TILE_WRAP_M / 16.0);
            n_off += dg.yzw * scree_m * scree_fade * 0.15;
        }
        let lost_scree = 0.15 * scree_m * (1.0 - scree_fade);
        variance += lost_scree * lost_scree;
    }

    // Snow last: covers whatever it claims; slight drift sub-structure so
    // fields aren't flat white, and it smothers the detail normal beneath.
    if snow_m > 0.0 {
        let drift = fbm3_periodic(p / 128.0, 2, TILE_WRAP_M / 128.0) - 0.5;
        let snow_albedo = SNOW_ALBEDO * (1.0 + 0.06 * drift);
        albedo = mix(albedo, snow_albedo, snow_m);
        roughness = mix(roughness, 0.38, snow_m);
        reflectance = mix(reflectance, SNOW_REFLECTANCE, snow_m);
        n_off = n_off * (1.0 - 0.8 * snow_m);
        // Snow smothers the relief beneath it, so it smothers that relief's
        // unresolved variance too — a snowfield really is the one smooth
        // surface out here, and handing it borrowed roughness would flatten
        // the low-roughness sheen that makes it read as snow.
        variance = variance * (1.0 - 0.8 * snow_m);
    }

    // ── Substrate + macro mottle, mirroring udlod's `eval_material_stack` ──
    //
    // Applied to the settled layer stack rather than to the meadow alone, so
    // every class inherits the same regional drift instead of each being
    // internally uniform. Snow is exempted from both: it is near-saturating,
    // and mottling it reads as dirt rather than as snow.
    let not_snow = 1.0 - snow_m;
    let variation = macro_variation(p);

    // Soil and rock show through vegetated ground as it steepens — udlod's
    // `grade_surface`, on the same shared palette constants. `slope_t` is
    // steepness in [0, 1] (0 flat, 1 vertical), the same parameterisation.
    let slope_t = clamp(1.0 - cosg, 0.0, 1.0);
    let open_share = not_snow * (1.0 - rock_m);
    if open_share > 0.0 {
        var ground = albedo;
        // Slope is the only thing that used to expose soil, so a plain — where
        // `slope_t` is ~0 by definition — came out as one hue no matter how
        // much tonal grain sat on it. Real open ground breaks cover on its own
        // small relief: hummock crests, stock tracks, thin patches. The contact
        // grain is that relief, so its positive tail exposes soil, on the same
        // shared palette the slope term uses. Rides the grain's own footprint
        // band, so the far field's hue is untouched.
        //
        // Thresholded rather than washed on: a linear mix tints the whole
        // plain slightly brown, which is a colour error, while a threshold
        // gives *patches* of exposed ground between intact cover — the thing
        // that actually reads.
        //
        // The threshold rides the grain and nothing else. An earlier revision
        // also gated it on the macro field's dry tail, reasoning that bare
        // ground belongs on dry rises — but `macro_variation` is a 250 m
        // mottle, so across the near field it is very nearly a constant, and
        // the gate simply switched the whole effect off (measured: hue σ fell
        // back to the ungated baseline). Dry-rise-versus-damp-hollow is a
        // *local* relation, and at this scale the grain is the only field that
        // carries it: its positive tail is the crest, its negative tail — read
        // by the wet term below — is the hollow.
        let bare = smoothstep(0.0, 0.38, contact_grain);
        let soil_t = clamp(
            smoothstep(SLOPE_SOIL_LO, SLOPE_SOIL_HI, slope_t) * SOIL_STRENGTH
                + bare * GRAIN_SOIL_STRENGTH,
            0.0,
            1.0,
        );
        ground = mix(ground, substrate_soil_color(), soil_t);
        let rock_t = smoothstep(SLOPE_ROCK_LO, SLOPE_ROCK_HI, slope_t) * ROCK_STRENGTH;
        ground = mix(ground, substrate_rock_color(alpine), rock_t);
        // Damp hollows: the macro field's low tail stands in for the moisture
        // channel the provider does not carry yet. The contact grain's own
        // negative tail joins it, so the small hollows between hummocks hold
        // damp the same way the regional ones do — the other half of the hue
        // variety `bare` gives the crests, and the reason open ground reads as
        // ground rather than as one green with noise on it.
        let wet = clamp(-variation - GRAIN_WET_TAIL * contact_grain, 0.0, 1.0)
            * WET_STRENGTH;
        ground = mix(ground, substrate_wet_color(), wet * (1.0 - rock_t));
        albedo = mix(albedo, ground, open_share);
    }

    albedo = albedo * (1.0 + variation * MACRO_VAR_AMT * not_snow);

    var out: LayerResult;
    out.albedo = albedo;
    out.roughness = roughness;
    out.reflectance = reflectance;
    out.normal_offset = n_off;
    out.normal_variance = variance;
    return out;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // Geometric world normal drives the stable-CSM receiver offset (a
    // perturbed N would wobble it) — same rule as every other receiver.
    let n = normalize(pbr_input.world_normal);
    let hard_shadow = sun_shadow_factor_nrm(
        pbr_input.world_position.xyz,
        n,
        recv_shadow,
        recv_shadow_map_0,
        recv_shadow_map_1,
        recv_shadow_map_2,
    );
    // The cloud deck is just the largest occluder of the same beam, so it folds
    // into the same gate as craft/structure shadows. With the direct/indirect
    // split below, the physical `exp(-τ)` transmittance now safely gates the
    // DIRECT term only — a thick cell kills the sunbeam but never the sky fill.
    // (The remaining §3.5 half — darkening the *ambient* under heavy overcast
    // by the overhead optical depth — still needs the planet-scale tail this
    // cascade does not have yet.)
    let cloud_t = cloud_sun_transmittance(
        cloud_shadow,
        cloud_shadow_tex,
        cloud_shadow_samp,
        pbr_input.world_position.xyz,
    );
    let shadow_f = hard_shadow * cloud_t;
    // Diagnostic paint (`THALOS_CLOUD_SHADOW=show`): the transmittance the
    // cascade actually holds at this fragment, unlit. Separates a wrong march
    // from a wrong projection — the split that matters most for this term,
    // since the producer and this shader reach the same lookup frame by
    // different routes.
    if cloud_shadow_debug(cloud_shadow) {
        out.color = vec4<f32>(
            cloud_shadow_payload(
                cloud_shadow,
                cloud_shadow_tex,
                cloud_shadow_samp,
                pbr_input.world_position.xyz,
            ),
            1.0,
        );
        return out;
    }
    let ambient_gate = ambient_daylight_gate(pbr_input.world_position.xyz);

    if tile_params.style == 1u && lights.n_directional_lights > 0u {
        let albedo = pbr_input.material.base_color.rgb;
        let v = pbr_input.V;
        let n_dot_v = max(dot(n, v), 1.0e-3);
        let w = hapke_w_from_albedo(albedo);
        // Sum ALL directional lights (the scene carries sun + moonlight; the
        // sun's slot index is not guaranteed).
        var direct = vec3<f32>(0.0);
        let n_lights = min(lights.n_directional_lights, 4u);
        for (var i = 0u; i < n_lights; i = i + 1u) {
            let light = lights.directional_lights[i];
            let l = light.direction_to_light;
            let r = hapke_brdf_rgb(
                max(dot(n, l), 0.0),
                n_dot_v,
                dot(v, l),
                pbr_input.material.perceptual_roughness,
                w,
            );
            direct += r * HAPKE_PHOTOMETRIC_SCALE * light.color.rgb;
        }
        // Ambient stays keyed to the flat albedo (it stands in for fill the
        // lobe doesn't model — same rule as `shade_hapke_surface`). Airless
        // bodies have no atmosphere fill, so this is the scene's authored
        // space/starlight term, day/night gated per fragment so it doesn't
        // wash the night hemisphere (`ambient_daylight_gate`). This branch is
        // already direct/ambient-split, so the shadow gates direct fully — on
        // an airless body a shadowed crater floor really is near-black.
        direct *= shadow_f;
        let ambient = albedo * lights.ambient_color.rgb * ambient_gate;
        out.color = vec4<f32>(
            (direct + ambient) * view.exposure,
            pbr_input.material.base_color.a,
        );
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    } else {
#ifdef VERTEX_UVS_A
#ifdef VERTEX_UVS_B
#ifdef VERTEX_COLORS
        // ── NTR-X4 material layer stack (vegetated tiles) ────────────────
        // Wrapped body-fixed position + canonical band weights from the
        // spare vertex channels. The forest weight rides the RAW vertex
        // alpha (`in.color.a`) — `alpha_discard` has already forced the
        // opaque base alpha back to 1, so the material base_color can't
        // carry it here.
        let p = vec3<f32>(in.uv.x, in.uv.y, in.uv_b.x);
        let eco_altitude_m = in.uv_b.y;
        let canopy_w = in.color.a;

        // Body-frame geometric normal + per-fragment ground footprint.
        let q = tile_params.orient;
        let n_body = quat_rotate(quat_conj(q), n);
        let footprint_m = length(fwidth(p));

        let layers = material_layers(
            p,
            n_body,
            pbr_input.material.base_color.rgb,
            eco_altitude_m,
            canopy_w,
            footprint_m,
        );
        pbr_input.material.base_color = vec4<f32>(layers.albedo, 1.0);
        // Roughness carries back the normal detail this footprint cannot
        // resolve (see `specular_aa_roughness`) — the far-field half of the
        // "terrain reads as satin" fix. Without it the detail normals above
        // only clean up the near field and distance restores the sheen.
        pbr_input.material.perceptual_roughness =
            specular_aa_roughness(layers.roughness, layers.normal_variance);
        // Per-class specular: vegetation reflects far less than the stock
        // dielectric the base material carries (see `VEG_REFLECTANCE`).
        pbr_input.material.reflectance = vec3<f32>(layers.reflectance);
        // Detail normal: perturb in the body frame, rotate back to world.
        // Shading normal only — `world_normal`/`n` stay geometric for the
        // stable-CSM receiver offset above.
        // `inspect == 2` keeps the geometric normal, so a frame's roughness can
        // be attributed to the detail normals or to the mesh, not argued about.
        let detail_off = select(layers.normal_offset, vec3<f32>(0.0), tile_params.inspect == 2u);
        let n_pert = normalize(n_body + detail_off);
        pbr_input.N = normalize(quat_rotate(q, n_pert));
#endif
#endif
#endif
        // Fullbright: the layer stack's own albedo, unlit and unfogged. The
        // capture-only diagnostic udlod has had since the mask work — it is
        // what separates a paint problem from a light problem in one shot.
        if tile_params.inspect == 1u {
            out.color = vec4<f32>(pbr_input.material.base_color.rgb, 1.0);
            return out;
        }
        // Day/night gate on every INDIRECT term Bevy adds — the flat ambient,
        // the environment map, irradiance volumes — all of which
        // `apply_pbr_lighting` scales by these two occlusion inputs and
        // nothing else (direct light is untouched, as it should be: its `n·l`
        // already zeroes on the night side). Multiplied in rather than
        // assigned so a real AO source keeps its say.
        pbr_input.diffuse_occlusion *= vec3<f32>(ambient_gate);
        pbr_input.specular_occlusion *= ambient_gate;
        // Direct/indirect split (exact, by linearity): occlusions zeroed ⇒ pure
        // exposure·direct, so the shadow (sun cascade × cloud transmittance)
        // subtracts only the sun's share and shadowed ground keeps its whole
        // sky fill — the deep, sky-tinted shadows the flat multiply flattened.
        var pbr_direct = pbr_input;
        pbr_direct.diffuse_occlusion = vec3<f32>(0.0);
        pbr_direct.specular_occlusion = 0.0;
        let direct_lit = apply_pbr_lighting(pbr_direct);
        out.color = apply_pbr_lighting(pbr_input);
        out.color = vec4<f32>(
            max(out.color.rgb - (1.0 - shadow_f) * direct_lit.rgb, vec3<f32>(0.0)),
            out.color.a,
        );
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    }
#endif

    return out;
}
