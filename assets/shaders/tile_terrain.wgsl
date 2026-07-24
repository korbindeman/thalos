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
//   rock   — slope-driven (steep faces expose rock regardless of the macro
//            band), with strata banding along a stable pseudo-bedding frame,
//            fall-line gully striation, and a Perlin detail normal.
//   scree  — the ~27–38° shoulder below rock faces: lighter, granular
//            debris blending into vegetation.
//   snow   — the canonical climate band weight (vertex-carried), sharpened
//            by a noise-broken line and shed from steep faces so rock ribs
//            poke through; low roughness.
//   forest — canonical forest band weight (vertex-carried): per-canopy cell
//            stippling (luminance + normal dimples) so aerial forest has
//            per-tree grain long before real scatter trees load.
//   meadow — everything else: the canonical albedo with fine grass mottle.
//
// Selection inputs come through the standard mesh pipeline's spare vertex
// channels (see `tiles::build_tile_mesh`):
//   uv      = wrapped body-fixed position .xy   (metres, mod TILE_WRAP_M)
//   uv_b.x  = wrapped body-fixed position .z
//   uv_b.y  = canonical snow band weight  [0,1]
//   color   = canonical macro albedo (rgb) + canonical forest band weight (a)
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
#import thalos::lighting::{hapke_brdf_rgb, hapke_w_from_albedo}
#import thalos::landcover::{hash13, wrap_lattice, value_noise_3d_periodic, fbm3_periodic}

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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // Body→world rotation (unit quaternion, xyzw), written per frame by the
    // game's tile driver (`update_tile_material_params`).
    orient: vec4<f32>,
    // Radial "up" at the view anchor, body-fixed (xyz; w unused). Up varies
    // ~1° per 50 km, so one uniform serves every fragment in frame; by the
    // time the error matters the ground is at the limb and sub-pixel.
    up_body: vec4<f32>,
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

// Same floor as `shadowed_standard.wgsl` / the hull, so every standard-path
// surface darkens identically in shadow.
const SHADOW_FLOOR: f32 = 0.4;

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

// Slope thresholds as cos(grade): 1 = flat. Rock above ~32°, scree on the
// ~22–32° shoulder — the reference's slope-driven exposure rule. Tuned LOW
// relative to real-world angles: the 90 m mesh understates true face slopes
// (a 4 km face over 12 km of 90 m verts averages well under 40°), so the
// thresholds must meet the geometry the mesh actually carries.
const ROCK_COS: f32 = 0.848;  // cos 32°
const SCREE_COS: f32 = 0.927; // cos 22°
// Noise jitter applied to the thresholds so transitions follow local texture,
// not contour lines.
const SLOPE_JITTER: f32 = 0.06;

// Layer palettes (linear RGB), anchored to the canonical macro band anchors
// in `procedural.rs::albedo_from_bands` so layers and palette agree.
const ROCK_ALBEDO: vec3<f32> = vec3<f32>(0.118, 0.120, 0.122);
const SCREE_ALBEDO: vec3<f32> = vec3<f32>(0.168, 0.158, 0.138);
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

// The NTR-X4 vegetated material stack's output.
struct LayerResult {
    albedo: vec3<f32>,
    roughness: f32,
    // Detail-normal offset in the body frame (added to n_body, renormalised).
    normal_offset: vec3<f32>,
}

fn material_layers(
    p: vec3<f32>,          // wrapped body-fixed position, metres
    n_body: vec3<f32>,     // geometric normal, body frame
    base_albedo: vec3<f32>,
    snow_w: f32,
    forest_w: f32,
    footprint_m: f32,
) -> LayerResult {
    let up = tile_params.up_body.xyz;
    let cosg = clamp(dot(n_body, up), -1.0, 1.0);

    // ── Selection fields ──────────────────────────────────────────────────
    let jitter = (fbm3_periodic(p / 256.0, 3, TILE_WRAP_M / 256.0) - 0.5) * 2.0;
    let cosg_j = cosg + jitter * SLOPE_JITTER;

    // Rock where steep; scree on the shoulder below it.
    let rock_m = 1.0 - smoothstep(ROCK_COS - 0.04, ROCK_COS + 0.04, cosg_j);
    let scree_m = (1.0 - smoothstep(SCREE_COS - 0.05, SCREE_COS + 0.05, cosg_j)) * (1.0 - rock_m);

    // Snow: canonical climate weight sharpened into a noise-broken line and
    // shed from steep faces so rock ribs poke through (reference pt. 5).
    // Full snow only holds on benches/hollows (< ~21°); everything steeper
    // sheds progressively — this is what breaks the "white blob" massif.
    let snow_break = (fbm3_periodic(p / 128.0, 3, TILE_WRAP_M / 128.0) - 0.5) * 2.0;
    let shed = smoothstep(0.80, 0.93, cosg + 0.04 * jitter);
    let snow_m = smoothstep(0.30, 0.60, snow_w + 0.28 * snow_break) * shed;

    // Forest: canonical band weight, gated off rock/snow and steep ground.
    let forest_gate = smoothstep(0.55, 0.75, cosg) * (1.0 - rock_m) * (1.0 - snow_m);
    let forest_density = smoothstep(0.15, 0.55, forest_w + 0.20 * jitter) * forest_gate;

    // ── Layers ────────────────────────────────────────────────────────────
    var albedo = base_albedo;
    var roughness = 0.95;
    var n_off = vec3<f32>(0.0);

    // Meadow mottle: fine grass-tone variation on open vegetated ground.
    let meadow_fade = footprint_fade(footprint_m, 8.0, 32.0);
    if meadow_fade > 0.0 {
        let mottle = fbm3_periodic(p / 16.0, 3, TILE_WRAP_M / 16.0) - 0.5;
        albedo = albedo * (1.0 + meadow_fade * 0.35 * mottle);
    }

    // Forest canopy stipple: per-canopy cells darken/vary luminance and
    // dimple the normal — aerial forest grain (reference pt. 4).
    let canopy_fade = footprint_fade(footprint_m, 12.0, 48.0);
    if forest_density > 0.0 {
        let cell = value_noise_3d_periodic(p / 24.0, TILE_WRAP_M / 24.0);
        let cell2 = value_noise_3d_periodic(p / 12.0, TILE_WRAP_M / 12.0);
        let canopy_tone = 0.55 + 0.9 * (0.6 * cell + 0.4 * cell2);
        let forest_albedo = base_albedo * vec3<f32>(0.52, 0.62, 0.50) * canopy_tone;
        albedo = mix(albedo, forest_albedo, forest_density * max(canopy_fade, 0.25));
        roughness = mix(roughness, 1.0, forest_density);
        if canopy_fade > 0.0 {
            let dg = fbm3_perlin_grad(p / 18.0, 2, TILE_WRAP_M / 18.0);
            n_off += dg.yzw * (forest_density * canopy_fade * 0.30);
        }
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

        // Fall-line gully striation: anisotropic noise stretched downhill.
        // (Slope-local coordinates — aperiodic by construction; a wrap seam
        // in this high-frequency streaking is not resolvable at the
        // distances rock faces are viewed from.)
        let fall = normalize(up - n_body * cosg + vec3<f32>(1.0e-5, 0.0, 0.0));
        let across = normalize(cross(n_body, fall));
        // ~24 m across-slope, ~150 m along the fall line: couloir-scale
        // striation that stays resolvable from the showcase framings (finer
        // scales aliased into speckle — round-2 finding).
        let gully_uv = vec3<f32>(dot(p, across) / 24.0, dot(p, fall) / 150.0, 0.0);
        let gully = fbm3_periodic(gully_uv, 2, 1.0e6) - 0.5;

        let rock_fade = footprint_fade(footprint_m, 30.0, 140.0);
        var rock_albedo = mix(ROCK_ALBEDO, base_albedo, 0.22);
        let macro_tone = fbm3_periodic(p / 1024.0, 3, TILE_WRAP_M / 1024.0) - 0.5;
        rock_albedo = rock_albedo * (1.0 + 0.30 * macro_tone);
        // Structure leads, grain follows: the banding + striation carry most
        // of the albedo modulation so faces read as bedded rock, not static.
        rock_albedo = rock_albedo * (1.0 + rock_fade * (0.42 * strata + 0.30 * gully));

        albedo = mix(albedo, rock_albedo, rock_m);
        roughness = mix(roughness, 0.88, rock_m);

        // Detail normal: strata ledges along the bedding axis + gully
        // striation across the fall line, with only a whisper of isotropic
        // grain — grain-dominant perturbation reads as speckle from the
        // showcase distances (round-2 finding).
        let detail_fade = footprint_fade(footprint_m, 15.0, 90.0);
        if detail_fade > 0.0 {
            let dg = fbm3_perlin_grad(p / 64.0, 2, TILE_WRAP_M / 64.0);
            let ledge = smoothstep(0.25, 0.75, band) - smoothstep(0.75, 0.98, band);
            n_off += (dg.yzw * 0.22 + normalize(DIP_K) * ledge * 0.7 + across * gully * 0.9)
                * rock_m * detail_fade * 0.5;
        }
    }

    // Scree: lighter granular debris apron between rock and vegetation.
    if scree_m > 0.0 {
        let scree_fade = footprint_fade(footprint_m, 10.0, 50.0);
        let grain = fbm3_periodic(p / 8.0, 2, TILE_WRAP_M / 8.0) - 0.5;
        var scree_albedo = mix(SCREE_ALBEDO, base_albedo, 0.30);
        scree_albedo = scree_albedo * (1.0 + scree_fade * 0.35 * grain);
        albedo = mix(albedo, scree_albedo, scree_m * 0.85);
        roughness = mix(roughness, 0.97, scree_m);
        if scree_fade > 0.0 {
            let dg = fbm3_perlin_grad(p / 16.0, 2, TILE_WRAP_M / 16.0);
            n_off += dg.yzw * scree_m * scree_fade * 0.15;
        }
    }

    // Snow last: covers whatever it claims; slight drift sub-structure so
    // fields aren't flat white, and it smothers the detail normal beneath.
    if snow_m > 0.0 {
        let drift = fbm3_periodic(p / 128.0, 2, TILE_WRAP_M / 128.0) - 0.5;
        let snow_albedo = SNOW_ALBEDO * (1.0 + 0.06 * drift);
        albedo = mix(albedo, snow_albedo, snow_m);
        roughness = mix(roughness, 0.38, snow_m);
        n_off = n_off * (1.0 - 0.8 * snow_m);
    }

    var out: LayerResult;
    out.albedo = albedo;
    out.roughness = roughness;
    out.normal_offset = n_off;
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
    let shadow_f = sun_shadow_factor_nrm(
        pbr_input.world_position.xyz,
        n,
        recv_shadow,
        recv_shadow_map_0,
        recv_shadow_map_1,
        recv_shadow_map_2,
    );

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
        // space/starlight term.
        direct *= max(shadow_f, SHADOW_FLOOR);
        let ambient = albedo * lights.ambient_color.rgb;
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
        let snow_w = in.uv_b.y;
        let forest_w = in.color.a;

        // Body-frame geometric normal + per-fragment ground footprint.
        let q = tile_params.orient;
        let n_body = quat_rotate(quat_conj(q), n);
        let footprint_m = length(fwidth(p));

        let layers = material_layers(
            p,
            n_body,
            pbr_input.material.base_color.rgb,
            snow_w,
            forest_w,
            footprint_m,
        );
        pbr_input.material.base_color = vec4<f32>(layers.albedo, 1.0);
        pbr_input.material.perceptual_roughness = layers.roughness;
        // Detail normal: perturb in the body frame, rotate back to world.
        // Shading normal only — `world_normal`/`n` stay geometric for the
        // stable-CSM receiver offset above.
        let n_pert = normalize(n_body + layers.normal_offset);
        pbr_input.N = normalize(quat_rotate(q, n_pert));
#endif
#endif
#endif
        out.color = apply_pbr_lighting(pbr_input);
        out.color = vec4<f32>(
            out.color.rgb * max(shadow_f, SHADOW_FLOOR),
            out.color.a,
        );
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    }
#endif

    return out;
}
