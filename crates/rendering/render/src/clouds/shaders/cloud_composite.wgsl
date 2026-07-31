// Canonical fullscreen cloud composite. The custom BodySky atmosphere renders
// earlier in the frame; this pass owns both the near-volume layer and the
// weather-derived orbital projection.

#import bevy_pbr::mesh_view_bindings::view
#import thalos::atmosphere::{
    AtmosphereBlock,
    weather_column_from_texel,
    orbital_cloud_altitude,
    orbital_cloud_shade,
    sample_weather_soft,
    orbital_cloud_normal_body,
    cloud_surface_density,
    WEATHER_TEXEL_ANGLE,
    cloud_cell_field,
    cloud_cell_style,
    cloud_strata_warp,
    cloud_far_ownership,
    cloud_march_stop_m,
    CLOUD_MARCH_FADE_FRACTION,
}
#import thalos::lighting::{SCENE_FLUX_SCALE, SURFACE_DIRECT_SCALE}
#import thalos::volumetrics::water_cloud_albedo

// The diffusion-limit reflectance both tiers are anchored to now has ONE
// definition, in `thalos::volumetrics`. It used to be written out here with a
// comment saying it MUST equal `CLOUD_MS_ALBEDO` in clouds_compute.wgsl — read
// `water_cloud_albedo()` instead. (It cannot be a `const` here: naga_oil does not
// carry imported consts across, and a `const` initializer cannot call.)
// `orbital_cloud_shade`'s radiance scale for a fully lit, optically thick,
// storm-free column (lit = 1, core ≈ 0.9): 0.72 · (0.18 + 0.82) · 0.85.
const FAR_SHADE_LIT: f32 = 0.61;
// Chroma is the AUTHORED climate albedo (`cloud_albedo_coverage.rgb`) and
// nothing else. The old extra `(0.90, 0.93, 0.97)` was headroom for phase
// peaks, but it is per-channel, so stacked on Thalos's authored (0.94, 0.96,
// 1.0) it gave "white" clouds a 15% blue bias — a second, independent reason
// they never read white. Peak headroom belongs in the near march's Reinhard
// white point, which is achromatic.
const FAR_CLOUD_TINT: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

@group(3) @binding(0) var<uniform> cloud_atmosphere: AtmosphereBlock;

struct CloudCompositeParams {
    sun_dir_flux:              vec4<f32>,
    planet_center_radius:      vec4<f32>,
    world_to_body_orientation: vec4<f32>,
    cloud_band_radii:          vec4<f32>,
    ocean:                     vec4<f32>,
    ocean_color_depth:         vec4<f32>,
    ocean_camera_phase:        vec4<f32>,
    ocean_low_phase:           vec4<f32>,
    ocean_high_phase:          vec4<f32>,
    ocean_slope_amplitudes:    vec4<f32>,
    ocean_spectrum:            vec4<f32>,
    ocean_wind_basis:          vec4<f32>,
    ocean_crosswind_basis:     vec4<f32>,
    tile_lookup:               vec4<f32>,
    tile_atlas_uv:             vec4<f32>,
    // x = near-march view step count; the composite mirrors the marcher's
    // per-ray far ownership from it (see far_ownership).
    // y = tier diagnostic, z = far footprint-mip mode, w = far aggregation.
    cloud_march:               vec4<f32>,
    // Far-tier opacity response: 16 nodes of expected near-column opacity vs
    // the profile-weighted strata mean (node i at i/15), derived per body by
    // `fill_lut::derive_fill_calibration` together with the near threshold
    // curve. Rendering this LUT is what keeps far thickness equal to the near
    // volume's — never replace it with a hand-tuned curve
    // (BL-20260723T214730Z).
    fill_response:             array<vec4<f32>, 4>,
    // Resolved-footprint pairing, derived WITH fill_response by the same
    // calibration: per strata-mean node, the cell-field value above which a
    // column is cloud (edge) and the occupied columns' mean opacity (solid).
    // `(cell > edge) · solid` equals fill_response in expectation by
    // construction — it concentrates the same opacity into cell interiors
    // instead of a smooth translucent veil.
    fill_cell_edge:            array<vec4<f32>, 4>,
    fill_cell_solid:           array<vec4<f32>, 4>,
    // Sky-ambient radiance pair (drive_clouds' `clouds_ambient_color_*`, the
    // near march's exact ambient inputs — SkyAmbient irradiance/π).
    cloud_ambient_top:         vec4<f32>,
    cloud_ambient_bottom:      vec4<f32>,
    // x = cell-scale cloud evolution, sim seconds; yzw reserved. Must match the
    // near marcher's `config.cell_evolution_s` or the tiers disagree on shape.
    cloud_time:                vec4<f32>,
}
@group(3) @binding(1) var<uniform> cloud_params: CloudCompositeParams;
@group(3) @binding(2) var scene_depth_texture: texture_depth_2d;
@group(3) @binding(3) var weather_texture: texture_cube<f32>;
@group(3) @binding(4) var weather_sampler: sampler;
@group(3) @binding(5) var cloud_layer_texture: texture_2d<f32>;
@group(3) @binding(6) var cloud_distance_texture: texture_2d<f32>;
@group(3) @binding(7) var surface_density_texture: texture_cube<f32>;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4(in.position.xy, 1.0, 1.0);
    return out;
}

fn rotate_quat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

struct CloudOverlay {
    premul_rgb: vec3<f32>,
    opacity: f32,
}

fn reconstruct_ray(pixel: vec2<f32>) -> vec3<f32> {
    let ndc_x = (pixel.x / view.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y / view.viewport.w) * 2.0;
    let cam_right = view.world_from_view[0].xyz;
    let cam_up = view.world_from_view[1].xyz;
    let cam_fwd = -view.world_from_view[2].xyz;
    return normalize(
        cam_right * (ndc_x / view.clip_from_view[0][0])
        + cam_up * (ndc_y / view.clip_from_view[1][1])
        + cam_fwd
    );
}

fn scene_distance(pixel: vec2<f32>) -> f32 {
    let depth = textureLoad(scene_depth_texture, vec2<i32>(pixel), 0);
    if depth <= 0.0 {
        return 1.0e30;
    }
    let ndc_x = (pixel.x / view.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y / view.viewport.w) * 2.0;
    let view_h = view.view_from_clip * vec4<f32>(ndc_x, ndc_y, depth, 1.0);
    return length(view_h.xyz / view_h.w);
}

fn sample_near_cloud(pixel: vec2<f32>) -> vec4<f32> {
    let dims = textureDimensions(cloud_layer_texture);
    let cloud_res = vec2<f32>(dims);
    let uv = pixel / view.viewport.zw;
    let p = uv * cloud_res - 0.5;
    let base = floor(p);
    let f = p - base;
    // Last rows/columns store the temporal camera metadata.
    let bilinear_max = vec2<i32>(i32(dims.x) - 2, i32(dims.y) - 4);
    let cb = clamp(vec2<i32>(base), vec2<i32>(0), bilinear_max);
    let cs00 = textureLoad(cloud_layer_texture, cb, 0);
    let cs10 = textureLoad(cloud_layer_texture, cb + vec2<i32>(1, 0), 0);
    let cs01 = textureLoad(cloud_layer_texture, cb + vec2<i32>(0, 1), 0);
    let cs11 = textureLoad(cloud_layer_texture, cb + vec2<i32>(1, 1), 0);
    let cd00 = textureLoad(cloud_distance_texture, cb, 0).r;
    let cd10 = textureLoad(cloud_distance_texture, cb + vec2<i32>(1, 0), 0).r;
    let cd01 = textureLoad(cloud_distance_texture, cb + vec2<i32>(0, 1), 0).r;
    let cd11 = textureLoad(cloud_distance_texture, cb + vec2<i32>(1, 1), 0).r;
    let ref_coord = cb + vec2<i32>(select(0, 1, f.x >= 0.5), select(0, 1, f.y >= 0.5));
    let cloud_near = textureLoad(cloud_distance_texture, ref_coord, 0).r;
    let depth_scale = max(cloud_near, 2000.0);
    let hit_ref = cloud_near < 1.0e8;
    let dw00 = select(0.0, select(1.0, exp(-abs(cd00 - cloud_near) / depth_scale), hit_ref && cd00 < 1.0e8), hit_ref == (cd00 < 1.0e8));
    let dw10 = select(0.0, select(1.0, exp(-abs(cd10 - cloud_near) / depth_scale), hit_ref && cd10 < 1.0e8), hit_ref == (cd10 < 1.0e8));
    let dw01 = select(0.0, select(1.0, exp(-abs(cd01 - cloud_near) / depth_scale), hit_ref && cd01 < 1.0e8), hit_ref == (cd01 < 1.0e8));
    let dw11 = select(0.0, select(1.0, exp(-abs(cd11 - cloud_near) / depth_scale), hit_ref && cd11 < 1.0e8), hit_ref == (cd11 < 1.0e8));
    let w00 = (1.0 - f.x) * (1.0 - f.y) * dw00;
    let w10 = f.x * (1.0 - f.y) * dw10;
    let w01 = (1.0 - f.x) * f.y * dw01;
    let w11 = f.x * f.y * dw11;
    let weight = max(w00 + w10 + w01 + w11, 1.0e-5);
    return (cs00 * w00 + cs10 * w10 + cs01 * w01 + cs11 * w11) / weight;
}

/// Fraction of the near marcher's result that survives occlusion by opaque
/// geometry at `scene_t`.
///
/// The marcher integrates its whole in-shell chord without knowing where the
/// ground is, so cloud BEHIND a mountain is in the result and has to be taken
/// back out here. What the march hands over is the chord's total transmittance
/// plus its extinction slab (`slab_near`..`slab_far`, matched to the real
/// profile's first moment in `slab_far_distance`).
///
/// The partition is therefore taken in OPTICAL DEPTH, not in opacity: the depth
/// in front of the scene is `tau * frac`, and the opacity that survives is
/// `1 - exp(-tau * frac)`, renormalized against the full-chord opacity the
/// caller is scaling. Ramping opacity linearly instead — as the predecessor did,
/// against a CONSTANT 5.4 km denominator — is what drew solid cumulus as
/// see-through smoke wherever terrain sat behind them, and left them solid only
/// against open sky (INC-20260729T051500Z).
fn near_visibility(
    cloud_near: f32,
    slab_far: f32,
    transmittance: f32,
    scene_t: f32,
    oc_len_sq: f32,
    b: f32,
) -> f32 {
    let r_base = cloud_params.cloud_band_radii.x;
    let r_top = cloud_params.cloud_band_radii.y;
    if scene_t >= 1.0e29 || r_top <= r_base {
        return 1.0;
    }
    let cam_r = sqrt(oc_len_sq);
    let disc_base = b * b - (oc_len_sq - r_base * r_base);
    let disc_top = b * b - (oc_len_sq - r_top * r_top);
    let sqrt_base = sqrt(max(disc_base, 0.0));
    let sqrt_top = sqrt(max(disc_top, 0.0));
    var band_far = 1.0e30;
    if cam_r < r_base {
        band_far = max(-b + sqrt_top, 0.0);
    } else if cam_r <= r_top {
        var exit = -b + sqrt_top;
        let base_down = -b - sqrt_base;
        if disc_base > 0.0 && base_down > 0.0 {
            exit = min(exit, base_down);
        }
        band_far = max(exit, 1.0);
    } else if disc_base > 0.0 {
        band_far = max(-b - sqrt_base, 0.0);
    } else {
        band_far = max(-b + sqrt_top, 0.0);
    }
    let near = min(cloud_near, band_far);
    // The marched slab, clipped to the shell chord. No constant lives here any
    // more: the extent is the one the march measured on THIS ray, so a scene hit
    // beyond the cloud's own back face partitions to exactly 1.
    let extent = max(min(slab_far, band_far) - near, 0.0);
    let frac = select(
        clamp((scene_t - near) / extent, 0.0, 1.0),
        select(0.0, 1.0, scene_t >= near),
        extent <= 1.0,
    );
    // Renormalized against the full-chord opacity, so `frac == 1` returns
    // exactly 1 for any transmittance and the caller's scaling is a no-op when
    // nothing occludes the cloud.
    let tau = -log(clamp(transmittance, 1.0e-4, 1.0));
    let opacity_full = 1.0 - exp(-tau);
    if opacity_full <= 1.0e-5 {
        return frac;
    }
    return clamp((1.0 - exp(-tau * frac)) / opacity_full, 0.0, 1.0);
}

// Share of this ray owned by the far/orbital projection. The near marcher now
// always COMPLETES its in-shell chord (it floors its step on the chord budget
// instead of stopping at a reach frontier), so there is no partial result to
// partition against and no mid-view seam to place: ownership is purely a
// question of whether cell-scale morphology is still resolvable.
//
// This replaces a three-component reach mirror (fade_begin, t_stop, entry
// ownership) that had to track the marcher's banded step law exactly. Keeping
// two integrators in lockstep on a *distance* ladder is what produced the
// ownership arc, the entry-window over-count veil, and finally the ascent
// fade-out where the far tier renders a near-empty wash over regions the near
// tier drew as solid cells. Footprint is a per-ray scalar both sides can
// compute independently and cannot drift on.
fn far_ownership(oc_len_sq: f32, b: f32, pixel_angle: f32) -> vec3<f32> {
    let r_base = cloud_params.cloud_band_radii.x;
    let r_top = cloud_params.cloud_band_radii.y;
    let cam_r = sqrt(oc_len_sq);
    let disc_top = b * b - (oc_len_sq - r_top * r_top);
    if disc_top <= 0.0 {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let sq_top = sqrt(disc_top);
    let tt0 = -b - sq_top;
    let disc_base = b * b - (oc_len_sq - r_base * r_base);
    let hit_base = disc_base > 0.0;
    var tb1 = 0.0;
    if hit_base {
        tb1 = -b + sqrt(disc_base);
    }
    // Shell entry, matching `get_ray`'s three camera regimes.
    var seg_start = 0.0;
    if cam_r > r_top {
        seg_start = tt0;
    } else if cam_r < r_base {
        if hit_base && tb1 > 0.0 { seg_start = tb1; } else { seg_start = max(tt0, 0.0); }
    }
    seg_start = max(seg_start, 0.0);
    // z = whole-ray ownership (cells sub-pixel). xy = the marcher's budget
    // frontier, which only grazing rays reach; beyond it this tier owns the
    // tail because the near march has run out of probes, not because anything
    // about the field changed.
    let steps = max(cloud_params.cloud_march.x, 1.0);
    let stop = cloud_march_stop_m(steps, seg_start, pixel_angle);
    let fade_begin = mix(seg_start, stop, CLOUD_MARCH_FADE_FRACTION);
    return vec3<f32>(fade_begin, stop, cloud_far_ownership(seg_start * pixel_angle));
}

// Reduced surface-density band march: the far estimator's answer to grazing
// geometry. K stratified samples traverse the ray's shell band accumulating
// column occupancy × local vertical profile, so the limb gets true vertical
// thickness and distant horizon decks break where columns are clear — instead
// of one representative sphere whose tangent painted a hard-edged solid wall.
// Weather varies at ≥ ~5 km/texel at the base level, so K samples across even a 500 km chord
// stay well-sampled (no 3-D noise is touched here).
const ORBITAL_MARCH_SAMPLES = 6u;

/// How hard the shared cell field modulates this tier's response-LUT input.
/// It is a spatial REDISTRIBUTION of a strata mean the LUT already calibrates,
/// so it is mean-preserving to first order and must not be reached for when
/// the far tier looks too thick or too thin — that is the fill pairing's job,
/// and re-tuning it here is what the derived `fill_lut` exists to prevent.
const FAR_CELL_AMPLITUDE: f32 = 0.55;

/// Chords longer than this aggregate per SEGMENT instead of through one
/// chord-mean → one LUT lookup → one cell anchor. A grazing chord crosses many
/// cells; collapsing it to its mean before the response LUT renders exactly
/// the mean — a structureless translucent veil over the whole distant band
/// (the 2026-07-29 horizon "smear"), with no gaps for the terrain no matter
/// how broken the field is. Two coarsest cell periods: below that a single
/// anchor is inside one cell and the mean path is both correct and cheaper.
const FAR_CHORD_STRUCTURE_MIN_M: f32 = 10800.0;

// Sample a 16-node LUT packed as 4 × vec4 (nodes over [0, 1], linear between
// nodes). The parameter copy gives naga the `var` reference dynamic indexing
// needs.
fn lut16_sample(lut_in: array<vec4<f32>, 4>, x: f32) -> f32 {
    var lut = lut_in;
    let t = clamp(x, 0.0, 1.0) * 15.0;
    let i = u32(min(t, 14.0));
    let f = t - f32(i);
    let a = lut[i / 4u][i % 4u];
    let b = lut[(i + 1u) / 4u][(i + 1u) % 4u];
    return mix(a, b, f);
}

// Sample the derived far opacity response (see `fill_response` above).
fn fill_response_sample(strata_mean: f32) -> f32 {
    return lut16_sample(cloud_params.fill_response, strata_mean);
}

/// Antialiasing half-width of the resolved cell-edge threshold, in cell-field
/// units. The field's own footprint fade supplies the coarse-footprint
/// softening; this only keeps the resolved edge from aliasing at one texel.
const FAR_CELL_EDGE_AA: f32 = 0.05;

/// Resolved-footprint far opacity for one column crossing: cells above the
/// derived edge render at the derived solid opacity, gaps render nothing, and
/// the whole term relaxes to the smooth expected response exactly as cell
/// morphology goes sub-pixel (`cloud_far_ownership`, the ADR-20260726T000929Z
/// resolvability scale).
fn far_column_opacity(profile: f32, cell: f32, footprint_m: f32) -> f32 {
    let smooth_a = fill_response_sample(
        clamp(profile + (cell - 0.5) * FAR_CELL_AMPLITUDE, 0.0, 1.0),
    );
    let edge = lut16_sample(cloud_params.fill_cell_edge, profile);
    let solid = lut16_sample(cloud_params.fill_cell_solid, profile);
    let occupied = smoothstep(edge - FAR_CELL_EDGE_AA, edge + FAR_CELL_EDGE_AA, cell);
    return mix(occupied * solid, smooth_a, cloud_far_ownership(footprint_m));
}

fn sample_orbital_cloud(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    planet_center: vec3<f32>,
    planet_radius: f32,
    oc_len_sq: f32,
    b: f32,
    scene_t: f32,
) -> CloudOverlay {
    if cloud_atmosphere.cloud_albedo_coverage.w <= 0.0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }
    let r_base = cloud_params.cloud_band_radii.x;
    let r_top = cloud_params.cloud_band_radii.y;
    if r_top <= r_base {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }

    // First forward slab of the shell band, same regime logic as the marcher.
    let disc_top = b * b - (oc_len_sq - r_top * r_top);
    if disc_top <= 0.0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }
    let sq_top = sqrt(disc_top);
    let tt0 = -b - sq_top;
    let tt1 = -b + sq_top;
    let disc_base = b * b - (oc_len_sq - r_base * r_base);
    let hit_base = disc_base > 0.0;
    var tb0 = 0.0;
    var tb1 = 0.0;
    if hit_base {
        let sq_base = sqrt(disc_base);
        tb0 = -b - sq_base;
        tb1 = -b + sq_base;
    }
    let cam_r = sqrt(oc_len_sq);
    var t0 = 0.0;
    var t1 = 0.0;
    if cam_r > r_top {
        t0 = tt0;
        if hit_base && tb0 > tt0 { t1 = tb0; } else { t1 = tt1; }
    } else if cam_r < r_base {
        if hit_base && tb1 > 0.0 { t0 = tb1; } else { t0 = max(tt0, 0.0); }
        t1 = tt1;
    } else {
        t0 = 0.0;
        if hit_base && tb0 > 0.0 { t1 = tb0; } else { t1 = tt1; }
    }
    t0 = max(t0, 0.0);
    t1 = min(t1, scene_t);
    if t1 <= t0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }

    // Per-pixel angle of THIS full-resolution pass — the far tier's own
    // sampling/LOD footprint.
    let pixel_angle = 2.0 / max(view.viewport.z * view.clip_from_view[0][0], 1.0);
    // Ownership, however, must reproduce the MARCHER's budget law, and the
    // marcher's pixel is the resolution-scaled cloud target's, not this
    // pass's: at the default 2/3 scale the two angles differ by 1.5×, which
    // is enough to put the two integrators in DIFFERENT reach regimes on the
    // same ray — the composite blocking the far tier over rays the marcher
    // had already abandoned (2026-07-31, the LEO bare annulus). The target
    // width rides in `cloud_time.y`; fall back to this pass's angle if a
    // driver ever leaves it unset.
    var march_pixel_angle = pixel_angle;
    if (cloud_params.cloud_time.y >= 1.0) {
        march_pixel_angle = 2.0 / max(cloud_params.cloud_time.y * view.clip_from_view[0][0], 1.0);
    }
    let reach = far_ownership(oc_len_sq, b, march_pixel_angle);
    let base_alt = max(cloud_atmosphere.cloud_shape.x, 0.0);
    let thickness = max(cloud_atmosphere.cloud_shape.y, 1.0);

    let seg = (t1 - t0) / f32(ORBITAL_MARCH_SAMPLES);
    let inv_thickness = 1.0 / thickness;
    // Footprint mip: one march segment's angular span over the weather cube.
    let lod_chord = clamp(
        log2(max((seg / planet_radius) / WEATHER_TEXEL_ANGLE, 1.0)),
        0.0,
        7.0,
    );
    // `col.opacity` is an AREAL FRACTION (see weather_cloud_opacity), so the
    // chord takes the strongest single column plus a damped stacking term —
    // compounding fractions as independent opaque slabs saturated a
    // 46%-coverage planet into solid overcast.
    var best_c = 0.0;
    var sum_c = 0.0;
    var sum_t = 0.0;
    // Occupancy-weighted cloud type along the chord, for the morphology style
    // below. Accumulated here rather than re-fetched at `n_morph` because the
    // loop already has the texel, and because weighting it exactly like `sum_t`
    // is what makes the style belong to the same column the morphology is
    // anchored to.
    var sum_type = 0.0;
    var best_s = 0.0;
    var sum_s = 0.0;
    var sample_weight = 0.0;
    var best_t = -1.0;
    var best_lod = 7.0;
    // Per-segment record for the chord-structured aggregation below. Sizes are
    // ORBITAL_MARCH_SAMPLES; zero-initialized, and segments that never enter
    // the layer stay zero (their `continue` skips the store).
    var seg_profile: array<f32, 6>;
    var seg_w: array<f32, 6>;
    var seg_t_m: array<f32, 6>;
    var seg_type: array<f32, 6>;
    // Height at the running segment boundary, as a fraction of the shell.
    var h_prev = (length(cam_pos + t0 * ray_dir - planet_center) - planet_radius - base_alt)
        * inv_thickness;
    for (var i = 0u; i < ORBITAL_MARCH_SAMPLES; i++) {
        let t_a = t0 + f32(i) * seg;
        let t_b = t_a + seg;
        let t_m = 0.5 * (t_a + t_b);
        let h_a = h_prev;
        let h_b = (length(cam_pos + t_b * ray_dir - planet_center) - planet_radius - base_alt)
            * inv_thickness;
        h_prev = h_b;
        let p = cam_pos + t_m * ray_dir - planet_center;
        let n_l = rotate_quat(cloud_params.world_to_body_orientation, normalize(p));
        let pixel_world_m = t_m * pixel_angle;
        let lod_pixel = clamp(
            log2(max((pixel_world_m / planet_radius) / WEATHER_TEXEL_ANGLE, 1.0)),
            0.0,
            7.0,
        );
        let lod = mix(lod_chord, lod_pixel, clamp(cloud_params.cloud_march.z, 0.0, 1.0));
        let weather = textureSampleLevel(weather_texture, weather_sampler, n_l, lod);
        let col = weather_column_from_texel(weather);
        // The 0.75 lod floor (sample_weather_soft's lattice softener) only
        // where the footprint is genuinely unresolved: at handoff ranges the
        // pixel sits deep inside one ~5 km texel (pure magnification, already
        // bilinear-smooth), and forcing the extra half-mip of blur there
        // widened every strata cell by kilometres of low-alpha halo — a large
        // share of the far tier's measured 3× areal excess over the near
        // volume at the 2026-07-24 A/B framing.
        let strata_floor = 0.75 * smoothstep(550.0, 1600.0, pixel_world_m);
        // Resolved footprints warp the strata lookup (shared contract with
        // the marcher's homogenized bands) so the ~5 km texel lattice reads
        // as organic cells instead of rounded squares (user ascent verdict).
        // Unconditional, matching the marcher: the warp is measure-preserving,
        // and gating it by footprint made the two tiers sample DIFFERENT
        // directions wherever the gate disagreed across the handoff.
        let n_s = cloud_strata_warp(n_l, 1.0);
        let strata = textureSampleLevel(
            surface_density_texture,
            weather_sampler,
            n_s,
            max(lod, strata_floor),
        );
        // Analytic per-segment layer overlap, evaluated in LAYER-RELATIVE
        // space: clip the segment's height span against this texel's
        // [base, top] and integrate the strata profile over the clipped
        // portion only. Segments that never enter the layer contribute
        // nothing AND don't dilute the mean — averaging over all six
        // segments made a long grazing chord divide its one in-layer hit
        // toward zero, which erased every cloud from the limb view.
        let seg_lo = min(h_a, h_b);
        let seg_hi = max(h_a, h_b);
        let ov_lo = max(seg_lo, col.base_frac);
        let ov_hi = min(seg_hi, col.top_frac);
        if ov_hi <= ov_lo {
            continue;
        }
        // Soft overlap weight: a segment fades in as its clipped span grows
        // toward half the layer thickness. A hard on/off clip made the
        // contribution DISCONTINUOUS in view angle, and tangent-geometry ray
        // families swept that step into razor-edged "knife blade" streaks
        // across the sky (user frames, 2026-07-23). Fully-inside grazing
        // segments all carry the same small weight, so the weighted mean —
        // and therefore grazing opacity — is unchanged.
        let layer_span = max(col.top_frac - col.base_frac, 0.02);
        let overlap_w = clamp((ov_hi - ov_lo) / (0.5 * layer_span), 0.0, 1.0);
        let w_seg = overlap_w;
        sample_weight += w_seg;
        let inv_layer = 1.0 / layer_span;
        let l_a = (ov_lo - col.base_frac) * inv_layer;
        let l_b = (ov_hi - col.base_frac) * inv_layer;
        let density_a = cloud_surface_density(strata, l_a);
        let density_m = cloud_surface_density(strata, 0.5 * (l_a + l_b));
        let density_b = cloud_surface_density(strata, l_b);
        let profile = 0.25 * density_a + 0.50 * density_m + 0.25 * density_b;
        if profile <= 1.0e-4 {
            continue;
        }
        // Pure occupancy signal: `mean_c` below must be exactly the variable
        // the derived `fill_response` LUT was conditioned on (the
        // profile-weighted strata mean). Thinness/optical-depth response is
        // already inside the LUT (it stores expected near-column OPACITY),
        // so multiplying a per-column thinness term here would double-count.
        let candidate = profile * w_seg;
        // The legacy stacked A/B path (cloud_march.w = 0) keeps its explicit
        // thinness multiplier — it predates the derived response.
        let a_col = mix(0.70, 1.0, 1.0 - exp(-col.optical_depth * 1.4));
        let candidate_stacked = candidate * a_col;
        if candidate > best_c {
            best_c = candidate;
            best_t = t_m;
            best_lod = lod;
        }
        sum_c += candidate;
        sum_t += t_m * candidate;
        sum_type += weather.g * candidate;
        best_s = max(best_s, candidate_stacked);
        sum_s += candidate_stacked;
        seg_profile[i] = profile;
        seg_w[i] = w_seg;
        seg_t_m[i] = t_m;
        seg_type[i] = weather.g;
    }
    let stacked_opacity = best_s + 0.35 * (sum_s - best_s) * (1.0 - best_s);
    let mean_c = sum_c / max(sample_weight, 1.0e-4);
    if best_t < 0.0 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }
    let p_best = cam_pos + best_t * ray_dir - planet_center;

    // Cell-scale morphology — THE SAME FIELD THE MARCHER FORMS CLOUD FROM
    // (`cloud_cell_field`, thalos::atmosphere). The strata cube's ~5 km texels
    // can only render smooth blobs with wide halos; the cells that make a deck
    // read as a deck live at 1–5 km and are analytic. Perturbing the LUT input
    // with them concentrates opacity into cell interiors and drops the halos
    // below the response toe — mean-preserving to first order, because the
    // field's own mean is 0.5 by construction.
    //
    // This replaces the earlier value-noise "mottle": a smooth perturbation at
    // roughly the right scale reads as dither, not as clouds, which is what
    // the orbital captures showed. Both tiers now perturb by the identical
    // field at the identical amplitude, so the crossfade is between two
    // integrators of one morphology rather than two textures that merely have
    // similar statistics.
    //
    // The field is anchored at the occupancy-weighted mean chord position —
    // NOT `p_best`: the best-segment argmax flips discontinuously between
    // adjacent rays, and feeding that into an opacity-bearing term cut
    // straight "torn seam" lines across every cell (first capture round).
    // Continuity guard: as occupancy fades to zero the weighted mean would
    // divide by the epsilon and jump by orders of magnitude — a hard seam in
    // the coordinate exactly along every cell edge. Blend toward the
    // (continuous) chord midpoint below a small occupancy, where the LUT
    // renders nothing anyway.
    //
    // Alias safety is the field's own `filter_m` fade now, fed the projected
    // pixel footprint: the ad-hoc resolve/fine-resolve gates this code used to
    // carry are exactly that computation done twice, by hand, per octave.
    let t_occ = sum_t / max(sum_c, 1.0e-4);
    let t_mid = 0.5 * (t0 + t1);
    let t_morph = mix(t_mid, t_occ, clamp(sum_c * 64.0, 0.0, 1.0));
    let p_morph = cam_pos + t_morph * ray_dir - planet_center;
    let n_morph = normalize(rotate_quat(cloud_params.world_to_body_orientation, p_morph));
    let px_morph = t_morph * pixel_angle;
    // Same per-place style the marcher forms cloud through, so a region that
    // rolls into streets up close still rolls into streets from orbit. The
    // occupancy guard mirrors `t_morph`'s: below it the response LUT renders
    // nothing, and a type divided by an epsilon would jump discontinuously.
    let type_morph = mix(0.45, sum_type / max(sum_c, 1.0e-4), clamp(sum_c * 64.0, 0.0, 1.0));
    let cell = cloud_cell_field(
        n_morph,
        planet_radius,
        px_morph,
        cloud_cell_style(n_morph, type_morph),
        cloud_params.cloud_time.x,
    );

    // One derived response for every footprint regime: the LUT stores the
    // expected near-column opacity for this strata mean, so the far tier's
    // rendered thickness equals the near volume's by construction. The old
    // resolved/unresolved split existed to compensate two hand-tuned curves
    // (`mean_c · 0.60` vs `smoothstep(0.06, 0.40, mean_c) · 0.95`); the
    // saturating resolved branch is what painted a moderate-strata field as a
    // near-solid veil at mid-altitude (fill 1.00 vs the near tier's 0.04 at
    // the 2026-07-24 A/B framing).
    //
    // Aggregation splits by chord length. A steep/short chord sits inside one
    // cell: one chord mean → one LUT response → one cell anchor is correct and
    // cheap. A grazing chord crosses many cells and columns; there the mean
    // erases every gap and the single anchor cannot restore them, so each
    // segment gets its own LUT response — modulated by the SAME cell field at
    // the segment's own position and footprint — and the segments composite
    // front-to-back like the distinct columns they are. The LUT remains the
    // sole per-column thickness authority (the cell term is still the
    // calibrated redistribution, per FAR_CELL_AMPLITUDE's contract); stacking
    // is damped by each segment's layer-overlap weight, mirroring the
    // occupancy weighting the mean path applies through `sample_weight`.
    var coverage_opacity: f32;
    if t1 - t0 > FAR_CHORD_STRUCTURE_MIN_M {
        var chord_trans = 1.0;
        for (var s = 0u; s < ORBITAL_MARCH_SAMPLES; s++) {
            if seg_profile[s] <= 1.0e-4 {
                continue;
            }
            let p_s = cam_pos + seg_t_m[s] * ray_dir - planet_center;
            let n_seg = normalize(rotate_quat(cloud_params.world_to_body_orientation, p_s));
            let px_seg = seg_t_m[s] * pixel_angle;
            let cell_s = cloud_cell_field(
                n_seg,
                planet_radius,
                px_seg,
                cloud_cell_style(n_seg, seg_type[s]),
                cloud_params.cloud_time.x,
            );
            let a_col = far_column_opacity(seg_profile[s], cell_s, px_seg);
            chord_trans *= 1.0 - min(a_col * seg_w[s], 0.95);
        }
        coverage_opacity = 1.0 - chord_trans;
    } else {
        coverage_opacity = far_column_opacity(mean_c, cell, px_morph);
    }
    var march_opacity = clamp(
        mix(stacked_opacity, coverage_opacity, clamp(cloud_params.cloud_march.w, 0.0, 1.0)),
        0.0,
        0.95,
    );
    // Near/far partition, applied to the CONVERGED OUTPUT opacity — weighting
    // the accumulation inputs instead pushes the weight through the response
    // LUT's nonlinear toe while the near tier fades linearly in alpha, so the
    // two halves stop summing to unity (the 2026-07-24 ascent "circle"). The
    // far-only diagnostic (tier_diagnostic = +1) bypasses ownership entirely.
    let ownership_active = cloud_params.cloud_march.y < 0.5;
    if ownership_active {
        // Near ownership is the union of "cells are resolvable" (whole-ray) and
        // "the near march still has probes here" (per-distance).
        var near_own = 1.0 - reach.z;
        if reach.y - reach.x > 1.0 {
            near_own *= 1.0 - smoothstep(reach.x, reach.y, t_morph);
        } else {
            near_own *= select(1.0, 0.0, t_morph > reach.y);
        }
        march_opacity *= 1.0 - near_own;
    }
    if march_opacity <= 1.0e-3 {
        return CloudOverlay(vec3<f32>(0.0), 0.0);
    }

    // Light the dominant sample through the shared soft column + height-moment
    // normal so this projection shades exactly like SolidPlanet's. The moment
    // normal's ~15 km relief detail is only meaningful while a weather texel
    // still spans several pixels; at long range it flickered per-texel through
    // the terminator (amber/white confetti at sunset), so it relaxes toward
    // the smooth sphere normal with distance.
    let cloud_alt = length(p_best) - planet_radius;
    let cloud_n_w = normalize(p_best);
    let cloud_n_l = rotate_quat(cloud_params.world_to_body_orientation, cloud_n_w);
    let column = weather_column_from_texel(
        sample_weather_soft(weather_texture, weather_sampler, cloud_n_l, lod_chord),
    );
    let n_body_lit_raw =
        orbital_cloud_normal_body(weather_texture, weather_sampler, cloud_n_l, lod_chord);
    // The moment normal is fetched at the footprint-matched mip, so its relief
    // is resolved-scale by construction; keep a stronger floor at range so the
    // disc shows cell relief instead of flat sphere shading (the veil look).
    let normal_detail = smoothstep(220000.0, 60000.0, best_t);
    let n_body_lit = normalize(mix(cloud_n_l, n_body_lit_raw, 0.40 + 0.60 * normal_detail));
    let q = cloud_params.world_to_body_orientation;
    let cloud_n_lit_w = rotate_quat(vec4(-q.xyz, q.w), n_body_lit);
    let n_dot_l = dot(cloud_n_lit_w, cloud_params.sun_dir_flux.xyz);
    let view_mu = max(dot(cloud_n_lit_w, -ray_dir), 0.0);
    let shade = orbital_cloud_shade(column, n_dot_l, view_mu);
    // Day/night and warm-hour chroma follow the GEOMETRIC solar elevation at
    // the sample (radial up · sun). Driving them from the relief-perturbed
    // shading normal painted midday cells that merely tilt away from the sun
    // in sunset orange — the pink limb/horizon fringes.
    let sun_elev_geo = dot(cloud_n_w, cloud_params.sun_dir_flux.xyz);
    let night = smoothstep(-0.12, 0.08, sun_elev_geo);
    // Warm hour is a narrow band near the terminator (< ~9° solar elevation).
    // Completing the transition at 0.45 tinted two-thirds of the lit disc
    // cream — daylit clouds must read white. The day endpoint is NEUTRAL: the
    // former (1.0, 0.97, 0.92) baked a permanent warm bias into every daylit
    // far cloud, which against the near tier's blue-sky-ambient white read as
    // a beige smear across the whole distant band (user, 2026-07-29). Daytime
    // warmth, where wanted, belongs to the warm-hour ramp — not the endpoint.
    let sun_chroma = mix(
        vec3<f32>(1.0, 0.45, 0.18),
        vec3<f32>(1.0, 1.0, 1.0),
        smoothstep(0.02, 0.16, clamp(sun_elev_geo, 0.0, 1.0)),
    );
    // Occupancy comes from the band march; `shade.y` would double-count it.
    let opacity = march_opacity * night;
    // Far-tier radiance prefactor, DERIVED against the near tier's photometric
    // anchor instead of eyeballed at the handoff.
    //
    // The near march's diffusion reservoir (`CLOUD_MS_ALBEDO` in
    // clouds_compute.wgsl) puts a fully lit, optically thick cell at
    // `A · flux · SCENE_FLUX_SCALE · SURFACE_DIRECT_SCALE` — exactly a
    // Lambertian surface of albedo A facing the same sun, which is the anchor
    // every spine surface already uses. This overlay must land on the same
    // number for the same cell, so with
    // `radiance = tint.g · flux · SCENE_FLUX_SCALE · K · shade`:
    //     K = A · SURFACE_DIRECT_SCALE / (tint.g · FAR_SHADE_LIT)
    // The former 0.55 → 0.68 pair were hand-matched at the handoff while the
    // near tier rendered single-scatter-only radiance (~4× too dark), so they
    // carried that error as compensating brightness. Do not re-tune this by
    // eye against the near tier; re-derive it if either side's photometry
    // moves.
    let far_radiance_k =
        water_cloud_albedo() * SURFACE_DIRECT_SCALE / (FAR_CLOUD_TINT.g * FAR_SHADE_LIT);
    var radiance = cloud_atmosphere.cloud_albedo_coverage.rgb
        * FAR_CLOUD_TINT
        * sun_chroma
        * cloud_params.sun_dir_flux.w
        * SCENE_FLUX_SCALE
        * far_radiance_k
        * shade.x
        * night;
    // Sky-ambient in-scatter — the same SkyAmbient radiance `drive_clouds`
    // feeds the near march's every sample. The near tier's thick lit cell
    // converges to (ambient + direct); the anchor above supplies only the
    // direct half, which is why this tier alone rendered warm-gray against
    // near white-blue clouds (measured 2026-07-29: neither chroma nor
    // airlight could account for the band's color). Tops-dominant view
    // factor: what this overlay shows of a cell is its sunward/skyward face;
    // undersides are already carried by `shade`. Like the near tier's, the
    // ambient is NOT albedo-tinted (chroma lives in the sun/sky colors).
    let amb_far = mix(
        cloud_params.cloud_ambient_bottom.rgb,
        cloud_params.cloud_ambient_top.rgb,
        0.75,
    ) * 0.85 * night;
    radiance = radiance + amb_far;

    // Foreground airlight: this overlay draws OVER the already-integrated sky,
    // so its opacity would otherwise delete the air in front of the cloud —
    // distant decks then read as extinction-only brown. Approximate the veil
    // with the same analytic β the near march uses and restore it here.
    let dens = exp(-max(cloud_alt, 0.0) / 8000.0);
    // Equivalent air path via a Schueler-style air mass: one scale height over
    // view inclination. The former flat 60 km/mu cap modelled a horizontal
    // in-atmosphere path; applied to an orbital top-down ray it attenuated
    // ~74 % of the blue and turned the whole disc's clouds beige.
    let mu_view = abs(dot(cloud_n_w, ray_dir));
    let d_air = min(best_t, 8000.0 / max(mu_view + 0.10, 0.15));
    let tau = vec3<f32>(5.5e-6, 1.3e-5, 3.2e-5) * dens * d_air;
    let view_T = exp(-tau);
    let day = smoothstep(-0.05, 0.30, sun_elev_geo);
    // NOTE (2026-07-29, measured): at this framing the whole airlight term
    // contributes < 1/255 to the distant band at any plausible gain — a 100×
    // red probe moved the band by only +33/255. The band's remaining warm
    // read vs the neighbouring haze is a MISSING SKY-AMBIENT term in this
    // tier's shading (the near march adds `clouds_ambient_color_*` to every
    // sample; this overlay is direct-sun only) — plumb the ambient in and
    // re-derive the anchor rather than re-tuning this constant.
    let e_air = vec3<f32>(0.40, 0.55, 0.85)
        * cloud_params.sun_dir_flux.w
        * SCENE_FLUX_SCALE
        * 0.12
        * day;
    radiance = radiance * view_T + e_air * (vec3<f32>(1.0) - view_T);

    return CloudOverlay(radiance * opacity, opacity);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Only the body currently projected by the one ship-view cloud compute
    // pass enables this material. The guard also prevents out-of-bounds reads
    // from the inactive 1x1 fallback textures.
    if cloud_params.cloud_band_radii.w < 0.5 {
        discard;
    }

    let ray_dir = reconstruct_ray(in.clip_position.xy);
    let cam_pos = view.world_position;
    let planet_center = cloud_params.planet_center_radius.xyz;
    let planet_radius = cloud_params.planet_center_radius.w;
    let oc = cam_pos - planet_center;
    let oc_len_sq = dot(oc, oc);
    let b = dot(oc, ray_dir);
    let scene_t = scene_distance(in.clip_position.xy);

    let near_sample = sample_near_cloud(in.clip_position.xy);
    let dims = textureDimensions(cloud_layer_texture);
    let uv = in.clip_position.xy / view.viewport.zw;
    let ref_coord = clamp(
        vec2<i32>(uv * vec2<f32>(dims)),
        vec2<i32>(0),
        vec2<i32>(dims) - vec2<i32>(1),
    );
    let cloud_span = textureLoad(cloud_distance_texture, ref_coord, 0).rg;
    let near_vis = near_visibility(
        cloud_span.x,
        cloud_span.y,
        near_sample.a,
        scene_t,
        oc_len_sq,
        b,
    );
    let tier_diagnostic = cloud_params.cloud_march.y;
    let near_enabled = select(1.0, 0.0, tier_diagnostic > 0.5);
    let far_enabled = select(1.0, 0.0, tier_diagnostic < -0.5);
    let near_cloud = CloudOverlay(
        near_sample.rgb * near_vis * near_enabled,
        (1.0 - near_sample.a) * near_vis * near_enabled,
    );
    let orbital = sample_orbital_cloud(
        cam_pos, ray_dir, planet_center, planet_radius, oc_len_sq, b, scene_t,
    );
    let cloud = CloudOverlay(
        near_cloud.premul_rgb + orbital.premul_rgb * far_enabled * (1.0 - near_cloud.opacity),
        1.0 - (1.0 - near_cloud.opacity) * (1.0 - orbital.opacity * far_enabled),
    );
    if cloud.opacity <= 1.0e-5 {
        discard;
    }
    return vec4(cloud.premul_rgb, clamp(cloud.opacity, 0.0, 1.0));
}
