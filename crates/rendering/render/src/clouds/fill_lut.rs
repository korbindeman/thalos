//! CPU-derived shared cloud fill/opacity response (BL-20260723T214730Z).
//!
//! The near volumetric tier and the far/orbital projection must agree on how
//! much sky a given weather column fills. The recorded contract (cloud doc,
//! round 3) is that the near tier's areal fill EQUALS the strata density the
//! far tier reads directly — but the near tier realizes fill through a
//! quantile threshold on its periodic 3-D shape noise, whose exceedance
//! distribution is far too steep to hand-tune (every previous constant pair
//! was fitted against captures, and the last one against a corrupted weather
//! cube — see INC-20260723T221126Z).
//!
//! This module *derives* the pairing instead, at body spawn, from the actual
//! authored weather field:
//!
//! 1. A faithful CPU mirror of the marcher's density math (the same 64³
//!    tileable noise volume, domain transforms, thresholds, profiles, and
//!    erosion as `clouds_compute.wgsl` — keep them in lockstep) Monte-Carlo
//!    marches vertical columns through texels sampled from the real cube.
//! 2. The near threshold curve `T(env) = mix(lo, hi, env)` is fitted so the
//!    simulated column fill tracks the far tier's own conditioning variable
//!    (the profile-weighted strata mean) — i.e. fill ≈ strata density where
//!    the envelope makes that reachable.
//! 3. With the fitted curve, the expected column opacity `E[1 − T_column]`
//!    per strata-mean bin is emitted as a 16-entry LUT. The far tier renders
//!    that LUT directly, so far opacity equals the near tier's expected
//!    result BY CONSTRUCTION — one derived response, not two hand-tuned
//!    curves.
//!
//! Statistical fidelity is what matters here: the mirror reproduces the
//! shader's *distributions* (same algorithms, same constants), so the derived
//! quantiles transfer even though individual GPU texels are never read back.

use bevy::math::{IVec3, Vec3, Vec4};
use rayon::prelude::*;
use thalos_terrain::cubemap::{CubemapFace, face_uv_to_dir};

/// Entries in the far-tier opacity response LUT (packed as 4 × vec4 in the
/// composite uniform). Entry `i` is the response at strata mean `i / 15`.
pub const FILL_RESPONSE_ENTRIES: usize = 16;

/// Nodes in the near-tier formation-threshold curve `T(env)` (piecewise
/// linear over env node `i / 7`). A 2-parameter linear curve cannot track the
/// fill contract: the shape noise's exceedance distribution is nearly a cliff,
/// so with only two knobs the count-heavy deep-deck bins pull moderate bins
/// into a 2–3× fill overshoot (measured in the first derivation run).
pub const THRESHOLD_NODES: usize = 8;

/// Version of the derivation algorithm below. **Bump this whenever anything
/// that affects `derive_fill_calibration`'s output changes** (the march
/// cadence, shape/erosion mirrors, threshold fit, column count…): it is part
/// of the disk-cache key the game uses to skip the multi-second Monte-Carlo
/// at boot, and a stale hit silently calibrates yesterday's renderer —
/// exactly the `GENERATOR_VERSION` rule the terrain tile cache follows.
pub const FILL_LUT_VERSION: u32 = 1;

/// Derived near-threshold curve + far opacity response for one body/climate.
#[derive(Clone, Copy, Debug)]
pub struct CloudFillCalibration {
    /// Near-tier formation threshold vs strata density (node `i` at `i / 7`,
    /// non-increasing: denser strata form at lower thresholds).
    pub threshold_nodes: [f32; THRESHOLD_NODES],
    /// Expected near-column opacity per strata-mean node (`i / 15`).
    pub far_response: [f32; FILL_RESPONSE_ENTRIES],
    /// Expected per-sample filtered density `E[shaped | env]` (node `i / 15`)
    /// — the band-limited stand-in for the marcher's Cartesian shape term.
    /// The long-reach march's coarse bands render this instead of the noise
    /// volume (BL-20260724T003705Z), so the homogenized field is
    /// mean-preserving by construction.
    pub shape_response: [f32; FILL_RESPONSE_ENTRIES],
}

impl CloudFillCalibration {
    /// Pack the response for the composite uniform (`fill_response` field).
    pub fn far_response_vec4s(&self) -> [Vec4; 4] {
        let r = &self.far_response;
        [
            Vec4::new(r[0], r[1], r[2], r[3]),
            Vec4::new(r[4], r[5], r[6], r[7]),
            Vec4::new(r[8], r[9], r[10], r[11]),
            Vec4::new(r[12], r[13], r[14], r[15]),
        ]
    }

    /// Pack the threshold curve for the compute uniform.
    pub fn threshold_vec4s(&self) -> [Vec4; 2] {
        let t = &self.threshold_nodes;
        [
            Vec4::new(t[0], t[1], t[2], t[3]),
            Vec4::new(t[4], t[5], t[6], t[7]),
        ]
    }

    /// Pack the homogenized-field response for the compute uniform.
    pub fn shape_response_vec4s(&self) -> [Vec4; 4] {
        let r = &self.shape_response;
        [
            Vec4::new(r[0], r[1], r[2], r[3]),
            Vec4::new(r[4], r[5], r[6], r[7]),
            Vec4::new(r[8], r[9], r[10], r[11]),
            Vec4::new(r[12], r[13], r[14], r[15]),
        ]
    }
}

/// Piecewise-linear threshold evaluation (the CPU twin of the shader's
/// `formation_threshold`).
fn threshold_at(nodes: &[f32; THRESHOLD_NODES], env: f32) -> f32 {
    let t = env.clamp(0.0, 1.0) * (THRESHOLD_NODES - 1) as f32;
    let i = (t as usize).min(THRESHOLD_NODES - 2);
    let f = t - i as f32;
    nodes[i] + (nodes[i + 1] - nodes[i]) * f
}

/// Inputs mirroring exactly what the marcher sees at runtime. Values must be
/// the PRODUCTION ones (`drive_clouds` / `init_cloud_appearance`), not the
/// crate defaults, or the derived response calibrates a different renderer.
pub struct FillCalibrationInput<'a> {
    /// RGBA8 weather texels, face-major in `CubemapFace::ALL` order.
    pub weather_texels: &'a [[u8; 4]],
    /// RGBA8 four-stratum surface-density texels, same layout.
    pub strata_texels: &'a [[u8; 4]],
    pub face_size: u32,
    /// `CloudsConfig::clouds_coverage` (global coverage scale).
    pub coverage_scale: f32,
    /// `CloudsConfig::clouds_density` (extinction multiplier, 1/m).
    pub density: f32,
    pub detail_strength: f32,
    pub base_edge_softness: f32,
    pub bottom_softness: f32,
    pub base_shape_scale_m: f32,
    pub detail_scale_m: f32,
    /// Shell base/top altitudes above the surface, metres.
    pub bottom_height_m: f32,
    pub top_height_m: f32,
    pub planet_radius_m: f32,
    /// Monte-Carlo seed; fixed by the caller for deterministic boots.
    pub seed: u64,
}

/// One pre-marched density sample: everything `column_alpha` needs to
/// re-evaluate the marcher's density under a candidate threshold curve
/// without touching the noise volume again.
#[derive(Clone, Copy)]
struct SampleRecord {
    /// Broad shape with the per-column macro variety term already folded in
    /// (the marcher adds it to the threshold; folding keeps `column_alpha` a
    /// pure function of the candidate curve).
    shape: f32,
    /// Anvil shape base (`0.72·broad.r + 0.28·broad.a`), macro term folded.
    anvil_base: f32,
    env: f32,
    envelope: f32,
    /// Height-typed erosion factor: `mix(detail.b, 1 − detail.b,
    /// smoothstep(0.10, 0.32, h)) · (0.80 + 0.55 · h)`, pre-folded because
    /// `h` is fixed per sample (mirrors the marcher's erosion character).
    erode: f32,
    profile: f32,
    vertical_narrow: f32,
    /// `anvil_profile * storm_w` (0 outside storm columns).
    anvil_gate: f32,
}

struct ColumnRecord {
    /// The far tier's conditioning variable for this column: the
    /// profile-weighted strata mean over the full layer (the vertical-ray
    /// analogue of the composite's per-segment quadrature).
    strata_mean: f32,
    samples: Vec<SampleRecord>,
}

/// March step along the column, mirroring the marcher's full-density cadence.
const STEP_M: f32 = 120.0;
/// A column "fills" its footprint when its integrated opacity is visible —
/// aligned with the capture-side fill measurement threshold (~12/255).
const VISIBLE_ALPHA: f32 = 0.05;
const COLUMNS: usize = 16_384;

pub fn derive_fill_calibration(input: &FillCalibrationInput<'_>) -> CloudFillCalibration {
    let volume = NoiseVolume::generate();
    let columns = sample_columns(input, &volume);

    // Stage 1 — linear init: fit T(env) = mix(lo, hi, env) on a coarse grid.
    // Loss is the weighted squared gap between per-bin fill and the bin's own
    // strata mean (the identity contract).
    let mut best: (f32, f32, f32) = (f32::INFINITY, 0.81, 0.44);
    for &(lo, hi) in &grid_pairs(0.45, 0.98, 0.10, 0.98, 0.02) {
        let mut nodes = [0.0; THRESHOLD_NODES];
        for (i, node) in nodes.iter_mut().enumerate() {
            *node = lo + (hi - lo) * i as f32 / (THRESHOLD_NODES - 1) as f32;
        }
        let loss = threshold_loss(&columns, input, &nodes);
        if loss < best.0 {
            best = (loss, lo, hi);
        }
    }
    let mut nodes = [0.0; THRESHOLD_NODES];
    for (i, node) in nodes.iter_mut().enumerate() {
        *node = best.1 + (best.2 - best.1) * i as f32 / (THRESHOLD_NODES - 1) as f32;
    }

    // Stage 2 — coordinate descent with monotonicity BY CONSTRUCTION:
    // parameters are the top-end (env = 1) threshold plus 7 non-negative
    // upward deltas toward env = 0 (`node[i] = node[i+1] + delta[i]`). A
    // clamp-after-move formulation silently blocked every threshold-raising
    // move (the raised node snapped back to its predecessor), which froze the
    // fit at the linear init and left moderate bins overshooting ~2×.
    let mut top = nodes[THRESHOLD_NODES - 1];
    let mut deltas = [0.0f32; THRESHOLD_NODES - 1];
    for i in 0..THRESHOLD_NODES - 1 {
        deltas[i] = (nodes[i] - nodes[i + 1]).max(0.0);
    }
    let assemble = |top: f32, deltas: &[f32; THRESHOLD_NODES - 1]| {
        let mut nodes = [0.0f32; THRESHOLD_NODES];
        nodes[THRESHOLD_NODES - 1] = top;
        for i in (0..THRESHOLD_NODES - 1).rev() {
            nodes[i] = nodes[i + 1] + deltas[i];
        }
        nodes
    };
    let mut best_loss = threshold_loss(&columns, input, &nodes);
    for &step in &[0.06f32, 0.025, 0.01, 0.004] {
        loop {
            let mut improved = false;
            for parameter in 0..THRESHOLD_NODES {
                for direction in [-1.0f32, 1.0] {
                    let (candidate_top, mut candidate_deltas) = (top, deltas);
                    if parameter == 0 {
                        let moved = (candidate_top + direction * step).clamp(0.05, 1.2);
                        let loss =
                            threshold_loss(&columns, input, &assemble(moved, &candidate_deltas));
                        if loss < best_loss {
                            best_loss = loss;
                            top = moved;
                            nodes = assemble(top, &deltas);
                            improved = true;
                        }
                        continue;
                    }
                    let index = parameter - 1;
                    candidate_deltas[index] =
                        (candidate_deltas[index] + direction * step).max(0.0);
                    let loss = threshold_loss(
                        &columns,
                        input,
                        &assemble(candidate_top, &candidate_deltas),
                    );
                    if loss < best_loss {
                        best_loss = loss;
                        deltas = candidate_deltas;
                        nodes = assemble(top, &deltas);
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }

    // Final pass: expected column opacity per strata-mean node with the
    // fitted curve — the far tier's response.
    let mut node_sum = [0.0f64; FILL_RESPONSE_ENTRIES];
    let mut node_weight = [0.0f64; FILL_RESPONSE_ENTRIES];
    for column in &columns {
        let alpha = column_alpha(column, input, &nodes);
        let t = column.strata_mean.clamp(0.0, 1.0) * (FILL_RESPONSE_ENTRIES - 1) as f32;
        let i = (t as usize).min(FILL_RESPONSE_ENTRIES - 2);
        let f = f64::from(t - i as f32);
        node_sum[i] += f64::from(alpha) * (1.0 - f);
        node_weight[i] += 1.0 - f;
        node_sum[i + 1] += f64::from(alpha) * f;
        node_weight[i + 1] += f;
    }
    let mut far_response = [f32::NAN; FILL_RESPONSE_ENTRIES];
    far_response[0] = 0.0;
    for i in 1..FILL_RESPONSE_ENTRIES {
        if node_weight[i] >= 8.0 {
            far_response[i] = (node_sum[i] / node_weight[i]) as f32;
        }
    }
    fill_gaps_monotone(&mut far_response);

    // Homogenized-field response: expected PER-SAMPLE filtered density vs the
    // sample's own strata value, with the fitted curve. The long-reach
    // march's coarse bands render this in place of the Cartesian shape term.
    let detail_w = detail_weight(input);
    let edge_soft = input.base_edge_softness.max(0.015);
    let mut shape_sum = [0.0f64; FILL_RESPONSE_ENTRIES];
    let mut shape_weight = [0.0f64; FILL_RESPONSE_ENTRIES];
    for column in &columns {
        for s in &column.samples {
            let shaped = sample_shaped(s, input, &nodes, detail_w, edge_soft);
            let t = s.env.clamp(0.0, 1.0) * (FILL_RESPONSE_ENTRIES - 1) as f32;
            let i = (t as usize).min(FILL_RESPONSE_ENTRIES - 2);
            let f = f64::from(t - i as f32);
            shape_sum[i] += f64::from(shaped) * (1.0 - f);
            shape_weight[i] += 1.0 - f;
            shape_sum[i + 1] += f64::from(shaped) * f;
            shape_weight[i + 1] += f;
        }
    }
    let mut shape_response = [f32::NAN; FILL_RESPONSE_ENTRIES];
    shape_response[0] = 0.0;
    for i in 1..FILL_RESPONSE_ENTRIES {
        if shape_weight[i] >= 32.0 {
            shape_response[i] = (shape_sum[i] / shape_weight[i]) as f32;
        }
    }
    fill_gaps_monotone(&mut shape_response);

    // Convergence record: per strata-mean bin, the identity target vs the
    // fill the fitted curve achieves (and the opacity the LUT will render).
    // Low bins are envelope-limited and cannot reach their target; that is
    // expected — parity between the tiers holds regardless, because the far
    // tier renders the ACHIEVED response.
    let mut table = vec![(0.0f32, 0.0f32, 0.0f32, 0u32); 16];
    for column in &columns {
        let bin = ((column.strata_mean * 16.0) as usize).min(15);
        let alpha = column_alpha(column, input, &nodes);
        let entry = &mut table[bin];
        entry.0 += column.strata_mean;
        entry.1 += if alpha > VISIBLE_ALPHA { 1.0 } else { 0.0 };
        entry.2 += alpha;
        entry.3 += 1;
    }
    for entry in &mut table {
        let n = (entry.3 as f32).max(1.0);
        entry.0 /= n;
        entry.1 /= n;
        entry.2 /= n;
    }
    bevy::log::info!(
        target: "thalos::clouds",
        ?nodes,
        "fill calibration fit: per-bin (strata target, achieved fill, mean opacity, n) = {table:?}"
    );

    CloudFillCalibration {
        threshold_nodes: nodes,
        far_response,
        shape_response,
    }
}

/// Region-restricted prediction of the near tier's rendered fill/opacity —
/// the dev-probe cross-check against a pixel-measured capture at the same
/// site (mirror validation; not used at runtime).
#[derive(Clone, Copy, Debug)]
pub struct RegionFillStats {
    pub fill: f32,
    pub mean_alpha: f32,
    pub mean_strata: f32,
    pub columns: usize,
}

pub fn predict_region_fill(
    input: &FillCalibrationInput<'_>,
    calibration: &CloudFillCalibration,
    center_dir: Vec3,
    cos_radius: f32,
    column_target: usize,
) -> RegionFillStats {
    let volume = NoiseVolume::generate();
    let size = input.face_size as usize;
    let mut rng = Rng::new(input.seed ^ 0xD1CE_CAFE);
    let mut columns = Vec::new();
    let mut attempts = 0usize;
    while columns.len() < column_target && attempts < column_target * 4000 {
        attempts += 1;
        let face_index = (rng.next_f32() * 6.0) as usize % 6;
        let x = 1.0 + rng.next_f32() * (size as f32 - 2.0);
        let y = 1.0 + rng.next_f32() * (size as f32 - 2.0);
        let dir = face_uv_to_dir(CubemapFace::ALL[face_index], x / size as f32, y / size as f32)
            .normalize();
        if dir.dot(center_dir) < cos_radius {
            continue;
        }
        let weather_v = bilinear_texel(input.weather_texels, size, face_index, x, y);
        let strata = bilinear_texel(input.strata_texels, size, face_index, x, y);
        let weather = [weather_v.x, weather_v.y, weather_v.z, weather_v.w];
        let strata_mean = 0.25 * surface_shape(strata, 0.0)
            + 0.50 * surface_shape(strata, 0.5)
            + 0.25 * surface_shape(strata, 1.0);
        columns.push(march_column(
            input,
            &volume,
            dir,
            weather,
            strata,
            strata_mean,
            &mut rng,
        ));
    }
    let mut fill = 0.0;
    let mut alpha_sum = 0.0;
    let mut strata_sum = 0.0;
    for column in &columns {
        let alpha = column_alpha(column, input, &calibration.threshold_nodes);
        fill += if alpha > VISIBLE_ALPHA { 1.0 } else { 0.0 };
        alpha_sum += alpha;
        strata_sum += column.strata_mean;
    }
    let n = (columns.len() as f32).max(1.0);
    RegionFillStats {
        fill: fill / n,
        mean_alpha: alpha_sum / n,
        mean_strata: strata_sum / n,
        columns: columns.len(),
    }
}

fn grid_pairs(lo_min: f32, lo_max: f32, hi_min: f32, hi_max: f32, step: f32) -> Vec<(f32, f32)> {
    let mut pairs = Vec::new();
    let mut lo = lo_min;
    while lo <= lo_max + 1.0e-6 {
        let mut hi = hi_min;
        while hi <= hi_max.min(lo) + 1.0e-6 {
            pairs.push((lo, hi));
            hi += step;
        }
        lo += step;
    }
    pairs
}

fn threshold_loss(
    columns: &[ColumnRecord],
    input: &FillCalibrationInput<'_>,
    nodes: &[f32; THRESHOLD_NODES],
) -> f32 {
    const BINS: usize = 16;
    let (fill, target, count) = columns
        .par_chunks(2048)
        .map(|chunk| {
            let mut fill = [0.0f64; BINS];
            let mut target = [0.0f64; BINS];
            let mut count = [0.0f64; BINS];
            for column in chunk {
                let bin = ((column.strata_mean * BINS as f32) as usize).min(BINS - 1);
                let alpha = column_alpha(column, input, nodes);
                fill[bin] += if alpha > VISIBLE_ALPHA { 1.0 } else { 0.0 };
                target[bin] += f64::from(column.strata_mean);
                count[bin] += 1.0;
            }
            (fill, target, count)
        })
        .reduce(
            || ([0.0; BINS], [0.0; BINS], [0.0; BINS]),
            |mut a, b| {
                for i in 0..BINS {
                    a.0[i] += b.0[i];
                    a.1[i] += b.1[i];
                    a.2[i] += b.2[i];
                }
                a
            },
        );
    let mut loss = 0.0f64;
    for b in 0..BINS {
        if count[b] < 16.0 {
            continue;
        }
        let gap = (fill[b] - target[b]) / count[b];
        // sqrt-count weighting: the strata distribution has a huge deep-deck
        // spike; raw count weighting let that one bin sacrifice every
        // moderate (visually load-bearing, broken-field) bin.
        loss += count[b].sqrt() * gap * gap;
    }
    loss as f32
}

/// One recorded sample's filtered density factor (`shaped`, pre profile ×
/// envelope × extinction) under a candidate threshold curve. Mirrors
/// `get_cloud_map_density`'s post-noise math — the same function feeds the
/// column-opacity re-evaluation AND the homogenized-field expectation.
fn sample_shaped(
    s: &SampleRecord,
    input: &FillCalibrationInput<'_>,
    nodes: &[f32; THRESHOLD_NODES],
    detail_weight: f32,
    edge_softness: f32,
) -> f32 {
    let threshold = threshold_at(nodes, s.env);
    let mut mass = s.shape - threshold - s.vertical_narrow;
    if s.anvil_gate > 0.0 {
        let anvil_shape = s.anvil_base - (threshold - 0.06);
        mass = mass.max(anvil_shape * s.anvil_gate);
    }
    let edge = 1.0 - smoothstep(0.02, 0.34, mass);
    if edge * detail_weight > 1.0e-3 {
        mass -= s.erode * edge * detail_weight * input.detail_strength * 0.55;
    }
    smoothstep(0.0, edge_softness, mass)
}

/// Re-evaluate one recorded column's integrated opacity under a candidate
/// threshold curve. Mirrors `get_cloud_map_density`'s post-noise math.
fn column_alpha(
    column: &ColumnRecord,
    input: &FillCalibrationInput<'_>,
    nodes: &[f32; THRESHOLD_NODES],
) -> f32 {
    let detail_weight = detail_weight(input);
    let edge_softness = input.base_edge_softness.max(0.015);
    let mut optical_depth = 0.0f32;
    for s in &column.samples {
        let shaped = sample_shaped(s, input, nodes, detail_weight, edge_softness);
        let density = (shaped * s.profile * s.envelope * input.density).max(0.0);
        optical_depth += density * STEP_M;
    }
    1.0 - (-optical_depth).exp()
}

fn detail_weight(input: &FillCalibrationInput<'_>) -> f32 {
    let detail_feature_m = input.detail_scale_m.max(50.0);
    1.0 - smoothstep(detail_feature_m * 0.25, detail_feature_m * 0.50, STEP_M)
}

/// Bilinear fetch on one cube face (RGBA8 → [0, 1] f32s). `x`/`y` are
/// continuous texel-centre coordinates. Face borders are not wrapped — the
/// samplers below keep positions ≥ 1 texel inside the face (a statistically
/// negligible exclusion at 1024² faces), because the runtime's seamless
/// cube filtering has no cheap CPU mirror.
fn bilinear_texel(texels: &[[u8; 4]], size: usize, face: usize, x: f32, y: f32) -> Vec4 {
    let xf = (x - 0.5).clamp(0.0, (size - 2) as f32);
    let yf = (y - 0.5).clamp(0.0, (size - 2) as f32);
    let x0 = xf as usize;
    let y0 = yf as usize;
    let fx = xf - x0 as f32;
    let fy = yf - y0 as f32;
    let base = face * size * size;
    let fetch = |xi: usize, yi: usize| {
        let t = texels[base + yi * size + xi];
        Vec4::new(
            f32::from(t[0]),
            f32::from(t[1]),
            f32::from(t[2]),
            f32::from(t[3]),
        ) / 255.0
    };
    let top = fetch(x0, y0).lerp(fetch(x0 + 1, y0), fx);
    let bottom = fetch(x0, y0 + 1).lerp(fetch(x0 + 1, y0 + 1), fx);
    top.lerp(bottom, fy)
}

fn sample_columns(input: &FillCalibrationInput<'_>, volume: &NoiseVolume) -> Vec<ColumnRecord> {
    let size = input.face_size as usize;
    let face_texels = size * size;
    assert_eq!(input.weather_texels.len(), 6 * face_texels);
    assert_eq!(input.strata_texels.len(), 6 * face_texels);
    (0..COLUMNS)
        .into_par_iter()
        .filter_map(|i| {
            let mut rng = Rng::new(input.seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let face_index = (rng.next_f32() * 6.0) as usize % 6;
            // Continuous position: the LUT must be conditioned on the same
            // BILINEAR field the far shader samples spatially. Deriving on
            // point texels (mostly binary cells) mismatched the two
            // distributions — the shader fed kilometres of inter-cell ramp
            // values into a response learned only at texel values, which lit
            // the whole ramp skirt and kept the far tier ~3× the near area.
            let x = 1.0 + rng.next_f32() * (size as f32 - 2.0);
            let y = 1.0 + rng.next_f32() * (size as f32 - 2.0);
            let weather_v = bilinear_texel(input.weather_texels, size, face_index, x, y);
            let strata = bilinear_texel(input.strata_texels, size, face_index, x, y);
            let weather = [weather_v.x, weather_v.y, weather_v.z, weather_v.w];
            // The far conditioning variable: the composite's 1/4-1/2-1/4
            // quadrature over the full (vertical-ray) layer span.
            let strata_mean = 0.25 * surface_shape(strata, 0.0)
                + 0.50 * surface_shape(strata, 0.5)
                + 0.25 * surface_shape(strata, 1.0);
            // Clear columns pin the LUT's zero node analytically; skip the
            // march (their alpha is 0 by construction).
            if strata.max_element() < 0.008 {
                return None;
            }
            let dir =
                face_uv_to_dir(CubemapFace::ALL[face_index], x / size as f32, y / size as f32)
                    .normalize();
            Some(march_column(input, volume, dir, weather, strata, strata_mean, &mut rng))
        })
        .collect()
}

fn march_column(
    input: &FillCalibrationInput<'_>,
    volume: &NoiseVolume,
    dir: Vec3,
    weather: [f32; 4],
    strata: Vec4,
    strata_mean: f32,
    rng: &mut Rng,
) -> ColumnRecord {
    let cov = (weather[0] * input.coverage_scale).clamp(0.0, 1.0);
    let local_base = weather[2].clamp(0.0, 0.92);
    let local_top = weather[3].clamp(0.02, 1.0).max(local_base + 0.02);
    let shell_thickness = (input.top_height_m - input.bottom_height_m).max(1.0);
    let mut samples = Vec::new();
    if cov > 1.0e-3 {
        // Type weights and the shape spectrum are per column (weather is one
        // texel here, as it is per-ray-context in the marcher).
        let stratus_w = 1.0 - smoothstep(0.18, 0.38, weather[1]);
        let storm_w = smoothstep(0.72, 0.88, weather[1]);
        let cumulus_w = (1.0 - stratus_w - storm_w).max(0.0);
        let column_tall = smoothstep(0.30, 0.65, local_top - local_base);
        let shape_scale = input.base_shape_scale_m.max(500.0);
        let macro_period = shape_scale * 2.7;
        let bottom_softness = input.bottom_softness.max(0.01);

        // Per-column macro context, as the marcher samples it once per ray.
        let mid_alt =
            input.bottom_height_m + 0.5 * (local_base + local_top) * shell_thickness;
        let mid_pos = dir * (input.planet_radius_m + mid_alt);
        let macro_sample = volume.trilinear(
            mid_pos + phase_offset(&weather, macro_period) + Vec3::new(-7300.0, 2100.0, 4900.0),
            macro_period,
        );
        let macro_term = (0.5 - macro_sample.w) * 0.05;

        let alt_lo = input.bottom_height_m + local_base * shell_thickness;
        let alt_hi = input.bottom_height_m + local_top * shell_thickness;
        let mut alt = alt_lo + rng.next_f32() * STEP_M;
        while alt < alt_hi {
            let shell_h = (alt - input.bottom_height_m) / shell_thickness;
            // Strata layer height uses the RAW base/top channels, exactly as
            // the marcher's step loop does before `get_cloud_map_density`
            // re-clamps for its own vertical response.
            let layer_h_raw = (shell_h - weather[2]) / (weather[3] - weather[2]).max(0.02);
            let env = surface_shape_gated(strata, layer_h_raw);
            let h = (shell_h - local_base) / (local_top - local_base);
            if h <= 0.0 || h >= 1.0 {
                alt += STEP_M;
                continue;
            }

            let pos = dir * (input.planet_radius_m + alt);
            let broad = volume.trilinear(
                rotated_domain(pos)
                    + phase_offset(&weather, shape_scale)
                    + Vec3::new(1800.0, -4200.0, 900.0),
                shape_scale,
            );
            let shape_squat = broad.x * 0.52 + broad.y * 0.24 + broad.w * 0.24;
            let shape_tall = broad.x * 0.64 + broad.y * 0.06 + broad.w * 0.30;
            let shape = shape_squat + (shape_tall - shape_squat) * column_tall;
            let detail = volume.trilinear(
                rotated_domain(pos) + Vec3::new(270.0, -610.0, 130.0),
                input.detail_scale_m.max(50.0) * 8.0,
            );

            // Round-7 dome sculpting + thin top skins + height-typed erosion:
            // exact mirrors of `get_cloud_map_density` — keep in lockstep.
            let vertical_narrow = h * 0.04 * stratus_w
                + (h * h)
                    * (0.42 * cumulus_w + 0.30 * storm_w)
                    * (1.0 - 0.45 * column_tall);
            let anvil_profile =
                smoothstep(0.62, 0.76, h) * (1.0 - smoothstep(0.90, 1.0, h));
            let stratus_profile = smoothstep(0.0, bottom_softness * 0.45, h)
                * (1.0 - smoothstep(0.72, 1.0, h));
            let cumulus_profile = smoothstep(0.0, bottom_softness * 0.75, h)
                * (1.0 - smoothstep(0.93, 1.0, h));
            let storm_profile = smoothstep(0.0, bottom_softness * 0.35, h)
                * (1.0 - smoothstep(0.94, 1.0, h));
            let erode_flip = smoothstep(0.10, 0.32, h);
            let erode = (detail.z + (1.0 - 2.0 * detail.z) * erode_flip)
                * (0.80 + 0.55 * h);
            let profile = stratus_profile * stratus_w
                + cumulus_profile * cumulus_w
                + storm_profile * storm_w;

            samples.push(SampleRecord {
                shape: shape - macro_term,
                anvil_base: broad.x * 0.72 + broad.w * 0.28 - macro_term,
                env,
                envelope: smoothstep(0.04, 0.42, env),
                erode,
                profile,
                vertical_narrow,
                anvil_gate: anvil_profile * storm_w,
            });
            alt += STEP_M;
        }
    }
    ColumnRecord {
        strata_mean,
        samples,
    }
}

/// `cloud_surface_shape` mirror (linear strata reconstruction, in-layer).
fn surface_shape(strata: Vec4, layer_height: f32) -> f32 {
    let z = layer_height.clamp(0.0, 1.0) * 4.0 - 0.5;
    if z <= 0.0 {
        strata.x
    } else if z < 1.0 {
        strata.x + (strata.y - strata.x) * z
    } else if z < 2.0 {
        strata.y + (strata.z - strata.y) * (z - 1.0)
    } else if z < 3.0 {
        strata.z + (strata.w - strata.z) * (z - 2.0)
    } else {
        strata.w
    }
}

/// `cloud_surface_shape` mirror including the out-of-layer hard zero.
fn surface_shape_gated(strata: Vec4, layer_height: f32) -> f32 {
    if layer_height <= -0.04 || layer_height >= 1.04 {
        return 0.0;
    }
    surface_shape(strata, layer_height)
}

fn rotated_domain(p: Vec3) -> Vec3 {
    Vec3::new(
        0.72 * p.x + 0.41 * p.y - 0.56 * p.z,
        -0.35 * p.x + 0.91 * p.y + 0.22 * p.z,
        0.60 * p.x + 0.05 * p.y + 0.80 * p.z,
    )
}

fn phase_offset(weather: &[f32; 4], period: f32) -> Vec3 {
    period
        * Vec3::new(
            1.35 * (weather[0] - 0.5) + 0.65 * (weather[2] - 0.5),
            -1.10 * (weather[3] - 0.5) + 0.55 * (weather[0] - 0.5),
            1.20 * (weather[0] - 0.5) - 0.75 * (weather[2] - 0.5)
                + 0.45 * (weather[3] - 0.5),
        )
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Interpolate NaN gaps and enforce a monotone non-decreasing response.
fn fill_gaps_monotone(nodes: &mut [f32; FILL_RESPONSE_ENTRIES]) {
    let mut last_known = 0usize;
    for i in 1..FILL_RESPONSE_ENTRIES {
        if nodes[i].is_nan() {
            continue;
        }
        let gap = i - last_known;
        if gap > 1 {
            for j in (last_known + 1)..i {
                let f = (j - last_known) as f32 / gap as f32;
                nodes[j] = nodes[last_known] + (nodes[i] - nodes[last_known]) * f;
            }
        }
        last_known = i;
    }
    for i in (last_known + 1)..FILL_RESPONSE_ENTRIES {
        nodes[i] = nodes[last_known];
    }
    for i in 1..FILL_RESPONSE_ENTRIES {
        nodes[i] = nodes[i].max(nodes[i - 1]);
    }
}

// ── CPU mirror of the marcher's 64³ tileable noise volume ──────────────────
// Algorithms are byte-for-byte ports of `common.wgsl` (`hash13`, `value_hash`,
// `hash_based_noise`, `voronoi`, `tilable_*`) and `render_clouds_volume` /
// `cloud_volume` in `clouds_compute.wgsl`. WGSL `fract` is floor-based —
// `wgsl_fract` below — NOT Rust's trunc-based `f32::fract`.

const WORLEY_RESOLUTION: usize = 64;

struct NoiseVolume {
    texels: Vec<Vec4>,
}

impl NoiseVolume {
    fn generate() -> Self {
        let texels = (0..WORLEY_RESOLUTION * WORLEY_RESOLUTION * WORLEY_RESOLUTION)
            .into_par_iter()
            .map(|i| {
                let x = i % WORLEY_RESOLUTION;
                let y = (i / WORLEY_RESOLUTION) % WORLEY_RESOLUTION;
                let z = i / (WORLEY_RESOLUTION * WORLEY_RESOLUTION);
                let coord = Vec3::new(
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                    z as f32 + 0.5,
                ) / WORLEY_RESOLUTION as f32;
                let perlin = tilable_perlin_fbm(coord, 3, 2.0);
                let cellular = tilable_voronoi(coord + Vec3::new(0.17, 0.31, 0.07), 2, 3.0);
                let erosion = tilable_voronoi(coord + Vec3::new(0.53, 0.11, 0.43), 1, 8.0);
                let macro_noise = tilable_perlin_fbm(coord + Vec3::new(0.29, 0.47, 0.61), 3, 1.0);
                Vec4::new(perlin, cellular, erosion, macro_noise)
                    .clamp(Vec4::ZERO, Vec4::ONE)
            })
            .collect();
        Self { texels }
    }

    fn corner(&self, c: IVec3) -> Vec4 {
        let w = (c + IVec3::splat(WORLEY_RESOLUTION as i32))
            .rem_euclid(IVec3::splat(WORLEY_RESOLUTION as i32));
        self.texels[w.z as usize * WORLEY_RESOLUTION * WORLEY_RESOLUTION
            + w.y as usize * WORLEY_RESOLUTION
            + w.x as usize]
    }

    /// `cloud_volume`: wrap-first, trilinearly filtered fetch.
    fn trilinear(&self, position: Vec3, period: f32) -> Vec4 {
        let p = wgsl_fract3(position / period) * WORLEY_RESOLUTION as f32;
        let pf = p - 0.5;
        let base = pf.floor();
        let mut f = pf - base;
        f = f * f * (Vec3::splat(3.0) - 2.0 * f);
        let b = base.as_ivec3();
        let c000 = self.corner(b);
        let c100 = self.corner(b + IVec3::new(1, 0, 0));
        let c010 = self.corner(b + IVec3::new(0, 1, 0));
        let c110 = self.corner(b + IVec3::new(1, 1, 0));
        let c001 = self.corner(b + IVec3::new(0, 0, 1));
        let c101 = self.corner(b + IVec3::new(1, 0, 1));
        let c011 = self.corner(b + IVec3::new(0, 1, 1));
        let c111 = self.corner(b + IVec3::new(1, 1, 1));
        let x00 = c000.lerp(c100, f.x);
        let x10 = c010.lerp(c110, f.x);
        let x01 = c001.lerp(c101, f.x);
        let x11 = c011.lerp(c111, f.x);
        x00.lerp(x10, f.y).lerp(x01.lerp(x11, f.y), f.z)
    }
}

fn wgsl_fract(x: f32) -> f32 {
    x - x.floor()
}

fn wgsl_fract3(v: Vec3) -> Vec3 {
    Vec3::new(wgsl_fract(v.x), wgsl_fract(v.y), wgsl_fract(v.z))
}

/// WGSL `%` on floats is trunc-based (like Rust's `%` on f32).
fn wgsl_rem3(v: Vec3, tile: f32) -> Vec3 {
    Vec3::new(v.x % tile, v.y % tile, v.z % tile)
}

fn hash13(p3: Vec3) -> f32 {
    let mut p = wgsl_fract3(p3 * 1031.1031);
    p += Vec3::splat(p.dot(Vec3::new(p.y, p.z, p.x) + 19.19));
    wgsl_fract((p.x + p.y) * p.z)
}

fn value_hash(p3: Vec3) -> f32 {
    let mut p = wgsl_fract3(p3 * 0.1031);
    p += Vec3::splat(p.dot(Vec3::new(p.y, p.z, p.x) + 19.19));
    wgsl_fract((p.x + p.y) * p.z)
}

fn hash_based_noise(x: Vec3, tile: f32) -> f32 {
    let p = x.floor();
    let mut f = x - p;
    f = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let corner = |dx: f32, dy: f32, dz: f32| {
        value_hash(wgsl_rem3(p + Vec3::new(dx, dy, dz), tile))
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x0 = lerp(
        lerp(corner(0.0, 0.0, 0.0), corner(1.0, 0.0, 0.0), f.x),
        lerp(corner(0.0, 1.0, 0.0), corner(1.0, 1.0, 0.0), f.x),
        f.y,
    );
    let x1 = lerp(
        lerp(corner(0.0, 0.0, 1.0), corner(1.0, 0.0, 1.0), f.x),
        lerp(corner(0.0, 1.0, 1.0), corner(1.0, 1.0, 1.0), f.x),
        f.y,
    );
    lerp(x0, x1, f.z)
}

fn voronoi(x: Vec3, tile: f32) -> f32 {
    let p = x.floor();
    let f = x - p;
    let mut res = 100.0f32;
    for k in -1..=1 {
        for j in -1..=1 {
            for i in -1..=1 {
                let b = Vec3::new(i as f32, j as f32, k as f32);
                let c = wgsl_rem3(p + b, tile);
                let r = b - f + Vec3::splat(hash13(c));
                res = res.min(r.dot(r));
            }
        }
    }
    1.0 - res
}

fn tilable_voronoi(p: Vec3, octaves: u32, base_freq: f32) -> f32 {
    let mut freq = base_freq;
    let mut amplitude = 1.0;
    let mut noise = 0.0;
    let mut w = 0.0;
    for _ in 0..octaves {
        noise += amplitude * voronoi(p * freq, freq);
        w += amplitude;
        freq *= 2.0;
        amplitude *= 0.5;
    }
    noise / w
}

fn tilable_perlin_fbm(p: Vec3, octaves: u32, base_freq: f32) -> f32 {
    let mut freq = base_freq;
    let mut amplitude = 1.0;
    let mut noise = 0.0;
    let mut w = 0.0;
    for _ in 0..octaves {
        noise += amplitude * hash_based_noise(p * freq, freq);
        w += amplitude;
        freq *= 2.0;
        amplitude *= 0.5;
    }
    noise / w
}

/// Small deterministic xorshift for the Monte-Carlo sampling.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32) / ((1u64 << 24) as f32)
    }
}
