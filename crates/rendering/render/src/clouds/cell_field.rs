//! CPU mirror of the shared cell-scale cloud morphology field
//! (`thalos::atmosphere`'s `cloud_cell_*` in `atmosphere.wgsl`).
//!
//! The aperiodic, direction-parameterized column field that carries cell-scale
//! morphology in the near render tier, plus the per-place **style** that decides
//! whether a region's convection reads as wind-aligned rolls, round open cells,
//! a coarse storm cluster, or a lane-cut sheet.
//!
//! This module exists so the mirror has **one** Rust home. It previously lived
//! inside [`super::fill_lut`] as private helpers, and the weather producer
//! (`thalos_runtime::solar_system_state`) needed the same style to stretch its
//! own mesoscale breakup along the same axis — which would have made three
//! copies of one contract to keep in lockstep instead of two. There are now
//! exactly two: this file and the WGSL.
//!
//! **Keep in lockstep with `atmosphere.wgsl`.** A divergence here silently fits
//! the near tier's threshold curve to a different renderer than the one that
//! draws the frame, and that failure is invisible in a screenshot — it shows up
//! as the far tier and the near tier disagreeing about how cloudy the planet is.

use bevy::math::Vec3;

// ── The one rule this design exists to obey ─────────────────────────────────
//
// **A per-place style may never scale the sampling domain.**
//
// The field is sampled at `dir * (radius / period)` — ~590 lattice units on a
// Thalos-sized body. Let the period vary across the planet and the chain rule
// adds a second term to the sampling gradient: the field runs at a LOCAL
// frequency well above the one its period nominally sets, while the octave fade
// still band-limits against that nominal period. The octave is under-filtered by
// exactly that factor, and it renders as fine feathered hatching.
//
// Measured by `live_style_field_stays_coherent_across_style_boundaries`: the
// shipped varying period ran 1.70× finer than nominal; this design runs at
// 1.13×, and that residual is the roll variant's genuinely finer across-street
// spacing, which its own band-limit accounts for. A varying `zonal_aspect` has
// the identical flaw, since it scales the domain by √a.
//
// Everything here is therefore built from operations that cannot add a
// sampling-gradient term: octave WEIGHTS (multiply already-sampled values),
// BILLOW (blends two values at the same point), and a ROLL BLEND between two
// globally constant transforms. Do not reintroduce a varying period or aspect —
// it is not recoverable by tuning.

/// Cell period. Globally constant — see the note above.
const CELL_PERIOD_M: f32 = 5400.0;
/// The fixed anisotropic variant of the arrangement octave.
const CELL_ROLL_ASPECT: f32 = 3.0;
/// Shear off the exact east–west line; non-zero on purpose, because mixing
/// longitude back into the lattice y is what stops a high aspect degenerating
/// into pure latitude bands (circular contours on a sphere).
const CELL_ROLL_TILT: f32 = 0.35;
const CELL_LACUNARITY: f32 = 2.37;
const CELL_BILLOW_MEAN: f32 = 0.302816;
/// Spread correction and soft-saturation knee — see the WGSL notes. Without the
/// gain the field's std is 0.115 and the threshold is a cliff; without the soft
/// knee the gain saturates ~10 % of the sky flat and the deck renders as mesas.
const CELL_GAIN: f32 = 3.2;
const CELL_KNEE: f32 = 0.45;
const CELL_BILLOW: [f32; 3] = [0.0, 0.75, 0.35];
/// Quadratic fit of one octave's σ against its billow blend (worst error
/// 0.0023 over 11 measured points), and the σ of the default mix that every
/// style is renormalized back onto. See the calibration note in the WGSL.
const CELL_SIGMA_FIT: [f32; 3] = [0.184571, -0.200902, 0.230569];
const CELL_SIGMA_REF: f32 = 0.123275;

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(f32::EPSILON)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// How a place's clouds are ORGANIZED, as opposed to how much of them there is.
/// Mirrors WGSL `CloudCellStyle` field for field.
#[derive(Clone, Copy, Debug)]
pub struct CellStyle {
    /// Octave weights — the cell-size control. Must sum to 1 so the field's
    /// mean stays 0.5. This is what replaces a varying period.
    pub weights: [f32; 3],
    /// Blend toward the fixed anisotropic variant of the arrangement octave:
    /// 0 = round cells, 1 = wind-aligned rolls and lanes.
    pub roll: f32,
    /// Scale on the per-octave billow ladder.
    pub billow: f32,
    /// `CELL_SIGMA_REF / σ(style)`, so the field's distribution — and therefore
    /// the meaning of the authored coverage channel — does not vary with style.
    pub spread_norm: f32,
}

/// `strata_warp_hash` mirror: integer hash of a lattice corner.
fn cell_hash(p: Vec3) -> f32 {
    let q = [
        (p.x.floor() as i32 as u32).wrapping_mul(1_597_334_673),
        (p.y.floor() as i32 as u32).wrapping_mul(3_812_015_801),
        (p.z.floor() as i32 as u32).wrapping_mul(2_798_796_415),
    ];
    let n = (q[0] ^ q[1] ^ q[2]).wrapping_mul(1_597_334_673);
    n as f32 * (1.0 / 4_294_967_295.0)
}

/// `strata_warp_noise` mirror: smoothstep-interpolated lattice value noise.
pub fn cell_noise(x: Vec3) -> f32 {
    let p = x.floor();
    let mut f = x - p;
    f = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let c = |dx: f32, dy: f32, dz: f32| cell_hash(p + Vec3::new(dx, dy, dz));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(c(0.0, 0.0, 0.0), c(1.0, 0.0, 0.0), f.x);
    let x10 = lerp(c(0.0, 1.0, 0.0), c(1.0, 1.0, 0.0), f.x);
    let x01 = lerp(c(0.0, 0.0, 1.0), c(1.0, 0.0, 1.0), f.x);
    let x11 = lerp(c(0.0, 1.0, 1.0), c(1.0, 1.0, 1.0), f.x);
    lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z)
}

/// `cloud_cell_spread_norm` mirror: analytic σ of the weighted octave mix.
/// Both the weights and the billow move σ, so both are folded in — that is what
/// keeps authored coverage meaning the same thing everywhere under one
/// planet-wide threshold fit.
fn cell_spread_norm(weights: [f32; 3], billow_scale: f32) -> f32 {
    let mut acc = 0.0;
    for i in 0..3 {
        let b = CELL_BILLOW[i] * billow_scale;
        let s = CELL_SIGMA_FIT[0] + CELL_SIGMA_FIT[1] * b + CELL_SIGMA_FIT[2] * b * b;
        acc += weights[i] * weights[i] * s * s;
    }
    CELL_SIGMA_REF / acc.sqrt().max(1.0e-5)
}

/// `cloud_cell_style` mirror. `cloud_type` is the weather cube's type channel
/// (0 = sheet, 1 = deep convection).
pub fn cell_style(dir: Vec3, cloud_type: f32) -> CellStyle {
    let org_raw = cell_noise(dir * 5.0 + Vec3::new(61.0, -23.0, 14.0));

    let abs_lat = dir.y.clamp(-1.0, 1.0).abs();
    let roll_belt = smoothstep(0.08, 0.30, abs_lat) * (1.0 - smoothstep(0.60, 0.88, abs_lat));
    let not_storm = 1.0 - smoothstep(0.70, 0.90, cloud_type);
    let roll = smoothstep(0.44, 0.80, org_raw) * roll_belt * not_storm;

    let storm_w = smoothstep(0.72, 0.88, cloud_type);
    let sheet_w = 1.0 - smoothstep(0.14, 0.42, cloud_type);

    // Cell size comes from the octave WEIGHTS, never from a scaled period — see
    // the phase-gradient note at the top. Both endpoints sum to 1 so the mean
    // stays 0.5 whatever the blend.
    let size_t =
        (smoothstep(0.20, 0.80, org_raw) - 0.35 * storm_w - 0.15 * sheet_w).clamp(0.0, 1.0);
    let weights = [
        mix(0.78, 0.44, size_t),
        mix(0.16, 0.34, size_t),
        mix(0.06, 0.22, size_t),
    ];
    let billow = mix(1.15, 0.30, sheet_w);
    // A blend weight between two CONSTANT transforms, not a transform parameter.
    let polar_fade = 1.0 - smoothstep(0.62, 0.90, abs_lat);
    let roll_blend =
        ((roll + 0.55 * sheet_w * smoothstep(0.50, 0.86, org_raw)) * polar_fade).clamp(0.0, 1.0);

    CellStyle {
        weights,
        roll: roll_blend,
        billow,
        spread_norm: cell_spread_norm(weights, billow),
    }
}

/// `cloud_cell_domain` mirror: the sampling domain at a CONSTANT anisotropy.
pub fn cell_domain(dir: Vec3, radius: f32, period_m: f32, aspect: f32, tilt: f32) -> Vec3 {
    let a = aspect.max(1.0);
    // Constant cell area: across ÷ √a, along × √a.
    let k = radius / (period_m / a.sqrt());
    let inv = 1.0 / a;
    let p = Vec3::new(dir.x * k * inv, dir.y * k, dir.z * k * inv);
    Vec3::new(p.x, p.y + tilt * (p.x + p.z), p.z)
}

/// `cloud_cell_shape` mirror at full detail (no `filter_m` fade: the
/// calibration is derived at full detail, which is the field the near march
/// integrates wherever fill is decided, and the fade is mean-preserving by
/// construction so one derivation serves every distance).
fn cell_shape(v: f32, billow: f32) -> f32 {
    let b = 0.5 + ((2.0 * v - 1.0).abs() - CELL_BILLOW_MEAN);
    v + (b - v) * billow
}

fn cell_octave(dir: Vec3, radius: f32, period_m: f32, offset: Vec3, billow: f32) -> f32 {
    cell_shape(
        cell_noise(cell_domain(dir, radius, period_m, 1.0, 0.0) + offset),
        billow,
    )
}

/// `cloud_cell_arrangement` mirror: isotropic cross-faded toward the fixed
/// anisotropic variant, renormalized for the variance a blend of two
/// decorrelated fields loses.
fn cell_arrangement(
    dir: Vec3,
    radius: f32,
    period_m: f32,
    offset: Vec3,
    billow: f32,
    roll: f32,
) -> f32 {
    let iso = cell_octave(dir, radius, period_m, offset, billow);
    if roll <= 1.0e-3 {
        return iso;
    }
    let rolled = cell_shape(
        cell_noise(cell_domain(dir, radius, period_m, CELL_ROLL_ASPECT, CELL_ROLL_TILT) + offset),
        billow,
    );
    let blended = mix(iso, rolled, roll);
    let shrink = (roll * roll + (1.0 - roll) * (1.0 - roll)).sqrt();
    0.5 + (blended - 0.5) / shrink.max(1.0e-3)
}

/// `cloud_cell_field` mirror at full detail. Octave periods are global
/// constants; only the weights vary per place.
pub fn cell_field(dir: Vec3, radius: f32, style: &CellStyle) -> f32 {
    let p0 = CELL_PERIOD_M;
    let p1 = p0 / CELL_LACUNARITY;
    let p2 = p1 / CELL_LACUNARITY;
    let b = [
        CELL_BILLOW[0] * style.billow,
        CELL_BILLOW[1] * style.billow,
        CELL_BILLOW[2] * style.billow,
    ];
    let o0 = cell_arrangement(
        dir,
        radius,
        p0,
        Vec3::new(11.3, -4.1, 27.9),
        b[0],
        style.roll,
    );
    let o1 = cell_octave(dir, radius, p1, Vec3::new(-23.7, 8.4, 3.2), b[1]);
    let o2 = cell_octave(dir, radius, p2, Vec3::new(5.9, 31.2, -17.6), b[2]);
    let raw = style.weights[0] * o0 + style.weights[1] * o1 + style.weights[2] * o2;
    let x = (raw - 0.5) * style.spread_norm * CELL_GAIN;
    0.5 + 0.5 * x / (CELL_KNEE + x.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden-ratio spiral over the sphere — a deterministic, dependency-free
    /// low-discrepancy direction sequence.
    fn dir_at(i: u32, n: u32) -> Vec3 {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = (i as f64) * 2.399_963_229_728_653;
        Vec3::new((r * phi.cos()) as f32, z as f32, (r * phi.sin()) as f32)
    }

    fn field_stats(style_of: impl Fn(Vec3) -> CellStyle) -> (f32, f32) {
        const N: u32 = 200_000;
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        for i in 0..N {
            let dir = dir_at(i, N);
            let v = cell_field(dir, 3_186_000.0, &style_of(dir)) as f64;
            let d = v - mean;
            mean += d / (i + 1) as f64;
            m2 += d * (v - mean);
        }
        (mean as f32, (m2 / N as f64).sqrt() as f32)
    }

    /// The near tier's formation threshold is ONE Monte-Carlo fit over the whole
    /// planet, so authored coverage only keeps its meaning if the cell field's
    /// distribution is style-INVARIANT. The octave weights and the billow blend
    /// both move σ, which is what `spread_norm` corrects.
    #[test]
    fn cell_field_distribution_is_style_invariant() {
        let fixed = |weights: [f32; 3], roll: f32, billow: f32| CellStyle {
            weights,
            roll,
            billow,
            spread_norm: cell_spread_norm(weights, billow),
        };
        let coarse = [0.78, 0.16, 0.06];
        let fine = [0.44, 0.34, 0.22];
        let base = [0.62, 0.26, 0.12];
        let styles = [
            ("default", fixed(base, 0.0, 1.0)),
            ("large cells", fixed(coarse, 0.0, 1.15)),
            ("small cells", fixed(fine, 0.0, 1.15)),
            ("rolls", fixed(base, 1.0, 1.15)),
            ("half-rolled", fixed(coarse, 0.5, 1.15)),
            ("sheet lanes", fixed(coarse, 1.0, 0.30)),
        ];
        for (name, style) in styles {
            let (mean, std) = field_stats(|_| style);
            assert!(
                (mean - 0.5).abs() < 0.012,
                "{name}: mean {mean} drifted off 0.5"
            );
            assert!(
                (std - 0.2036).abs() < 0.012,
                "{name}: std {std} drifted off the calibrated 0.2036"
            );
        }
    }

    /// The live style field (which varies every knob together across the planet)
    /// must land on the same distribution as the un-styled field, or the whole
    /// planet's coverage is re-biased.
    #[test]
    fn live_style_field_matches_calibrated_distribution() {
        for cloud_type in [0.05f32, 0.45, 0.80] {
            let (mean, std) = field_stats(|dir| cell_style(dir, cloud_type));
            assert!(
                (mean - 0.5).abs() < 0.012,
                "type {cloud_type}: mean {mean} drifted off 0.5"
            );
            assert!(
                (std - 0.2036).abs() < 0.012,
                "type {cloud_type}: std {std} drifted off the calibrated 0.2036"
            );
        }
    }

    /// Move `dir` by `arc_m` of surface arc, tangentially.
    fn step_dir(dir: Vec3, arc_m: f32, radius: f32) -> Vec3 {
        let helper = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let tangent = dir.cross(helper).normalize();
        (dir + tangent * (arc_m / radius)).normalize()
    }

    /// **The regression guard for the phase-gradient rule.**
    ///
    /// A style that scales the sampling domain does not deform the field, it
    /// decorrelates it: the domain is ~590 lattice units, so a 1 % period change
    /// moves the sample by ~6 whole cells. That renders as fine feathered
    /// hatching wherever the style varies, and it is exactly what shipped twice
    /// (2026-07-26) before being isolated by capture.
    ///
    /// The invariant, stated so it cannot be tuned away: over a step MUCH
    /// shorter than the cell period, the live field must change much less than
    /// it does over a full period. A decorrelating field scores ~1.0 here; a
    /// coherent one scores well under 0.5. Reintroduce a varying period or
    /// aspect and this fails.
    #[test]
    fn live_style_field_stays_coherent_across_style_boundaries() {
        const RADIUS: f32 = 3_186_000.0;

        /// Mean |Δ| over a step far shorter than the cell period — a direct
        /// measure of the field's LOCAL spatial frequency. A style that scales
        /// the sampling domain adds its own phase gradient on top of the
        /// intrinsic one, so the field runs finer than its nominal period while
        /// the band-limit fade still filters for the nominal period. That gap is
        /// the aliasing, and this number is what exposes it.
        fn local_gradient(field: impl Fn(Vec3) -> f64) -> f64 {
            const N: u32 = 120_000;
            let mut acc = 0.0f64;
            for i in 0..N {
                let dir = dir_at(i, N);
                acc += (field(step_dir(dir, 120.0, RADIUS)) - field(dir)).abs();
            }
            acc / N as f64
        }

        let baseline = local_gradient(|dir| {
            let style = CellStyle {
                weights: [0.62, 0.26, 0.12],
                roll: 0.0,
                billow: 1.0,
                spread_norm: cell_spread_norm([0.62, 0.26, 0.12], 1.0),
            };
            cell_field(dir, RADIUS, &style) as f64
        });
        let live =
            local_gradient(|dir| cell_field(dir, RADIUS, &cell_style(dir, 0.45)) as f64) / baseline;

        // The metric has to be able to FAIL, or it guards nothing. This is the
        // defect as it actually shipped: the same field with the period scaled
        // by the same smooth style field it used to be scaled by.
        let buggy = local_gradient(|dir| {
            let org = cell_noise(dir * 5.0 + Vec3::new(61.0, -23.0, 14.0));
            let period = CELL_PERIOD_M * mix(1.55, 0.78, smoothstep(0.20, 0.80, org));
            let b = CELL_BILLOW;
            let o0 = cell_shape(
                cell_noise(
                    cell_domain(dir, RADIUS, period, 1.0, 0.0) + Vec3::new(11.3, -4.1, 27.9),
                ),
                b[0],
            );
            let p1 = period / CELL_LACUNARITY;
            let o1 = cell_shape(
                cell_noise(cell_domain(dir, RADIUS, p1, 1.0, 0.0) + Vec3::new(-23.7, 8.4, 3.2)),
                b[1],
            );
            let p2 = p1 / CELL_LACUNARITY;
            let o2 = cell_shape(
                cell_noise(cell_domain(dir, RADIUS, p2, 1.0, 0.0) + Vec3::new(5.9, 31.2, -17.6)),
                b[2],
            );
            let raw = 0.62 * o0 + 0.26 * o1 + 0.12 * o2;
            let x = (raw - 0.5) * CELL_GAIN;
            (0.5 + 0.5 * x / (CELL_KNEE + x.abs())) as f64
        }) / baseline;

        println!("local gradient vs constant-period baseline: live {live:.3}, buggy {buggy:.3}");
        assert!(
            buggy > 1.5,
            "the metric no longer detects a domain-scaling style (buggy {buggy:.3}× baseline); \
             it has stopped guarding anything"
        );
        assert!(
            live < 1.4,
            "the field runs {live:.3}× finer than its nominal period (the known defect scores \
             {buggy:.3}×) — a style knob is scaling the sampling domain again, so the band-limit \
             fade is filtering for a period the field no longer has"
        );
    }
}
