//! NTR-X2d: the coarse **conditioning chart** handed to the terrain-diffusion
//! producer — five channels plus a landform-province classification, derived
//! from Thalos's canonical macro terrain.
//!
//! # Why this exists
//!
//! The producer details what it is asked for. The first Thalos conditioning
//! (`export_thalos_macro.rs`, NTR-X2a) asked for one fractal relief field, a
//! latitude temperature ramp, and one moisture noise, on 3 of the 5 channels —
//! so the planet detailed into mountains more or less everywhere, and the two
//! seasonality channels fell back to the pipeline's own Perlin. The NTR-X2c
//! bench established that authored provinces *do* yield distinct landform
//! families, and that an authored precipitation gradient alone produces visible
//! rain-shadow asymmetry. This module is that finding turned into the chart.
//!
//! # What it may and may not do
//!
//! - **Elevation is the canonical height, untouched.** One height authority
//!   (`ProceduralSurface`); the chart never invents continents, or the runway
//!   site moves out from under the spaceport. Landform diversity is expressed
//!   through *climate* steering the producer's morphology, plus the province
//!   classification, never by rewriting the height field.
//! - **Thresholds are measured, not guessed.** Province cuts are quantiles of
//!   this body's own orogeny/relief distributions, so the classification does
//!   not silently degenerate if the canonical field's range changes.
//! - **Imports must be post-`finalize`.** The producer applies
//!   `finalize_synthetic_map` only on its *synthetic* path; a custom
//!   conditioning import bypasses it. So this module reproduces all four of its
//!   transforms itself — precipitation-dependent lapse rate, the 20 °C contrast
//!   stretch, the temperature-sd baseline, and the precipitation-CV damping.
//!   NTR-X2a applied only the lapse rate.
//!
//! Channel units are the producer's internal ones (`tiff_export::CHANNEL_FILES`):
//! elevation m, temperature °C, temperature sd **°C × 100** (WorldClim BIO4),
//! precipitation mm/yr, precipitation CV (BIO15).

use glam::DVec3;

use crate::procedural::ProceduralSurface;

/// Native detail scale of the released 90 m producer.
pub const NATIVE_PX_M: f64 = 90.0;
/// Native pixels per coarse conditioning cell.
pub const NATIVE_PER_COARSE: f64 = 256.0;
/// One coarse conditioning cell at the equator: 23.04 km.
pub const COARSE_PX_M: f64 = NATIVE_PX_M * NATIVE_PER_COARSE;

/// Envelope of the producer's own synthetic conditioning prior, measured over a
/// 512² cell window (`bench_conditioning.py calibrate`, 2026-07-29). Authored
/// channels are held inside it — outside, the model is extrapolating.
pub mod prior {
    /// Temperature sd (°C × 100): p1 / p99 of the prior.
    pub const TEMP_SD_MIN: f64 = 176.0;
    pub const TEMP_SD_MAX: f64 = 1_943.0;
    /// Annual precipitation (mm): p1 / p99.
    pub const PRECIP_MIN: f64 = 9.0;
    pub const PRECIP_MAX: f64 = 3_645.0;
    /// Precipitation CV: p1 / p99.
    pub const PRECIP_CV_MIN: f64 = 5.0;
    pub const PRECIP_CV_MAX: f64 = 141.0;
    /// Temperature (°C) — `finalize` clips to this range.
    pub const TEMP_MIN: f64 = -10.0;
    pub const TEMP_MAX: f64 = 40.0;
}

/// Sea-level temperature at the equator, °C.
pub const TEMP_EQUATOR_C: f64 = 28.0;
/// Sea-level temperature drop from equator to pole, °C. Chosen so polar sea
/// level (`28 - 34 = -6 °C`) stays above `finalize`'s -10 °C clip: high ground
/// still gets cold, but through the lapse rate rather than by saturating.
pub const TEMP_POLE_DROP_C: f64 = 34.0;

/// Landform province — the geomorphic class the conditioning asks the producer
/// to detail into. Derived, never authored directly: a projection of the
/// canonical height/orogeny/relief fields, in the spirit of
/// ADR-20260725T004758Z's "a blend, never a class" (this is the coarse
/// *generation-side* view; the render-side biome blend is a separate seam).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LandformProvince {
    Ocean,
    /// Submerged but shallow — the continental shelf.
    Shelf,
    /// Low-lying, low relief.
    Plain,
    /// Old craton: low relief, low orogeny, continental interior.
    Shield,
    /// High-standing **and** flat — the dissected-plateau candidate.
    Plateau,
    /// Active orogeny.
    Range,
    /// Interior low surrounded by higher ground.
    Basin,
}

impl LandformProvince {
    pub const ALL: [Self; 7] = [
        Self::Ocean,
        Self::Shelf,
        Self::Plain,
        Self::Shield,
        Self::Plateau,
        Self::Range,
        Self::Basin,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Ocean => "ocean",
            Self::Shelf => "shelf",
            Self::Plain => "plain",
            Self::Shield => "shield",
            Self::Plateau => "plateau",
            Self::Range => "range",
            Self::Basin => "basin",
        }
    }
}

/// An equirectangular coarse conditioning chart. Row 0 is the north pole edge,
/// column 0 is longitude 0 — the layout `thalos_export.py` and
/// `DiffusionSurface` both assume.
pub struct ConditioningChart {
    pub width: usize,
    pub height: usize,
    /// Metres above sea level (0 m datum). Canonical height, verbatim.
    pub elevation_m: Vec<f32>,
    /// °C, post-`finalize` (lapse applied, clipped, 20 °C stretch applied).
    pub temperature_c: Vec<f32>,
    /// °C × 100 (BIO4).
    pub temperature_sd_c100: Vec<f32>,
    /// mm/yr (BIO12).
    pub precipitation_mm: Vec<f32>,
    /// BIO15, post-damping.
    pub precipitation_cv: Vec<f32>,
    pub province: Vec<LandformProvince>,
    /// Metres of relief within the neighbourhood window — the plateau/range
    /// discriminator, kept for verification and province re-tuning.
    pub relief_m: Vec<f32>,
    /// Kilometres to the nearest ocean cell (continentality).
    pub coast_distance_km: Vec<f32>,
    /// Fraction of precipitation removed by upwind barriers, in `[0, 1]`.
    pub rain_shadow: Vec<f32>,
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    if (b - a).abs() < f64::EPSILON {
        return if x < a { 0.0 } else { 1.0 };
    }
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Quantile of an unsorted slice, by copy-and-sort. Charts are ~869×434, so
/// this is negligible against the field evaluation.
fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    v[idx]
}

/// Latitude of row `y`'s centre, radians.
fn row_lat(y: usize, height: usize) -> f64 {
    (90.0 - (y as f64 + 0.5) / height as f64 * 180.0).to_radians()
}

/// Unit direction of cell `(x, y)`. Matches `runway::latlon_dir`.
fn cell_dir(x: usize, y: usize, width: usize, height: usize) -> DVec3 {
    let lat = row_lat(y, height);
    let lon = ((x as f64 + 0.5) / width as f64 * 360.0).to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin())
}

/// Prevailing wind, as a signed column step: Earth's three-cell circulation —
/// tropical easterlies, mid-latitude westerlies, polar easterlies. The sign is
/// the direction the wind blows *toward*, so upwind is the opposite step.
fn prevailing_wind_step(lat_deg: f64) -> i32 {
    let a = lat_deg.abs();
    if a < 30.0 {
        -1 // easterlies: air moves westward
    } else if a < 60.0 {
        1 // westerlies
    } else {
        -1
    }
}

impl ConditioningChart {
    /// Sample `surface` onto a `width × width/2` equirect grid and derive every
    /// conditioning channel. `width` is normally
    /// `round(2π · radius / COARSE_PX_M)` so the chart is metrically exact at
    /// the equator.
    pub fn build(surface: &ProceduralSurface, width: usize, lod_m: f32) -> Self {
        let width = width.max(8);
        let height = (width / 2).max(4);
        let n = width * height;

        // --- pass 1: canonical signals -------------------------------------
        let mut elevation = vec![0f64; n];
        let mut orogeny = vec![0f64; n];
        let mut continentalness = vec![0f64; n];
        let mut moisture = vec![0f64; n];
        for y in 0..height {
            for x in 0..width {
                let s = surface.macro_signals(cell_dir(x, y, width, height), lod_m);
                let i = y * width + x;
                elevation[i] = s.height_m;
                orogeny[i] = s.orogeny;
                continentalness[i] = s.continentalness;
                moisture[i] = s.moisture;
            }
        }

        // --- pass 2: derived geometry --------------------------------------
        let relief = neighbourhood_relief(&elevation, width, height, 2);
        let coast_km = coast_distance_km(&elevation, width, height);
        let shadow = orographic_shadow(&elevation, width, height);

        // Province cuts from this body's own distributions. Land only — ocean
        // cells would dominate every quantile and drag the cuts to zero.
        let land: Vec<usize> = (0..n).filter(|&i| elevation[i] > 0.0).collect();
        let land_oro: Vec<f64> = land.iter().map(|&i| orogeny[i]).collect();
        let land_relief: Vec<f64> = land.iter().map(|&i| relief[i]).collect();
        let land_elev: Vec<f64> = land.iter().map(|&i| elevation[i]).collect();
        let oro_range = quantile(&land_oro, 0.70);
        let relief_flat = quantile(&land_relief, 0.35);
        let relief_rough = quantile(&land_relief, 0.75);
        let elev_high = quantile(&land_elev, 0.75);

        let mut out = Self {
            width,
            height,
            elevation_m: vec![0.0; n],
            temperature_c: vec![0.0; n],
            temperature_sd_c100: vec![0.0; n],
            precipitation_mm: vec![0.0; n],
            precipitation_cv: vec![0.0; n],
            province: vec![LandformProvince::Ocean; n],
            relief_m: relief.iter().map(|&v| v as f32).collect(),
            coast_distance_km: coast_km.iter().map(|&v| v as f32).collect(),
            rain_shadow: shadow.iter().map(|&v| v as f32).collect(),
        };

        for y in 0..height {
            let sin_lat = row_lat(y, height).sin().abs();
            for x in 0..width {
                let i = y * width + x;
                let h = elevation[i];

                // --- province ----------------------------------------------
                let p = if h <= -200.0 {
                    LandformProvince::Ocean
                } else if h <= 0.0 {
                    LandformProvince::Shelf
                } else if orogeny[i] >= oro_range && relief[i] >= relief_rough {
                    LandformProvince::Range
                } else if h >= elev_high && relief[i] <= relief_flat {
                    LandformProvince::Plateau
                } else if is_interior_low(&elevation, width, height, x, y) {
                    LandformProvince::Basin
                } else if relief[i] <= relief_flat && coast_km[i] > 600.0 {
                    LandformProvince::Shield
                } else {
                    LandformProvince::Plain
                };
                out.province[i] = p;

                // --- continentality ----------------------------------------
                // Saturates at ~1500 km inland, where maritime moderation of
                // the annual cycle has essentially stopped.
                let cont = smoothstep(80.0, 1_500.0, coast_km[i]);

                // --- precipitation (BIO12) ---------------------------------
                // Canonical macro moisture is the base; the orographic shadow
                // is the new term, and the bench showed it is enough on its own
                // to produce visibly asymmetric morphology across a range.
                let m = ((moisture[i] + 1.0) * 0.5).clamp(0.0, 1.0);
                let wet_base = 40.0 + 1_800.0 * m.powf(1.5);
                let precip = (wet_base * (1.0 - 0.80 * shadow[i]))
                    .clamp(prior::PRECIP_MIN, prior::PRECIP_MAX);

                // --- temperature (BIO1), post-finalize ---------------------
                // NTR-X2a used `27 - 52·sin²lat`, which put **land p1 through
                // p25 all at -17.5 °C** — exactly `finalize`'s clamp floor. A
                // quarter of the planet's temperature conditioning was
                // saturated, and saturated conditioning steers nothing. The
                // curve below keeps sea-level polar temperature just above the
                // -10 °C clip so the lapse rate, not the clamp, decides how
                // cold high ground gets.
                let temp_sea_level = TEMP_EQUATOR_C - TEMP_POLE_DROP_C * sin_lat * sin_lat;
                let temp = finalize_temperature(temp_sea_level, h, precip);

                // --- temperature seasonality (BIO4) ------------------------
                // Grows poleward and inland; the two are multiplicative, which
                // is what separates a maritime west coast from a continental
                // interior at the same latitude.
                let lat_t = smoothstep(0.05, 0.85, sin_lat);
                let season = lat_t.powf(1.15) * (0.35 + 0.65 * cont);
                let temp_sd = (prior::TEMP_SD_MIN
                    + (prior::TEMP_SD_MAX - prior::TEMP_SD_MIN) * season)
                    .clamp(prior::TEMP_SD_MIN, prior::TEMP_SD_MAX);

                // --- precipitation variability (BIO15) ---------------------
                // Arid and continental climates are the seasonal ones; the
                // damping term is `finalize`'s, and drives CV to 0 as
                // precipitation rises.
                let cv_raw = prior::PRECIP_CV_MIN
                    + (prior::PRECIP_CV_MAX - prior::PRECIP_CV_MIN)
                        * (1.0 - m).powf(1.2)
                        * (0.35 + 0.65 * cont);
                let precip_cv = finalize_precip_cv(cv_raw, precip);

                out.elevation_m[i] = h as f32;
                out.temperature_c[i] = temp as f32;
                out.temperature_sd_c100[i] = temp_sd as f32;
                out.precipitation_mm[i] = precip as f32;
                out.precipitation_cv[i] = precip_cv as f32;
            }
        }
        out
    }

    /// Fraction of cells in each province, for verification. Ocean included.
    /// Per-cell surface area weight, normalised so the equator row is 1.0.
    ///
    /// **Every statistic over this chart must use it.** An equirect grid has
    /// uniform rows in *latitude*, so a raw cell count over-represents polar
    /// rows severely — at 80° a cell covers 17 % of an equatorial one. Counting
    /// cells reported this planet's land fraction as 0.301 against the 0.352
    /// that `just map` measures from actual surface area, and made it look like
    /// the world missed its ~0.35 lore target when it does not.
    pub fn cell_weight(&self, y: usize) -> f64 {
        row_lat(y, self.height).cos().abs()
    }

    fn weighted_fraction(&self, pred: impl Fn(usize) -> bool) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for y in 0..self.height {
            let w = self.cell_weight(y);
            for x in 0..self.width {
                den += w;
                if pred(y * self.width + x) {
                    num += w;
                }
            }
        }
        if den > 0.0 { num / den } else { 0.0 }
    }

    /// Share of the body's **surface area** in each province. Ocean included.
    pub fn province_fractions(&self) -> Vec<(LandformProvince, f64)> {
        LandformProvince::ALL
            .iter()
            .map(|&p| (p, self.weighted_fraction(|i| self.province[i] == p)))
            .collect()
    }

    /// Share of the body's **surface area** above sea level.
    pub fn land_fraction(&self) -> f64 {
        self.weighted_fraction(|i| self.elevation_m[i] > 0.0)
    }
}

/// `finalize_synthetic_map`'s temperature path: precipitation-dependent lapse
/// rate, clip, then the sub-20 °C contrast stretch about the 20 °C pivot.
pub fn finalize_temperature(temp_sea_level_c: f64, elevation_m: f64, precip_mm: f64) -> f64 {
    let lapse = (-6.5 + 0.0015 * precip_mm).clamp(-9.8, -4.0) / 1000.0;
    let t =
        (temp_sea_level_c + lapse * elevation_m.max(0.0)).clamp(prior::TEMP_MIN, prior::TEMP_MAX);
    if t > 20.0 {
        t
    } else {
        (t - 20.0) * 1.25 + 20.0
    }
}

/// `finalize_synthetic_map`'s precipitation-CV damping: variability falls to
/// zero as annual precipitation approaches ~4500 mm.
pub fn finalize_precip_cv(cv: f64, precip_mm: f64) -> f64 {
    cv * ((185.0 - 0.04111 * precip_mm) / 185.0).max(0.0)
}

/// Max-minus-min elevation over a `(2r+1)²` window, with longitude wrap and
/// clamped rows. This is the plateau/range discriminator: a plateau is high
/// *and* flat, a range is high and rough.
fn neighbourhood_relief(elev: &[f64], width: usize, height: usize, r: usize) -> Vec<f64> {
    let mut out = vec![0f64; width * height];
    for y in 0..height {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(height - 1);
        for x in 0..width {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for yy in y0..=y1 {
                for dx in 0..=(2 * r) {
                    let xx = (x + width + dx - r) % width;
                    let v = elev[yy * width + xx];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            out[y * width + x] = (hi - lo).max(0.0);
        }
    }
    out
}

/// Distance from each cell to the nearest ocean cell, in km, by two chamfer
/// sweeps with longitude wrap. Step costs use the true metric spacing, which
/// compresses east-west as `cos(latitude)` on an equirect grid — ignoring that
/// would report polar interiors as far more continental than they are.
fn coast_distance_km(elev: &[f64], width: usize, height: usize) -> Vec<f64> {
    const FAR: f64 = 1.0e9;
    let mut d = vec![FAR; width * height];
    for i in 0..d.len() {
        if elev[i] <= 0.0 {
            d[i] = 0.0;
        }
    }
    // North-south spacing is uniform on an equirect grid; east-west is not.
    let dy_km = COARSE_PX_M / 1000.0;
    let dx_km: Vec<f64> = (0..height)
        .map(|y| (COARSE_PX_M / 1000.0) * row_lat(y, height).cos().abs().max(1e-3))
        .collect();

    // Two passes converge for the wrapped axis in practice; a third is cheap
    // insurance against a landmass that wraps the whole sphere.
    for _ in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let mut best = d[i];
                if y > 0 {
                    best = best.min(d[(y - 1) * width + x] + dy_km);
                }
                let xl = (x + width - 1) % width;
                best = best.min(d[y * width + xl] + dx_km[y]);
                d[i] = best;
            }
        }
        for y in (0..height).rev() {
            for x in (0..width).rev() {
                let i = y * width + x;
                let mut best = d[i];
                if y + 1 < height {
                    best = best.min(d[(y + 1) * width + x] + dy_km);
                }
                let xr = (x + 1) % width;
                best = best.min(d[y * width + xr] + dx_km[y]);
                d[i] = best;
            }
        }
    }
    d
}

/// Orographic rain shadow in `[0, 1]`: how much of a cell's moisture an upwind
/// barrier has already wrung out. Marches upwind along the prevailing wind for
/// the row's latitude belt, tracking the highest barrier crossed relative to
/// the cell's own elevation.
fn orographic_shadow(elev: &[f64], width: usize, height: usize) -> Vec<f64> {
    const MARCH_CELLS: usize = 14; // ~320 km upwind at 23 km/cell
    let mut barrier = vec![0f64; width * height];
    for y in 0..height {
        let upwind = -prevailing_wind_step(row_lat(y, height).to_degrees());
        for x in 0..width {
            let i = y * width + x;
            let own = elev[i];
            if own <= 0.0 {
                continue;
            }
            let mut b = 0.0f64;
            for step in 1..=MARCH_CELLS {
                let xx =
                    ((x as i64 + upwind as i64 * step as i64).rem_euclid(width as i64)) as usize;
                // Decay with distance: a range 300 km upwind casts a weaker
                // shadow than the same range just over the ridge.
                let fall = 1.0 - (step as f64 - 1.0) / MARCH_CELLS as f64;
                b = b.max((elev[y * width + xx] - own) * fall);
            }
            barrier[i] = b;
        }
    }

    // Threshold against **this body's** barrier distribution, not a guessed
    // metre value. Hardcoding smoothstep(300, 2200) put land p75 at exactly
    // zero on Thalos — a rain-shadow term that never fired anywhere, because
    // this planet's ranges are lower than the constant assumed. Quantiles keep
    // the term live on a low-relief world and stop it saturating on a high one.
    let positives: Vec<f64> = barrier.iter().copied().filter(|&b| b > 0.0).collect();
    if positives.is_empty() {
        return vec![0.0; width * height];
    }
    let lo = quantile(&positives, 0.55).max(60.0);
    let hi = quantile(&positives, 0.97).max(lo + 120.0);
    barrier.iter().map(|&b| smoothstep(lo, hi, b)).collect()
}

/// Is this land cell lower than the ring of land around it — an interior low
/// with no short path to the sea?
fn is_interior_low(elev: &[f64], width: usize, height: usize, x: usize, y: usize) -> bool {
    const R: usize = 3;
    let h = elev[y * width + x];
    if h <= 0.0 {
        return false;
    }
    let y0 = y.saturating_sub(R);
    let y1 = (y + R).min(height - 1);
    let mut ring_min = f64::INFINITY;
    let mut ring_max = f64::NEG_INFINITY;
    for yy in y0..=y1 {
        for dx in 0..=(2 * R) {
            let xx = (x + width + dx - R) % width;
            if yy == y && xx == x {
                continue;
            }
            let v = elev[yy * width + xx];
            ring_min = ring_min.min(v);
            ring_max = ring_max.max(v);
        }
    }
    // Enclosed: everything around is higher, and the rim stands well above.
    ring_min >= h - 20.0 && ring_max >= h + 250.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The producer skips `finalize` on the import path, so these two are the
    /// transforms our chart owes it. Both are regression-pinned against the
    /// upstream formulas.
    #[test]
    fn finalize_temperature_matches_upstream() {
        // Above the pivot: lapse only, no stretch.
        let hot = finalize_temperature(30.0, 0.0, 500.0);
        assert!((hot - 30.0).abs() < 1e-6, "{hot}");
        // Below the pivot: expanded 1.25x about 20 C.
        let cool = finalize_temperature(10.0, 0.0, 500.0);
        assert!((cool - 7.5).abs() < 1e-6, "{cool}");
        // Lapse pulls high ground down, and wet lapse is shallower than dry.
        let dry = finalize_temperature(27.0, 3_000.0, 100.0);
        let wet = finalize_temperature(27.0, 3_000.0, 3_000.0);
        assert!(wet > dry, "wet lapse must be shallower: {wet} vs {dry}");
        // Never escapes the range the producer clips to.
        let polar = finalize_temperature(-40.0, 4_000.0, 100.0);
        assert!(polar >= (prior::TEMP_MIN - 20.0) * 1.25 + 20.0);
    }

    #[test]
    fn precip_cv_damps_to_zero_when_wet() {
        assert!((finalize_precip_cv(100.0, 0.0) - 100.0).abs() < 1e-6);
        assert!(finalize_precip_cv(100.0, 2_000.0) < 60.0);
        assert_eq!(finalize_precip_cv(100.0, 6_000.0), 0.0);
    }

    #[test]
    fn relief_wraps_longitude() {
        // A ridge at column 0 must be seen by the last column, or every chart
        // grows a false flat seam down the date line.
        let (w, h) = (8usize, 4usize);
        let mut elev = vec![0f64; w * h];
        for y in 0..h {
            elev[y * w] = 1_000.0;
        }
        let relief = neighbourhood_relief(&elev, w, h, 2);
        assert!(relief[w - 1] >= 1_000.0, "{}", relief[w - 1]);
    }

    #[test]
    fn coast_distance_is_zero_at_sea_and_grows_inland() {
        let (w, h) = (16usize, 8usize);
        let mut elev = vec![-1_000f64; w * h];
        // A land block in the middle of one row band.
        for y in 3..5 {
            for x in 4..12 {
                elev[y * w + x] = 500.0;
            }
        }
        let d = coast_distance_km(&elev, w, h);
        assert_eq!(d[3 * w], 0.0);
        let edge = d[3 * w + 4];
        let middle = d[3 * w + 8];
        assert!(
            middle > edge,
            "interior must be farther: {middle} vs {edge}"
        );
    }

    #[test]
    fn rain_shadow_falls_on_the_lee_side_only() {
        // Mid-latitude row -> westerlies -> upwind is west (-1 column).
        let (w, h) = (64usize, 32usize);
        let lee_row = (0..h)
            .find(|&y| {
                let a = row_lat(y, h).to_degrees().abs();
                (30.0..60.0).contains(&a)
            })
            .expect("a mid-latitude row exists");
        let mut elev = vec![200f64; w * h];
        for y in 0..h {
            elev[y * w + 20] = 3_500.0; // a north-south wall
        }
        let s = orographic_shadow(&elev, w, h);
        let lee = s[lee_row * w + 22]; // east of the wall = downwind
        let windward = s[lee_row * w + 18]; // west of the wall = upwind
        assert!(lee > 0.5, "lee must be shadowed: {lee}");
        assert!(windward < 0.05, "windward must be clear: {windward}");
    }

    /// Both of these fired on the first NTR-X2d chart and are the reason the
    /// module calibrates against measured distributions instead of constants.
    #[test]
    fn conditioning_channels_are_not_saturated_or_dead() {
        let surface = ProceduralSurface::new(3_186_000.0, 2);
        let chart = ConditioningChart::build(&surface, 256, 23_040.0);
        let land: Vec<usize> = (0..chart.elevation_m.len())
            .filter(|&i| chart.elevation_m[i] > 0.0)
            .collect();
        assert!(!land.is_empty());

        // A value pinned to `finalize`'s clamp floor carries no information.
        // NTR-X2a's curve saturated 25 % of land here.
        let floor = (prior::TEMP_MIN - 20.0) * 1.25 + 20.0;
        let pinned = land
            .iter()
            .filter(|&&i| (f64::from(chart.temperature_c[i]) - floor).abs() < 0.05)
            .count();
        let frac = pinned as f64 / land.len() as f64;
        assert!(
            frac < 0.05,
            "{:.1} % of land is pinned at the temperature floor {floor} °C",
            frac * 100.0
        );

        // A rain-shadow term that is zero almost everywhere is not modelling
        // anything. The first version had land p75 = 0.0 exactly.
        let mut shadowed: Vec<f64> = land
            .iter()
            .map(|&i| f64::from(chart.rain_shadow[i]))
            .collect();
        shadowed.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90 = shadowed[(shadowed.len() - 1) * 9 / 10];
        assert!(p90 > 0.05, "rain shadow is dead: land p90 = {p90}");
    }

    /// Equirect cell counts over-represent the poles, so every reported share
    /// must weight by `cos(lat)`. Found by cross-checking against `just map`'s
    /// area-based land fraction, which disagreed with the cell count by ~5 pp.
    #[test]
    fn area_weighting_is_applied_to_reported_fractions() {
        let surface = ProceduralSurface::new(3_186_000.0, 2);
        let chart = ConditioningChart::build(&surface, 256, 23_040.0);

        // Province shares are a partition of the surface: they must sum to 1.
        let total: f64 = chart.province_fractions().iter().map(|(_, f)| f).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "provinces sum to {total}, not 1"
        );

        // The weighted land fraction must differ from the naive cell count,
        // because this planet's land is not uniformly distributed in latitude.
        // If these ever coincide, the weighting has been silently dropped.
        let naive = chart.elevation_m.iter().filter(|&&h| h > 0.0).count() as f64
            / chart.elevation_m.len() as f64;
        let weighted = chart.land_fraction();
        assert!(
            (weighted - naive).abs() > 0.005,
            "weighted {weighted:.4} vs naive {naive:.4} — area weighting looks inert"
        );

        // Polar rows must carry less weight than equatorial ones.
        assert!(chart.cell_weight(0) < chart.cell_weight(chart.height / 2));
    }

    #[test]
    fn chart_is_deterministic_and_in_prior_envelope() {
        let surface = ProceduralSurface::new(3_186_000.0, 2);
        let a = ConditioningChart::build(&surface, 64, 23_040.0);
        let b = ConditioningChart::build(&surface, 64, 23_040.0);
        assert_eq!(a.temperature_c, b.temperature_c);
        assert_eq!(a.precipitation_mm, b.precipitation_mm);

        // Every authored channel must land inside the distribution the producer
        // was primed on, or it is extrapolating and the output is unreliable.
        for (i, &p) in a.precipitation_mm.iter().enumerate() {
            assert!(
                (prior::PRECIP_MIN as f32..=prior::PRECIP_MAX as f32).contains(&p),
                "precip {p} out of prior envelope at {i}"
            );
            let sd = a.temperature_sd_c100[i];
            assert!(
                (prior::TEMP_SD_MIN as f32..=prior::TEMP_SD_MAX as f32).contains(&sd),
                "temp sd {sd} out of prior envelope at {i}"
            );
            let cv = a.precipitation_cv[i];
            assert!(
                (0.0..=prior::PRECIP_CV_MAX as f32).contains(&cv),
                "precip cv {cv} out of prior envelope at {i}"
            );
        }
    }
}
