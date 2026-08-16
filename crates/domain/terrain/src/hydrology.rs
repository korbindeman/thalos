//! Offline raster hydrology for a completed planetary heightfield.
//!
//! Drainage is downstream of terrain generation: the final solve must consume
//! the completed neural DEM so mountain passes, learned valleys, and detailed
//! divides all participate. This module therefore knows nothing about
//! [`SurfaceQuery`](crate::query::SurfaceQuery) or terrain synthesis. The
//! current `bake_rivers` example samples a coarse preview raster and hands it
//! here; the full neural bake will hand over its authored height band directly.
//!
//! The solver produces two deliberately separate quantities on one topology:
//! geometric upstream catchment area and climate-weighted annual-mean
//! discharge. Catchment describes the watershed and channel hierarchy;
//! discharge distinguishes a large arid basin from a perennial river.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rayon::prelude::*;

/// Receiver sentinel for ocean cells and cells outside the land network.
pub const NO_RECEIVER: u32 = u32::MAX;

const SECONDS_PER_YEAR: f32 = 31_556_952.0;

#[derive(Debug, Clone, Copy)]
pub struct HydrologyConfig {
    pub width: usize,
    pub height: usize,
    pub planet_radius_m: f64,
    /// Strictly positive rise added per priority-flood step. This breaks filled
    /// flats into deterministic downhill paths without materially changing the
    /// DEM. The per-cell multiplier is randomized deterministically to avoid a
    /// flood-sweep direction imprint.
    pub fill_epsilon_m: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HydrologyError {
    InvalidDimensions,
    GridTooLarge,
    HeightLength { expected: usize, actual: usize },
    RunoffLength { expected: usize, actual: usize },
    NoOceanSeed,
    NonFiniteHeight { index: usize },
    InvalidRunoff { index: usize },
}

impl std::fmt::Display for HydrologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions => {
                write!(f, "hydrology grid dimensions and radius must be positive")
            }
            Self::GridTooLarge => write!(f, "hydrology grid exceeds the u32 cell-address space"),
            Self::HeightLength { expected, actual } => {
                write!(f, "height raster has {actual} cells, expected {expected}")
            }
            Self::RunoffLength { expected, actual } => {
                write!(f, "runoff raster has {actual} cells, expected {expected}")
            }
            Self::NoOceanSeed => write!(f, "hydrology raster has no height <= 0 ocean seed"),
            Self::NonFiniteHeight { index } => {
                write!(
                    f,
                    "height raster contains a non-finite value at cell {index}"
                )
            }
            Self::InvalidRunoff { index } => {
                write!(
                    f,
                    "runoff raster contains a negative or non-finite value at cell {index}"
                )
            }
        }
    }
}

impl std::error::Error for HydrologyError {}

#[derive(Debug)]
pub struct DrainageSolve {
    /// One receiver per cell. Ocean cells use [`NO_RECEIVER`].
    pub receiver: Vec<u32>,
    /// Land cells ordered from upstream/high to downstream/low. Every donor
    /// precedes its receiver, so secondary extensive fields can reuse it.
    pub descending_land: Vec<u32>,
    /// Geometric upstream area in km². Ocean cells contain the total delivered
    /// to that coastal outlet; land cells contain their own area plus donors.
    pub catchment_km2: Vec<f32>,
    /// Climate-weighted annual-mean flow in m³/s on the same topology.
    pub discharge_m3_s: Vec<f32>,
    pub raised_cell_count: usize,
    pub mean_fill_depth_m: f64,
    pub max_fill_depth_m: f32,
}

/// Convert the canonical macro-moisture field `[-1, 1]` to a long-term runoff
/// depth. This is a deterministic runoff proxy, not a second climate model.
///
/// It mirrors the moisture→precipitation curve used by the neural conditioning
/// chart, then removes moisture-dependent interception/infiltration. A small
/// direct-runoff fraction keeps arid cells non-zero; wet headwaters still
/// contribute orders of magnitude more water and carry that flow through dry
/// downstream terrain.
pub fn annual_runoff_mm(moisture: f32) -> f32 {
    let m = ((moisture + 1.0) * 0.5).clamp(0.0, 1.0);
    let precipitation_mm = 40.0 + 1_800.0 * m.powf(1.5);
    let initial_loss_mm = 350.0 - 200.0 * m;
    let excess_mm = (precipitation_mm - initial_loss_mm).max(0.0);
    precipitation_mm * 0.015 + excess_mm * (0.18 + 0.47 * m)
}

/// Solve drainage on an equirectangular DEM. Longitude wraps; latitude clamps.
/// Heights at or below zero are ocean outlets.
pub fn solve_equirectangular(
    config: HydrologyConfig,
    height_m: &[f32],
    annual_runoff_mm: &[f32],
) -> Result<DrainageSolve, HydrologyError> {
    let HydrologyConfig {
        width,
        height,
        planet_radius_m,
        fill_epsilon_m,
    } = config;
    if width == 0
        || height == 0
        || !planet_radius_m.is_finite()
        || planet_radius_m <= 0.0
        || !fill_epsilon_m.is_finite()
        || fill_epsilon_m <= 0.0
    {
        return Err(HydrologyError::InvalidDimensions);
    }
    let cell_count = width
        .checked_mul(height)
        .ok_or(HydrologyError::GridTooLarge)?;
    if cell_count > u32::MAX as usize {
        return Err(HydrologyError::GridTooLarge);
    }
    if height_m.len() != cell_count {
        return Err(HydrologyError::HeightLength {
            expected: cell_count,
            actual: height_m.len(),
        });
    }
    if annual_runoff_mm.len() != cell_count {
        return Err(HydrologyError::RunoffLength {
            expected: cell_count,
            actual: annual_runoff_mm.len(),
        });
    }
    if let Some(index) = height_m.iter().position(|v| !v.is_finite()) {
        return Err(HydrologyError::NonFiniteHeight { index });
    }
    if let Some(index) = annual_runoff_mm
        .iter()
        .position(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(HydrologyError::InvalidRunoff { index });
    }

    let lat_of = |y: usize| (0.5 - (y as f64 + 0.5) / height as f64) * std::f64::consts::PI;
    let dy_m = std::f64::consts::PI * planet_radius_m / height as f64;
    let row_dx_m: Vec<f64> = (0..height)
        .map(|y| std::f64::consts::TAU * planet_radius_m * lat_of(y).cos() / width as f64)
        .collect();
    let row_area_km2: Vec<f64> = row_dx_m.iter().map(|dx_m| dx_m * dy_m / 1.0e6).collect();

    let key = |v: f32| -> u32 {
        let bits = v.to_bits();
        if v >= 0.0 { bits | 0x8000_0000 } else { !bits }
    };
    let mut filled = height_m.to_vec();
    let mut done = vec![false; cell_count];
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    for (i, &height) in height_m.iter().enumerate() {
        if height <= 0.0 {
            done[i] = true;
            heap.push(Reverse((key(height), i as u32)));
        }
    }
    if heap.is_empty() {
        return Err(HydrologyError::NoOceanSeed);
    }

    let mut raised_cell_count = 0usize;
    let mut fill_depth_sum_m = 0.0f64;
    let mut max_fill_depth_m = 0.0f32;
    while let Some(Reverse((_, i))) = heap.pop() {
        let i = i as usize;
        let (y, x) = (i / width, i % width);
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let ny = y as i64 + dy;
                if ny < 0 || ny >= height as i64 {
                    continue;
                }
                let nx = (x as i64 + dx).rem_euclid(width as i64);
                let j = ny as usize * width + nx as usize;
                if done[j] {
                    continue;
                }
                done[j] = true;
                let jitter = randomized_fill_step(j);
                let target = filled[i] + fill_epsilon_m * jitter;
                if filled[j] < target {
                    let depth = target - filled[j];
                    raised_cell_count += 1;
                    fill_depth_sum_m += f64::from(depth);
                    max_fill_depth_m = max_fill_depth_m.max(depth);
                    filled[j] = target;
                }
                heap.push(Reverse((key(filled[j]), j as u32)));
            }
        }
    }

    let receiver: Vec<u32> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let dx_m = row_dx_m[y];
            let filled = &filled;
            (0..width)
                .map(move |x| {
                    let i = y * width + x;
                    if height_m[i] <= 0.0 {
                        return NO_RECEIVER;
                    }
                    let mut best = NO_RECEIVER;
                    let mut best_slope = 0.0f64;
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let ny = y as i64 + dy;
                            if ny < 0 || ny >= height as i64 {
                                continue;
                            }
                            let nx = (x as i64 + dx).rem_euclid(width as i64);
                            let j = ny as usize * width + nx as usize;
                            let run_m =
                                ((dx as f64 * dx_m).powi(2) + (dy as f64 * dy_m).powi(2)).sqrt();
                            let slope = f64::from(filled[i] - filled[j]) / run_m;
                            if slope > best_slope {
                                best_slope = slope;
                                best = j as u32;
                            }
                        }
                    }
                    best
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut descending_land: Vec<u32> = (0..cell_count as u32)
        .filter(|&i| height_m[i as usize] > 0.0)
        .collect();
    descending_land.sort_unstable_by(|a, b| filled[*b as usize].total_cmp(&filled[*a as usize]));

    let catchment_km2 = accumulate_downstream(&descending_land, &receiver, |i| {
        row_area_km2[i / width] as f32
    });
    // 1 km² receiving 1 mm/year contributes 1,000 m³/year.
    let discharge_m3_s = accumulate_downstream(&descending_land, &receiver, |i| {
        row_area_km2[i / width] as f32 * annual_runoff_mm[i] * 1_000.0 / SECONDS_PER_YEAR
    });

    Ok(DrainageSolve {
        receiver,
        descending_land,
        catchment_km2,
        discharge_m3_s,
        raised_cell_count,
        mean_fill_depth_m: if raised_cell_count == 0 {
            0.0
        } else {
            fill_depth_sum_m / raised_cell_count as f64
        },
        max_fill_depth_m,
    })
}

fn randomized_fill_step(index: usize) -> f32 {
    let mut z = (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z ^= z >> 29;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    0.5 + ((z >> 40) as f32 / 16_777_216.0)
}

fn accumulate_downstream(
    descending_land: &[u32],
    receiver: &[u32],
    mut own: impl FnMut(usize) -> f32,
) -> Vec<f32> {
    let mut accumulated = vec![0.0; receiver.len()];
    for &i in descending_land {
        let i = i as usize;
        let total = accumulated[i] + own(i);
        accumulated[i] = total;
        let r = receiver[i];
        if r != NO_RECEIVER {
            accumulated[r as usize] += total;
        }
    }
    accumulated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(width: usize, height: usize) -> HydrologyConfig {
        HydrologyConfig {
            width,
            height,
            planet_radius_m: 1_000_000.0,
            fill_epsilon_m: 1.0e-3,
        }
    }

    #[test]
    fn runoff_proxy_is_monotone_and_separates_arid_from_humid_climate() {
        let dry = annual_runoff_mm(-1.0);
        let steppe = annual_runoff_mm(-0.5);
        let temperate = annual_runoff_mm(0.0);
        let humid = annual_runoff_mm(1.0);

        assert!(dry > 0.0 && dry < 2.0, "dry runoff {dry}");
        assert!(dry < steppe && steppe < temperate && temperate < humid);
        assert!(
            temperate > 100.0 && temperate < 300.0,
            "temperate runoff {temperate}"
        );
        assert!(humid > 900.0, "humid runoff {humid}");
    }

    #[test]
    fn downstream_accumulation_preserves_area_and_runoff_contrast() {
        let order = [0, 1, 2, 3];
        let receiver = [2, 2, 3, NO_RECEIVER];
        let area = accumulate_downstream(&order, &receiver, |_| 1.0);
        let runoff = [1.0, 10.0, 0.0, 0.0];
        let discharge = accumulate_downstream(&order, &receiver, |i| runoff[i]);

        assert_eq!(area, [1.0, 1.0, 3.0, 4.0]);
        assert_eq!(discharge, [1.0, 10.0, 11.0, 11.0]);
    }

    #[test]
    fn priority_flood_connects_an_enclosed_pit_without_touching_input() {
        let width = 7;
        let height = 5;
        let mut dem = vec![-1.0; width * height];
        for y in 1..height - 1 {
            for x in 0..width {
                dem[y * width + x] = 10.0;
            }
        }
        dem[2 * width + 3] = 1.0;
        let original = dem.clone();
        let runoff = vec![100.0; dem.len()];

        let solved = solve_equirectangular(config(width, height), &dem, &runoff).unwrap();

        assert_eq!(dem, original, "solver must not mutate the authored DEM");
        assert!(solved.raised_cell_count > 0);
        assert!(solved.max_fill_depth_m >= 9.0);
        assert!(
            solved
                .descending_land
                .iter()
                .all(|&i| solved.receiver[i as usize] != NO_RECEIVER)
        );
    }

    #[test]
    fn solve_is_byte_deterministic() {
        let width = 9;
        let height = 5;
        let dem: Vec<f32> = (0..width * height)
            .map(|i| {
                if i < width {
                    -1.0
                } else {
                    1.0 + (i % 7) as f32
                }
            })
            .collect();
        let runoff: Vec<f32> = (0..dem.len()).map(|i| 10.0 + (i % 5) as f32).collect();

        let a = solve_equirectangular(config(width, height), &dem, &runoff).unwrap();
        let b = solve_equirectangular(config(width, height), &dem, &runoff).unwrap();

        assert_eq!(a.receiver, b.receiver);
        assert_eq!(a.catchment_km2, b.catchment_km2);
        assert_eq!(a.discharge_m3_s, b.discharge_m3_s);
    }

    #[test]
    fn solve_conserves_spherical_area_and_runoff_at_ocean_outlets() {
        let width = 12;
        let height = 7;
        let cfg = config(width, height);
        let mut dem = vec![0.0; width * height];
        let mut runoff = vec![0.0; dem.len()];
        for y in 1..height - 1 {
            for x in 0..width {
                let i = y * width + x;
                dem[i] = 100.0 + (height - y) as f32 * 10.0 + (x % 3) as f32;
                runoff[i] = 25.0 + (x * 7 + y) as f32;
            }
        }

        let solved = solve_equirectangular(cfg, &dem, &runoff).unwrap();
        let outlet_area_km2: f64 = dem
            .iter()
            .enumerate()
            .filter(|(_, height)| **height <= 0.0)
            .map(|(i, _)| f64::from(solved.catchment_km2[i]))
            .sum();
        let outlet_discharge_m3_s: f64 = dem
            .iter()
            .enumerate()
            .filter(|(_, height)| **height <= 0.0)
            .map(|(i, _)| f64::from(solved.discharge_m3_s[i]))
            .sum();

        let dy_m = std::f64::consts::PI * cfg.planet_radius_m / height as f64;
        let mut expected_area_km2 = 0.0;
        let mut expected_discharge_m3_s = 0.0;
        for y in 0..height {
            let lat = (0.5 - (y as f64 + 0.5) / height as f64) * std::f64::consts::PI;
            let dx_m = std::f64::consts::TAU * cfg.planet_radius_m * lat.cos() / width as f64;
            let cell_area_km2 = dx_m * dy_m / 1.0e6;
            for x in 0..width {
                let i = y * width + x;
                if dem[i] > 0.0 {
                    expected_area_km2 += cell_area_km2;
                    expected_discharge_m3_s += cell_area_km2 * f64::from(runoff[i]) * 1_000.0
                        / f64::from(SECONDS_PER_YEAR);
                }
            }
        }

        let area_error = (outlet_area_km2 - expected_area_km2).abs() / expected_area_km2;
        let discharge_error =
            (outlet_discharge_m3_s - expected_discharge_m3_s).abs() / expected_discharge_m3_s;
        assert!(area_error < 1.0e-6, "relative area error {area_error}");
        assert!(
            discharge_error < 1.0e-6,
            "relative discharge error {discharge_error}"
        );
    }

    #[test]
    fn rejects_missing_ocean_outlet() {
        let dem = vec![1.0; 12];
        let runoff = vec![1.0; 12];
        assert!(matches!(
            solve_equirectangular(config(4, 3), &dem, &runoff),
            Err(HydrologyError::NoOceanSeed)
        ));
    }
}
