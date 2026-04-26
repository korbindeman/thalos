//! Topography — compose the final height field.
//!
//! Reads the plate layout + the tectonic cubemaps written by Tectonics
//! (orogen_intensity, orogen_age_myr) and composes them with multi-scale
//! fbm to produce the final `BodyBuilder::height_contributions` field.
//!
//! Composition:
//! - Per-cell *continentalness* in [0, 1]: a smooth weighted blend of all
//!   plate kinds, with weight falling off exponentially with angular
//!   distance from each plate centroid. Near the centroid of a
//!   continental plate the cell is ~fully continental (1.0); near an
//!   oceanic centroid it's ~fully oceanic (0.0); the transition at plate
//!   boundaries is a narrow soft band. This preserves the plate-identity
//!   narrative (continents track continental plates) without the
//!   half-a-texel-wide discontinuities that nearest-texel plate-id lookup
//!   produces at cube-face seams — those were turning into visible
//!   seam-lines under the shader's finite-difference normal computation.
//! - Isostatic baseline lerped by continentalness between
//!   `oceanic_baseline_m` and `continental_baseline_m`.
//! - Multi-scale regional fbm (three bands at different frequencies)
//!   sampled through an IQ-style nested domain warp. Adds continent-scale
//!   relief that can credibly override the baseline near plate margins,
//!   so coastlines trace a combined field (plate-driven + noise) and
//!   inland seas / open-ocean islands emerge where the fbm dominates.
//! - High-frequency roughness — continent-interior detail, mountain-scale
//!   relief for the shader's normal perturbation.
//! - Orogen bumps + ridge noise where `orogen_intensity > 0` — mountain
//!   belts along tectonic boundaries, rising above the regional field.
//! - Sea-level percentile normalization: subtracts the p-th percentile of
//!   the distribution so `h = 0` is the shoreline and a target fraction
//!   of cells falls below it.
//! - Near-sea-level coastal detail: second pass adding high-frequency fbm
//!   weighted by a gaussian on height-above-sea.
//!
//! See `docs/gen/thalos_processes.md §Topography`.

use rayon::prelude::*;
use serde::Deserialize;

use glam::Vec3;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::noise::fbm3;
use crate::stage::Stage;
use crate::types::PlateKind;

#[derive(Debug, Clone, Deserialize)]
pub struct Topography {
    /// Baseline height (m) at the center of a continental plate, where
    /// continentalness ≈ 1. At 0 m the plate center sits exactly at sea
    /// level; regional fbm + orogen bumps add relief on top.
    #[serde(default = "default_continental_baseline_m")]
    pub continental_baseline_m: f32,
    /// Baseline height (m) at the center of an oceanic plate, where
    /// continentalness ≈ 0. Should be negative. Regional fbm variation
    /// on top determines actual seafloor depth per cell.
    #[serde(default = "default_oceanic_baseline_m")]
    pub oceanic_baseline_m: f32,
    /// Standard deviation (in 1−cos angular units) of the gaussian kernel
    /// that weights each plate's contribution to per-cell continentalness.
    /// At `d = bandwidth` the weight has dropped to ~0.37; at `d = 2 ×
    /// bandwidth` to ~0.018. Smaller values keep continentalness tightly
    /// localised to the nearest plate; larger values smear neighbours in
    /// for a softer landmass field.
    /// Typical: 0.04 ≈ 16° ≈ 900 km (Earth-like crisp continents) up to
    /// 0.08 ≈ 23° ≈ 1300 km (very fuzzy margins).
    #[serde(default = "default_continentalness_bandwidth")]
    pub continentalness_bandwidth: f32,
    #[serde(default = "default_peak_orogen_m")]
    pub peak_orogen_m: f32,
    /// High-frequency fbm amplitude (m). Applied uniformly to every cell;
    /// continentalness does NOT gate it, because that would reintroduce a
    /// plate-kind-driven discontinuity in the height cubemap.
    #[serde(default = "default_roughness_m")]
    pub roughness_m: f32,
    /// Age (in Myr) at which orogen height has decayed to ~1/e of its
    /// active-boundary peak. 1500 Myr gives a visible split between
    /// young alpine orogens and worn-down ancient ones.
    #[serde(default = "default_orogen_age_scale_myr")]
    pub orogen_age_scale_myr: f32,
    /// fBm base frequency for continental roughness. 4.0 gives
    /// continent-scale lobes; higher values give finer topography.
    #[serde(default = "default_noise_frequency")]
    pub noise_frequency: f32,
    /// Domain-warp amplitude, in sphere radii. The plate-kind lookup
    /// direction is perturbed by a low-frequency 3-vector noise of this
    /// amplitude before the Voronoi lookup, so sharp plate edges become
    /// ragged coastlines rather than straight polygon sides. 0.0 disables
    /// warping (retains the raw Voronoi shape).
    #[serde(default = "default_warp_amplitude")]
    pub warp_amplitude: f32,
    /// Frequency (cycles per unit sphere) of the domain-warp noise. ~2-3
    /// gives continent-scale warping without over-fragmenting plates.
    #[serde(default = "default_warp_frequency")]
    pub warp_frequency: f32,
    /// Middle-frequency nested warp, between the outer and inner levels.
    /// Sampled at the inner-warped position; its output then perturbs the
    /// sampling point of the outer warp. 0.0 disables this level (fallback:
    /// two-level nesting of outer + inner).
    #[serde(default = "default_warp_amplitude_mid")]
    pub warp_amplitude_mid: f32,
    #[serde(default = "default_warp_frequency_mid")]
    pub warp_frequency_mid: f32,
    /// Innermost high-frequency nested warp. Sampled at the raw direction;
    /// its output perturbs the sampling point of the middle warp. Gives
    /// small-scale coastline wiggle that survives the outer-scale warps
    /// unchanged. 0.0 disables this level (fallback: single outer warp).
    #[serde(default = "default_warp_amplitude_hi")]
    pub warp_amplitude_hi: f32,
    #[serde(default = "default_warp_frequency_hi")]
    pub warp_frequency_hi: f32,
    /// Regional (continent-scale) mid-frequency fbm, applied to every cell
    /// regardless of plate kind. When this is comparable to or larger than
    /// the continental/oceanic baseline gap, land/ocean distribution is
    /// driven by the combined height field rather than by plate identity —
    /// so inland seas and open-ocean islands emerge naturally, and
    /// coastlines fractalise at continent scale. 0.0 disables.
    #[serde(default = "default_regional_height_m")]
    pub regional_height_m: f32,
    #[serde(default = "default_regional_frequency")]
    pub regional_frequency: f32,
    /// Very-low-frequency regional fbm (continent-scale). Decides where
    /// landmasses *broadly* sit on the sphere — the largest silhouette
    /// feature. Composes additively with `regional_height_m` and
    /// `regional_height_hi_m`. 0.0 disables.
    #[serde(default = "default_regional_height_lo_m")]
    pub regional_height_lo_m: f32,
    #[serde(default = "default_regional_frequency_lo")]
    pub regional_frequency_lo: f32,
    /// Higher-frequency regional fbm. Local subcontinent-scale variation
    /// — fills in relief between the large silhouette features. Composes
    /// additively with the lower two regional bands. 0.0 disables.
    #[serde(default = "default_regional_height_hi_m")]
    pub regional_height_hi_m: f32,
    #[serde(default = "default_regional_frequency_hi")]
    pub regional_frequency_hi: f32,
    /// High-frequency detail concentrated near sea level via a gaussian
    /// weight: `coastal_detail_m * exp(-(h/scale)^2) * fbm(freq)`. Applied
    /// in a second pass after sea-level normalization so the weight uses
    /// true height-above-sea. Produces fjords, rias, archipelago-like
    /// scatter, and coastline wiggle at pixel-scale without polluting
    /// mountains or deep ocean. 0.0 disables.
    #[serde(default = "default_coastal_detail_m")]
    pub coastal_detail_m: f32,
    #[serde(default = "default_coastal_detail_frequency")]
    pub coastal_detail_frequency: f32,
    /// Gaussian width in meters for the coastal detail falloff. Cells
    /// within ±this altitude of sea level get the full detail; farther
    /// away the weight falls off rapidly. Typical: 400–1000 m.
    #[serde(default = "default_coastal_detail_scale_m")]
    pub coastal_detail_scale_m: f32,
    /// After composing the height field, subtract the value at this
    /// percentile of the distribution so the result has sea level at 0
    /// and this fraction of cells fall below it. 0.0 disables the pass
    /// (heights remain absolute; sea level is whatever the baseline
    /// implies). Intended for Earth-like planets where a specific land
    /// fraction is the design target. Typical: 0.55–0.65.
    #[serde(default = "default_sea_level_percentile")]
    pub sea_level_percentile: f32,
}

fn default_continental_baseline_m() -> f32 {
    500.0
}
fn default_oceanic_baseline_m() -> f32 {
    -4500.0
}
fn default_continentalness_bandwidth() -> f32 {
    0.06
}
fn default_peak_orogen_m() -> f32 {
    7000.0
}
fn default_roughness_m() -> f32 {
    400.0
}
fn default_orogen_age_scale_myr() -> f32 {
    1500.0
}
fn default_noise_frequency() -> f32 {
    4.0
}
fn default_warp_amplitude() -> f32 {
    0.10
}
fn default_warp_frequency() -> f32 {
    2.5
}
fn default_warp_amplitude_mid() -> f32 {
    0.0
}
fn default_warp_frequency_mid() -> f32 {
    5.0
}
fn default_warp_amplitude_hi() -> f32 {
    0.04
}
fn default_warp_frequency_hi() -> f32 {
    8.0
}
fn default_regional_height_m() -> f32 {
    0.0
}
fn default_regional_frequency() -> f32 {
    1.5
}
fn default_regional_height_lo_m() -> f32 {
    0.0
}
fn default_regional_frequency_lo() -> f32 {
    0.4
}
fn default_regional_height_hi_m() -> f32 {
    0.0
}
fn default_regional_frequency_hi() -> f32 {
    4.0
}
fn default_coastal_detail_m() -> f32 {
    0.0
}
fn default_coastal_detail_frequency() -> f32 {
    12.0
}
fn default_coastal_detail_scale_m() -> f32 {
    800.0
}
fn default_sea_level_percentile() -> f32 {
    0.0
}

impl Stage for Topography {
    fn name(&self) -> &str {
        "topography"
    }
    fn dependencies(&self) -> &[&str] {
        &["plates", "tectonics"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let seed = builder.stage_seed();
        let res = builder.cubemap_resolution;
        let inv_res = 1.0 / res as f32;

        // Snapshot read-only data — Cloning the Plate list is cheap (tens
        // of entries); cubemap fields are accessed via disjoint-field
        // borrows in the loop below.
        let plates = builder
            .plates
            .as_ref()
            .expect("Topography requires Plates/Tectonics to run first")
            .plates
            .clone();

        let continental_base = self.continental_baseline_m;
        let oceanic_base = self.oceanic_baseline_m;
        let bandwidth = self.continentalness_bandwidth.max(1.0e-4);
        let peak_orogen = self.peak_orogen_m;
        let rough_amp = self.roughness_m;
        let age_scale_myr = self.orogen_age_scale_myr;
        let noise_freq = self.noise_frequency;
        let warp_amp = self.warp_amplitude;
        let warp_freq = self.warp_frequency;
        let warp_amp_mid = self.warp_amplitude_mid;
        let warp_freq_mid = self.warp_frequency_mid;
        let warp_amp_hi = self.warp_amplitude_hi;
        let warp_freq_hi = self.warp_frequency_hi;
        let regional_amp = self.regional_height_m;
        let regional_freq = self.regional_frequency;
        let regional_amp_lo = self.regional_height_lo_m;
        let regional_freq_lo = self.regional_frequency_lo;
        let regional_amp_hi = self.regional_height_hi_m;
        let regional_freq_hi = self.regional_frequency_hi;
        let sea_level_pct = self.sea_level_percentile;
        // Sub-seeds for fbm3 are u32. Fold the body seed's high+low
        // halves so both 32-bit halves contribute, then xor a per-band
        // magic to decorrelate fields.
        let body_seed = (seed as u32) ^ ((seed >> 32) as u32);
        // Outer warp seeds (largest scale).
        let warp_seed_x: u32 = body_seed ^ 0x5E6F_7081;
        let warp_seed_y: u32 = body_seed ^ 0x9876_5432;
        let warp_seed_z: u32 = body_seed ^ 0x89AB_CDEF;
        // Middle warp seeds.
        let warp_seed_xm: u32 = body_seed ^ 0xF3D8_B7A0;
        let warp_seed_ym: u32 = body_seed ^ 0x5C94_A721;
        let warp_seed_zm: u32 = body_seed ^ 0x29B0_F643;
        // Inner warp seeds (smallest scale).
        let warp_seed_xh: u32 = body_seed ^ 0x2E3D_5F06;
        let warp_seed_yh: u32 = body_seed ^ 0xC5D6_E708;
        let warp_seed_zh: u32 = body_seed ^ 0xEF01_2345;
        // Regional field seeds (one set per scale so the three bands are
        // decorrelated — otherwise low-freq and high-freq would share zero
        // crossings and the composition would look mono-scale).
        let regional_seed: u32 = body_seed ^ 0xABCD_EF01;
        let regional_seed_lo: u32 = body_seed ^ 0x3218_C704;
        let regional_seed_hi: u32 = body_seed ^ 0x5A32_B608;
        let roughness_seed: u32 = body_seed ^ 0x7E_E9_2D_F1;
        let coastal_seed: u32 = body_seed ^ 0x4F8D_1753;

        let oi_cm = &builder.orogen_intensity;
        let oa_cm = &builder.orogen_age_myr;
        let height = &mut builder.height_contributions.height;

        for face in CubemapFace::ALL {
            let oi = oi_cm.face_data(face);
            let oa = oa_cm.face_data(face);
            let h_out = height.face_data_mut(face);

            h_out.par_iter_mut().enumerate().for_each(|(idx, h_v)| {
                    let y = (idx as u32) / res;
                    let x = (idx as u32) % res;
                    let u = (x as f32 + 0.5) * inv_res;
                    let v = (y as f32 + 0.5) * inv_res;
                    let dir = face_uv_to_dir(face, u, v);

                    // Nested domain warp (IQ-style recursive `f(p + s(p))`
                    // where `s(p) = fbm(p + r(p))` and `r(p) = fbm(p +
                    // q(p))` and `q(p) = fbm(p)`). Three levels:
                    //   inner  — small amplitude, high frequency; perturbs
                    //            the sampling point of the middle level
                    //   middle — medium amplitude/frequency; perturbs the
                    //            sampling point of the outer level
                    //   outer  — large amplitude, low frequency; its
                    //            output IS what perturbs `dir` below.
                    // Nesting (as opposed to plain additive) is what gives
                    // multi-scale fractal structure in the regional fbm
                    // fields sampled through `warped_dir` below — peninsulas
                    // at continent scale, bays inside, inlets inside those.
                    // A level with amplitude 0 contributes nothing and the
                    // next outer level samples at the un-warped sub-input.
                    let q = if warp_amp_hi > 0.0 {
                        Vec3::new(
                            fbm3(
                                dir.x * warp_freq_hi,
                                dir.y * warp_freq_hi,
                                dir.z * warp_freq_hi,
                                warp_seed_xh,
                                2,
                                0.5,
                                2.1,
                            ),
                            fbm3(
                                dir.x * warp_freq_hi + 13.7,
                                dir.y * warp_freq_hi,
                                dir.z * warp_freq_hi,
                                warp_seed_yh,
                                2,
                                0.5,
                                2.1,
                            ),
                            fbm3(
                                dir.x * warp_freq_hi,
                                dir.y * warp_freq_hi + 23.1,
                                dir.z * warp_freq_hi,
                                warp_seed_zh,
                                2,
                                0.5,
                                2.1,
                            ),
                        ) * warp_amp_hi
                    } else {
                        Vec3::ZERO
                    };
                    let sample_mid = dir + q;
                    let r = if warp_amp_mid > 0.0 {
                        Vec3::new(
                            fbm3(
                                sample_mid.x * warp_freq_mid,
                                sample_mid.y * warp_freq_mid,
                                sample_mid.z * warp_freq_mid,
                                warp_seed_xm,
                                3,
                                0.55,
                                2.1,
                            ),
                            fbm3(
                                sample_mid.x * warp_freq_mid + 11.1,
                                sample_mid.y * warp_freq_mid,
                                sample_mid.z * warp_freq_mid,
                                warp_seed_ym,
                                3,
                                0.55,
                                2.1,
                            ),
                            fbm3(
                                sample_mid.x * warp_freq_mid,
                                sample_mid.y * warp_freq_mid + 19.4,
                                sample_mid.z * warp_freq_mid,
                                warp_seed_zm,
                                3,
                                0.55,
                                2.1,
                            ),
                        ) * warp_amp_mid
                    } else {
                        Vec3::ZERO
                    };
                    let sample_outer = dir + r;
                    let s = if warp_amp > 0.0 {
                        Vec3::new(
                            fbm3(
                                sample_outer.x * warp_freq,
                                sample_outer.y * warp_freq,
                                sample_outer.z * warp_freq,
                                warp_seed_x,
                                3,
                                0.55,
                                2.1,
                            ),
                            fbm3(
                                sample_outer.x * warp_freq + 17.3,
                                sample_outer.y * warp_freq,
                                sample_outer.z * warp_freq,
                                warp_seed_y,
                                3,
                                0.55,
                                2.1,
                            ),
                            fbm3(
                                sample_outer.x * warp_freq,
                                sample_outer.y * warp_freq + 29.7,
                                sample_outer.z * warp_freq,
                                warp_seed_z,
                                3,
                                0.55,
                                2.1,
                            ),
                        ) * warp_amp
                    } else {
                        Vec3::ZERO
                    };
                    let warped_dir = if warp_amp > 0.0
                        || warp_amp_mid > 0.0
                        || warp_amp_hi > 0.0
                    {
                        (dir + s).normalize()
                    } else {
                        dir
                    };

                    // 1. Continentalness — kernel-weighted blend of every
                    // plate's kind by angular distance from its centroid.
                    //
                    // Earlier this used `exp(−d/bandwidth)` which has a
                    // SHARP peak (non-zero derivative) at d = 0 — i.e. at
                    // each centroid. That spike combined additively with
                    // the regional fbm to print bright polygon-shaped
                    // hotspots in the height field, one per centroid.
                    //
                    // A gaussian kernel `exp(−d²/bandwidth²)` has the same
                    // smooth-blend-across-the-planet behaviour but ZERO
                    // derivative at the centroid, so the peak is flat and
                    // the height field gets no centroid spike. Other
                    // properties of the all-centroids weighted average are
                    // preserved: continentalness varies smoothly across
                    // the sphere (no plate-step discontinuities) and
                    // adjacent cells at cube-face seams have nearly-
                    // identical weights.
                    //
                    // `warped_dir` gives the blend the same nested-IQ
                    // fractal warping as the regional fbm below, so
                    // continent silhouettes remain crooked at multiple
                    // scales.
                    let inv_bw_sq = 1.0 / (bandwidth * bandwidth);
                    let mut cont_sum = 0.0f32;
                    let mut weight_sum = 0.0f32;
                    for p in plates.iter() {
                        let cos_angle = warped_dir.dot(p.centroid);
                        let angular = (1.0 - cos_angle).max(0.0);
                        let weight = (-(angular * angular) * inv_bw_sq).exp();
                        let kind_val = if p.kind == PlateKind::Continental {
                            1.0
                        } else {
                            0.0
                        };
                        cont_sum += kind_val * weight;
                        weight_sum += weight;
                    }
                    let continentalness = cont_sum / weight_sum.max(1.0e-9);

                    // 2. Isostatic baseline lerped by continentalness.
                    // Purely continuous in `warped_dir`, so no face-seam
                    // steps into the height cubemap.
                    let base = oceanic_base
                        + (continental_base - oceanic_base) * continentalness;

                    // 3. Orogen bump — applied wherever tectonic intensity
                    // is nonzero, regardless of plate kind. The intensity
                    // field is already a per-cell continuous function
                    // (distance-falloff from the nearest plate boundary),
                    // so no discrete gate is needed. In practice this
                    // means island arcs on oceanic-oceanic boundaries
                    // still rise as ridges — physically reasonable.
                    let intensity = oi[idx];
                    let orogen_bump = if intensity > 0.0 {
                        let age = oa[idx];
                        let age_factor = (-age / age_scale_myr).exp();
                        peak_orogen * intensity * age_factor
                    } else {
                        0.0
                    };

                    // 4. Roughness via fbm3. Uniform amplitude across the
                    // sphere — gating on continentalness would re-introduce
                    // a near-discrete jump at the plate boundary (the
                    // bandwidth is narrow enough that the lerp shifts
                    // roughness amplitude over a few texels, which reads
                    // as a seam). Active-orogen cells still get a boost
                    // from the `intensity * 2.0` factor, which is already
                    // continuous from Tectonics.
                    let roughness_amp = rough_amp * (1.0 + intensity * 2.0);
                    // 7 octaves extends the FBM spectrum down to ~60 km
                    // features at `noise_frequency = 4`, so mountain-scale
                    // relief shows up in the shader's height-gradient
                    // normal perturbation.
                    let n = fbm3(
                        dir.x * noise_freq,
                        dir.y * noise_freq,
                        dir.z * noise_freq,
                        roughness_seed,
                        7,
                        0.55,
                        2.1,
                    );
                    let rough = n * roughness_amp;

                    // 4. Multi-scale regional height field — three
                    // independent fbm bands (continent / regional / local)
                    // with decorrelated seeds. Sampled through `warped_dir`
                    // so the nested warp turns smooth fbm into IQ-style
                    // fractally-warped fbm — coastlines get peninsulas at
                    // continent scale, bays inside, inlets inside those,
                    // and the same warp structure informs the
                    // continentalness blend above so regional relief lines
                    // up with the warped plate-kind field rather than
                    // cutting across it.
                    let regional_lo = if regional_amp_lo > 0.0 {
                        let p = fbm3(
                            warped_dir.x * regional_freq_lo,
                            warped_dir.y * regional_freq_lo,
                            warped_dir.z * regional_freq_lo,
                            regional_seed_lo,
                            4,
                            0.55,
                            2.1,
                        );
                        p * regional_amp_lo
                    } else {
                        0.0
                    };
                    let regional_mid = if regional_amp > 0.0 {
                        let p = fbm3(
                            warped_dir.x * regional_freq,
                            warped_dir.y * regional_freq,
                            warped_dir.z * regional_freq,
                            regional_seed,
                            4,
                            0.55,
                            2.1,
                        );
                        p * regional_amp
                    } else {
                        0.0
                    };
                    let regional_hi = if regional_amp_hi > 0.0 {
                        let p = fbm3(
                            warped_dir.x * regional_freq_hi,
                            warped_dir.y * regional_freq_hi,
                            warped_dir.z * regional_freq_hi,
                            regional_seed_hi,
                            4,
                            0.55,
                            2.1,
                        );
                        p * regional_amp_hi
                    } else {
                        0.0
                    };
                    let regional = regional_lo + regional_mid + regional_hi;

                    // Ridge crest / peak structure now comes from
                    // `orogen_intensity` itself — OrogenDla stamps
                    // depth-weighted ridge geometry into that field with
                    // its own falloff, so we no longer add a
                    // `boundary_distance_km`-driven crest here. Sampling
                    // the Voronoi distance would re-embed the polygon
                    // edges the DLA pass exists to escape.

                    *h_v += base + orogen_bump + rough + regional;
                });
        }

        // Sea-level renormalization. Collect all heights, find the value at
        // the requested percentile via quickselect (O(n), vs O(n log n) for
        // a sort — at 2048² × 6 ≈ 25 M f32s this matters), and subtract it
        // from every cell. After this pass, sea level is exactly 0 and the
        // requested fraction of cells falls below. Skips entirely when
        // percentile = 0 so non-Earth-like bodies pay zero cost.
        if sea_level_pct > 0.0 {
            let height = &mut builder.height_contributions.height;
            let mut all: Vec<f32> = CubemapFace::ALL
                .iter()
                .flat_map(|f| height.face_data(*f).iter().copied())
                .collect();
            let n = all.len();
            let pct = sea_level_pct.clamp(0.0, 1.0);
            let idx = ((pct * (n as f32 - 1.0)).round() as usize).min(n - 1);
            let (_, nth, _) = all.select_nth_unstable_by(idx, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let sea_level = *nth;
            for face in CubemapFace::ALL {
                for v in height.face_data_mut(face) {
                    *v -= sea_level;
                }
            }
        }

        // Near-sea-level coastal detail pass. Adds a high-frequency fbm
        // layer weighted by a gaussian on height-above-sea, so the
        // amplitude peaks exactly at the coast and decays exponentially
        // above and below. Concentrates roughness where the coastline
        // actually is — producing fjords, rias, island scatter, and
        // pixel-scale coastline wiggle — without polluting mountain peaks
        // or deep ocean. Runs after sea-level normalization so the weight
        // uses true height-above-sea (= 0 at the coastline).
        if self.coastal_detail_m > 0.0 {
            let amp = self.coastal_detail_m;
            let freq = self.coastal_detail_frequency;
            let scale = self.coastal_detail_scale_m.max(1.0);
            let inv_scale = 1.0 / scale;

            let height = &mut builder.height_contributions.height;
            for face in CubemapFace::ALL {
                let h_out = height.face_data_mut(face);
                h_out.par_iter_mut().enumerate().for_each(|(idx, h_v)| {
                    let y = (idx as u32) / res;
                    let x = (idx as u32) % res;
                    let u = (x as f32 + 0.5) * inv_res;
                    let v = (y as f32 + 0.5) * inv_res;
                    let dir = face_uv_to_dir(face, u, v);
                    let h = *h_v;
                    let norm = h * inv_scale;
                    let weight = (-(norm * norm)).exp();
                    if weight < 1e-4 {
                        return;
                    }
                    let n = fbm3(
                        dir.x * freq,
                        dir.y * freq,
                        dir.z * freq,
                        coastal_seed,
                        5,
                        0.55,
                        2.1,
                    );
                    *h_v += n * amp * weight;
                });
            }
        }
    }
}

