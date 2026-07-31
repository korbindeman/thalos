use std::sync::Arc;

use bevy::prelude::*;
use thalos_body_render::CLOUD_BAND_COUNT;
use thalos_physics_canonical::{
    body_trajectory_provider::BodyTrajectoryProvider, canonical::Epoch, simulation::Simulation,
    types::BodyStates,
};
use thalos_terrain::DynamicSurfaceState;
use thalos_terrain::cubemap::{CubemapFace, face_uv_to_dir};
use thalos_weather::{WeatherSim, WeatherSimParams};
use thalos_world::{BodyId, CloudClimate, SolarSystemDefinition};

use crate::SimStage;

/// Central simulation state: the long-lived authority that advances time,
/// craft state, flight plans, and the active body trajectory provider.
#[derive(Resource)]
pub struct SimulationState {
    pub simulation: Simulation,
    pub system: SolarSystemDefinition,
    pub ephemeris: Arc<dyn BodyTrajectoryProvider>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudBandEnvironmentState {
    pub phases: [f64; CLOUD_BAND_COUNT],
    pub scroll_rate_rad_s: f64,
    pub differential_rotation: f64,
}

impl CloudBandEnvironmentState {
    // Constructed once banded-cloud bodies are wired at spawn (see
    // `install_cloud_band_state`); kept as the clamping constructor.
    #[allow(dead_code)]
    pub fn new(scroll_rate_rad_s: f64, differential_rotation: f64) -> Self {
        Self {
            phases: [0.0; CLOUD_BAND_COUNT],
            scroll_rate_rad_s,
            differential_rotation: differential_rotation.clamp(0.0, 1.0),
        }
    }

    pub fn advance(&mut self, dt: f64) {
        if dt == 0.0 || self.scroll_rate_rad_s.abs() < 1.0e-12 {
            return;
        }

        for i in 0..CLOUD_BAND_COUNT {
            let sin2 = i as f64 / (CLOUD_BAND_COUNT - 1) as f64;
            let lat_factor = 1.0 - self.differential_rotation * sin2;
            let omega = self.scroll_rate_rad_s * lat_factor;
            self.phases[i] = (self.phases[i] + omega * dt).rem_euclid(std::f64::consts::TAU);
        }
    }
}

/// Per-face resolution of the canonical cubemap weather field. A 90° face at
/// Thalos's 3,186 km radius spans about 5,005 km, so 1024 texels resolve
/// ~4.9 km at the face centre (the former 256 field was ~19.5 km/texel, not
/// the stale ~60 km estimate). CLOUD-3 still supplies finer 3-D shape detail.
pub const CLOUD_WEATHER_FACE_SIZE: u32 = 1024;

/// Re-export of the one coverage trim so out-of-crate consumers (the
/// `cloud_weather_probe` example) calibrate against the same value the
/// renderer uses. Defined in [`crate::rendering::clouds`].
pub const CLOUD_COVERAGE_SCALE: f32 = crate::rendering::clouds::COVERAGE_SCALE;

/// Mutable, per-body weather and broad-density authority. Both cubemap payloads
/// are stored face-major in [`CubemapFace::ALL`] order. `texels` carries RGBA8
/// coverage, cloud type, normalized base, and normalized top;
/// `surface_density_texels` carries the broad body-space shape signal at four
/// normalized-height strata. Every render projection consumes these together;
/// no projection owns an independent pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct CloudWeatherField {
    pub seed: u64,
    pub face_size: u32,
    pub texels: Vec<[u8; 4]>,
    pub surface_density_texels: Vec<[u8; 4]>,
    pub coverage_mean: f32,
    pub base_altitude_m: f32,
    pub top_altitude_m: f32,
    pub albedo: [f32; 3],
    pub wind_m_s: [f32; 2],
    /// Consumers re-upload or reject temporal history when this changes.
    pub version: u32,
}

/// One sampled texel's inputs plus the trace of its lowest stratum, for the
/// `cloud_weather_probe` diagnostic. Carries the authored coverage alongside
/// the derivation's internals so the probe can condition on it — the question
/// that matters is what the derivation emits *where coverage says clear*.
#[derive(Clone, Copy, Debug)]
pub struct WeatherTraceSample {
    pub coverage: f32,
    pub cloud_type: f32,
    pub base: f32,
    pub top: f32,
    pub trace: SurfaceDensityTrace,
}

impl CloudWeatherField {
    pub fn from_climate(climate: &CloudClimate) -> Self {
        Self::from_climate_traced(climate, 0).0
    }

    /// [`Self::from_climate`] with optional instrumentation. A non-zero
    /// `trace_stride` records one [`WeatherTraceSample`] every `trace_stride`
    /// texels (lowest stratum), which is how `cloud_weather_probe` attributes a
    /// too-cloudy planet to a specific term of the derivation instead of
    /// guessing from the emitted cube. Stride keeps the sample set to a few MB;
    /// the production path passes 0 and allocates nothing.
    pub fn from_climate_traced(
        climate: &CloudClimate,
        trace_stride: usize,
    ) -> (Self, Vec<WeatherTraceSample>) {
        let mut trace_samples: Vec<WeatherTraceSample> = Vec::new();
        let mut texel_index: usize = 0;
        let face_size = CLOUD_WEATHER_FACE_SIZE;
        let mut texels = Vec::with_capacity((face_size * face_size * 6) as usize);
        let mut surface_density_texels = Vec::with_capacity((face_size * face_size * 6) as usize);
        let mix_sum = climate.type_mix.iter().copied().sum::<f32>();
        let type_mix = if mix_sum > 1.0e-6 {
            climate.type_mix.map(|value| value.max(0.0) / mix_sum)
        } else {
            [0.25, 0.55, 0.20]
        };

        // Planet-scale circulation constants, fixed per seed. The ITCZ sits
        // OFF the equator (Earth's annual mean is ~6°N — a centered band is a
        // tell of synthetic weather), and one hemisphere's storm track is
        // stronger than the other's (Earth: the ocean hemisphere).
        let itcz_lat = 0.10 * if hash01(climate.seed, 0x17C2) < 0.5 { 1.0 } else { -1.0 };
        let storm_boost = if hash01(climate.seed, 0x570B) < 0.5 {
            [1.15, 0.85]
        } else {
            [0.85, 1.15]
        };

        // ── The synoptic layer is SIMULATED, not painted ────────────────
        // A seeded shallow-water spin-up (thalos_weather) supplies the
        // planet's synoptic organization: jets, Rossby-wave trains, cyclones
        // with wound fronts and dry slots, ITCZ convergence rain. Painted
        // approximations of this structure — curl-warped noise, analytic
        // vortex rotations, ridged-noise fronts — all converged on the same
        // marbled-fluid, spirograph verdict (2026-07-31); shapes read as
        // weather only when they are histories of an actual flow. The sim is
        // deterministic per seed; ~24 model days is past spin-up transients.
        let sim = {
            let mut sim = WeatherSim::new(WeatherSimParams {
                seed: climate.seed,
                itcz_lat_rad: itcz_lat,
                ..WeatherSimParams::default()
            });
            sim.run_days(24.0);
            sim
        };
        let (sim_nx, sim_ny) = (sim.nx(), sim.ny());
        let sim_cloud = sim.cloud();
        // Rain normalized to [0, 1] with the same scaling the sim's own cloud
        // diagnostic uses: strong precipitation marks fronts and the ITCZ.
        let sim_rain: Vec<f32> = {
            let raw = sim.precip_field();
            let mut out = vec![0.0f32; raw.len()];
            for j in 0..sim_ny {
                let qs = sim.qsat_of_row(j);
                for i in 0..sim_nx {
                    let idx = j * sim_nx + i;
                    out[idx] = smoothstep(0.45, 1.2, raw[idx] / (2.8e-6 * qs));
                }
            }
            out
        };
        // Bilinear equirect sampler (zonal wrap, meridional clamp).
        let sample_sim = |map: &[f32], dir: Vec3| -> f32 {
            let lon = dir.z.atan2(dir.x);
            let lat = dir.y.clamp(-1.0, 1.0).asin();
            let xf = (lon / std::f32::consts::TAU + 0.5) * sim_nx as f32 - 0.5;
            let yf = ((lat / std::f32::consts::PI) + 0.5) * sim_ny as f32 - 0.5;
            let yf = yf.clamp(0.0, (sim_ny - 1) as f32);
            let y0 = yf.floor() as usize;
            let y1 = (y0 + 1).min(sim_ny - 1);
            let ty = yf - y0 as f32;
            let xw = xf.rem_euclid(sim_nx as f32);
            let x0 = xw.floor() as usize % sim_nx;
            let x1 = (x0 + 1) % sim_nx;
            let tx = xw - xw.floor();
            let a = map[y0 * sim_nx + x0] + (map[y0 * sim_nx + x1] - map[y0 * sim_nx + x0]) * tx;
            let b = map[y1 * sim_nx + x0] + (map[y1 * sim_nx + x1] - map[y1 * sim_nx + x0]) * tx;
            a + (b - a) * ty
        };

        for face in CubemapFace::ALL {
            for y in 0..face_size {
                let v = (y as f32 + 0.5) / face_size as f32;
                for x in 0..face_size {
                    let u = (x as f32 + 0.5) / face_size as f32;
                    let dir_raw = face_uv_to_dir(face, u, v).normalize();
                    // ── Simulated synoptic fields ────────────────────────
                    // The sim's cloud field carries the system-scale
                    // organization (Rossby-wave trains, cyclones with dry
                    // slots, jets); its rain field marks fronts and the ITCZ.
                    // Only the SYNOPTIC scale comes from the sim: the
                    // mesoscale and cell-scale texture stay noise layers in
                    // the UNWARPED domain — an organized flow carrying a
                    // granular convective texture, never a marbled fluid.
                    // (That anti-marble split survives from the warp era; the
                    // warps themselves are gone because painted rotations of
                    // static noise read as spirograph, however tuned —
                    // 2026-07-31 verdict, see thalos_weather's module notes.)
                    let sim_occ_raw = sample_sim(&sim_cloud, dir_raw);
                    let rain = sample_sim(&sim_rain, dir_raw);
                    let dir = dir_raw;
                    let dir_meso = dir_raw;
                    // Meteorological banding is a *bias*, not a paint ring:
                    // warp the latitude the profile reads (so band edges
                    // meander like jet streams) and gate its strength with a
                    // continental-scale field (so bands appear as broken
                    // segments, not complete rings). The straight gaussian
                    // rings previously survived the far projection's coverage
                    // threshold as conspicuous synthetic stripes.
                    let band_warp = 0.55
                        * (fbm3(
                            dir * 3.1 + Vec3::new(7.0, -3.0, 11.0),
                            climate.seed ^ 0xBA9D,
                            3,
                        ) - 0.5);
                    // The gate keeps a floor: a belt may thin to half strength
                    // regionally but never vanish — at the old 0.35 floor whole
                    // quadrants lost their climatology and the disc read as one
                    // uniform texture at every latitude.
                    let band_gate = (0.55
                        + 0.90
                            * fbm3(
                                dir * 2.1 + Vec3::new(-2.0, 9.0, 5.0),
                                climate.seed ^ 0x6A7E,
                                3,
                            ))
                    .clamp(0.0, 1.0);
                    // The ITCZ is a train of convective clusters, not a solid
                    // line: a zonally-stretched cluster field turns the band
                    // into beads-on-a-string with gaps of comparable size.
                    let itcz_beads = 0.15
                        + 1.10
                            * smoothstep(
                                0.32,
                                0.64,
                                fbm3(
                                    Vec3::new(dir.x * 2.4, dir.y * 7.0, dir.z * 2.4)
                                        + Vec3::new(9.0, 17.0, -4.0),
                                    climate.seed ^ 0x17C2,
                                    3,
                                ),
                            );
                    // The meander is a JET feature: mid-latitude belt edges
                    // wander ±16° with it, but the ITCZ holds within a few
                    // degrees of its mean latitude — warped at full amplitude
                    // the 5°-wide band scattered into unconnected patches and
                    // never read as a line (probe round 4).
                    let lat_raw = dir.y.clamp(-1.0, 1.0).asin();
                    let meander_scale = 0.22 + 0.78 * smoothstep(0.18, 0.55, lat_raw.abs());
                    let lat_banded = lat_raw + band_warp * meander_scale;
                    let itcz_gauss = {
                        let z = (lat_banded - itcz_lat) / 0.09;
                        (-z * z).exp()
                    };
                    let itcz = itcz_beads * itcz_gauss;
                    let band = zonal_climatology(lat_banded, itcz, storm_boost) * band_gate;
                    // Fronts come from the sim's precipitation: long thin
                    // bands of strong rain along convergence lines, wound
                    // around the lows that made them. The ridged-noise front
                    // field this replaces drew the same worm filament at
                    // every latitude, which is exactly the painted look the
                    // sim exists to kill.
                    let frontal = rain;
                    // Coverage needs two meteorological scales. The synoptic
                    // component establishes planetary bands and fronts while
                    // the mesoscale component breaks those systems into the
                    // distinct cells that the same field must expose both to
                    // the near volume and to an orbital projection. Cloud type
                    // remains categorical and must not be abused as a shape
                    // mask: doing so produces conspicuous cubemap-sized blocks.
                    // Mesoscale is partially advected (`dir_meso`); cell scale
                    // is not advected at all (`dir_raw`) — see the domain-split
                    // note at the warp.
                    let mesoscale_warp = fbm3(
                        dir_meso * 17.0 + Vec3::new(13.0, -4.0, 29.0),
                        climate.seed ^ 0x5CA1_0A11,
                        3,
                    ) - 0.5;
                    let mesoscale = fbm3(
                        dir_meso * 52.0
                            + Vec3::new(-11.0, 23.0, 7.0)
                            + Vec3::splat(2.4 * mesoscale_warp),
                        climate.seed ^ 0x5CA1_E5CA,
                        4,
                    );
                    let cellular_mass = fbm3(
                        dir_raw * 128.0
                            + Vec3::new(19.0, -5.0, 37.0)
                            + Vec3::splat(-3.1 * mesoscale_warp),
                        climate.seed ^ 0xCE11_C10D,
                        3,
                    );
                    let cellular_cut_raw = fbm3(
                        dir_raw * 211.0 + Vec3::new(-31.0, 47.0, 5.0),
                        climate.seed ^ 0xCE11_5EED,
                        2,
                    );
                    let cellular_cut = 1.0 - (2.0 * cellular_cut_raw - 1.0).abs();
                    let cellular = 0.68 * cellular_mass + 0.32 * cellular_cut;

                    // ── Regime-structured occupancy (BL-20260723T165923Z) ──
                    // Real skies are organized, not statistically uniform: a
                    // synoptic OCCUPANCY field thresholded into weather systems
                    // with genuinely clear air between them, and a coherent
                    // REGIME per region (scattered-cumulus field / stratus
                    // sheet / storm cluster, plus frontal ridges) that sets the
                    // local coverage texture, cloud type, and vertical extent.
                    // The previous producer summed fixed-scale noises around
                    // one mean, which rendered the whole planet as the same
                    // mid-cumulus speckle (2026-07-23 user verdict).
                    //
                    // Occupancy: threshold the synoptic field at the quantile
                    // matching authored mean coverage; soft edges so systems
                    // thin out rather than shear off.
                    // The sim's honest domain ends at its polar sponge
                    // (~66°); poleward, occupancy hands over to a broken
                    // stratus-sheet climatology (real polar caps run 0.6–0.7
                    // cloud fraction — the first wired cube rendered them
                    // pitch black).
                    let polar_occ = 0.52 + 0.55 * (mesoscale - 0.5);
                    let sim_occ = sim_occ_raw
                        + (polar_occ - sim_occ_raw) * smoothstep(1.05, 1.30, lat_raw.abs());
                    // The synoptic driver is the SIM's cloud field, centered
                    // so the authored-coverage quantile mapping below keeps
                    // its meaning (sim mean ~0.38 → recentered near 0.5). The
                    // climatology band rides on top as a bias — belts, clear
                    // subtropics, the beaded ITCZ — and the mesoscale term
                    // breaks system edges below sim resolution.
                    let system_field = 0.12
                        + sim_occ
                        + 1.2 * climate.band_strength * band
                        + 0.08 * (mesoscale - 0.5);
                    let occ_threshold = 0.70 - 0.36 * climate.coverage.clamp(0.0, 1.0);
                    // The gate's transition has to be wide relative to the
                    // synoptic field's own spread, or it acts as a binary cut:
                    // at the original ±0.08 it was narrower than one fbm
                    // standard deviation, and 43% of the planet came out at
                    // exactly zero coverage while the rest sat near saturation
                    // — the "either caked or nothing" verdict. Widening it all
                    // the way to ±0.21 went too far the other way and left the
                    // planet with no clear sky at all, so a system now thins
                    // out over a realistic few-hundred-km margin and no more.
                    let system_edge =
                        smoothstep(occ_threshold - 0.13, occ_threshold + 0.15, system_field);
                    // No part of a real terrestrial planet is empty. Even the
                    // subtropical highs — the clearest air there is — carry
                    // broken shallow cumulus, and that is what fills the space
                    // between systems in the reference imagery. Deliberately a
                    // SPARSE population (low coverage, full optical depth), so
                    // it reads as scattered cells rather than the grey wash
                    // that scaling optical depth would give.
                    let fair_weather = 0.20
                        * smoothstep(0.52, 0.84, 0.58 * cellular + 0.42 * mesoscale)
                        * (0.40 + 0.60 * system_edge);
                    // ITCZ convective clusters occupy DIRECTLY: deep tropical
                    // convection is driven by surface convergence, not by the
                    // mid-latitude synoptic state, and routing it through the
                    // shared threshold left the band visible only where
                    // `regional` already happened to sit near the cut — no
                    // coherent line at any additive weight (probe rounds 3–5).
                    let itcz_occ = smoothstep(0.45, 0.85, itcz);
                    let occupancy = system_edge.max(fair_weather).max(itcz_occ);
                    // 0 at a system's fringe, 1 deep inside: deep systems are
                    // more developed (storm potential, higher fill).
                    // ITCZ clusters and strongly precipitating frontal bands
                    // count as developed systems: deep convection shares the
                    // storm/depth pathway the synoptic cores use.
                    let intensity = smoothstep(occ_threshold + 0.02, occ_threshold + 0.22, system_field)
                        .max(0.6 * itcz_occ)
                        .max(0.55 * rain);

                    // Regime selector: an independent low-frequency partition,
                    // uniformized so the authored type_mix reads as area
                    // fractions of the cloudy world. Storms additionally need a
                    // developed system core.
                    let regime = fbm3(
                        dir * 3.6 + Vec3::new(23.0, 5.0, -12.0),
                        climate.seed ^ 0xC0DE,
                        3,
                    );
                    let regime_x = smoothstep(0.36, 0.64, regime);
                    let m_stratus = type_mix[0].clamp(0.0, 0.9);
                    let m_storm = type_mix[2].clamp(0.0, 0.9);
                    let stratus_region = smoothstep(
                        1.0 - m_stratus - 0.06,
                        (1.0 - m_stratus + 0.06).min(1.0),
                        regime_x,
                    );
                    let storm_region = (1.0
                        - smoothstep((m_storm - 0.06).max(0.0), m_storm + 0.06, regime_x))
                        * (0.30 + 0.70 * intensity);
                    let cumulus_region = (1.0 - stratus_region - storm_region).max(0.0);

                    // Per-regime coverage texture. `variation` scales how deep
                    // the mesoscale/cellular breakup cuts.
                    let breakup = (0.55 + climate.variation).clamp(0.5, 1.5);
                    let cell_broken = smoothstep(0.42, 0.78, 0.62 * cellular + 0.38 * mesoscale);
                    let sheet_holes = smoothstep(0.66, 0.84, cellular);
                    let storm_core = smoothstep(0.52, 0.78, 0.66 * mesoscale + 0.34 * cellular);
                    let frontal_boost = frontal * occupancy;
                    let cumulus_cov = (0.66 - 0.52 * breakup * (1.0 - cell_broken)
                        + 0.10 * intensity)
                        .clamp(0.0, 1.0);
                    let stratus_cov = 0.94 - 0.30 * breakup * sheet_holes;
                    let storm_cov = (0.40 + 0.46 * storm_core + 0.10 * intensity).clamp(0.0, 1.0);
                    let cov_regime = stratus_region * stratus_cov
                        + cumulus_region * cumulus_cov
                        + storm_region * storm_cov;
                    // The coverage channel is a smooth SYNOPTIC field: the local
                    // probability that a place is cloudy, an areal fraction. It
                    // is deliberately NOT a realized cloud field — the single
                    // realization point is `cloud_surface_density_cpu`, which
                    // thresholds the cell-scale shape noise against exactly
                    // this value. Realizing it here as well (tried 2026-07-25)
                    // binarizes the field twice, and stacked thresholds turn
                    // every partly-cloudy region into harsh salt-and-pepper
                    // dither instead of broken cloud.
                    // (Dry slots need no explicit carve-out any more — the
                    // sim's subsidence drying digs them into `sim_occ`.)
                    let deck_cov = (occupancy * cov_regime + 0.28 * frontal_boost).clamp(0.0, 1.0);

                    // ── High thin veil (cirrus / cirrostratus) ───────────
                    // The single largest visual difference between this
                    // producer and orbital photography of a real terrestrial
                    // planet: high ice cloud is WIDE, streaky, optically thin,
                    // and it survives over the very regions where the low deck
                    // is absent. Without it a planet reads as opaque systems
                    // scattered on a bare ball.
                    //
                    // It carries no new texture channel — the weather cube's
                    // RGBA is full. A veil is expressible in the channels that
                    // already exist as exactly what it physically is: a thin
                    // sheet (stratus-shaped vertical profile) with a HIGH base
                    // and a small strata amplitude. Amplitude, not opacity, is
                    // the knob: the far tier converts occupancy to opacity
                    // through the derived response LUT, so a low-amplitude
                    // stratum renders as a translucent veil in both tiers
                    // without anyone multiplying a coverage term into optical
                    // depth (which is the rejected grey-wash failure — see
                    // `weather_column_from_texel`).
                    //
                    // Cirrus is sheared far harder than the deck: it lives in
                    // the jet, so its filaments are strongly zonal and it
                    // trails downstream of the rain bands (the `frontal` bias
                    // below is sim rain).
                    let veil_dir = zonal_shear(
                        zonal_shear(dir_raw, climate.seed ^ 0xC144_05EE),
                        climate.seed ^ 0xC144_1EE5,
                    );
                    let veil_field = fbm3(
                        Vec3::new(veil_dir.x * 2.4, veil_dir.y * 6.0, veil_dir.z * 2.4)
                            + Vec3::new(-17.0, 41.0, 8.0),
                        climate.seed ^ 0xC144_1005,
                        4,
                    );
                    // Outflow: real cirrus streams off the tops of deep
                    // convection and ahead of warm fronts, so the veil is
                    // biased toward — but deliberately offset from — the
                    // deck's own systems.
                    let veil_raw = veil_field + 0.22 * frontal + 0.18 * intensity;
                    // Strongly selective. A veil is a feature of PARTICULAR
                    // regions — jet-stream bands, warm-front approaches, storm
                    // outflow — and the single most damaging thing it can do is
                    // become a planet-wide film. At a low threshold it covered
                    // most of the disc, and a thin grey wash over everything
                    // desaturated the ocean and muted the cloud tops at the same
                    // time: an A/B with the veil zeroed showed deep blue ocean
                    // and white cloud tops returning immediately (2026-07-25).
                    // It has to read as streaks over a clear planet, never as
                    // atmosphere.
                    let veil_cov =
                        0.22 * smoothstep(0.68, 0.92, veil_raw) * (0.55 + 0.45 * band_gate);
                    // Union the two decks: where the low deck is solid the veil
                    // is invisible behind it, so it only adds where there is
                    // sky left to fill.
                    let veil_add = veil_cov * (1.0 - deck_cov);
                    let coverage = (deck_cov + veil_add).clamp(0.0, 1.0);
                    // 1 = this texel is pure veil, 0 = pure low deck. Drives
                    // the geometry blend and the strata amplitude below.
                    let veil_share = (veil_add / coverage.max(1.0e-4)).clamp(0.0, 1.0);

                    // Cloud type follows the regime: sheets read stratus, storm
                    // clusters read cumulonimbus at their cores, and building
                    // cells inside deep cumulus fields turn congestus. Fronts
                    // push toward storm so ridge lines carry tall cloud.
                    // Building cells appear in ordinary cumulus fields too, not
                    // only deep systems — within-horizon vertical hierarchy is
                    // the main local monotony breaker (2026-07-23 round 2).
                    let congestus = storm_core * (0.30 + 0.70 * intensity);
                    let deck_type = stratus_region * 0.08
                        + cumulus_region * (0.42 + 0.30 * congestus)
                        + storm_region * (0.78 + 0.18 * storm_core)
                        + 0.10 * frontal_boost;
                    // A veil is a sheet, so it reads as the stratus end of the
                    // type channel — that is what selects the flat-profile
                    // vertical response in all three density implementations.
                    let cloud_type = deck_type.lerp(0.04, veil_share).clamp(0.02, 0.97);
                    let vertical_noise = fbm3(
                        dir_meso * 34.0 + Vec3::new(31.0, -7.0, 13.0),
                        climate.seed ^ 0xA11E,
                        3,
                    );
                    // Local base/top are fractions of the authored shell, per
                    // regime: thin low stratus decks, cumulus growing with its
                    // cells, storm towers claiming most of the thickness so
                    // limb silhouettes keep height.
                    let base_stratus = 0.09 + 0.05 * vertical_noise;
                    let base_cumulus = 0.10 + 0.07 * vertical_noise;
                    let base_storm = 0.05 + 0.04 * vertical_noise;
                    let top_stratus = base_stratus + 0.10 + 0.08 * vertical_noise;
                    // Ordinary cumulus fields develop real depth (round 7):
                    // the former 0.09 baseline gave plain fair-weather
                    // columns <1 km of a 10.5 km shell — everything below
                    // congestus rendered as a squat sheet. Building cells now
                    // carry more of the growth so broken fields read as
                    // mixed-height puffs rather than one flat deck.
                    // Round 8 (2026-07-30): **judge cumulus depth against the
                    // CELL WIDTH, not against the shell.** Cells here are
                    // 5.4 km across (`CELL_PERIOD_M`), so the round-7 envelope
                    // — measured p50 1,647 m — drew every median column at an
                    // aspect ratio near 0.3, i.e. a pancake, and read flat no
                    // matter how well the dome sculpted its top. That is why
                    // the "clouds are flat" verdict survived round 7's top
                    // sculpting: the sculpting was working (58.9% of columns
                    // reached their own top) on columns with nothing to sculpt.
                    // Convective cells are roughly as tall as they are wide;
                    // this puts the median near 3 km (aspect ~0.55) and leaves
                    // the growth term to carry congestus past 6 km.
                    // `cloud_weather_probe`'s HEIGHT table is the check —
                    // re-measure there rather than adjusting these by eye.
                    let top_cumulus = base_cumulus
                        + 0.24
                        + 0.78 * (0.42 * cell_broken + 0.58 * congestus)
                        + 0.09 * vertical_noise;
                    let top_storm = 0.60 + 0.38 * storm_core;
                    let deck_base = stratus_region * base_stratus
                        + cumulus_region * base_cumulus
                        + storm_region * base_storm;
                    let deck_top = (stratus_region * top_stratus
                        + cumulus_region * top_cumulus
                        + storm_region * top_storm)
                        .max(deck_base + 0.04);
                    // Veil geometry: a thin sheet high in the shell. It must
                    // clear the deck's tops or the two would render as one
                    // fused mass on the limb.
                    let veil_base = 0.74 + 0.09 * vertical_noise;
                    let veil_top = veil_base + 0.07 + 0.05 * vertical_noise;
                    let base = deck_base.lerp(veil_base, veil_share);
                    let top = deck_top.lerp(veil_top, veil_share).max(base + 0.04);
                    // Canonical surface-space broad shape. These fields live on
                    // the unit direction sphere, so they are seamless across
                    // cubemap faces and never inherit the near volume's small
                    // Cartesian repeat. Four correlated strata let towers lean
                    // and split with height without storing a full 3-D shell.
                    // Coverage/type/base/top remain separate climate controls;
                    // shaders apply their shared threshold/profile contract.
                    // Cell-scale shape stays in the UNWARPED domain: this is the
                    // field that has to read as individual puffy cloud bodies,
                    // and advecting it is what turned round cells into ribbons.
                    //
                    // It is also UNSTYLED: see the note on the deleted
                    // `styled_domain` — a per-place domain transform here has
                    // the same decorrelating phase gradient that the near
                    // tier's varying period did.
                    let shape_warp = fbm3(
                        dir_raw * 41.0 + Vec3::new(43.0, -17.0, 9.0),
                        climate.seed ^ 0x5A11_FACE,
                        2,
                    ) - 0.5;
                    let shape_mass = fbm3(
                        dir_raw * 128.0
                            + Vec3::new(-37.0, 61.0, 23.0)
                            + Vec3::splat(7.5 * shape_warp),
                        climate.seed ^ 0xD315_17A1,
                        3,
                    );
                    let shape_cut_raw = fbm3(
                        dir_raw * 211.0
                            + Vec3::new(71.0, -29.0, 53.0)
                            + Vec3::splat(-5.0 * shape_warp),
                        climate.seed ^ 0xCE11_B0D1,
                        2,
                    );
                    let shape_cut = 1.0 - (2.0 * shape_cut_raw - 1.0).abs();
                    // Cluster-scale octave. This is the field the one formation
                    // threshold below cuts against, so it is where a system's
                    // INTERNAL structure comes from — whether a 500 km cloud
                    // mass is a featureless slab or a field of distinct cloud
                    // clusters. Every octave in the producer used to top out at
                    // 211 cycles (~95 km features) on a cube whose texels are
                    // 4.9 km, so systems had no interior at all and rendered as
                    // smooth washes.
                    //
                    // The ceiling is a viewing limit, not the cube's: a
                    // planet-disc pixel is ~6 km, and structure below ~30 km
                    // lands on 2–3 pixels and reads as dither rather than as
                    // cloud. Real cumulus fields dodge this by being far
                    // smaller than a pixel and averaging into smooth haze;
                    // content sitting exactly at pixel scale is the worst case.
                    // fbm doubles per octave, so the TOP octave sets the limit
                    // — this bottoms out at ~31 km.
                    let shape_cluster = fbm3(
                        dir_raw * 160.0
                            + Vec3::new(53.0, -11.0, 29.0)
                            + Vec3::splat(4.0 * shape_warp),
                        climate.seed ^ 0xCE11_5CA1,
                        3,
                    );
                    // Weights stay convex per stratum: the formation threshold
                    // is compared against these levels directly, so a sum that
                    // drifts off ~0.5 mean silently re-biases planetary
                    // coverage.
                    let surface_shape = [
                        0.40 * shape_mass
                            + 0.24 * shape_cluster
                            + 0.20 * shape_cut
                            + 0.16 * cellular,
                        0.34 * shape_mass
                            + 0.22 * shape_cluster
                            + 0.21 * shape_cut
                            + 0.13 * regime_x
                            + 0.10 * cellular,
                        0.28 * shape_mass
                            + 0.18 * shape_cluster
                            + 0.20 * shape_cut
                            + 0.19 * vertical_noise
                            + 0.15 * regime_x,
                        0.23 * shape_mass
                            + 0.14 * shape_cluster
                            + 0.18 * shape_cut
                            + 0.26 * vertical_noise
                            + 0.19 * regime_x,
                    ];
                    // Strata are LAYER-RELATIVE: four samples across the local
                    // [base, top] interval, not the whole shell. Fixed shell
                    // heights had a dead zone — a ~2 km deck sitting between
                    // two sampling heights read zero from every stratum, so
                    // the far tier showed clear sky over a solid near-volume
                    // deck (2026-07-23). Consumers map their shell height
                    // through the same weather base/top channels.
                    let top_c = top.max(base + 0.02);
                    // Ice cloud holds a small fraction of the condensate a
                    // convective column does. Scaling the stored OCCUPANCY is
                    // the honest way to say so here: the consumers' response
                    // curve turns occupancy into opacity, so a veil comes out
                    // translucent in both tiers without anyone multiplying
                    // coverage into optical depth.
                    let amplitude = 1.0_f32.lerp(0.26, veil_share);
                    let surface_density = [0.125, 0.375, 0.625, 0.875].map(|q| {
                        amplitude
                            * cloud_surface_density_cpu(
                                surface_shape,
                                base + q * (top_c - base),
                                coverage,
                                cloud_type,
                                base,
                                top_c,
                            )
                    });
                    if trace_stride > 0 && texel_index % trace_stride == 0 {
                        trace_samples.push(WeatherTraceSample {
                            coverage,
                            cloud_type,
                            base,
                            top: top_c,
                            trace: cloud_surface_density_traced(
                                surface_shape,
                                base + 0.125 * (top_c - base),
                                coverage,
                                cloud_type,
                                base,
                                top_c,
                            ),
                        });
                    }
                    texel_index += 1;
                    let encode = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    texels.push([
                        encode(coverage),
                        encode(cloud_type),
                        encode(base),
                        encode(top_c),
                    ]);
                    surface_density_texels.push(surface_density.map(encode));
                }
            }
        }

        (
            Self {
                seed: climate.seed,
                face_size,
                texels,
                surface_density_texels,
                coverage_mean: climate.coverage.clamp(0.0, 1.0),
                base_altitude_m: climate.base_altitude_m.max(0.0),
                top_altitude_m: (climate.base_altitude_m + climate.thickness_m).max(0.0),
                albedo: climate.albedo,
                wind_m_s: climate.wind_m_s,
                version: 0,
            },
            trace_samples,
        )
    }

    /// Number of mip levels the weather cube carries (1024 → 8 px faces).
    /// Far projections select a level from their projected footprint; without
    /// this chain the mesoscale/cellular coverage aliases into ring/speckle
    /// moiré at disc scale.
    pub const MIP_LEVELS: u32 = 8;

    /// Full RGBA8 cube payload with a box-filtered mip chain, laid out
    /// layer-major (face0[mip0..], face1[mip0..], …) to match wgpu's
    /// `TextureDataOrder::LayerMajor` default used by Bevy's image uploads.
    pub fn rgba8_mip_chain(&self) -> Vec<u8> {
        rgba8_cube_mip_chain(&self.texels, self.face_size, Self::MIP_LEVELS)
    }

    /// Full four-stratum surface-density cube payload with the same layout and
    /// mip contract as [`Self::rgba8_mip_chain`].
    pub fn surface_density_rgba8_mip_chain(&self) -> Vec<u8> {
        rgba8_cube_mip_chain(
            &self.surface_density_texels,
            self.face_size,
            Self::MIP_LEVELS,
        )
    }
}

fn rgba8_cube_mip_chain(texels: &[[u8; 4]], face_size: u32, mip_levels: u32) -> Vec<u8> {
    let size = face_size as usize;
    let face_texels = size * size;
    assert_eq!(texels.len(), 6 * face_texels, "cloud cubemap texel count");
    let mut out = Vec::new();
    for face in 0..6 {
        let base = &texels[face * face_texels..(face + 1) * face_texels];
        let mut level: Vec<[u8; 4]> = base.to_vec();
        let mut level_size = size;
        out.extend(level.iter().flatten());
        for _ in 1..mip_levels {
            let next_size = (level_size / 2).max(1);
            let mut next = Vec::with_capacity(next_size * next_size);
            for y in 0..next_size {
                for x in 0..next_size {
                    let mut acc = [0u32; 4];
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let sy = (y * 2 + dy).min(level_size - 1);
                        let sx = (x * 2 + dx).min(level_size - 1);
                        let t = level[sy * level_size + sx];
                        for c in 0..4 {
                            acc[c] += u32::from(t[c]);
                        }
                    }
                    next.push([
                        (acc[0] / 4) as u8,
                        (acc[1] / 4) as u8,
                        (acc[2] / 4) as u8,
                        (acc[3] / 4) as u8,
                    ]);
                }
            }
            out.extend(next.iter().flatten());
            level = next;
            level_size = next_size;
        }
    }
    out
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(f32::EPSILON)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// NOTE (2026-07-26): a `styled_domain` helper here applied the cell field's
// per-place anisotropy to the cube's shape lookups, so the far tier's texture
// would point the same way as the near tier's cells. It is DELETED, not tuned:
// it carried the same phase-gradient defect as the varying period in
// `cell_field` — a spatially varying domain scale on a lookup at frequency
// 128–211 decorrelates the field between neighbouring texels instead of
// deforming it. Far/near registration of cell ORIENTATION is a nice-to-have on
// a tier that CLOUD-6 already tracks as a smooth wash; it is not worth
// reintroducing the defect the near tier just had removed. If it comes back, it
// must use the same fix the shader uses — blend between constant transforms.

/// Intermediates of [`cloud_surface_density_cpu`], for the `cloud_weather_probe`
/// diagnostic. The emitted strata cube is what every far/orbital projection
/// renders, so when the planet reads cloudier than its authored coverage the
/// question is always *which* term lifted it — the formation threshold, the
/// shape field it is compared against, or the areal realization. Reading that
/// off the final density alone is impossible; this makes each term observable
/// without a second copy of the math to keep in lockstep.
#[derive(Clone, Copy, Debug, Default)]
pub struct SurfaceDensityTrace {
    /// Normalized height within the local `[base, top]` deck.
    pub h: f32,
    /// Coverage after `COVERAGE_SCALE`, i.e. the authored areal fraction.
    pub cov: f32,
    /// Reconstructed vertical shape-noise value at this height.
    pub shape: f32,
    /// Formation threshold `shape` must clear for cloud to exist here.
    pub threshold: f32,
    /// Height-rising dome-sculpting term subtracted from `mass`.
    pub vertical_narrow: f32,
    /// `shape - threshold - vertical_narrow` (after the anvil branch).
    pub mass: f32,
    /// Sub-texel areal fraction: the share of the texel that is cloudy.
    pub areal_fraction: f32,
    /// Type-blended vertical profile applied on top of the areal fraction.
    pub vertical_profile: f32,
    /// The emitted value — `areal_fraction * vertical_profile`.
    pub density: f32,
}

/// CPU producer for the canonical four-stratum density payload. Nonlinear
/// formation/profile response happens before the mip chain is built, so far
/// footprints average occupied area instead of thresholding an averaged raw
/// signal into salt-and-pepper cloud pixels.
fn cloud_surface_density_cpu(
    surface_shape: [f32; 4],
    normalized_height: f32,
    coverage: f32,
    cloud_type: f32,
    local_base: f32,
    local_top: f32,
) -> f32 {
    cloud_surface_density_traced(
        surface_shape,
        normalized_height,
        coverage,
        cloud_type,
        local_base,
        local_top,
    )
    .density
}

/// [`cloud_surface_density_cpu`] with its intermediates exposed. The production
/// path routes through this, so there is exactly one copy of the derivation.
pub fn cloud_surface_density_traced(
    surface_shape: [f32; 4],
    normalized_height: f32,
    coverage: f32,
    cloud_type: f32,
    local_base: f32,
    local_top: f32,
) -> SurfaceDensityTrace {
    let h = (normalized_height - local_base) / (local_top - local_base).max(0.02);
    let cov = (coverage * crate::rendering::clouds::COVERAGE_SCALE).clamp(0.0, 1.0);
    if h <= 0.0 || h >= 1.0 || cov <= 1.0e-3 {
        return SurfaceDensityTrace {
            h,
            cov,
            ..Default::default()
        };
    }

    let stratus_w = 1.0 - smoothstep(0.18, 0.38, cloud_type);
    let storm_w = smoothstep(0.72, 0.88, cloud_type);
    let cumulus_w = (1.0 - stratus_w - storm_w).max(0.0);
    // Formation threshold, calibrated so the emitted column occupancy TRACKS
    // the coverage channel (`cloud_weather_probe` reports the ratio). This
    // matters because the far tier renders the strata cube, not coverage.
    //
    // **DERIVED, not hand-fitted** — `cloud_weather_probe`'s THRESHOLD FIT
    // table prints it. For occupancy to equal coverage `c`, the threshold must
    // be the (1-c) quantile of the `shape - vertical_narrow` comparand, and
    // the probe measures exactly that per coverage decile and least-squares
    // fits this line. Re-derive it there after ANY change to `surface_shape`
    // or the vertical terms; do not nudge these constants by eye (hand-fitting
    // has failed here twice — see BL-20260723T214730Z).
    //
    // The 1.03 → 0.33 line this replaces was co-tuned with the anvil mass
    // floor below, and only made sense in its presence: `shape` has a NARROW
    // distribution (σ ≈ 0.09 about 0.562), so a threshold sweeping 1.03 → 0.33
    // sits almost entirely outside the field's support. Nothing formed on its
    // own merits and the 0.5 floor supplied the planet's cloud instead. With
    // the floor removed that line emitted 0.34× the authored coverage.
    let threshold = 0.675 + (0.458 - 0.675) * cov;
    // Round-7 dome sculpting: convective tops are carved by a quadratically
    // height-rising threshold (a per-lobe noise isosurface — strong lobes
    // tower, weak lobes stay squat) instead of the former linear thinning;
    // tall congestus/storm columns keep more mass with height so towers stay
    // coherent. Mirrored in `get_cloud_map_density` (clouds_compute.wgsl) and
    // `march_column` (fill_lut.rs) — keep the three in lockstep.
    // **Scale these against `shape`'s spread, not in the abstract.** This term
    // is a height-rising addition to the effective threshold, so what decides
    // how hard it carves is its size in units of σ(`shape`) ≈ 0.09 — that is
    // what converts a threshold shift into an occupancy change.
    //
    // The former 0.04/0.42/0.30 were scaled against a threshold line spanning
    // 1.03 → 0.33. At h→1 the cumulus term reached 0.42 ≈ **4.7σ**, which
    // drives top-stratum occupancy to ~0: every column lost its upper half and
    // the deck rendered as flat pancakes with no vertical development. That was
    // survivable only while the anvil mass floor (INC-20260729T061228Z) was
    // pinning every stratum at 0.5 regardless.
    //
    // Rescaled by 0.31 for the derived threshold line's 0.675 → 0.458 span.
    // Two independent routes agree on that factor: the threshold-range ratio
    // (0.217 / 0.70) and the σ argument (want ~1.4σ at the top, so 0.42 →
    // 0.13). Occupancy now tapers ~32% → ~6% from base to top stratum, which
    // is a dome rather than a cut. Keep the RATIOS between the three
    // coefficients — they are the authored per-type shape; only the common
    // scale is derived.
    let column_tall = smoothstep(0.30, 0.65, local_top - local_base);
    let vertical_narrow = h * 0.012 * stratus_w
        + (h * h) * (0.130 * cumulus_w + 0.093 * storm_w) * (1.0 - 0.45 * column_tall);

    // C1 reconstruction of the four vertical shape samples, matching
    // `cloud_surface_shape` (thalos::atmosphere) and `surface_shape`
    // (fill_lut.rs) — keep the three in lockstep. The piecewise-linear form
    // this replaces broke slope at the knots, and a deck viewed edge-on
    // rendered those breaks as horizontal shelves (2026-07-26).
    let z = (normalized_height.clamp(0.0, 1.0) * 4.0 - 0.5).clamp(0.0, 3.0);
    let k = z.floor().clamp(0.0, 2.0);
    let t = z - k;
    let spline = |p0: f32, p1: f32, p2: f32, p3: f32| {
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    };
    let shape = if k < 1.0 {
        spline(
            surface_shape[0],
            surface_shape[0],
            surface_shape[1],
            surface_shape[2],
        )
    } else if k < 2.0 {
        spline(
            surface_shape[0],
            surface_shape[1],
            surface_shape[2],
            surface_shape[3],
        )
    } else {
        spline(
            surface_shape[1],
            surface_shape[2],
            surface_shape[3],
            surface_shape[3],
        )
    }
    .clamp(0.0, 1.0);
    let mut mass = shape - threshold - vertical_narrow;
    // Cumulonimbus anvils broaden again near the tropopause, but only where the
    // storm channel permits them.
    //
    // **Never fold the gate into the value and then use it as a floor.** The
    // shipped form was `mass.max(anvil_shape * anvil_profile * storm_w)`, and
    // outside an anvil that product is exactly 0.0 — so the line degraded to
    // `mass.max(0.0)` and `mass` could not go negative anywhere on the planet.
    // `anvil_profile` is zero for the lower strata, so it fired at essentially
    // every height outside storm columns. The realization below is a smoothstep
    // CENTRED on zero, so a floor of exactly 0.0 lands on its midpoint and
    // emitted `areal_fraction = 0.5` for clear sky, planet-wide: an orbital
    // cloud floor no climate trim or downstream response curve could remove
    // (INC-20260729T*-orbital-cloud-floor). The near marcher and `fill_lut`
    // carried the same `max` but realize with `smoothstep(0.0, …)`, where a
    // zero floor maps to zero density — which is why the defect was invisible
    // from the surface and catastrophic from orbit.
    //
    // Blending by the gate cannot clamp at any gate value: 0 leaves `mass`
    // untouched, 1 applies the full anvil floor. Mirrored in
    // `get_cloud_map_density` (clouds_compute.wgsl) and `march_column`
    // (fill_lut.rs) — keep the three in lockstep.
    let anvil_profile = smoothstep(0.62, 0.76, h) * (1.0 - smoothstep(0.90, 1.0, h));
    let anvil_gate = anvil_profile * storm_w;
    let anvil_mass = shape - (threshold - 0.06);
    mass += (mass.max(anvil_mass) - mass) * anvil_gate;

    // NOTE (2026-07-25): surface-space boundary erosion was tried here — fine
    // noise added to `mass`, gated on proximity to the threshold, to rough up
    // the smooth oval silhouettes that value-noise fbm produces. It does not
    // work at this cube resolution, at any strength or gate width tested: the
    // gate must be narrow relative to `shape`'s spread to stay a boundary
    // effect, but at 4.9 km texels a narrow band is only 1–2 texels wide, so
    // the perturbation is at texel frequency and renders as planet-wide
    // salt-and-pepper stipple rather than ragged edges. It also pushed the
    // strata/coverage calibration to 1.29. Silhouette detail belongs to the
    // consumers' sub-texel morphology, not to this cube.

    let bottom_softness = 0.16;
    // Thin condensation top skins (the dome term above owns top shape);
    // stratus stays a genuine sheet. Lockstep with the marcher + fill_lut.
    let stratus_profile =
        smoothstep(0.0, bottom_softness * 0.45, h) * (1.0 - smoothstep(0.72, 1.0, h));
    let cumulus_profile =
        smoothstep(0.0, bottom_softness * 0.75, h) * (1.0 - smoothstep(0.93, 1.0, h));
    let storm_profile =
        smoothstep(0.0, bottom_softness * 0.35, h) * (1.0 - smoothstep(0.94, 1.0, h));
    let vertical_profile =
        stratus_profile * stratus_w + cumulus_profile * cumulus_w + storm_profile * storm_w;
    // ── Areal fraction, not an isosurface ────────────────────────────────
    // This value is what a whole ~5 km weather texel contains, and cloud
    // cells are 1–8 km across, so the honest answer is a FRACTION: the share
    // of the texel's area whose shape noise clears the formation threshold.
    // The former `smoothstep(0.0, 0.055, mass)` asked a different question —
    // "is the noise above threshold at this one point" — and 0.055 is far
    // narrower than the sub-texel spread of `shape`, so every texel answered
    // 0 or 1. That is the whole "either fully caked or no clouds in sight"
    // failure: coverage moved the threshold (how much AREA is cloudy) but
    // could never move the amount, so a 30 %-coverage region rendered as 30 %
    // of texels at full opacity instead of a broken field.
    //
    // Widening the transition to the sub-texel RMS of `shape` is the analytic
    // form of supersampling the threshold test — same expected value, ~1/9th
    // the noise evaluations of a 3×3 sub-sample, and it stays smooth so the
    // mip chain has something to filter. Coverage keeps its single meaning
    // (areal occupancy); optical depth is untouched, so this cannot regress
    // into the rejected grey-veil look (`weather_column_from_texel`).
    //
    // NOT in lockstep with the marcher's `edge_softness`: that one shapes
    // sub-texel lobes from a point sample and must stay sharp. This one
    // integrates over the texel. The near tier reads the result through its
    // spawn-derived formation curve, which re-fits to whatever this emits.
    // Width is the sub-texel RMS of `shape`, and it wants to stay SMALL.
    //
    // A real cloud field is locally high-contrast and globally graded: an
    // individual cell is optically thick with clear air beside it, while the
    // cloudy *fraction* varies smoothly over hundreds of km. Both failure
    // modes seen on 2026-07-25 come from confusing those two scales. The
    // original 0.055 was fine here — the binariness was never at cell scale,
    // it was the synoptic occupancy gate upstream. Widening this to 0.12–0.19
    // to compensate put every texel inside the transition, so nothing was
    // opaque anywhere and the planet rendered as a flat grey film: the exact
    // grey-veil signature `weather_column_from_texel` warns about, reached
    // from the occupancy side instead of the optical-depth side.
    //
    // It is not zero, though: at a 5 km texel and 1–8 km cells the honest
    // answer is still a fraction, and a hard step quantizes the smooth
    // upstream field back into 5 km blocks (the crumb-edged look). This is
    // wide enough to keep the mip chain something to filter and no wider.
    const SUB_TEXEL_RMS: f32 = 0.035;
    let areal_fraction = smoothstep(-SUB_TEXEL_RMS, SUB_TEXEL_RMS, mass);
    SurfaceDensityTrace {
        h,
        cov,
        shape,
        threshold,
        vertical_narrow,
        mass,
        areal_fraction,
        vertical_profile,
        density: areal_fraction * vertical_profile,
    }
}

/// Latitude-dependent rotation about the spin axis.
///
/// Zonal wind is the dominant organizing motion of a rotating atmosphere: it
/// stretches everything east–west and, because it varies with latitude, tilts
/// features into the leaning bands and trailing fronts of the reference
/// imagery. Applied as a domain rotation (not a coverage bias) so it deforms
/// structure instead of painting stripes — the failure mode the band profile
/// already carries a warning about.
fn zonal_shear(dir: Vec3, seed: u64) -> Vec3 {
    let sin_lat = dir.y.clamp(-1.0, 1.0);
    // Mid-latitude jets in each hemisphere, weaker easterlies at the equator.
    let jet = 1.35 * sin_lat * (1.0 - sin_lat * sin_lat) - 0.28 * sin_lat.abs();
    // Meander so the shear itself is not a perfect function of latitude.
    let meander = 0.35 * (fbm3(dir * 2.3 + Vec3::new(-9.0, 4.0, 17.0), seed, 2) - 0.5);
    let angle = jet + meander;
    let (sin_a, cos_a) = angle.sin_cos();
    Vec3::new(
        dir.x * cos_a + dir.z * sin_a,
        dir.y,
        -dir.x * sin_a + dir.z * cos_a,
    )
}

/// Zonal cloud climatology: the W-curve every terrestrial full-disc image
/// shows. Three cloudy belts — a sharp ITCZ (deliberately OFF the equator;
/// `itcz_lat` carries the seeded offset, ~±6°), two broad mid-latitude storm
/// tracks — separated by deep clear subtropical belts, with a mild polar
/// decline. Numbers follow the satellite climatology shape: subtropical
/// minima near 24°, storm-track maxima near 53°, and one hemisphere's track
/// stronger than the other's (`storm_boost` = [north, south] scale).
///
/// `itcz_beads` multiplies only the ITCZ term: the real band is a train of
/// convective clusters, and a solid bright ring is the giveaway of a painted
/// climatology.
///
/// `lat` is signed radians (callers pass the jet-meander-warped latitude so
/// belt edges wander instead of tracing perfect circles).
fn zonal_climatology(lat: f32, itcz: f32, storm_boost: [f32; 2]) -> f32 {
    let gauss =
        |x: f32, center: f32, width: f32| (-((x - center) / width) * ((x - center) / width)).exp();
    let a = lat.abs();
    let hemisphere = if lat >= 0.0 {
        storm_boost[0]
    } else {
        storm_boost[1]
    };
    // The ITCZ reads as a band only because the trades around it are CLEAR:
    // a broad equatorial suppression (the trade-cumulus belts, sparse by
    // nature) with the sharp beaded peak riding on top. Without the
    // suppression the deep tropics grew full-strength synoptic worms and the
    // band vanished into them (probe round 3). `itcz` arrives precomputed
    // (beads × line gaussian) because the caller also feeds it into occupancy
    // directly — see the note there.
    // Polar decline is MILD: real polar caps carry 0.6–0.7 broken stratus,
    // well above the subtropical minima — the caps must not go dark.
    1.35 * itcz + 0.95 * hemisphere * gauss(a, 0.92, 0.24) - 0.40 * gauss(a, 0.0, 0.22)
        - 0.90 * gauss(a, 0.40, 0.17)
        - 0.10 * gauss(a, std::f32::consts::FRAC_PI_2, 0.22)
}

/// Small deterministic uniform in [0, 1) keyed off the climate seed — the
/// per-planet circulation constants (ITCZ side, storm-track asymmetry) come
/// from here so they are stable per seed and independent of the texel loop.
fn hash01(seed: u64, key: u64) -> f32 {
    let mut h = seed ^ key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0x00FF_FFFF) as f32 / 16_777_216.0
}

fn hash3(p: IVec3, seed: u64) -> f32 {
    let mut h = (p.x as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (p.y as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (p.z as i64 as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ seed;
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0x00FF_FFFF) as f32 / 16_777_216.0
}

fn value_noise3(p: Vec3, seed: u64) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let i = i.as_ivec3();
    let corner = |dx: i32, dy: i32, dz: i32| hash3(i + IVec3::new(dx, dy, dz), seed);
    let x00 = corner(0, 0, 0) + (corner(1, 0, 0) - corner(0, 0, 0)) * u.x;
    let x10 = corner(0, 1, 0) + (corner(1, 1, 0) - corner(0, 1, 0)) * u.x;
    let x01 = corner(0, 0, 1) + (corner(1, 0, 1) - corner(0, 0, 1)) * u.x;
    let x11 = corner(0, 1, 1) + (corner(1, 1, 1) - corner(0, 1, 1)) * u.x;
    let y0 = x00 + (x10 - x00) * u.y;
    let y1 = x01 + (x11 - x01) * u.y;
    y0 + (y1 - y0) * u.z
}

fn fbm3(p: Vec3, seed: u64, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 0.5;
    let mut norm = 0.0;
    let mut q = p;
    for _ in 0..octaves {
        sum += amplitude * value_noise3(q, seed);
        norm += amplitude;
        amplitude *= 0.5;
        q = q * 2.03 + Vec3::new(13.1, 7.7, 19.3);
    }
    sum / norm.max(f32::EPSILON)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BodyEnvironmentState {
    /// Mutable runtime state for terrain-owned dynamic layers: seasonal ice,
    /// active dunes, and later weather/tide-driven surface overlays.
    pub dynamic_surface: Option<DynamicSurfaceState>,
    /// Atmospheric cloud-band motion and phases. Kept here, not on render
    /// components, so map impostors, ship impostors, terrain skies, and future
    /// weather systems all see the same cloud state.
    pub cloud_bands: Option<CloudBandEnvironmentState>,
    /// Canonical large-scale volumetric-cloud weather. `None` mirrors an
    /// authored `CloudClimate::None`; renderers must not install defaults.
    pub cloud_weather: Option<CloudWeatherField>,
}

/// Canonical evaluated solar-system state for the current game frame.
///
/// This is the source that projections consume. Bevy entities, impostor
/// materials, terrain tile providers, map snapshots, and atmosphere passes may
/// cache derived data, but they should not independently evaluate or own body
/// state. Future wind, storms, tides, and dune migration belong in
/// [`BodyEnvironmentState`] so every projection reads the same runtime
/// environment for a body.
///
/// **Sole writer:** [`sync_solar_system_state`] (in [`SimStage::Sync`]). All
/// other systems read it; environment mutators go through `environment_mut`.
#[derive(Resource, Debug, Default)]
pub struct SolarSystemState {
    pub states: Option<BodyStates>,
    pub time: f64,
    pub environment: Vec<BodyEnvironmentState>,
}

impl SolarSystemState {
    pub fn environment_mut(&mut self, body_id: BodyId) -> Option<&mut BodyEnvironmentState> {
        self.environment.get_mut(body_id)
    }

    fn ensure_body_capacity(&mut self, body_count: usize) {
        if self.environment.len() < body_count {
            self.environment
                .resize_with(body_count, BodyEnvironmentState::default);
        }
    }

    // Forward environment-install API, ready for spawn-time wiring:
    // `install_cloud_band_state` lights up the `update_cloud_bands` drift
    // loop the moment a body is given cloud bands. Kept symmetric with the live
    // `install_cloud_weather`.
    #[allow(dead_code)]
    pub fn install_cloud_band_state(&mut self, body_id: BodyId, state: CloudBandEnvironmentState) {
        self.ensure_body_capacity(body_id + 1);
        self.environment[body_id].cloud_bands = Some(state);
    }

    pub fn install_cloud_weather(&mut self, body_id: BodyId, state: CloudWeatherField) {
        self.ensure_body_capacity(body_id + 1);
        self.environment[body_id].cloud_weather = Some(state);
    }
}

pub fn sync_solar_system_state(
    sim: Res<SimulationState>,
    mut solar_system: ResMut<SolarSystemState>,
) {
    let epoch = Epoch(sim.simulation.sim_time());
    if solar_system.states.is_some() && (solar_system.time - epoch.0).abs() < f64::EPSILON {
        return;
    }

    if let Some(states) = solar_system.states.as_mut() {
        sim.ephemeris.states_into(epoch, states);
    } else {
        let mut states = Vec::with_capacity(sim.ephemeris.body_count());
        sim.ephemeris.states_into(epoch, &mut states);
        solar_system.states = Some(states);
    }
    solar_system.time = epoch.0;
    solar_system.ensure_body_capacity(sim.ephemeris.body_count());
}

pub struct SolarSystemStatePlugin;

impl Plugin for SolarSystemStatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SolarSystemState>()
            .add_systems(Update, sync_solar_system_state.in_set(SimStage::Sync));
    }
}

#[cfg(test)]
mod cloud_site_probe {
    use super::*;

    /// Dev probe for the BL-20260723T214730Z thickness-parity protocol: scan
    /// the authored Thalos weather field for *cloudy* sites near the runway's
    /// daylight longitude, so tier A/B captures can frame real cloud (the
    /// default spaceport column is authored nearly clear). Prints
    /// `THALOS_RUNWAY_SITE` candidates.
    ///
    /// Run: `cargo test -p thalos_runtime --lib cloud_site_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "dev probe: prints cloudy THALOS_RUNWAY_SITE candidates"]
    fn print_cloudy_sites() {
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets");
        let system = thalos_world::parsing::load_solar_system_from_dir(&assets)
            .expect("load authored solar system");
        let thalos_id = system.name_to_id["Thalos"];
        let climate = system.bodies[thalos_id]
            .terrestrial_atmosphere
            .as_ref()
            .and_then(|atmosphere| atmosphere.clouds.clone())
            .expect("Thalos authored cloud climate");
        let field = CloudWeatherField::from_climate(&climate);

        // 2°x2° bins over the runway's daylight window (lon near 178°).
        const LAT_MIN: f32 = -45.0;
        const LAT_MAX: f32 = 45.0;
        const LON_MIN: f32 = 150.0;
        const LON_MAX: f32 = 206.0;
        const BIN_DEG: f32 = 2.0;
        let lat_bins = ((LAT_MAX - LAT_MIN) / BIN_DEG) as usize;
        let lon_bins = ((LON_MAX - LON_MIN) / BIN_DEG) as usize;
        #[derive(Clone, Copy, Default)]
        struct Bin {
            n: u32,
            cov: f64,
            sd_col: f64,
            cloudy: u32,
        }
        let mut bins = vec![Bin::default(); lat_bins * lon_bins];
        let size = field.face_size as usize;
        for (face_index, face) in CubemapFace::ALL.into_iter().enumerate() {
            for y in (0..size).step_by(2) {
                let v = (y as f32 + 0.5) / size as f32;
                for x in (0..size).step_by(2) {
                    let u = (x as f32 + 0.5) / size as f32;
                    let dir = face_uv_to_dir(face, u, v).normalize();
                    let lat = dir.y.asin().to_degrees();
                    let lon = dir.z.atan2(dir.x).to_degrees().rem_euclid(360.0);
                    if !(LAT_MIN..LAT_MAX).contains(&lat) || !(LON_MIN..LON_MAX).contains(&lon) {
                        continue;
                    }
                    let bin = &mut bins[((lat - LAT_MIN) / BIN_DEG) as usize * lon_bins
                        + ((lon - LON_MIN) / BIN_DEG) as usize];
                    let index = face_index * size * size + y * size + x;
                    let weather = field.texels[index];
                    let strata = field.surface_density_texels[index];
                    let cov = f64::from(weather[0]) / 255.0;
                    let col = strata
                        .iter()
                        .map(|&s| f64::from(s) / 255.0)
                        .fold(0.0f64, f64::max);
                    bin.n += 1;
                    bin.cov += cov;
                    bin.sd_col += col;
                    bin.cloudy += u32::from(cov > 0.25);
                }
            }
        }

        // Rank by "broken moderate field" suitability: mean column strata near
        // 0.42 with substantial (but not total) cloudy-texel fraction.
        let mut ranked: Vec<(f32, f32, f64, f64, f64)> = Vec::new();
        for (i, bin) in bins.iter().enumerate() {
            if bin.n < 32 {
                continue;
            }
            let lat = LAT_MIN + (i / lon_bins) as f32 * BIN_DEG + BIN_DEG * 0.5;
            let lon = LON_MIN + (i % lon_bins) as f32 * BIN_DEG + BIN_DEG * 0.5;
            let n = f64::from(bin.n);
            ranked.push((
                lat,
                lon,
                bin.cov / n,
                bin.sd_col / n,
                f64::from(bin.cloudy) / n,
            ));
        }
        ranked.sort_by(|a, b| {
            // Broken moderate field wanted: real cloudy texels, mid strata.
            let score = |r: &(f32, f32, f64, f64, f64)| (r.3 - 0.42).abs() - 0.6 * r.4.min(0.6);
            score(a).total_cmp(&score(b))
        });

        // Local sun elevation at the runway morning boot epoch, so candidates
        // are known-daylit before spending a cold capture on them.
        use thalos_physics_canonical::body_trajectory_provider::BodyTrajectoryProvider;
        let provider =
            thalos_physics_canonical::patched_conics::PatchedConics::new(&system, 3.156e11);
        let states = provider.states(Epoch(59_100.0));
        let star = states.first().map(|s| s.position).unwrap_or_default();
        let thalos_state = &states[thalos_id];
        let sun_elevation_deg = |lat_deg: f32, lon_deg: f32| -> f64 {
            let lat = f64::from(lat_deg).to_radians();
            let lon = f64::from(lon_deg).to_radians();
            let dir_body =
                bevy::math::DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin());
            let up_world = thalos_state.orientation * dir_body;
            let to_sun =
                (star - (thalos_state.position + up_world * thalos_state.radius_m)).normalize();
            90.0 - up_world.angle_between(to_sun).to_degrees()
        };

        println!("lat, lon, mean_cov, mean_col_strata, cloudy_frac, sun_elev_deg");
        for (lat, lon, cov, sd, cloudy) in ranked.iter().take(30) {
            println!(
                "{lat:7.1} {lon:7.1}   {cov:5.3}   {sd:5.3}   {cloudy:5.3}   {:6.1}",
                sun_elevation_deg(*lat, *lon)
            );
        }
        // Reference: the default runway site's bin.
        let default_bin = &bins[((7.6 - LAT_MIN) / BIN_DEG) as usize * lon_bins
            + ((178.0 - LON_MIN) / BIN_DEG) as usize];
        if default_bin.n > 0 {
            let n = f64::from(default_bin.n);
            println!(
                "default site (7.6, 178.0): cov {:5.3} col_strata {:5.3} cloudy_frac {:5.3}",
                default_bin.cov / n,
                default_bin.sd_col / n,
                f64::from(default_bin.cloudy) / n,
            );
        }
    }

    /// Dev probe: run the shared fill derivation on the real Thalos field and
    /// print the fitted curve + far response, without booting a capture.
    /// The per-bin convergence table lands in the log output (init a
    /// subscriber below so it prints).
    ///
    /// Run: `cargo test -p thalos_runtime --lib derive_fill -- --ignored --nocapture`
    #[test]
    #[ignore = "dev probe: prints the derived cloud fill calibration"]
    fn derive_fill_calibration_probe() {
        let subscriber = tracing_subscriber_fmt();
        let _guard = tracing::subscriber::set_default(subscriber);
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets");
        let system = thalos_world::parsing::load_solar_system_from_dir(&assets)
            .expect("load authored solar system");
        let thalos_id = system.name_to_id["Thalos"];
        let body = &system.bodies[thalos_id];
        let climate = body
            .terrestrial_atmosphere
            .as_ref()
            .and_then(|atmosphere| atmosphere.clouds.clone())
            .expect("Thalos authored cloud climate");
        let field = CloudWeatherField::from_climate(&climate);
        let start = std::time::Instant::now();
        let calibration = crate::rendering::derive_body_fill_calibration_for_probe(
            &field,
            &climate,
            body.radius_m as f32,
        );
        println!(
            "derived in {:?}: threshold nodes {:?}\nfar_response {:?}",
            start.elapsed(),
            calibration.threshold_nodes,
            calibration.far_response,
        );

        // Cross-check the CPU mirror against the pixel-measured tier A/B at
        // the measurement site (22.0 N, 153.0 E, ~15 km crop).
        let lat = 22.0f32.to_radians();
        let lon = 153.0f32.to_radians();
        let site = Vec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin());
        let climate_bottom = climate.base_altitude_m.max(0.0);
        let input = thalos_body_render::FillCalibrationInput {
            weather_texels: &field.texels,
            strata_texels: &field.surface_density_texels,
            face_size: field.face_size,
            coverage_scale: crate::rendering::clouds::COVERAGE_SCALE,
            density: 0.0026 * climate.density.max(0.0),
            detail_strength: 0.16,
            base_edge_softness: 0.055,
            bottom_softness: 0.16,
            base_shape_scale_m: climate.base_shape_scale_m.max(500.0),
            detail_scale_m: climate.detail_scale_m.max(50.0),
            bottom_height_m: climate_bottom,
            top_height_m: (climate.base_altitude_m + climate.thickness_m).max(climate_bottom + 1.0),
            planet_radius_m: body.radius_m as f32,
            seed: field.seed,
        };
        for radius_km in [8.0f32, 20.0, 60.0] {
            let cos_radius = (radius_km * 1000.0 / body.radius_m as f32).cos();
            let stats = thalos_body_render::fill_lut::predict_region_fill(
                &input,
                &calibration,
                site,
                cos_radius,
                4000,
            );
            println!("site prediction r={radius_km} km: {stats:?}");
        }
    }

    fn tracing_subscriber_fmt() -> impl tracing::Subscriber + Send + Sync {
        use bevy::log::tracing_subscriber::{self, layer::SubscriberExt};
        tracing_subscriber::registry().with(tracing_subscriber::fmt::layer())
    }
}
