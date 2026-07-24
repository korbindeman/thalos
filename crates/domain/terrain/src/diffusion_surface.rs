//! NTR-X2a — the terrain-diffusion surface backing for Thalos.
//!
//! Geometry comes from the terrain-diffusion reference pipeline's own bands,
//! exported by `thalos_export.py` (terrain-diffusion checkout) **conditioned
//! on Thalos's canonical macro terrain** (`export_thalos_macro.rs`), so the
//! continents are the game's own geography rendered in the model's coastline
//! and relief language:
//!
//! - **Planetary band** — the model's coarse chart (23.04 km/px at the
//!   equator) as a global equirect raster, plus chart-conditioned sub-Nyquist
//!   analytic octaves (higher reference terrain → stronger hills; no content
//!   above the chart's Nyquist is invented). Ocean is ours: the released
//!   coarse band clamps the sea to 0, so bathymetry is a smoothed-landmask
//!   shelf toward the abyss — which also preserves the Thalos "shoreline at
//!   height 0" datum convention exactly.
//! - **Regional band** — the model's native 90 m detail output around the
//!   spaceport site, applied as a residual against the planetary band
//!   (mip-matched: it vanishes into the parent at coarse footprints), edge
//!   feathered.
//! - **Fine band** — sub-model octaves conditioned by the model's own local
//!   relief energy inside the detail window (rugged model terrain gets rugged
//!   fine detail, its plains stay smooth), by chart relief outside.
//!
//! Landcover stays canonical: albedo / moisture / climate come from the inner
//! [`ProceduralSurface`]'s own fields via
//! [`ProceduralSurface::macro_albedo_for`], driven by the diffusion height —
//! one coloring model regardless of geometry backing, and the biome geography
//! the game already has (the conditioning makes the two agree).
//!
//! Band-composition invariant (ADR-20260722T105147Z part 3): every band is
//! footprint-gated / mip-matched, so refinement adds bandwidth only —
//! ported from the probe's `field.rs`, which the FT-2 benchmark verified
//! reproduces the flat reference statistics.

use std::path::Path;

use glam::{DVec3, Vec3};

use crate::procedural::{MacroBiome, ProceduralSurface};
use crate::query::{Region, SurfaceQuery, SurfaceSample};

// --- deterministic gradient noise (probe field.rs, verbatim) -----------------

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn cell_gradient(ix: i64, iy: i64, iz: i64, seed: u64) -> DVec3 {
    let h = splitmix(
        seed ^ (ix as u64).wrapping_mul(0x8da6_b343)
            ^ (iy as u64).wrapping_mul(0xd816_3841)
            ^ (iz as u64).wrapping_mul(0xcb1a_b31f),
    );
    let a = (h & 0xffff) as f64 / 65535.0 * core::f64::consts::TAU;
    let z = ((h >> 16) & 0xffff) as f64 / 65535.0 * 2.0 - 1.0;
    let r = (1.0 - z * z).max(0.0).sqrt();
    DVec3::new(r * a.cos(), r * a.sin(), z)
}

fn smootherstep(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn grad_noise(p: DVec3, seed: u64) -> f64 {
    let base = p.floor();
    let f = p - base;
    let (ix, iy, iz) = (base.x as i64, base.y as i64, base.z as i64);
    let corner = |dx: i64, dy: i64, dz: i64| -> f64 {
        let g = cell_gradient(ix + dx, iy + dy, iz + dz, seed);
        g.dot(f - DVec3::new(dx as f64, dy as f64, dz as f64))
    };
    let (u, v, w) = (smootherstep(f.x), smootherstep(f.y), smootherstep(f.z));
    let mut acc = [[0.0f64; 2]; 2];
    for dy in 0..2i64 {
        for dz in 0..2i64 {
            let a = corner(0, dy, dz);
            let b = corner(1, dy, dz);
            acc[dy as usize][dz as usize] = a + (b - a) * u;
        }
    }
    let y0 = acc[0][0] + (acc[0][1] - acc[0][0]) * w;
    let y1 = acc[1][0] + (acc[1][1] - acc[1][0]) * w;
    (y0 + (y1 - y0) * v) * 1.9
}

/// Footprint gate: wavelength λ contributes fully when resolved (λ ≥ 4·fp),
/// not at all when λ ≤ 2·fp.
fn footprint_gate(wavelength_m: f64, footprint_m: f64) -> f64 {
    if footprint_m <= 0.0 {
        return 1.0;
    }
    let t = ((wavelength_m / footprint_m - 2.0) / 2.0).clamp(0.0, 1.0);
    smootherstep(t)
}

// --- rasters -------------------------------------------------------------------

/// A W×H raster with a box-filtered mip chain, bilinear at a footprint-matched
/// mip. `wrap_x` for the global equirect chart (longitude wraps).
struct Raster {
    width: usize,
    height: usize,
    px_m: f64,
    wrap_x: bool,
    mips: Vec<(usize, usize, Vec<f32>)>,
}

impl Raster {
    fn from_data(data: Vec<f32>, width: usize, height: usize, px_m: f64, wrap_x: bool) -> Self {
        assert_eq!(width * height, data.len(), "raster dims mismatch");
        let mut mips = vec![(width, height, data)];
        let (mut w, mut h) = (width, height);
        while w > 8 && h > 4 {
            let (nw, nh) = (w / 2, h / 2);
            let prev = &mips.last().unwrap().2;
            let mut next = vec![0f32; nw * nh];
            for y in 0..nh {
                for x in 0..nw {
                    next[y * nw + x] = 0.25
                        * (prev[(2 * y) * w + 2 * x]
                            + prev[(2 * y) * w + (2 * x + 1).min(w - 1)]
                            + prev[(2 * y + 1).min(h - 1) * w + 2 * x]
                            + prev[(2 * y + 1).min(h - 1) * w + (2 * x + 1).min(w - 1)]);
                }
            }
            mips.push((nw, nh, next));
            w = nw;
            h = nh;
        }
        Self { width, height, px_m, wrap_x, mips }
    }

    fn load(path: &Path, width: usize, height: usize, px_m: f64, wrap_x: bool) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() != width * height * 4 {
            return Err(format!(
                "{}: {} bytes, expected {}x{} f32",
                path.display(),
                raw.len(),
                width,
                height
            ));
        }
        let mut data = Vec::with_capacity(width * height);
        for c in raw.chunks_exact(4) {
            data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        Ok(Self::from_data(data, width, height, px_m, wrap_x))
    }

    /// Bilinear sample at fractional mip-0 pixel coords, footprint-matched.
    fn sample_px(&self, px: f64, py: f64, footprint_m: f64) -> f64 {
        let mip = if footprint_m <= self.px_m {
            0
        } else {
            ((footprint_m / self.px_m).log2().floor() as usize).min(self.mips.len() - 1)
        };
        let scale = (1usize << mip) as f64;
        let (w, h, data) = &self.mips[mip];
        let fx = px / scale - 0.5;
        let fy = (py / scale - 0.5).clamp(0.0, (*h - 1) as f64);
        let (y0, ty) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(h - 1);
        let (x0f, tx) = (fx.floor(), fx - fx.floor());
        let (x0, x1) = if self.wrap_x {
            let m = *w as i64;
            let a = ((x0f as i64) % m + m) % m;
            (a as usize, ((a + 1) % m) as usize)
        } else {
            let a = x0f.clamp(0.0, (*w - 1) as f64) as usize;
            (a, (a + 1).min(w - 1))
        };
        data[y0 * w + x0] as f64 * (1.0 - tx) * (1.0 - ty)
            + data[y0 * w + x1] as f64 * tx * (1.0 - ty)
            + data[y1 * w + x0] as f64 * (1.0 - tx) * ty
            + data[y1 * w + x1] as f64 * tx * ty
    }
}

/// Naive JSON number grab (sidecar files are machine-written; a serde
/// dependency is not warranted for two fields).
fn json_num(json: &str, key: &str) -> Option<f64> {
    json.split(&format!("\"{key}\"")).nth(1).and_then(|rest| {
        rest.trim_start_matches([':', ' ', '['])
            .split(|c: char| c == ',' || c == '}' || c == ']' || c == '\n')
            .next()?
            .trim()
            .parse()
            .ok()
    })
}

// --- the surface ------------------------------------------------------------------

struct DetailWindow {
    raster: Raster,
    /// |mip0 − mip2|: the model's own 90–360 m relief energy, conditioning the
    /// fine band.
    rough: Raster,
    site_dir: DVec3,
    east: DVec3,
    north: DVec3,
}

/// Analytic gap-filler octaves (probe `field.rs`, verbatim amplitudes — the
/// FT-2 benchmark tuned these against the reference statistics).
const OCTAVES: [(f64, f64); 12] = [
    (6_000_000.0, 1_800.0),
    (3_000_000.0, 1_100.0),
    (1_400_000.0, 700.0),
    (700_000.0, 450.0),
    (350_000.0, 280.0),
    (160_000.0, 170.0),
    (75_000.0, 100.0),
    (34_000.0, 60.0),
    (15_000.0, 35.0),
    (6_500.0, 20.0),
    (2_800.0, 12.0),
    (1_200.0, 7.0),
];
/// First octave below the coarse chart's ~46 km Nyquist — what analytic detail
/// may add on chart land without double-counting chart content.
const SUB_CHART_OCTAVE0: usize = 7;

/// Sub-model fine band (below the 90 m detail resolution down to mesh scale).
const FINE_OCTAVES: [(f64, f64); 4] = [(700.0, 7.0), (300.0, 4.0), (130.0, 2.4), (55.0, 1.5)];

/// Geometry from the diffusion bands, landcover from the canonical
/// [`ProceduralSurface`]. See the module docs.
pub struct DiffusionSurface {
    radius_m: f64,
    chart: Raster,
    landmask: Raster,
    detail: Option<DetailWindow>,
    /// Canonical climate/landcover/albedo authority (geometry unused).
    landcover: ProceduralSurface,
    seed: u64,
    height_range_m: f32,
    /// FNV-1a over the raster payloads — cache-namespace identity.
    pub content_fingerprint: u64,
}

impl DiffusionSurface {
    /// Load from a directory of `thalos_export.py` outputs:
    /// `thalos_chart_elev.f32/.json` (required),
    /// `thalos_site_detail_<side>_90m.f32/.json` (optional).
    pub fn load(dir: &Path, radius_m: f32, body_seed: u32) -> Result<Self, String> {
        let chart_json = std::fs::read_to_string(dir.join("thalos_chart_elev.json"))
            .map_err(|e| format!("chart sidecar: {e}"))?;
        let width = json_num(&chart_json, "width").ok_or("chart sidecar: width")? as usize;
        let height = json_num(&chart_json, "height").ok_or("chart sidecar: height")? as usize;
        let px_m = json_num(&chart_json, "px_m_equator").unwrap_or(23_040.0);
        let chart = Raster::load(&dir.join("thalos_chart_elev.f32"), width, height, px_m, true)?;

        let mask: Vec<f32> = chart.mips[0]
            .2
            .iter()
            .map(|&h| if h > 0.5 { 1.0 } else { 0.0 })
            .collect();
        let landmask = Raster::from_data(mask, width, height, px_m, true);

        // Optional site detail window (pick the largest present).
        let mut detail = None;
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut best: Option<(usize, std::path::PathBuf)> = None;
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if let Some(side) = name
                    .strip_prefix("thalos_site_detail_")
                    .and_then(|r| r.strip_suffix("_90m.json"))
                    .and_then(|s| s.parse::<usize>().ok())
                    && best.as_ref().is_none_or(|(b, _)| side > *b)
                {
                    best = Some((side, e.path()));
                }
            }
            if let Some((side, json_path)) = best {
                let json = std::fs::read_to_string(&json_path)
                    .map_err(|e| format!("detail sidecar: {e}"))?;
                let lon = json_num(&json, "site_lon_deg").ok_or("detail sidecar: site_lon_deg")?;
                let lat = json_num(&json, "site_lat_deg").ok_or("detail sidecar: site_lat_deg")?;
                let raster = Raster::load(
                    &json_path.with_extension("f32"),
                    side,
                    side,
                    90.0,
                    false,
                )?;
                // Relief-energy conditioning raster (probe field.rs recipe).
                let mut rough = vec![0f32; side * side];
                {
                    let m0 = &raster.mips[0].2;
                    let (w2, _, m2) = &raster.mips[2];
                    for y in 0..side {
                        for x in 0..side {
                            let c = m2[(y / 4).min(w2 - 1) * w2 + (x / 4).min(w2 - 1)];
                            rough[y * side + x] = (m0[y * side + x] - c).abs();
                        }
                    }
                }
                let (lat_r, lon_r) = (lat.to_radians(), lon.to_radians());
                let site_dir = DVec3::new(
                    lat_r.cos() * lon_r.cos(),
                    lat_r.sin(),
                    lat_r.cos() * lon_r.sin(),
                )
                .normalize();
                // Tangent frame aligned with the export's map axes: east =
                // +longitude (raster +x), north = +latitude (raster −y).
                // ENU north is east × up — `up × east` points SOUTH, which
                // rendered the whole detail window vertically mirrored about
                // the site (INC-20260724T170955Z: invisible at the window
                // center, where every X2a verification framed).
                let east = DVec3::new(-lon_r.sin(), 0.0, lon_r.cos());
                let north = east.cross(site_dir).normalize();
                detail = Some(DetailWindow {
                    raster,
                    rough: Raster::from_data(rough, side, side, 90.0, false),
                    site_dir,
                    east,
                    north,
                });
            }
        }

        // Content identity for cache namespaces: FNV-1a over the chart payload
        // (+ detail payload length — cheap and sufficient to key re-exports).
        let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
        for v in &chart.mips[0].2 {
            for b in v.to_le_bytes() {
                fnv ^= u64::from(b);
                fnv = fnv.wrapping_mul(0x100_0000_01b3);
            }
        }
        if let Some(d) = &detail {
            fnv ^= d.raster.mips[0].2.len() as u64;
            fnv = fnv.wrapping_mul(0x100_0000_01b3);
        }

        let chart_max = chart.mips[0].2.iter().cloned().fold(0.0f32, f32::max);
        Ok(Self {
            radius_m: f64::from(radius_m),
            chart,
            landmask,
            detail,
            landcover: ProceduralSurface::new(radius_m, body_seed),
            seed: 0x7ea1_0f0d,
            // Chart peaks + sub-chart/fine octave headroom + shelf floor.
            height_range_m: (chart_max + 1_500.0).max(4_800.0),
            content_fingerprint: fnv,
        })
    }

    fn chart_px(&self, dir: DVec3) -> (f64, f64) {
        let lat = dir.y.clamp(-1.0, 1.0).asin();
        let lon = dir.z.atan2(dir.x).rem_euclid(core::f64::consts::TAU);
        (
            lon / core::f64::consts::TAU * self.chart.width as f64,
            (0.5 - lat / core::f64::consts::PI) * self.chart.height as f64,
        )
    }

    fn octave_sum(&self, dir: DVec3, footprint_m: f64, first: usize, scale: f64, ridged_w: f64) -> f64 {
        let mut h = 0.0;
        for (i, (wavelength, amp)) in OCTAVES.iter().enumerate().skip(first) {
            let gate = footprint_gate(*wavelength, footprint_m);
            if gate <= 0.0 {
                break;
            }
            let n = grad_noise(dir * (self.radius_m / wavelength), self.seed.wrapping_add(i as u64 * 977));
            let ridged = (1.0 - n.abs()) * 2.0 - 1.0;
            let shaped = n + (ridged - n) * ridged_w.clamp(0.0, 1.0);
            h += shaped * amp * scale * gate;
        }
        h
    }

    /// Planetary band: global chart land + chart-conditioned sub-Nyquist
    /// relief; smoothed-landmask shelf bathymetry; continuous shore blend
    /// (probe `field.rs::planetary`, minus the drape edge — the chart is
    /// global).
    fn planetary(&self, dir: DVec3, footprint_m: f64) -> f64 {
        let (px, py) = self.chart_px(dir);
        let chart_h = self.chart.sample_px(px, py, footprint_m.max(self.chart.px_m));
        let shelf = self
            .landmask
            .sample_px(px, py, footprint_m.max(self.chart.px_m * 2.0));

        let relief_w = (chart_h.max(0.0) / 700.0 + 0.25).clamp(0.25, 2.0);
        let land_h = chart_h.max(0.0)
            + self.octave_sum(dir, footprint_m, SUB_CHART_OCTAVE0, 3.8 * relief_w, 0.45);
        let depth = -150.0 - 3_300.0 * smootherstep((0.55 - shelf).clamp(0.0, 0.55) / 0.55);
        let ocean_h = depth + self.octave_sum(dir, footprint_m, SUB_CHART_OCTAVE0, 0.25, 0.0);
        let shore = smootherstep(((shelf - 0.30) / 0.40).clamp(0.0, 1.0))
            .max(smootherstep((chart_h / 90.0).clamp(0.0, 1.0)));
        ocean_h + (land_h - ocean_h) * shore
    }

    /// The detail raster's pixel coords for `dir`, or None outside the window
    /// (tangent drape at the site, axes matched to the export's map axes).
    fn detail_px(&self, dir: DVec3) -> Option<(f64, f64)> {
        let d = self.detail.as_ref()?;
        let ang = dir.dot(d.site_dir).clamp(-1.0, 1.0).acos();
        let side = d.raster.width as f64;
        // Cheap reject: outside the window's circumscribed radius.
        if ang * self.radius_m > side * 90.0 {
            return None;
        }
        let du = dir.dot(d.east) * self.radius_m;
        let dv = dir.dot(d.north) * self.radius_m;
        let dx = side * 0.5 + du / 90.0;
        let dy = side * 0.5 - dv / 90.0;
        if (0.0..side).contains(&dx) && (0.0..side).contains(&dy) {
            Some((dx, dy))
        } else {
            None
        }
    }

    /// Regional band: the model's detail output as a residual against the
    /// planetary band, mip-matched and edge-feathered; plus the roughness-
    /// conditioned fine band.
    fn regional(&self, dir: DVec3, footprint_m: f64, planetary_h: f64) -> f64 {
        let Some(d) = &self.detail else {
            return self.fine_band(dir, footprint_m, self.chart_rough_scale(dir));
        };
        let Some((dx, dy)) = self.detail_px(dir) else {
            return self.fine_band(dir, footprint_m, self.chart_rough_scale(dir));
        };
        let detail_h = d.raster.sample_px(dx, dy, footprint_m.max(90.0));
        let side = d.raster.width as f64;
        let f = (dx / (side * 0.08))
            .min((side - 1.0 - dx) / (side * 0.08))
            .min(dy / (side * 0.08))
            .min((side - 1.0 - dy) / (side * 0.08))
            .clamp(0.0, 1.0);
        let residual = (detail_h - planetary_h) * smootherstep(f);
        let rough_scale = (d.rough.sample_px(dx, dy, footprint_m.max(90.0)) / 18.0).clamp(0.15, 2.2);
        residual + self.fine_band(dir, footprint_m, rough_scale)
    }

    /// Fine-band amplitude conditioning outside the detail window: chart
    /// relief, kept timid (uniform noise reads as crumple, not features —
    /// probe M4 user finding).
    fn chart_rough_scale(&self, dir: DVec3) -> f64 {
        let (px, py) = self.chart_px(dir);
        let ch = self.chart.sample_px(px, py, self.chart.px_m);
        (ch.max(0.0) / 900.0 + 0.15).clamp(0.15, 0.7)
    }

    fn fine_band(&self, dir: DVec3, footprint_m: f64, rough_scale: f64) -> f64 {
        let mut h = 0.0;
        for (i, (wavelength, amp)) in FINE_OCTAVES.iter().enumerate() {
            let gate = footprint_gate(*wavelength, footprint_m);
            if gate <= 0.0 {
                break;
            }
            let n = grad_noise(
                dir * (self.radius_m / wavelength),
                self.seed.wrapping_add(0x5eed + i as u64 * 7919),
            );
            let ridged = (1.0 - n.abs()) * 2.0 - 1.0;
            h += (n * 0.8 + ridged * 0.2) * amp * rough_scale * gate;
        }
        h
    }

    fn height(&self, dir: DVec3, footprint_m: f64) -> f64 {
        let planetary = self.planetary(dir, footprint_m);
        planetary + self.regional(dir, footprint_m, planetary)
    }

    /// Sample + dominant biome class — the `world_map` export's view, mirroring
    /// [`ProceduralSurface::sample_biome_d`]. Classification comes from the
    /// same canonical landcover evaluation the albedo uses.
    pub fn sample_biome_d(&self, dir: DVec3, lod_m: f32) -> (SurfaceSample, MacroBiome) {
        let dir = dir.normalize_or_zero();
        if dir == DVec3::ZERO {
            return (
                SurfaceSample {
                    height_m: 0.0,
                    albedo_linear: Vec3::ZERO,
                    roughness: 0.5,
                    moisture: 0.0,
                },
                MacroBiome::Ocean,
            );
        }
        let height_m = self.height(dir, f64::from(lod_m));
        let (albedo_linear, moisture, biome) =
            self.landcover.macro_albedo_for(dir, height_m, lod_m);
        (
            SurfaceSample {
                height_m: height_m as f32,
                albedo_linear,
                roughness: 0.92,
                moisture: moisture as f32,
            },
            biome,
        )
    }
}

impl SurfaceQuery for DiffusionSurface {
    fn sample(&self, dir: Vec3, lod_m: f32) -> SurfaceSample {
        self.sample_d(dir.as_dvec3(), lod_m)
    }

    fn sample_d(&self, dir: DVec3, lod_m: f32) -> SurfaceSample {
        self.sample_biome_d(dir, lod_m).0
    }

    fn sample_height_m(&self, dir: Vec3, lod_m: f32) -> f32 {
        let dir = dir.as_dvec3().normalize_or_zero();
        if dir == DVec3::ZERO {
            return 0.0;
        }
        self.height(dir, f64::from(lod_m)) as f32
    }

    fn landcover_moisture(&self, dir: DVec3) -> f32 {
        self.landcover.landcover_moisture(dir)
    }

    fn sample_bands_d(&self, dir: DVec3, lod_m: f32) -> (SurfaceSample, crate::query::MaterialBands) {
        let dir = dir.normalize_or_zero();
        if dir == DVec3::ZERO {
            return (
                SurfaceSample {
                    height_m: 0.0,
                    albedo_linear: Vec3::ZERO,
                    roughness: 0.5,
                    moisture: 0.0,
                },
                crate::query::MaterialBands::default(),
            );
        }
        let height_m = self.height(dir, f64::from(lod_m));
        let (albedo_linear, moisture, _biome, bands) =
            self.landcover.macro_albedo_bands_for(dir, height_m, lod_m);
        (
            SurfaceSample {
                height_m: height_m as f32,
                albedo_linear,
                roughness: 0.92,
                moisture: moisture as f32,
            },
            bands,
        )
    }

    fn radius_m(&self) -> f32 {
        self.radius_m as f32
    }

    fn height_range_m(&self) -> f32 {
        self.height_range_m
    }

    fn prewarm(&self, _region: Region, _lod_m: f32) {}
}
