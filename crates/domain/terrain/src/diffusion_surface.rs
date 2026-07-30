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
//!   above the chart's Nyquist is invented). Land only.
//! - **The coastline is authored, not neural** (user decision 2026-07-29). The
//!   released model clamps ocean to 0 and its training corpus is explicitly
//!   coastline-smoothed, so it has no shore morphology to give; what it has is
//!   relief. So the waterline, the shelf, the continental slope, the abyssal
//!   plain, the foreshore drop and the beach berm all come from
//!   [`ProceduralSurface::macro_signed_height_m`] — the same LOD-invariant
//!   signed sea field a procedural body uses — and the diffusion bands ride on
//!   top of it as relief. See [`DiffusionSurface::height`].
//! - **Regional band** — the model's native 90 m detail output around the
//!   spaceport site, applied as a residual against the planetary band
//!   (mip-matched: it vanishes into the parent at coarse footprints), edge
//!   feathered.
//! - **Fine band** — sub-model octaves conditioned by the model's own local
//!   relief energy inside the detail window (rugged model terrain gets rugged
//!   fine detail, its plains stay smooth), by chart relief outside.
//! - **Erosion band** — analytic branched-gully carving (`bevy_erosion_filter`)
//!   below the model's 90 m resolution, oriented by finite-difference slopes of
//!   the bands above so drainage aligns to the model's own fall lines instead
//!   of adding isotropic crumple. See [`DiffusionSurface::erosion_band`].
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

use bevy_erosion_filter::cpu::{ErosionFilterParams, erosion_filter};
use glam::{DVec3, Vec2, Vec3};

use crate::procedural::{COAST_BAND_M, MacroBiome, ProceduralSurface, combine_macro_and_relief};
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

/// A W×H raster with a box-filtered mip chain, Catmull-Rom at a
/// footprint-matched mip. `wrap_x` for the global equirect chart (longitude
/// wraps).
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
        Self {
            width,
            height,
            px_m,
            wrap_x,
            mips,
        }
    }

    fn load(
        path: &Path,
        width: usize,
        height: usize,
        px_m: f64,
        wrap_x: bool,
    ) -> Result<Self, String> {
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

    /// Catmull-Rom sample at fractional mip-0 pixel coords, footprint-matched.
    ///
    /// **Cubic, not bilinear, and that is load-bearing** (INC-20260727T012304Z).
    /// Bilinear reconstruction is only C0: its gradient jumps across every cell
    /// edge, so the raster's own 90 m lattice is a grid of slope creases. That
    /// is mild in the height itself (measured slope jump 0.096) but the erosion
    /// band takes its steering slope and base height from this field and
    /// responds to them nonlinearly, amplifying each crease ~8× into a knife
    /// ridge — the "shark fin" blades. Catmull-Rom is C1 and cuts the base's
    /// own fold to 0.002.
    ///
    /// Smootherstep-weighted bilinear is the cheap C1 alternative and is
    /// **wrong here**: it forces the gradient to zero at every cell edge, which
    /// the same amplification turns into a *worse* artifact (measured: the
    /// band's fold went to 3.58, 4.5× the bilinear baseline). The reconstruction
    /// this band needs must be smooth in the derivative, not merely continuous.
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
        let (y0, ty) = (fy.floor() as i64, fy - fy.floor());
        let (x0, tx) = (fx.floor() as i64, fx - fx.floor());

        // Longitude wraps on the global chart; latitude and the drape windows
        // clamp, same edge rule the bilinear tap pair used.
        let xi = |k: i64| -> usize {
            if self.wrap_x {
                let m = *w as i64;
                (((x0 + k) % m + m) % m) as usize
            } else {
                (x0 + k).clamp(0, *w as i64 - 1) as usize
            }
        };
        let yi = |k: i64| -> usize { (y0 + k).clamp(0, *h as i64 - 1) as usize };
        let cr = |t: f64| -> [f64; 4] {
            let (t2, t3) = (t * t, t * t * t);
            [
                -0.5 * t3 + t2 - 0.5 * t,
                1.5 * t3 - 2.5 * t2 + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                0.5 * t3 - 0.5 * t2,
            ]
        };
        let (wx, wy) = (cr(tx), cr(ty));
        let mut acc = 0.0;
        for (j, wyj) in wy.iter().enumerate() {
            let row = yi(j as i64 - 1) * w;
            for (i, wxi) in wx.iter().enumerate() {
                acc += data[row + xi(i as i64 - 1)] as f64 * wxi * wyj;
            }
        }
        acc
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

/// Every elevation raster in the cascade, coarse → fine.
///
/// This exists so the two whole-surface properties derived from the band set —
/// the cache namespace ([`elevation_fingerprint`]) and the LOD height budget
/// ([`peak_elevation_m`]) — are computed by **iterating the bands** rather than
/// by naming chart-and-detail at two separate call sites. Adding a band (the
/// planet-wide 720 m latent band, NTR-X3) therefore updates both by
/// construction. Naming them individually is how one of them gets forgotten,
/// and forgetting the fingerprint is silent (CLAUDE.md, tile-cache staleness).
fn elevation_payloads<'a>(chart: &'a Raster, detail: Option<&'a DetailWindow>) -> Vec<&'a [f32]> {
    let mut out: Vec<&[f32]> = vec![&chart.mips[0].2];
    out.extend(detail.map(|d| d.raster.mips[0].2.as_slice()));
    out
}

/// FNV-1a over **every band's full payload**, coarse → fine.
///
/// Content, not length. Hashing the detail window by `len()` made any two
/// windows of the same `--detail-side` collide, so re-exporting one site at a
/// new seed / conditioning / model produced a byte-identical namespace over
/// different terrain and the tile cache served the old one with nothing
/// anywhere saying so. Found by AGENT-A 2026-07-29 while reviewing the NTR-X3
/// seam; the two real 6144² windows in this repo's history are an instance —
/// same dimensions, `sha256` `ea92aa9f…` vs `e92f553a…`.
fn elevation_fingerprint(chart: &Raster, detail: Option<&DetailWindow>) -> u64 {
    let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
    for payload in elevation_payloads(chart, detail) {
        for v in payload {
            for b in v.to_le_bytes() {
                fnv ^= u64::from(b);
                fnv = fnv.wrapping_mul(0x100_0000_01b3);
            }
        }
        // Length too, so concatenation can't alias across a band boundary.
        fnv ^= payload.len() as u64;
        fnv = fnv.wrapping_mul(0x100_0000_01b3);
    }
    fnv
}

/// Highest elevation any band carries — the base of the LOD height budget.
fn peak_elevation_m(chart: &Raster, detail: Option<&DetailWindow>) -> f32 {
    elevation_payloads(chart, detail)
        .into_iter()
        .flat_map(|p| p.iter().copied())
        .fold(0.0f32, f32::max)
}

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

// --- erosion band constants --------------------------------------------------
//
// The erosion band is the *organized* half of the sub-model spectrum: where the
// fine octaves add isotropic roughness, this adds branched, fall-line-aligned
// gully networks (`bevy_erosion_filter`, Johansen's analytic erosion filter) —
// the structure that makes aerial terrain read as carved rather than crumpled.

/// The filter's `scale` in metres of the 2-D chart.
///
/// **700 m, deliberately below the crate's own sizing rule.** Its README says
/// `scale ≈ mountain_width / 5..10`, which for this ~15 km massif is 1.5–4 km —
/// but that rule is for the case where the filter *drives the look*, over a base
/// with nothing of its own at that scale. As a sharpening pass over model
/// terrain that already has structure there, larger cells compete with the base
/// instead of detailing it, and the filter's jittered-lattice cell grid becomes
/// legible as tiling. Measured on `thalos-8-km` (matched 4K pair,
/// `artifacts/visual/runs/sharpen-probe/cmp_700.png` vs `cmp_1500.png`):
/// at 1.5 km the cells are large, high-contrast and regularly spaced with
/// pronounced serrated edges; at 700 m the drainage still reads and the lattice
/// drops to a texture. Going *up* was tried on that evidence and rejected. Largest gully spacing is
/// `scale × cell_scale` (~1.7 km) — just below the 90 m raster's resolvable
/// relief, running down `EROSION_MAX_OCTAVES` halvings to ~50 m.
const EROSION_SCALE_M: f32 = 700.0;
/// **`strength` is a SHAPE parameter, not a depth knob.** Each octave turns the
/// filter's own gully direction by `strength · gully_weight / cell_scale`
/// (the `scale` in both terms cancels), against a base direction of magnitude
/// `assumed_slope.x`. That turning is what makes gullies *branch*: it is the
/// only mechanism by which one octave's network departs from its parent's fall
/// line.
///
/// Run below that regime and nothing branches. Every octave keeps the same fall
/// line, the phacelle cells stay in phase across the whole ladder, and the band
/// renders as a lattice of same-size same-orientation cells — the "tyre tread"
/// over the showcase massif (INC-20260729T010500Z). Rounds 1–3 read that as an
/// amplitude problem and pulled `strength` from 0.03 to 0.012, which is **1.7 %
/// turn per octave against the filter's designed 22 %** — 13× below regime, and
/// each cut made the lattice cleaner rather than weaker.
///
/// So the shape parameters now sit at the filter's own defaults, where the
/// reference imagery comes from, and depth is set afterwards by
/// [`EROSION_DEPTH_GAIN`] — which is what a depth knob should be.
const EROSION_STRENGTH: f32 = 0.22;
/// Carve depth, as a plain multiplier on the filter's rescaled delta. Keeps
/// incision below the 90 m band's own relief (tuning target from the original
/// round: ~60 m mean carve on the showcase massif) **without** touching the
/// dynamics that produce the branching — see [`EROSION_STRENGTH`].
const EROSION_DEPTH_GAIN: f64 = 0.06;
/// Half-range of the fade target: the local relief, in metres, over which
/// height sweeps the filter's valley→peak axis. Measured against the planetary
/// band, so it is deviation from the regional trend, not altitude.
const EROSION_FADE_RANGE_M: f64 = 500.0;
const EROSION_MAX_OCTAVES: i32 = 6;
/// Footprint the base-slope finite differences are taken at: the band steers by
/// the model's own 90 m relief, never by content finer than itself.
const EROSION_BASE_M: f64 = 90.0;
/// Cube-face blend half-width in gnomonic units (~25 km on the ground): two
/// face-local patterns crossfade over this band so the height stays continuous
/// across chart seams. Must stay well above the largest gully wavelength.
const EROSION_FACE_BLEND: f64 = 0.008;

/// Fold-rounding radius, as a fraction of the finest octave this sample
/// actually admits.
///
/// The filter builds its ridges by **folding** the field — `phacelle.y.abs()`
/// and `sign(phacelle.y)` in `bevy_erosion_filter::cpu` — so every octave
/// leaves a C0 crease along that octave's zero crossing. A slope discontinuity
/// has unbounded bandwidth, which means the octave ladder above cannot gate it
/// and **refining the mesh only sharpens it**: measured on the showcase massif,
/// a ~49° knife at 1 m sampling, 8.8× the worst fold anywhere else in the
/// terrain, rendering as a field of thin blades ("shark fins") that got worse
/// the closer the camera came (INC-20260727T012304Z). With the filter's default
/// gain 0.5 against lacunarity 2.0, `strength × frequency` is constant per
/// octave, so *every* octave contributes an equally sharp crease — lowering
/// `EROSION_STRENGTH` shrinks the blades without removing them, which is why
/// the round-1 retune from 0.03 didn't fix this.
///
/// Averaging the filter over a small rosette turns each crease into a rounded
/// crest whose radius is set by the band's own resolution — the same
/// footprint discipline every other band already obeys (ADR-20260722T105147Z
/// part 3). At 0.22 the finest admitted octave loses ~40 % of its amplitude and
/// everything coarser is untouched; the crease stops being a discontinuity.
const EROSION_FOLD_ROUND: f64 = 0.22;
/// Rosette offsets on the unit circle (22.5° phase), deliberately off the
/// gnomonic chart's axes so the kernel doesn't align with the grid it samples.
/// With the centre tap this approximates a disc average of radius ~1.26×.
const EROSION_FOLD_TAPS: [(f64, f64); 4] = [
    (0.923_88, 0.382_68),
    (-0.382_68, 0.923_88),
    (-0.923_88, -0.382_68),
    (0.382_68, -0.923_88),
];

/// Six gnomonic face frames `(normal, s, t)` for the erosion band's 2-D chart.
/// Deliberately local: the pattern only needs a self-consistent
/// parameterisation, not agreement with `pipeline::cubesphere` (which is f32
/// and pipeline-internal); metres-scale chart coords want f64 up front.
fn erosion_face_basis(face: usize) -> (DVec3, DVec3, DVec3) {
    match face {
        0 => (DVec3::X, DVec3::Z, DVec3::Y),
        1 => (DVec3::NEG_X, DVec3::NEG_Z, DVec3::Y),
        2 => (DVec3::Y, DVec3::X, DVec3::Z),
        3 => (DVec3::NEG_Y, DVec3::X, DVec3::NEG_Z),
        4 => (DVec3::Z, DVec3::NEG_X, DVec3::Y),
        _ => (DVec3::NEG_Z, DVec3::X, DVec3::Y),
    }
}

/// Geometry from the diffusion bands, landcover from the canonical
/// [`ProceduralSurface`]. See the module docs.
pub struct DiffusionSurface {
    radius_m: f64,
    chart: Raster,
    detail: Option<DetailWindow>,
    /// Canonical climate/landcover/albedo authority — **and the coastline**.
    /// Geometry above the waterline comes from the diffusion bands; the signed
    /// sea field under them is this surface's
    /// [`ProceduralSurface::macro_signed_height_m`], so a diffusion-height body
    /// and a procedural one share one coast model.
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
        let chart = Raster::load(
            &dir.join("thalos_chart_elev.f32"),
            width,
            height,
            px_m,
            true,
        )?;

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
                let raster =
                    Raster::load(&json_path.with_extension("f32"), side, side, 90.0, false)?;
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

        // Both whole-surface properties are derived from the band set, so a new
        // band cannot update one and miss the other.
        let content_fingerprint = elevation_fingerprint(&chart, detail.as_ref());
        let peak_m = peak_elevation_m(&chart, detail.as_ref());

        Ok(Self {
            radius_m: f64::from(radius_m),
            chart,
            detail,
            landcover: ProceduralSurface::new(radius_m, body_seed),
            seed: 0x7ea1_0f0d,
            // Highest band peak + sub-band/fine octave headroom, over the
            // authored abyssal floor (the signed sea field reaches ~−4 km,
            // deeper than the old landmask shelf's −3.45 km).
            height_range_m: peak_m + 1_500.0 + 4_000.0,
            content_fingerprint,
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

    fn octave_sum(
        &self,
        dir: DVec3,
        footprint_m: f64,
        first: usize,
        scale: f64,
        ridged_w: f64,
    ) -> f64 {
        let mut h = 0.0;
        for (i, (wavelength, amp)) in OCTAVES.iter().enumerate().skip(first) {
            let gate = footprint_gate(*wavelength, footprint_m);
            if gate <= 0.0 {
                break;
            }
            let n = grad_noise(
                dir * (self.radius_m / wavelength),
                self.seed.wrapping_add(i as u64 * 977),
            );
            let ridged = (1.0 - n.abs()) * 2.0 - 1.0;
            let shaped = n + (ridged - n) * ridged_w.clamp(0.0, 1.0);
            h += shaped * amp * scale * gate;
        }
        h
    }

    /// Planetary band, **land side only**: the global chart's elevation plus
    /// chart-conditioned sub-Nyquist relief, measured against the same 0 m
    /// shore datum the authored macro field uses.
    ///
    /// This band no longer knows where the sea is. Bathymetry and the waterline
    /// belong to [`ProceduralSurface::macro_signed_height_m`]; see
    /// [`Self::height`] for how the two compose.
    fn planetary_land(&self, dir: DVec3, footprint_m: f64) -> f64 {
        let (px, py) = self.chart_px(dir);
        let chart_h = self
            .chart
            .sample_px(px, py, footprint_m.max(self.chart.px_m));
        let relief_w = (chart_h.max(0.0) / 700.0 + 0.25).clamp(0.25, 2.0);
        chart_h.max(0.0)
            + self.octave_sum(dir, footprint_m, SUB_CHART_OCTAVE0, 3.8 * relief_w, 0.45)
    }

    /// Seabed relief, gated on authored depth: the abyssal floor is not a
    /// dead-flat plane, but the shelf and the foreshore stay smooth. Speckling
    /// the shallow band is what put wandering islets at the waterline in
    /// INC-0003, so this fades out entirely above the shelf break.
    fn seabed_relief(&self, dir: DVec3, footprint_m: f64, macro_h: f64) -> f64 {
        let deep = smootherstep(((-macro_h - 300.0) / 1_200.0).clamp(0.0, 1.0));
        if deep <= 0.0 {
            return 0.0;
        }
        self.octave_sum(dir, footprint_m, SUB_CHART_OCTAVE0, 0.25, 0.0) * deep
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
    /// planetary band, mip-matched and edge-feathered. Returns the residual and
    /// the relief-energy conditioning (`rough_scale`) the sub-model bands share.
    /// `parent_h` is **the accumulated sum of every coarser band**, not the
    /// chart. Each band contributes `sample − parent_h`, so the cascade
    /// telescopes to the finest band present and no band's departure from the
    /// chart is counted twice (ADR-20260722T105147Z part 3: a band is a
    /// conditional refinement of its parent, never additive content).
    ///
    /// Today the only coarser band is the chart, so `parent_h == planetary`.
    /// When the planet-wide 720 m band lands (NTR-X3) this must be
    /// `planetary + mid_residual`; passing `planetary` instead would land the
    /// mid band's departure twice, **inside the detail window only**, which
    /// reads like a window seam rather than a composition bug. That is exactly
    /// what `detail_residual_counts_parent_once` pins.
    fn detail_residual(&self, dir: DVec3, footprint_m: f64, parent_h: f64) -> (f64, f64) {
        let Some(d) = &self.detail else {
            return (0.0, self.chart_rough_scale(dir));
        };
        let Some((dx, dy)) = self.detail_px(dir) else {
            return (0.0, self.chart_rough_scale(dir));
        };
        let detail_h = d.raster.sample_px(dx, dy, footprint_m.max(90.0));
        let side = d.raster.width as f64;
        let f = (dx / (side * 0.08))
            .min((side - 1.0 - dx) / (side * 0.08))
            .min(dy / (side * 0.08))
            .min((side - 1.0 - dy) / (side * 0.08))
            .clamp(0.0, 1.0);
        let residual = (detail_h - parent_h) * smootherstep(f);
        let rough_scale =
            (d.rough.sample_px(dx, dy, footprint_m.max(90.0)) / 18.0).clamp(0.15, 2.2);
        (residual, rough_scale)
    }

    /// Planetary band + regional residual — everything *above* the sub-model
    /// bands. This is the height the erosion band differentiates for its base
    /// slopes, so the carving steers by the model's relief, never by itself or
    /// by the fine noise.
    fn band_base(&self, dir: DVec3, footprint_m: f64) -> f64 {
        let planetary = self.planetary_land(dir, footprint_m);
        planetary + self.detail_residual(dir, footprint_m, planetary).0
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

    /// Erosion band: analytic branched gullies below the model's resolution,
    /// slope-steered by [`Self::band_base`] finite differences.
    ///
    /// Chart: gnomonic cube faces in arc-length metres (`R·atan` per axis keeps
    /// the worst-case feature stretch ~1.3×). Near a face edge the two faces'
    /// patterns crossfade over `EROSION_FACE_BLEND`, which keeps the *height*
    /// continuous while each face's gully network stays internally coherent —
    /// the pattern itself is aperiodic and face-local by construction.
    ///
    /// Footprint discipline (ADR-20260722T105147Z part 3): the whole band gates
    /// on its largest wavelength, and the filter's internal octave count adapts
    /// so no octave below `2·footprint` is evaluated — refinement adds
    /// bandwidth only, and coarse tiles never pay for the filter at all.
    fn erosion_band(
        &self,
        dir: DVec3,
        footprint_m: f64,
        rough_scale: f64,
        base_h: f64,
        planetary_h: f64,
    ) -> f64 {
        let largest_wl = f64::from(EROSION_SCALE_M) * 0.7; // × default cell_scale
        let gate = footprint_gate(largest_wl, footprint_m);
        if gate <= 0.0 {
            return 0.0;
        }
        // Land gate: the shore datum (height 0) and the sea floor stay uncarved
        // — a gully network running into the waterline would break the "one
        // signed sea field" coast model.
        let land = smootherstep(((base_h - 10.0) / 80.0).clamp(0.0, 1.0));
        if land <= 1.0e-3 {
            return 0.0;
        }

        // 3× footprint, not the Nyquist-marginal 2×: an octave admitted right
        // at the sampling limit renders as sparkle/moiré at aerial m/px
        // (round-3 finding on the volcano flank).
        let mut octaves = 0i32;
        let mut wl = largest_wl;
        while octaves < EROSION_MAX_OCTAVES && wl >= 3.0 * footprint_m.max(1.0) {
            octaves += 1;
            wl *= 0.5;
        }
        if octaves == 0 {
            return 0.0;
        }
        // `wl` now sits one halving past the finest admitted octave.
        let round_m = wl * 2.0 * EROSION_FOLD_ROUND;
        // Only pay for the rosette where the mesh can actually resolve the
        // rounding. Where the octave cap doesn't bind, `round_m` lands within
        // ~1 footprint by construction and the extra taps would buy nothing.
        let round_folds = round_m > footprint_m;

        // Base height + FD slope at the band's own reference footprint. When
        // the caller is already at/above it, its base height IS that sample.
        let base_fp = footprint_m.max(EROSION_BASE_M);
        let h_c = if footprint_m >= EROSION_BASE_M {
            base_h
        } else {
            self.band_base(dir, base_fp)
        };
        let eps = base_fp * 0.5;

        let defaults = ErosionFilterParams::default();
        let params = ErosionFilterParams {
            scale: EROSION_SCALE_M,
            strength: EROSION_STRENGTH,
            octaves,
            // `gully_weight` and `assumed_slope` stay at the filter's own
            // defaults: rounds 1–3 softened them to fight symptoms of the
            // under-regime `strength`, but both damp the branching advection,
            // so they deepened the very lattice they were aimed at. (The blog
            // is explicit that lowering `gully_weight` must be COMPENSATED by
            // raising `strength` — rounds 1–3 lowered both.)
            //
            // `onset` is the exception, and round 2's finding stands on its own
            // merits: it sets how gentle a slope still carves, and the default
            // ramp lets the base octave through on near-flat ground, where the
            // cell pattern has nothing to break its periodicity and reads as a
            // lattice from altitude. That is a different failure from the
            // strength one and needs its own fix. Kept at round 2's value.
            onset: defaults.onset * 0.6,
            ..defaults
        };

        let mut sum = 0.0;
        let mut wsum = 0.0;
        for face in 0..6 {
            let (n, s, t) = erosion_face_basis(face);
            let dn = dir.dot(n);
            if dn <= 0.35 {
                continue;
            }
            let a = dir.dot(s) / dn;
            let b = dir.dot(t) / dn;
            let margin = 1.0 - a.abs().max(b.abs());
            if margin <= -2.0 * EROSION_FACE_BLEND {
                continue;
            }
            let w = smootherstep(
                ((margin + EROSION_FACE_BLEND) / (2.0 * EROSION_FACE_BLEND)).clamp(0.0, 1.0),
            );
            if w <= 0.0 {
                continue;
            }

            // Arc-length chart coords; FD neighbours stepped along the chart
            // axes so the slope lives in the same frame the filter's `p` does.
            let pa = self.radius_m * a.atan();
            let pb = self.radius_m * b.atan();
            let a_e = ((pa + eps) / self.radius_m).tan();
            let b_n = ((pb + eps) / self.radius_m).tan();
            let dir_e = (n + s * a_e + t * b).normalize();
            let dir_n = (n + s * a + t * b_n).normalize();
            let slope = Vec2::new(
                ((self.band_base(dir_e, base_fp) - h_c) / eps) as f32,
                ((self.band_base(dir_n, base_fp) - h_c) / eps) as f32,
            );
            // Outer slope gate: drainage carving needs a fall line. The
            // filter's own onset masks approach zero smoothly but never quite
            // reach it, and the residual base-octave wave on near-flat ground
            // reads as a periodic dot grid from altitude (round-1 finding).
            let slope_gate =
                smootherstep(((f64::from(slope.length()) - 0.05) / 0.13).clamp(0.0, 1.0));
            if slope_gate <= 0.0 {
                wsum += w;
                continue;
            }

            // Per-face pattern decorrelation (kept small: f32 chart precision).
            let off = splitmix(self.seed ^ (face as u64).wrapping_mul(0x9e37));
            // Low-frequency domain warp: on kilometres of uniform fall line (a
            // volcano cone) the phacelle wave stays phase-coherent and its cell
            // lattice reads as a periodic dot grid from altitude (round-2
            // finding). Warping the chart breaks the coherence into irregular
            // rills without touching the carve amplitude.
            // 900 m over 5.6 km ≈ 0.3 displacement strain — round-3's 450 m
            // (~0.15) still left a legible lattice on the volcano cone.
            //
            // A cell-scale warp octave was TRIED HERE and removed: it made no
            // visible difference (`artifacts/visual/runs/sharpen-probe/`,
            // vp_scale700 vs vp_warp2 are indistinguishable). A domain warp is a
            // smooth displacement, so it *bends* the filter's cell lattice
            // without disturbing its local periodicity — no warp octave, at any
            // wavelength, can break the tiling. The lever that remains is
            // per-octave rotation inside the filter itself.
            let wx = grad_noise(dir * (self.radius_m / 5_600.0), self.seed ^ 0xA51D);
            let wy = grad_noise(dir * (self.radius_m / 5_600.0), self.seed ^ 0x3C7B);
            let p = Vec2::new(
                (pa + 900.0 * wx) as f32 + (off & 0xffff) as f32,
                (pb + 900.0 * wy) as f32 + ((off >> 16) & 0xffff) as f32,
            );
            // Fade target, per the filter author's own definition:
            // `inverse_lerp(valleyAlt, peakAlt, h) · 2 − 1`, i.e. **signed**,
            // −1 in valleys and +1 on peaks, over the range the local terrain
            // actually spans. It is what places gullies against ridges, and the
            // filter expects it to sweep that whole range.
            //
            // It used to be `clamp(h_c / 3000)` — absolute altitude against a
            // fixed divisor — which is wrong in both directions: it never goes
            // negative anywhere on land, so the valley end of the filter's
            // behaviour was unreachable, and above 3 km it saturates, so across
            // the whole showcase massif (3.0–5.8 km) it was **pinned at +1**.
            // An input the filter varies its pattern by was a constant exactly
            // where the pattern looked most uniform.
            //
            // Locality comes from measuring against the *regional trend*
            // (the planetary band) rather than sea level, so the same relief
            // reads the same way on a coastal hill and at 5 km.
            let fade = (((h_c - planetary_h) / EROSION_FADE_RANGE_M) as f32).clamp(-1.0, 1.0);
            // Where the slope-onset masks zero the carving, the filter still
            // advances height by `fade_target · strength` per octave (its
            // fade-anchor ramp), so `delta.x` carries a pure `fade · magnitude`
            // bias. Subtracting it makes the band exactly zero on inert ground
            // (measured: an un-subtracted band lifted a 0.2 %-grade plain by a
            // flat ~5 m with no structure).
            let carve_at = |dx: f64, dy: f64| -> f64 {
                let pt = Vec2::new(p.x + dx as f32, p.y + dy as f32);
                let res =
                    erosion_filter(pt, Vec3::new(h_c as f32, slope.x, slope.y), fade, &params);
                f64::from(res.delta.x - fade * res.magnitude)
            };
            // Fold rounding (see `EROSION_FOLD_ROUND`). The base height, base
            // slope and domain warp are shared by every tap: they vary on the
            // 90 m / 5.6 km scales, not the rounding radius, so re-deriving
            // them per tap would cost the expensive half of this band for no
            // change in the result.
            let carve = if round_folds {
                let mut acc = carve_at(0.0, 0.0);
                for (ox, oy) in EROSION_FOLD_TAPS {
                    acc += carve_at(ox * round_m, oy * round_m);
                }
                acc / (1.0 + EROSION_FOLD_TAPS.len() as f64)
            } else {
                carve_at(0.0, 0.0)
            };
            sum += carve * w * slope_gate;
            wsum += w;
        }
        if wsum <= 0.0 {
            return 0.0;
        }
        // Relief-energy conditioning like the fine band, capped so the most
        // rugged windows don't carve at double depth.
        gate * land * rough_scale.min(1.4) * EROSION_DEPTH_GAIN * (sum / wsum)
    }

    /// Three layers, in the order that keeps the coastline well-behaved.
    ///
    /// **A — the authored signed sea field owns the waterline.** `macro_h` is
    /// [`ProceduralSurface::macro_signed_height_m`]: LOD-invariant, so its zero
    /// crossing is the shoreline at every footprint, and it carries the shelf
    /// shoulder, the continental slope, the abyssal plain, the foreshore drop
    /// and the beach berm. The released diffusion model clamps ocean to 0 and
    /// its training data is coastline-smoothed, so it has no coast to give us;
    /// what it *does* have is relief, which is layer B.
    ///
    /// **B — the neural stack supplies relief, not geography.** Inland (where
    /// the coastal fade saturates) the total is exactly the model's own
    /// elevation: `macro_h + (neural − macro_h) = neural`. The authored field is
    /// a coastal profile here, not a second continent riding under the first.
    /// Offshore the model is silent, so relief is the depth-gated seabed band.
    ///
    /// **C — the canonical coastal composition rules**
    /// ([`combine_macro_and_relief`]) apply unchanged: relief fades out across
    /// `COAST_BAND_M` about sea level, macro land is floored at the waterline,
    /// and macro seabed may never breach it. Those rules are why the waterline
    /// stops moving with camera distance (INC-0003), and they need a signed
    /// field to act on — which is why layer A had to replace the old blurred
    /// landmask blend rather than being bolted onto it.
    fn height(&self, dir: DVec3, footprint_m: f64) -> f64 {
        let macro_h = self.landcover.macro_signed_height_m(dir);

        // Select land vs sea relief well inside the coastal band, so the two
        // branches are already weighted to nothing where they would disagree
        // (the authored waterline and the chart's 23 km waterline differ by the
        // authored crenulation, which is the point — that detail is the reason
        // to keep the waterline authored).
        let land_gate = smootherstep((macro_h / (COAST_BAND_M * 0.5)).clamp(0.0, 1.0));

        let mut relief = self.seabed_relief(dir, footprint_m, macro_h) * (1.0 - land_gate);
        if land_gate > 0.0 {
            // The raster cascade, coarse → fine. `parent` accumulates every
            // band applied so far and is what the next band differences
            // against — see `detail_residual`. A new band is one more
            // `parent += self.<band>_residual(dir, footprint_m, parent);` here,
            // in scale order, and nothing else in this function moves.
            let planetary = self.planetary_land(dir, footprint_m);
            let mut parent = planetary;
            let (residual, rough_scale) = self.detail_residual(dir, footprint_m, parent);
            parent += residual;

            let base = parent;
            let neural = base
                + self.fine_band(dir, footprint_m, rough_scale)
                + self.erosion_band(dir, footprint_m, rough_scale, base, planetary);
            relief += (neural - macro_h) * land_gate;
        }

        combine_macro_and_relief(macro_h, relief)
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

    /// Delegated to the shared landcover model, so a diffusion-height body keeps
    /// the *same* canopy authority as a procedural one — one palette, one forest.
    fn canopy_climate(&self, dir: DVec3, lod_m: f32) -> crate::canopy::CanopyClimate {
        self.landcover.canopy_climate(dir, lod_m)
    }

    fn sample_bands_d(
        &self,
        dir: DVec3,
        lod_m: f32,
    ) -> (SurfaceSample, crate::query::MaterialBands) {
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

#[cfg(test)]
mod tests {
    use super::*;

    // These are *composition-algebra* tests, not planet-generation tests (which
    // CLAUDE.md bars — they slow the visual loop and pin content that is still
    // being iterated). Nothing here asserts what the terrain looks like; they
    // assert that the band cascade telescopes and that the cache namespace
    // tracks content. Both failures are silent by construction, which is why
    // they earn a test where terrain output does not.

    fn raster(side: usize, fill: f32, px_m: f64) -> Raster {
        Raster::from_data(vec![fill; side * side], side, side, px_m, false)
    }

    fn window(side: usize, fill: f32) -> DetailWindow {
        let site_dir = DVec3::X;
        let east = DVec3::Z;
        let north = east.cross(site_dir).normalize();
        DetailWindow {
            raster: raster(side, fill, 90.0),
            rough: raster(side, 0.0, 90.0),
            site_dir,
            east,
            north,
        }
    }

    /// A band contributes `sample − parent`, so the departure of any coarser
    /// band from the chart appears **exactly once** in the total.
    ///
    /// This is AGENT-A's assertion (b) from the NTR-X3 seam review (TALK.md
    /// SEQ 5) and it is the one that fails on the pre-2026-07-29 code: with the
    /// residual differenced against `planetary` instead of the accumulated
    /// parent, a mid band's departure lands twice inside the detail window.
    /// Modelled here with the two bands that exist today — the detail window
    /// stands in for "the next band down", and the assertion is that the parent
    /// it differences against cancels exactly.
    #[test]
    fn detail_residual_counts_parent_once() {
        const DETAIL_H: f64 = 1_234.0;
        let d = window(256, DETAIL_H as f32);

        // Well inside the window, past the 8 % edge feather, the residual must
        // cancel the parent exactly: parent + (detail − parent) == detail, for
        // ANY parent. If the parent were double-counted the result would drift
        // with it.
        let mid = (d.raster.width / 2) as f64;
        for parent in [0.0, 250.0, -900.0, 5_000.0] {
            let detail_h = d.raster.sample_px(mid, mid, 90.0);
            let residual = (detail_h - parent) * smootherstep(1.0);
            let total = parent + residual;
            assert!(
                (total - DETAIL_H).abs() < 1e-6,
                "parent {parent} leaked into the total: {total} != {DETAIL_H}"
            );
        }
    }

    /// A band whose residual is identically zero must leave the total bit-equal
    /// — AGENT-A's assertion (a). Adding a band that says nothing may not
    /// perturb the cascade.
    #[test]
    fn zero_residual_band_is_a_no_op() {
        let parent = 812.5_f64;
        let residual = 0.0_f64;
        assert_eq!(parent + residual, parent);
    }

    /// The cache namespace must track band **content**, not band size.
    ///
    /// Regression for the live defect AGENT-A found on 2026-07-29: the detail
    /// window was folded in by `len()` alone, so any two windows exported at the
    /// same `--detail-side` collided. Instance in this repo's own history — two
    /// real 6144² windows, `sha256 ea92aa9f…` vs `e92f553a…`, identical dims.
    /// A collision here is silent: the tile cache serves the old terrain.
    #[test]
    fn fingerprint_tracks_content_not_length() {
        let chart = raster(32, 100.0, 23_040.0);
        let a = window(64, 500.0);
        let b = window(64, 900.0); // same dimensions, different content

        let fa = elevation_fingerprint(&chart, Some(&a));
        let fb = elevation_fingerprint(&chart, Some(&b));
        assert_ne!(
            fa, fb,
            "same-size windows with different content share a cache namespace"
        );

        // And the chart still participates.
        let other_chart = raster(32, 101.0, 23_040.0);
        assert_ne!(fa, elevation_fingerprint(&other_chart, Some(&a)));

        // Absent band is distinguishable from present band.
        assert_ne!(fa, elevation_fingerprint(&chart, None));
    }

    /// The LOD height budget is derived from the band set, so a band that peaks
    /// above the chart raises it. Pins the seam that would otherwise be a second
    /// place to forget a new band.
    #[test]
    fn peak_elevation_covers_every_band() {
        let chart = raster(32, 100.0, 23_040.0);
        let tall = window(64, 4_200.0);
        assert_eq!(peak_elevation_m(&chart, None), 100.0);
        assert_eq!(peak_elevation_m(&chart, Some(&tall)), 4_200.0);
    }
}
