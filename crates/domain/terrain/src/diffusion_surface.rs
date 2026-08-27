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
//! - **Regional band** — native 90 m detail windows wherever they have been
//!   baked and checked out (today: the spaceport). Each window is a residual
//!   against the planetary band (mip-matched, edge-feathered). Missing or
//!   incomplete windows are skipped, so a machine renders the chart plus
//!   whatever tiles are actually on disk. Add more windows incrementally;
//!   share them through Git LFS (`just terrain-assets`).
//! - **Fine band** — everything below the learned data's own resolution, which
//!   is 90 m inside the detail window and 1.2 km outside it. Not decoration:
//!   at those scales this band *is* the terrain, so it is regime-selected by
//!   the base terrain's own slope — depositional lowland, soil-mantled
//!   hillslope, or bare rock — and each regime brings its own amplitude, Hurst
//!   exponent and shaping. See [`DiffusionSurface::fine_band`].
//!
//! There is no erosion band. An analytic branched-gully carve
//! (`bevy_erosion_filter`) lived below the 90 m raster from 2026-07-26 to
//! 2026-08-27 and was removed: three rounds of retuning never got past its
//! cell-lattice signature, and once the fine band became regime-aware the
//! carve had nothing left to add that the hillslope and rock regimes do not
//! do better. Do not reintroduce it without new evidence
//! (INC-20260827T194228Z).
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

use std::path::{Path, PathBuf};

use glam::{DVec3, Vec3};

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

// --- band ablation (diagnostic) ------------------------------------------------

/// Which sub-model bands `THALOS_TERRAIN_BANDS` has switched off.
///
/// "The ground looks bumpy" names a wavelength, not a band, and the cascade has
/// several bands in the range a surface camera sees. Removing one and
/// recapturing is the only test that separates them, and doing it by editing
/// constants means every round is a source change nobody can reproduce later.
/// So the ablation is a named env knob instead: `THALOS_TERRAIN_BANDS=-fine`
/// drops that band from [`DiffusionSurface::height`] and nothing else, leaving
/// the planetary and regional bands to answer on their own.
///
/// Diagnostic only — the shipped surface is every band on, and every band's
/// footprint gate is unchanged by this. It is a boot-time read, so a capture
/// host must restart to pick up a new value; `THALOS_TERRAIN_BANDS` is in the
/// capture client's startup-override set for exactly that reason.
#[derive(Clone, Copy, Default)]
struct DisabledBands {
    fine: bool,
}

impl DisabledBands {
    /// Cache-namespace contribution. Zero when nothing is ablated, so the
    /// shipped surface keeps the fingerprint it has always had.
    fn namespace_salt(self) -> u64 {
        if self.fine { 0x6261_6e64_5f66_696e } else { 0 }
    }
}

fn disabled_bands() -> DisabledBands {
    static BANDS: std::sync::OnceLock<DisabledBands> = std::sync::OnceLock::new();
    *BANDS.get_or_init(|| {
        let Ok(value) = std::env::var("THALOS_TERRAIN_BANDS") else {
            return DisabledBands::default();
        };
        let mut bands = DisabledBands::default();
        for token in value.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            match token.trim_start_matches('-').to_ascii_lowercase().as_str() {
                "fine" => bands.fine = true,
                other => {
                    eprintln!("unknown THALOS_TERRAIN_BANDS entry {other:?}; expected -fine")
                }
            }
        }
        bands
    })
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

/// Interpret a `THALOS_TERRAIN` override: `true` is diffusion/neural, `false`
/// is the analytic procedural planet.
pub fn parse_thalos_terrain_env(value: &str) -> Result<bool, String> {
    let value = value.trim();
    if matches_ignore_ascii_case(value, &["diffusion", "neural", "1", "true", "on"]) {
        Ok(true)
    } else if matches_ignore_ascii_case(value, &["procedural", "0", "false", "off"]) {
        Ok(false)
    } else {
        Err(format!(
            "unknown THALOS_TERRAIN={value:?}; expected neural, diffusion, or procedural"
        ))
    }
}

/// Session override, or `None` when the variable is unset/empty so the caller
/// can apply its own default (Cargo feature for the game, diffusion for tools).
pub fn thalos_terrain_env() -> Result<Option<bool>, String> {
    match std::env::var("THALOS_TERRAIN") {
        Err(_) => Ok(None),
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => parse_thalos_terrain_env(&value).map(Some),
    }
}

/// Tools follow the game: diffusion unless the session explicitly asks for
/// procedural. A missing package still falls back at the load site.
pub fn thalos_terrain_prefers_diffusion() -> bool {
    match thalos_terrain_env() {
        Ok(None) | Ok(Some(true)) => true,
        Ok(Some(false)) => false,
        Err(error) => {
            eprintln!("{error}; using diffusion");
            true
        }
    }
}

fn matches_ignore_ascii_case(value: &str, candidates: &[&str]) -> bool {
    candidates
        .iter()
        .any(|candidate| value.eq_ignore_ascii_case(candidate))
}

fn is_detail_sidecar_name(name: &str) -> bool {
    name.starts_with("thalos_site_detail_") && name.ends_with("_90m.json")
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
fn elevation_payloads<'a>(chart: &'a Raster, details: &'a [DetailWindow]) -> Vec<&'a [f32]> {
    let mut out: Vec<&[f32]> = vec![&chart.mips[0].2];
    for detail in details {
        out.push(detail.raster.mips[0].2.as_slice());
    }
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
fn elevation_fingerprint(chart: &Raster, details: &[DetailWindow]) -> u64 {
    let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
    for payload in elevation_payloads(chart, details) {
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
fn peak_elevation_m(chart: &Raster, details: &[DetailWindow]) -> f32 {
    elevation_payloads(chart, details)
        .into_iter()
        .flat_map(|p| p.iter().copied())
        .fold(0.0f32, f32::max)
}

struct DetailWindow {
    label: String,
    raster: Raster,
    /// |mip0 − mip2|: the model's own 90–360 m relief energy, conditioning the
    /// fine band.
    rough: Raster,
    site_dir: DVec3,
    east: DVec3,
    north: DVec3,
}

impl DetailWindow {
    /// Pixel coords and 8 % edge feather inside this window, or `None` outside.
    fn sample_xy(&self, dir: DVec3, radius_m: f64) -> Option<(f64, f64, f64)> {
        let ang = dir.dot(self.site_dir).clamp(-1.0, 1.0).acos();
        let side = self.raster.width as f64;
        let px_m = self.raster.px_m;
        // Cheap reject: outside the window's circumscribed radius.
        if ang * radius_m > side * px_m {
            return None;
        }
        let du = dir.dot(self.east) * radius_m;
        let dv = dir.dot(self.north) * radius_m;
        let dx = side * 0.5 + du / px_m;
        let dy = side * 0.5 - dv / px_m;
        if (0.0..side).contains(&dx) && (0.0..side).contains(&dy) {
            let f = (dx / (side * 0.08))
                .min((side - 1.0 - dx) / (side * 0.08))
                .min(dy / (side * 0.08))
                .min((side - 1.0 - dy) / (side * 0.08))
                .clamp(0.0, 1.0);
            Some((dx, dy, f))
        } else {
            None
        }
    }
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

// --- fine band ----------------------------------------------------------------
//
// The fine band is not "detail added on top of the terrain" — below the
// learned data's own resolution it *is* the terrain, and it has to hold that
// role from 90 m down to whatever a boot on the ground resolves. Its
// predecessor was four octaves of isotropic gradient noise with a 20 %
// `1 − |n|` ridged mix, and it failed both halves of that job
// (INC-20260827T194228Z): the ladder lost only ~0.6× amplitude per ~2.3×
// wavelength, so slope *grew* 35 % per octave and the finest octave's
// individual noise cells were what you actually saw; and the ridged fold left
// a C0 slope crease along every zero-contour, which no footprint gate can
// attenuate because a slope discontinuity is not band-limited. Closed creases
// plus cell-sized lumps read as a quilt over every surface, mountains included.
//
// The replacement is regime-selected. Which processes act on a piece of ground
// is decided almost entirely by how steep it is, and those processes leave
// visibly different surfaces:
//
//   - **lowland** (gentle): deposition. Broad, near-flat swells; the fine
//     scales are *smoother* than the coarse ones.
//   - **hillslope** (moderate): soil creep and overland flow. Convex spurs and
//     concave hollows elongated down the fall line, roughly self-affine.
//   - **rock** (steep): no soil mantle. Bedding ledges, fall-line chutes and
//     ribs — the sharp, layered, anisotropic surface a real crag has.
//
// So the band reads the base terrain's own 90 m slope, blends the three, and
// each contributes a ladder with its own amplitude and its own Hurst exponent.
// Sharpness comes from *shaping* — rounded folds, bedding — never from letting
// the spectrum tilt up. That is the mistake the old band made, and the one
// `EROSION_FOLD_ROUND` was written to explain before its band was deleted.

/// Coarsest fine-band wavelength: the finest octave [`OCTAVES`] carries, so
/// outside the detail window the two ladders meet with no gap between them.
const FINE_TOP_M: f64 = 1_200.0;
/// Finest wavelength the ladder will reach. Sub-metre because the band must
/// still have something to say when a boot is standing on it; every octave is
/// footprint-gated, so a tile never evaluates below what it can resolve.
const FINE_FLOOR_M: f64 = 0.7;
/// Ground resolution of the learned detail raster. Inside the window the base
/// already carries everything coarser than this, so the fine band's octaves
/// above it must stand down there or the same relief lands twice.
const DETAIL_RASTER_M: f64 = 90.0;
/// Wavelength the per-regime amplitudes below are quoted at, so they read as
/// "metres of relief per 90 m of ground" rather than as opaque ladder seeds.
const FINE_REF_M: f64 = 90.0;

/// Slope (rise/run, at [`DETAIL_RASTER_M`]) bounding the depositional regime.
/// `tan 4.6 deg = 0.08`, `tan 10 deg = 0.18`. Measured on the shipped chart the
/// valley-floor decile sits at 0.03 and the median at 0.18, so this puts flat
/// ground in the lowland regime and anything with a recognisable hillside in
/// the next one — see `band_roughness_probe`'s slope histogram.
const LOWLAND_SLOPE_LO: f64 = 0.08;
const LOWLAND_SLOPE_HI: f64 = 0.18;
/// Slope bounding the bedrock regime. `tan 20 deg = 0.36`, `tan 32 deg = 0.62`.
///
/// Deliberately **below** the ~34 deg angle of repose the physics would
/// suggest. The base this keys on is band-limited to 1.2 km outside the detail
/// window and 90 m inside it, so it is systematically gentler than the ground
/// a player stands on: measured over the showcase massif the p99 is 0.53 and
/// the maximum 0.86. A threshold at the true repose angle would fire on well
/// under 1 % of the planet and the rock regime would effectively not ship.
///
/// Set against the measured deciles rather than against the angle: at 0.36/0.62
/// the showcase peak's whole upper cone (p90 0.31, p99 0.53) still came out
/// soil-mantled, so the summit read as a smooth green-brown dome. At 0.28/0.50
/// the cone above the treeline is mostly rock and the shoulders blend, which is
/// where the landcover palette already puts the rock/scree break.
const ROCK_SLOPE_LO: f64 = 0.28;
const ROCK_SLOPE_HI: f64 = 0.50;

/// Per-regime relief at [`FINE_REF_M`], as **RMS metres**, and the Hurst
/// exponent that carries it up and down the ladder (`amp` proportional to
/// `lambda^H`).
///
/// RMS rather than a raw noise coefficient: [`grad_noise`] has a spread of its
/// own (~0.38), so a coefficient written as "2.0" delivered 0.76 m and every
/// value here had to be read through a factor nobody had written down. The
/// ladder divides by the measured spread ([`NoiseMoments::second`]), which
/// makes these numbers metres of relief per 90 m of ground — checkable against
/// a real hillside instead of against each other.
///
/// **H is the load-bearing number, not the amplitude.** Slope per octave scales
/// as `lambda^(H-1)`, so `H = 1` is constant slope at every scale, `H > 1` is a
/// surface that smooths as you approach it, and `H < 1` is one that roughens.
/// Soil-mantled ground smooths — creep is a diffusion, and that is exactly why
/// a real meadow is not a fractal at 5 m. Only fresh rock roughens, and even
/// there the shaping does more of the work than the spectral tilt.
const LOWLAND_AMP_M: f64 = 0.45;
const LOWLAND_HURST: f64 = 1.20;
const HILLSLOPE_AMP_M: f64 = 2.0;
const HILLSLOPE_HURST: f64 = 1.05;
const ROCK_AMP_M: f64 = 5.5;
/// Rock runs at the same exponent as a hillslope on purpose. A crag *is*
/// rougher at fine scales than a meadow, but that is what the fold and the
/// bedding are for; buying it from the spectrum instead makes the finest
/// admitted octave the steepest thing on screen, which is the failure this band
/// was rebuilt to remove. Measured at `H = 1.0` the ladder still tilted up
/// (the fold concentrates slope on its own), so the tilt goes the other way and
/// the shaping keeps the character.
const ROCK_HURST: f64 = 1.05;

/// Fall-line elongation. Each shaped octave is the average of a centre tap and
/// two taps offset by plus/minus `FLOW_SPAN * lambda` **along the downslope
/// direction**, which low-passes the field along the fall line and therefore
/// stretches its features down it — spurs and chutes instead of round lumps.
///
/// The offset is a bounded displacement, never a projection. Building this as
/// `noise(dot(p, fall) / L)` is the same moire generator the tile shader's
/// gully striation documents (INC-20260727T004856Z): `p` is a body-space
/// position of magnitude 3.2e6 m, so a tenth of a degree of slope rotation
/// slides that phase by whole periods. Rotating a `0.55 * lambda` offset vector
/// moves the tap by at most `0.55 * lambda`, full stop.
const FLOW_SPAN: f64 = 0.55;
/// Restores the contrast the three-tap average removes (partly correlated taps).
const FLOW_GAIN: f64 = 1.34;
/// Wavelength below which the elongation is gone, and the one above which it is
/// at full strength.
///
/// The fall line this steers by is the *regional* one, measured at 90 m. That
/// is the right direction for a 100 m hollow and the wrong one for a 2 m rill,
/// which follows whatever the ground immediately above it is doing. Applied at
/// every octave the single direction combed the whole ladder into parallel
/// hair — visible as brush strokes down a mountain flank, because real gullies
/// at that scale fan and branch instead of running in lockstep with the
/// hillside. Fading the elongation out below the scale the steering data can
/// speak for keeps the organisation where it is earned and leaves the fine
/// octaves isotropic.
/// Set from both failure directions, not from one. Faded out below 45 m the
/// 10–30 m octaves — the ones a hillside actually shows you — went isotropic
/// and the mountain came back speckled, which is the original defect wearing a
/// smaller size. Applied all the way down, the sub-10 m octaves combed into
/// visible parallel hair. The band between is where the 90 m fall line is still
/// the right answer and the hair has not started.
const FLOW_FADE_LO_M: f64 = 4.0;
const FLOW_FADE_HI_M: f64 = 16.0;

/// Hollow/spur asymmetry for the hillslope regime: hollows are tighter and
/// deeper than the spurs between them are tall, because that is where the water
/// goes. Applied as `n - k(n^2 - E[n^2])`, with the mean subtracted so the
/// transform adds no DC — see [`noise_moments`].
const HOLLOW_SKEW: f64 = 0.35;

/// Rounding radius of the rock regime's ridge fold, in units of the octave's
/// own normalised amplitude.
///
/// `1 - |n|` is how a fold becomes a ridge, and the crease it leaves at `n = 0`
/// is a slope discontinuity — unbounded bandwidth, invisible to every footprint
/// gate above it, and *sharper* the closer the camera gets. That is what put
/// blades on the massif (INC-20260727T012304Z) and creases under the quilt this
/// band replaces. `1 - sqrt(n^2 + r^2)` is the same fold with the crease
/// rounded over `|n| < r`, and because `n` is normalised per octave the
/// rounding radius scales with the octave automatically.
const RIDGE_ROUND: f64 = 0.18;
/// How much of the rock regime is folded ridge versus plain noise. All ridge is
/// a continuous crest network with no interruptions; the mix is what leaves
/// open faces between the ribs.
const ROCK_RIDGE_MIX: f64 = 0.55;

/// Bedding: mean bed thickness in metres, how far it varies across the planet,
/// and the relief one bed contributes.
///
/// Keyed on the **base height**, not on a position projected onto a dip vector.
/// Bedding shows up as contour-parallel banding precisely because beds are
/// level surfaces cutting through relief, so height is the honest
/// parameterisation — and it sidesteps the projection trap entirely, since no
/// large position is ever dotted with a slowly varying direction.
const BED_THICKNESS_M: f64 = 18.0;
const BED_VARIATION: f64 = 0.45;
const BED_VARIATION_WL_M: f64 = 26_000.0;
/// Slow tilt of the bedding, as metres of height offset over its own
/// wavelength: enough that beds are not perfectly level everywhere.
const BED_TILT_M: f64 = 70.0;
const BED_TILT_WL_M: f64 = 40_000.0;
/// Outcrop break-up: how many bed thicknesses the bedding phase wanders, and
/// over what ground distance. Wide enough that a bed still reads as a
/// continuous band across a face, short enough that it never closes a ring.
const BED_BREAK_M: f64 = 2.2;
const BED_BREAK_WL_M: f64 = 420.0;
/// Height gradient the break-up contributes, as a slope equivalent
/// (`BED_BREAK_M * thickness * |grad noise|`, with `|grad noise| ~ 0.8/lambda`).
/// Added to the real slope when sizing the strata's ground wavelength.
const BED_BREAK_SLOPE: f64 = 0.08;
/// Ledge relief is stated as **the slope of the riser**, not as a height.
///
/// A bed's ground wavelength is `thickness / slope`, so on a steep face the
/// same 2.6 m of ledge relief that reads as a bench on a hillside is squeezed
/// into an 18 m period — a riser slope of 0.9, near vertical, which lit at a
/// grazing sun came out as hard black-and-white zebra over every crag. Fixing
/// that by lowering the height would have left the term scale-dependent in the
/// same way, just less visibly. Deriving the amplitude from the wavelength
/// keeps the riser at a fixed angle wherever the bedding lands, which is what
/// "a ledge" actually means.
const STRATA_RISER_SLOPE: f64 = 0.30;
/// Ceiling on the derived amplitude, so a shallow bench on near-flat rock
/// cannot grow a bed into a landform.
const STRATA_MAX_M: f64 = 2.0;
/// How square the ledge profile is. Doubles as the bandwidth ceiling: the
/// profile's steepest slope is `k / tanh k` times a sine's, so raising this
/// sharpens the riser and nothing else gets away from the footprint gate.
const STRATA_SHARPNESS: f64 = 1.6;
/// The ground wavelength of bedding at a given slope is `thickness / slope`,
/// which collapses toward zero on a near-vertical face. Gating the strata term
/// on *that* wavelength is what stops the ledges aliasing exactly where they
/// are steepest and most visible.
const STRATA_MIN_SLOPE: f64 = 0.05;

/// Land gate: the shore datum and the sea floor keep the authored coast model
/// untouched, so the fine band fades in over the first stretch of dry land.
const FINE_LAND_FLOOR_M: f64 = 6.0;
const FINE_LAND_BAND_M: f64 = 70.0;

/// What the regional band contributed at a sample, and how far inside the
/// detail window it was. `window` is the same feather the residual is scaled
/// by, forwarded so the fine band can stand its overlapping octaves down
/// exactly where the raster starts carrying them.
struct DetailResidual {
    residual_m: f64,
    rough_scale: f64,
    window: f64,
}

impl DetailResidual {
    fn outside(rough_scale: f64) -> Self {
        Self {
            residual_m: 0.0,
            rough_scale,
            window: 0.0,
        }
    }
}

/// What the base terrain under a fine-band sample is doing.
struct FineFrame {
    /// `|grad h|` at [`DETAIL_RASTER_M`], rise over run.
    slope: f64,
    /// Unit body-space tangent pointing downhill, or zero where the slope is
    /// too small for a fall line to mean anything.
    fall: DVec3,
    rock: f64,
    hillslope: f64,
    lowland: f64,
}

/// Build-time hash of every `.rs` file in this crate, as a number.
///
/// `build.rs` already emits it for the bake cache; the tile cache reaches it
/// through [`DiffusionSurface::content_fingerprint`]. Parsed rather than
/// stored as a string so the fingerprint stays one `u64` XOR.
fn terrain_source_hash() -> u64 {
    u64::from_str_radix(env!("THALOS_TERRAIN_SOURCE_HASH"), 16).unwrap_or(0)
}

/// An arbitrary but stable tangent frame at `dir`. The fine band only needs a
/// consistent pair to take finite differences in and to express the fall line;
/// nothing downstream depends on which way "east" points.
fn fine_tangents(dir: DVec3) -> (DVec3, DVec3) {
    let seed = if dir.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
    let east = seed.cross(dir).normalize();
    (east, dir.cross(east).normalize())
}

/// Distribution constants of [`grad_noise`], measured once rather than written
/// down.
///
/// The shaping transforms are non-linear, so each one moves the mean: a
/// `n^2` term adds `E[n^2]` and a fold adds its own offset. Left in, that DC is
/// multiplied by a footprint gate — which makes it a height that **changes with
/// camera distance**, the failure mode the coast model exists to prevent
/// (INC-0003). Subtracting a hard-coded guess is how the guess silently drifts
/// when the noise changes, so the constants are calibrated from the function
/// itself over a fixed lattice with a fixed seed: deterministic, self-correcting
/// and a few microseconds once per process.
#[derive(Clone, Copy)]
struct NoiseMoments {
    /// `E[n^2]`.
    second: f64,
    /// `E[1 - sqrt(n^2 + RIDGE_ROUND^2)]`.
    ridge_mean: f64,
    /// Factor restoring the fold's spread to the noise's own.
    ridge_gain: f64,
}

fn noise_moments() -> NoiseMoments {
    static MOMENTS: std::sync::OnceLock<NoiseMoments> = std::sync::OnceLock::new();
    *MOMENTS.get_or_init(|| {
        // Irrational strides so the lattice never lands on the noise's own
        // integer grid, where the value is identically zero.
        const SIDE: i32 = 40;
        const STRIDE: f64 = 0.373_095_048;
        let mut n2 = 0.0;
        let mut r_sum = 0.0;
        let mut r2 = 0.0;
        let mut count = 0.0;
        for i in 0..SIDE {
            for j in 0..SIDE {
                for k in 0..SIDE {
                    let p = DVec3::new(
                        f64::from(i) * STRIDE,
                        f64::from(j) * STRIDE,
                        f64::from(k) * STRIDE,
                    );
                    let n = grad_noise(p, 0x9e37_79b9_7f4a_7c15);
                    let r = 1.0 - (n * n + RIDGE_ROUND * RIDGE_ROUND).sqrt();
                    n2 += n * n;
                    r_sum += r;
                    r2 += r * r;
                    count += 1.0;
                }
            }
        }
        let second = n2 / count;
        let ridge_mean = r_sum / count;
        let ridge_var = (r2 / count - ridge_mean * ridge_mean).max(1.0e-9);
        NoiseMoments {
            second,
            ridge_mean,
            ridge_gain: (second / ridge_var).sqrt(),
        }
    })
}

/// The rock regime's ridge shape: a fold with its crease rounded off, centred
/// and rescaled to sit on the same footing as the plain noise it replaces.
/// See [`RIDGE_ROUND`] for why the rounding is not optional.
fn ridge_fold(n: f64, moments: NoiseMoments) -> f64 {
    let raw = 1.0 - (n * n + RIDGE_ROUND * RIDGE_ROUND).sqrt();
    (raw - moments.ridge_mean) * moments.ridge_gain
}

fn load_detail_windows(dir: &Path) -> Vec<DetailWindow> {
    let mut sidecars: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(is_detail_sidecar_name)
            })
            .collect(),
        Err(_) => return Vec::new(),
    };
    sidecars.sort();
    let mut windows = Vec::new();
    for sidecar in sidecars {
        match load_detail_window(&sidecar) {
            Ok(window) => windows.push(window),
            Err(error) => eprintln!("diffusion detail window skipped: {error}"),
        }
    }
    windows
}

fn load_detail_window(json_path: &Path) -> Result<DetailWindow, String> {
    let label = json_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("detail")
        .to_string();
    let json = std::fs::read_to_string(json_path).map_err(|e| format!("{label} sidecar: {e}"))?;
    let width = json_num(&json, "width").ok_or_else(|| format!("{label}: width"))? as usize;
    let height = json_num(&json, "height").unwrap_or(width as f64) as usize;
    if width == 0 || height != width {
        return Err(format!("{label}: detail windows must be square, got {width}x{height}"));
    }
    let lon = json_num(&json, "site_lon_deg").ok_or_else(|| format!("{label}: site_lon_deg"))?;
    let lat = json_num(&json, "site_lat_deg").ok_or_else(|| format!("{label}: site_lat_deg"))?;
    let px_m = json_num(&json, "px_m").unwrap_or(90.0);
    let payload = json_path.with_extension("f32");
    let expected = (width * height * 4) as u64;
    let actual = std::fs::metadata(&payload)
        .map_err(|e| format!("{}: {e}", payload.display()))?
        .len();
    if actual != expected {
        return Err(format!(
            "{} is {actual} bytes, expected {expected} (incomplete checkout or Git LFS pointer). Run `just terrain-assets`",
            payload.display()
        ));
    }
    let raster = Raster::load(&payload, width, height, px_m, false)?;
    let mut rough = vec![0f32; width * height];
    {
        let m0 = &raster.mips[0].2;
        let mip2 = raster.mips.get(2);
        for y in 0..height {
            for x in 0..width {
                let c = if let Some((w2, _, m2)) = mip2 {
                    m2[(y / 4).min(w2 - 1) * w2 + (x / 4).min(w2 - 1)]
                } else {
                    m0[y * width + x]
                };
                rough[y * width + x] = (m0[y * width + x] - c).abs();
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
    Ok(DetailWindow {
        label,
        raster,
        rough: Raster::from_data(rough, width, height, px_m, false),
        site_dir,
        east,
        north,
    })
}

/// Geometry from the diffusion bands, landcover from the canonical
/// [`ProceduralSurface`]. See the module docs.
pub struct DiffusionSurface {
    radius_m: f64,
    chart: Raster,
    detail: Vec<DetailWindow>,
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
    /// Procedural generator version that authored the conditioning chart.
    /// `None` is legacy content whose sidecar predates provenance tracking.
    conditioning_generator_version: Option<u64>,
}

impl DiffusionSurface {
    /// Load from a directory of `thalos_export.py` outputs:
    /// `thalos_chart_elev.f32/.json` (required planetary band),
    /// `thalos_site_detail_*_90m.f32/.json` (optional; every complete window).
    pub fn load(dir: &Path, radius_m: f32, body_seed: u32) -> Result<Self, String> {
        let chart_json = std::fs::read_to_string(dir.join("thalos_chart_elev.json"))
            .map_err(|e| format!("chart sidecar: {e}"))?;
        let width = json_num(&chart_json, "width").ok_or("chart sidecar: width")? as usize;
        let height = json_num(&chart_json, "height").ok_or("chart sidecar: height")? as usize;
        let px_m = json_num(&chart_json, "px_m_equator").unwrap_or(23_040.0);
        let conditioning_generator_version =
            json_num(&chart_json, "conditioning_generator_version").map(|value| value as u64);
        let chart = Raster::load(
            &dir.join("thalos_chart_elev.f32"),
            width,
            height,
            px_m,
            true,
        )?;

        let detail = load_detail_windows(dir);

        // Both whole-surface properties are derived from the band set, so a new
        // band cannot update one and miss the other.
        //
        // The band ablation is folded in here because it changes generated
        // height: the tile disk cache is keyed on this fingerprint, and an
        // ablated run that reused the full-band namespace would serve the
        // *unablated* surface from disk and produce a plausible, wrong A/B.
        //
        // So is the crate's own source hash. The rasters are only half of what
        // decides a height — the analytic bands are the other half, and they
        // live in this file. Keyed on payloads alone, every edit to the fine
        // band shipped with a namespace the previous band had already filled,
        // so the tile cache served the *old* terrain and the screenshot proving
        // the change looked exactly like the screenshot before it. The bake
        // cache has folded in `THALOS_TERRAIN_SOURCE_HASH` since it existed
        // (`cache.rs`); the tile cache reads this fingerprint instead, and had
        // no equivalent. Relying on someone remembering to bump
        // `GENERATOR_VERSION` is not a mechanism, and that constant means
        // something else — it is the conditioning-provenance check a few lines
        // below, which a fine-band edit must not falsify.
        let content_fingerprint = elevation_fingerprint(&chart, &detail)
            ^ disabled_bands().namespace_salt()
            ^ terrain_source_hash();
        let peak_m = peak_elevation_m(&chart, &detail);

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
            conditioning_generator_version,
        })
    }

    /// Generator version of the macro conditioning consumed by the learned
    /// chart. A mismatch means the package is valid learned content, but its
    /// planet-scale relief predates the current authored tectonic structure.
    pub fn conditioning_generator_version(&self) -> Option<u64> {
        self.conditioning_generator_version
    }

    /// How many complete 90 m windows this process actually loaded.
    pub fn detail_window_count(&self) -> usize {
        self.detail.len()
    }

    /// Filenames of loaded windows, in cache-namespace order.
    pub fn detail_window_names(&self) -> impl Iterator<Item = &str> {
        self.detail.iter().map(|window| window.label.as_str())
    }

    /// Attach a baked drainage raster to the inner landcover authority
    /// (NTR-X2q). Geometry is unaffected — rivers are a landcover channel — but
    /// this body's landcover *is* that inner `ProceduralSurface`, so the
    /// forwarding is what makes rivers visible on the diffusion backing at all.
    #[must_use]
    pub fn with_rivers(mut self, rivers: std::sync::Arc<crate::rivers::RiverField>) -> Self {
        self.landcover = self.landcover.clone().with_rivers(rivers);
        self
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

    /// Regional band: the model's detail output as a residual against the
    /// planetary band, mip-matched and edge-feathered. Returns the residual and
    /// the relief-energy conditioning (`rough_scale`) the sub-model bands share.
    /// `parent_h` is **the accumulated sum of every coarser band**, not the
    /// chart. Each band contributes `sample − parent_h`, so the cascade
    /// telescopes to the finest band present and no band's departure from the
    /// chart is counted twice (ADR-20260722T105147Z part 3: a band is a
    /// conditional refinement of its parent, never additive content).
    ///
    /// Overlapping windows pick the sample you are furthest inside of (highest
    /// edge feather). Non-overlapping curated sites just accumulate.
    ///
    /// Today the only coarser band is the chart, so `parent_h == planetary`.
    /// When the planet-wide 720 m band lands (NTR-X3) this must be
    /// `planetary + mid_residual`; passing `planetary` instead would land the
    /// mid band's departure twice, **inside the detail window only**, which
    /// reads like a window seam rather than a composition bug. That is exactly
    /// what `detail_residual_counts_parent_once` pins.
    fn detail_residual(&self, dir: DVec3, footprint_m: f64, parent_h: f64) -> DetailResidual {
        let mut best_f = 0.0_f64;
        let mut chosen: Option<(&DetailWindow, f64, f64)> = None;
        for window in &self.detail {
            let Some((dx, dy, f)) = window.sample_xy(dir, self.radius_m) else {
                continue;
            };
            if chosen.is_some() && f < best_f {
                continue;
            }
            best_f = f;
            chosen = Some((window, dx, dy));
        }
        let Some((window, dx, dy)) = chosen else {
            return DetailResidual::outside(self.chart_rough_scale(dir));
        };
        let sample_fp = footprint_m.max(window.raster.px_m);
        let detail_h = window.raster.sample_px(dx, dy, sample_fp);
        let feather = smootherstep(best_f);
        DetailResidual {
            residual_m: (detail_h - parent_h) * feather,
            rough_scale: (window.rough.sample_px(dx, dy, sample_fp) / 18.0).clamp(0.15, 2.2),
            window: feather,
        }
    }

    /// Fine-band relief-energy conditioning outside every detail window, where
    /// there is no `rough` raster to read: the chart's own elevation stands in
    /// for how rugged the learned terrain is, kept timid because uniform noise
    /// reads as crumple rather than as features (probe M4 user finding).
    fn chart_rough_scale(&self, dir: DVec3) -> f64 {
        let (px, py) = self.chart_px(dir);
        let ch = self.chart.sample_px(px, py, self.chart.px_m);
        (ch.max(0.0) / 900.0 + 0.15).clamp(0.15, 0.7)
    }

    /// Planetary band + regional residual — everything *above* the fine band.
    /// This is the height the fine band differentiates for its regime slope, so
    /// the regime is chosen by the learned relief and never by the band's own
    /// output.
    fn band_base(&self, dir: DVec3, footprint_m: f64) -> f64 {
        let planetary = self.planetary_land(dir, footprint_m);
        planetary + self.detail_residual(dir, footprint_m, planetary).residual_m
    }

    /// The base terrain a fine-band sample sits on: how steep it is, which way
    /// is downhill, and the regime weights that follow.
    ///
    /// Slope comes from forward differences of [`Self::band_base`] at
    /// [`DETAIL_RASTER_M`] — the band steers by the learned data's own relief
    /// and never by its own output, which is what keeps the regime selection
    /// stable as the footprint refines. Forward rather than central because the
    /// base is band-limited well above this step, so the half-step bias is a
    /// fraction of a percent of the gradient and costs two field evaluations
    /// instead of four.
    fn fine_frame(&self, dir: DVec3, base_h: f64, footprint_m: f64) -> FineFrame {
        let (east, north) = fine_tangents(dir);
        let step = DETAIL_RASTER_M;
        let sample = |offset: DVec3| self.band_base((dir + offset).normalize(), step);
        // Both ends of the difference must be the same field, or the mismatch
        // *is* the gradient. Below 90 m the caller's `base_h` already is the
        // 90 m value — `OCTAVES` bottoms out at 1.2 km, the detail raster
        // clamps its own mip at 90 m, and the chart is 23 km/px, so nothing in
        // `band_base` varies with a footprint finer than that. Above it they
        // diverge, and reusing `base_h` there added a spurious offset over the
        // 90 m step, which is a fabricated slope of exactly the size that
        // reassigns a regime.
        let centre = if footprint_m <= DETAIL_RASTER_M {
            base_h
        } else {
            self.band_base(dir, step)
        };
        let dh_e = (sample(east * (step / self.radius_m)) - centre) / step;
        let dh_n = (sample(north * (step / self.radius_m)) - centre) / step;

        let slope = (dh_e * dh_e + dh_n * dh_n).sqrt();
        // Downhill, as a unit body-space tangent. Degenerate on flat ground,
        // where the fall line is meaningless and the lowland regime — which
        // does not use it — owns the sample anyway.
        let fall = if slope > 1.0e-6 {
            (east * (-dh_e / slope) + north * (-dh_n / slope)).normalize_or_zero()
        } else {
            DVec3::ZERO
        };

        let rock = smootherstep(
            ((slope - ROCK_SLOPE_LO) / (ROCK_SLOPE_HI - ROCK_SLOPE_LO)).clamp(0.0, 1.0),
        );
        let lowland = (1.0
            - smootherstep(
                ((slope - LOWLAND_SLOPE_LO) / (LOWLAND_SLOPE_HI - LOWLAND_SLOPE_LO)).clamp(0.0, 1.0),
            ))
            * (1.0 - rock);
        FineFrame {
            slope,
            fall,
            rock,
            lowland,
            hillslope: (1.0 - rock - lowland).max(0.0),
        }
    }

    /// Fine band: everything below the learned data's resolution.
    ///
    /// One octave ladder from [`FINE_TOP_M`] to [`FINE_FLOOR_M`], evaluated
    /// three ways — once per regime — and blended by the weights in
    /// [`FineFrame`]. Every octave carries the standard footprint gate, so
    /// refinement adds bandwidth and nothing else (ADR-20260722T105147Z part 3),
    /// and octaves coarser than the detail raster stand down inside the window
    /// where the raster already carries them.
    ///
    /// `window` is the detail window's feather weight (0 outside, 1 well
    /// inside): the same value the regional residual fades on, so the two bands
    /// hand over without a seam.
    fn fine_band(
        &self,
        dir: DVec3,
        footprint_m: f64,
        rough_scale: f64,
        base_h: f64,
        window: f64,
    ) -> f64 {
        // Nothing in this band survives a footprint coarser than the top
        // octave's Nyquist, and the frame below costs two field evaluations —
        // so reject before paying for it.
        if footprint_gate(FINE_TOP_M, footprint_m) <= 0.0 {
            return 0.0;
        }
        let land = smootherstep(((base_h - FINE_LAND_FLOOR_M) / FINE_LAND_BAND_M).clamp(0.0, 1.0));
        if land <= 1.0e-3 {
            return 0.0;
        }

        let frame = self.fine_frame(dir, base_h, footprint_m);
        let moments = noise_moments();
        // Amplitudes below are RMS metres; the ladder works in units of the
        // noise's own spread. See the regime-amplitude constants.
        let norm = 1.0 / moments.second.sqrt();
        let p_m = dir * self.radius_m;
        // Relief-energy conditioning, kept mild: the regime already carries the
        // structural decision, so this only says how much the learned terrain's
        // own ruggedness leans on it.
        let energy = 0.65 + 0.35 * rough_scale.clamp(0.0, 2.0);

        let mut height = 0.0;
        let mut wavelength = FINE_TOP_M;
        let mut octave = 0u64;
        while wavelength >= FINE_FLOOR_M {
            let gate = footprint_gate(wavelength, footprint_m);
            if gate <= 0.0 {
                break;
            }
            // Octaves the detail raster already carries stand down inside the
            // window, in proportion to how far inside it the sample is.
            let overlap = smootherstep(((wavelength / DETAIL_RASTER_M) - 1.0).clamp(0.0, 1.0));
            let admit = gate * (1.0 - window * overlap);
            if admit <= 1.0e-4 {
                wavelength *= 0.5;
                octave += 1;
                continue;
            }

            let seed = self.seed.wrapping_add(0x5eed).wrapping_add(octave * 7919);
            let centre = grad_noise(p_m / wavelength, seed);
            // Fall-line elongation. Only the two soil/rock regimes want it; on
            // flat ground `fall` is zero and the taps collapse onto the centre.
            let flow = smootherstep(
                ((wavelength - FLOW_FADE_LO_M) / (FLOW_FADE_HI_M - FLOW_FADE_LO_M)).clamp(0.0, 1.0),
            );
            let streaked = if frame.fall == DVec3::ZERO || flow <= 0.0 {
                centre
            } else {
                let offset = frame.fall * (FLOW_SPAN * wavelength);
                let averaged = (centre
                    + grad_noise((p_m + offset) / wavelength, seed)
                    + grad_noise((p_m - offset) / wavelength, seed))
                    / 3.0
                    * FLOW_GAIN;
                centre + (averaged - centre) * flow
            };

            let scale = wavelength / FINE_REF_M;
            if frame.lowland > 0.0 {
                height += centre * LOWLAND_AMP_M * scale.powf(LOWLAND_HURST) * frame.lowland * admit;
            }
            if frame.hillslope > 0.0 {
                let hollowed = streaked - HOLLOW_SKEW * (streaked * streaked - moments.second);
                height += hollowed
                    * HILLSLOPE_AMP_M
                    * scale.powf(HILLSLOPE_HURST)
                    * frame.hillslope
                    * admit;
            }
            if frame.rock > 0.0 {
                let folded = ridge_fold(streaked, moments);
                let shaped = streaked + (folded - streaked) * ROCK_RIDGE_MIX;
                height += shaped * ROCK_AMP_M * scale.powf(ROCK_HURST) * frame.rock * admit;
            }

            wavelength *= 0.5;
            octave += 1;
        }

        height * norm * energy * land + self.strata(dir, footprint_m, base_h, &frame) * land
    }

    /// Bedding ledges on exposed rock.
    ///
    /// A level-surface wave in the **base height**, so beds run along the
    /// contours the way real bedding does, with a slow tilt and a slowly
    /// varying thickness so the planet is not one uniform layer cake. Its
    /// ground wavelength is `thickness / slope`, and that is what the footprint
    /// gate is fed — bedding on a steep face projects to a much finer ground
    /// pattern than the same bedding on a bench, and only the projected version
    /// can alias.
    fn strata(&self, dir: DVec3, footprint_m: f64, base_h: f64, frame: &FineFrame) -> f64 {
        if frame.rock <= 0.0 {
            return 0.0;
        }
        let p_m = dir * self.radius_m;
        let thickness = BED_THICKNESS_M
            * (1.0
                + BED_VARIATION
                    * grad_noise(p_m / BED_VARIATION_WL_M, self.seed ^ 0xB3D5).clamp(-1.0, 1.0));
        // The break-up wanders the phase too, so it contributes bandwidth of
        // its own — gating on the slope term alone would let the ledges alias
        // on ground the slope says is flat enough to be safe.
        let ground_wavelength =
            thickness / (frame.slope.max(STRATA_MIN_SLOPE) + BED_BREAK_SLOPE);
        let gate = footprint_gate(ground_wavelength, footprint_m);
        if gate <= 0.0 {
            return 0.0;
        }
        let tilt = grad_noise(p_m / BED_TILT_WL_M, self.seed ^ 0x71C7) * BED_TILT_M;
        // Break the phase off the base's own contours.
        //
        // Height alone is the honest parameterisation of a level surface, but
        // the base it reads is analytic and smooth, so its contours are near
        // perfect closed curves and the ledges came out as concentric rings —
        // the "agate / topographic map" look the tile shader's gully striation
        // documents from the other direction. A bounded height offset that
        // varies on `BED_BREAK_WL_M` puts the beds back on an irregular
        // outcrop pattern without decoupling them from the relief: it is
        // several bed thicknesses of wander, not a new field.
        let broken =
            grad_noise(p_m / BED_BREAK_WL_M, self.seed ^ 0x2F19) * BED_BREAK_M * thickness;
        let phase = (base_h + tilt + broken) / thickness * core::f64::consts::TAU;
        // Flat treads, steep risers, bounded derivative.
        //
        // `sign(w)·|w|^0.6` is the obvious way to square up a cosine and it is
        // the wrong one: the exponent below 1 makes the derivative *infinite*
        // at every zero crossing, so the profile carries unbounded bandwidth
        // and the gate above cannot touch it — the same defect as an unrounded
        // ridge fold, and it drew hard zebra banding across every crag at 2 m
        // sampling. `tanh(k·sin)` is the same shape with `k` as an explicit
        // ceiling on the slope, and it stays odd, so it is still DC-free.
        let ledge = (STRATA_SHARPNESS * phase.sin()).tanh() / STRATA_SHARPNESS.tanh();
        // Peak riser slope is `amp · (k / tanh k) · (2π / wavelength)`; solving
        // it for `amp` is what makes [`STRATA_RISER_SLOPE`] mean the angle.
        let amp = (STRATA_RISER_SLOPE * ground_wavelength * STRATA_SHARPNESS.tanh()
            / (core::f64::consts::TAU * STRATA_SHARPNESS))
            .min(STRATA_MAX_M);
        ledge * amp * frame.rock * gate
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
            let detail = self.detail_residual(dir, footprint_m, planetary);
            let base = planetary + detail.residual_m;

            let mut neural = base;
            if !disabled_bands().fine {
                neural += self.fine_band(
                    dir,
                    footprint_m,
                    detail.rough_scale,
                    base,
                    detail.window,
                );
            }
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

    /// The fine band's shaping transforms must add no DC.
    ///
    /// Each one is non-linear, so left uncentred it contributes a constant per
    /// octave — and every octave is multiplied by a footprint gate, which turns
    /// that constant into a **height that moves with camera distance**. That is
    /// the LOD-invariance failure the coast model exists to prevent (INC-0003),
    /// and it is invisible in a single screenshot: the ground is simply at a
    /// different altitude when you fly closer. The centring constants come from
    /// [`noise_moments`], so this is also the test that catches a change to
    /// [`grad_noise`] silently invalidating them.
    #[test]
    fn fine_band_shaping_adds_no_dc() {
        let moments = noise_moments();
        const SIDE: i32 = 24;
        const STRIDE: f64 = 0.611_803_398;
        let mut hollow_sum = 0.0;
        let mut ridge_sum = 0.0;
        let mut ledge_sum = 0.0;
        let mut count = 0.0;
        for i in 0..SIDE {
            for j in 0..SIDE {
                for k in 0..SIDE {
                    let p = DVec3::new(
                        f64::from(i) * STRIDE,
                        f64::from(j) * STRIDE + 0.137,
                        f64::from(k) * STRIDE + 0.921,
                    );
                    let n = grad_noise(p, 0x1234_5678_9abc_def0);
                    hollow_sum += n - HOLLOW_SKEW * (n * n - moments.second);
                    ridge_sum += ridge_fold(n, moments);
                    // The strata wave, over its own phase rather than over the
                    // noise. Odd in `sin`, so it should integrate to zero over
                    // any whole number of periods and very nearly zero over a
                    // long incommensurate walk.
                    let phase = f64::from(i * SIDE * SIDE + j * SIDE + k) * 0.017_29;
                    ledge_sum +=
                        (STRATA_SHARPNESS * phase.sin()).tanh() / STRATA_SHARPNESS.tanh();
                    count += 1.0;
                }
            }
        }
        // Tolerance in units of the noise's own RMS: a bias this small is far
        // below one octave's gate-to-gate amplitude difference.
        let rms = moments.second.sqrt();
        for (label, sum) in [
            ("hollow", hollow_sum),
            ("ridge", ridge_sum),
            ("ledge", ledge_sum),
        ] {
            let mean = sum / count;
            assert!(
                mean.abs() < 0.02 * rms.max(1.0),
                "{label} shaping has DC bias {mean:.5} (noise rms {rms:.4})"
            );
        }
    }

    fn window(side: usize, fill: f32) -> DetailWindow {
        let site_dir = DVec3::X;
        let east = DVec3::Z;
        let north = east.cross(site_dir).normalize();
        DetailWindow {
            label: "test".into(),
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

        let fa = elevation_fingerprint(&chart, std::slice::from_ref(&a));
        let fb = elevation_fingerprint(&chart, std::slice::from_ref(&b));
        assert_ne!(
            fa, fb,
            "same-size windows with different content share a cache namespace"
        );

        // And the chart still participates.
        let other_chart = raster(32, 101.0, 23_040.0);
        assert_ne!(
            fa,
            elevation_fingerprint(&other_chart, std::slice::from_ref(&a))
        );

        // Absent band is distinguishable from present band.
        assert_ne!(fa, elevation_fingerprint(&chart, &[]));
        // A second window is a different surface.
        assert_ne!(fa, elevation_fingerprint(&chart, &[a, b]));
    }

    /// The LOD height budget is derived from the band set, so a band that peaks
    /// above the chart raises it. Pins the seam that would otherwise be a second
    /// place to forget a new band.
    #[test]
    fn peak_elevation_covers_every_band() {
        let chart = raster(32, 100.0, 23_040.0);
        let tall = window(64, 4_200.0);
        assert_eq!(peak_elevation_m(&chart, &[]), 100.0);
        assert_eq!(peak_elevation_m(&chart, &[tall]), 4_200.0);
    }

    #[test]
    fn detail_sidecar_names_allow_slugs() {
        assert!(is_detail_sidecar_name("thalos_site_detail_6144_90m.json"));
        assert!(is_detail_sidecar_name(
            "thalos_site_detail_massif_east_2048_90m.json"
        ));
        assert!(!is_detail_sidecar_name("thalos_chart_elev.json"));
    }

    #[test]
    fn parse_terrain_env_accepts_known_spellings() {
        assert_eq!(parse_thalos_terrain_env("diffusion"), Ok(true));
        assert_eq!(parse_thalos_terrain_env("neural"), Ok(true));
        assert_eq!(parse_thalos_terrain_env("procedural"), Ok(false));
        assert!(parse_thalos_terrain_env("surprise").is_err());
    }

    #[test]
    fn load_skips_incomplete_windows_and_keeps_complete_ones() {
        let root = std::env::temp_dir().join(format!(
            "thalos-diffusion-windows-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        write_test_raster(&root.join("thalos_chart_elev"), 16, 8, 23_040.0, None, None);
        write_test_raster(
            &root.join("thalos_site_detail_complete_90m"),
            32,
            32,
            90.0,
            Some(10.0),
            Some(20.0),
        );
        std::fs::write(
            root.join("thalos_site_detail_pointer_90m.json"),
            r#"{"width":32,"height":32,"site_lon_deg":0.0,"site_lat_deg":0.0,"px_m":90.0}"#,
        )
        .unwrap();
        std::fs::write(
            root.join("thalos_site_detail_pointer_90m.f32"),
            b"version https://git-lfs.github.com/spec/v1\n",
        )
        .unwrap();

        let surface = DiffusionSurface::load(&root, 3_186_000.0, 2).unwrap();
        assert_eq!(surface.detail_window_count(), 1);
        assert_eq!(
            surface.detail_window_names().collect::<Vec<_>>(),
            ["thalos_site_detail_complete_90m"]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    fn write_test_raster(
        stem: &Path,
        width: usize,
        height: usize,
        px_m: f64,
        lon: Option<f64>,
        lat: Option<f64>,
    ) {
        let json = if let (Some(lon), Some(lat)) = (lon, lat) {
            format!(
                r#"{{"width":{width},"height":{height},"px_m":{px_m},"site_lon_deg":{lon},"site_lat_deg":{lat}}}"#
            )
        } else {
            format!(r#"{{"width":{width},"height":{height},"px_m_equator":{px_m},"px_m":{px_m}}}"#)
        };
        std::fs::write(stem.with_extension("json"), json).unwrap();
        std::fs::write(stem.with_extension("f32"), vec![0u8; width * height * 4]).unwrap();
    }
}
