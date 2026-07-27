//! The canonical **canopy stand structure** — the one field that decides where
//! forest stands actually sit *within* the climate envelope that permits them.
//!
//! # Why this lives here
//!
//! Canopy coverage is the product of two independent things:
//!
//! - the **climate envelope** — "can closed forest grow here at all", the
//!   moisture/treeline-driven `forest` band of
//!   [`procedural`](crate::procedural)'s macro landcover, and
//! - the **stand structure** — "where inside that envelope are the actual
//!   stands, and where are the plains and glades between them", this module.
//!
//! Both were already computed, but in *different crates*: the envelope in
//! `thalos_terrain`, driving the baked macro albedo's dark canopy anchor; the
//! stand structure in `thalos_body_render::ground::scatter`, driving tree
//! placement. Nothing tied them together, so the ground painted closed-canopy
//! green over open plains while the trees stood on ground the palette called
//! meadow — two answers to "where is the forest", visibly disagreeing from the
//! air. (The old `scatter::lattice_value` doc claimed `body_terrain.wgsl`
//! mirrored this noise so the tint would line up; no such mirror existed in
//! either terrain shader.)
//!
//! So the stand structure moves behind the terrain seam, the envelope keeps
//! owning climate, and their product — [`MaterialBands::canopy`] — is the
//! single canopy authority every consumer reads:
//!
//! | Consumer | Reads it as |
//! |---|---|
//! | tree / shrub / ground-cover placement | [`SurfaceQuery::canopy_coverage`] |
//! | baked macro albedo (the dark canopy anchor) | mixed at this weight |
//! | tile material (aerial canopy grain, understory un-mix) | per-vertex `MaterialBands::canopy` |
//! | grass far-ring cull (grass returns in the glades) | `SurfaceQuery::canopy_coverage` |
//!
//! Carrying it as **vertex data** rather than re-deriving it in WGSL is what
//! makes the agreement structural: there is one evaluation, on the CPU, and no
//! shader-side mirror of this noise to drift out of sync.
//!
//! [`MaterialBands::canopy`]: crate::query::MaterialBands::canopy
//! [`SurfaceQuery::canopy_coverage`]: crate::query::SurfaceQuery::canopy_coverage

use glam::DVec3;

/// The **slowly-varying** half of canopy coverage: the climate terms that are
/// expensive to evaluate but change over tens of kilometres.
///
/// Coverage decomposes by *evaluation cost*, not by scale:
///
/// | Term | Cost | Varies over | Where it's evaluated |
/// |---|---|---|---|
/// | moisture, orogeny | domain warp + several fBm octaves | ~100 km | **once per tile** — this struct |
/// | altitude chain (coast / treeline / snowline) | a few `smoothstep`s on the candidate's own height | metres | per candidate |
/// | stand structure | cheap value noise | ~400 m | per candidate |
///
/// The altitude chain deliberately stays per-candidate even though it is the
/// cheap part: hoisting it would quantise the treeline and the shoreline to the
/// tile grid, and a treeline that staircases in 200 m steps is far more visible
/// than a moisture field that lags by one tile.
///
/// Hoist this with [`SurfaceQuery::canopy_climate`], then call
/// [`Self::coverage`] per candidate. A one-off caller can use
/// [`SurfaceQuery::canopy_coverage`], which does both.
///
/// [`SurfaceQuery::canopy_climate`]: crate::query::SurfaceQuery::canopy_climate
/// [`SurfaceQuery::canopy_coverage`]: crate::query::SurfaceQuery::canopy_coverage
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CanopyClimate {
    pub(crate) moisture: f64,
    pub(crate) orogeny: f64,
}

impl CanopyClimate {
    /// Canopy coverage in `[0, 1]` at `dir`, for ground standing at `height_m`.
    ///
    /// **The one definition of coverage.** Everything else — the per-vertex
    /// `MaterialBands::canopy`, the albedo bake's anchor weight, vegetation
    /// placement, the grass cull — is this number, reached by a different route.
    pub fn coverage(&self, dir: DVec3, height_m: f32) -> f32 {
        let cold_lift = crate::procedural::climate_cold_lift_m(dir.y.abs());
        let warmth = crate::procedural::climate_warmth(cold_lift);
        crate::procedural::macro_band_ts(
            dir,
            f64::from(height_m),
            self.orogeny,
            self.moisture,
            cold_lift,
            warmth,
        )
        .material_bands()
        .canopy
    }
}

/// Angular frequency of the stand-structure mask. At planet radius this sets
/// the patch scale: `~ radius / FREQ` metres per lattice cell (≈ 400 m on a
/// ~3200 km body), so stands read as groves the player can walk between — not
/// the near-constant ~100 km blanket a low frequency gives.
pub const STAND_FREQ: f64 = 8000.0;

/// Stand window over the raw fBM mask — the centre of the ecotone ramp in
/// [`canopy_stand`], which widens it by −0.06 / +0.12 for a soft transition.
///
/// **Calibrated against the value-noise distribution of [`stand_fbm`]**, which
/// is roughly symmetric about 0.5 — *not* against a Perlin basis in `[-1, 1]`.
/// If the basis ever changes, re-measure the distribution before re-picking
/// these; a threshold inherited across a basis change is the classic way to
/// silently lose (or flood) coverage.
///
/// Deliberately high: only the upper part of the noise carries forest, so
/// stands read as groves over mostly-open ground rather than the
/// near-continuous blanket a centred `smoothstep(0.40, 0.60)` gives. Lower
/// [`STAND_LO`] for more forest, raise it for emptier plains.
pub const STAND_LO: f32 = 0.52;
/// Upper edge of the stand window — see [`STAND_LO`].
pub const STAND_HI: f32 = 0.72;

/// Glade noise frequency, as a multiple of [`STAND_FREQ`]: the medium-scale
/// field that carves internal clearings inside a stand. ≈130 m glades on a
/// ~3200 km body — clearing-sized, so a stand interior isn't a solid fill.
pub const GLADE_FREQ_MUL: f64 = 3.0;

/// Position-only **stand structure** in `[0, 1]`: how much of a closed canopy
/// this spot carries from stand geometry alone, before any climate envelope or
/// per-sample terrain coupling. Two scales, so forest reads as landscape rather
/// than a stamped patch:
///
/// - a **wide ecotone ramp** on the large-scale stand field (no hard edge, no
///   flat plateau), so density feathers in over a broad transition band, and
/// - a **medium-scale glade field** that opens real internal clearings and
///   breaks the uniform interior.
///
/// This is deliberately *not* the whole answer to "is there forest here" —
/// multiply it by the climate envelope (see the module docs). Terrain-form
/// coupling (denser hollows, thinner ridges) needs a height sample this
/// position-only field never sees and stays with the placement gate.
pub fn canopy_stand(dir: DVec3) -> f32 {
    let mask = stand_fbm(dir);
    // Wide ecotone: the stand ramps in gradually over a broad band around the
    // STAND_LO/HI window, so edges thin out instead of cutting off.
    let stand = smoothstep(STAND_LO - 0.06, STAND_HI + 0.12, mask);
    // Glades: medium-scale dips open clearings within the stand. Closed canopy
    // is the majority (most of the field is above the upper edge), glades the
    // minority — but they break the solid interior and feather the margins.
    let glade = value_fbm(dir * (STAND_FREQ * GLADE_FREQ_MUL), 3);
    let canopy = smoothstep(0.30, 0.60, glade);
    stand * canopy
}

/// The raw domain-warped stand mask (≈`[0, 1]`, centred near 0.5), before the
/// [`STAND_LO`]/[`STAND_HI`] contrast. Warp-then-sample so patch edges aren't
/// grid-aligned.
pub fn stand_fbm(dir: DVec3) -> f32 {
    let warp = f64::from(value_fbm(dir * (STAND_FREQ * 0.45), 2));
    value_fbm(dir * STAND_FREQ + DVec3::splat(warp * 5.0), 4)
}

/// Value-noise fBM over an already-frequency-scaled position. Cheap,
/// deterministic, no trig — a placement/coverage mask, never terrain height.
fn value_fbm(p: DVec3, octaves: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 0.5f32;
    let mut freq = 1.0f64;
    let mut norm = 0.0f32;
    for _ in 0..octaves.max(1) {
        sum += amp * value_noise(p * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm.max(f32::EPSILON)
}

/// Trilinearly-interpolated value noise in `[0, 1]`.
fn value_noise(p: DVec3) -> f32 {
    let pf = p.floor();
    let (ix, iy, iz) = (pf.x as i64, pf.y as i64, pf.z as i64);
    let f = p - pf;
    let s = DVec3::new(smooth01(f.x), smooth01(f.y), smooth01(f.z));
    let c = |dx: i64, dy: i64, dz: i64| lattice_value(ix + dx, iy + dy, iz + dz);
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), s.x as f32);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), s.x as f32);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), s.x as f32);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), s.x as f32);
    let y0 = lerp(x00, x10, s.y as f32);
    let y1 = lerp(x01, x11, s.y as f32);
    lerp(y0, y1, s.z as f32)
}

/// u32 integer hash. Kept free of `u64` so it stays trivially portable, but
/// note that nothing mirrors it in WGSL and nothing should: canopy coverage
/// reaches shaders as vertex data (see the module docs), which is what keeps
/// the CPU and GPU answers identical by construction.
fn lattice_value(ix: i64, iy: i64, iz: i64) -> f32 {
    let x = (ix as i32) as u32;
    let y = (iy as i32) as u32;
    let z = (iz as i32) as u32;
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(z.wrapping_mul(2_246_822_519));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    h as f32 / 4_294_967_296.0
}

fn smooth01(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stand field must be a *structure*, not a constant: a body-scale
    /// sweep has to produce both closed canopy and genuinely open ground.
    /// Guards the failure mode where a basis or threshold change collapses the
    /// field toward its mean (everything half-forested) — the shape of bug that
    /// reads as "the forest is everywhere and nowhere".
    #[test]
    fn stand_field_spans_open_and_closed() {
        let mut closed = 0;
        let mut open = 0;
        let n = 40;
        for i in 0..n {
            for j in 0..n {
                let u = (i as f64 + 0.5) / n as f64 * std::f64::consts::TAU;
                let v = ((j as f64 + 0.5) / n as f64) * 2.0 - 1.0;
                let r = (1.0 - v * v).max(0.0).sqrt();
                let dir = DVec3::new(r * u.cos(), v, r * u.sin());
                let c = canopy_stand(dir);
                assert!((0.0..=1.0).contains(&c), "stand out of range: {c}");
                if c > 0.6 {
                    closed += 1;
                }
                if c < 0.05 {
                    open += 1;
                }
            }
        }
        assert!(closed > 0, "no closed-canopy stands anywhere on the body");
        assert!(open > 0, "no open ground anywhere on the body");
    }

    /// Deterministic: the placement gate and the albedo bake evaluate this from
    /// different call sites and must agree exactly.
    #[test]
    fn stand_field_is_deterministic() {
        let dir = DVec3::new(0.31, 0.62, -0.72).normalize();
        assert_eq!(canopy_stand(dir), canopy_stand(dir));
    }

    /// A flatten pad levels **height**, not climate: canopy coverage must survive
    /// the [`FlattenedSurface`](crate::query::FlattenedSurface) wrapper untouched.
    ///
    /// Thalos's surface is always wrapped for the spaceport pad, so a decorator
    /// that forgets to forward a defaulted seam method silently zeroes it
    /// *planet-wide* — which is exactly what happened here: coverage read 0.0
    /// everywhere, no tree placed anywhere, and no error anywhere.
    #[test]
    fn flatten_decorator_passes_canopy_through() {
        use crate::procedural::ProceduralSurface;
        use crate::query::{FlattenedSurface, SurfaceQuery, flatten_handle};
        use std::sync::Arc;

        const R: f32 = 3_186_000.0;
        let inner: Arc<dyn SurfaceQuery> = Arc::new(ProceduralSurface::new(R, 0x5EED));
        // An empty handle is the transparent case, which is precisely the one
        // that must not zero the field: the wrapper is present on Thalos
        // whether or not any pad is installed.
        let wrapped = FlattenedSurface::new(Arc::clone(&inner), flatten_handle());

        let mut saw_canopy = false;
        for j in 0..16 {
            let a = 0.9 + j as f64 * 0.41;
            let b = -0.3 + j as f64 * 0.11;
            let dir = DVec3::new(a.cos() * b.cos(), b.sin(), a.sin() * b.cos()).normalize();
            let h = inner.sample_d(dir, 16.0).height_m;
            let bare = inner.canopy_coverage(dir, h, 16.0);
            saw_canopy |= bare > 0.01;
            assert_eq!(
                bare,
                wrapped.canopy_coverage(dir, h, 16.0),
                "FlattenedSurface dropped canopy_coverage at {dir:?}"
            );
        }
        // Guard the guard: an all-zero sweep would pass the equality trivially.
        assert!(saw_canopy, "sample sweep found no canopy to compare");
    }

    /// **The invariant this whole module exists for.** Canopy coverage reaches
    /// consumers two ways — per-vertex via `MaterialBands::canopy` (the tile
    /// mesh, and the weight the albedo bake mixed its canopy anchor at) and
    /// per-candidate via `SurfaceQuery::canopy_coverage` (vegetation placement,
    /// grass cull). The second takes a deliberately cheaper route to the *same*
    /// band evaluation, so the two must return the identical number. If they
    /// ever diverge, the ground paints forest where nothing grows again — which
    /// is exactly the defect that motivated unifying the fields.
    #[test]
    fn point_query_matches_per_vertex_band() {
        use crate::procedural::ProceduralSurface;
        use crate::query::SurfaceQuery;

        let surface = ProceduralSurface::new(3_186_000.0, 0x5EED);
        for (i, lod_m) in [0.5_f32, 16.0, 512.0].into_iter().enumerate() {
            for j in 0..12 {
                let a = 0.7 + i as f64 * 0.31 + j as f64 * 0.53;
                let b = -0.4 + j as f64 * 0.19;
                let dir = DVec3::new(a.cos() * b.cos(), b.sin(), a.sin() * b.cos()).normalize();
                let (sample, bands) = surface.sample_bands_d(dir, lod_m);
                let point = surface.canopy_coverage(dir, sample.height_m, lod_m);
                assert_eq!(
                    bands.canopy, point,
                    "canopy disagreement at dir={dir:?} lod={lod_m}: \
                     per-vertex {} vs point query {point}",
                    bands.canopy
                );
            }
        }
    }
}
