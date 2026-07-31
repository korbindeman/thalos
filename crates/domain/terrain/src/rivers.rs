//! NTR-X2q(a) — the baked river channel, read as **landcover wetness**.
//!
//! Drainage cannot be computed in the field: flow accumulation is a global,
//! ordered, downhill traversal and `SurfaceQuery` is a per-point black box with
//! no neighbourhood. So `examples/bake_rivers.rs` computes it once offline and
//! this reads the result.
//!
//! It is **wetness, not water**. The analytic ocean is one sphere at r=R and
//! draws exactly one water level (ADR-20260720T185954Z), so a river at 300 m
//! cannot be rendered as a water surface. What it can do is drive moisture: a
//! trunk river crossing a dry belt becomes a green ribbon through steppe, which
//! is what a river looks like from the air. Rendered water at arbitrary
//! altitude is step (c) of NTR-X2q and a renderer decision.
//!
//! **The raster is backing-specific.** Rivers follow the terrain they were
//! baked on; loading a raster baked on the other backing puts them running
//! uphill. The sidecar records which, and [`RiverField::load`] refuses a
//! mismatch rather than shipping wrong geography quietly.

use glam::DVec3;

/// Baked catchment, u8 log-scaled, equirect, row 0 = north pole edge.
impl std::fmt::Debug for RiverField {
    /// Never print the payload — it is 50 MB.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RiverField")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("backing", &self.backing)
            .finish_non_exhaustive()
    }
}

pub struct RiverField {
    width: usize,
    height: usize,
    /// `255 * log10(catchment_km2) / log_decades`, 0 where there is no land.
    data: Vec<u8>,
    log_decades: f64,
    /// Which surface backing this was baked from — `"procedural"`/`"diffusion"`.
    pub backing: String,
}

fn json_num(json: &str, key: &str) -> Option<f64> {
    json.split(&format!("\"{key}\""))
        .nth(1)?
        .trim_start_matches([':', ' '])
        .split([',', '}', '\n'])
        .next()?
        .trim()
        .parse()
        .ok()
}

fn json_str(json: &str, key: &str) -> Option<String> {
    let rest = json.split(&format!("\"{key}\""))
        .nth(1)?
        .trim_start_matches([':', ' ']);
    let rest = rest.strip_prefix('"')?;
    Some(rest.split('"').next()?.to_owned())
}

impl RiverField {
    /// Load the raster for `backing`, or `Ok(None)` when none is installed —
    /// rivers are an optional channel and their absence is not an error.
    ///
    /// A raster baked from a *different* backing **is** an error: it would put
    /// rivers on the wrong terrain, which reads as a bug in the terrain rather
    /// than in the asset.
    pub fn load(dir: &std::path::Path, backing: &str) -> Result<Option<Self>, String> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(None);
        };
        let mut best: Option<(u64, std::path::PathBuf)> = None;
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(px) = name
                .strip_prefix("thalos_rivers_")
                .and_then(|r| r.strip_suffix("m.json"))
                .and_then(|s| s.parse::<u64>().ok())
                && best.as_ref().is_none_or(|(b, _)| px < *b)
            {
                // Finest available wins.
                best = Some((px, e.path()));
            }
        }
        let Some((_, json_path)) = best else {
            return Ok(None);
        };
        let json = std::fs::read_to_string(&json_path).map_err(|e| format!("rivers sidecar: {e}"))?;
        let width = json_num(&json, "width").ok_or("rivers sidecar: width")? as usize;
        let height = json_num(&json, "height").ok_or("rivers sidecar: height")? as usize;
        let log_decades = json_num(&json, "log_decades").unwrap_or(7.0);
        let baked = json_str(&json, "backing").unwrap_or_default();
        if baked != backing {
            return Err(format!(
                "river raster was baked from the {baked:?} backing but this body renders \
                 {backing:?} — rivers would run uphill. Re-bake with \
                 `THALOS_TERRAIN={backing} cargo run -p thalos_terrain --release --example bake_rivers`"
            ));
        }
        let data = std::fs::read(json_path.with_extension("u8"))
            .map_err(|e| format!("rivers payload: {e}"))?;
        if data.len() != width * height {
            return Err(format!(
                "rivers payload is {} bytes, expected {width}x{height}",
                data.len()
            ));
        }
        Ok(Some(Self {
            width,
            height,
            data,
            log_decades,
            backing: baked,
        }))
    }

    fn sample_u8(&self, dir: DVec3) -> f64 {
        let lat = dir.y.clamp(-1.0, 1.0).asin();
        let lon = dir.z.atan2(dir.x).rem_euclid(core::f64::consts::TAU);
        let fx = lon / core::f64::consts::TAU * self.width as f64 - 0.5;
        let fy = (0.5 - lat / core::f64::consts::PI) * self.height as f64 - 0.5;
        let (x0, tx) = (fx.floor(), fx - fx.floor());
        let (y0, ty) = (fy.floor(), fy - fy.floor());
        let xi = |k: i64| -> usize {
            let m = self.width as i64;
            (((x0 as i64 + k) % m + m) % m) as usize
        };
        let yi = |k: i64| -> usize {
            (y0 as i64 + k).clamp(0, self.height as i64 - 1) as usize
        };
        let at = |x: usize, y: usize| self.data[y * self.width + x] as f64;
        let (x_0, x_1, y_0, y_1) = (xi(0), xi(1), yi(0), yi(1));
        let a = at(x_0, y_0) + (at(x_1, y_0) - at(x_0, y_0)) * tx;
        let b = at(x_0, y_1) + (at(x_1, y_1) - at(x_0, y_1)) * tx;
        a + (b - a) * ty
    }

    /// Upstream catchment in km² at `dir`, 0 off-network.
    pub fn catchment_km2(&self, dir: DVec3) -> f64 {
        let v = self.sample_u8(dir.normalize_or_zero());
        if v <= 0.0 {
            return 0.0;
        }
        10f64.powf(v / 255.0 * self.log_decades)
    }

    /// Smooth ramp of catchment between two thresholds, `[0, 1]`.
    fn ramp(&self, dir: DVec3, lo_km2: f64, hi_km2: f64) -> f64 {
        let km2 = self.catchment_km2(dir);
        if km2 <= lo_km2 {
            return 0.0;
        }
        let t = ((km2.log10() - lo_km2.log10()) / (hi_km2.log10() - lo_km2.log10())).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// **Riparian corridor** — the vegetated floodplain either side of a river.
    ///
    /// Starts at [`RIPARIAN_CORRIDOR_LO_KM2`]. That threshold is measured, not
    /// chosen: Horton-Strahler on the baked network gives Rb 4.87 (Earth 3-5),
    /// so the *structure* is right, and what made it read as "too frequent" was
    /// drawing every order-1 gully. A 10^3 km² head puts 3.4 % of land in
    /// channel; 3x10^4 puts 0.49 %, which is the density an atlas draws.
    pub fn corridor(&self, dir: DVec3) -> f64 {
        self.ramp(dir, RIPARIAN_CORRIDOR_LO_KM2, RIPARIAN_CORRIDOR_HI_KM2)
    }

    /// **Channel core** — the wet bed itself.
    ///
    /// This is how hierarchy gets its *width*. Catchment falls off away from a
    /// channel, so a higher threshold is automatically a narrower line: the
    /// corridor is broad along a trunk and vanishes on a creek, and the core
    /// only appears on trunks at all. Brightness alone conveys no hierarchy —
    /// that was the whole of the "doesn't feel hierarchical" report — so the
    /// band must vary in extent, not just in strength.
    pub fn channel(&self, dir: DVec3) -> f64 {
        self.ramp(dir, RIPARIAN_CHANNEL_LO_KM2, RIPARIAN_CHANNEL_HI_KM2)
    }

    /// Landcover wetness contributed by drainage, `[0, 1]` — the moisture
    /// term. Same ramp as the corridor, so vegetation and palette agree.
    pub fn wetness(&self, dir: DVec3) -> f64 {
        self.corridor(dir)
    }
}

/// Catchment at which a floodplain starts reading as watered, and at which it
/// is fully so. See [`RiverField::corridor`] — 3x10^4 is the measured
/// atlas-density head, not a guess.
const RIPARIAN_CORRIDOR_LO_KM2: f64 = 30_000.0;
const RIPARIAN_CORRIDOR_HI_KM2: f64 = 1_000_000.0;
/// The wet bed appears only on trunk rivers, which is what makes it narrow.
const RIPARIAN_CHANNEL_LO_KM2: f64 = 300_000.0;
const RIPARIAN_CHANNEL_HI_KM2: f64 = 3_000_000.0;
