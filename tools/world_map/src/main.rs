//! Whole-planet map export of `ProceduralSurface`, for iterating on the
//! continent / ocean macro shape **and the biome geography** without launching
//! the game.
//!
//! Two render modes over two projections:
//!
//! - `WORLD_MODE=biome` (default) — the planet in its **true in-game macro
//!   palette** (`sample_d().albedo_linear`, exactly what the impostor / orbit
//!   view shades from), hillshaded, plus a second flat **biome-class map**
//!   (`ProceduralSurface::sample_biome_d`) with a 30° graticule. Prints
//!   area-weighted per-biome coverage (% of planet / % of land) so palette and
//!   climate tuning has numbers, not just pixels.
//! - `WORLD_MODE=hypso` — the legacy hypsometric + moisture-tinted ramp.
//!
//! - `WORLD_PROJ=mercator` (default) — web-mercator (±85.05°), square output.
//! - `WORLD_PROJ=equirect` — the old equirectangular frame (includes poles).
//!
//! Run (defaults to Thalos: radius 3,186 km, seed 2 = its body id):
//!   just map                    # cargo run --release -p thalos_world_map
//! Override: `WORLD_SEED=7 WORLD_RADIUS_KM=2000 WORLD_W=4096 ...`
//! Output: target/world_map.png (+ target/world_biomes.png in biome mode).
//!
//! Also: `WORLD_ZOOM="lat,lon[,half_km]"` tangent-plane crop,
//! `WORLD_TRANSECT="lat,lon,az_deg,length_km"` height profile (unchanged).

use glam::DVec3;
use rayon::prelude::*;
use thalos_terrain::query::{SurfaceQuery, SurfaceSample};
use thalos_terrain::{DiffusionSurface, MacroBiome, ProceduralSurface};

const OUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/world_map.png");
const OUT_BIOMES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/world_biomes.png");

// Runway site (matches `thalos_runtime::runway` / the ProceduralSurface scaffold).
const RUNWAY_LAT_DEG: f64 = 7.6;
const RUNWAY_LON_DEG: f64 = 178.0;
/// One site for both terrain backings (the conditioned diffusion export keeps
/// the canonical continents, and the site holds — see `runway.rs`).
fn runway_site_latlon() -> (f64, f64) {
    (RUNWAY_LAT_DEG, RUNWAY_LON_DEG)
}

/// Web-mercator latitude clamp: `atan(sinh(π))` ≈ 85.05113°, which makes the
/// full-range map exactly square.
const MERC_MAX_Y: f64 = std::f64::consts::PI;

fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

#[derive(Clone, Copy, PartialEq)]
enum Proj {
    Mercator,
    Equirect,
}

/// The lat/lon frame of the output image: per-row latitude (pixel centres),
/// per-row sphere-area weight, and forward projection for markers/graticule.
struct Frame {
    w: usize,
    h: usize,
    proj: Proj,
    /// Pixel-centre latitude per row (degrees, north at row 0).
    lat_deg: Vec<f64>,
    /// Relative sphere-area weight of one pixel in this row: equirect rows
    /// span equal dφ (weight ∝ cos φ); mercator rows span dφ ∝ cos φ
    /// (weight ∝ cos² φ).
    area_w: Vec<f64>,
}

impl Frame {
    fn new(proj: Proj, w: usize) -> Self {
        let h = match proj {
            Proj::Mercator => w,
            Proj::Equirect => w / 2,
        };
        let (lat_deg, area_w): (Vec<f64>, Vec<f64>) = (0..h)
            .map(|j| {
                let v = (j as f64 + 0.5) / h as f64; // 0 top → 1 bottom
                match proj {
                    Proj::Mercator => {
                        let y = MERC_MAX_Y * (1.0 - 2.0 * v);
                        let lat = y.sinh().atan();
                        (lat.to_degrees(), lat.cos().powi(2))
                    }
                    Proj::Equirect => {
                        let lat_deg = 90.0 - v * 180.0;
                        (lat_deg, lat_deg.to_radians().cos().max(0.0))
                    }
                }
            })
            .unzip();
        Self {
            w,
            h,
            proj,
            lat_deg,
            area_w,
        }
    }

    fn lon_deg(&self, i: usize) -> f64 {
        -180.0 + (i as f64 + 0.5) / self.w as f64 * 360.0
    }

    /// Forward projection: (lat, lon) in degrees → pixel indices (may be
    /// outside the image for out-of-range latitudes under mercator).
    fn project(&self, lat_deg: f64, lon_deg: f64) -> (i64, i64) {
        let i = ((lon_deg + 180.0) / 360.0 * self.w as f64) as i64;
        let j = match self.proj {
            Proj::Mercator => {
                let y = lat_deg.to_radians().tan().asinh();
                ((MERC_MAX_Y - y) / (2.0 * MERC_MAX_Y) * self.h as f64) as i64
            }
            Proj::Equirect => ((90.0 - lat_deg) / 180.0 * self.h as f64) as i64,
        };
        (i, j)
    }

    /// Ground metres per pixel (east, north) at row `j`, for the hillshade.
    fn m_per_px(&self, radius_m: f64, j: usize) -> (f64, f64) {
        let lat = self.lat_deg[j].to_radians();
        let mx = std::f64::consts::TAU * radius_m * lat.cos().max(1e-3) / self.w as f64;
        let jd = j.saturating_sub(1);
        let ju = (j + 1).min(self.h - 1);
        let dlat = (self.lat_deg[jd] - self.lat_deg[ju]).to_radians() / (ju - jd).max(1) as f64;
        (mx, radius_m * dlat)
    }
}

#[derive(Clone, Copy)]
struct Px {
    height_m: f32,
    moisture: f32,
    albedo: [f32; 3],
    biome: MacroBiome,
}

/// The tool's surface: the canonical procedural planet, or the NTR-X2a
/// terrain-diffusion backing when `THALOS_TERRAIN=diffusion` (same toggle and
/// data directory as the game's `BodySurfaceRegistry`).
enum MapSurface {
    Procedural(ProceduralSurface),
    Diffusion(DiffusionSurface),
}

impl MapSurface {
    fn sample_biome_d(&self, dir: DVec3, lod_m: f32) -> (SurfaceSample, MacroBiome) {
        match self {
            Self::Procedural(s) => s.sample_biome_d(dir, lod_m),
            Self::Diffusion(s) => s.sample_biome_d(dir, lod_m),
        }
    }

    fn query(&self) -> &dyn SurfaceQuery {
        match self {
            Self::Procedural(s) => s,
            Self::Diffusion(s) => s,
        }
    }
}

/// Attach the baked drainage raster when one is installed for this backing
/// (NTR-X2q). Absent is fine — rivers are an optional landcover channel — but a
/// raster baked from the *other* backing is a hard error, because rivers that
/// followed different terrain run visibly uphill.
fn attach_rivers(surface: ProceduralSurface, backing: &str) -> ProceduralSurface {
    let dir = std::path::Path::new("assets/terrain_packages/thalos_rivers");
    match thalos_terrain::RiverField::load(dir, backing) {
        Ok(Some(r)) => {
            println!("world_map: rivers attached ({} backing)", r.backing);
            surface.with_rivers(std::sync::Arc::new(r))
        }
        Ok(None) => surface,
        Err(e) => {
            println!("world_map: rivers NOT attached — {e}");
            surface
        }
    }
}

fn main() {
    let radius_m = env_f64("WORLD_RADIUS_KM")
        .map(|km| km * 1000.0)
        .unwrap_or(3_186_000.0);
    let seed = env_f64("WORLD_SEED").map(|s| s as u32).unwrap_or(2);
    let diffusion = std::env::var("THALOS_TERRAIN")
        .map(|v| v.trim().eq_ignore_ascii_case("diffusion"))
        .unwrap_or(false);
    let surface = if diffusion {
        let dir = std::path::Path::new("assets/terrain_packages/thalos_diffusion");
        match DiffusionSurface::load(dir, radius_m as f32, seed) {
            Ok(s) => {
                println!("world_map: terrain-diffusion surface ({})", dir.display());
                let rd = std::path::Path::new("assets/terrain_packages/thalos_rivers");
                let s = match thalos_terrain::RiverField::load(rd, "diffusion") {
                    Ok(Some(r)) => {
                        println!("world_map: rivers attached (diffusion backing)");
                        s.with_rivers(std::sync::Arc::new(r))
                    }
                    Ok(None) => s,
                    Err(e) => {
                        println!("world_map: rivers NOT attached — {e}");
                        s
                    }
                };
                MapSurface::Diffusion(s)
            }
            Err(e) => {
                eprintln!("THALOS_TERRAIN=diffusion: {e}; falling back to procedural");
                MapSurface::Procedural(attach_rivers(
                    ProceduralSurface::new(radius_m as f32, seed),
                    "procedural",
                ))
            }
        }
    } else {
        MapSurface::Procedural(attach_rivers(
            ProceduralSurface::new(radius_m as f32, seed),
            "procedural",
        ))
    };

    // Zoom mode: WORLD_ZOOM="lat,lon,half_km" renders a tangent-plane crop
    // (finer LOD) instead of the global map, to check that coastlines and
    // relief stay fractal/believable as you descend toward the surface.
    if let Ok(spec) = std::env::var("WORLD_ZOOM") {
        render_zoom(surface.query(), radius_m, &spec);
        return;
    }

    // Transect mode: WORLD_TRANSECT="lat,lon,az_deg,length_km" walks a straight
    // tangent line and prints a height profile — used to measure shelf width and
    // continental-slope steepness (is the land→abyss transition a natural ramp
    // or a cliff?).
    if let Ok(spec) = std::env::var("WORLD_TRANSECT") {
        print_transect(surface.query(), radius_m, &spec);
        return;
    }

    let proj = match std::env::var("WORLD_PROJ").as_deref() {
        Ok("equirect") => Proj::Equirect,
        _ => Proj::Mercator,
    };
    let biome_mode = !matches!(std::env::var("WORLD_MODE").as_deref(), Ok("hypso"));
    let w = env_f64("WORLD_W")
        .map(|v| v as usize)
        .unwrap_or(2048)
        .max(64);
    let frame = Frame::new(proj, w);
    let (w, h) = (frame.w, frame.h);

    // Coarse LOD: the macro continent/ocean/biome shape is LOD-invariant, so
    // this just controls how many relief octaves come in.
    let lod_m = 4_000.0_f32;

    println!(
        "world_map: radius {:.0} km, seed {}, height_range ±{:.0} m, {}x{} {} ({})",
        radius_m / 1000.0,
        seed,
        surface.query().height_range_m(),
        w,
        h,
        if proj == Proj::Mercator {
            "mercator ±85°"
        } else {
            "equirect"
        },
        if biome_mode {
            "biome palette"
        } else {
            "hypso ramp"
        },
    );

    // Pass 1 (parallel rows): one full sample per pixel — height, macro
    // moisture, the true in-game macro albedo, and the dominant biome class.
    let px: Vec<Px> = (0..h)
        .into_par_iter()
        .flat_map_iter(|j| {
            let lat = frame.lat_deg[j];
            let frame = &frame;
            let surface = &surface;
            (0..w).map(move |i| {
                let dir = latlon_dir(lat, frame.lon_deg(i));
                let (s, biome) = surface.sample_biome_d(dir, lod_m);
                Px {
                    height_m: s.height_m,
                    moisture: s.moisture,
                    albedo: s.albedo_linear.to_array(),
                    biome,
                }
            })
        })
        .collect();

    // Area-weighted stats: land fraction + per-biome coverage.
    let mut total_area = 0.0f64;
    let mut land_area = 0.0f64;
    let mut biome_area = [0.0f64; MacroBiome::ALL.len()];
    let (mut hmin, mut hmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for j in 0..h {
        let wlat = frame.area_w[j];
        for i in 0..w {
            let p = &px[j * w + i];
            total_area += wlat;
            if p.height_m >= 0.0 {
                land_area += wlat;
            }
            let bi = MacroBiome::ALL
                .iter()
                .position(|b| *b == p.biome)
                .unwrap_or(0);
            biome_area[bi] += wlat;
            hmin = hmin.min(p.height_m as f64);
            hmax = hmax.max(p.height_m as f64);
        }
    }
    let land_frac = land_area / total_area.max(1.0);
    println!(
        "land fraction {:.1}%   height min {:.0} m  max {:.0} m{}",
        land_frac * 100.0,
        hmin,
        hmax,
        if proj == Proj::Mercator {
            "   (stats over the ±85° window)"
        } else {
            ""
        },
    );

    let mut ranked: Vec<(MacroBiome, f64)> = MacroBiome::ALL
        .iter()
        .enumerate()
        .map(|(i, b)| (*b, biome_area[i] / total_area.max(1.0)))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("biome coverage (% of planet | % of land):");
    for (b, frac) in ranked.iter().filter(|(_, f)| *f > 0.0005) {
        let of_land = if *b == MacroBiome::Ocean {
            "    —".to_string()
        } else {
            format!("{:5.1}", frac / (land_frac).max(1e-9) * 100.0)
        };
        println!("  {:<9} {:5.1} | {}", b.label(), frac * 100.0, of_land);
    }

    // Land dryness distribution — the tuning diagnostic for the moisture
    // geography: the palette transfer lives at dryness 0.55 (dry-grass onset),
    // ~0.72 (steppe dominates), 0.88 (bare soil / desert). If p90 never
    // reaches the upper thresholds, no belt tuning of the *palette* will ever
    // produce deserts.
    let mut dry: Vec<f32> = px
        .iter()
        .filter(|p| p.height_m >= 0.0)
        .map(|p| 0.5 - 0.5 * p.moisture)
        .collect();
    if !dry.is_empty() {
        dry.sort_by(f32::total_cmp);
        let pct = |q: f64| dry[((dry.len() - 1) as f64 * q) as usize];
        let above = |t: f32| dry.iter().filter(|d| **d > t).count() as f64 / dry.len() as f64;
        println!(
            "land dryness: p10 {:.2}  p50 {:.2}  p90 {:.2}  |  >0.55 {:.1}%  >0.72 {:.1}%  >0.88 {:.1}%",
            pct(0.10),
            pct(0.50),
            pct(0.90),
            above(0.55) * 100.0,
            above(0.72) * 100.0,
            above(0.88) * 100.0,
        );
    }

    // Per-|latitude|-band breakdown: where the dry land actually is, and what
    // claims it — the climate-belt tuning view.
    println!("per-|lat| band (land): mean dryness | %>0.72 | top biome");
    for band in 0..6 {
        let (lo, hi) = (band as f64 * 15.0, band as f64 * 15.0 + 15.0);
        let mut n = 0.0f64;
        let mut dsum = 0.0f64;
        let mut ndry = 0.0f64;
        let mut counts = [0.0f64; MacroBiome::ALL.len()];
        for j in 0..h {
            let alat = frame.lat_deg[j].abs();
            if alat < lo || alat >= hi {
                continue;
            }
            let wlat = frame.area_w[j];
            for i in 0..w {
                let p = &px[j * w + i];
                if p.height_m < 0.0 {
                    continue;
                }
                let d = (0.5 - 0.5 * p.moisture) as f64;
                n += wlat;
                dsum += wlat * d;
                if d > 0.72 {
                    ndry += wlat;
                }
                let bi = MacroBiome::ALL
                    .iter()
                    .position(|b| *b == p.biome)
                    .unwrap_or(0);
                counts[bi] += wlat;
            }
        }
        if n < 1.0 {
            continue;
        }
        let top = counts
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, c)| format!("{} {:.0}%", MacroBiome::ALL[i].label(), c / n * 100.0))
            .unwrap_or_default();
        println!(
            "  {:2.0}–{:2.0}°  {:.2} | {:4.1}% | {}",
            lo,
            hi,
            dsum / n,
            ndry / n * 100.0,
            top
        );
    }

    // Runway-site report.
    let (site_lat, site_lon) = runway_site_latlon();
    let site = latlon_dir(site_lat, site_lon);
    let site_h = surface.query().sample_height_m(site.as_vec3(), 30.0) as f64;
    println!(
        "runway site (lat {:.1}, lon {:.1}): height {:.0} m  ({})",
        site_lat,
        site_lon,
        site_h,
        if site_h >= 0.0 {
            "LAND"
        } else {
            "OCEAN — fix the bias!"
        }
    );

    // Pass 2: the shaded map. Biome mode renders land in the true macro
    // palette (linear → sRGB) so the map IS the impostor/orbit transfer;
    // ocean keeps the legibility depth ramp (in-game water colour comes from
    // the water renderer, not the terrain albedo). Hypso mode keeps the
    // legacy climate-shifted ramp.
    let light = DVec3::new(-0.5, -0.4, 0.75).normalize();
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for j in 0..h {
        let (mx, my) = frame.m_per_px(radius_m, j);
        for i in 0..w {
            let p = &px[j * w + i];
            let z = p.height_m as f64;
            let base = if z < 0.0 {
                hypso_color(z)
            } else if biome_mode {
                srgb8(p.albedo)
            } else {
                let lat = frame.lat_deg[j];
                let sin_lat = lat.to_radians().sin().abs();
                let cold_lift = thalos_terrain::climate_cold_lift_m(sin_lat);
                let warmth = thalos_terrain::climate_warmth(cold_lift);
                moisture_tinted(
                    hypso_color(z + cold_lift),
                    z + cold_lift,
                    p.moisture,
                    warmth,
                )
            };
            let shade = if z < 0.0 {
                1.0
            } else {
                let il = i.saturating_sub(1);
                let ir = (i + 1).min(w - 1);
                let jd = j.saturating_sub(1);
                let ju = (j + 1).min(h - 1);
                let dzdx = (px[j * w + ir].height_m - px[j * w + il].height_m) as f64 / (mx * 2.0);
                let dzdy = (px[ju * w + i].height_m - px[jd * w + i].height_m) as f64 / (my * 2.0);
                let normal = DVec3::new(-dzdx, -dzdy, 1.0).normalize();
                (normal.dot(light).max(0.0) * 0.7 + 0.3).clamp(0.0, 1.0)
            };
            img.put_pixel(
                i as u32,
                j as u32,
                image::Rgb([
                    (base[0] as f64 * shade) as u8,
                    (base[1] as f64 * shade) as u8,
                    (base[2] as f64 * shade) as u8,
                ]),
            );
        }
    }
    mark_site(&mut img, &frame);
    img.save(OUT).expect("save png");
    println!("wrote {OUT}");

    // Flat biome-class map + graticule: the tuning view (which class claims
    // which region), companion to the true-colour render.
    if biome_mode {
        let mut cls = image::RgbImage::new(w as u32, h as u32);
        for j in 0..h {
            for i in 0..w {
                cls.put_pixel(
                    i as u32,
                    j as u32,
                    image::Rgb(biome_color(px[j * w + i].biome)),
                );
            }
        }
        draw_graticule(&mut cls, &frame);
        mark_site(&mut cls, &frame);
        cls.save(OUT_BIOMES).expect("save png");
        println!("wrote {OUT_BIOMES}");
    }
}

/// Flat class colours for the biome map (sRGB) — deliberately more saturated
/// than the render palette so regions read at a glance.
fn biome_color(b: MacroBiome) -> [u8; 3] {
    match b {
        MacroBiome::Ocean => [24, 52, 100],
        MacroBiome::Beach => [222, 205, 156],
        MacroBiome::Grassland => [110, 170, 78],
        MacroBiome::Steppe => [190, 168, 106],
        MacroBiome::Desert => [238, 202, 118],
        MacroBiome::Forest => [30, 100, 44],
        MacroBiome::Tundra => [150, 156, 130],
        MacroBiome::Upland => [124, 138, 106],
        MacroBiome::Rock => [128, 126, 124],
        MacroBiome::Snow => [242, 246, 250],
    }
}

/// Linear-RGB → sRGB8.
fn srgb8(c: [f32; 3]) -> [u8; 3] {
    let enc = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    [enc(c[0]), enc(c[1]), enc(c[2])]
}

/// 30° graticule (equator emphasised) so regions are addressable — the
/// coordinates feed straight back into `WORLD_ZOOM` / `WORLD_TRANSECT`.
fn draw_graticule(img: &mut image::RgbImage, frame: &Frame) {
    let blend = |img: &mut image::RgbImage, i: i64, j: i64, t: f64| {
        if i >= 0 && (i as u32) < img.width() && j >= 0 && (j as u32) < img.height() {
            let p = img.get_pixel_mut(i as u32, j as u32);
            for c in 0..3 {
                p.0[c] = (p.0[c] as f64 * (1.0 - t)) as u8;
            }
        }
    };
    for lon in (-150..=150).step_by(30) {
        let (i, _) = frame.project(0.0, lon as f64);
        for j in 0..frame.h as i64 {
            blend(img, i, j, 0.25);
        }
    }
    for lat in (-60..=60).step_by(30) {
        let (_, j) = frame.project(lat as f64, 0.0);
        let t = if lat == 0 { 0.4 } else { 0.25 };
        for i in 0..frame.w as i64 {
            blend(img, i, j, t);
        }
    }
}

/// Red cross at the fixed runway site.
fn mark_site(img: &mut image::RgbImage, frame: &Frame) {
    let (site_lat, site_lon) = runway_site_latlon();
    let (si, sj) = frame.project(site_lat, site_lon);
    for d in -10..=10i64 {
        for (a, b) in [(si + d, sj), (si, sj + d)] {
            if a >= 0 && a < frame.w as i64 && b >= 0 && b < frame.h as i64 {
                img.put_pixel(a as u32, b as u32, image::Rgb([255, 40, 40]));
            }
        }
    }
}

/// Tangent-plane crop centred on `lat,lon` with `half_km` half-extent, at a LOD
/// matched to the pixel spacing so the relief cascade fades in as it would near
/// the camera. Writes target/world_zoom.png.
fn render_zoom(surface: &dyn SurfaceQuery, radius_m: f64, spec: &str) {
    let parts: Vec<f64> = spec
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (clat, clon, half_km) = match parts.as_slice() {
        [a, b, c] => (*a, *b, *c),
        [a, b] => (*a, *b, 200.0),
        _ => {
            eprintln!("WORLD_ZOOM must be \"lat,lon[,half_km]\"");
            return;
        }
    };
    let center = latlon_dir(clat, clon);
    let up = center;
    let east = DVec3::Y.cross(up).normalize();
    let north = up.cross(east).normalize();
    let half_m = half_km * 1000.0;
    let n = 1024usize;
    let px = 2.0 * half_m / (n - 1) as f64;
    // LOD ≈ pixel spacing, so the cascade resolves what the grid can show.
    let lod_m = px.max(0.5) as f32;

    let zoom_biome = matches!(std::env::var("WORLD_MODE").as_deref(), Ok("biome"));
    let sample = |ex: f64, ny: f64| -> f64 {
        let p = up * radius_m + east * ex + north * ny;
        surface.sample_height_m(p.normalize().as_vec3(), lod_m) as f64
    };

    let mut h = vec![0.0f64; n * n];
    let (mut hmin, mut hmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for j in 0..n {
        let ny = (j as f64 / (n - 1) as f64 - 0.5) * 2.0 * half_m;
        for i in 0..n {
            let ex = (i as f64 / (n - 1) as f64 - 0.5) * 2.0 * half_m;
            let v = sample(ex, ny);
            h[j * n + i] = v;
            hmin = hmin.min(v);
            hmax = hmax.max(v);
        }
    }
    println!(
        "world_zoom: center (lat {clat}, lon {clon}), ±{half_km} km, lod {lod_m:.1} m  \
         height min {hmin:.0} m  max {hmax:.0} m"
    );

    let light = DVec3::new(-0.5, -0.4, 0.75).normalize();
    let mut img = image::RgbImage::new(n as u32, n as u32);
    for j in 0..n {
        for i in 0..n {
            let z = h[j * n + i];
            let il = i.saturating_sub(1);
            let ir = (i + 1).min(n - 1);
            let jd = j.saturating_sub(1);
            let ju = (j + 1).min(n - 1);
            let dzdx = (h[j * n + ir] - h[j * n + il]) / (px * (ir - il) as f64);
            let dzdy = (h[ju * n + i] - h[jd * n + i]) / (px * (ju - jd) as f64);
            let normal = DVec3::new(-dzdx, -dzdy, 1.0).normalize();
            let shade = (normal.dot(light).max(0.0) * 0.8 + 0.2).clamp(0.0, 1.0);
            // `WORLD_MODE=biome` colours by the real landcover albedo, which is
            // the only view that can show landcover-only channels such as the
            // riparian band (NTR-X2q) — the hypso ramp is a function of height
            // alone and is structurally blind to them.
            let base = if zoom_biome {
                let ex = (i as f64 / (n - 1) as f64 - 0.5) * 2.0 * half_m;
                let ny = (j as f64 / (n - 1) as f64 - 0.5) * 2.0 * half_m;
                let d = (up * radius_m + east * ex + north * ny).normalize();
                srgb8(surface.sample_d(d, lod_m).albedo_linear.to_array())
            } else {
                hypso_color(z)
            };
            img.put_pixel(
                i as u32,
                j as u32,
                image::Rgb([
                    (base[0] as f64 * shade) as u8,
                    (base[1] as f64 * shade) as u8,
                    (base[2] as f64 * shade) as u8,
                ]),
            );
        }
    }
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/world_zoom.png");
    img.save(out).expect("save png");
    println!("wrote {out}  (north = up, east = right)");
}

/// Walk a straight tangent line and print a height profile (a depth/altitude
/// transect), so the shelf width and continental-slope steepness are legible as
/// numbers, not just pixels.
fn print_transect(surface: &dyn SurfaceQuery, radius_m: f64, spec: &str) {
    let parts: Vec<f64> = spec
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let [clat, clon, az_deg, len_km] = parts.as_slice() else {
        eprintln!("WORLD_TRANSECT must be \"lat,lon,az_deg,length_km\"");
        return;
    };
    let center = latlon_dir(*clat, *clon);
    let east = DVec3::Y.cross(center).normalize();
    let north = center.cross(east).normalize();
    let az = az_deg.to_radians();
    let dir_step = (north * az.cos() + east * az.sin()).normalize();
    let len_m = len_km * 1000.0;
    let steps = 60usize;
    println!("transect from (lat {clat}, lon {clon}) az {az_deg}° over {len_km} km:");
    let mut prev_h = None::<f64>;
    for i in 0..=steps {
        let d = (i as f64 / steps as f64 - 0.5) * len_m;
        let p = center * radius_m + dir_step * d;
        let h = surface.sample_height_m(p.normalize().as_vec3(), 60.0) as f64;
        let slope = prev_h.map(|p| (h - p) / (len_m / steps as f64) * 1000.0); // m per km
        println!(
            "{:+6.0} km  {:+6.0} m {} {}",
            d / 1000.0,
            h,
            if h >= 0.0 { "L" } else { "~" },
            slope.map(|s| format!("({s:+.0} m/km)")).unwrap_or_default(),
        );
        prev_h = Some(h);
    }
}

/// Hypsometric colour ramp (sRGB-ish, for eyeballing). Ocean = depth ramp;
/// land = the runway_relief band ramp with a beach at the coast.
fn hypso_color(z: f64) -> [u8; 3] {
    if z < 0.0 {
        // Deep abyss → shelf → coast.
        let bands = [
            (-4000.0, [8, 18, 48]),  // abyss
            (-2000.0, [16, 36, 78]), //
            (-400.0, [28, 64, 110]), // slope
            (-120.0, [44, 96, 140]), // shelf
            (0.0, [70, 130, 165]),   // shallow / coast
        ];
        return ramp(z, &bands);
    }
    // Land bands mirror the in-game ecological transfer (landcover.wgsl):
    // green through the lush belt (~1.5 km eco), grey-green upland toward the
    // treeline (~2.4–3.0 km), scree grey, then snow (~3.1–4.0 km) — the old
    // "upland brown at 900 m" band painted temperate uplands brown and made
    // the map lie about how the game renders them.
    let bands = [
        (0.0, [205, 200, 150]),    // beach
        (60.0, [70, 120, 60]),     // green lowland
        (1500.0, [88, 118, 66]),   // upper lush belt
        (2400.0, [110, 118, 95]),  // grey-green upland (treeline)
        (3000.0, [150, 148, 142]), // alpine scree
        (3300.0, [235, 238, 245]), // snow
        (4800.0, [255, 255, 255]),
    ];
    ramp(z, &bands)
}

fn ramp(z: f64, bands: &[(f64, [u8; 3])]) -> [u8; 3] {
    if z <= bands[0].0 {
        return bands[0].1;
    }
    for w in bands.windows(2) {
        let (z0, c0) = w[0];
        let (z1, c1) = w[1];
        if z <= z1 {
            let t = ((z - z0) / (z1 - z0)).clamp(0.0, 1.0);
            return [
                lerp_u8(c0[0], c1[0], t),
                lerp_u8(c0[1], c1[1], t),
                lerp_u8(c0[2], c1[2], t),
            ];
        }
    }
    bands[bands.len() - 1].1
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t).round() as u8
}

/// `smoothstep` accepting descending edges (WGSL-style), for the mirror above.
fn smoothstep64(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Blend the low-altitude land ramp by macro moisture: dry regions toward tan
/// (hot-desert sand while `warmth` is high, cold steppe otherwise), wet
/// regions toward dark forest green. Fades out with (eco) altitude — uplands
/// keep the hypsometric ramp — and leaves the ocean untouched.
fn moisture_tinted(base: [u8; 3], eco_z: f64, moisture: f32, warmth: f64) -> [u8; 3] {
    if eco_z < 0.0 {
        return base;
    }
    let low = 1.0 - ((eco_z - 400.0) / 800.0).clamp(0.0, 1.0);
    // Mirror the in-game transfer (landcover.wgsl `vegetation_color`): dryness
    // reads as tan only past ~0.55 and as sand/soil past ~0.8, forest ramps in
    // below dryness ~0.58 — so the map is a faithful proxy of the render, not
    // an exaggeration of the raw field.
    let dryness = (0.5 - 0.5 * moisture as f64).clamp(0.0, 1.0);
    let (target, t) = if dryness > 0.5 {
        let steppe = [150u8, 138u8, 96u8];
        let sand = [205u8, 178u8, 120u8];
        let sand_t = smoothstep64(0.80, 0.95, dryness) * warmth;
        let dry = [
            lerp_u8(steppe[0], sand[0], sand_t),
            lerp_u8(steppe[1], sand[1], sand_t),
            lerp_u8(steppe[2], sand[2], sand_t),
        ];
        (dry, smoothstep64(0.55, 0.88, dryness) * 0.85)
    } else {
        ([34u8, 78u8, 36u8], smoothstep64(0.42, 0.18, dryness) * 0.70)
    };
    let t = t * low;
    [
        lerp_u8(base[0], target[0], t),
        lerp_u8(base[1], target[1], t),
        lerp_u8(base[2], target[2], t),
    ]
}
