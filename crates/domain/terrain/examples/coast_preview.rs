//! Visual probe: what does a coastline actually look like on this backing?
//!
//! `just map` shows the planet and `coastline_lod` / `shelf_breach_probe`
//! measure the waterline, but neither shows a coast at the scale a player sees
//! one. This renders a plan view of a searched coastal region plus a cross-shore
//! elevation profile through its centre, so the foreshore drop and the beach
//! berm are visible as a curve rather than only as constants in the source.
//!
//! Honours `THALOS_TERRAIN=diffusion` (same toggle as the game and `just map`),
//! so it is the offline A/B for "do both backings have the same coast?".
//!
//! Run: `cargo run --release -p thalos_terrain --example coast_preview`
//!      `THALOS_TERRAIN=diffusion cargo run --release -p thalos_terrain --example coast_preview`
//! Output: `target/coast_preview[_diffusion].png`

use glam::DVec3;
use rayon::prelude::*;
use thalos_terrain::query::SurfaceQuery;
use thalos_terrain::{DiffusionSurface, ProceduralSurface};

const RADIUS_M: f64 = 3_186_000.0;
// Thalos's body id — the seed the game and `just map` use. A different
// value here is a different planet, which silently makes any A/B against the
// diffusion backing (loaded with body id 2) a comparison of two worlds.
const SEED: u32 = 2;

/// Plan-view size and ground scale. 40 m/px over 64 km: wide enough for an
/// archipelago and a bay, fine enough that the strand is several pixels.
const SIZE: usize = 1600;
const PX_M: f64 = 40.0;
/// Height of the profile panel below the plan view.
const PROFILE_H: usize = 420;
/// Profile vertical range (m about sea level).
const PROF_LO: f64 = -90.0;
const PROF_HI: f64 = 210.0;

fn load_surface() -> (Box<dyn SurfaceQuery>, &'static str) {
    let diffusion = std::env::var("THALOS_TERRAIN")
        .map(|v| v.trim().eq_ignore_ascii_case("diffusion"))
        .unwrap_or(false);
    if diffusion {
        let dir = std::path::Path::new("assets/terrain_packages/thalos_diffusion");
        match DiffusionSurface::load(dir, RADIUS_M as f32, 2) {
            Ok(surface) => return (Box::new(surface), "diffusion"),
            Err(error) => println!("diffusion load failed ({error}); falling back"),
        }
    }
    (
        Box::new(ProceduralSurface::new(RADIUS_M as f32, SEED)),
        "procedural",
    )
}

fn tangent_basis(dir: DVec3) -> (DVec3, DVec3) {
    let seed = if dir.y.abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let east = seed.cross(dir).normalize();
    let north = dir.cross(east).normalize();
    (east, north)
}

/// Direction at plan-view offset `(du, dv)` metres from `site`.
fn offset_dir(site: DVec3, east: DVec3, north: DVec3, du: f64, dv: f64) -> DVec3 {
    (site + (east * du + north * dv) / RADIUS_M).normalize()
}

/// Search for a coast worth looking at: land and sea both present within the
/// frame, and as many waterline crossings around a ring as possible — which is
/// what picks an indented, island-strewn stretch over a straight one.
fn find_interesting_coast(surface: &dyn SurfaceQuery) -> DVec3 {
    const CANDIDATES: usize = 6_000;
    const GOLDEN: f64 = 2.399_963_229_728_653;
    let ring_m = SIZE as f64 * PX_M * 0.35;

    let scored: Vec<(f64, DVec3)> = (0..CANDIDATES)
        .into_par_iter()
        .filter_map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / CANDIDATES as f64;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = GOLDEN * i as f64;
            let site = DVec3::new(theta.cos() * r, y, theta.sin() * r).normalize();
            let h0 = surface.sample_height_m(site.as_vec3(), 64.0) as f64;
            // Centre the frame near the waterline, on the land side.
            if !(0.0..=60.0).contains(&h0) {
                return None;
            }
            let (east, north) = tangent_basis(site);
            let mut crossings = 0usize;
            let mut land = 0usize;
            let mut prev: Option<bool> = None;
            for k in 0..96 {
                let a = k as f64 / 96.0 * std::f64::consts::TAU;
                let d = offset_dir(site, east, north, ring_m * a.cos(), ring_m * a.sin());
                let wet = surface.sample_height_m(d.as_vec3(), 64.0) <= 0.0;
                if !wet {
                    land += 1;
                }
                if let Some(p) = prev
                    && p != wet
                {
                    crossings += 1;
                }
                prev = Some(wet);
            }
            // Want a genuine mix of land and sea, and a complex boundary.
            if !(24..=72).contains(&land) {
                return None;
            }
            Some((crossings as f64, site))
        })
        .collect();

    scored
        .into_iter()
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, d)| d)
        .unwrap_or(DVec3::Y)
}

/// Land shading: sand at the strand, vegetation above it, rock/snow high up,
/// Lambert-lit from a fixed low sun so relief and the berm read.
fn land_color(h: f64, normal: DVec3, up: DVec3, east: DVec3, north: DVec3) -> [f64; 3] {
    let sun = (up * 0.34 + east * 0.62 + north * 0.71).normalize();
    let lambert = normal.dot(sun).max(0.0);
    let shade = 0.28 + 0.72 * lambert;

    let sand = [0.80, 0.73, 0.55];
    let grass = [0.31, 0.44, 0.24];
    let upland = [0.44, 0.42, 0.35];
    let rock = [0.55, 0.54, 0.52];

    // The strand: the beach berm's first few metres are the sand band, so a
    // working berm shows up as a visible fringe hugging every waterline.
    let t_sand = (1.0 - (h / 14.0).clamp(0.0, 1.0)).powf(1.4);
    let t_up = ((h - 250.0) / 700.0).clamp(0.0, 1.0);
    let t_rock = ((h - 900.0) / 1_400.0).clamp(0.0, 1.0);

    let mut c = [0.0; 3];
    for i in 0..3 {
        let veg = grass[i] * (1.0 - t_up) + upland[i] * t_up;
        let hi = veg * (1.0 - t_rock) + rock[i] * t_rock;
        c[i] = (hi * (1.0 - t_sand) + sand[i] * t_sand) * shade;
    }
    c
}

/// Water: depth-tinted, with the shallow band on the same e-folding the ocean
/// renderer uses, so the foreshore drop is legible as a narrow pale fringe
/// rather than a wide translucent apron.
fn water_color(depth: f64) -> [f64; 3] {
    let shallow = [0.36, 0.66, 0.70];
    let mid = [0.10, 0.30, 0.52];
    let deep = [0.03, 0.09, 0.26];
    let t1 = 1.0 - (-depth / 22.0).exp();
    let t2 = (depth / 900.0).clamp(0.0, 1.0);
    let mut c = [0.0; 3];
    for i in 0..3 {
        let a = shallow[i] * (1.0 - t1) + mid[i] * t1;
        c[i] = a * (1.0 - t2) + deep[i] * t2;
    }
    c
}

fn put(buf: &mut [u8], w: usize, x: usize, y: usize, c: [f64; 3]) {
    let o = (y * w + x) * 3;
    for i in 0..3 {
        buf[o + i] = (c[i].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
    }
}

fn main() {
    let (surface, label) = load_surface();
    let surface = surface.as_ref();
    println!("backing: {label}");

    let site = std::env::args()
        .nth(1)
        .and_then(|s| {
            let v: Vec<f64> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            (v.len() == 3).then(|| DVec3::new(v[0], v[1], v[2]).normalize())
        })
        .unwrap_or_else(|| find_interesting_coast(surface));
    let (east, north) = tangent_basis(site);
    println!(
        "site ({:+.4},{:+.4},{:+.4})  span {:.0} km at {PX_M:.0} m/px",
        site.x,
        site.y,
        site.z,
        SIZE as f64 * PX_M / 1000.0
    );

    let half = SIZE as f64 * PX_M * 0.5;
    let lod = PX_M as f32;
    let total_h = SIZE + PROFILE_H;
    let mut buf = vec![0u8; SIZE * total_h * 3];

    // --- plan view ---------------------------------------------------------
    let rows: Vec<Vec<[f64; 3]>> = (0..SIZE)
        .into_par_iter()
        .map(|y| {
            let dv = half - (y as f64 + 0.5) * PX_M;
            (0..SIZE)
                .map(|x| {
                    let du = (x as f64 + 0.5) * PX_M - half;
                    let d = offset_dir(site, east, north, du, dv);
                    let h = surface.sample_height_m(d.as_vec3(), lod) as f64;
                    if h > 0.0 {
                        // Central-difference normal in the local tangent frame.
                        let e = PX_M;
                        let hx = surface.sample_height_m(
                            offset_dir(site, east, north, du + e, dv).as_vec3(),
                            lod,
                        ) as f64
                            - surface.sample_height_m(
                                offset_dir(site, east, north, du - e, dv).as_vec3(),
                                lod,
                            ) as f64;
                        let hy = surface.sample_height_m(
                            offset_dir(site, east, north, du, dv + e).as_vec3(),
                            lod,
                        ) as f64
                            - surface.sample_height_m(
                                offset_dir(site, east, north, du, dv - e).as_vec3(),
                                lod,
                            ) as f64;
                        let n = (d * (2.0 * e) - east * hx - north * hy).normalize();
                        land_color(h, n, d, east, north)
                    } else {
                        water_color(-h)
                    }
                })
                .collect()
        })
        .collect();
    for (y, row) in rows.iter().enumerate() {
        for (x, c) in row.iter().enumerate() {
            put(&mut buf, SIZE, x, y, *c);
        }
    }

    // Scale bar: 10 km.
    let bar = (10_000.0 / PX_M) as usize;
    for x in 40..40 + bar {
        for y in SIZE - 60..SIZE - 52 {
            put(&mut buf, SIZE, x, y, [1.0, 1.0, 1.0]);
        }
    }

    // --- cross-shore profile through the centre row ------------------------
    let prof_y0 = SIZE;
    for y in prof_y0..total_h {
        for x in 0..SIZE {
            put(&mut buf, SIZE, x, y, [0.06, 0.07, 0.09]);
        }
    }
    let to_row = |h: f64| -> usize {
        let t = ((h - PROF_LO) / (PROF_HI - PROF_LO)).clamp(0.0, 1.0);
        prof_y0 + PROFILE_H - 1 - (t * (PROFILE_H - 1) as f64).round() as usize
    };
    // Sea-level reference line.
    let sea_row = to_row(0.0);
    for x in 0..SIZE {
        put(&mut buf, SIZE, x, sea_row, [0.30, 0.45, 0.60]);
    }
    // The profile is sampled at 1 m, not at the plan view's 40 m: the foreshore
    // drop and the berm are metre-scale features and would be filtered away at
    // the coarser footprint.
    let heights: Vec<f64> = (0..SIZE)
        .into_par_iter()
        .map(|x| {
            let du = (x as f64 + 0.5) * PX_M - half;
            let d = offset_dir(site, east, north, du, 0.0);
            surface.sample_height_m(d.as_vec3(), 1.0) as f64
        })
        .collect();
    for (x, &h) in heights.iter().enumerate() {
        let r = to_row(h);
        // Fill the column between the curve and sea level so land reads as a
        // body and water as a trough.
        let (lo, hi) = if r < sea_row {
            (r, sea_row)
        } else {
            (sea_row, r)
        };
        for y in lo..=hi {
            let c = if h > 0.0 {
                [0.42, 0.38, 0.26]
            } else {
                [0.10, 0.24, 0.40]
            };
            put(&mut buf, SIZE, x, y, c);
        }
        for dy in 0..3 {
            let y = (r + dy).min(total_h - 1);
            put(&mut buf, SIZE, x, y, [0.95, 0.86, 0.55]);
        }
    }

    let min_h = heights.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_h = heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let crossings = heights
        .windows(2)
        .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
        .count();
    println!(
        "profile: {crossings} waterline crossing(s) across {:.0} km, height {min_h:.1} … {max_h:.1} m",
        SIZE as f64 * PX_M / 1000.0
    );
    println!("profile panel range {PROF_LO:.0} … {PROF_HI:.0} m, sampled at 1 m footprint");

    let out = if label == "diffusion" {
        "target/coast_preview_diffusion.png"
    } else {
        "target/coast_preview.png"
    };
    image::save_buffer(
        out,
        &buf,
        SIZE as u32,
        total_h as u32,
        image::ColorType::Rgb8,
    )
    .expect("write png");
    println!("wrote {out}");
}
