//! Diagnostic probe: how much of the offshore shelf renders ABOVE sea level
//! (h > 0) as the sampling LOD coarsens?
//!
//! The in-game ocean is an analytic sphere at h = 0; any terrain sample that
//! comes out above 0 renders as land. If the LOD-aware relief cascade still
//! carries enough amplitude at coarse `lod_m` to breach shallow shelves, the
//! breach cap turns broad offshore areas into speckle fields of 0..+14 m
//! "skerries" that appear/disappear with camera distance — the dotted
//! water/terrain artifact seen from orbit.
//!
//! For several coasts: walk transects offshore from the fine-LOD waterline and
//! report, per lod_m, the fraction of offshore samples with h > 0 and the mean
//! breach height.
//!
//! Run: `cargo run -p thalos_terrain --example shelf_breach_probe`

use glam::DVec3;
use thalos_terrain::query::SurfaceQuery;
use thalos_terrain::{DiffusionSurface, ProceduralSurface};

const RADIUS_M: f64 = 3_186_000.0; // Thalos
// Thalos's body id — the seed the game and `just map` use. A different
// value here is a different planet, which silently makes any A/B against the
// diffusion backing (loaded with body id 2) a comparison of two worlds.
const SEED: u32 = 2;

/// Same toggle the game and `just map` use — see `coastline_lod.rs`.
fn load_surface() -> Box<dyn SurfaceQuery> {
    let diffusion = std::env::var("THALOS_TERRAIN")
        .map(|v| v.trim().eq_ignore_ascii_case("diffusion"))
        .unwrap_or(false);
    if diffusion {
        let dir = std::path::Path::new("assets/terrain_packages/thalos_diffusion");
        match DiffusionSurface::load(dir, RADIUS_M as f32, 2) {
            Ok(surface) => {
                println!("backing: terrain-diffusion ({})", dir.display());
                return Box::new(surface);
            }
            Err(error) => {
                println!("backing: procedural (diffusion load failed: {error})");
            }
        }
    } else {
        println!("backing: procedural");
    }
    Box::new(ProceduralSurface::new(RADIUS_M as f32, SEED))
}

fn lod_m_for(lod: u32) -> f32 {
    let inner_texels = 508.0_f64;
    let face_radians = std::f64::consts::FRAC_PI_2 / (1u64 << lod) as f64;
    ((RADIUS_M * face_radians / inner_texels).max(1.0)) as f32
}

fn tangent_basis(dir: DVec3) -> (DVec3, DVec3) {
    let up = if dir.y.abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let east = up.cross(dir).normalize();
    let north = dir.cross(east).normalize();
    (east, north)
}

fn step_dir(dir: DVec3, tangent: DVec3, ang: f64) -> DVec3 {
    (dir * ang.cos() + tangent * ang.sin()).normalize()
}

fn main() {
    let surface = load_surface();
    let surface = surface.as_ref();
    let lods: Vec<(String, f32)> = [0u32, 2, 4, 6, 8, 10]
        .iter()
        .map(|&l| (format!("LOD{l:>2} ({:>8.1} m)", lod_m_for(l)), lod_m_for(l)))
        .chain(std::iter::once(("fine  (     0.3 m)".to_string(), 0.3f32)))
        .collect();

    let n_probe = 4000usize;
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut found = 0usize;

    for i in 0..n_probe {
        if found >= 6 {
            break;
        }
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / n_probe as f64;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * i as f64;
        let center = DVec3::new(theta.cos() * r, y, theta.sin() * r).normalize();
        let (_east, north) = tangent_basis(center);

        // Need: center on land, a point 6 km north at sea (fine LOD) → coast.
        let h_here = surface.sample_height_m(center.as_vec3(), 0.3);
        let ahead = step_dir(center, north, 6_000.0 / RADIUS_M);
        let h_ahead = surface.sample_height_m(ahead.as_vec3(), 0.3);
        if !(h_here > 0.0 && h_ahead < 0.0) {
            continue;
        }
        found += 1;

        // Find the fine waterline along north, then sample 2..40 km offshore.
        let mut s_coast = 0.0;
        for k in 0..1200 {
            let s = 6000.0 * k as f64 / 1200.0;
            if surface.sample_height_m(step_dir(center, north, s / RADIUS_M).as_vec3(), 0.3) < 0.0 {
                s_coast = s;
                break;
            }
        }

        println!(
            "── Coast #{found} at ({:+.3},{:+.3},{:+.3}), waterline at {:.0} m north",
            center.x, center.y, center.z, s_coast
        );
        for (label, lod_m) in &lods {
            let mut above = 0usize;
            let mut total = 0usize;
            let mut breach_sum = 0.0f64;
            let mut depth_sum = 0.0f64;
            for k in 0..2000 {
                let s = s_coast + 2_000.0 + 38_000.0 * k as f64 / 2000.0;
                let h = surface
                    .sample_height_m(step_dir(center, north, s / RADIUS_M).as_vec3(), *lod_m)
                    as f64;
                total += 1;
                if h > 0.0 {
                    above += 1;
                    breach_sum += h;
                } else {
                    depth_sum += h;
                }
            }
            let above_pct = 100.0 * above as f64 / total as f64;
            let mean_breach = if above > 0 {
                breach_sum / above as f64
            } else {
                0.0
            };
            let mean_depth = if above < total {
                depth_sum / (total - above) as f64
            } else {
                0.0
            };
            println!(
                "     {label}: {above_pct:5.1}% of offshore samples ABOVE sea \
                 (mean breach {mean_breach:+6.1} m, mean wet depth {mean_depth:+7.1} m)"
            );
        }
        println!();
    }

    if found == 0 {
        println!("No land→sea coast found in probe set.");
    }
}
