//! Diagnostic probe: how far does the **waterline** (terrain height == 0
//! crossing) move as the render LOD changes?
//!
//! The in-game ocean is analytic; the visible shore is wherever the *rendered
//! terrain* crosses sea level (0 m). Rendered tiles are sampled at a
//! per-tile `lod_m` that depends on camera distance (coarse from orbit, fine
//! up close), so if the coastal height field is LOD-dependent the waterline
//! shifts horizontally as you fly in — exactly the "coastline shape/distance
//! differs with camera distance" symptom.
//!
//! This walks a great-circle arc across a coast, finds the zero crossing at a
//! range of `lod_m` values (mirroring the game's `tile_lod_m` for LOD 0..15 on
//! Thalos), and reports how many metres the crossing moves between LODs, plus
//! the local coastal slope (which sets how much a given height error becomes a
//! horizontal shift).
//!
//! Run: `cargo run -p thalos_terrain --example coastline_lod`

use glam::DVec3;
use thalos_terrain::ProceduralSurface;
use thalos_terrain::query::SurfaceQuery;

const RADIUS_M: f64 = 3_186_000.0; // Thalos
const SEED: u32 = 1;

// tile_lod_m for LOD 0..15 (mirrors body_render's `tile_lod_m`):
//   radius * (pi/2 / 2^lod) / inner_texels,  inner_texels = 512 - 2*2 = 508.
fn lod_m_for(lod: u32) -> f32 {
    let inner_texels = 508.0_f64;
    let face_radians = std::f64::consts::FRAC_PI_2 / (1u64 << lod) as f64;
    ((RADIUS_M * face_radians / inner_texels).max(1.0)) as f32
}

/// Build an orthonormal tangent basis at `dir`.
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

/// Rotate `dir` by `ang` radians toward `tangent` (great-circle step).
fn step_dir(dir: DVec3, tangent: DVec3, ang: f64) -> DVec3 {
    (dir * ang.cos() + tangent * ang.sin()).normalize()
}

fn main() {
    let surface = ProceduralSurface::new(RADIUS_M as f32, SEED);
    println!("Thalos radius {RADIUS_M:.0} m, seed {SEED}");
    println!("LOD → lod_m (game tile sampling scale):");
    for lod in [0u32, 3, 6, 9, 12, 15] {
        println!("  LOD {lod:>2}: lod_m = {:>9.2} m", lod_m_for(lod));
    }
    println!();

    // The lod_m ladder we test the waterline against (coarse orbit → fine ground).
    let lods: Vec<f32> = vec![
        lod_m_for(0),
        lod_m_for(2),
        lod_m_for(4),
        lod_m_for(6),
        lod_m_for(8),
        lod_m_for(10),
        lod_m_for(13),
        0.3,
    ];

    // Find several coastlines by scanning a Fibonacci-sphere set of arc centres
    // for a sign change of height (evaluated at fine detail) along local north.
    let n_probe = 4000usize;
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut found = 0usize;
    let arc_span_m = 40_000.0_f64; // ±40 km arc around the crossing
    let steps = 4000usize;

    for i in 0..n_probe {
        if found >= 8 {
            break;
        }
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / n_probe as f64;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * i as f64;
        let center = DVec3::new(theta.cos() * r, y, theta.sin() * r).normalize();
        let (_east, north) = tangent_basis(center);

        // Coarse scan north for a land/sea sign change near the centre.
        let h_here = surface.sample_height_m(center.as_vec3(), 0.3);
        let ahead = step_dir(center, north, 6_000.0 / RADIUS_M);
        let h_ahead = surface.sample_height_m(ahead.as_vec3(), 0.3);
        if h_here.signum() == h_ahead.signum() {
            continue;
        }
        found += 1;

        // Measure the waterline crossing along the arc at each lod_m.
        // Arc parameter s in metres, from -arc_span to +arc_span around centre.
        let crossing_at = |lod_m: f32| -> Option<f64> {
            let mut prev_s = -arc_span_m;
            let mut prev_h = surface
                .sample_height_m(step_dir(center, north, prev_s / RADIUS_M).as_vec3(), lod_m);
            for k in 1..=steps {
                let s = -arc_span_m + 2.0 * arc_span_m * k as f64 / steps as f64;
                let h =
                    surface.sample_height_m(step_dir(center, north, s / RADIUS_M).as_vec3(), lod_m);
                if h.signum() != prev_h.signum() {
                    // Linear interpolate the zero crossing in s.
                    let t = prev_h as f64 / (prev_h as f64 - h as f64);
                    return Some(prev_s + (s - prev_s) * t);
                }
                prev_s = s;
                prev_h = h;
            }
            None
        };

        // Local coastal slope at the fine crossing (m rise / m horizontal).
        let fine_cross = crossing_at(0.3);
        let slope = fine_cross.map(|s0| {
            let d = 50.0;
            let hp = surface
                .sample_height_m(step_dir(center, north, (s0 + d) / RADIUS_M).as_vec3(), 0.3);
            let hm = surface
                .sample_height_m(step_dir(center, north, (s0 - d) / RADIUS_M).as_vec3(), 0.3);
            ((hp - hm) as f64 / (2.0 * d)).abs()
        });

        println!(
            "── Coast #{found} at dir ({:+.3},{:+.3},{:+.3})  coastal slope ≈ {:.4} m/m ({:.1} m per km)",
            center.x,
            center.y,
            center.z,
            slope.unwrap_or(f64::NAN),
            slope.unwrap_or(f64::NAN) * 1000.0,
        );

        let mut positions = Vec::new();
        for &lod_m in &lods {
            match crossing_at(lod_m) {
                Some(s) => {
                    println!("     lod_m {lod_m:>9.2} m → waterline at s = {s:>+9.1} m");
                    positions.push(s);
                }
                None => {
                    println!("     lod_m {lod_m:>9.2} m → no crossing in ±{arc_span_m:.0} m arc")
                }
            }
        }
        if positions.len() >= 2 {
            let min = positions.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = positions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "     ⇒ waterline moves {:.0} m horizontally across LODs (orbit↔ground)",
                max - min
            );
        }
        println!();
    }

    if found == 0 {
        println!("No coastline found in probe set — adjust the scan.");
    }
}
