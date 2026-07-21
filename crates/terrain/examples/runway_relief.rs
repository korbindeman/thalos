//! Throwaway shaded-relief renderer for tuning the authored runway mountain
//! massif in `ProceduralSurface`. Samples a top-down region around the runway
//! site and writes a hillshaded, height-coloured PNG so the mountain *shape*
//! (erosion ridges/gullies) can be judged without launching the game.
//!
//! Run: `cargo run -p thalos_terrain --example runway_relief`
//! Output: written next to the path in the `OUT` const below.

use glam::DVec3;
use thalos_terrain::ProceduralSurface;
use thalos_terrain::query::SurfaceQuery;

const OUT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/runway_relief.png"
);

fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

fn main() {
    let radius_m = 3_186_000.0_f64;
    let surface = ProceduralSurface::new(radius_m as f32, 1);

    // Region centred between the runway and the massif so both are framed.
    let (clat, clon, half_km) = match std::env::var("RELIEF_ZOOM").ok().as_deref() {
        Some("massif") => (7.105, 177.137, 85.0),
        // Tight crop over the down-runway massif to judge erosion gully *scale*
        // (the wide presets are too coarse to resolve drainage texture).
        Some("massif_zoom") => (7.105, 177.137, 18.0),
        Some("ne") => (8.48, 178.51, 100.0),
        _ => (7.35, 177.57, 130.0),
    };
    let center = latlon_dir(clat, clon);
    let up = center;
    let east = DVec3::Y.cross(up).normalize();
    let north = up.cross(east).normalize();

    let half_m = half_km * 1000.0_f64;
    let n = 1024usize;
    // Sample roughly at the pixel footprint so a tight crop actually resolves
    // fine gullies instead of LOD-averaging them away.
    let lod = (2.0 * half_m / n as f64 * 0.75).clamp(4.0, 30.0) as f32;

    let sample = |ex: f64, ny: f64| -> f64 {
        let p = up * radius_m + east * ex + north * ny;
        surface.sample_height_m(p.normalize().as_vec3(), lod) as f64
    };

    // First pass: heights + min/max.
    let mut h = vec![0.0f64; n * n];
    let mut hmin = f64::INFINITY;
    let mut hmax = f64::NEG_INFINITY;
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
    println!("height min {hmin:.0} m  max {hmax:.0} m");

    let px = 2.0 * half_m / (n - 1) as f64; // metres per pixel
    let light = DVec3::new(-0.5, -0.4, 0.75).normalize(); // toward upper-left, high

    let mut img = image::RgbImage::new(n as u32, n as u32);
    for j in 0..n {
        for i in 0..n {
            let hc = h[j * n + i];
            // Central-difference normal in the local tangent plane (x east, y north, z up).
            let il = i.saturating_sub(1);
            let ir = (i + 1).min(n - 1);
            let jd = j.saturating_sub(1);
            let ju = (j + 1).min(n - 1);
            let dzdx = (h[j * n + ir] - h[j * n + il]) / (px * (ir - il) as f64);
            let dzdy = (h[ju * n + i] - h[jd * n + i]) / (px * (ju - jd) as f64);
            let normal = DVec3::new(-dzdx, -dzdy, 1.0).normalize();
            let shade = (normal.dot(light).max(0.0) * 0.85 + 0.15).clamp(0.0, 1.0);

            let base = height_color(hc);
            let r = (base[0] as f64 * shade) as u8;
            let g = (base[1] as f64 * shade) as u8;
            let b = (base[2] as f64 * shade) as u8;
            img.put_pixel(i as u32, j as u32, image::Rgb([r, g, b]));
        }
    }

    // Mark the runway site (lat 7.6, lon 178.0) with a red cross.
    let site = latlon_dir(7.6, 178.0);
    let rel = (site - up) * radius_m; // approx tangent offset
    let sx = rel.dot(east);
    let sy = rel.dot(north);
    let pi = ((sx / (2.0 * half_m) + 0.5) * (n - 1) as f64).round() as i64;
    let pj = ((sy / (2.0 * half_m) + 0.5) * (n - 1) as f64).round() as i64;
    for d in -8..=8i64 {
        for (a, b) in [(pi + d, pj), (pi, pj + d)] {
            if a >= 0 && a < n as i64 && b >= 0 && b < n as i64 {
                img.put_pixel(a as u32, b as u32, image::Rgb([255, 0, 0]));
            }
        }
    }

    img.save(OUT).expect("save png");
    println!("wrote {OUT}");
    println!("(red cross = runway site; north = up, east = right)");
}

fn height_color(h: f64) -> [u8; 3] {
    // Linear-ish band ramp, sRGB-ish output for eyeballing.
    if h < 0.0 {
        return [30, 60, 95]; // ocean
    }
    let bands = [
        (0.0, [70, 110, 55]),      // green lowland
        (300.0, [95, 120, 60]),    //
        (900.0, [120, 110, 80]),   // upland brown
        (1800.0, [120, 112, 100]), // rock
        (2600.0, [150, 145, 140]), // bare rock
        (3200.0, [235, 238, 245]), // snow
        (4200.0, [255, 255, 255]),
    ];
    for w in bands.windows(2) {
        let (h0, c0) = w[0];
        let (h1, c1) = w[1];
        if h <= h1 {
            let t = ((h - h0) / (h1 - h0)).clamp(0.0, 1.0);
            return [
                lerp_u8(c0[0], c1[0], t),
                lerp_u8(c0[1], c1[1], t),
                lerp_u8(c0[2], c1[2], t),
            ];
        }
    }
    [255, 255, 255]
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t).round() as u8
}
