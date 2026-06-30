//! Whole-planet "orbital view" map of `ProceduralSurface`, for iterating on the
//! continent / ocean macro shape without launching the game.
//!
//! Renders an equirectangular hypsometric + hillshaded map of the body, prints
//! the area-weighted land fraction (so the `CONTINENT_C0` threshold can be tuned
//! to a target), and marks the fixed runway site so we can confirm it lands on
//! dry, flat ground.
//!
//! Run (defaults to Thalos: radius 3,186 km, seed 2 = its body id):
//!   cargo run -p thalos_terrain --example world_map
//! Override: `WORLD_SEED=7 WORLD_RADIUS_KM=2000 cargo run ... --example world_map`
//! Output: target/world_map.png (next to the path in the `OUT` const).

use glam::DVec3;
use thalos_terrain::ProceduralSurface;
use thalos_terrain::query::SurfaceQuery;

const OUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/world_map.png");

// Runway site (matches `thalos_game::runway` / the ProceduralSurface scaffold).
const RUNWAY_LAT_DEG: f64 = 7.6;
const RUNWAY_LON_DEG: f64 = 178.0;

fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

fn main() {
    let radius_m = env_f64("WORLD_RADIUS_KM").map(|km| km * 1000.0).unwrap_or(3_186_000.0);
    let seed = env_f64("WORLD_SEED").map(|s| s as u32).unwrap_or(2);
    let surface = ProceduralSurface::new(radius_m as f32, seed);

    // Zoom mode: WORLD_ZOOM="lat,lon,half_km" renders a tangent-plane crop
    // (finer LOD) instead of the global equirect map, to check that coastlines
    // and relief stay fractal/believable as you descend toward the surface.
    if let Ok(spec) = std::env::var("WORLD_ZOOM") {
        render_zoom(&surface, radius_m, &spec);
        return;
    }

    // Transect mode: WORLD_TRANSECT="lat,lon,az_deg,length_km" walks a straight
    // tangent line and prints a height profile — used to measure shelf width and
    // continental-slope steepness (is the land→abyss transition a natural ramp
    // or a cliff?).
    if let Ok(spec) = std::env::var("WORLD_TRANSECT") {
        print_transect(&surface, radius_m, &spec);
        return;
    }

    // Equirectangular grid. Coarse LOD: the macro continent/ocean shape is
    // LOD-invariant, so this just controls how many relief octaves come in.
    let w = 2048usize;
    let h = 1024usize;
    let lod_m = 4_000.0_f32;

    println!(
        "world_map: radius {:.0} km, seed {}, height_range ±{:.0} m",
        radius_m / 1000.0,
        seed,
        surface.height_range_m()
    );

    // First pass: heights + area-weighted land fraction + extremes.
    let mut height = vec![0.0f64; w * h];
    let mut land_area = 0.0f64;
    let mut total_area = 0.0f64;
    let mut hmin = f64::INFINITY;
    let mut hmax = f64::NEG_INFINITY;
    for j in 0..h {
        // Pixel-centre latitude; cos(lat) is the area weight of the row.
        let lat = 90.0 - (j as f64 + 0.5) / h as f64 * 180.0;
        let wlat = lat.to_radians().cos().max(0.0);
        for i in 0..w {
            let lon = -180.0 + (i as f64 + 0.5) / w as f64 * 360.0;
            let dir = latlon_dir(lat, lon);
            let z = surface.sample_height_m(dir.as_vec3(), lod_m) as f64;
            height[j * w + i] = z;
            total_area += wlat;
            if z >= 0.0 {
                land_area += wlat;
            }
            hmin = hmin.min(z);
            hmax = hmax.max(z);
        }
    }
    let land_frac = land_area / total_area.max(1.0);
    println!(
        "land fraction {:.1}%   height min {:.0} m  max {:.0} m",
        land_frac * 100.0,
        hmin,
        hmax
    );

    // Runway-site report.
    let site = latlon_dir(RUNWAY_LAT_DEG, RUNWAY_LON_DEG);
    let site_h = surface.sample_height_m(site.as_vec3(), 30.0) as f64;
    println!(
        "runway site (lat {:.1}, lon {:.1}): height {:.0} m  ({})",
        RUNWAY_LAT_DEG,
        RUNWAY_LON_DEG,
        site_h,
        if site_h >= 0.0 { "LAND" } else { "OCEAN — fix the bias!" }
    );

    // Second pass: hypsometric colour + cheap hillshade from the height buffer.
    // Hillshade scale: degrees → metres along each axis (rough, just for relief
    // legibility, not a calibrated normal).
    let m_per_px_x = std::f64::consts::PI * radius_m / w as f64;
    let m_per_px_y = std::f64::consts::PI * radius_m / h as f64;
    let light = DVec3::new(-0.5, -0.4, 0.75).normalize();

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for j in 0..h {
        for i in 0..w {
            let z = height[j * w + i];
            let il = i.saturating_sub(1);
            let ir = (i + 1).min(w - 1);
            let jd = j.saturating_sub(1);
            let ju = (j + 1).min(h - 1);
            let dzdx = (height[j * w + ir] - height[j * w + il]) / (m_per_px_x * 2.0);
            let dzdy = (height[ju * w + i] - height[jd * w + i]) / (m_per_px_y * 2.0);
            let normal = DVec3::new(-dzdx, -dzdy, 1.0).normalize();
            let shade = (normal.dot(light).max(0.0) * 0.7 + 0.3).clamp(0.0, 1.0);

            let base = hypso_color(z);
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

    // Mark the runway site with a red cross.
    let si = (((RUNWAY_LON_DEG + 180.0) / 360.0) * w as f64) as i64;
    let sj = (((90.0 - RUNWAY_LAT_DEG) / 180.0) * h as f64) as i64;
    for d in -10..=10i64 {
        for (a, b) in [(si + d, sj), (si, sj + d)] {
            if a >= 0 && a < w as i64 && b >= 0 && b < h as i64 {
                img.put_pixel(a as u32, b as u32, image::Rgb([255, 40, 40]));
            }
        }
    }

    img.save(OUT).expect("save png");
    println!("wrote {OUT}");
}

/// Tangent-plane crop centred on `lat,lon` with `half_km` half-extent, at a LOD
/// matched to the pixel spacing so the relief cascade fades in as it would near
/// the camera. Writes target/world_zoom.png.
fn render_zoom(surface: &ProceduralSurface, radius_m: f64, spec: &str) {
    let parts: Vec<f64> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
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
            let base = hypso_color(z);
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
fn print_transect(surface: &ProceduralSurface, radius_m: f64, spec: &str) {
    let parts: Vec<f64> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
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
            (-4000.0, [8, 18, 48]),   // abyss
            (-2000.0, [16, 36, 78]),  //
            (-400.0, [28, 64, 110]),  // slope
            (-120.0, [44, 96, 140]),  // shelf
            (0.0, [70, 130, 165]),    // shallow / coast
        ];
        return ramp(z, &bands);
    }
    let bands = [
        (0.0, [205, 200, 150]),    // beach
        (60.0, [70, 120, 60]),     // green lowland
        (400.0, [95, 125, 62]),    //
        (900.0, [120, 110, 80]),   // upland brown
        (1800.0, [122, 112, 100]), // rock
        (2600.0, [150, 145, 140]), // bare rock
        (3200.0, [235, 238, 245]), // snow
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
