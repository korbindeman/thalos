//! Throwaway: render the terrain skyline as seen *from the runway*, looking
//! down the takeoff heading, accounting for planet curvature. Confirms the
//! authored massif clears the horizon and reads as distant mountains (not a
//! tiny bump or an absurd wall) before launching the game.
//!
//! Run: `cargo run -p thalos_terrain --example runway_skyline`

use glam::DVec3;
use thalos_terrain::ProceduralSurface;
use thalos_terrain::query::SurfaceQuery;

const OUT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../target/runway_skyline.png"
);

const RUNWAY_LAT: f64 = 7.6;
const RUNWAY_LON: f64 = 178.0;
const RUNWAY_HEADING_DEG: f64 = 30.0;

fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

fn main() {
    let radius_m = 3_186_000.0_f64;
    let surface = ProceduralSurface::new(radius_m as f32, 1);

    let site = latlon_dir(RUNWAY_LAT, RUNWAY_LON);
    let up = site;
    // Match the game's runway heading basis exactly (TerrainPatchBasis +
    // `tangent_x*cos + tangent_z*sin`), so "forward" here is the real
    // down-the-runway look direction.
    let tangent_x = DVec3::Y.cross(up).normalize();
    let tangent_z = tangent_x.cross(up).normalize();
    let az = RUNWAY_HEADING_DEG.to_radians();
    let fwd = (tangent_x * az.cos() + tangent_z * az.sin()).normalize();
    let right = fwd.cross(up).normalize();

    // The anchor lat/lon that would centre a massif `D` m down this heading.
    for d in [45_000.0, 55_000.0, 70_000.0] {
        let a = (up * (d / radius_m).cos() + fwd * (d / radius_m).sin()).normalize();
        println!(
            "anchor for {:.0} km down heading: lat {:.3}, lon {:.3}",
            d / 1000.0,
            a.y.asin().to_degrees(),
            a.z.atan2(a.x).to_degrees(),
        );
    }

    let ground_h = surface.sample_height_m(site.as_vec3(), 5.0) as f64;
    // Two viewpoints: standing on the runway, and on a 1.5 km approach.
    for (tag, eye_h) in [("ground", ground_h + 2.0), ("approach", ground_h + 1500.0)] {
        render(&surface, radius_m, up, fwd, right, eye_h, tag);
    }
}

fn render(
    surface: &ProceduralSurface,
    radius_m: f64,
    up: DVec3,
    fwd: DVec3,
    right: DVec3,
    eye_h: f64,
    tag: &str,
) {
    let panorama = std::env::var("PANORAMA").is_ok();
    let w = if panorama { 1440usize } else { 1200 };
    let h = if panorama { 200usize } else { 360 };
    let h_fov = if panorama {
        360.0_f64.to_radians()
    } else {
        70.0_f64.to_radians()
    };
    let v_top = 5.0_f64.to_radians();
    let v_bot = -3.0_f64.to_radians();

    let eye = up * (radius_m + eye_h);
    let mut img = image::RgbImage::new(w as u32, h as u32);

    let mut peak_angle = f64::NEG_INFINITY;
    let mut peak_yaw = 0.0;
    for col in 0..w {
        let yaw = (col as f64 / (w - 1) as f64 - 0.5) * h_fov;
        let dir = (fwd * yaw.cos() + right * yaw.sin()).normalize();

        // March outward; track the max elevation angle (the silhouette) and the
        // height/distance at that hit.
        let mut max_ang = f64::NEG_INFINITY;
        let mut hit_h = 0.0;
        let mut hit_d = 0.0;
        let mut d = 200.0;
        while d < 200_000.0 {
            // Surface point at arc distance d along `dir` from the eye base.
            let ang = d / radius_m;
            let p_unit = (up * ang.cos() + dir * ang.sin()).normalize();
            let lod = (d * 0.02).clamp(5.0, 800.0) as f32;
            let terr_h = surface.sample_height_m(p_unit.as_vec3(), lod) as f64;
            let p = p_unit * (radius_m + terr_h);
            let to = p - eye;
            // Elevation angle relative to local horizontal (perp to `up`).
            let radial = to.dot(up);
            let horiz = (to - up * radial).length();
            let elev = radial.atan2(horiz);
            if elev > max_ang {
                max_ang = elev;
                hit_h = terr_h;
                hit_d = d;
            }
            d += (d * 0.01).clamp(40.0, 400.0);
        }
        if max_ang > peak_angle {
            peak_angle = max_ang;
            peak_yaw = yaw;
        }

        for row in 0..h {
            let elev = v_top + (v_bot - v_top) * (row as f64 / (h - 1) as f64);
            let color = if elev <= max_ang {
                terrain_shade(hit_h, hit_d)
            } else {
                sky(elev)
            };
            img.put_pixel(col as u32, row as u32, image::Rgb(color));
        }
    }

    // Gridlines every 30° of bearing (absolute, from north).
    for col in 0..w {
        let yaw = (col as f64 / (w - 1) as f64 - 0.5) * h_fov;
        let bearing = (RUNWAY_HEADING_DEG + yaw.to_degrees()).rem_euclid(360.0);
        if (bearing / 30.0).fract() < (h_fov.to_degrees() / w as f64) / 30.0 {
            for row in (0..h).step_by(2) {
                img.put_pixel(col as u32, row as u32, image::Rgb([255, 80, 80]));
            }
        }
    }

    let out = OUT.replace("skyline", &format!("skyline_{tag}"));
    img.save(&out).expect("save");
    let peak_bearing = (RUNWAY_HEADING_DEG + peak_yaw.to_degrees()).rem_euclid(360.0);
    println!(
        "[{tag}] eye {eye_h:.0} m  peak silhouette {:.2}° at yaw {:+.0}° (bearing {:.0}°)  -> {out}",
        peak_angle.to_degrees(),
        peak_yaw.to_degrees(),
        peak_bearing,
    );
}

fn terrain_shade(h: f64, d: f64) -> [u8; 3] {
    // Height tint, hazed toward the sky with distance (aerial perspective).
    let base = if h < 0.0 {
        [40, 70, 100]
    } else if h < 400.0 {
        [70, 110, 55]
    } else if h < 1200.0 {
        [95, 115, 65]
    } else if h < 2200.0 {
        [115, 110, 95]
    } else if h < 3000.0 {
        [150, 148, 145]
    } else {
        [235, 240, 248]
    };
    let haze = (d / 200_000.0).clamp(0.0, 0.8);
    let sky = [150.0, 180.0, 210.0];
    [
        (base[0] as f64 * (1.0 - haze) + sky[0] * haze) as u8,
        (base[1] as f64 * (1.0 - haze) + sky[1] * haze) as u8,
        (base[2] as f64 * (1.0 - haze) + sky[2] * haze) as u8,
    ]
}

fn sky(elev: f64) -> [u8; 3] {
    let t = (elev.to_degrees() / 5.0).clamp(0.0, 1.0);
    [
        (150.0 + 20.0 * (1.0 - t)) as u8,
        (180.0 + 15.0 * (1.0 - t)) as u8,
        (215.0 + 10.0 * (1.0 - t)) as u8,
    ]
}
