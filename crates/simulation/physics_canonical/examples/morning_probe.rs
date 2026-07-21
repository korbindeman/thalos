//! Throwaway probe: find the boot epoch that lights the fixed runway site
//! (lat 7.6, lon 178) with a climbing morning sun. Reuses the real ephemeris.

use glam::DVec3;
use thalos_physics_canonical::body_trajectory_provider::BodyTrajectoryProvider;
use thalos_physics_canonical::canonical::Epoch;
use thalos_physics_canonical::patched_conics::PatchedConics;
use thalos_world::parsing::load_solar_system_from_dir;

const RUNWAY_SITE_LAT_DEG: f64 = 7.6;
const RUNWAY_SITE_LON_DEG: f64 = 178.0;
const RUNTIME_TIME_SPAN: f64 = 3.156e11;

fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

/// Sun elevation (deg) above the local horizon at the runway site, for the given
/// epoch. Positive = sun above the horizon.
fn sun_elevation_deg(eph: &PatchedConics, thalos: usize, epoch: Epoch) -> f64 {
    let bs = eph.state(thalos, epoch);
    // Pyros at heliocentric origin → sun direction is toward -position.
    let sun_inertial = (-bs.position).normalize();
    let sun_body = (bs.orientation.inverse() * sun_inertial).normalize();
    let site = latlon_dir(RUNWAY_SITE_LAT_DEG, RUNWAY_SITE_LON_DEG);
    // Elevation = 90 - angle(site_up, sun).
    let cos_zenith = site.dot(sun_body).clamp(-1.0, 1.0);
    90.0 - cos_zenith.acos().to_degrees()
}

/// Sub-stellar lat/lon at the epoch (sanity check against the runway comment).
fn substellar_latlon(eph: &PatchedConics, thalos: usize, epoch: Epoch) -> (f64, f64) {
    let bs = eph.state(thalos, epoch);
    let sun_inertial = (-bs.position).normalize();
    let s = (bs.orientation.inverse() * sun_inertial).normalize();
    let lat = s.y.clamp(-1.0, 1.0).asin().to_degrees();
    let lon = s.z.atan2(s.x).to_degrees();
    (lat, lon)
}

fn main() {
    let system = load_solar_system_from_dir(std::path::Path::new("assets"))
        .expect("load solar_system from assets/");
    let thalos = *system.name_to_id.get("Thalos").expect("Thalos present");
    let eph = PatchedConics::new(&system, RUNTIME_TIME_SPAN);
    let day_s = 76680.0_f64;

    let (lat0, lon0) = substellar_latlon(&eph, thalos, Epoch(0.0));
    println!(
        "boot epoch: sub-stellar lat {:.1} lon {:.1}, site elevation {:.1} deg",
        lat0,
        lon0,
        sun_elevation_deg(&eph, thalos, Epoch(0.0))
    );

    // Sweep one full Thalos day in 0.5 h steps; report elevation + whether the
    // sun is climbing (morning) over the next minute.
    println!("\n t/day_frac   epoch_s    elev   trend");
    let mut best: Option<(f64, f64)> = None; // (|elev-12|, epoch)
    let steps = 96;
    for i in 0..steps {
        let frac = i as f64 / steps as f64;
        let t = frac * day_s;
        let e = sun_elevation_deg(&eph, thalos, Epoch(t));
        let e_next = sun_elevation_deg(&eph, thalos, Epoch(t + 60.0));
        let climbing = e_next > e;
        if (4.0..=25.0).contains(&e) && climbing {
            let score = (e - 12.0).abs();
            if best.is_none_or(|(b, _)| score < b) {
                best = Some((score, t));
            }
        }
        if i % 2 == 0 {
            println!(
                "  {:.3}    {:>9.0}   {:>5.1}   {}",
                frac,
                t,
                e,
                if climbing { "rising" } else { "setting" }
            );
        }
    }

    if let Some((_, t)) = best {
        let (lat, lon) = substellar_latlon(&eph, thalos, Epoch(t));
        println!(
            "\nBEST morning epoch: {:.0} s  (elev {:.1} deg, rising; sub-stellar lat {:.1} lon {:.1})",
            t,
            sun_elevation_deg(&eph, thalos, Epoch(t)),
            lat,
            lon
        );
    } else {
        println!("\nno climbing 4..25 deg sample found");
    }
}
