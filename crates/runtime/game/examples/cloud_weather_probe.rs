//! Diagnostic for the cloud weather field's *distribution* — the question
//! "is the planet's cloud cover binary, and if so, which stage made it binary?"
//!
//! Renders the produced field as flat equirectangular maps and prints the
//! histograms that separate the three candidate stages:
//!
//! 1. the CPU producer's coverage channel (`CloudWeatherField::from_climate`),
//! 2. the four-stratum surface density it derives from that coverage,
//! 3. the far tier's derived coverage→opacity response LUT
//!    (`derive_fill_calibration`), which is what an orbital pixel actually
//!    shows.
//!
//! Run: `cargo run -p thalos_runtime --example cloud_weather_probe`
//! Output: `artifacts/diagnostics/cloud_weather/*.png` + stdout histograms.

use bevy::math::Vec3;
use thalos_body_render::{FillCalibrationInput, derive_fill_calibration};
use thalos_runtime::solar_system_state::CloudWeatherField;
use thalos_world::CloudClimate;

// 2048 wide puts one probe pixel at ~9.8 km on Thalos's equator — the same
// order as a pixel in a planet-disc capture (~6 km), so grain that reads as
// stipple in the render also reads as stipple here. At 1024 the map is too
// coarse to tell smooth cloud from dither.
const W: u32 = 2048;
const H: u32 = 1024;

fn thalos_climate() -> CloudClimate {
    // Mirrors `assets/bodies/thalos.ron`. Parsed from the asset would drag the
    // whole loader in; this probe only needs the numbers.
    let ron = r#"(
        seed: 0x7A105C10D5,
        coverage: 0.46,
        band_strength: 0.18,
        variation: 0.45,
        type_mix: (0.24, 0.56, 0.20),
        albedo: (0.94, 0.96, 1.0),
        scroll_rate: 4.7e-6,
        differential_rotation: 0.40,
        wind_m_s: (15.0, 2.0),
        base_altitude_m: 900.0,
        thickness_m: 10500.0,
        density: 1.0,
        precipitation_threshold: 0.72,
        storm_threshold: 0.86,
        weather_scale_km: 900.0,
        base_shape_scale_m: 8000.0,
        detail_scale_m: 450.0,
    )"#;
    ron::from_str(ron).expect("probe climate mirrors thalos.ron")
}

/// Sample a face-major cubemap by direction (nearest texel — the probe is
/// measuring the produced field, not a filtered projection of it).
fn sample_cube(texels: &[[u8; 4]], face_size: u32, dir: Vec3) -> [u8; 4] {
    let a = dir.abs();
    let (face, sc, tc, ma) = if a.x >= a.y && a.x >= a.z {
        if dir.x > 0.0 {
            (0usize, -dir.z, -dir.y, a.x)
        } else {
            (1, dir.z, -dir.y, a.x)
        }
    } else if a.y >= a.z {
        if dir.y > 0.0 {
            (2usize, dir.x, dir.z, a.y)
        } else {
            (3, dir.x, -dir.z, a.y)
        }
    } else if dir.z > 0.0 {
        (4usize, dir.x, -dir.y, a.z)
    } else {
        (5, -dir.x, -dir.y, a.z)
    };
    let u = (sc / ma + 1.0) * 0.5;
    let v = (tc / ma + 1.0) * 0.5;
    let x = ((u * face_size as f32) as u32).min(face_size - 1);
    let y = ((v * face_size as f32) as u32).min(face_size - 1);
    texels[face * (face_size * face_size) as usize + (y * face_size + x) as usize]
}

fn equirect_dir(px: u32, py: u32) -> Vec3 {
    let lon = ((px as f32 + 0.5) / W as f32) * std::f32::consts::TAU - std::f32::consts::PI;
    let lat = std::f32::consts::FRAC_PI_2 - ((py as f32 + 0.5) / H as f32) * std::f32::consts::PI;
    Vec3::new(lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()).normalize()
}

fn histogram(label: &str, values: &[f32]) {
    let mut bins = [0usize; 10];
    for &v in values {
        bins[((v.clamp(0.0, 0.999) * 10.0) as usize).min(9)] += 1;
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    // Fraction that is neither ~empty nor ~saturated: the number the user's
    // "binary" complaint is about.
    let mid = values.iter().filter(|v| **v > 0.15 && **v < 0.85).count() as f32 / n;
    let clear = values.iter().filter(|v| **v <= 0.02).count() as f32 / n;
    println!(
        "\n{label}  mean={mean:.3}  mid-tone(0.15..0.85)={:.1}%  clear(<=0.02)={:.1}%",
        mid * 100.0,
        clear * 100.0
    );
    for (i, count) in bins.iter().enumerate() {
        let frac = *count as f32 / n;
        println!(
            "  {:.1}-{:.1} {:>6.2}% {}",
            i as f32 / 10.0,
            (i + 1) as f32 / 10.0,
            frac * 100.0,
            "#".repeat((frac * 120.0) as usize)
        );
    }
}

fn write_gray(name: &str, values: &[f32]) {
    let buf: Vec<u8> = values
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let dir = std::path::Path::new("artifacts/diagnostics/cloud_weather");
    std::fs::create_dir_all(dir).expect("probe output dir");
    let path = dir.join(name);
    image::save_buffer(&path, &buf, W, H, image::ExtendedColorType::L8).expect("write probe png");
    println!("wrote {}", path.display());
}

fn main() {
    let climate = thalos_climate();
    let field = CloudWeatherField::from_climate(&climate);
    let face = field.face_size;

    let mut coverage = Vec::with_capacity((W * H) as usize);
    let mut cloud_type = Vec::with_capacity((W * H) as usize);
    let mut strata_mean = Vec::with_capacity((W * H) as usize);
    for py in 0..H {
        for px in 0..W {
            let dir = equirect_dir(px, py);
            let w = sample_cube(&field.texels, face, dir);
            let s = sample_cube(&field.surface_density_texels, face, dir);
            coverage.push(f32::from(w[0]) / 255.0);
            cloud_type.push(f32::from(w[1]) / 255.0);
            strata_mean.push(s.iter().map(|c| f32::from(*c) / 255.0).sum::<f32>() / 4.0);
        }
    }

    histogram("STAGE 1  weather coverage (producer)", &coverage);
    histogram("STAGE 2  strata mean (far-tier input)", &strata_mean);
    // Calibration check. The strata cube is the column OCCUPANCY the far tier
    // renders, so its mean must track the coverage channel: if strata runs
    // systematically denser, the planet renders cloudier than the authored
    // coverage says and no amount of trimming the coverage channel helps.
    let cm = coverage.iter().sum::<f32>() / coverage.len() as f32;
    let sm = strata_mean.iter().sum::<f32>() / strata_mean.len() as f32;
    println!(
        "\nCALIBRATION  strata/coverage = {:.3}  (target ~1.0; >1 renders cloudier than authored)",
        sm / cm
    );
    write_gray("coverage.png", &coverage);
    write_gray("cloud_type.png", &cloud_type);
    write_gray("strata_mean.png", &strata_mean);

    // Stage 3: the derived far-tier response. `strata_mean` is (near enough)
    // the LUT's input variable, so pushing the measured field through it gives
    // the opacity distribution an orbital pixel actually renders.
    let calibration = derive_fill_calibration(&FillCalibrationInput {
        weather_texels: &field.texels,
        strata_texels: &field.surface_density_texels,
        face_size: face,
        coverage_scale: thalos_runtime::solar_system_state::CLOUD_COVERAGE_SCALE,
        density: 0.0026,
        detail_strength: 0.16,
        base_edge_softness: 0.055,
        bottom_softness: 0.16,
        base_shape_scale_m: 8000.0,
        detail_scale_m: 450.0,
        bottom_height_m: 900.0,
        top_height_m: 11400.0,
        planet_radius_m: 3_186_000.0,
        seed: field.seed,
    });
    println!("\nSTAGE 3  far response LUT (input 0..1 -> opacity)");
    for (i, node) in calibration.far_response.iter().enumerate() {
        println!("  {:.3} -> {node:.3}", i as f32 / 15.0);
    }
    let opacity: Vec<f32> = strata_mean
        .iter()
        .map(|m| {
            let t = (m.clamp(0.0, 1.0)) * 15.0;
            let i = (t as usize).min(14);
            let f = t - i as f32;
            calibration.far_response[i] * (1.0 - f) + calibration.far_response[i + 1] * f
        })
        .collect();
    histogram("STAGE 3  rendered far opacity", &opacity);
    write_gray("far_opacity.png", &opacity);
}
