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
//! Stage 4 measures the OTHER orbital consumer: the full-disc impostor
//! (`solid_planet.wgsl`). It never received the derived LUT, so it combines the
//! same strata with `cloud_surface_column_density` — a MAX-of-strata reducer —
//! and multiplies a thinness response instead. Stages 3 and 4 draw the same
//! planet at neighbouring camera distances, so any gap between their
//! distributions is a visible discontinuity, and a Stage-4 mean far above the
//! authored coverage is a planet that renders overcast whatever the climate
//! says.
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

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Mirror of `thalos::atmosphere`'s `cloud_surface_column_density`: the
/// reducer the IMPOSTOR uses to collapse four strata into one occupancy.
/// Max-dominated, so a column with any one dense stratum reads dense —
/// deliberately unlike the mean the far tier's LUT is conditioned on.
fn column_density_max(strata: [f32; 4]) -> f32 {
    let best = strata.iter().copied().fold(0.0f32, f32::max);
    let sum: f32 = strata.iter().sum();
    (best + 0.22 * (sum - best) * (1.0 - best)).clamp(0.0, 1.0)
}

/// Mirror of `weather_column_from_texel`'s `optical_depth` term.
fn optical_depth(weather: [u8; 4]) -> f32 {
    let cov = f32::from(weather[0]) / 255.0;
    let ty = f32::from(weather[1]) / 255.0;
    let local_base = (f32::from(weather[2]) / 255.0).clamp(0.0, 0.92);
    let local_top = (f32::from(weather[3]) / 255.0)
        .clamp(0.02, 1.0)
        .max(local_base + 0.02);
    let thickness = local_top - local_base;
    let stratus_w = 1.0 - smoothstep(0.18, 0.38, ty);
    let storm_w = smoothstep(0.72, 0.88, ty);
    let cumulus_w = (1.0 - stratus_w - storm_w).max(0.0);
    let type_density = 0.55 * stratus_w + 0.95 * cumulus_w + 1.35 * storm_w;
    (0.30 + 0.90 * smoothstep(0.06, 0.50, cov)) * type_density * (0.40 + 2.6 * thickness)
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
    // Stride 37 is coprime with the face size, so samples do not land on a
    // single column of every face.
    let (field, traces) = CloudWeatherField::from_climate_traced(&climate, 37);
    let face = field.face_size;

    let mut coverage = Vec::with_capacity((W * H) as usize);
    let mut cloud_type = Vec::with_capacity((W * H) as usize);
    let mut strata_mean = Vec::with_capacity((W * H) as usize);
    let mut impostor_opacity = Vec::with_capacity((W * H) as usize);
    let mut strata_all: Vec<[f32; 4]> = Vec::with_capacity((W * H) as usize);
    let mut base_frac = Vec::with_capacity((W * H) as usize);
    let mut top_frac = Vec::with_capacity((W * H) as usize);
    for py in 0..H {
        for px in 0..W {
            let dir = equirect_dir(px, py);
            let w = sample_cube(&field.texels, face, dir);
            let s = sample_cube(&field.surface_density_texels, face, dir);
            coverage.push(f32::from(w[0]) / 255.0);
            cloud_type.push(f32::from(w[1]) / 255.0);
            let strata = [
                f32::from(s[0]) / 255.0,
                f32::from(s[1]) / 255.0,
                f32::from(s[2]) / 255.0,
                f32::from(s[3]) / 255.0,
            ];
            strata_mean.push(strata.iter().sum::<f32>() / 4.0);
            strata_all.push(strata);
            base_frac.push(f32::from(w[2]) / 255.0);
            top_frac.push(f32::from(w[3]) / 255.0);
            // `solid_planet.wgsl`'s `surface_opacity`, verbatim.
            let od = optical_depth(w);
            impostor_opacity.push(
                (column_density_max(strata) * (0.70 + 0.30 * (1.0 - (-od * 1.4).exp())))
                    .clamp(0.0, 0.95),
            );
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
    histogram(
        "STAGE 3  rendered far opacity (composite far tier, LUT)",
        &opacity,
    );
    write_gray("far_opacity.png", &opacity);

    histogram(
        "STAGE 4  rendered impostor opacity (solid_planet.wgsl)",
        &impostor_opacity,
    );
    write_gray("impostor_opacity.png", &impostor_opacity);

    // ── The transfer that matters ────────────────────────────────────────
    // Coverage is the authored areal fraction; strata is what the far tiers
    // actually render. If clear sky (coverage ~0) does not come out of the
    // derivation as strata ~0, the planet has a cloud FLOOR: no climate trim
    // and no downstream response curve can put clear sky back, because the
    // information is already gone by then. This is the one table to read
    // first — a monotone ramp starting near zero is healthy; a nonzero value
    // in the 0.0-0.1 row is the defect.
    println!("\nTRANSFER  coverage decile -> mean rendered value (clear must stay clear)");
    println!("  coverage        n     strata   far(LUT)  impostor");
    for d in 0..10 {
        let lo = d as f32 / 10.0;
        let hi = (d + 1) as f32 / 10.0;
        let idx: Vec<usize> = (0..coverage.len())
            .filter(|&i| coverage[i] >= lo && coverage[i] < hi)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let avg = |v: &[f32]| idx.iter().map(|&i| v[i]).sum::<f32>() / idx.len() as f32;
        println!(
            "  {lo:.1}-{hi:.1}  {:>9}   {:>6.3}    {:>6.3}    {:>6.3}",
            idx.len(),
            avg(&strata_mean),
            avg(&opacity),
            avg(&impostor_opacity),
        );
    }

    // Localize the pedestal. Which of the four strata carries it, and is the
    // carrier the low deck or the high cirrus veil? The veil lives at
    // base ~0.74 of the shell and reads type ~0.04, so a clear-coverage bin
    // whose density sits in the TOP strata at high base and low type is a
    // veil floor; one whose density sits in the bottom strata is a low-deck
    // formation-threshold floor. The two have entirely different fixes.
    println!("\nPEDESTAL  per-decile strata breakdown (q=0.125/0.375/0.625/0.875 of local shell)");
    println!("  coverage    s0     s1     s2     s3    type   base");
    for d in 0..10 {
        let lo = d as f32 / 10.0;
        let hi = (d + 1) as f32 / 10.0;
        let idx: Vec<usize> = (0..coverage.len())
            .filter(|&i| coverage[i] >= lo && coverage[i] < hi)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let n = idx.len() as f32;
        let s = |k: usize| idx.iter().map(|&i| strata_all[i][k]).sum::<f32>() / n;
        let avg = |v: &[f32]| idx.iter().map(|&i| v[i]).sum::<f32>() / n;
        println!(
            "  {lo:.1}-{hi:.1}  {:>5.3}  {:>5.3}  {:>5.3}  {:>5.3}  {:>5.3}  {:>5.3}",
            s(0),
            s(1),
            s(2),
            s(3),
            avg(&cloud_type),
            avg(&base_frac),
        );
    }

    // ── Attribution ──────────────────────────────────────────────────────
    // Which term of the derivation lifts clear sky off zero? `mass` is
    // `shape - threshold - vertical_narrow`; the areal fraction is a narrow
    // smoothstep of it (±0.035), so per texel it is essentially binary.
    // `frac>0.5` therefore separates two very different worlds: if it tracks
    // the mean density, the pedestal is a MINORITY of texels rendering fully
    // cloudy inside nominally clear regions (a threshold/shape miss); if it
    // is near zero while density is not, the pedestal is every texel sitting
    // inside the smoothstep's transition band (a realization-width miss).
    println!("\nATTRIBUTION  derivation terms by coverage decile (lowest stratum)");
    println!("  coverage        n    shape  thresh  v_narrow    mass  areal  frac>0.5  vprof");
    for d in 0..10 {
        let lo = d as f32 / 10.0;
        let hi = (d + 1) as f32 / 10.0;
        let sel: Vec<&thalos_runtime::solar_system_state::WeatherTraceSample> = traces
            .iter()
            .filter(|s| s.coverage >= lo && s.coverage < hi)
            .collect();
        if sel.is_empty() {
            continue;
        }
        let n = sel.len() as f32;
        let m = |f: fn(&thalos_runtime::solar_system_state::WeatherTraceSample) -> f32| {
            sel.iter().map(|s| f(s)).sum::<f32>() / n
        };
        let hot = sel.iter().filter(|s| s.trace.areal_fraction > 0.5).count() as f32 / n;
        println!(
            "  {lo:.1}-{hi:.1}  {:>7}  {:>6.3}  {:>6.3}  {:>8.3}  {:>6.3}  {:>5.3}  {:>8.3}  {:>5.3}",
            sel.len(),
            m(|s| s.trace.shape),
            m(|s| s.trace.threshold),
            m(|s| s.trace.vertical_narrow),
            m(|s| s.trace.mass),
            m(|s| s.trace.areal_fraction),
            hot,
            m(|s| s.trace.vertical_profile),
        );
    }

    // ── Threshold calibration ────────────────────────────────────────────
    // The formation threshold decides what fraction of a texel forms cloud:
    // `areal ~ P(shape - vertical_narrow > threshold)`. For the emitted
    // occupancy to TRACK the authored coverage channel, the threshold at
    // coverage c must be the (1-c) quantile of that comparand's distribution.
    // Deriving it here — rather than hand-fitting constants, which has gone
    // wrong twice before — makes the line reproducible from the field itself.
    let mut fit: Vec<(f32, f32)> = Vec::new();
    println!("\nTHRESHOLD FIT  empirical (coverage, threshold) for occupancy = coverage");
    println!("  coverage        n   comparand      needed  shipped");
    for d in 0..10 {
        let lo = d as f32 / 10.0;
        let hi = (d + 1) as f32 / 10.0;
        let mut vals: Vec<f32> = traces
            .iter()
            .filter(|s| s.coverage >= lo && s.coverage < hi && s.trace.cov > 1.0e-3)
            .map(|s| s.trace.shape - s.trace.vertical_narrow)
            .collect();
        if vals.len() < 64 {
            continue;
        }
        let c: f32 = traces
            .iter()
            .filter(|s| s.coverage >= lo && s.coverage < hi && s.trace.cov > 1.0e-3)
            .map(|s| s.trace.cov)
            .sum::<f32>()
            / vals.len() as f32;
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        // The (1-c) quantile: the value only the cloudiest `c` share exceeds.
        let k = ((c * vals.len() as f32) as usize).clamp(1, vals.len() - 1);
        let needed = vals[k];
        let comparand = vals.iter().sum::<f32>() / vals.len() as f32;
        fit.push((c, needed));
        println!(
            "  {lo:.1}-{hi:.1}  {:>7}      {comparand:>6.3}      {needed:>6.3}   {:>6.3}",
            vals.len(),
            1.03 - 0.70 * c,
        );
    }
    // Least squares through the derived points: threshold = a + b*coverage.
    let n = fit.len() as f32;
    let sx: f32 = fit.iter().map(|p| p.0).sum();
    let sy: f32 = fit.iter().map(|p| p.1).sum();
    let sxx: f32 = fit.iter().map(|p| p.0 * p.0).sum();
    let sxy: f32 = fit.iter().map(|p| p.0 * p.1).sum();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let a = (sy - b * sx) / n;
    println!(
        "\n  DERIVED LINE  threshold = {a:.3} + ({:.3}) * coverage    [shipped: 1.030 + (-0.700) * coverage]",
        b
    );

    // ── Column height ────────────────────────────────────────────────────
    // "The clouds look flat" has two possible causes with OPPOSITE fixes, and
    // a screenshot cannot separate them: either the vertical ENVELOPE the
    // producer authors (`base`..`top`) is too shallow, or the dome term is
    // carving the tops off columns that do have room. This measures both.
    //
    // `envelope` is what the producer allows; `rendered` is how much of it
    // survives the formation threshold + dome. `reach_top` is the share of
    // cloudy columns whose highest stratum still has density — the direct
    // tower-vs-squat contrast number that round 7's dome exists to control.
    const SHELL_M: f32 = 10_500.0;
    const BASE_M: f32 = 900.0;
    let mut env_m: Vec<f32> = Vec::new();
    let mut top_m: Vec<f32> = Vec::new();
    let mut reach_top = 0usize;
    for i in 0..coverage.len() {
        let st = strata_all[i];
        if st.iter().copied().fold(0.0f32, f32::max) < 0.05 {
            continue;
        }
        let b = base_frac[i];
        let t = top_frac[i].max(b + 0.02);
        env_m.push((t - b) * SHELL_M);
        // Highest stratum carrying density, mapped back to absolute altitude.
        let q = [0.125f32, 0.375, 0.625, 0.875];
        let mut hi = 0.0f32;
        for k in 0..4 {
            if st[k] >= 0.05 {
                hi = q[k];
            }
        }
        if st[3] >= 0.05 {
            reach_top += 1;
        }
        top_m.push(BASE_M + (b + hi * (t - b)) * SHELL_M);
    }
    let pct = |v: &mut Vec<f32>, q: f32| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((q * v.len() as f32) as usize).min(v.len() - 1)]
    };
    let n_cloudy = env_m.len();
    let mut e = env_m.clone();
    let mut t = top_m.clone();
    println!(
        "
HEIGHT  over {n_cloudy} cloudy columns"
    );
    println!(
        "  authored envelope (top-base), m:  p50 {:.0}  p90 {:.0}  p99 {:.0}",
        pct(&mut e, 0.50),
        pct(&mut e, 0.90),
        pct(&mut e, 0.99)
    );
    println!(
        "  RENDERED cloud top, m amsl:       p50 {:.0}  p90 {:.0}  p99 {:.0}",
        pct(&mut t, 0.50),
        pct(&mut t, 0.90),
        pct(&mut t, 0.99)
    );
    println!(
        "  columns reaching their own top:   {:.1}%  (tower-vs-squat contrast;          ~0% = dome cuts every top, ~100% = flat slab deck, no contrast)",
        reach_top as f32 / n_cloudy as f32 * 100.0
    );

    let fm = opacity.iter().sum::<f32>() / opacity.len() as f32;
    let im = impostor_opacity.iter().sum::<f32>() / impostor_opacity.len() as f32;
    println!(
        "\nTIER AGREEMENT  far(LUT)={fm:.3}  impostor={im:.3}  ratio={:.2}x  \
         (target ~1.0; the two draw the same planet at neighbouring distances)",
        im / fm.max(1.0e-4)
    );
    println!(
        "AUTHORED        coverage={:.3}  -> impostor renders {:.2}x the authored cloud fraction",
        climate.coverage,
        im / climate.coverage.max(1.0e-4)
    );
}
