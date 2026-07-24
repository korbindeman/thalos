//! NTR-X2a: export Thalos's canonical macro terrain as diffusion-conditioning
//! rasters.
//!
//! Samples the same `ProceduralSurface` the game renders (body id 2, radius
//! 3,186 km) onto an equirect grid at the terrain-diffusion pipeline's coarse
//! cell scale (23.04 km/px at the equator) and writes:
//!
//! - `thalos_macro_elev.f32` — elevation, metres (conditioning channel 0)
//! - `thalos_macro_temp.f32` — approximate surface temperature, °C (channel 1)
//! - `thalos_macro_precip.f32` — approximate annual precipitation, mm (ch 3)
//!
//! The diffusion pipeline's `set_custom_conditioning_import` consumes these
//! (see `thalos_export.py` in the terrain-diffusion checkout), so the model
//! generates coarse + detail bands **conditioned on Thalos's own continents**
//! — the runway site stays on its real coastline and `just map` stays
//! recognisable. Temperature/precip are crude climate proxies (latitude +
//! lapse rate; macro moisture) — conditioning guidance, not physics.
//!
//! Convention notes: `dir = (cos lat · cos lon, sin lat, cos lat · sin lon)`
//! (matches `runway::latlon_dir`); raster row 0 = north pole edge, column 0 =
//! longitude 0, row-major little-endian f32 — the same layout the Python side
//! and the future `DiffusionSurface` sampler assume.
//!
//! Usage: `cargo run -p thalos_terrain --release --example export_thalos_macro
//! [-- out_dir]` (default out_dir: `target/thalos_macro`).

use std::io::Write as _;
use std::path::PathBuf;

use glam::DVec3;
use thalos_terrain::ProceduralSurface;

const THALOS_RADIUS_M: f32 = 3_186_000.0;
const THALOS_BODY_SEED: u32 = 2;
const COARSE_PX_M: f64 = 90.0 * 256.0; // 23.04 km — one diffusion conditioning cell
const SAMPLE_LOD_M: f32 = 23_040.0;

fn write_f32(path: &PathBuf, data: &[f32]) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in data {
        out.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/thalos_macro"));
    std::fs::create_dir_all(&out_dir)?;

    let width =
        (std::f64::consts::TAU * f64::from(THALOS_RADIUS_M) / COARSE_PX_M).round() as usize; // ~869
    let height = width / 2;
    let surface = ProceduralSurface::new(THALOS_RADIUS_M, THALOS_BODY_SEED);

    let mut elev = vec![0f32; width * height];
    let mut temp = vec![0f32; width * height];
    let mut precip = vec![0f32; width * height];

    for y in 0..height {
        let lat = (90.0 - (y as f64 + 0.5) / height as f64 * 180.0).to_radians();
        for x in 0..width {
            let lon = ((x as f64 + 0.5) / width as f64 * 360.0).to_radians();
            let dir = DVec3::new(
                lat.cos() * lon.cos(),
                lat.sin(),
                lat.cos() * lon.sin(),
            );
            let (sample, _) = surface.sample_biome_d(dir, SAMPLE_LOD_M);
            let h = sample.height_m;
            let sin_lat = lat.sin().abs();
            let idx = y * width + x;
            elev[idx] = h;
            // Latitude curve + dry-adiabatic-ish lapse on land. Ocean rows use
            // the sea-surface value (h clamped to 0).
            temp[idx] =
                (27.0 - 52.0 * (sin_lat * sin_lat) as f32 - 6.5 * (h.max(0.0) / 1000.0)) as f32;
            // Macro moisture in [-1, 1] → plausible annual millimetres.
            let m = f64::from(sample.moisture).clamp(-1.0, 1.0);
            precip[idx] = (120.0 + 1_500.0 * ((m + 1.0) * 0.5).powf(1.4)) as f32;
        }
        if y % 64 == 0 {
            println!("row {y}/{height}");
        }
    }

    write_f32(&out_dir.join("thalos_macro_elev.f32"), &elev)?;
    write_f32(&out_dir.join("thalos_macro_temp.f32"), &temp)?;
    write_f32(&out_dir.join("thalos_macro_precip.f32"), &precip)?;

    let land_frac = elev.iter().filter(|h| **h > 0.0).count() as f64 / elev.len() as f64;
    let meta = format!(
        "{{\n  \"width\": {width},\n  \"height\": {height},\n  \"px_m_equator\": {COARSE_PX_M},\n  \"planet_radius_m\": {THALOS_RADIUS_M},\n  \"body_seed\": {THALOS_BODY_SEED},\n  \"sample_lod_m\": {SAMPLE_LOD_M},\n  \"land_frac\": {land_frac:.4},\n  \"mapping\": \"equirect, row 0 = north, col 0 = lon 0, dir = (cos lat cos lon, sin lat, cos lat sin lon)\"\n}}\n"
    );
    std::fs::write(out_dir.join("thalos_macro.json"), &meta)?;
    println!("{meta}");
    println!("wrote {}", out_dir.display());
    Ok(())
}
