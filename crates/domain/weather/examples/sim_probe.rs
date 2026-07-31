//! Fast visual iteration on the weather sim: run a seeded spin-up, dump the
//! diagnostic maps as PNGs, print phase timings and field statistics.
//!
//! Run: `cargo run --release -p thalos_weather --example sim_probe [days] [seed]`
//! Output: `artifacts/diagnostics/weather_sim/*.png`
//!
//! This exists so the edit → image loop lives in a leaf crate that compiles in
//! seconds — iterating on weather through the runtime crate costs a 1–2 min
//! release rebuild per constant change.

use std::time::Instant;

use thalos_weather::{WeatherSim, WeatherSimParams};

fn save_gray(path: &str, w: usize, h: usize, values: &[f32], lo: f32, hi: f32) {
    let px: Vec<u8> = values
        .iter()
        .map(|v| (((v - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    // Row 0 of the sim is the SOUTH edge; images read top = north.
    let mut flipped = vec![0u8; px.len()];
    for j in 0..h {
        flipped[j * w..(j + 1) * w].copy_from_slice(&px[(h - 1 - j) * w..(h - j) * w]);
    }
    image::save_buffer(path, &flipped, w as u32, h as u32, image::ColorType::L8)
        .expect("write probe png");
}

fn stats(name: &str, values: &[f32]) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("  {name:<10} mean {mean:+.4}  std {:.4}  min {min:+.3}  max {max:+.3}", var.sqrt());
}

fn main() {
    let mut args = std::env::args().skip(1);
    let days: f32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(12.0);
    let seed: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0x7A10);

    let params = WeatherSimParams {
        seed,
        ..WeatherSimParams::default()
    };
    let (nx, ny) = (params.nx, params.ny);
    println!(
        "weather sim probe: {nx}x{ny}, dt {}s, {days} days, seed {seed:#x}",
        params.dt_s
    );

    let t0 = Instant::now();
    let mut sim = WeatherSim::new(params);
    println!("init: {:?}", t0.elapsed());

    let t1 = Instant::now();
    sim.run_days(days);
    let steps = (days * 86_400.0 / sim.params.dt_s).ceil();
    println!(
        "run:  {:?} for {steps} steps ({:.1} ms/step)",
        t1.elapsed(),
        t1.elapsed().as_secs_f64() * 1000.0 / f64::from(steps)
    );

    let out_dir = "artifacts/diagnostics/weather_sim";
    std::fs::create_dir_all(out_dir).expect("create weather_sim dir");

    let cloud = sim.cloud();
    let vort = sim.vorticity();
    let div = sim.divergence();
    let t2 = Instant::now();
    save_gray(&format!("{out_dir}/cloud.png"), nx, ny, &cloud, 0.0, 1.0);
    save_gray(&format!("{out_dir}/moisture.png"), nx, ny, sim.q_field(), 0.0, 1.1);
    save_gray(&format!("{out_dir}/vorticity.png"), nx, ny, &vort, -8.0e-5, 8.0e-5);
    save_gray(&format!("{out_dir}/divergence.png"), nx, ny, &div, -4.0e-5, 4.0e-5);
    save_gray(&format!("{out_dir}/u_wind.png"), nx, ny, sim.u_field(), -30.0, 30.0);
    save_gray(&format!("{out_dir}/h_anom.png"), nx, ny, sim.h_field(), -400.0, 400.0);
    println!("png:  {:?} -> {out_dir}", t2.elapsed());

    stats("cloud", &cloud);
    stats("q", sim.q_field());
    stats("u", sim.u_field());
    stats("v", sim.v_field());
    stats("h", sim.h_field());
    stats("vort", &vort);
}
