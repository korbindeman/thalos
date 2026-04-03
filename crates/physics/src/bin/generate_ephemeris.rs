//! Precompute and save an N-body ephemeris to disk.
//!
//! Usage:
//!   generate_ephemeris [YEARS] [--validate]
//!
//! YEARS defaults to 10000.  You can start small and extend later — existing
//! progress is kept:
//!
//!   just generate 1000     # generate 1,000 years
//!   just generate 5000     # extends to 5,000 years (keeps first 1,000)
//!   just generate 10000    # extends to 10,000 years
//!
//! Flags:
//!   --validate   Run only energy conservation check (no ephemeris saved).
//!
//! Generation is resumable — kill the process and re-run to pick up where
//! you left off.  Progress is checkpointed to `assets/ephemeris_work/`.

use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use thalos_physics::ephemeris::{
    check_stability, validate_energy_conservation_with_progress,
    EnergyValidationReport, Ephemeris,
};
use thalos_physics::ephemeris_generator::EphemerisGenerator;
use thalos_physics::types::{load_solar_system, SolarSystemDefinition};

/// One Julian year in seconds.
const JULIAN_YEAR: f64 = 3.156e7;

/// Default generation span if no argument given.
const DEFAULT_YEARS: f64 = 10_000.0;

/// Energy drift threshold in parts per million.
const ENERGY_THRESHOLD_PPM: f64 = 1.0;

/// Stability check: sample this many points across the ephemeris.
const STABILITY_CHECK_POINTS: usize = 10_000;

/// Steps per batch before updating the progress bar.
const BATCH_SIZE: u64 = 10_000;

const KDL_PATH: &str = "assets/solar_system.kdl";
const OUTPUT_PATH: &str = "assets/ephemeris.bin";
const WORK_DIR: &str = "assets/ephemeris_work";

fn make_bar(total: u64, msg: &str) -> ProgressBar {
    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::with_template(
            "  {msg} [{bar:40.cyan/dim}] {percent}% ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    bar.set_message(msg.to_string());
    bar
}

fn parse_years(args: &[String]) -> f64 {
    for arg in args.iter().skip(1) {
        if arg.starts_with('-') {
            continue;
        }
        if let Ok(y) = arg.parse::<f64>() {
            return y;
        }
    }
    DEFAULT_YEARS
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let validate_only = args.iter().any(|a| a == "--validate");
    let years = parse_years(&args);
    let time_span = years * JULIAN_YEAR;

    let kdl_source = std::fs::read_to_string(KDL_PATH)
        .expect("Could not read assets/solar_system.kdl — run from the workspace root");
    let system =
        load_solar_system(&kdl_source).expect("Failed to parse solar_system.kdl");

    println!("System:  {}", system.name);
    println!("Bodies:  {}", system.bodies.len());
    println!("Target:  {:.0} years", years);

    if validate_only {
        println!(
            "\n── Energy Conservation Validation ({:.0} years, {:.1} ppm threshold) ──",
            years, ENERGY_THRESHOLD_PPM,
        );
        let report = run_validation(&system, time_span);
        print_energy_report(&system, &report);
        if !report.all_passed {
            std::process::exit(1);
        }
        return;
    }

    run_generation(&system, time_span, years);
}

// ---------------------------------------------------------------------------
// Streaming generation
// ---------------------------------------------------------------------------

fn run_generation(system: &SolarSystemDefinition, time_span: f64, years: f64) {
    let work_dir = Path::new(WORK_DIR);
    let output = Path::new(OUTPUT_PATH);

    println!("\n── Generating {:.0}-year ephemeris ──", years);

    let mut generator = EphemerisGenerator::open(system, time_span, work_dir)
        .expect("Failed to open ephemeris generator");

    let (start_step, total) = generator.progress();
    let bar = make_bar(total, "Integrating");
    bar.set_position(start_step);

    while !generator.is_done() {
        generator.advance(BATCH_SIZE)
            .expect("Integration step failed");
        let (step, _) = generator.progress();
        bar.set_position(step);
    }
    bar.finish_and_clear();

    // Print segment stats.
    println!("  Total segments: {}", generator.total_segment_count());
    for (i, &count) in generator.segment_counts().iter().enumerate() {
        if i == generator.star_id() {
            continue;
        }
        println!("    body {}: {} segments", i, count);
    }

    // Assemble final file.
    println!("\n── Assembling {} ──", output.display());
    let total_bytes = generator.finalize_total_bytes();
    let bar = make_bar(total_bytes, "Assembling");
    generator.finalize(output, |written, _total| {
        bar.set_position(written);
    }).expect("Failed to finalize ephemeris");
    bar.finish_and_clear();

    let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    println!("  Saved {:.1} MB", file_size as f64 / 1_048_576.0);

    // Load the assembled file for validation and stability checks.
    println!("\n── Loading for validation ──");
    let ephemeris = Ephemeris::load(output).expect("Failed to load assembled ephemeris");

    // Energy conservation.
    println!(
        "\n── Energy Conservation ({:.0} years, {:.1} ppm threshold) ──",
        years, ENERGY_THRESHOLD_PPM,
    );
    let energy_report = run_validation(system, time_span);
    print_energy_report(system, &energy_report);

    if !energy_report.all_passed {
        eprintln!("\nWarning: energy conservation check failed.");
        eprintln!("The ephemeris has been saved but may have accuracy issues.");
        eprintln!("Consider tightening the RK4 timestep (DT) in ephemeris.rs.");
    }

    // Stability / noteworthy events.
    println!("\n── Noteworthy Events ──");
    let events = check_stability(system, &ephemeris, STABILITY_CHECK_POINTS);
    if events.is_empty() {
        println!("  No noteworthy events detected over {:.0} years.", years);
    } else {
        println!("  {} event(s) detected:\n", events.len());
        for event in &events {
            let event_years = event.time_s / JULIAN_YEAR;
            println!("  t = {:.1} y  {}", event_years, event.description);
        }
    }

    println!("\nDone.");
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn run_validation(
    system: &SolarSystemDefinition,
    duration: f64,
) -> EnergyValidationReport {
    let bar = make_bar(1000, "Validating");
    let report = validate_energy_conservation_with_progress(
        system,
        duration,
        ENERGY_THRESHOLD_PPM,
        |step, total| bar.set_position(step * 1000 / total),
    );
    bar.finish_and_clear();
    report
}

// ---------------------------------------------------------------------------
// Report printer
// ---------------------------------------------------------------------------

fn print_energy_report(system: &SolarSystemDefinition, report: &EnergyValidationReport) {
    let name_width = system
        .bodies
        .iter()
        .map(|b| b.name.len())
        .max()
        .unwrap_or(8);

    let header_status = "Status";
    println!(
        "  {:width$}  {:>14}  {:>14}  {:>12}  {header_status}",
        "Body", "E_initial", "E_final", "Max drift",
        width = name_width,
    );
    println!("  {}", "─".repeat(name_width + 50));

    for body_report in &report.bodies {
        let name = &system.bodies[body_report.body_id].name;
        let status = if body_report.passed { "  OK" } else { "FAIL" };
        println!(
            "  {:width$}  {:>14.6e}  {:>14.6e}  {:>9.3} ppm  {}",
            name,
            body_report.initial_energy,
            body_report.final_energy,
            body_report.max_drift_ppm,
            status,
            width = name_width,
        );
    }

    println!();
    if report.all_passed {
        println!("  All bodies within {:.1} ppm threshold.", ENERGY_THRESHOLD_PPM);
    } else {
        println!(
            "  Some bodies exceed {:.1} ppm threshold!",
            ENERGY_THRESHOLD_PPM,
        );
        println!("  Consider tightening the RK4 timestep (DT) in ephemeris.rs.");
    }
}
