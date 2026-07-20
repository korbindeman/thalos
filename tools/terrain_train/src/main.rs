#![recursion_limit = "256"]

#[cfg(all(feature = "cpu", feature = "gpu"))]
compile_error!("select exactly one of the cpu or gpu features");
#[cfg(not(any(feature = "cpu", feature = "gpu")))]
compile_error!("select one of the cpu or gpu features");

mod config;
mod dem;
mod grid;
mod output;
mod pyramid;
mod synthetic;
mod train;
mod validate;

use std::path::{Path, PathBuf};

use serde::Serialize;

use config::Config;

#[derive(Serialize)]
struct CorpusEntry {
    index: usize,
    seed: u64,
    split: &'static str,
    parameters: synthetic::Parameters,
    height_rms_m: f32,
    slope_rms: f32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("thalos_terrain_train: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "smoke".into());
    if command == "prepare-dem" || command == "prepare-sldem" {
        return prepare_dem(args.collect(), command == "prepare-sldem");
    }
    let flag = args.next().unwrap_or_else(|| "--config".into());
    if flag != "--config" {
        return Err("usage: thalos_terrain_train <prepare|smoke> --config <path>".into());
    }
    let config_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "tools/terrain_train/configs/smoke.toml".into()),
    );
    let config = Config::load(&config_path)?;
    match command.as_str() {
        "prepare" => {
            let samples = prepare(&config)?;
            println!(
                "prepared {} deterministic airless patches in {}",
                samples.len(),
                config.run.data_dir.display()
            );
        }
        "smoke" => smoke(&config)?,
        _ => return Err(format!("unknown command {command:?}; expected prepare or smoke").into()),
    }
    Ok(())
}

fn prepare_dem(
    arguments: Vec<String>,
    sldem_strip: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !arguments.len().is_multiple_of(2) {
        return Err("prepare-dem options must be --name value pairs".into());
    }
    let options: std::collections::HashMap<_, _> = arguments
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| (pair[0].as_str(), pair[1].as_str()))
        .collect();
    let required = |name: &str| -> Result<&str, Box<dyn std::error::Error>> {
        options
            .get(name)
            .copied()
            .ok_or_else(|| format!("missing prepare-dem option {name}").into())
    };
    let prepare = dem::PrepareOptions {
        input: required("--input")?.into(),
        output: required("--output")?.into(),
        source_id: required("--source-id")?.into(),
        split: required("--split")?.into(),
        expected_sha256: required("--sha256")?.into(),
        native_metres_per_pixel: required("--native-mpp")?.parse()?,
        target_metres_per_pixel: required("--target-mpp")?.parse()?,
        patch_size: required("--patch-size")?.parse()?,
        stride: required("--stride")?.parse()?,
    };
    let count = if sldem_strip {
        dem::prepare_sldem_strip(&dem::SldemStripOptions {
            prepare: prepare.clone(),
            source_width: required("--source-width")?.parse()?,
            crop_x: required("--crop-x")?.parse()?,
        })?
    } else {
        dem::prepare(&prepare)?
    };
    println!(
        "prepared {count} verified {} patches in {}",
        prepare.split,
        prepare.output.display()
    );
    Ok(())
}

fn prepare(config: &Config) -> Result<Vec<synthetic::Sample>, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&config.run.data_dir)?;
    let validation_count = ((config.data.sample_count as f32 * config.data.validation_fraction)
        .ceil() as usize)
        .max(1);
    let training_count = config.data.sample_count - validation_count;
    let mut samples = Vec::with_capacity(config.data.sample_count);
    let mut entries = Vec::with_capacity(config.data.sample_count);
    for index in 0..config.data.sample_count {
        let seed = splitmix64(config.run.seed.wrapping_add(index as u64));
        let sample = synthetic::generate(&config.data, seed);
        let split = if index < training_count {
            "train"
        } else {
            "validation"
        };
        entries.push(CorpusEntry {
            index,
            seed,
            split,
            parameters: sample.parameters.clone(),
            height_rms_m: sample.height.rms(),
            slope_rms: sample.height.slope_rms(config.data.metres_per_pixel),
        });
        write_f32le(
            &config.run.data_dir.join(format!("sample_{index:04}.f32le")),
            &sample.height.values,
        )?;
        samples.push(sample);
    }
    std::fs::write(
        config.run.data_dir.join("index.json"),
        serde_json::to_vec_pretty(&entries)?,
    )?;
    Ok(samples)
}

fn smoke(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&config.run.output_dir)?;
    let samples = prepare(config)?;
    let first = &samples[0];
    let pyramid = pyramid::build(&first.height);
    output::save_height_u16(&config.run.output_dir.join("height.png"), &first.height)?;
    for (index, band) in pyramid.full_resolution_bands.iter().enumerate() {
        output::save_height_u16(&config.run.output_dir.join(format!("s{index}.png")), band)?;
    }
    output::save_contact_sheet(
        &config.run.output_dir.join("laplacian_contact_sheet.png"),
        &[
            ("height", &first.height),
            ("s0", &pyramid.full_resolution_bands[0]),
            ("s1", &pyramid.full_resolution_bands[1]),
            ("s2", &pyramid.full_resolution_bands[2]),
            ("s3", &pyramid.full_resolution_bands[3]),
        ],
    )?;

    let report = train::run(config, &samples, &config.run.output_dir)?;
    std::fs::write(
        config.run.output_dir.join("training_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    std::fs::write(
        config.run.output_dir.join("MODEL_CARD.md"),
        model_card(config, &report),
    )?;
    println!(
        "MIRA-1 smoke complete: {} batches, loss {:.5} -> {:.5}, {:.2}s, outputs {}",
        report.batches,
        report.initial_loss,
        report.final_loss,
        report.elapsed_seconds,
        config.run.output_dir.display()
    );
    Ok(())
}

fn model_card(config: &Config, report: &train::TrainingReport) -> String {
    format!(
        "# {}\n\nExperimental MIRA-1 S3 airless residual denoiser. Not accepted for package production.\n\n- Framework: Burn {} ({})\n- Seed: {}\n- Samples: {} synthetic; pinned lunar teachers are prepared but not yet mixed into this run\n- Patch: {} px at {} m/px\n- Diffusion: {} linear-beta steps, {} deterministic DDIM sample steps\n- Training: {} epochs, {} batches; resumed from epoch {}\n- Noise-prediction MSE: {:.6} → {:.6}\n- Runtime: {:.3} s\n- EMA decay: {:.5}\n- EMA tensor SHA-256: `{}`\n- Raw tensor SHA-256: `{}`\n- Overlap windows: {}, disagreement RMS {:.6} m\n- Repeat determinism max delta: {:.9} m\n- Tensor contract: 8 input channels, one S3 noise channel\n- Limitations: real-data mixing, spectral/slope/SFD acceptance, and campaign-scale quality remain open.\n- Intended use: offline patch proof only\n",
        config.run.name,
        report.burn_version,
        report.backend,
        config.run.seed,
        config.data.sample_count,
        config.data.patch_size,
        config.data.metres_per_pixel,
        config.diffusion.timesteps,
        report.validation.sample_steps,
        report.epochs,
        report.batches,
        report.resumed_from_epoch,
        report.initial_loss,
        report.final_loss,
        report.elapsed_seconds,
        report.requested_ema_decay,
        report.model_tensor_sha256,
        report.raw_model_tensor_sha256,
        report.validation.windows,
        report.validation.overlap_prediction_rms_m,
        report.validation.repeated_generation_max_abs_delta_m,
    )
}

fn write_f32le(path: &Path, values: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
