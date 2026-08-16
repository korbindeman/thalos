use std::{env, ffi::OsString, path::PathBuf};

use anyhow::{Context, Result, bail};
use bevy::{math::DVec3, prelude::Resource};

const DEFAULT_CAPTURE_WIDTH: u32 = 1600;
const DEFAULT_CAPTURE_HEIGHT: u32 = 900;
const DEFAULT_VIEWPOINTS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/viewpoints.json");

#[derive(Clone, Debug, Resource)]
pub struct RunConfig {
    pub viewpoints_path: PathBuf,
    pub initial_viewpoint: ViewpointSelection,
    pub capture: Option<HeadlessCapture>,
    pub spatial: SpatialMode,
    /// Authored local civil time for deterministic startup/capture, in hours.
    pub local_time_hours: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpatialMode {
    #[default]
    Planar,
    Ellipsoid,
}

impl RunConfig {
    pub fn is_headless(&self) -> bool {
        self.capture.is_some()
    }
}

#[derive(Clone, Debug)]
pub struct HeadlessCapture {
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub enum ViewpointSelection {
    Default,
    Preset(u8),
    Named(String),
    Arbitrary { position_m: DVec3, target_m: DVec3 },
}

pub enum CliAction {
    Run(RunConfig),
    Help,
}

pub fn parse() -> Result<CliAction> {
    let args = env::args_os().skip(1).collect();
    parse_args(
        args,
        env::var_os("KORSOU_CAPTURE"),
        env::var_os("KORSOU_CAMERA_PRESET"),
        env::var_os("KORSOU_VIEWPOINTS"),
    )
}

fn parse_args(
    args: Vec<OsString>,
    env_capture: Option<OsString>,
    env_preset: Option<OsString>,
    env_viewpoints: Option<OsString>,
) -> Result<CliAction> {
    let args = args
        .into_iter()
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow::anyhow!("command-line arguments must be valid UTF-8"))
        })
        .collect::<Result<Vec<_>>>()?;

    if args
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        return Ok(CliAction::Help);
    }

    if args.first().is_some_and(|argument| argument == "capture") {
        return parse_capture(&args[1..], env_viewpoints).map(CliAction::Run);
    }

    let (viewpoints_path, spatial, local_time_hours) =
        parse_interactive_options(&args, env_viewpoints)?;
    let initial_viewpoint = env_preset
        .map(|value| {
            let preset = value
                .to_string_lossy()
                .parse::<u8>()
                .context("KORSOU_CAMERA_PRESET must be 1, 2, or 3")?;
            if !(1..=3).contains(&preset) {
                bail!("KORSOU_CAMERA_PRESET must be 1, 2, or 3");
            }
            Ok(preset)
        })
        .transpose()?
        .map_or(ViewpointSelection::Default, ViewpointSelection::Preset);

    let capture = env_capture.map(|output| HeadlessCapture {
        output: PathBuf::from(output),
        width: DEFAULT_CAPTURE_WIDTH,
        height: DEFAULT_CAPTURE_HEIGHT,
    });

    Ok(CliAction::Run(RunConfig {
        viewpoints_path,
        initial_viewpoint,
        capture,
        spatial,
        local_time_hours,
    }))
}

fn parse_interactive_options(
    args: &[String],
    env_viewpoints: Option<OsString>,
) -> Result<(PathBuf, SpatialMode, Option<f64>)> {
    let mut viewpoints_path = env_viewpoints
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_VIEWPOINTS_PATH));
    let mut index = 0;
    let mut spatial = SpatialMode::Planar;
    let mut local_time_hours = None;
    while index < args.len() {
        match args[index].as_str() {
            "--viewpoints" => {
                viewpoints_path = PathBuf::from(option_value(args, &mut index, "--viewpoints")?);
            }
            "--spatial" => {
                spatial = parse_spatial(option_value(args, &mut index, "--spatial")?)?;
            }
            "--time" => {
                local_time_hours =
                    Some(parse_local_time(option_value(args, &mut index, "--time")?)?);
            }
            argument => bail!("unknown argument `{argument}`\n\n{}", help_text()),
        }
        index += 1;
    }
    Ok((viewpoints_path, spatial, local_time_hours))
}

fn parse_capture(args: &[String], env_viewpoints: Option<OsString>) -> Result<RunConfig> {
    let Some(output) = args.first().filter(|value| !value.starts_with('-')) else {
        bail!("capture requires an output PNG path\n\n{}", help_text());
    };

    let mut viewpoints_path = env_viewpoints
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_VIEWPOINTS_PATH));
    let mut named = None;
    let mut position = None;
    let mut target = None;
    let mut size = (DEFAULT_CAPTURE_WIDTH, DEFAULT_CAPTURE_HEIGHT);
    let mut spatial = SpatialMode::Planar;
    let mut local_time_hours = None;
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--viewpoint" => named = Some(option_value(args, &mut index, "--viewpoint")?.into()),
            "--position" => {
                position = Some(parse_vec3(option_value(args, &mut index, "--position")?)?)
            }
            "--look-at" => target = Some(parse_vec3(option_value(args, &mut index, "--look-at")?)?),
            "--size" => size = parse_size(option_value(args, &mut index, "--size")?)?,
            "--viewpoints" => {
                viewpoints_path = PathBuf::from(option_value(args, &mut index, "--viewpoints")?);
            }
            "--spatial" => {
                spatial = parse_spatial(option_value(args, &mut index, "--spatial")?)?;
            }
            "--time" => {
                local_time_hours =
                    Some(parse_local_time(option_value(args, &mut index, "--time")?)?);
            }
            argument => bail!("unknown capture argument `{argument}`\n\n{}", help_text()),
        }
        index += 1;
    }

    if named.is_some() && (position.is_some() || target.is_some()) {
        bail!("use either --viewpoint or --position with --look-at, not both");
    }
    let initial_viewpoint = match (named, position, target) {
        (Some(name), None, None) => ViewpointSelection::Named(name),
        (None, Some(position_m), Some(target_m)) => {
            if position_m == target_m {
                bail!("--position and --look-at must not be the same point");
            }
            ViewpointSelection::Arbitrary {
                position_m,
                target_m,
            }
        }
        (None, None, None) => ViewpointSelection::Default,
        _ => bail!("arbitrary captures require both --position and --look-at"),
    };

    Ok(RunConfig {
        viewpoints_path,
        initial_viewpoint,
        capture: Some(HeadlessCapture {
            output: PathBuf::from(output),
            width: size.0,
            height: size.1,
        }),
        spatial,
        local_time_hours,
    })
}

fn parse_spatial(value: &str) -> Result<SpatialMode> {
    match value {
        "planar" => Ok(SpatialMode::Planar),
        "ellipsoid" => Ok(SpatialMode::Ellipsoid),
        _ => bail!("--spatial must be `planar` or `ellipsoid`"),
    }
}

fn parse_local_time(value: &str) -> Result<f64> {
    let Some((hours, minutes)) = value.split_once(':') else {
        bail!("--time must use local HH:MM, for example 17:30");
    };
    let hours = hours
        .parse::<u8>()
        .with_context(|| format!("--time `{value}` must use local HH:MM"))?;
    let minutes = minutes
        .parse::<u8>()
        .with_context(|| format!("--time `{value}` must use local HH:MM"))?;
    if hours >= 24 || minutes >= 60 {
        bail!("--time `{value}` must use local HH:MM within 00:00–23:59");
    }
    Ok(f64::from(hours) + f64::from(minutes) / 60.0)
}

fn option_value<'a>(args: &'a [String], index: &mut usize, option: &str) -> Result<&'a str> {
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .with_context(|| format!("{option} requires a value"))
}

fn parse_vec3(value: &str) -> Result<DVec3> {
    let values = value
        .split(',')
        .map(|part| part.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("`{value}` must contain three comma-separated numbers"))?;
    if values.len() != 3 || values.iter().any(|value| !value.is_finite()) {
        bail!("`{value}` must contain three finite comma-separated numbers");
    }
    Ok(DVec3::new(values[0], values[1], values[2]))
}

fn parse_size(value: &str) -> Result<(u32, u32)> {
    let Some((width, height)) = value.split_once('x') else {
        bail!("--size must use WIDTHxHEIGHT, for example 1920x1080");
    };
    let width = width
        .parse::<u32>()
        .context("capture width must be an integer")?;
    let height = height
        .parse::<u32>()
        .context("capture height must be an integer")?;
    if width == 0 || height == 0 {
        bail!("capture dimensions must be greater than zero");
    }
    Ok((width, height))
}

pub fn help_text() -> &'static str {
    "Kòrsou — lightweight real-world island explorer\n\n\
Usage:\n\
  korsou [--viewpoints FILE] [--spatial planar|ellipsoid] [--time HH:MM]\n\
  korsou capture OUTPUT.png [--viewpoint NAME] [--size WIDTHxHEIGHT] [--spatial planar|ellipsoid] [--time HH:MM]\n\
  korsou capture OUTPUT.png --position X,Y,Z --look-at X,Y,Z\n\n\
Capture runs offscreen without creating a window. --time is Curaçao local civil time (UTC-4). Coordinates are local metres: \
+X east, +Y up, -Z north. The default viewpoint file is viewpoints.json."
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn parses_named_headless_capture() {
        let CliAction::Run(config) = parse_args(
            strings(&[
                "capture",
                "artifacts/test.png",
                "--viewpoint",
                "Viewpoint 003",
                "--size",
                "1920x1080",
                "--time",
                "17:30",
            ]),
            None,
            None,
            None,
        )
        .unwrap() else {
            panic!("expected run configuration");
        };

        let capture = config.capture.unwrap();
        assert_eq!(capture.output, PathBuf::from("artifacts/test.png"));
        assert_eq!((capture.width, capture.height), (1920, 1080));
        assert_eq!(config.local_time_hours, Some(17.5));
        assert!(matches!(
            config.initial_viewpoint,
            ViewpointSelection::Named(ref name) if name == "Viewpoint 003"
        ));
    }

    #[test]
    fn rejects_invalid_local_time() {
        let error = parse_args(
            strings(&["capture", "out.png", "--time", "25:00"]),
            None,
            None,
            None,
        )
        .err()
        .unwrap();

        assert!(error.to_string().contains("HH:MM"));
    }

    #[test]
    fn arbitrary_capture_requires_position_and_target() {
        let error = parse_args(
            strings(&["capture", "out.png", "--position", "1,2,3"]),
            None,
            None,
            None,
        )
        .err()
        .unwrap();

        assert!(error.to_string().contains("require both"));
    }

    #[test]
    fn legacy_preset_must_name_an_existing_preset() {
        let error = parse_args(Vec::new(), None, Some(OsString::from("9")), None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("must be 1, 2, or 3"));
    }
}
