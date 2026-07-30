//! Portable control-plane types shared by capture clients and the renderer.
//!
//! This crate intentionally contains no ECS, GPU, camera, or application
//! types. It describes requests and results; `thalos_capture_runtime` turns
//! them into work performed by the complete Bevy game renderer.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

pub const CAPTURE_PROTOCOL_SCHEMA: u32 = 5;
pub const VIEWPOINT_CATALOG_SCHEMA: &str = "thalos.viewpoints.v2";
const LEGACY_VIEWPOINT_CATALOG_SCHEMA: &str = "thalos.viewpoints.v1";

pub const FULL_FRAME_GATE_WIDTH_MM: f32 = 36.0;
pub const MIN_FOCAL_LENGTH_MM: f32 = 12.0;
pub const MAX_FOCAL_LENGTH_MM: f32 = 400.0;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CameraLensModel {
    #[default]
    FullFrameHorizontal,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CameraLens {
    pub model: CameraLensModel,
    pub focal_length_mm: f32,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SensorCrop {
    #[default]
    Full,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CameraSensor {
    pub gate_width_mm: f32,
    /// Reduced sensor-window aspect ratio, never an output pixel extent.
    pub aspect: [u32; 2],
    pub crop: SensorCrop,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CameraOptics {
    pub lens: CameraLens,
    pub sensor: CameraSensor,
}

impl Default for CameraOptics {
    fn default() -> Self {
        Self::from_vertical_fov(45.0_f32.to_radians(), [16, 9])
            .expect("default camera optics are valid")
    }
}

impl CameraOptics {
    pub fn new(focal_length_mm: f32, aspect: [u32; 2]) -> Result<Self, String> {
        let aspect = reduced_aspect(aspect)?;
        let optics = Self {
            lens: CameraLens {
                model: CameraLensModel::FullFrameHorizontal,
                focal_length_mm,
            },
            sensor: CameraSensor {
                gate_width_mm: FULL_FRAME_GATE_WIDTH_MM,
                aspect,
                crop: SensorCrop::Full,
            },
        };
        optics.validate()?;
        Ok(optics)
    }

    pub fn from_vertical_fov(vertical_fov_rad: f32, aspect: [u32; 2]) -> Result<Self, String> {
        if !vertical_fov_rad.is_finite()
            || !(1.0_f32.to_radians()..179.0_f32.to_radians()).contains(&vertical_fov_rad)
        {
            return Err(format!("invalid vertical FOV {vertical_fov_rad}"));
        }
        let [width, height] = reduced_aspect(aspect)?;
        let aspect = width as f32 / height as f32;
        let horizontal_fov_rad = 2.0 * ((vertical_fov_rad * 0.5).tan() * aspect).atan();
        let mut focal_length_mm =
            FULL_FRAME_GATE_WIDTH_MM / (2.0 * (horizontal_fov_rad * 0.5).tan());
        // Trig round-trips at an authored endpoint can land a few ULPs outside
        // the lens range (12 mm became 11.999999 mm in practice). Snap only
        // that numerical fringe; genuinely out-of-range framing is rejected.
        if (focal_length_mm - MIN_FOCAL_LENGTH_MM).abs() <= 1.0e-4 {
            focal_length_mm = MIN_FOCAL_LENGTH_MM;
        } else if (focal_length_mm - MAX_FOCAL_LENGTH_MM).abs() <= 1.0e-3 {
            focal_length_mm = MAX_FOCAL_LENGTH_MM;
        }
        Self::new(focal_length_mm, [width, height])
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.lens.model != CameraLensModel::FullFrameHorizontal {
            return Err("unsupported camera lens model".into());
        }
        if !self.lens.focal_length_mm.is_finite()
            || !(MIN_FOCAL_LENGTH_MM..=MAX_FOCAL_LENGTH_MM).contains(&self.lens.focal_length_mm)
        {
            return Err(format!(
                "focal length {} mm is outside {MIN_FOCAL_LENGTH_MM}..={MAX_FOCAL_LENGTH_MM}",
                self.lens.focal_length_mm
            ));
        }
        if !self.sensor.gate_width_mm.is_finite()
            || (self.sensor.gate_width_mm - FULL_FRAME_GATE_WIDTH_MM).abs() > 1.0e-4
        {
            return Err(format!(
                "full-frame-horizontal optics require a {FULL_FRAME_GATE_WIDTH_MM} mm gate"
            ));
        }
        if reduced_aspect(self.sensor.aspect)? != self.sensor.aspect {
            return Err("camera sensor aspect must be reduced".into());
        }
        Ok(())
    }

    pub fn horizontal_fov_rad(&self) -> f32 {
        2.0 * (self.sensor.gate_width_mm / (2.0 * self.lens.focal_length_mm)).atan()
    }

    pub fn vertical_fov_rad(&self) -> f32 {
        let aspect = self.sensor.aspect[0] as f32 / self.sensor.aspect[1] as f32;
        2.0 * ((self.horizontal_fov_rad() * 0.5).tan() / aspect).atan()
    }

    pub fn with_focal_length_mm(mut self, focal_length_mm: f32) -> Result<Self, String> {
        self.lens.focal_length_mm = focal_length_mm;
        self.validate()?;
        Ok(self)
    }
}

pub fn reduced_aspect([width, height]: [u32; 2]) -> Result<[u32; 2], String> {
    if width == 0 || height == 0 {
        return Err("camera sensor aspect dimensions must be non-zero".into());
    }
    let divisor = gcd(width, height);
    Ok([width / divisor, height / divisor])
}

const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Compiled procedural driver capabilities available to catalog entries.
///
/// Keep this list aligned with the runtime's `ScreenshotPreset` enum. The
/// developer-facing registry is `assets/viewpoints.json`; this list validates
/// that its scripted entries name real executors and supports internal boot
/// compatibility scheduling.
pub const CAPTURE_PRESETS: &[&str] = &[
    "latest-perspective",
    "spaceport-aerial",
    "runway-atmosphere",
    "paved-ground",
    "hub",
    "dry-belt",
    "forest-stand",
    "earth-reference",
    "ocean",
    "ocean-slopes",
    "coastline",
    "mira-orbit",
    "mira-surface",
    "mira-eva",
    "mira-disc",
    "mira-approach",
    "mira-rim",
    "cloud-runway",
    "cloud-motion",
    "cloud-cruise",
    "cloud-interior",
    "cloud-limb",
    "cloud-planet",
    "cloud-sunset",
    "cloud-godray",
    "plume",
    "plume-skyline",
    "reentry",
    "interstage",
    "orbit-hull",
    "massif-aerial",
    "massif-ridge",
    "massif-valley",
];

/// Canonical scene builder behind an authored viewpoint.
///
/// A viewpoint restores camera framing, not a partial save game. This value
/// tells interactive/headless consumers which normal scene supplies the world
/// around that camera.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ViewpointSpawn {
    Orbit,
    Polar,
    Eva,
    Landing,
    Final,
    Runway,
    RunwayApproach,
    Launch,
    Cruise,
}

/// One stable, exchangeable camera point in a body's authored surface frame.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Viewpoint {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub saved_unix_ms: u128,
    pub body: String,
    pub spawn: ViewpointSpawn,
    #[serde(default)]
    pub boots_hub: bool,
    /// Canonical simulation time at which the view was saved. Headless replay
    /// uses this epoch by default, but callers may override it per capture.
    ///
    /// This remains camera/environment metadata rather than a save-game
    /// snapshot: craft dynamics still come from the canonical spawn.
    pub sim_time_s: f64,
    pub camera_position_body_m: [f64; 3],
    pub camera_rotation_body_xyzw: [f64; 4],
    pub optics: CameraOptics,
}

impl Viewpoint {
    pub fn validate(&self) -> Result<(), String> {
        if !valid_viewpoint_id(&self.id) {
            return Err(format!(
                "viewpoint id {:?} must contain only lowercase ASCII letters, digits, and single '-' separators",
                self.id
            ));
        }
        if self.name.trim().is_empty() {
            return Err(format!("viewpoint {:?} has no display name", self.id));
        }
        if self.body.trim().is_empty() {
            return Err(format!("viewpoint {:?} has no target body", self.id));
        }
        if !self.sim_time_s.is_finite()
            || !self
                .camera_position_body_m
                .iter()
                .all(|value| value.is_finite())
            || !self
                .camera_rotation_body_xyzw
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(format!(
                "viewpoint {:?} contains a non-finite number",
                self.id
            ));
        }
        let rotation_len_sq = self
            .camera_rotation_body_xyzw
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        if rotation_len_sq < 0.25 {
            return Err(format!(
                "viewpoint {:?} has an invalid camera rotation",
                self.id
            ));
        }
        self.optics
            .validate()
            .map_err(|error| format!("viewpoint {:?}: {error}", self.id))?;
        Ok(())
    }
}

#[derive(Deserialize)]
struct ViewpointWire {
    id: String,
    name: String,
    #[serde(default)]
    description: String,
    saved_unix_ms: u128,
    body: String,
    spawn: ViewpointSpawn,
    #[serde(default)]
    boots_hub: bool,
    sim_time_s: f64,
    camera_position_body_m: [f64; 3],
    camera_rotation_body_xyzw: [f64; 4],
    #[serde(default)]
    optics: Option<CameraOptics>,
    #[serde(default)]
    vertical_fov_rad: Option<f32>,
    #[serde(default)]
    viewport: Option<[u32; 2]>,
}

impl<'de> Deserialize<'de> for Viewpoint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ViewpointWire::deserialize(deserializer)?;
        let optics = match wire.optics {
            Some(optics) => optics,
            None => CameraOptics::from_vertical_fov(
                wire.vertical_fov_rad
                    .ok_or_else(|| D::Error::custom("viewpoint has no optics or legacy FOV"))?,
                wire.viewport
                    .ok_or_else(|| D::Error::custom("legacy viewpoint has no viewport"))?,
            )
            .map_err(D::Error::custom)?,
        };
        Ok(Self {
            id: wire.id,
            name: wire.name,
            description: wire.description,
            saved_unix_ms: wire.saved_unix_ms,
            body: wire.body,
            spawn: wire.spawn,
            boots_hub: wire.boots_hub,
            sim_time_s: wire.sim_time_s,
            camera_position_body_m: wire.camera_position_body_m,
            camera_rotation_body_xyzw: wire.camera_rotation_body_xyzw,
            optics,
        })
    }
}

/// A named agent view whose camera/focus is resolved by a procedural capture
/// driver. The catalog owns its public identity and metadata; `driver` names
/// the runtime capability that performs searches or diagnostic setup.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ScriptedViewpoint {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub driver: String,
}

impl ScriptedViewpoint {
    pub fn validate(&self) -> Result<(), String> {
        if !valid_viewpoint_id(&self.id) {
            return Err(format!("scripted viewpoint id {:?} is invalid", self.id));
        }
        if self.name.trim().is_empty() {
            return Err(format!(
                "scripted viewpoint {:?} has no display name",
                self.id
            ));
        }
        if !CAPTURE_PRESETS.contains(&self.driver.as_str()) || self.driver == "latest-perspective" {
            return Err(format!(
                "scripted viewpoint {:?} names unknown driver {:?}",
                self.id, self.driver
            ));
        }
        Ok(())
    }
}

/// Versioned source-of-truth shared by the in-game manager and capture tools.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ViewpointCatalog {
    pub schema: String,
    #[serde(default)]
    pub viewpoints: Vec<Viewpoint>,
    #[serde(default)]
    pub scripted_viewpoints: Vec<ScriptedViewpoint>,
}

impl Default for ViewpointCatalog {
    fn default() -> Self {
        Self {
            schema: VIEWPOINT_CATALOG_SCHEMA.to_owned(),
            viewpoints: Vec::new(),
            scripted_viewpoints: Vec::new(),
        }
    }
}

impl ViewpointCatalog {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != VIEWPOINT_CATALOG_SCHEMA {
            return Err(format!(
                "unsupported viewpoint catalog schema {:?}; expected {VIEWPOINT_CATALOG_SCHEMA}",
                self.schema
            ));
        }
        for (index, viewpoint) in self.viewpoints.iter().enumerate() {
            viewpoint.validate()?;
            if CAPTURE_PRESETS.contains(&viewpoint.id.as_str())
                || matches!(viewpoint.id.as_str(), "latest" | "perspective")
            {
                return Err(format!(
                    "viewpoint id {:?} is reserved by the capture interface",
                    viewpoint.id
                ));
            }
            if self.viewpoints[..index]
                .iter()
                .any(|other| other.id == viewpoint.id)
                || self
                    .scripted_viewpoints
                    .iter()
                    .any(|other| other.id == viewpoint.id)
            {
                return Err(format!("duplicate viewpoint id {:?}", viewpoint.id));
            }
        }
        for (index, viewpoint) in self.scripted_viewpoints.iter().enumerate() {
            viewpoint.validate()?;
            if matches!(
                viewpoint.id.as_str(),
                "latest" | "perspective" | "latest-perspective"
            ) || self.scripted_viewpoints[..index]
                .iter()
                .any(|other| other.id == viewpoint.id)
            {
                return Err(format!(
                    "duplicate or reserved viewpoint id {:?}",
                    viewpoint.id
                ));
            }
        }
        Ok(())
    }

    pub fn find(&self, id: &str) -> Option<&Viewpoint> {
        self.viewpoints.iter().find(|viewpoint| viewpoint.id == id)
    }

    pub fn find_scripted(&self, id: &str) -> Option<&ScriptedViewpoint> {
        self.scripted_viewpoints
            .iter()
            .find(|viewpoint| viewpoint.id == id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.find(id).is_some() || self.find_scripted(id).is_some()
    }

    pub fn latest(&self) -> Option<&Viewpoint> {
        self.viewpoints
            .iter()
            .max_by_key(|viewpoint| viewpoint.saved_unix_ms)
    }
}

pub fn valid_viewpoint_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && !id.starts_with('-')
        && !id.ends_with('-')
        && !id.contains("--")
}

/// Produce a valid initial id for a newly named viewpoint.
pub fn viewpoint_id_from_name(name: &str) -> String {
    let mut id = String::with_capacity(name.len().min(64));
    let mut separator_pending = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            let separator_len = usize::from(separator_pending && !id.is_empty());
            if id.len() + separator_len + 1 > 64 {
                break;
            }
            if separator_len == 1 {
                id.push('-');
            }
            id.push(ch.to_ascii_lowercase());
            separator_pending = false;
        } else {
            separator_pending = true;
        }
    }
    while id.ends_with('-') {
        id.pop();
    }
    id
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureAction {
    #[default]
    Capture,
    Shutdown,
    #[serde(other)]
    Unsupported,
}

#[derive(Deserialize)]
struct ViewpointCatalogWire {
    schema: String,
    #[serde(default)]
    viewpoints: Vec<Viewpoint>,
    #[serde(default)]
    scripted_viewpoints: Vec<ScriptedViewpoint>,
}

impl<'de> Deserialize<'de> for ViewpointCatalog {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ViewpointCatalogWire::deserialize(deserializer)?;
        let schema = match wire.schema.as_str() {
            LEGACY_VIEWPOINT_CATALOG_SCHEMA | VIEWPOINT_CATALOG_SCHEMA => {
                VIEWPOINT_CATALOG_SCHEMA.to_owned()
            }
            other => other.to_owned(),
        };
        Ok(Self {
            schema,
            viewpoints: wire.viewpoints,
            scripted_viewpoints: wire.scripted_viewpoints,
        })
    }
}

/// Workspace source floor attributed to a capture.
///
/// `build_fingerprint` covers Rust/Cargo inputs that require a host rebuild.
/// `fingerprint` additionally covers hot-reloaded shaders and authored capture
/// configuration. A controller may accept a renderer prepared from this state
/// or one reached later while the build was in flight; exact post-capture
/// equality is recorded separately. The Git fields are human-readable context.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureSourceSnapshot {
    #[serde(default)]
    pub fingerprint: String,
    #[serde(default)]
    pub build_fingerprint: String,
    #[serde(default)]
    pub git_revision: String,
    #[serde(default)]
    pub working_tree_dirty: bool,
}

/// Typed camera changes that may be applied to either a saved viewpoint or a
/// scripted capture framing without changing its pose.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct CaptureCameraOverride {
    #[serde(default)]
    pub focal_length_mm: Option<f32>,
}

impl CaptureCameraOverride {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(focal_length_mm) = self.focal_length_mm
            && (!focal_length_mm.is_finite()
                || !(MIN_FOCAL_LENGTH_MM..=MAX_FOCAL_LENGTH_MM).contains(&focal_length_mm))
        {
            return Err(format!(
                "capture focal length {focal_length_mm} mm is outside {MIN_FOCAL_LENGTH_MM}..={MAX_FOCAL_LENGTH_MM}"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CapturedCameraState {
    pub optics: CameraOptics,
    pub effective_focal_length_mm: f32,
    pub derived_vertical_fov_rad: f32,
    pub output_extent: [u32; 2],
}

/// Graphics preferences applied to one capture request.
///
/// Optional fields make this a patch over the renderer's deterministic capture
/// defaults.
/// Add new user-facing graphics controls here as they become capture-relevant;
/// the request remains backward-compatible because absent fields retain the
/// deterministic capture default.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureGraphicsOverrides {
    #[serde(default)]
    pub clouds: Option<bool>,
    #[serde(default)]
    pub grass: Option<bool>,
}

impl CaptureGraphicsOverrides {
    /// Parse the CLI/environment adapter form, e.g. `clouds=off,grass=on`.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let mut parsed = Self::default();
        if raw.trim().is_empty() {
            return Err("graphics settings cannot be empty".into());
        }
        for assignment in raw.split(',') {
            let (name, value) = assignment
                .trim()
                .split_once('=')
                .ok_or_else(|| format!("graphics setting {assignment:?} must be NAME=VALUE"))?;
            let enabled = parse_capture_bool(value).ok_or_else(|| {
                format!("graphics setting {name:?} expects on/off, got {value:?}")
            })?;
            let slot = match name.trim() {
                "clouds" => &mut parsed.clouds,
                "grass" => &mut parsed.grass,
                other => return Err(format!("unknown graphics setting {other:?}")),
            };
            if slot.replace(enabled).is_some() {
                return Err(format!(
                    "graphics setting {:?} was supplied twice",
                    name.trim()
                ));
            }
        }
        Ok(parsed)
    }
}

fn parse_capture_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Effective graphics settings recorded with a completed capture.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureGraphicsSettings {
    pub clouds: bool,
    pub grass: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CaptureRequest {
    pub schema_version: u32,
    pub id: String,
    #[serde(default)]
    pub action: CaptureAction,
    #[serde(default)]
    pub preset: String,
    #[serde(default)]
    pub overrides: BTreeMap<String, String>,
    #[serde(default)]
    pub source: CaptureSourceSnapshot,
    #[serde(default)]
    pub camera: CaptureCameraOverride,
    #[serde(default)]
    pub graphics: CaptureGraphicsOverrides,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CaptureResponse {
    pub schema_version: u32,
    pub id: String,
    pub ok: bool,
    pub message: String,
    pub output: Option<String>,
    pub completed_unix_ms: u128,
    #[serde(default)]
    pub source: CaptureSourceSnapshot,
    #[serde(default)]
    pub camera: Option<CapturedCameraState>,
    #[serde(default)]
    pub graphics: Option<CaptureGraphicsSettings>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureServerState {
    pub schema_version: u32,
    pub pid: u32,
    pub preset: String,
    /// Presets that can be captured without rebuilding this boot world.
    #[serde(default)]
    pub compatible_presets: Vec<String>,
    pub width: u32,
    pub height: u32,
    pub ready: bool,
    pub busy: bool,
    pub completed_captures: u64,
    pub shader_reload_unix_ms: u128,
    pub heartbeat_unix_ms: u128,
    /// Source snapshot supplied by the controller when this host was launched.
    #[serde(default)]
    pub source: CaptureSourceSnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_request_deserializes() {
        let request: CaptureRequest = serde_json::from_str(
            r#"{"schema_version":5,"id":"abc","action":"capture","preset":"hub","overrides":{}}"#,
        )
        .expect("deserialize request");
        assert_eq!(request.action, CaptureAction::Capture);
        assert_eq!(request.preset, "hub");
        assert_eq!(request.source, CaptureSourceSnapshot::default());
        assert_eq!(request.camera, CaptureCameraOverride::default());
        assert_eq!(request.graphics, CaptureGraphicsOverrides::default());
    }

    #[test]
    fn capture_graphics_parser_is_typed_and_partial() {
        let parsed = CaptureGraphicsOverrides::parse("clouds=off, grass=on").unwrap();
        assert_eq!(
            parsed,
            CaptureGraphicsOverrides {
                clouds: Some(false),
                grass: Some(true),
            }
        );
        assert!(CaptureGraphicsOverrides::parse("clouds=maybe").is_err());
        assert!(CaptureGraphicsOverrides::parse("trees=off").is_err());
        assert!(CaptureGraphicsOverrides::parse("grass=on,grass=off").is_err());
    }

    #[test]
    fn capture_source_snapshot_round_trips() {
        let source = CaptureSourceSnapshot {
            fingerprint: "capture-hash".into(),
            build_fingerprint: "build-hash".into(),
            git_revision: "deadbeef".into(),
            working_tree_dirty: true,
        };
        let encoded = serde_json::to_string(&source).unwrap();
        assert_eq!(
            serde_json::from_str::<CaptureSourceSnapshot>(&encoded).unwrap(),
            source
        );
    }

    #[test]
    fn viewpoint_catalog_rejects_duplicate_ids() {
        let viewpoint = Viewpoint {
            id: "ridge-light".into(),
            name: "Ridge light".into(),
            description: String::new(),
            saved_unix_ms: 1,
            body: "Mira".into(),
            spawn: ViewpointSpawn::Orbit,
            boots_hub: false,
            sim_time_s: 0.0,
            camera_position_body_m: [1.0, 2.0, 3.0],
            camera_rotation_body_xyzw: [0.0, 0.0, 0.0, 1.0],
            optics: CameraOptics::from_vertical_fov(1.0, [16, 9]).unwrap(),
        };
        let catalog = ViewpointCatalog {
            schema: VIEWPOINT_CATALOG_SCHEMA.into(),
            viewpoints: vec![viewpoint.clone(), viewpoint],
            scripted_viewpoints: Vec::new(),
        };
        assert!(catalog.validate().is_err());
    }

    #[test]
    fn viewpoint_name_becomes_stable_slug() {
        assert_eq!(
            viewpoint_id_from_name("  Mira: Ridge @ Dawn  "),
            "mira-ridge-dawn"
        );
        assert_eq!(viewpoint_id_from_name("A--B"), "a-b");
    }

    #[test]
    fn checked_in_viewpoint_catalog_matches_the_protocol() {
        let catalog: ViewpointCatalog =
            serde_json::from_str(include_str!("../../../../assets/viewpoints.json"))
                .expect("checked-in viewpoint JSON parses");
        catalog
            .validate()
            .expect("checked-in viewpoint catalog validates");
        let expected_drivers = CAPTURE_PRESETS
            .iter()
            .copied()
            .filter(|driver| *driver != "latest-perspective")
            .collect::<Vec<_>>();
        assert_eq!(
            catalog.scripted_viewpoints.len(),
            expected_drivers.len(),
            "every compiled agent-view driver must be represented in the unified catalog"
        );
        for driver in expected_drivers {
            assert!(
                catalog
                    .scripted_viewpoints
                    .iter()
                    .any(|viewpoint| viewpoint.driver == driver),
                "missing catalog entry for agent-view driver {driver}"
            );
        }
    }

    #[test]
    fn full_frame_horizontal_conversion_round_trips_common_aspects() {
        for aspect in [[4, 3], [3, 2], [16, 9], [21, 9]] {
            for focal_length_mm in [12.0, 24.0, 35.0, 50.0, 85.0, 135.0, 400.0] {
                let optics = CameraOptics::new(focal_length_mm, aspect).unwrap();
                let round_trip =
                    CameraOptics::from_vertical_fov(optics.vertical_fov_rad(), aspect).unwrap();
                assert!(
                    (round_trip.lens.focal_length_mm - focal_length_mm).abs() < 1.0e-3,
                    "aspect={aspect:?} focal={focal_length_mm} round_trip={round_trip:?}"
                );
            }
        }
    }

    #[test]
    fn horizontal_fov_is_sensor_aspect_invariant() {
        let expected = CameraOptics::new(50.0, [16, 9])
            .unwrap()
            .horizontal_fov_rad();
        for aspect in [[4, 3], [3, 2], [21, 9]] {
            let actual = CameraOptics::new(50.0, aspect)
                .unwrap()
                .horizontal_fov_rad();
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn legacy_viewpoint_migrates_fov_and_discards_pixel_dimensions() {
        let legacy = r#"{
            "id":"legacy","name":"Legacy","saved_unix_ms":1,"body":"Thalos",
            "spawn":"orbit","sim_time_s":0.0,
            "camera_position_body_m":[1.0,2.0,3.0],
            "camera_rotation_body_xyzw":[0.0,0.0,0.0,1.0],
            "vertical_fov_rad":0.7853982,"viewport":[3840,2160]
        }"#;
        let viewpoint: Viewpoint = serde_json::from_str(legacy).unwrap();
        assert_eq!(viewpoint.optics.sensor.aspect, [16, 9]);
        assert!((viewpoint.optics.vertical_fov_rad() - 0.7853982).abs() < 1.0e-6);
        let encoded = serde_json::to_string(&viewpoint).unwrap();
        assert!(!encoded.contains("vertical_fov_rad"));
        assert!(!encoded.contains("viewport"));
    }
}
