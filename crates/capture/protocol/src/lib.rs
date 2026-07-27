//! Portable control-plane types shared by capture clients and the renderer.
//!
//! This crate intentionally contains no ECS, GPU, camera, or application
//! types. It describes requests and results; `thalos_capture_runtime` turns
//! them into work performed by the complete Bevy game renderer.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const CAPTURE_PROTOCOL_SCHEMA: u32 = 2;
pub const VIEWPOINT_CATALOG_SCHEMA: &str = "thalos.viewpoints.v1";

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
    "plume",
    "plume-skyline",
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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
    /// Provenance only. Replay uses the spawn's canonical time rather than
    /// restoring a half-snapshot of craft/simulation state.
    pub sim_time_s: f64,
    pub camera_position_body_m: [f64; 3],
    pub camera_rotation_body_xyzw: [f64; 4],
    pub vertical_fov_rad: f32,
    pub viewport: [u32; 2],
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
        if !self.vertical_fov_rad.is_finite()
            || !(1.0f32.to_radians()..179.0f32.to_radians()).contains(&self.vertical_fov_rad)
        {
            return Err(format!(
                "viewpoint {:?} has an invalid vertical FOV",
                self.id
            ));
        }
        if self.viewport[0] == 0 || self.viewport[1] == 0 {
            return Err(format!("viewpoint {:?} has an empty viewport", self.id));
        }
        Ok(())
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureRequest {
    pub schema_version: u32,
    pub id: String,
    #[serde(default)]
    pub action: CaptureAction,
    #[serde(default)]
    pub preset: String,
    #[serde(default)]
    pub overrides: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureResponse {
    pub schema_version: u32,
    pub id: String,
    pub ok: bool,
    pub message: String,
    pub output: Option<String>,
    pub completed_unix_ms: u128,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_request_deserializes() {
        let request: CaptureRequest = serde_json::from_str(
            r#"{"schema_version":2,"id":"abc","action":"capture","preset":"hub","overrides":{}}"#,
        )
        .expect("deserialize request");
        assert_eq!(request.action, CaptureAction::Capture);
        assert_eq!(request.preset, "hub");
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
            vertical_fov_rad: 1.0,
            viewport: [1920, 1080],
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
}
