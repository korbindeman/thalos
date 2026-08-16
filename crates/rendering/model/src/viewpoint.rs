//! Portable saved-viewpoint model shared by interactive apps and capture.

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

use crate::CameraOptics;

pub const VIEWPOINT_CATALOG_SCHEMA: &str = "thalos.viewpoints.v3";
const LEGACY_VIEWPOINT_CATALOG_SCHEMAS: [&str; 2] =
    ["thalos.viewpoints.v1", "thalos.viewpoints.v2"];

/// The one declaration of game-side procedural viewpoint drivers.
///
/// The lightweight viewer carries only these names as catalog validation data;
/// executors remain behind the full game's capture capability.
#[macro_export]
macro_rules! capture_preset_catalog {
    ($receiver:ident) => {
        $receiver! {
            LatestPerspective => "latest-perspective",
            SpaceportAerial => "spaceport-aerial",
            RunwayAtmosphere => "runway-atmosphere",
            CraftStance => "craft-stance",
            PavedGround => "paved-ground",
            Hub => "hub",
            DryBelt => "dry-belt",
            ForestStand => "forest-stand",
            EarthReference => "earth-reference",
            Ocean => "ocean",
            OceanSlopes => "ocean-slopes",
            Coastline => "coastline",
            MiraOrbit => "mira-orbit",
            MiraSurface => "mira-surface",
            MiraEva => "mira-eva",
            MiraDisc => "mira-disc",
            MiraApproach => "mira-approach",
            MiraRim => "mira-rim",
            CloudRunway => "cloud-runway",
            CloudMotion => "cloud-motion",
            CloudCruise => "cloud-cruise",
            CloudInterior => "cloud-interior",
            CloudLimb => "cloud-limb",
            CloudLeo => "cloud-leo",
            CloudPlanet => "cloud-planet",
            CloudSunset => "cloud-sunset",
            CloudGodray => "cloud-godray",
            Plume => "plume",
            PlumeSkyline => "plume-skyline",
            Reentry => "reentry",
            VaporCone => "vapor-cone",
            Interstage => "interstage",
            OrbitHull => "orbit-hull",
            MassifAerial => "massif-aerial",
            MassifRidge => "massif-ridge",
            MassifValley => "massif-valley",
        }
    };
}

macro_rules! capture_preset_names {
    ($($variant:ident => $name:literal,)*) => {
        pub const CAPTURE_PRESETS: &[&str] = &[$($name),*];
    };
}

capture_preset_catalog!(capture_preset_names);

/// Canonical scene builder behind an authored body-fixed viewpoint.
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

/// Stable spatial frame in which a saved camera pose is authored.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ViewpointFrame {
    /// A body's authored surface frame, replayed through its current state.
    AuthoredBodyFixed {
        body: String,
        spawn: ViewpointSpawn,
        #[serde(default)]
        boots_hub: bool,
        sim_time_s: f64,
    },
    /// A bounded projected-local metric frame owned by an application adapter.
    ProjectedLocal { reference: String },
}

impl ViewpointFrame {
    pub fn label(&self) -> &str {
        match self {
            Self::AuthoredBodyFixed { body, .. } => body,
            Self::ProjectedLocal { reference } => reference,
        }
    }
}

/// One stable, exchangeable camera point.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Viewpoint {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub saved_unix_ms: u128,
    pub frame: ViewpointFrame,
    pub camera_position_m: [f64; 3],
    pub camera_rotation_xyzw: [f64; 4],
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
        match &self.frame {
            ViewpointFrame::AuthoredBodyFixed {
                body, sim_time_s, ..
            } => {
                if body.trim().is_empty() {
                    return Err(format!("viewpoint {:?} has no target body", self.id));
                }
                if !sim_time_s.is_finite() {
                    return Err(format!(
                        "viewpoint {:?} contains a non-finite simulation time",
                        self.id
                    ));
                }
            }
            ViewpointFrame::ProjectedLocal { reference } if reference.trim().is_empty() => {
                return Err(format!(
                    "viewpoint {:?} has no projected frame reference",
                    self.id
                ));
            }
            ViewpointFrame::ProjectedLocal { .. } => {}
        }
        if !self.camera_position_m.iter().all(|value| value.is_finite())
            || !self
                .camera_rotation_xyzw
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(format!(
                "viewpoint {:?} contains a non-finite camera number",
                self.id
            ));
        }
        let rotation_len_sq = self
            .camera_rotation_xyzw
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

    pub fn authored_body(&self) -> Option<(&str, ViewpointSpawn, bool, f64)> {
        match &self.frame {
            ViewpointFrame::AuthoredBodyFixed {
                body,
                spawn,
                boots_hub,
                sim_time_s,
            } => Some((body, *spawn, *boots_hub, *sim_time_s)),
            ViewpointFrame::ProjectedLocal { .. } => None,
        }
    }
}

#[derive(Deserialize)]
struct ViewpointWire {
    id: String,
    name: String,
    #[serde(default)]
    description: String,
    saved_unix_ms: u128,
    #[serde(default)]
    frame: Option<ViewpointFrame>,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    spawn: Option<ViewpointSpawn>,
    #[serde(default)]
    boots_hub: bool,
    #[serde(default)]
    sim_time_s: Option<f64>,
    #[serde(alias = "camera_position_body_m")]
    camera_position_m: [f64; 3],
    #[serde(alias = "camera_rotation_body_xyzw")]
    camera_rotation_xyzw: [f64; 4],
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
        let frame = match wire.frame {
            Some(frame) => frame,
            None => ViewpointFrame::AuthoredBodyFixed {
                body: wire
                    .body
                    .ok_or_else(|| D::Error::custom("legacy viewpoint has no body"))?,
                spawn: wire
                    .spawn
                    .ok_or_else(|| D::Error::custom("legacy viewpoint has no spawn"))?,
                boots_hub: wire.boots_hub,
                sim_time_s: wire
                    .sim_time_s
                    .ok_or_else(|| D::Error::custom("legacy viewpoint has no simulation time"))?,
            },
        };
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
            frame,
            camera_position_m: wire.camera_position_m,
            camera_rotation_xyzw: wire.camera_rotation_xyzw,
            optics,
        })
    }
}

/// A named view whose pose is resolved by an application capability.
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

/// Versioned source of truth shared by interactive apps and capture tools.
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
        let schema = if wire.schema == VIEWPOINT_CATALOG_SCHEMA
            || LEGACY_VIEWPOINT_CATALOG_SCHEMAS.contains(&wire.schema.as_str())
        {
            VIEWPOINT_CATALOG_SCHEMA.to_owned()
        } else {
            wire.schema
        };
        Ok(Self {
            schema,
            viewpoints: wire.viewpoints,
            scripted_viewpoints: wire.scripted_viewpoints,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_body_fixed_viewpoint_migrates_to_a_typed_frame() {
        let json = r#"{
            "id":"ridge","name":"Ridge","description":"","saved_unix_ms":1,
            "body":"Mira","spawn":"orbit","boots_hub":false,"sim_time_s":42.0,
            "camera_position_body_m":[1.0,2.0,3.0],
            "camera_rotation_body_xyzw":[0.0,0.0,0.0,1.0],
            "optics":{"lens":{"model":"full-frame-horizontal","focal_length_mm":35.0},"sensor":{"gate_width_mm":36.0,"aspect":[16,9],"crop":"full"}}
        }"#;
        let viewpoint: Viewpoint = serde_json::from_str(json).unwrap();
        assert_eq!(
            viewpoint.authored_body(),
            Some(("Mira", ViewpointSpawn::Orbit, false, 42.0))
        );
        assert_eq!(viewpoint.camera_position_m, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn projected_local_viewpoint_round_trips() {
        let viewpoint = Viewpoint {
            id: "westpunt".into(),
            name: "Westpunt".into(),
            description: String::new(),
            saved_unix_ms: 1,
            frame: ViewpointFrame::ProjectedLocal {
                reference: "EPSG:32619".into(),
            },
            camera_position_m: [1.0, 2.0, 3.0],
            camera_rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            optics: CameraOptics::default(),
        };
        let encoded = serde_json::to_string(&viewpoint).unwrap();
        assert_eq!(
            serde_json::from_str::<Viewpoint>(&encoded).unwrap(),
            viewpoint
        );
    }
}
