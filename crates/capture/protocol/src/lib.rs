//! Portable control-plane types shared by capture clients and the renderer.
//!
//! This crate intentionally contains no ECS, GPU, camera, or application
//! types. It describes requests and results; `thalos_capture_runtime` turns
//! them into work performed by the complete Bevy game renderer.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
pub use thalos_render_model::RenderPlan;

pub const CAPTURE_PROTOCOL_SCHEMA: u32 = 5;

pub use thalos_render_model::{
    CameraLens, CameraLensModel, CameraOptics, CameraSensor, CapturedCameraState,
    FULL_FRAME_GATE_WIDTH_MM, MAX_FOCAL_LENGTH_MM, MIN_FOCAL_LENGTH_MM, SensorCrop, reduced_aspect,
};

pub use thalos_render_model::{
    CAPTURE_PRESETS, ScriptedViewpoint, VIEWPOINT_CATALOG_SCHEMA, Viewpoint, ViewpointCatalog,
    ViewpointFrame, ViewpointSpawn, capture_preset_catalog, valid_viewpoint_id,
    viewpoint_id_from_name,
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureAction {
    #[default]
    Capture,
    Shutdown,
    #[serde(other)]
    Unsupported,
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
    #[serde(default)]
    pub foliage: Option<bool>,
}

impl CaptureGraphicsOverrides {
    /// Parse the CLI/environment adapter form, e.g.
    /// `clouds=off,grass=on,foliage=off`.
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
                "foliage" => &mut parsed.foliage,
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
    #[serde(default = "capture_graphics_default_on")]
    pub foliage: bool,
    #[serde(default = "capture_graphics_default_shadow_cascades")]
    pub shadow_cascades: u8,
    #[serde(default = "capture_graphics_default_shadow_map_size")]
    pub shadow_map_size_px: u32,
}

fn capture_graphics_default_on() -> bool {
    true
}

fn capture_graphics_default_shadow_cascades() -> u8 {
    4
}

fn capture_graphics_default_shadow_map_size() -> u32 {
    4096
}

/// Tile-terrain residency at the moment the image was read back.
///
/// A capture can render the ground **coarser than the preset authored** and
/// still exit zero with a plausible PNG: the tile memory brake coarsens
/// selection when residency runs over this process's VRAM share, and headless
/// capture deliberately runs a smaller allowance than an interactive session.
/// Before this existed, `just diag` could report that a session braked but no
/// reader could map that onto a *file* — so this rides in the receipt beside
/// the image, the same way source provenance does.
///
/// `split_scale` is 1.0 when the brake is not holding anything back;
/// `MIN_SPLIT_SCALE` (≈ 0.333) is the floor. `worst_split_scale` covers the
/// whole request — settle included — because a brake that bit during warmup and
/// released before readback did not affect the image but does explain a slow
/// shot.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CaptureTerrainResidency {
    /// Split scale at readback. Below 1.0 means this image is coarser than the
    /// preset authored.
    pub split_scale: f64,
    /// Worst split scale seen between arming the request and reading it back.
    pub worst_split_scale: f64,
    /// Landed tiles at readback, and what the selector wanted before the brake.
    pub resident_tiles: usize,
    pub desired_tiles: usize,
    /// True only when the desired leaf set is fully covered and no tile work
    /// remains in flight at readback. A plausible PNG with this false is not
    /// settled terrain evidence even when `split_scale == 1`.
    #[serde(default)]
    pub settled: bool,
    /// Wall-clock seconds the capture held after its ordinary frame warmup for
    /// desired coverage to settle.
    #[serde(default)]
    pub settle_wait_s: f64,
    pub resident_mib: f64,
    /// This process's share of the machine-wide tile budget, MiB. `None` when
    /// the budget is disabled (`THALOS_TILE_BUDGET_MB=0`).
    pub budget_mib: Option<f64>,
    /// Live renderer instances the share was divided across. More than one
    /// means a peer halved this host's allowance — read it before blaming the
    /// preset (INC-20260725T012104Z).
    pub instances: usize,
    /// Seconds the readback waited for the brake to release. Non-zero with
    /// `split_scale` back at 1.0 is the gate working; non-zero with
    /// `split_scale` below 1.0 means the wait timed out.
    pub brake_wait_s: f64,
}

impl CaptureTerrainResidency {
    /// Did this image render coarser than the preset authored?
    pub fn braked(&self) -> bool {
        self.split_scale < 1.0
    }
}

/// Where a shot's canonical simulation time came from.
///
/// Recorded per shot because the lighting of a capture is entirely a function of
/// it, and because the failure this enum exists to expose is silent: see
/// [`CaptureClock::sim_time_s`].
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureTimeSource {
    /// The spawn scenario's authored boot epoch — what an untimed shot resolves
    /// to, and what makes repeat shots of one preset comparable.
    PresetBootEpoch,
    /// A saved viewpoint's recorded `sim_time_s`.
    ViewpointMetadata,
    /// An explicit `--time` / `THALOS_SCREENSHOT_TIME` from the caller.
    CallerOverride,
    /// Nothing pinned the time: the scenario authors no epoch and the caller
    /// asked for none, so the image was rendered at whatever the host's clock
    /// had reached. **The image is not reproducible** — a rerun on a warm host,
    /// or after any other shot, can light it differently. Treated as a defect
    /// to be closed by giving the scenario an epoch, not as a normal mode.
    HostClock,
}

/// The clock the renderer produced this image under.
///
/// A **wall** clock advances the world by however long each frame took, so the
/// same preset settles to a *different* world state on a busy machine than on
/// an idle one — visible wherever anything is time-dependent (cloud advection,
/// plumes, the settle gate). A **driven** clock advances every frame by exactly
/// `driven_dt_s` regardless of render cost, which is what makes a rerun
/// comparable and what lets frame *n* of a sequence land at *n · dt*.
///
/// `driven_dt_s = None` means the wall clock. That is also what a host predating
/// the driven mode reports, and it is the truth for it — so an absent field is
/// never ambiguous.
///
/// `driven_dt_s` is a **boot** property of the host; `sim_time_s` /
/// `sim_time_source` are **per request**.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct CaptureClock {
    #[serde(default)]
    pub driven_dt_s: Option<f64>,
    /// Canonical simulation time (s) the world was seated to for this shot —
    /// i.e. the time of day the image is lit by.
    ///
    /// Recorded because a wrong value here is otherwise undetectable from the
    /// artifact: a resident host serves many requests, and until
    /// BL-20260731T202657Z every request without an explicit `--time` simply
    /// left the clock wherever the previous request had put it. That produced a
    /// correct-looking PNG at another shot's sun, exit 0, and no way for the
    /// agent reading the image to tell. Compare it against the preset's epoch
    /// (or against the sibling shots of a comparison) before trusting a matched
    /// pair.
    ///
    /// `None` only from a host predating the field.
    #[serde(default)]
    pub sim_time_s: Option<f64>,
    #[serde(default)]
    pub sim_time_source: Option<CaptureTimeSource>,
}

impl CaptureClock {
    pub const WALL: Self = Self {
        driven_dt_s: None,
        sim_time_s: None,
        sim_time_source: None,
    };

    pub fn driven(dt_s: f64) -> Self {
        Self {
            driven_dt_s: Some(dt_s),
            ..Self::WALL
        }
    }

    pub fn is_driven(&self) -> bool {
        self.driven_dt_s.is_some()
    }

    /// This shot's time was pinned by the preset, a viewpoint, or the caller —
    /// so a rerun reproduces its lighting.
    pub fn sim_time_pinned(&self) -> bool {
        !matches!(
            self.sim_time_source,
            Some(CaptureTimeSource::HostClock) | None
        )
    }
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
    /// Validated restart-time renderer composition that produced the image.
    /// Absent only on responses from a host predating render-plan selection.
    #[serde(default)]
    pub render_plan: Option<RenderPlan>,
    /// Ground residency at readback. `None` on the legacy udlod path or a body
    /// the tile renderer has not installed on — "not applicable", not "fine".
    #[serde(default)]
    pub terrain: Option<CaptureTerrainResidency>,
    /// Wall or driven clock. Defaults to wall, which is what a host predating
    /// the mode ran on — so the field needs no schema bump to stay truthful.
    #[serde(default)]
    pub clock: CaptureClock,
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
        let parsed = CaptureGraphicsOverrides::parse("clouds=off, grass=on, foliage=off").unwrap();
        assert_eq!(
            parsed,
            CaptureGraphicsOverrides {
                clouds: Some(false),
                grass: Some(true),
                foliage: Some(false),
            }
        );
        assert!(CaptureGraphicsOverrides::parse("clouds=maybe").is_err());
        assert!(CaptureGraphicsOverrides::parse("trees=off").is_err());
        assert!(CaptureGraphicsOverrides::parse("grass=on,grass=off").is_err());
    }

    #[test]
    fn legacy_capture_graphics_defaults_new_quality_fields() {
        let parsed: CaptureGraphicsSettings =
            serde_json::from_str(r#"{"clouds":true,"grass":false}"#).unwrap();
        assert!(parsed.foliage);
        assert_eq!(parsed.shadow_cascades, 4);
        assert_eq!(parsed.shadow_map_size_px, 4096);
    }

    /// A receipt written before the driven clock existed must still read as
    /// "wall", not as an unknown. That is what lets the field ship without a
    /// schema bump.
    #[test]
    fn a_receipt_without_a_clock_block_reads_as_wall() {
        let response: CaptureResponse = serde_json::from_str(
            r#"{"schema_version":5,"id":"abc","ok":true,"message":"",
                "output":null,"completed_unix_ms":1}"#,
        )
        .expect("legacy response deserializes");
        assert_eq!(response.clock, CaptureClock::WALL);
        assert!(!response.clock.is_driven());

        let driven = CaptureClock::driven(1.0 / 60.0);
        let round_trip: CaptureClock =
            serde_json::from_str(&serde_json::to_string(&driven).unwrap()).unwrap();
        assert_eq!(round_trip, driven);
        assert!(round_trip.is_driven());
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
            frame: ViewpointFrame::AuthoredBodyFixed {
                body: "Mira".into(),
                spawn: ViewpointSpawn::Orbit,
                boots_hub: false,
                sim_time_s: 0.0,
            },
            camera_position_m: [1.0, 2.0, 3.0],
            camera_rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
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
