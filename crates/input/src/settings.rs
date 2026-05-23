use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use bevy::prelude::{KeyCode, MouseButton, Resource};
use bevy_enhanced_input::prelude::Binding;
use serde::Deserialize;

pub const INPUT_SETTINGS_VERSION: u32 = 1;

#[derive(Resource, Debug, Clone, PartialEq)]
pub struct InputSettings {
    pub version: u32,
    pub game: GameInputSettings,
    pub planet_editor: BindingSection,
    pub shipyard: BindingSection,
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            version: INPUT_SETTINGS_VERSION,
            game: GameInputSettings::default(),
            planet_editor: defaults::planet_editor(),
            shipyard: defaults::shipyard(),
        }
    }
}

impl InputSettings {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, InputLoadError> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|source| InputLoadError::Read {
            path: path.into(),
            source,
        })?;
        Self::from_ron_str(&source).map_err(|source| InputLoadError::Parse {
            path: path.into(),
            source: Box::new(source),
        })
    }

    pub fn from_ron_str(source: &str) -> Result<Self, InputSettingsError> {
        let file: InputFile = ron::from_str(source).map_err(InputSettingsError::Ron)?;
        Self::from_file(file)
    }

    pub fn from_file(file: InputFile) -> Result<Self, InputSettingsError> {
        validate_file(&file)?;
        let mut settings = Self {
            version: file.version.unwrap_or(INPUT_SETTINGS_VERSION),
            ..Default::default()
        };
        settings.game.merge(file.game);
        settings.planet_editor.merge(file.planet_editor);
        settings.shipyard.merge(file.shipyard);
        Ok(settings)
    }
}

#[derive(Debug)]
pub enum InputSettingsError {
    Ron(ron::error::SpannedError),
    Validation(InputValidationError),
}

impl fmt::Display for InputSettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ron(source) => write!(f, "{source}"),
            Self::Validation(source) => write!(f, "{source}"),
        }
    }
}

impl std::error::Error for InputSettingsError {}

#[derive(Debug)]
pub struct InputValidationError {
    path: String,
    message: String,
}

impl InputValidationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for InputValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl std::error::Error for InputValidationError {}

#[derive(Debug)]
pub enum InputLoadError {
    Read {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: std::path::PathBuf,
        source: Box<InputSettingsError>,
    },
}

impl fmt::Display for InputLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(
                    f,
                    "could not read input bindings {}: {source}",
                    path.display()
                )
            }
            Self::Parse { path, source } => {
                write!(
                    f,
                    "could not parse input bindings {}: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for InputLoadError {}

fn validate_file(file: &InputFile) -> Result<(), InputSettingsError> {
    let defaults = InputSettings::default();
    validate_section_file("game.system", &file.game.system, &defaults.game.system)?;
    validate_section_file("game.flight", &file.game.flight, &defaults.game.flight)?;
    validate_section_file("game.view", &file.game.view, &defaults.game.view)?;
    validate_section_file("game.camera", &file.game.camera, &defaults.game.camera)?;
    validate_section_file("game.eva", &file.game.eva, &defaults.game.eva)?;
    validate_section_file(
        "game.eva_move",
        &file.game.eva_move,
        &defaults.game.eva_move,
    )?;
    validate_section_file(
        "game.maneuver",
        &file.game.maneuver,
        &defaults.game.maneuver,
    )?;
    validate_section_file(
        "game.maneuver_precision",
        &file.game.maneuver_precision,
        &defaults.game.maneuver_precision,
    )?;
    validate_section_file(
        "planet_editor",
        &file.planet_editor,
        &defaults.planet_editor,
    )?;
    validate_section_file("shipyard", &file.shipyard, &defaults.shipyard)?;
    Ok(())
}

fn validate_section_file(
    path: &str,
    file: &BindingSectionFile,
    defaults: &BindingSection,
) -> Result<(), InputSettingsError> {
    for action in file.bindings.keys() {
        if !defaults.bindings.contains_key(action) {
            return Err(InputSettingsError::Validation(InputValidationError::new(
                format!("{path}.bindings.{action}"),
                "unknown action",
            )));
        }
    }
    // Binding values are validated by serde at parse time — KeyCode and
    // MouseButton variants are real enum types, not loose strings.

    for (axis, _spec) in &file.axes {
        if !defaults.axes.contains_key(axis) {
            return Err(InputSettingsError::Validation(InputValidationError::new(
                format!("{path}.axes.{axis}"),
                "unknown axis",
            )));
        }
        // Axis sources are also validated by serde.
    }

    Ok(())
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
pub struct InputFile {
    pub version: Option<u32>,
    #[serde(default)]
    pub game: GameInputFile,
    #[serde(default)]
    pub planet_editor: BindingSectionFile,
    #[serde(default)]
    pub shipyard: BindingSectionFile,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GameInputSettings {
    pub system: BindingSection,
    pub flight: BindingSection,
    pub warp: BindingSection,
    pub view: BindingSection,
    pub camera: BindingSection,
    pub eva: BindingSection,
    pub eva_move: BindingSection,
    pub maneuver: BindingSection,
    pub maneuver_precision: BindingSection,
}

impl Default for GameInputSettings {
    fn default() -> Self {
        Self {
            system: defaults::game_system(),
            flight: defaults::game_flight(),
            warp: defaults::game_warp(),
            view: defaults::game_view(),
            camera: defaults::game_camera(),
            eva: defaults::game_eva(),
            eva_move: defaults::game_eva_move(),
            maneuver: defaults::game_maneuver(),
            maneuver_precision: defaults::game_maneuver_precision(),
        }
    }
}

impl GameInputSettings {
    fn merge(&mut self, file: GameInputFile) {
        self.system.merge(file.system);
        self.flight.merge(file.flight);
        self.warp.merge(file.warp);
        self.view.merge(file.view);
        self.camera.merge(file.camera);
        self.eva.merge(file.eva);
        self.eva_move.merge(file.eva_move);
        self.maneuver.merge(file.maneuver);
        self.maneuver_precision.merge(file.maneuver_precision);
    }
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
pub struct GameInputFile {
    #[serde(default)]
    pub system: BindingSectionFile,
    #[serde(default)]
    pub flight: BindingSectionFile,
    #[serde(default)]
    pub warp: BindingSectionFile,
    #[serde(default)]
    pub view: BindingSectionFile,
    #[serde(default)]
    pub camera: BindingSectionFile,
    #[serde(default)]
    pub eva: BindingSectionFile,
    #[serde(default)]
    pub eva_move: BindingSectionFile,
    #[serde(default)]
    pub maneuver: BindingSectionFile,
    #[serde(default)]
    pub maneuver_precision: BindingSectionFile,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BindingSection {
    pub bindings: BTreeMap<String, Vec<BindingSpec>>,
    pub axes: BTreeMap<String, AxisSpec>,
}

impl BindingSection {
    pub fn merge(&mut self, file: BindingSectionFile) {
        for (action, bindings) in file.bindings {
            self.bindings.insert(action, bindings);
        }
        for (axis, spec) in file.axes {
            self.axes.insert(axis, spec);
        }
    }

    pub fn bindings(&self, action: &str) -> Vec<Binding> {
        self.bindings
            .get(action)
            .map(|bindings| bindings.iter().map(BindingSpec::to_binding).collect())
            .unwrap_or_default()
    }

    pub fn axis_positive(&self, axis: &str) -> Vec<Binding> {
        self.axes
            .get(axis)
            .map(|axis| axis.positive.iter().map(BindingSpec::to_binding).collect())
            .unwrap_or_default()
    }

    pub fn axis_negative(&self, axis: &str) -> Vec<Binding> {
        self.axes
            .get(axis)
            .map(|axis| axis.negative.iter().map(BindingSpec::to_binding).collect())
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
pub struct BindingSectionFile {
    #[serde(default)]
    pub bindings: BTreeMap<String, Vec<BindingSpec>>,
    #[serde(default)]
    pub axes: BTreeMap<String, AxisSpec>,
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
pub struct AxisSpec {
    #[serde(default)]
    pub positive: Vec<BindingSpec>,
    #[serde(default)]
    pub negative: Vec<BindingSpec>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum BindingSpec {
    Key(KeyCode),
    MouseButton(MouseButton),
    MouseMotion,
    MouseWheel,
}

impl BindingSpec {
    pub fn key(key: KeyCode) -> Self {
        Self::Key(key)
    }

    pub fn mouse_button(button: MouseButton) -> Self {
        Self::MouseButton(button)
    }

    pub fn to_binding(&self) -> Binding {
        match self {
            Self::Key(key) => Binding::from(*key),
            Self::MouseButton(button) => Binding::from(*button),
            Self::MouseMotion => Binding::mouse_motion(),
            Self::MouseWheel => Binding::mouse_wheel(),
        }
    }
}

pub mod defaults {
    use super::{AxisSpec, BindingSection, BindingSpec};
    use bevy::prelude::{KeyCode, MouseButton};

    pub fn game_system() -> BindingSection {
        section(
            [
                ("escape", keys([KeyCode::Escape])),
                ("screenshot", keys([KeyCode::F2])),
                ("toggle_free_cam", keys([KeyCode::F3])),
            ],
            [],
        )
    }

    pub fn game_flight() -> BindingSection {
        section(
            [
                ("toggle_sas", keys([KeyCode::KeyT])),
                ("throttle_full", keys([KeyCode::KeyZ])),
                ("throttle_cut", keys([KeyCode::KeyX])),
                ("stage", keys([KeyCode::Space])),
            ],
            [
                ("pitch", axis([KeyCode::KeyW], [KeyCode::KeyS])),
                ("yaw", axis([KeyCode::KeyD], [KeyCode::KeyA])),
                ("roll", axis([KeyCode::KeyE], [KeyCode::KeyQ])),
                (
                    "throttle_ramp",
                    axis(
                        [KeyCode::ShiftLeft, KeyCode::ShiftRight],
                        [KeyCode::ControlLeft, KeyCode::ControlRight],
                    ),
                ),
            ],
        )
    }

    pub fn game_warp() -> BindingSection {
        section(
            [
                ("warp_to_maneuver", keys([KeyCode::KeyG])),
                ("warp_increase", keys([KeyCode::Period])),
                ("warp_decrease", keys([KeyCode::Comma])),
                ("warp_reset", keys([KeyCode::Backslash])),
            ],
            [],
        )
    }

    pub fn game_view() -> BindingSection {
        section(
            [
                ("toggle_view", keys([KeyCode::KeyM])),
                ("toggle_photo_mode", keys([KeyCode::F1, KeyCode::KeyP])),
                ("cycle_ship_camera", keys([KeyCode::KeyV])),
            ],
            [],
        )
    }

    pub fn game_camera() -> BindingSection {
        section(
            [
                ("primary", mouse_buttons([MouseButton::Left])),
                ("motion", vec![BindingSpec::MouseMotion]),
                ("wheel", vec![BindingSpec::MouseWheel]),
            ],
            [],
        )
    }

    pub fn game_eva() -> BindingSection {
        section([("toggle_player_controller", keys([KeyCode::KeyF]))], [])
    }

    pub fn game_eva_move() -> BindingSection {
        section(
            [
                ("jump", keys([KeyCode::Space])),
                ("sprint", keys([KeyCode::ShiftLeft, KeyCode::ShiftRight])),
            ],
            [
                ("forward", axis([KeyCode::KeyW], [KeyCode::KeyS])),
                ("strafe", axis([KeyCode::KeyD], [KeyCode::KeyA])),
            ],
        )
    }

    pub fn game_maneuver() -> BindingSection {
        section(
            [
                ("toggle_place_node", keys([KeyCode::KeyN])),
                ("delete_node", keys([KeyCode::Delete, KeyCode::Backspace])),
            ],
            [],
        )
    }

    pub fn game_maneuver_precision() -> BindingSection {
        section(
            [
                ("fine", keys([KeyCode::ShiftLeft, KeyCode::ShiftRight])),
                ("ultra", keys([KeyCode::ControlLeft, KeyCode::ControlRight])),
            ],
            [],
        )
    }

    pub fn planet_editor() -> BindingSection {
        section(
            [
                ("primary", mouse_buttons([MouseButton::Left])),
                ("camera_motion", vec![BindingSpec::MouseMotion]),
                ("camera_wheel", vec![BindingSpec::MouseWheel]),
                ("toggle_fullbright", keys([KeyCode::KeyF])),
                ("overlay_suppress", keys([KeyCode::Space])),
            ],
            [],
        )
    }

    pub fn shipyard() -> BindingSection {
        section(
            [
                ("primary", mouse_buttons([MouseButton::Left])),
                ("camera_motion", vec![BindingSpec::MouseMotion]),
                ("camera_wheel", vec![BindingSpec::MouseWheel]),
                (
                    "precision_slow",
                    keys([KeyCode::ShiftLeft, KeyCode::ShiftRight]),
                ),
            ],
            [],
        )
    }

    fn section(
        bindings: impl IntoIterator<Item = (&'static str, Vec<BindingSpec>)>,
        axes: impl IntoIterator<Item = (&'static str, AxisSpec)>,
    ) -> BindingSection {
        BindingSection {
            bindings: bindings
                .into_iter()
                .map(|(name, bindings)| (name.to_string(), bindings))
                .collect(),
            axes: axes
                .into_iter()
                .map(|(name, axis)| (name.to_string(), axis))
                .collect(),
        }
    }

    fn keys<const N: usize>(keys: [KeyCode; N]) -> Vec<BindingSpec> {
        keys.into_iter().map(BindingSpec::key).collect()
    }

    fn mouse_buttons<const N: usize>(buttons: [MouseButton; N]) -> Vec<BindingSpec> {
        buttons.into_iter().map(BindingSpec::mouse_button).collect()
    }

    fn axis<const P: usize, const N: usize>(
        positive: [KeyCode; P],
        negative: [KeyCode; N],
    ) -> AxisSpec {
        AxisSpec {
            positive: keys(positive),
            negative: keys(negative),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::KeyCode;

    #[test]
    fn assets_input_ron_parses() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/input.ron"
        ));
        let settings = InputSettings::from_ron_str(source).expect("assets/input.ron should parse");
        assert_eq!(settings.version, INPUT_SETTINGS_VERSION);
        assert!(!settings.game.system.bindings("escape").is_empty());
        assert!(!settings.game.flight.axis_positive("pitch").is_empty());
        assert!(
            !settings
                .game
                .eva
                .bindings("toggle_player_controller")
                .is_empty()
        );
        assert!(!settings.game.eva_move.axis_positive("forward").is_empty());
        assert!(!settings.planet_editor.bindings("camera_motion").is_empty());
        assert!(!settings.shipyard.bindings("primary").is_empty());
    }

    #[test]
    fn missing_bindings_merge_with_defaults() {
        let settings = InputSettings::from_ron_str("#![enable(implicit_some)]\n(version: 1, game: (flight: (bindings: {\"toggle_sas\": [Key(KeyF)]},),),)")
            .expect("settings should parse");
        assert_eq!(
            settings.game.flight.bindings.get("toggle_sas"),
            Some(&vec![BindingSpec::key(KeyCode::KeyF)])
        );
        assert_eq!(
            settings.game.warp.bindings.get("warp_increase"),
            Some(&vec![BindingSpec::key(KeyCode::Period)])
        );
    }

    #[test]
    fn invalid_action_source_reports_key_name() {
        // Unknown action names are caught by validation, not serde.
        let error = InputSettings::from_ron_str(
            "#![enable(implicit_some)]\n(version: 1, game: (system: (bindings: {\"escape\": [Key(Nope)]},),),)",
        )
        .expect_err("settings should reject unknown keys");
        let msg = error.to_string();
        // The RON parser catches Nope as an unknown KeyCode variant.
        assert!(msg.contains("Nope"), "error should mention Nope: {msg}");
    }

    #[test]
    fn invalid_action_name_reports_path() {
        let error = InputSettings::from_ron_str(
            "#![enable(implicit_some)]\n(version: 1, game: (system: (bindings: {\"bogus\": [Key(Escape)]},),),)",
        )
        .expect_err("settings should reject unknown actions")
        .to_string();
        assert!(error.contains("game.system.bindings.bogus"));
        assert!(error.contains("unknown action"));
    }

    #[test]
    fn axis_config_maps_positive_and_negative() {
        let settings = InputSettings::default();
        assert_eq!(
            settings
                .game
                .flight
                .axes
                .get("pitch")
                .map(|axis| &axis.positive),
            Some(&vec![BindingSpec::key(KeyCode::KeyW)])
        );
        assert_eq!(
            settings
                .game
                .flight
                .axes
                .get("pitch")
                .map(|axis| &axis.negative),
            Some(&vec![BindingSpec::key(KeyCode::KeyS)])
        );
    }
}
