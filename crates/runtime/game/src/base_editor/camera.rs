//! God-view camera for the base editor.
//!
//! The orbit state and the per-frame driver live in the shared
//! [`crate::god_view`] module (the space-center hub uses the same camera); this
//! module only gathers the editor's raw input, resolves its focus via
//! [`super::compute_focus`], and hands both to [`god_view::drive_god_view`]. The
//! flight camera systems (`SimStage::Camera`) are gated off while the editor is
//! open, so they don't fight this; they resume — snapping back to the ship — the
//! moment the editor closes.
//!
//! Controls (raw input, since gameplay contexts are suppressed): right-drag
//! orbits, scroll zooms, WASD pans across the ground.

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_physics_local::HeightSourceRegistry;

use crate::camera::ShipCamera;
use crate::god_view::{self, GodViewFocus, GodViewInput, GodViewOrbit};
use crate::rendering::{SimulationState, SolarSystemState};
use crate::structures::StructureRegistry;

use super::{BaseEditor, BaseEditorMode, base_editor_open, compute_focus};

pub(super) struct BaseEditorCameraPlugin;

impl Plugin for BaseEditorCameraPlugin {
    fn build(&self, app: &mut App) {
        // `GodViewOrbit` lives in `god_view::GodViewPlugin`.
        app.add_systems(
            Update,
            (
                reset_orbit_on_open,
                drive_base_editor_camera.in_set(god_view::GodViewCameraSet),
            )
                .chain()
                .run_if(base_editor_open),
        );
    }
}

/// Fresh establishing view whenever the editor opens (or switches focus mode).
///
/// When the focus is an existing base — placing buildings or picking a launch
/// point — it frames the whole base ([`GodViewOrbit::reset_over_base`]), the one
/// default god-view per base shared with the space-center hub; when picking a
/// *new* site on open terrain there is no base to frame, so it opens at the close
/// default. Keyed on `is_changed()` (not just the open edge) so the
/// PickSite → PlaceBuildings transition re-frames onto the freshly-flattened base
/// — safe because `BaseEditor` only mutates on genuine open/mode/site changes,
/// never per-frame.
fn reset_orbit_on_open(editor: Res<BaseEditor>, mut orbit: ResMut<GodViewOrbit>) {
    if editor.is_changed() && editor.open {
        match editor.mode {
            BaseEditorMode::PlaceBuildings | BaseEditorMode::SelectLaunch => {
                orbit.reset_over_base()
            }
            BaseEditorMode::PickSite => orbit.reset(),
        }
    }
}

/// Place the ship camera at the god-view around the editor's focus point.
/// Ungated by `SimStage` so it runs while the editor pauses the sim.
#[allow(clippy::too_many_arguments)]
fn drive_base_editor_camera(
    editor: Res<BaseEditor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    registry: Res<StructureRegistry>,
    ui_gate: Res<crate::hud::UiPointerGate>,
    ui_keyboard: Res<crate::hud::UiKeyboardGate>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut orbit: ResMut<GodViewOrbit>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut camera: Query<(&mut Transform, &mut CellCoord), With<ShipCamera>>,
) {
    // Always drain the input streams so they don't pile up across frames.
    let drag: Vec2 = motion.read().map(|m| m.delta).sum();
    let scroll: f32 = wheel.read().map(|w| w.y).sum();

    let Some(focus) = compute_focus(&editor, &sim, &solar, &height_sources, &registry) else {
        return;
    };
    let Ok(root_grid) = root_grid.single() else {
        return;
    };
    let Ok((mut transform, mut cell)) = camera.single_mut() else {
        return;
    };

    god_view::drive_god_view(
        GodViewFocus {
            center_world: focus.center_world,
            up_world: focus.up_world,
        },
        &mut orbit,
        &keys,
        GodViewInput {
            over_ui: ui_gate.hovered,
            text_entry: ui_keyboard.text_entry(),
            orbit_held: mouse_buttons.pressed(MouseButton::Right),
            drag,
            scroll,
            dt: time.delta_secs(),
        },
        root_grid,
        &mut transform,
        &mut cell,
    );
}
