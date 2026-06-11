//! In-game shipyard editor — the native Bevy-UI front-end over
//! `thalos_shipyard::editor` (`ShipEditorCorePlugin`).
//!
//! The editor is a **modal pause mode**, following the pause-menu /
//! scenario-menu pattern: [`ShipyardEditor::open`] is a sim-clock pause
//! source (see `crate::sim_clock`), not an `AppState`. While open:
//!
//! - the scene cameras (ship + map) deactivate and a dedicated
//!   [`scene::EditorCamera`] on [`crate::coords::EDITOR_LAYER`] renders the
//!   build world — editor entities never bleed into flight or map views,
//!   and the flight world stays exactly where it was;
//! - all gameplay input contexts deactivate and the `ShipyardContext`
//!   (orbit drag, placement clicks) activates — see
//!   `crate::input::gate_enhanced_input_sources`;
//! - the HUD hides (photo-mode pattern) and the editor's Bevy-UI panels
//!   (`ui` module, styled from [`crate::hud::theme::HudTheme`]) show.
//!
//! Entry points: the pause menu's SHIPYARD button, or launching with
//! `just game shipyard`. Escape closes (owned by
//! `pause_menu::handle_escape_input`'s priority chain).
//!
//! The build world is partitioned from the flight ship by the
//! `EditorPart` marker — see `thalos_shipyard::editor` docs. Build state
//! persists across open/close (entities are hidden, not despawned), so a
//! design in progress survives flying around in between.

pub mod scene;
pub mod ui;

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy_egui::EguiContexts;

use thalos_input::shipyard::ShipyardInputPlugin;
use thalos_shipyard::editor::{EditorPart, EditorUiGate, PreviewGhost, ShipEditorCorePlugin};

use crate::camera::{ActiveCamera, MapCamera, ShipCamera};
use crate::coords::EDITOR_LAYER;
use crate::hud::HudPanel;
use crate::loading::AppState;
use crate::view::ViewMode;

pub use ui::EditorTextFocus;

/// Whether the in-game shipyard editor is open. A sim-clock pause source.
///
/// **Sole writer of `open`:** the pause menu's SHIPYARD button, Escape via
/// `pause_menu::handle_escape_input`, and `just game shipyard` startup.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct ShipyardEditor {
    pub open: bool,
}

/// Open the editor the instant the game finishes loading (set by `main.rs`
/// for `just game shipyard`). The editor must **not** open during
/// `AppState::Loading`: while it is open the three `SimStage` sets are gated
/// off (it's a separate scene), and the terrain/body-state systems that
/// *complete* loading live in those sets — so opening early would deadlock
/// the loading screen. Deferring the open to `OnEnter(Running)` keeps the
/// world load running with the editor closed.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct OpenShipyardOnStart(pub bool);

/// Run condition: the shipyard editor is closed.
pub fn editor_closed(editor: Option<Res<ShipyardEditor>>) -> bool {
    editor.map(|e| !e.open).unwrap_or(true)
}

/// Run condition: the shipyard editor is open.
pub fn editor_open(editor: Option<Res<ShipyardEditor>>) -> bool {
    editor.map(|e| e.open).unwrap_or(false)
}

pub struct ShipyardEditorPlugin;

impl Plugin for ShipyardEditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ShipyardInputPlugin)
            .add_plugins(ShipEditorCorePlugin)
            .add_plugins(scene::EditorScenePlugin)
            .add_plugins(ui::EditorUiPlugin)
            .init_resource::<ShipyardEditor>()
            .init_resource::<OpenShipyardOnStart>()
            .add_systems(
                PreUpdate,
                sync_editor_ui_gate.before(bevy::picking::PickingSystems::Hover),
            )
            // Deferred open: only once the world has loaded (see
            // `OpenShipyardOnStart`).
            .add_systems(OnEnter(AppState::Running), open_on_start)
            .add_systems(Update, apply_open_state)
            .add_systems(PostUpdate, propagate_editor_render_layers);
    }
}

/// Open the editor on entry to `AppState::Running` when launched with
/// `just game shipyard` — after the world has loaded, never during it.
fn open_on_start(flag: Res<OpenShipyardOnStart>, mut editor: ResMut<ShipyardEditor>) {
    if flag.0 {
        editor.open = true;
    }
}

/// React to open/close: flip the cameras, show/hide the build world, and
/// hide/restore the flight HUD (photo-mode pattern). Sim pause is handled
/// by `sim_clock::sync_sim_clock` reading [`ShipyardEditor`] directly.
fn apply_open_state(
    editor: Res<ShipyardEditor>,
    mut view: ResMut<ViewMode>,
    mut commands: Commands,
    mut cameras: ParamSet<(
        Query<(Entity, &mut Camera), Or<(With<MapCamera>, With<ShipCamera>)>>,
        Query<(Entity, &mut Camera), With<scene::EditorCamera>>,
    )>,
    mut visibilities: ParamSet<(
        Query<&mut Visibility, Or<(With<EditorPart>, With<PreviewGhost>, With<scene::EditorSceneEntity>)>>,
        Query<&mut Visibility, With<HudPanel>>,
    )>,
) {
    if !editor.is_changed() {
        return;
    }
    let open = editor.open;

    if open {
        // The editor owns the screen: both scene cameras off (markers
        // stripped so the close path can cleanly reassert them), hangar
        // camera on and carrying `IsDefaultUiCamera` — bevy_ui renders to
        // the default UI camera, and an inactive default means no UI at
        // all. `view::apply_active_camera` stands down while we're open.
        for (entity, mut camera) in cameras.p0().iter_mut() {
            if camera.is_active {
                camera.is_active = false;
            }
            commands
                .entity(entity)
                .remove::<(ActiveCamera, IsDefaultUiCamera)>();
        }
        for (entity, mut camera) in cameras.p1().iter_mut() {
            camera.is_active = true;
            commands.entity(entity).insert(IsDefaultUiCamera);
        }
    } else {
        for (entity, mut camera) in cameras.p1().iter_mut() {
            camera.is_active = false;
            commands.entity(entity).remove::<IsDefaultUiCamera>();
        }
        // Hand the screen back: poke ViewMode's change tick so
        // `view::apply_active_camera` (the owner of scene-camera activity
        // and the ActiveCamera/IsDefaultUiCamera markers) reasserts the
        // current view's camera.
        view.set_changed();
    }

    let build_target = if open {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in visibilities.p0().iter_mut() {
        if *vis != build_target {
            *vis = build_target;
        }
    }

    let hud_target = if open {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut vis in visibilities.p1().iter_mut() {
        if *vis != hud_target {
            *vis = hud_target;
        }
    }
}

/// Mirror the Bevy-UI hover gate (plus egui, for the debug overlays) into
/// the editor core's [`EditorUiGate`], so its picking observers and the
/// placement preview stand down while the cursor is over panels.
fn sync_editor_ui_gate(
    ui_pointer: Res<crate::hud::UiPointerGate>,
    mut contexts: EguiContexts,
    mut gate: ResMut<EditorUiGate>,
) {
    let egui_busy = contexts
        .ctx_mut()
        .map(|ctx| ctx.is_pointer_over_area() || ctx.wants_pointer_input())
        .unwrap_or(false);
    let busy = ui_pointer.hovered || egui_busy;
    if gate.pointer_busy != busy {
        gate.pointer_busy = busy;
    }
}

/// Keep every editor-owned entity (parts, their rebuilt mesh children, the
/// preview ghost, the hangar scene) on [`EDITOR_LAYER`], so only the editor
/// camera draws them. Mesh children are respawned freely by the core's
/// rebuild systems, so this reasserts layers down each tree every frame —
/// the same pattern as `view::propagate_view_render_layers`.
fn propagate_editor_render_layers(
    mut commands: Commands,
    roots: Query<
        Entity,
        Or<(
            With<EditorPart>,
            With<PreviewGhost>,
            With<scene::EditorSceneEntity>,
        )>,
    >,
    children_q: Query<&Children>,
    layers_q: Query<&RenderLayers>,
) {
    let target = RenderLayers::layer(EDITOR_LAYER);
    for root in &roots {
        let mut stack: Vec<Entity> = vec![root];
        while let Some(e) = stack.pop() {
            let needs = layers_q.get(e).map(|rl| rl != &target).unwrap_or(true);
            if needs {
                commands.entity(e).insert(target.clone());
            }
            if let Ok(c) = children_q.get(e) {
                stack.extend(c.iter());
            }
        }
    }
}
