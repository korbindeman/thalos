//! In-game shipyard editor — the native Bevy-UI front-end over the
//! [`core`] editor logic (`ShipEditorCorePlugin`).
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
//! `EditorPart` marker — see the [`core`] docs. Build state persists across
//! open/close (entities are hidden, not despawned), so a design in progress
//! survives flying around in between.

pub mod core;
pub mod scene;
pub mod ui;

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;

use thalos_input::shipyard::ShipyardInputPlugin;

use self::core::{EditorPart, EditorUiGate, PreviewGhost, ShipEditorCorePlugin};

use crate::coords::EDITOR_LAYER;
use crate::hud::HudPanel;
use crate::loading::AppState;
use crate::photo_mode::HideInPhotoMode;

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
/// `just game shipyard` (or via the start screen's SHIPYARD button) — after
/// the world has loaded, never during it.
fn open_on_start(mut flag: ResMut<OpenShipyardOnStart>, mut editor: ResMut<ShipyardEditor>) {
    if flag.0 {
        editor.open = true;
        // One-shot: `OnEnter(Running)` fires again on every later loading
        // pass (start-screen runway starts), which must not re-open the
        // editor. The start screen's SHIPYARD button re-arms this.
        flag.0 = false;
    }
}

/// React to open/close: show/hide the build world and hide/restore the flight
/// overlays — both `HudPanel`s and the photo-mode (`HideInPhotoMode`) set, since
/// the editor camera becomes the default UI camera and would otherwise draw any
/// flight UI left visible (the navball leaked through when only `HudPanel`s were
/// hidden).
///
/// Camera activation is **not** owned here anymore: the single authority
/// `view::apply_active_camera` selects the editor camera from
/// [`GameContext::Vab`](crate::game_context::GameContext) (F6 UI-flow
/// unification). Sim pause is handled by `sim_clock::sync_sim_clock` reading the
/// same `GameContext`.
fn apply_open_state(
    editor: Res<ShipyardEditor>,
    mut visibilities: ParamSet<(
        Query<
            &mut Visibility,
            Or<(
                With<EditorPart>,
                With<PreviewGhost>,
                With<scene::EditorSceneEntity>,
            )>,
        >,
        Query<&mut Visibility, With<HudPanel>>,
        // Flight overlays that opt out of the clean scene via the photo-mode
        // marker rather than `HudPanel` — the navball above all. These are
        // `bevy_ui` nodes / world overlays that would otherwise render into the
        // editor (the editor camera becomes the default UI camera), so the
        // editor must hide the *same* set photo mode does, not just `HudPanel`.
        Query<&mut Visibility, With<HideInPhotoMode>>,
    )>,
) {
    if !editor.is_changed() {
        return;
    }
    let open = editor.open;

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
    for mut vis in visibilities.p2().iter_mut() {
        if *vis != hud_target {
            *vis = hud_target;
        }
    }
}

/// Mirror the Bevy-UI hover gate into the editor core's [`EditorUiGate`], so its
/// picking observers and the placement preview stand down while the cursor is
/// over panels.
fn sync_editor_ui_gate(
    ui_pointer: Res<crate::hud::UiPointerGate>,
    mut gate: ResMut<EditorUiGate>,
) {
    let busy = ui_pointer.hovered;
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
