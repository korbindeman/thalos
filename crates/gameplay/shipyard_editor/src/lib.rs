// Bevy system signatures routinely exceed clippy's argument and type
// complexity budgets; same crate-level allowance as thalos_runtime.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! In-game shipyard editor — the native Bevy-UI front-end over the
//! [`core`] editor logic (`ShipEditorCorePlugin`).
//!
//! The editor is a **modal pause mode**, following the pause-menu /
//! scenario-menu pattern: [`ShipyardEditor::open`] is a sim-clock pause
//! source (see the runtime's `sim_clock`), not an `AppState`. While open:
//!
//! - the scene cameras (ship + map) deactivate and a dedicated
//!   [`scene::EditorCamera`] on [`thalos_game_state::coords::EDITOR_LAYER`] renders the
//!   build world — editor entities never bleed into flight or map views,
//!   and the flight world stays exactly where it was;
//! - all gameplay input contexts deactivate and the `ShipyardContext`
//!   (orbit drag, placement clicks) activates — see
//!   the runtime's `gate_enhanced_input_sources`;
//! - the HUD hides (photo-mode pattern) and the editor's Bevy-UI panels
//!   (`ui` module, styled from [`thalos_ui::HudTheme`]) show.
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

use thalos_game_state::coords::EDITOR_LAYER;
use thalos_game_state::ui::HideInPhotoMode;
use thalos_game_state::ui::HudPanel;

/// Whether the in-game shipyard editor is open. A sim-clock pause source.
///
/// `open` is a **derived mirror** of [`GameContext::Vab`](thalos_game_state::context::GameContext)
/// (Phase 3): its sole writer is `game_context::mirror_context_to_booleans`. The
/// VAB is entered by setting `NextState<GameContext>` — the pause menu's SHIPYARD
/// button, the hub's VAB facility, `just game shipyard` via
/// [`InitialContext`](thalos_game_state::context::InitialContext), and Escape / Exit
/// back-out via [`ContextHistory`](thalos_game_state::context::ContextHistory).
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct ShipyardEditor {
    pub open: bool,
}

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
            .add_systems(
                PreUpdate,
                sync_editor_ui_gate.before(bevy::picking::PickingSystems::Hover),
            )
            .add_systems(Update, apply_open_state)
            .add_systems(PostUpdate, propagate_editor_render_layers);
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
/// [`GameContext::Vab`](thalos_game_state::context::GameContext) (F6 UI-flow
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
    ui_pointer: Res<thalos_game_state::ui::UiPointerGate>,
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
