use bevy::prelude::*;
use thalos_input::enhanced::{ActionSources, ContextActivity, EnhancedInputSystems};
use thalos_input::game::{
    GameEvaContext, GameEvaMoveContext, GameFlightContext, GameManeuverContext,
    GameManeuverPrecisionContext, GameViewContext, GameWarpContext,
};
use thalos_input::shipyard::ShipyardContext;

pub struct GameInputGatePlugin;

impl Plugin for GameInputGatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PreUpdate,
            gate_enhanced_input_sources.before(EnhancedInputSystems::Update),
        );
    }
}

fn gate_enhanced_input_sources(
    mut commands: Commands,
    mut action_sources: ResMut<ActionSources>,
    app_state: Res<State<crate::loading::AppState>>,
    ui_pointer_gate: Option<Res<crate::hud::UiPointerGate>>,
    freecam: Option<Res<crate::freecam::FreeCam>>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    shipyard_editor: Option<Res<crate::shipyard_editor::ShipyardEditor>>,
    base_editor: Option<Res<crate::base_editor::BaseEditor>>,
    space_center: Option<Res<crate::space_center::SpaceCenter>>,
    viewpoint_manager: Option<Res<thalos_viewer::ViewpointUiState>>,
    settings_menu: Option<Res<crate::settings_menu::SettingsMenu>>,
    ui_keyboard: Option<Res<crate::hud::UiKeyboardGate>>,
    // Bundled to stay within Bevy's 16-param system limit.
    context_queries: (
        Query<(Entity, &ContextActivity<GameFlightContext>)>,
        Query<(Entity, &ContextActivity<GameWarpContext>)>,
        Query<(Entity, &ContextActivity<GameViewContext>)>,
        Query<(Entity, &ContextActivity<GameEvaContext>)>,
        Query<(Entity, &ContextActivity<GameEvaMoveContext>)>,
        Query<(Entity, &ContextActivity<GameManeuverContext>)>,
        Query<(Entity, &ContextActivity<GameManeuverPrecisionContext>)>,
        Query<(Entity, &ContextActivity<ShipyardContext>)>,
    ),
) {
    let (flight, warp, view, eva, eva_move, maneuver, precision, shipyard_ctx) = context_queries;
    let bevy_ui_pointer_busy = ui_pointer_gate
        .as_deref()
        .map(|gate| gate.hovered)
        .unwrap_or(false);
    let freecam_active = freecam.as_deref().map(|f| f.active).unwrap_or(false);
    let editor_open = shipyard_editor.as_deref().map(|e| e.open).unwrap_or(false);
    // A text-entry surface (the shipyard name, settings HOTAS inputs, the F9
    // prompt, the egui F8 manager's name/id/notes fields) swallows the
    // keyboard so raw keys edit the field instead of tripping flight/system
    // bindings. One gate covers both UI systems — see `hud::input_gate`.
    let editor_text_focused = ui_keyboard
        .as_deref()
        .is_some_and(crate::hud::UiKeyboardGate::text_entry);
    // The start screen owns the frame like the shipyard editor does: every
    // gameplay context deactivates (the system context stays active for
    // Escape / screenshot / viewpoint manager). Folded into `editor_open` since the suppression
    // set is identical.
    let editor_open = editor_open || *app_state.get() == crate::loading::AppState::MainMenu;
    let base_editor_open = base_editor.as_deref().map(|e| e.open).unwrap_or(false);
    let space_center_open = space_center.as_deref().map(|s| s.open).unwrap_or(false);
    let viewpoint_manager_open = viewpoint_manager
        .as_deref()
        .map(thalos_viewer::ViewpointUiState::is_open)
        .unwrap_or(false);
    let settings_open = settings_menu.as_deref().is_some_and(|menu| menu.open);
    // The shipyard/base editors, the space-center hub, and the start screen all
    // deactivate every gameplay context; only the shipyard editor / start screen
    // own the `ShipyardContext`, so the two are tracked separately.
    let gameplay_suppressed = editor_open
        || base_editor_open
        || space_center_open
        || viewpoint_manager_open
        || settings_open;

    thalos_input::gating::set_mouse_sources(&mut action_sources, !bevy_ui_pointer_busy);
    // GameSystemContext stays active for Escape/screenshot/perspective save. Text entry only
    // disables gameplay contexts — except the shipyard editor's own text
    // field, which reads raw key events and must swallow everything
    // (including Escape) while focused.
    thalos_input::gating::set_keyboard_source(&mut action_sources, !editor_text_focused);

    let player_controller_active = player
        .as_deref()
        .map(|state| state.is_active())
        .unwrap_or(false);

    // Freecam suspends flight input so WASD/QE drive the camera; the EVA
    // controller likewise owns WASD while active. The shipyard editor owns
    // everything except the system context while open.
    set_context_activity(
        &mut commands,
        &flight,
        !freecam_active && !player_controller_active && !gameplay_suppressed,
    );
    // Warp controls (pause, speed up/down, warp-to-maneuver) are sim-time
    // meta-controls — they must remain available in every mode, including
    // EVA and freecam. Only the shipyard editor (which force-pauses the sim)
    // suppresses them. (Text-input focus is handled by the keyboard-source
    // gate above, not by deactivating contexts.)
    set_context_activity(&mut commands, &warp, !gameplay_suppressed);
    set_context_activity(&mut commands, &view, !gameplay_suppressed);
    set_context_activity(&mut commands, &eva, !gameplay_suppressed);
    set_context_activity(
        &mut commands,
        &eva_move,
        !freecam_active && player_controller_active && !gameplay_suppressed,
    );
    set_context_activity(&mut commands, &maneuver, !gameplay_suppressed);
    if gameplay_suppressed {
        set_context_activity(&mut commands, &precision, false);
    }
    // The shipyard input context (orbit drag, placement clicks, precision
    // wheel) only runs while the editor is open, so its mouse passthrough
    // and Shift modifier never collide with flight controls.
    set_context_activity(&mut commands, &shipyard_ctx, editor_open);
}

fn set_context_activity<C: Component>(
    commands: &mut Commands,
    query: &Query<(Entity, &ContextActivity<C>)>,
    active: bool,
) {
    for (entity, current) in query {
        if **current != active {
            commands
                .entity(entity)
                .insert(thalos_input::gating::context_activity::<C>(active));
        }
    }
}
