use bevy::prelude::*;
use bevy_egui::EguiContexts;
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
    mut contexts: EguiContexts,
    ui_pointer_gate: Option<Res<crate::hud::UiPointerGate>>,
    freecam: Option<Res<crate::freecam::FreeCam>>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    shipyard_editor: Option<Res<crate::shipyard_editor::ShipyardEditor>>,
    editor_text: Option<Res<crate::shipyard_editor::EditorTextFocus>>,
    flight: Query<(Entity, &ContextActivity<GameFlightContext>)>,
    warp: Query<(Entity, &ContextActivity<GameWarpContext>)>,
    view: Query<(Entity, &ContextActivity<GameViewContext>)>,
    eva: Query<(Entity, &ContextActivity<GameEvaContext>)>,
    eva_move: Query<(Entity, &ContextActivity<GameEvaMoveContext>)>,
    maneuver: Query<(Entity, &ContextActivity<GameManeuverContext>)>,
    precision: Query<(Entity, &ContextActivity<GameManeuverPrecisionContext>)>,
    shipyard_ctx: Query<(Entity, &ContextActivity<ShipyardContext>)>,
) {
    let (egui_pointer_busy, egui_keyboard_busy) = contexts
        .ctx_mut()
        .map(|ctx| (ctx.wants_pointer_input(), ctx.wants_keyboard_input()))
        .unwrap_or((false, false));
    let bevy_ui_pointer_busy = ui_pointer_gate
        .as_deref()
        .map(|gate| gate.hovered)
        .unwrap_or(false);
    let freecam_active = freecam.as_deref().map(|f| f.active).unwrap_or(false);
    let editor_open = shipyard_editor.as_deref().map(|e| e.open).unwrap_or(false);
    let editor_text_focused = editor_text.as_deref().map(|t| t.is_focused()).unwrap_or(false);

    thalos_input::gating::set_mouse_sources(
        &mut action_sources,
        !(egui_pointer_busy || bevy_ui_pointer_busy),
    );
    // GameSystemContext stays active for Escape/screenshot. Text entry only
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
        !egui_keyboard_busy && !freecam_active && !player_controller_active && !editor_open,
    );
    // Warp controls (pause, speed up/down, warp-to-maneuver) are sim-time
    // meta-controls — they must remain available in every mode, including
    // EVA and freecam. Only text-input focus and the shipyard editor
    // (which force-pauses the sim) suppress them.
    set_context_activity(&mut commands, &warp, !egui_keyboard_busy && !editor_open);
    set_context_activity(&mut commands, &view, !egui_keyboard_busy && !editor_open);
    set_context_activity(&mut commands, &eva, !egui_keyboard_busy && !editor_open);
    set_context_activity(
        &mut commands,
        &eva_move,
        !egui_keyboard_busy && !freecam_active && player_controller_active && !editor_open,
    );
    set_context_activity(&mut commands, &maneuver, !egui_keyboard_busy && !editor_open);
    if egui_keyboard_busy || editor_open {
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
