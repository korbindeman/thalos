use bevy::prelude::*;
use bevy_egui::EguiContexts;
use thalos_input::enhanced::{ActionSources, ContextActivity, EnhancedInputSystems};
use thalos_input::game::{
    GameFlightContext, GameManeuverContext, GameManeuverPrecisionContext, GameViewContext,
};

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
    flight: Query<(Entity, &ContextActivity<GameFlightContext>)>,
    view: Query<(Entity, &ContextActivity<GameViewContext>)>,
    maneuver: Query<(Entity, &ContextActivity<GameManeuverContext>)>,
    precision: Query<(Entity, &ContextActivity<GameManeuverPrecisionContext>)>,
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

    thalos_input::gating::set_mouse_sources(
        &mut action_sources,
        !(egui_pointer_busy || bevy_ui_pointer_busy),
    );
    // GameSystemContext stays active for Escape/screenshot. Text entry only
    // disables gameplay contexts.
    thalos_input::gating::set_keyboard_source(&mut action_sources, true);

    // Freecam suspends flight input so WASD/QE drive the camera, not the ship.
    set_context_activity(&mut commands, &flight, !egui_keyboard_busy && !freecam_active);
    set_context_activity(&mut commands, &view, !egui_keyboard_busy);
    set_context_activity(&mut commands, &maneuver, !egui_keyboard_busy);
    if egui_keyboard_busy {
        set_context_activity(&mut commands, &precision, false);
    }
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
