use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy_enhanced_input::prelude::*;

use crate::settings::{HotasAxisBinding, HotasDeviceSelector, InputSettings};

#[derive(Component)]
pub struct GameInputController;

#[derive(Component)]
pub struct GameSystemContext;

#[derive(Component)]
pub struct GameFlightContext;

/// Sim-time controls (pause, warp speed, warp-to-maneuver). Split from
/// `GameFlightContext` because these must remain available in every mode —
/// EVA, freecam, photo — not only when the player is flying a ship. Gated
/// only on egui text-input focus so typing in a text field doesn't trip
/// pause.
#[derive(Component)]
pub struct GameWarpContext;

#[derive(Component)]
pub struct GameViewContext;

#[derive(Component)]
pub struct GameCameraContext;

#[derive(Component)]
pub struct GameEvaContext;

#[derive(Component)]
pub struct GameEvaMoveContext;

#[derive(Component)]
pub struct GameManeuverContext;

#[derive(Component)]
pub struct GameManeuverPrecisionContext;

#[derive(InputAction)]
#[action_output(bool)]
pub struct EscapeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct ScreenshotAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct ToggleFreeCamAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct ToggleSasAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct WarpToManeuverAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct WarpIncreaseAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct WarpDecreaseAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct WarpResetAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct ThrottleFullAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct ThrottleCutAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct StageAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PitchPositiveAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PitchNegativeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct YawPositiveAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct YawNegativeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct RollPositiveAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct RollNegativeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct ThrottleRampPositiveAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct ThrottleRampNegativeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct ToggleViewAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct TogglePhotoModeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct CycleShipCameraAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct CameraPrimaryAction;

#[derive(InputAction)]
#[action_output(Vec2)]
pub struct CameraMotionAction;

#[derive(InputAction)]
#[action_output(Vec2)]
pub struct CameraWheelAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct TogglePlayerControllerAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PlayerForwardPositiveAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PlayerForwardNegativeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PlayerStrafePositiveAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PlayerStrafeNegativeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PlayerJumpAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PlayerSprintAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct TogglePlaceNodeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct DeleteNodeAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PrecisionFineAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PrecisionUltraAction;

#[derive(Resource, Debug, Default, Clone)]
pub struct GameInputIntent {
    pub escape: bool,
    pub screenshot: bool,
    pub toggle_free_cam: bool,
    pub toggle_sas: bool,
    pub warp_to_maneuver: bool,
    pub warp_increase: bool,
    pub warp_decrease: bool,
    pub warp_reset: bool,
    pub throttle_full: bool,
    pub throttle_cut: bool,
    /// Edge-triggered: the player advanced to the next stage this frame.
    pub stage: bool,
    pub throttle_up: bool,
    pub throttle_down: bool,
    /// Absolute HOTAS throttle command in `[0, 1]`. `None` leaves the
    /// frame to the discrete keyboard ramp/full/cut controls.
    pub throttle_absolute: Option<f32>,
    pub attitude: Vec3,
    pub toggle_view: bool,
    pub toggle_photo_mode: bool,
    pub cycle_ship_camera: bool,
    pub primary_pressed: bool,
    pub primary_started: bool,
    pub primary_released: bool,
    pub camera_motion: Vec2,
    pub camera_wheel: Vec2,
    pub toggle_player_controller: bool,
    pub player_move: Vec2,
    /// Edge-triggered: the player pressed jump this frame (on foot).
    pub player_jump: bool,
    /// Held: the player is sprinting (on foot).
    pub player_sprint: bool,
    pub toggle_place_node: bool,
    pub delete_node: bool,
    pub precision_fine: bool,
    pub precision_ultra: bool,
}

pub struct GameInputPlugin;

impl Plugin for GameInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EnhancedInputPlugin)
            .add_input_context::<GameSystemContext>()
            .add_input_context::<GameFlightContext>()
            .add_input_context::<GameWarpContext>()
            .add_input_context::<GameViewContext>()
            .add_input_context::<GameCameraContext>()
            .add_input_context::<GameEvaContext>()
            .add_input_context::<GameEvaMoveContext>()
            .add_input_context::<GameManeuverContext>()
            .add_input_context::<GameManeuverPrecisionContext>()
            .init_resource::<GameInputIntent>()
            .add_systems(Startup, spawn_game_input_controller)
            .add_systems(
                PreUpdate,
                reset_game_intent.before(EnhancedInputSystems::Apply),
            )
            .add_systems(
                PreUpdate,
                (
                    collect_system_intent,
                    collect_flight_toggle_intent,
                    collect_throttle_command_intent,
                    collect_attitude_intent,
                    collect_throttle_axis_intent,
                    collect_view_intent,
                    collect_camera_intent,
                    collect_player_controller_intent,
                    collect_maneuver_intent,
                    collect_precision_intent,
                    collect_hotas_intent,
                )
                    .chain()
                    .after(EnhancedInputSystems::Apply),
            );
    }
}

fn spawn_game_input_controller(mut commands: Commands, settings: Res<InputSettings>) {
    let settings = &*settings;
    let mut controller = commands.spawn((GameInputController, Name::new("GameInputController")));
    controller.insert((
        GameSystemContext,
        ContextPriority::<GameSystemContext>::new(100),
        actions!(GameSystemContext[
            (
                Action::<EscapeAction>::new(),
                ActionSettings {
                    consume_input: false,
                    ..default()
                },
                Bindings::spawn(settings.game.system.bindings("escape")),
            ),
            (
                Action::<ScreenshotAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.system.bindings("screenshot")),
            ),
            (
                Action::<ToggleFreeCamAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.system.bindings("toggle_free_cam")),
            ),
        ]),
    ));
    controller.insert((
        GameWarpContext,
        ContextPriority::<GameWarpContext>::new(50),
        actions!(GameWarpContext[
            (
                Action::<WarpToManeuverAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.warp.bindings("warp_to_maneuver")),
            ),
            (
                Action::<WarpIncreaseAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.warp.bindings("warp_increase")),
            ),
            (
                Action::<WarpDecreaseAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.warp.bindings("warp_decrease")),
            ),
            (
                Action::<WarpResetAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.warp.bindings("warp_reset")),
            ),
        ]),
    ));
    controller.insert((
        GameFlightContext,
        ContextPriority::<GameFlightContext>::new(20),
        actions!(GameFlightContext[
            (
                Action::<ToggleSasAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.bindings("toggle_sas")),
            ),
            (
                Action::<ThrottleFullAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.bindings("throttle_full")),
            ),
            (
                Action::<ThrottleCutAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.bindings("throttle_cut")),
            ),
            (
                Action::<StageAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.bindings("stage")),
            ),
            (
                Action::<PitchPositiveAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.axis_positive("pitch")),
            ),
            (
                Action::<PitchNegativeAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.axis_negative("pitch")),
            ),
            (
                Action::<YawPositiveAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.axis_positive("yaw")),
            ),
            (
                Action::<YawNegativeAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.axis_negative("yaw")),
            ),
            (
                Action::<RollPositiveAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.axis_positive("roll")),
            ),
            (
                Action::<RollNegativeAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.axis_negative("roll")),
            ),
            (
                Action::<ThrottleRampPositiveAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.axis_positive("throttle_ramp")),
            ),
            (
                Action::<ThrottleRampNegativeAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.flight.axis_negative("throttle_ramp")),
            ),
        ]),
    ));
    controller.insert((
        GameViewContext,
        ContextPriority::<GameViewContext>::new(30),
        actions!(GameViewContext[
            (
                Action::<ToggleViewAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.view.bindings("toggle_view")),
            ),
            (
                Action::<TogglePhotoModeAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.view.bindings("toggle_photo_mode")),
            ),
            (
                Action::<CycleShipCameraAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.view.bindings("cycle_ship_camera")),
            ),
        ]),
    ));
    controller.insert((
        GameCameraContext,
        ContextPriority::<GameCameraContext>::new(10),
        actions!(GameCameraContext[
            (
                Action::<CameraPrimaryAction>::new(),
                ActionSettings {
                    consume_input: false,
                    ..default()
                },
                Bindings::spawn(settings.game.camera.bindings("primary")),
            ),
            (
                Action::<CameraMotionAction>::new(),
                ActionSettings {
                    consume_input: false,
                    ..default()
                },
                Bindings::spawn(settings.game.camera.bindings("motion")),
            ),
            (
                Action::<CameraWheelAction>::new(),
                ActionSettings {
                    consume_input: false,
                    ..default()
                },
                Bindings::spawn(settings.game.camera.bindings("wheel")),
            ),
        ]),
    ));
    controller.insert((
        GameEvaContext,
        ContextPriority::<GameEvaContext>::new(35),
        actions!(GameEvaContext[(
            Action::<TogglePlayerControllerAction>::new(),
            consume_input(),
            Bindings::spawn(settings.game.eva.bindings("toggle_player_controller")),
        ),]),
    ));
    controller.insert((
        GameEvaMoveContext,
        ContextPriority::<GameEvaMoveContext>::new(95),
        ContextActivity::<GameEvaMoveContext>::INACTIVE,
        actions!(GameEvaMoveContext[
            (
                Action::<PlayerForwardPositiveAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.eva_move.axis_positive("forward")),
            ),
            (
                Action::<PlayerForwardNegativeAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.eva_move.axis_negative("forward")),
            ),
            (
                Action::<PlayerStrafePositiveAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.eva_move.axis_positive("strafe")),
            ),
            (
                Action::<PlayerStrafeNegativeAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.eva_move.axis_negative("strafe")),
            ),
            (
                Action::<PlayerJumpAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.eva_move.bindings("jump")),
            ),
            (
                Action::<PlayerSprintAction>::new(),
                ActionSettings {
                    consume_input: false,
                    ..default()
                },
                Bindings::spawn(settings.game.eva_move.bindings("sprint")),
            ),
        ]),
    ));
    controller.insert((
        GameManeuverContext,
        ContextPriority::<GameManeuverContext>::new(40),
        actions!(GameManeuverContext[
            (
                Action::<TogglePlaceNodeAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.maneuver.bindings("toggle_place_node")),
            ),
            (
                Action::<DeleteNodeAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.maneuver.bindings("delete_node")),
            ),
        ]),
    ));
    controller.insert((
        GameManeuverPrecisionContext,
        ContextPriority::<GameManeuverPrecisionContext>::new(90),
        ContextActivity::<GameManeuverPrecisionContext>::INACTIVE,
        actions!(GameManeuverPrecisionContext[
            (
                Action::<PrecisionFineAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.maneuver_precision.bindings("fine")),
            ),
            (
                Action::<PrecisionUltraAction>::new(),
                consume_input(),
                Bindings::spawn(settings.game.maneuver_precision.bindings("ultra")),
            ),
        ]),
    ));
}

fn consume_input() -> ActionSettings {
    ActionSettings {
        consume_input: true,
        ..default()
    }
}

fn reset_game_intent(mut intent: ResMut<GameInputIntent>) {
    *intent = GameInputIntent::default();
}

fn collect_system_intent(
    mut intent: ResMut<GameInputIntent>,
    escape: Query<(&Action<EscapeAction>, &ActionEvents)>,
    screenshot: Query<(&Action<ScreenshotAction>, &ActionEvents)>,
    toggle_free_cam: Query<(&Action<ToggleFreeCamAction>, &ActionEvents)>,
) {
    intent.escape = started(&escape);
    intent.screenshot = started(&screenshot);
    intent.toggle_free_cam = started(&toggle_free_cam);
}

fn collect_flight_toggle_intent(
    mut intent: ResMut<GameInputIntent>,
    toggle_sas: Query<(&Action<ToggleSasAction>, &ActionEvents)>,
    warp_to_maneuver: Query<(&Action<WarpToManeuverAction>, &ActionEvents)>,
    warp_increase: Query<(&Action<WarpIncreaseAction>, &ActionEvents)>,
    warp_decrease: Query<(&Action<WarpDecreaseAction>, &ActionEvents)>,
    warp_reset: Query<(&Action<WarpResetAction>, &ActionEvents)>,
) {
    intent.toggle_sas = started(&toggle_sas);
    intent.warp_to_maneuver = started(&warp_to_maneuver);
    intent.warp_increase = started(&warp_increase);
    intent.warp_decrease = started(&warp_decrease);
    intent.warp_reset = started(&warp_reset);
}

fn collect_throttle_command_intent(
    mut intent: ResMut<GameInputIntent>,
    throttle_full: Query<(&Action<ThrottleFullAction>, &ActionEvents)>,
    throttle_cut: Query<(&Action<ThrottleCutAction>, &ActionEvents)>,
    stage: Query<(&Action<StageAction>, &ActionEvents)>,
) {
    intent.throttle_full = started(&throttle_full);
    intent.throttle_cut = started(&throttle_cut);
    intent.stage = started(&stage);
}

fn collect_attitude_intent(
    mut intent: ResMut<GameInputIntent>,
    pitch_pos: Query<&Action<PitchPositiveAction>>,
    pitch_neg: Query<&Action<PitchNegativeAction>>,
    yaw_pos: Query<&Action<YawPositiveAction>>,
    yaw_neg: Query<&Action<YawNegativeAction>>,
    roll_pos: Query<&Action<RollPositiveAction>>,
    roll_neg: Query<&Action<RollNegativeAction>>,
) {
    intent.attitude = Vec3::new(
        axis_value(&pitch_pos, &pitch_neg),
        axis_value(&roll_pos, &roll_neg),
        axis_value(&yaw_pos, &yaw_neg),
    );
}

fn collect_throttle_axis_intent(
    mut intent: ResMut<GameInputIntent>,
    keys: Res<ButtonInput<KeyCode>>,
    throttle_up: Query<&Action<ThrottleRampPositiveAction>>,
    throttle_down: Query<&Action<ThrottleRampNegativeAction>>,
) {
    // Keep Shift/Ctrl available as throttle controls during normal flight, but
    // do not let OS command chords like cmd+shift-click or cmd+shift+2 leak
    // into gameplay as throttle input.
    if command_modifier_held(&keys) {
        intent.throttle_up = false;
        intent.throttle_down = false;
        return;
    }
    intent.throttle_up = held(&throttle_up);
    intent.throttle_down = held(&throttle_down);
}

fn collect_view_intent(
    mut intent: ResMut<GameInputIntent>,
    toggle_view: Query<(&Action<ToggleViewAction>, &ActionEvents)>,
    toggle_photo_mode: Query<(&Action<TogglePhotoModeAction>, &ActionEvents)>,
    cycle_ship_camera: Query<(&Action<CycleShipCameraAction>, &ActionEvents)>,
) {
    intent.toggle_view = started(&toggle_view);
    intent.toggle_photo_mode = started(&toggle_photo_mode);
    intent.cycle_ship_camera = started(&cycle_ship_camera);
}

fn collect_camera_intent(
    mut intent: ResMut<GameInputIntent>,
    primary: Query<(&Action<CameraPrimaryAction>, &ActionEvents)>,
    motion: Query<&Action<CameraMotionAction>>,
    wheel: Query<&Action<CameraWheelAction>>,
    scroll: Res<AccumulatedMouseScroll>,
) {
    intent.primary_pressed = held_with_events(&primary);
    intent.primary_started = started(&primary);
    intent.primary_released = completed(&primary);
    intent.camera_motion = vec2(&motion);
    intent.camera_wheel = crate::camera_scroll_delta(vec2(&wheel), scroll.unit);
}

#[allow(clippy::too_many_arguments)]
fn collect_player_controller_intent(
    mut intent: ResMut<GameInputIntent>,
    toggle: Query<(&Action<TogglePlayerControllerAction>, &ActionEvents)>,
    forward_pos: Query<&Action<PlayerForwardPositiveAction>>,
    forward_neg: Query<&Action<PlayerForwardNegativeAction>>,
    strafe_pos: Query<&Action<PlayerStrafePositiveAction>>,
    strafe_neg: Query<&Action<PlayerStrafeNegativeAction>>,
    jump: Query<(&Action<PlayerJumpAction>, &ActionEvents)>,
    sprint: Query<&Action<PlayerSprintAction>>,
) {
    intent.toggle_player_controller = started(&toggle);
    intent.player_move = Vec2::new(
        axis_value(&strafe_pos, &strafe_neg),
        axis_value(&forward_pos, &forward_neg),
    )
    .clamp_length_max(1.0);
    intent.player_jump = started(&jump);
    intent.player_sprint = held(&sprint);
}

fn collect_maneuver_intent(
    mut intent: ResMut<GameInputIntent>,
    toggle_place_node: Query<(&Action<TogglePlaceNodeAction>, &ActionEvents)>,
    delete_node: Query<(&Action<DeleteNodeAction>, &ActionEvents)>,
) {
    intent.toggle_place_node = started(&toggle_place_node);
    intent.delete_node = started(&delete_node);
}

fn collect_precision_intent(
    mut intent: ResMut<GameInputIntent>,
    fine: Query<&Action<PrecisionFineAction>>,
    ultra: Query<&Action<PrecisionUltraAction>>,
) {
    intent.precision_fine = held(&fine);
    intent.precision_ultra = held(&ultra);
}

fn collect_hotas_intent(
    mut intent: ResMut<GameInputIntent>,
    settings: Res<InputSettings>,
    flight: Query<&ContextActivity<GameFlightContext>>,
    gamepads: Query<(&Gamepad, Option<&Name>)>,
) {
    let hotas = &settings.game.hotas;
    if !hotas.enabled {
        return;
    }
    let flight_active = flight.single().map(|activity| **activity).unwrap_or(false);
    if !flight_active {
        return;
    }

    if let Some(value) = hotas_axis_value(hotas.axis("pitch"), &hotas.device, &gamepads) {
        intent.attitude.x = merge_hotas_axis(intent.attitude.x, value);
    }
    if let Some(value) = hotas_axis_value(hotas.axis("roll"), &hotas.device, &gamepads) {
        intent.attitude.y = merge_hotas_axis(intent.attitude.y, value);
    }
    if let Some(value) = hotas_axis_value(hotas.axis("yaw"), &hotas.device, &gamepads) {
        intent.attitude.z = merge_hotas_axis(intent.attitude.z, value);
    }
    if let Some(binding) = hotas.axis("throttle")
        && let Some(raw) = hotas_axis_raw(binding, &hotas.device, &gamepads)
    {
        intent.throttle_absolute = Some(hotas_throttle_value(raw, binding));
    }
}

fn started<A: InputAction<Output = bool>>(query: &Query<(&Action<A>, &ActionEvents)>) -> bool {
    query
        .single()
        .map(|(_, events)| events.contains(ActionEvents::START))
        .unwrap_or(false)
}

fn completed<A: InputAction<Output = bool>>(query: &Query<(&Action<A>, &ActionEvents)>) -> bool {
    query
        .single()
        .map(|(_, events)| events.contains(ActionEvents::COMPLETE))
        .unwrap_or(false)
}

fn held_with_events<A: InputAction<Output = bool>>(
    query: &Query<(&Action<A>, &ActionEvents)>,
) -> bool {
    query.single().map(|(action, _)| **action).unwrap_or(false)
}

fn held<A: InputAction<Output = bool>>(query: &Query<&Action<A>>) -> bool {
    query.single().map(|action| **action).unwrap_or(false)
}

fn command_modifier_held(keys: &ButtonInput<KeyCode>) -> bool {
    keys.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight])
}

fn axis_value<P, N>(positive: &Query<&Action<P>>, negative: &Query<&Action<N>>) -> f32
where
    P: InputAction<Output = bool>,
    N: InputAction<Output = bool>,
{
    held(positive) as i8 as f32 - held(negative) as i8 as f32
}

fn vec2<A: InputAction<Output = Vec2>>(query: &Query<&Action<A>>) -> Vec2 {
    query.single().map(|action| **action).unwrap_or(Vec2::ZERO)
}

fn hotas_axis_value(
    binding: Option<&HotasAxisBinding>,
    default_device: &HotasDeviceSelector,
    gamepads: &Query<(&Gamepad, Option<&Name>)>,
) -> Option<f32> {
    let binding = binding?;
    hotas_axis_raw(binding, default_device, gamepads).map(|raw| hotas_signed_value(raw, binding))
}

fn hotas_axis_raw(
    binding: &HotasAxisBinding,
    default_device: &HotasDeviceSelector,
    gamepads: &Query<(&Gamepad, Option<&Name>)>,
) -> Option<f32> {
    let selector = binding.device.as_ref().unwrap_or(default_device);
    gamepads
        .iter()
        .find(|(gamepad, name)| hotas_device_matches(selector, gamepad, *name))
        .and_then(|(gamepad, _)| gamepad.get(binding.axis))
}

fn hotas_device_matches(
    selector: &HotasDeviceSelector,
    gamepad: &Gamepad,
    name: Option<&Name>,
) -> bool {
    match selector {
        HotasDeviceSelector::Any => true,
        HotasDeviceSelector::NameContains(needle) => name
            .map(|name| {
                name.as_str()
                    .to_ascii_lowercase()
                    .contains(&needle.to_ascii_lowercase())
            })
            .unwrap_or(false),
        HotasDeviceSelector::Usb {
            vendor_id,
            product_id,
        } => {
            gamepad.vendor_id() == Some(*vendor_id)
                && product_id.map_or(true, |id| gamepad.product_id() == Some(id))
        }
    }
}

fn hotas_unit_value(raw: f32, binding: &HotasAxisBinding) -> f32 {
    ((raw - binding.min) / (binding.max - binding.min)).clamp(0.0, 1.0)
}

fn hotas_signed_value(raw: f32, binding: &HotasAxisBinding) -> f32 {
    let mut value = hotas_unit_value(raw, binding) * 2.0 - 1.0;
    if binding.invert {
        value = -value;
    }
    let magnitude = value.abs();
    if magnitude <= binding.deadzone {
        0.0
    } else {
        value.signum() * ((magnitude - binding.deadzone) / (1.0 - binding.deadzone)).min(1.0)
    }
}

fn hotas_throttle_value(raw: f32, binding: &HotasAxisBinding) -> f32 {
    let value = hotas_unit_value(raw, binding);
    if binding.invert { 1.0 - value } else { value }
}

fn merge_hotas_axis(current: f32, hotas: f32) -> f32 {
    if hotas.abs() > 1.0e-4 { hotas } else { current }
}

#[cfg(test)]
mod tests {
    use bevy::input::InputPlugin;
    use bevy::prelude::*;

    use super::*;
    use crate::settings::{HotasAxisBinding, InputSettings};

    fn input_app() -> App {
        input_app_with_settings(InputSettings::default())
    }

    fn input_app_with_settings(settings: InputSettings) -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, InputPlugin))
            .insert_resource(settings)
            .add_plugins(GameInputPlugin);
        app.finish();
        app.cleanup();
        app.update();
        app
    }

    fn press_key(app: &mut App, key: KeyCode) {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(key);
    }

    fn release_key(app: &mut App, key: KeyCode) {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .release(key);
    }

    fn set_precision_context(app: &mut App, active: bool) {
        let entity = app
            .world_mut()
            .query_filtered::<Entity, With<GameManeuverPrecisionContext>>()
            .single(app.world())
            .expect("precision context should exist");
        app.world_mut()
            .entity_mut(entity)
            .insert(ContextActivity::<GameManeuverPrecisionContext>::new(active));
    }

    fn set_flight_context(app: &mut App, active: bool) {
        let entity = app
            .world_mut()
            .query_filtered::<Entity, With<GameFlightContext>>()
            .single(app.world())
            .expect("flight context should exist");
        app.world_mut()
            .entity_mut(entity)
            .insert(ContextActivity::<GameFlightContext>::new(active));
    }

    fn hotas_binding(axis: GamepadAxis) -> HotasAxisBinding {
        HotasAxisBinding {
            axis,
            device: None,
            invert: false,
            deadzone: 0.0,
            min: -1.0,
            max: 1.0,
        }
    }

    fn hotas_settings() -> InputSettings {
        let mut settings = InputSettings::default();
        settings.game.hotas.enabled = true;
        settings.game.hotas.axes.insert(
            "pitch".to_string(),
            HotasAxisBinding {
                invert: true,
                ..hotas_binding(GamepadAxis::LeftStickY)
            },
        );
        settings
            .game
            .hotas
            .axes
            .insert("throttle".to_string(), hotas_binding(GamepadAxis::LeftZ));
        settings
    }

    fn spawn_gamepad_with_axes(app: &mut App, axes: impl IntoIterator<Item = (GamepadAxis, f32)>) {
        let mut gamepad = Gamepad::default();
        for (axis, value) in axes {
            gamepad.analog_mut().set(axis, value);
        }
        app.world_mut().spawn((Name::new("Test HOTAS"), gamepad));
    }

    #[test]
    fn shift_precision_suppresses_throttle_only_while_active() {
        let mut app = input_app();

        press_key(&mut app, KeyCode::ShiftLeft);
        app.update();
        {
            let intent = app.world().resource::<GameInputIntent>();
            assert!(intent.throttle_up);
            assert!(!intent.precision_fine);
        }

        release_key(&mut app, KeyCode::ShiftLeft);
        app.update();
        set_precision_context(&mut app, true);

        press_key(&mut app, KeyCode::ShiftLeft);
        app.update();
        {
            let intent = app.world().resource::<GameInputIntent>();
            assert!(!intent.throttle_up);
            assert!(intent.precision_fine);
        }

        release_key(&mut app, KeyCode::ShiftLeft);
        app.update();
        set_precision_context(&mut app, false);

        press_key(&mut app, KeyCode::ShiftLeft);
        app.update();
        let intent = app.world().resource::<GameInputIntent>();
        assert!(intent.throttle_up);
        assert!(!intent.precision_fine);
    }

    #[test]
    fn command_shift_does_not_emit_throttle_ramp() {
        let mut app = input_app();

        press_key(&mut app, KeyCode::SuperLeft);
        press_key(&mut app, KeyCode::ShiftLeft);
        app.update();
        {
            let intent = app.world().resource::<GameInputIntent>();
            assert!(!intent.throttle_up);
            assert!(!intent.throttle_down);
        }

        release_key(&mut app, KeyCode::SuperLeft);
        app.update();
        let intent = app.world().resource::<GameInputIntent>();
        assert!(intent.throttle_up);
    }

    #[test]
    fn eva_w_walks_forward_when_eva_move_context_active() {
        let mut app = input_app();
        // EVA move context starts INACTIVE; the game's gating system would
        // activate it when the player controller is alive. Simulate that.
        let entity = app
            .world_mut()
            .query_filtered::<Entity, With<GameEvaMoveContext>>()
            .single(app.world())
            .expect("eva move context should exist");
        app.world_mut()
            .entity_mut(entity)
            .insert(ContextActivity::<GameEvaMoveContext>::new(true));
        app.update();

        press_key(&mut app, KeyCode::KeyW);
        app.update();
        let intent = app.world().resource::<GameInputIntent>();
        assert_eq!(
            intent.player_move,
            Vec2::new(0.0, 1.0),
            "KeyW with active EVA move context should produce forward axis",
        );
    }

    #[test]
    fn hotas_axes_feed_attitude_and_absolute_throttle() {
        let mut app = input_app_with_settings(hotas_settings());
        spawn_gamepad_with_axes(
            &mut app,
            [(GamepadAxis::LeftStickY, -0.5), (GamepadAxis::LeftZ, 0.25)],
        );

        app.update();

        let intent = app.world().resource::<GameInputIntent>();
        assert_eq!(intent.attitude.x, 0.5);
        assert_eq!(intent.throttle_absolute, Some(0.625));
    }

    #[test]
    fn centered_hotas_axis_does_not_clear_keyboard_attitude() {
        let mut app = input_app_with_settings(hotas_settings());
        spawn_gamepad_with_axes(&mut app, [(GamepadAxis::LeftStickY, 0.0)]);

        press_key(&mut app, KeyCode::KeyW);
        app.update();

        let intent = app.world().resource::<GameInputIntent>();
        assert_eq!(intent.attitude.x, 1.0);
    }

    #[test]
    fn hotas_axes_follow_flight_context_activity() {
        let mut app = input_app_with_settings(hotas_settings());
        spawn_gamepad_with_axes(
            &mut app,
            [(GamepadAxis::LeftStickY, -0.5), (GamepadAxis::LeftZ, 1.0)],
        );
        set_flight_context(&mut app, false);

        app.update();

        let intent = app.world().resource::<GameInputIntent>();
        assert_eq!(intent.attitude, Vec3::ZERO);
        assert_eq!(intent.throttle_absolute, None);
    }
}
