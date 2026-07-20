use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy_enhanced_input::prelude::*;

use crate::settings::InputSettings;

#[derive(Component)]
pub struct ShipyardContext;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PrimaryAction;

/// Build-camera orbit button (right mouse). Kept separate from
/// [`PrimaryAction`] (left mouse) so placing / selecting parts and orbiting the
/// camera never contend for the same button.
#[derive(InputAction)]
#[action_output(bool)]
pub struct OrbitAction;

#[derive(InputAction)]
#[action_output(Vec2)]
pub struct CameraMotionAction;

#[derive(InputAction)]
#[action_output(Vec2)]
pub struct CameraWheelAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PrecisionSlowAction;

#[derive(Resource, Debug, Default, Clone)]
pub struct ShipyardInputIntent {
    pub primary_pressed: bool,
    pub primary_started: bool,
    pub primary_released: bool,
    /// Held: the camera-orbit (right mouse) button is down.
    pub orbit_pressed: bool,
    pub camera_motion: Vec2,
    pub camera_wheel: Vec2,
    pub precision_slow: bool,
}

pub struct ShipyardInputPlugin;

impl Plugin for ShipyardInputPlugin {
    fn build(&self, app: &mut App) {
        // The game adds this plugin (for its in-game shipyard editor) alongside
        // `GameInputPlugin`, which already registers `EnhancedInputPlugin`. Guard
        // so it works whether or not `EnhancedInputPlugin` is already present.
        if !app.is_plugin_added::<EnhancedInputPlugin>() {
            app.add_plugins(EnhancedInputPlugin);
        }
        app.add_input_context::<ShipyardContext>()
            .init_resource::<ShipyardInputIntent>()
            .add_systems(Startup, spawn_shipyard_input)
            .add_systems(
                PreUpdate,
                collect_shipyard_intent.after(EnhancedInputSystems::Apply),
            );
    }
}

fn spawn_shipyard_input(mut commands: Commands, settings: Res<InputSettings>) {
    let section = &settings.shipyard;
    commands.spawn((
        ShipyardContext,
        // Explicit activity so a host can gate the whole context off (the
        // game keeps it inactive unless its shipyard editor is open; the
        // standalone editor leaves it always-on).
        ContextActivity::<ShipyardContext>::ACTIVE,
        Name::new("ShipyardInputController"),
        actions!(ShipyardContext[
            (
                Action::<PrimaryAction>::new(),
                ActionSettings {
                    consume_input: false,
                    ..default()
                },
                Bindings::spawn(section.bindings("primary")),
            ),
            (
                Action::<OrbitAction>::new(),
                ActionSettings {
                    consume_input: false,
                    ..default()
                },
                Bindings::spawn(section.bindings("orbit")),
            ),
            (
                Action::<CameraMotionAction>::new(),
                ActionSettings {
                    consume_input: false,
                    ..default()
                },
                Bindings::spawn(section.bindings("camera_motion")),
            ),
            (
                Action::<CameraWheelAction>::new(),
                ActionSettings {
                    consume_input: false,
                    ..default()
                },
                Bindings::spawn(section.bindings("camera_wheel")),
            ),
            (
                Action::<PrecisionSlowAction>::new(),
                consume_input(),
                Bindings::spawn(section.bindings("precision_slow")),
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

fn collect_shipyard_intent(
    mut intent: ResMut<ShipyardInputIntent>,
    primary: Query<(&Action<PrimaryAction>, &ActionEvents)>,
    orbit: Query<&Action<OrbitAction>>,
    motion: Query<&Action<CameraMotionAction>>,
    wheel: Query<&Action<CameraWheelAction>>,
    scroll: Res<AccumulatedMouseScroll>,
    precision: Query<&Action<PrecisionSlowAction>>,
) {
    *intent = ShipyardInputIntent {
        primary_pressed: primary
            .single()
            .map(|(action, _)| **action)
            .unwrap_or(false),
        primary_started: primary
            .single()
            .map(|(_, events)| events.contains(ActionEvents::START))
            .unwrap_or(false),
        primary_released: primary
            .single()
            .map(|(_, events)| events.contains(ActionEvents::COMPLETE))
            .unwrap_or(false),
        orbit_pressed: orbit.single().map(|action| **action).unwrap_or(false),
        camera_motion: motion.single().map(|action| **action).unwrap_or(Vec2::ZERO),
        camera_wheel: crate::camera_scroll_delta(
            wheel.single().map(|action| **action).unwrap_or(Vec2::ZERO),
            scroll.unit,
        ),
        precision_slow: precision.single().map(|action| **action).unwrap_or(false),
    };
}
