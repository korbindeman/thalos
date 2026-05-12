use bevy::math::Vec2;
use bevy::prelude::*;
use bevy_enhanced_input::prelude::*;

use crate::settings::InputSettings;

#[derive(Component)]
pub struct ShipyardContext;

#[derive(InputAction)]
#[action_output(bool)]
pub struct PrimaryAction;

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
    pub camera_motion: Vec2,
    pub camera_wheel: Vec2,
    pub precision_slow: bool,
}

pub struct ShipyardInputPlugin;

impl Plugin for ShipyardInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EnhancedInputPlugin)
            .add_input_context::<ShipyardContext>()
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
    motion: Query<&Action<CameraMotionAction>>,
    wheel: Query<&Action<CameraWheelAction>>,
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
        camera_motion: motion.single().map(|action| **action).unwrap_or(Vec2::ZERO),
        camera_wheel: wheel.single().map(|action| **action).unwrap_or(Vec2::ZERO),
        precision_slow: precision.single().map(|action| **action).unwrap_or(false),
    };
}
