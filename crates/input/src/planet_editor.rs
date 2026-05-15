use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy_enhanced_input::prelude::*;

use crate::settings::InputSettings;

#[derive(Component)]
pub struct PlanetEditorContext;

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
pub struct ToggleFullbrightAction;

#[derive(InputAction)]
#[action_output(bool)]
pub struct OverlaySuppressAction;

#[derive(Resource, Debug, Default, Clone)]
pub struct PlanetEditorInputIntent {
    pub primary_pressed: bool,
    pub primary_started: bool,
    pub camera_motion: Vec2,
    pub camera_wheel: Vec2,
    pub toggle_fullbright: bool,
    pub overlay_suppress: bool,
}

pub struct PlanetEditorInputPlugin;

impl Plugin for PlanetEditorInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EnhancedInputPlugin)
            .add_input_context::<PlanetEditorContext>()
            .init_resource::<PlanetEditorInputIntent>()
            .add_systems(Startup, spawn_planet_editor_input)
            .add_systems(
                PreUpdate,
                collect_planet_editor_intent.after(EnhancedInputSystems::Apply),
            );
    }
}

fn spawn_planet_editor_input(mut commands: Commands, settings: Res<InputSettings>) {
    let section = &settings.planet_editor;
    commands.spawn((
        PlanetEditorContext,
        Name::new("PlanetEditorInputController"),
        actions!(PlanetEditorContext[
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
                Action::<ToggleFullbrightAction>::new(),
                consume_input(),
                Bindings::spawn(section.bindings("toggle_fullbright")),
            ),
            (
                Action::<OverlaySuppressAction>::new(),
                consume_input(),
                Bindings::spawn(section.bindings("overlay_suppress")),
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

fn collect_planet_editor_intent(
    mut intent: ResMut<PlanetEditorInputIntent>,
    primary: Query<(&Action<PrimaryAction>, &ActionEvents)>,
    motion: Query<&Action<CameraMotionAction>>,
    wheel: Query<&Action<CameraWheelAction>>,
    scroll: Res<AccumulatedMouseScroll>,
    fullbright: Query<(&Action<ToggleFullbrightAction>, &ActionEvents)>,
    suppress: Query<&Action<OverlaySuppressAction>>,
) {
    *intent = PlanetEditorInputIntent {
        primary_pressed: primary
            .single()
            .map(|(action, _)| **action)
            .unwrap_or(false),
        primary_started: primary
            .single()
            .map(|(_, events)| events.contains(ActionEvents::START))
            .unwrap_or(false),
        camera_motion: motion.single().map(|action| **action).unwrap_or(Vec2::ZERO),
        camera_wheel: crate::camera_scroll_delta(
            wheel.single().map(|action| **action).unwrap_or(Vec2::ZERO),
            scroll.unit,
        ),
        toggle_fullbright: fullbright
            .single()
            .map(|(_, events)| events.contains(ActionEvents::START))
            .unwrap_or(false),
        overlay_suppress: suppress.single().map(|action| **action).unwrap_or(false),
    };
}
