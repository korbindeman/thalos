use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy_enhanced_input::prelude::*;

use crate::settings::InputSettings;

#[derive(Component)]
pub struct BodyEditorContext;

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
pub struct BodyEditorInputIntent {
    pub primary_pressed: bool,
    pub primary_started: bool,
    pub camera_motion: Vec2,
    pub camera_wheel: Vec2,
    pub toggle_fullbright: bool,
    pub overlay_suppress: bool,
}

pub struct BodyEditorInputPlugin;

impl Plugin for BodyEditorInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EnhancedInputPlugin)
            .add_input_context::<BodyEditorContext>()
            .init_resource::<BodyEditorInputIntent>()
            .add_systems(Startup, spawn_body_editor_input)
            .add_systems(
                PreUpdate,
                collect_body_editor_intent.after(EnhancedInputSystems::Apply),
            );
    }
}

fn spawn_body_editor_input(mut commands: Commands, settings: Res<InputSettings>) {
    let section = &settings.body_editor;
    commands.spawn((
        BodyEditorContext,
        Name::new("BodyEditorInputController"),
        actions!(BodyEditorContext[
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

fn collect_body_editor_intent(
    mut intent: ResMut<BodyEditorInputIntent>,
    primary: Query<(&Action<PrimaryAction>, &ActionEvents)>,
    motion: Query<&Action<CameraMotionAction>>,
    wheel: Query<&Action<CameraWheelAction>>,
    scroll: Res<AccumulatedMouseScroll>,
    fullbright: Query<(&Action<ToggleFullbrightAction>, &ActionEvents)>,
    suppress: Query<&Action<OverlaySuppressAction>>,
) {
    *intent = BodyEditorInputIntent {
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
