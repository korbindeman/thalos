//! Frame-rate-independent input capture.
//!
//! Bevy's `ButtonInput` keeps every `just_pressed` and `just_released` flag, but
//! its `pressed` value is only the final state after all platform events for a
//! frame have been applied. `bevy_enhanced_input` samples that final value. A
//! complete press and release during one long frame therefore looks idle and
//! disappears at the action layer.
//!
//! This plugin records the raw keyboard, mouse-button, and gamepad-button
//! transitions, then exposes at most one transition per button per game frame.
//! Extra transitions remain queued for following frames. Continuous mouse
//! motion and scrolling already use Bevy's additive accumulators and do not
//! need replay.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use bevy::input::gamepad::{GamepadButtonStateChangedEvent, GamepadConnectionEvent};
use bevy::input::keyboard::{KeyboardFocusLost, KeyboardInput};
use bevy::input::mouse::MouseButtonInput;
use bevy::input::{ButtonInput, ButtonState, InputSystems};
use bevy::prelude::*;
use bevy_enhanced_input::prelude::EnhancedInputSystems;

/// Preserves every discrete input transition even when several arrive during
/// one rendered frame.
pub struct FrameIndependentInputPlugin;

impl Plugin for FrameIndependentInputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<KeyboardTransitions>()
            .init_resource::<MouseTransitions>()
            .init_resource::<GamepadTransitions>()
            .configure_sets(
                PreUpdate,
                (FrameInputSystems::Capture, FrameInputSystems::Replay)
                    .chain()
                    .after(InputSystems)
                    .before(EnhancedInputSystems::Prepare),
            )
            .add_systems(PreUpdate, capture_pre_update_state.before(InputSystems))
            .add_systems(
                PreUpdate,
                (
                    capture_keyboard_transitions,
                    capture_mouse_transitions,
                    capture_gamepad_transitions,
                )
                    .in_set(FrameInputSystems::Capture),
            )
            .add_systems(
                PreUpdate,
                (
                    replay_keyboard_transitions,
                    replay_mouse_transitions,
                    replay_gamepad_transitions,
                )
                    .in_set(FrameInputSystems::Replay),
            );
    }
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FrameInputSystems {
    Capture,
    Replay,
}

#[derive(Resource, Default)]
struct KeyboardTransitions(ButtonTransitionBuffer<KeyCode>);

#[derive(Resource, Default)]
struct MouseTransitions(ButtonTransitionBuffer<MouseButton>);

#[derive(Resource, Default)]
struct GamepadTransitions(ButtonTransitionBuffer<(Entity, GamepadButton)>);

struct ButtonTransitionBuffer<T> {
    tracked: HashSet<T>,
    logical_pressed: HashSet<T>,
    observed_pressed: HashSet<T>,
    pending: HashMap<T, VecDeque<bool>>,
}

impl<T> Default for ButtonTransitionBuffer<T> {
    fn default() -> Self {
        Self {
            tracked: HashSet::new(),
            logical_pressed: HashSet::new(),
            observed_pressed: HashSet::new(),
            pending: HashMap::new(),
        }
    }
}

impl<T> ButtonTransitionBuffer<T>
where
    T: Clone + Eq + Hash,
{
    fn initialize_pressed(&mut self, pressed: impl IntoIterator<Item = T>) {
        for input in pressed {
            if self.tracked.insert(input.clone()) {
                self.logical_pressed.insert(input.clone());
                self.observed_pressed.insert(input);
            }
        }
    }

    fn capture_snapshot(&mut self, pressed: impl IntoIterator<Item = T>) {
        let pressed: HashSet<_> = pressed.into_iter().collect();
        self.initialize_pressed(pressed.iter().cloned());

        let tracked: Vec<_> = self.tracked.iter().cloned().collect();
        for input in tracked {
            let is_pressed = pressed.contains(&input);
            if self.logical_pressed.contains(&input) != is_pressed {
                self.queue(
                    input,
                    if is_pressed {
                        ButtonState::Pressed
                    } else {
                        ButtonState::Released
                    },
                );
            }
        }
    }

    fn queue(&mut self, input: T, state: ButtonState) {
        let pressed = state.is_pressed();
        if self.tracked.insert(input.clone()) {
            // A state-change event tells us the previous state even when this
            // input was first seen after startup.
            if !pressed {
                self.logical_pressed.insert(input.clone());
                self.observed_pressed.insert(input.clone());
            }
        }

        if self.observed_pressed.contains(&input) == pressed {
            return;
        }

        if pressed {
            self.observed_pressed.insert(input.clone());
        } else {
            self.observed_pressed.remove(&input);
        }
        self.pending.entry(input).or_default().push_back(pressed);
    }

    fn queue_release_all(&mut self) {
        let pressed: Vec<_> = self.observed_pressed.iter().cloned().collect();
        for input in pressed {
            self.queue(input, ButtonState::Released);
        }
    }

    fn advance(&mut self) -> Vec<(T, bool)> {
        let inputs: Vec<_> = self.pending.keys().cloned().collect();
        let mut transitions = Vec::with_capacity(inputs.len());
        for input in inputs {
            let Some(queue) = self.pending.get_mut(&input) else {
                continue;
            };
            let Some(pressed) = queue.pop_front() else {
                continue;
            };
            if queue.is_empty() {
                self.pending.remove(&input);
            }
            if pressed {
                self.logical_pressed.insert(input.clone());
            } else {
                self.logical_pressed.remove(&input);
            }
            transitions.push((input, pressed));
        }
        transitions
    }

    fn forget_where(&mut self, mut predicate: impl FnMut(&T) -> bool) {
        self.tracked.retain(|input| !predicate(input));
        self.logical_pressed.retain(|input| !predicate(input));
        self.observed_pressed.retain(|input| !predicate(input));
        self.pending.retain(|input, _| !predicate(input));
    }
}

fn capture_pre_update_state(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    gamepads: Query<(Entity, &Gamepad)>,
    mut keyboard: ResMut<KeyboardTransitions>,
    mut mouse_buttons: ResMut<MouseTransitions>,
    mut gamepad_buttons: ResMut<GamepadTransitions>,
) {
    keyboard.0.capture_snapshot(keys.get_pressed().copied());
    mouse_buttons
        .0
        .capture_snapshot(mouse.get_pressed().copied());
    for (entity, gamepad) in &gamepads {
        gamepad_buttons.0.capture_snapshot(
            gamepad
                .get_pressed()
                .copied()
                .map(|button| (entity, button)),
        );
    }
}

fn capture_keyboard_transitions(
    mut events: MessageReader<KeyboardInput>,
    mut focus_lost: MessageReader<KeyboardFocusLost>,
    mut transitions: ResMut<KeyboardTransitions>,
) {
    for event in events.read() {
        if !event.repeat {
            transitions.0.queue(event.key_code, event.state);
        }
    }
    if focus_lost.read().next().is_some() {
        transitions.0.queue_release_all();
    }
}

fn capture_mouse_transitions(
    mut events: MessageReader<MouseButtonInput>,
    mut transitions: ResMut<MouseTransitions>,
) {
    for event in events.read() {
        transitions.0.queue(event.button, event.state);
    }
}

fn capture_gamepad_transitions(
    mut events: MessageReader<GamepadButtonStateChangedEvent>,
    mut connections: MessageReader<GamepadConnectionEvent>,
    mut transitions: ResMut<GamepadTransitions>,
) {
    for event in events.read() {
        transitions
            .0
            .queue((event.entity, event.button), event.state);
    }
    for event in connections.read() {
        if event.disconnected() {
            transitions
                .0
                .forget_where(|(entity, _)| *entity == event.gamepad);
        }
    }
}

fn replay_keyboard_transitions(
    mut input: ResMut<ButtonInput<KeyCode>>,
    mut transitions: ResMut<KeyboardTransitions>,
) {
    replay_button_input(&mut input, &mut transitions.0);
}

fn replay_mouse_transitions(
    mut input: ResMut<ButtonInput<MouseButton>>,
    mut transitions: ResMut<MouseTransitions>,
) {
    replay_button_input(&mut input, &mut transitions.0);
}

fn replay_button_input<T>(input: &mut ButtonInput<T>, transitions: &mut ButtonTransitionBuffer<T>)
where
    T: Clone + Eq + Hash + Send + Sync + 'static,
{
    let physically_pressed: Vec<_> = input.get_pressed().cloned().collect();
    for button in physically_pressed {
        if !transitions.logical_pressed.contains(&button) {
            input.release(button);
        }
    }
    for button in &transitions.logical_pressed {
        if !input.pressed(button.clone()) {
            input.press(button.clone());
        }
    }

    input.clear();
    for (button, pressed) in transitions.advance() {
        if pressed {
            input.press(button);
        } else {
            input.release(button);
        }
    }
}

fn replay_gamepad_transitions(
    mut gamepads: Query<(Entity, &mut Gamepad)>,
    mut transitions: ResMut<GamepadTransitions>,
) {
    for (entity, mut gamepad) in &mut gamepads {
        let input = gamepad.digital_mut();
        let physically_pressed: Vec<_> = input.get_pressed().copied().collect();
        for button in physically_pressed {
            if !transitions.0.logical_pressed.contains(&(entity, button)) {
                input.release(button);
            }
        }
        for (_, button) in transitions
            .0
            .logical_pressed
            .iter()
            .filter(|(gamepad, _)| *gamepad == entity)
        {
            if !input.pressed(*button) {
                input.press(*button);
            }
        }
        input.clear();
    }

    for ((entity, button), pressed) in transitions.0.advance() {
        let Ok((_, mut gamepad)) = gamepads.get_mut(entity) else {
            continue;
        };
        if pressed {
            gamepad.digital_mut().press(button);
        } else {
            gamepad.digital_mut().release(button);
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::input::{
        InputPlugin,
        gamepad::{RawGamepadButtonChangedEvent, RawGamepadEvent},
    };

    use super::*;

    #[test]
    fn transition_buffer_replays_every_edge_in_order() {
        let mut buffer = ButtonTransitionBuffer::<MouseButton>::default();
        buffer.queue(MouseButton::Left, ButtonState::Pressed);
        buffer.queue(MouseButton::Left, ButtonState::Released);
        buffer.queue(MouseButton::Left, ButtonState::Pressed);
        buffer.queue(MouseButton::Left, ButtonState::Released);

        assert_eq!(buffer.advance(), vec![(MouseButton::Left, true)]);
        assert_eq!(buffer.advance(), vec![(MouseButton::Left, false)]);
        assert_eq!(buffer.advance(), vec![(MouseButton::Left, true)]);
        assert_eq!(buffer.advance(), vec![(MouseButton::Left, false)]);
        assert!(buffer.advance().is_empty());
    }

    #[test]
    fn repeated_pressed_events_do_not_create_phantom_edges() {
        let mut buffer = ButtonTransitionBuffer::<KeyCode>::default();
        buffer.queue(KeyCode::KeyW, ButtonState::Pressed);
        buffer.queue(KeyCode::KeyW, ButtonState::Pressed);

        assert_eq!(buffer.advance(), vec![(KeyCode::KeyW, true)]);
        assert!(buffer.advance().is_empty());
    }

    #[test]
    fn gamepad_tap_between_updates_preserves_both_edges() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, InputPlugin, FrameIndependentInputPlugin));
        app.finish();
        app.cleanup();
        app.update();

        let gamepad = app.world_mut().spawn(Gamepad::default()).id();
        app.world_mut()
            .write_message(RawGamepadEvent::Button(RawGamepadButtonChangedEvent::new(
                gamepad,
                GamepadButton::South,
                1.0,
            )));
        app.world_mut()
            .write_message(RawGamepadEvent::Button(RawGamepadButtonChangedEvent::new(
                gamepad,
                GamepadButton::South,
                0.0,
            )));

        app.update();
        let input = app.world().entity(gamepad).get::<Gamepad>().unwrap();
        assert!(input.pressed(GamepadButton::South));
        assert!(input.just_pressed(GamepadButton::South));

        app.update();
        let input = app.world().entity(gamepad).get::<Gamepad>().unwrap();
        assert!(!input.pressed(GamepadButton::South));
        assert!(input.just_released(GamepadButton::South));
    }
}
