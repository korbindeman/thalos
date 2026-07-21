//! Minimal single-line text field.
//!
//! One focused field at a time ([`TextFieldFocus`]); raw key events feed the
//! focused field's `value`. Hosts gate their own keyboard bindings on
//! [`TextFieldFocus::is_focused`] so typing never trips game actions.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;

use crate::UiTheme;
use crate::tokens::*;

/// A single-line editable text field. Consumers read `value` (or react to
/// `Changed<UiTextField>`) and may write it to sync from the model.
#[derive(Component, Debug, Clone)]
pub struct UiTextField {
    pub value: String,
    pub placeholder: String,
    pub max_len: usize,
}

impl UiTextField {
    pub fn new(value: impl Into<String>, placeholder: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            placeholder: placeholder.into(),
            max_len: 48,
        }
    }
}

/// Marker on the text child inside a field.
#[derive(Component)]
pub struct TextFieldText;

/// Which field currently owns the keyboard.
///
/// **Sole writer:** [`focus_text_fields`] (and Enter/Escape in
/// [`apply_text_field_input`]).
#[derive(Resource, Debug, Default)]
pub struct TextFieldFocus {
    pub field: Option<Entity>,
}

impl TextFieldFocus {
    pub fn is_focused(&self) -> bool {
        self.field.is_some()
    }
}

/// Spawn a text field. Returns the field entity (carries [`UiTextField`] +
/// the caller's marker bundle).
pub fn spawn_text_field(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    field: UiTextField,
    width: Val,
    marker: impl Bundle,
) -> Entity {
    let value = field.value.clone();
    let placeholder = field.placeholder.clone();
    parent
        .spawn((
            Button,
            Node {
                width,
                height: Val::Px(CTRL_H),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(RADIUS_CTRL)),
                padding: UiRect::horizontal(Val::Px(SPACE_SM)),
                align_items: AlignItems::Center,
                overflow: Overflow::clip_x(),
                ..Default::default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.24)),
            BorderColor::all(Color::NONE),
            Interaction::None,
            field,
            marker,
        ))
        .with_children(|c| {
            let bundle = if value.is_empty() {
                theme.faint(placeholder)
            } else {
                theme.body(value)
            };
            c.spawn((bundle, TextFieldText));
        })
        .id()
}

/// Focus a field on click; blur when clicking anything else.
pub fn focus_text_fields(
    mut focus: ResMut<TextFieldFocus>,
    fields: Query<(Entity, &Interaction), (Changed<Interaction>, With<UiTextField>)>,
    other_presses: Query<(Entity, &Interaction), (Changed<Interaction>, Without<UiTextField>)>,
) {
    for (entity, interaction) in &fields {
        if matches!(interaction, Interaction::Pressed) {
            focus.field = Some(entity);
            return;
        }
    }
    if focus.is_focused()
        && other_presses
            .iter()
            .any(|(_, i)| matches!(i, Interaction::Pressed))
    {
        focus.field = None;
    }
}

/// Feed raw key events into the focused field's value. Drains events while
/// unfocused so stale ones never replay on the next focus.
pub fn apply_text_field_input(
    mut focus: ResMut<TextFieldFocus>,
    mut key_events: MessageReader<KeyboardInput>,
    mut fields: Query<&mut UiTextField>,
) {
    let Some(entity) = focus.field else {
        key_events.clear();
        return;
    };
    let Ok(mut field) = fields.get_mut(entity) else {
        focus.field = None;
        key_events.clear();
        return;
    };
    let mut value = field.value.clone();
    for event in key_events.read() {
        if !event.state.is_pressed() {
            continue;
        }
        match &event.logical_key {
            Key::Character(s) => {
                for c in s.chars().filter(|c| !c.is_control()) {
                    if value.len() < field.max_len {
                        value.push(c);
                    }
                }
            }
            Key::Space if value.len() < field.max_len => value.push(' '),
            Key::Backspace => {
                value.pop();
            }
            Key::Enter | Key::Escape => {
                focus.field = None;
            }
            _ => {}
        }
    }
    if field.value != value {
        field.value = value;
    }
}

/// Text + caret + focus-border visuals for every field.
pub fn update_text_field_visuals(
    theme: Res<UiTheme>,
    focus: Res<TextFieldFocus>,
    fields: Query<(Entity, &UiTextField, &Children)>,
    mut texts: Query<(&mut Text, &mut TextColor, &mut TextFont), With<TextFieldText>>,
    mut borders: Query<&mut BorderColor, With<UiTextField>>,
) {
    for (entity, field, children) in &fields {
        let focused = focus.field == Some(entity);
        let (shown, color, font) = if field.value.is_empty() && !focused {
            (field.placeholder.clone(), TEXT_FAINT, theme.font_ui.clone())
        } else if focused {
            (
                format!("{}▏", field.value),
                TEXT_PRIMARY,
                theme.font_ui.clone(),
            )
        } else {
            (field.value.clone(), TEXT_PRIMARY, theme.font_ui.clone())
        };
        for child in children.iter() {
            if let Ok((mut text, mut tc, mut tf)) = texts.get_mut(child) {
                if **text != shown {
                    **text = shown.clone();
                }
                if tc.0 != color {
                    tc.0 = color;
                }
                if tf.font != font {
                    tf.font = font.clone();
                }
            }
        }
        if let Ok(mut border) = borders.get_mut(entity) {
            let target = BorderColor::all(if focused { ACCENT } else { Color::NONE });
            if border.top != target.top {
                *border = target;
            }
        }
    }
}
