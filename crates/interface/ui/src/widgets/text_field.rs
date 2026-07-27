//! Minimal single-line text field.
//!
//! One focused field at a time ([`TextFieldFocus`]); raw key events feed the
//! focused field's `value`. Hosts gate their own keyboard bindings on
//! [`TextFieldFocus::is_focused`] so typing never trips game actions.
//!
//! There is no caret/selection model — the caret is always at the end. The one
//! exception is [`UiTextField::selected`]: a prefilled suggestion that renders
//! highlighted and is replaced wholesale by the first keystroke, so a
//! suggest-and-accept prompt is one keypress to take and one word to override.

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
    /// The whole value is selected: it renders highlighted, and the next
    /// keystroke replaces it instead of appending. Cleared by any edit, by
    /// clicking the field, and by losing focus.
    pub select_all: bool,
}

impl UiTextField {
    pub fn new(value: impl Into<String>, placeholder: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            placeholder: placeholder.into(),
            max_len: 48,
            select_all: false,
        }
    }

    /// Start with the value fully selected — for a prefilled suggestion the
    /// user should be able to accept with Enter or type straight over.
    pub fn selected(mut self) -> Self {
        self.select_all = true;
        self
    }
}

/// Emitted when a focused field is dismissed by the keyboard: Enter
/// (`accepted`) or Escape (cancelled). Hosts that need commit-vs-cancel read
/// this; hosts that only mirror the value can keep using `Changed<UiTextField>`.
#[derive(Message, Debug, Clone, Copy)]
pub struct TextFieldSubmit {
    pub field: Entity,
    pub accepted: bool,
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
            // The selection highlight is this node's background fill.
            c.spawn((bundle, BackgroundColor(Color::NONE), TextFieldText));
        })
        .id()
}

/// Focus a field on click; blur when clicking anything else.
pub fn focus_text_fields(
    mut focus: ResMut<TextFieldFocus>,
    mut fields: Query<(Entity, &Interaction, &mut UiTextField), Changed<Interaction>>,
    other_presses: Query<(Entity, &Interaction), (Changed<Interaction>, Without<UiTextField>)>,
) {
    for (entity, interaction, mut field) in &mut fields {
        if matches!(interaction, Interaction::Pressed) {
            focus.field = Some(entity);
            // A click means "edit this", not "replace it" — the caret lands at
            // the end rather than keeping a prefilled suggestion selected.
            if field.select_all {
                field.select_all = false;
            }
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
    mut submits: MessageWriter<TextFieldSubmit>,
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
    let mut select_all = field.select_all;
    // Any keystroke that edits the value first consumes the selection, which
    // is what makes a prefilled suggestion typeable-over.
    let take_selection = |value: &mut String, select_all: &mut bool| {
        if *select_all {
            value.clear();
            *select_all = false;
        }
    };
    let mut submitted = false;
    for event in key_events.read() {
        // Everything after Enter/Escape belongs to whatever the host opens
        // next, not to a field that has already closed.
        if !event.state.is_pressed() || submitted {
            continue;
        }
        match &event.logical_key {
            Key::Character(s) => {
                take_selection(&mut value, &mut select_all);
                for c in s.chars().filter(|c| !c.is_control()) {
                    if value.len() < field.max_len {
                        value.push(c);
                    }
                }
            }
            Key::Space => {
                take_selection(&mut value, &mut select_all);
                if value.len() < field.max_len {
                    value.push(' ');
                }
            }
            Key::Backspace | Key::Delete => {
                if select_all {
                    take_selection(&mut value, &mut select_all);
                } else {
                    value.pop();
                }
            }
            key @ (Key::Enter | Key::Escape) => {
                select_all = false;
                focus.field = None;
                submitted = true;
                submits.write(TextFieldSubmit {
                    field: entity,
                    accepted: matches!(key, Key::Enter),
                });
            }
            _ => {}
        }
    }
    if field.value != value {
        field.value = value;
    }
    if field.select_all != select_all {
        field.select_all = select_all;
    }
}

/// Text + caret + focus-border visuals for every field.
pub fn update_text_field_visuals(
    theme: Res<UiTheme>,
    focus: Res<TextFieldFocus>,
    fields: Query<(Entity, &UiTextField, &Children)>,
    mut texts: Query<
        (
            &mut Text,
            &mut TextColor,
            &mut TextFont,
            &mut BackgroundColor,
        ),
        With<TextFieldText>,
    >,
    mut borders: Query<&mut BorderColor, With<UiTextField>>,
) {
    for (entity, field, children) in &fields {
        let focused = focus.field == Some(entity);
        let selected = focused && field.select_all && !field.value.is_empty();
        let (shown, color, font) = if field.value.is_empty() && !focused {
            (field.placeholder.clone(), TEXT_FAINT, theme.font_ui.clone())
        } else if selected {
            // Selected text carries the caret in its highlight, not a bar.
            (field.value.clone(), ON_ACCENT, theme.font_ui.clone())
        } else if focused {
            (
                format!("{}▏", field.value),
                TEXT_PRIMARY,
                theme.font_ui.clone(),
            )
        } else {
            (field.value.clone(), TEXT_PRIMARY, theme.font_ui.clone())
        };
        let highlight = if selected { ACCENT } else { Color::NONE };
        for child in children.iter() {
            if let Ok((mut text, mut tc, mut tf, mut bg)) = texts.get_mut(child) {
                if **text != shown {
                    **text = shown.clone();
                }
                if tc.0 != color {
                    tc.0 = color;
                }
                if tf.font != font {
                    tf.font = font.clone();
                }
                if bg.0 != highlight {
                    bg.0 = highlight;
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
