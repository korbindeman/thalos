//! Small reusable Bevy-UI widgets for the shipyard editor, styled from
//! [`HudTheme`]: text buttons with hover/press feedback, a draggable value
//! slider (Bevy UI has no built-in one), wheel-scrollable columns, and a
//! minimal single-line text field.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use crate::hud::theme::HudTheme;

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------

/// Marker for editor buttons that take the shared hover/press styling.
/// `accent_when` lets toggle buttons render "latched" (accent border + text)
/// from a state flag the toggle system keeps in sync.
#[derive(Component, Default)]
pub struct EditorUiButton {
    pub latched: bool,
}

/// Spawn a one-line text button. `action` is the caller's click-marker
/// bundle; the shared [`style_editor_buttons`] system drives the visuals.
pub fn spawn_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    action: impl Bundle,
    label: &str,
    font_size: f32,
    height_px: f32,
) -> Entity {
    parent
        .spawn((
            Button,
            Node {
                height: Val::Px(height_px),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(2.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            EditorUiButton::default(),
            action,
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(font_size),
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
        })
        .id()
}

/// Hover / press / latched visuals for every [`EditorUiButton`]. Mirrors the
/// pause menu's button styling so the editor reads as the same UI family.
pub fn style_editor_buttons(
    theme: Res<HudTheme>,
    mut buttons: Query<
        (
            &Interaction,
            &EditorUiButton,
            &mut BorderColor,
            &mut BackgroundColor,
            &Children,
        ),
        With<Button>,
    >,
    mut text_q: Query<&mut TextColor>,
) {
    for (interaction, button, mut border, mut bg, children) in &mut buttons {
        let (border_color, bg_color, label_color) = match (interaction, button.latched) {
            (Interaction::Pressed, _) => {
                (theme.text_primary, theme.panel_border, theme.text_primary)
            }
            (Interaction::Hovered, _) => (theme.text_accent, theme.panel_bg, theme.text_accent),
            (Interaction::None, true) => (theme.text_accent, theme.panel_bg_alt, theme.text_accent),
            (Interaction::None, false) => (theme.panel_border, theme.panel_bg, theme.text_primary),
        };
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
        if bg.0 != bg_color {
            bg.0 = bg_color;
        }
        if let Some(&child) = children.first()
            && let Ok(mut tc) = text_q.get_mut(child)
            && tc.0 != label_color
        {
            tc.0 = label_color;
        }
    }
}

// ---------------------------------------------------------------------------
// Slider
// ---------------------------------------------------------------------------

/// How a slider's value renders in its value label.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SliderFormat {
    /// `12.5 m`
    Meters,
    /// `35°`
    Degrees,
    /// `0.12`
    Plain2,
    /// `420 L` (any unit suffix)
    Amount(&'static str),
}

impl SliderFormat {
    pub fn format(self, value: f32) -> String {
        match self {
            SliderFormat::Meters => format!("{value:.2} m"),
            SliderFormat::Degrees => format!("{value:.1}°"),
            SliderFormat::Plain2 => format!("{value:.2}"),
            SliderFormat::Amount(unit) => format!("{value:.0} {unit}"),
        }
    }
}

/// A horizontal drag slider. The track node carries this component plus
/// `Interaction` + `RelativeCursorPosition`; [`drive_sliders`] maps a held
/// press to `value`, and consumers react to `Changed<UiSlider>` (writes are
/// value-guarded, so change detection means "the user moved it" — or a
/// refresh system synced it from the model).
#[derive(Component, Debug, Clone)]
pub struct UiSlider {
    pub min: f32,
    pub max: f32,
    pub value: f32,
    pub format: SliderFormat,
}

impl UiSlider {
    pub fn fraction(&self) -> f32 {
        if self.max > self.min {
            ((self.value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Marker on the fill bar inside a slider track.
#[derive(Component)]
pub struct SliderFill;

/// Value label tied to a slider track entity.
#[derive(Component)]
pub struct SliderValueText(pub Entity);

/// Spawn a labelled slider row: `LABEL  [====    ]  12.5 m`. Returns the
/// track entity (which carries [`UiSlider`] + the caller's binding bundle).
pub fn spawn_slider_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    label: &str,
    slider: UiSlider,
    binding: impl Bundle,
) -> Entity {
    let mut track_entity = Entity::PLACEHOLDER;
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Px(20.0),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(theme.text_dim),
                Node {
                    width: Val::Px(96.0),
                    flex_shrink: 0.0,
                    ..default()
                },
            ));
            let fraction = slider.fraction();
            track_entity = row
                .spawn((
                    Node {
                        flex_grow: 1.0,
                        height: Val::Px(12.0),
                        border: UiRect::all(Val::Px(1.0)),
                        border_radius: BorderRadius::all(Val::Px(2.0)),
                        padding: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.02, 0.02, 0.02, 0.9)),
                    BorderColor::all(theme.panel_border),
                    Interaction::None,
                    RelativeCursorPosition::default(),
                    slider,
                    binding,
                ))
                .with_children(|track| {
                    track.spawn((
                        Node {
                            width: Val::Percent(fraction * 100.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(theme.text_accent.with_alpha(0.55)),
                        SliderFill,
                    ));
                })
                .id();
            row.spawn((
                Text::new(""),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(theme.text_primary),
                Node {
                    width: Val::Px(60.0),
                    flex_shrink: 0.0,
                    justify_content: JustifyContent::FlexEnd,
                    ..default()
                },
                SliderValueText(track_entity),
            ));
        });
    track_entity
}

/// Map a held press on a slider track to its value. `Interaction::Pressed`
/// latches for the whole drag, and `RelativeCursorPosition` keeps reporting
/// outside the node, so the drag keeps tracking when the cursor overshoots —
/// clamp to the range.
pub fn drive_sliders(mut sliders: Query<(&Interaction, &RelativeCursorPosition, &mut UiSlider)>) {
    for (interaction, rel, mut slider) in &mut sliders {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let Some(normalized) = rel.normalized else {
            continue;
        };
        let x = normalized.x.clamp(0.0, 1.0);
        let value = slider.min + x * (slider.max - slider.min);
        if (slider.value - value).abs() > 1.0e-5 {
            slider.value = value;
        }
    }
}

/// Keep each slider's fill width and value label in sync with its value.
pub fn update_slider_visuals(
    sliders: Query<(&UiSlider, &Children)>,
    mut fills: Query<&mut Node, With<SliderFill>>,
    mut labels: Query<(&SliderValueText, &mut Text)>,
) {
    for (slider, children) in &sliders {
        let target = Val::Percent(slider.fraction() * 100.0);
        for child in children.iter() {
            if let Ok(mut node) = fills.get_mut(child)
                && node.width != target
            {
                node.width = target;
            }
        }
    }
    for (value_text, mut text) in &mut labels {
        let Ok((slider, _)) = sliders.get(value_text.0) else {
            continue;
        };
        let formatted = slider.format.format(slider.value);
        if **text != formatted {
            **text = formatted;
        }
    }
}

// ---------------------------------------------------------------------------
// Scrolling
// ---------------------------------------------------------------------------

/// Marker for wheel-scrollable columns (`overflow: scroll_y`).
#[derive(Component)]
pub struct ScrollableColumn;

/// Wheel-scroll whichever scrollable column the cursor is over.
pub fn scroll_scrollables(
    mut wheel: MessageReader<MouseWheel>,
    mut scrollables: Query<(&RelativeCursorPosition, &mut ScrollPosition), With<ScrollableColumn>>,
) {
    for event in wheel.read() {
        let dy = match event.unit {
            MouseScrollUnit::Line => event.y * 28.0,
            MouseScrollUnit::Pixel => event.y,
        };
        if dy == 0.0 {
            continue;
        }
        for (rel, mut scroll) in &mut scrollables {
            if rel.cursor_over() {
                scroll.y -= dy;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Text input
// ---------------------------------------------------------------------------

/// Marker for the ship-name text field.
#[derive(Component)]
pub struct ShipNameField;

/// Which editor text field currently owns the keyboard. While focused, the
/// game's input gate disables the keyboard action source so raw key events
/// (consumed here) don't trip flight/system bindings.
#[derive(Resource, Debug, Default)]
pub struct EditorTextFocus {
    pub field: Option<Entity>,
}

impl EditorTextFocus {
    pub fn is_focused(&self) -> bool {
        self.field.is_some()
    }
}

/// Focus the name field on click; defocus when clicking anything else.
pub fn focus_text_field_on_click(
    mut focus: ResMut<EditorTextFocus>,
    fields: Query<(Entity, &Interaction), (Changed<Interaction>, With<ShipNameField>)>,
    other_presses: Query<(Entity, &Interaction), (Changed<Interaction>, Without<ShipNameField>)>,
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

/// Feed raw key events into the focused field's backing string. Returns the
/// edited string through the closure-style accessor pattern used by the
/// caller (`apply_text_input` in the panel module owns the model write).
pub fn collect_text_edits(
    focus: &mut EditorTextFocus,
    key_events: &mut MessageReader<KeyboardInput>,
    text: &mut String,
) {
    for event in key_events.read() {
        if !event.state.is_pressed() {
            continue;
        }
        match &event.logical_key {
            Key::Character(s) => {
                for c in s.chars().filter(|c| !c.is_control()) {
                    if text.len() < 48 {
                        text.push(c);
                    }
                }
            }
            Key::Space if text.len() < 48 => text.push(' '),
            Key::Backspace => {
                text.pop();
            }
            Key::Enter | Key::Escape => {
                focus.field = None;
            }
            _ => {}
        }
    }
}
