//! Shared native-Bevy-UI widget toolkit, styled from [`HudTheme`].
//!
//! Bevy UI ships no higher-level widgets, so player-facing panels (settings,
//! the celestial-bodies tree, the maneuver node editor) build on these:
//!
//! - [`UiButton`] — text button with hover/press/latched feedback (also the
//!   building block for tab strips and toggles).
//! - [`UiSlider`] — horizontal drag slider with a value label.
//! - [`UiCheckbox`] — labelled boolean toggle.
//! - [`UiCycle`] — `[<] value [>]` one-of-N picker (a dropdown without a popup,
//!   which sorts reliably under Bevy UI's z-order).
//! - [`ScrollableColumn`] — wheel-scrollable `overflow: scroll_y` column.
//! - [`TextField`] — minimal single-line text field with shared focus.
//!
//! Every widget stores its own state and is driven by the systems in
//! [`UiWidgetsPlugin`]; consumers react to `Changed<Widget>` and read the
//! widget's value field. Spawn helpers initialise the widget from the model;
//! the panel rebuilds its body when the backing model changes (the same
//! rebuild-on-demand pattern the shipyard inspector uses).
//!
//! These intentionally duplicate the shipyard editor's private widgets
//! (`shipyard_editor::ui::widgets`); folding the editor onto this toolkit is a
//! later cleanup once both have been runtime-verified.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use crate::hud::theme::HudTheme;

pub struct UiWidgetsPlugin;

impl Plugin for UiWidgetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TextFieldFocus>().add_systems(
            Update,
            (
                style_buttons,
                drive_sliders,
                update_slider_visuals.after(drive_sliders),
                drive_checkboxes,
                update_checkbox_visuals,
                drive_cycles,
                update_cycle_visuals.after(drive_cycles),
                scroll_scrollables,
                focus_text_field_on_click,
                edit_focused_text_field.after(focus_text_field_on_click),
                update_text_field_visuals.after(edit_focused_text_field),
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------

/// Marker for buttons taking the shared hover/press styling. `latched` renders
/// the button "active" (accent border + text) from a state flag the owner keeps
/// in sync — used for tab strips and toggles.
#[derive(Component, Default)]
pub struct UiButton {
    pub latched: bool,
}

/// Spawn a one-line text button. `binding` is the caller's click-marker bundle;
/// [`style_buttons`] drives the visuals.
pub fn spawn_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    binding: impl Bundle,
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
                padding: UiRect::axes(Val::Px(10.0), Val::Px(2.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            UiButton::default(),
            binding,
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size,
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
        })
        .id()
}

/// Hover / press / latched visuals for every [`UiButton`].
fn style_buttons(
    theme: Res<HudTheme>,
    mut buttons: Query<
        (
            &Interaction,
            &UiButton,
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
        apply_border_bg(&mut border, &mut bg, border_color, bg_color);
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
    /// `0.12`
    Plain2,
    /// `1.25×`
    Scale2,
}

impl SliderFormat {
    pub fn format(self, value: f32) -> String {
        match self {
            SliderFormat::Plain2 => format!("{value:.2}"),
            SliderFormat::Scale2 => format!("{value:.2}×"),
        }
    }
}

/// A horizontal drag slider. The track carries this + `Interaction` +
/// `RelativeCursorPosition`; [`drive_sliders`] maps a held press to `value`.
/// Consumers react to `Changed<UiSlider>`.
#[derive(Component, Debug, Clone)]
pub struct UiSlider {
    pub min: f32,
    pub max: f32,
    pub value: f32,
    /// Rounds the dragged value to a multiple of this (0 = continuous).
    pub step: f32,
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

/// Spawn a labelled slider row: `LABEL  [====    ]  12.5`. Returns the track
/// entity (which carries [`UiSlider`] + the caller's `binding` bundle).
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
            height: Val::Px(22.0),
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
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.text_dim),
                Node {
                    width: Val::Px(120.0),
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
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.text_primary),
                Node {
                    width: Val::Px(56.0),
                    flex_shrink: 0.0,
                    justify_content: JustifyContent::FlexEnd,
                    ..default()
                },
                SliderValueText(track_entity),
            ));
        });
    track_entity
}

/// Map a held press on a slider track to its value. `RelativeCursorPosition`
/// keeps reporting outside the node, so an overshooting drag keeps tracking —
/// clamp to range and quantise to `step`.
fn drive_sliders(mut sliders: Query<(&Interaction, &RelativeCursorPosition, &mut UiSlider)>) {
    for (interaction, rel, mut slider) in &mut sliders {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let Some(normalized) = rel.normalized else {
            continue;
        };
        let x = normalized.x.clamp(0.0, 1.0);
        let mut value = slider.min + x * (slider.max - slider.min);
        if slider.step > 0.0 {
            value = (value / slider.step).round() * slider.step;
        }
        value = value.clamp(slider.min, slider.max);
        if (slider.value - value).abs() > 1.0e-5 {
            slider.value = value;
        }
    }
}

/// Keep each slider's fill width and value label in sync with its value.
fn update_slider_visuals(
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
// Checkbox
// ---------------------------------------------------------------------------

/// A labelled boolean toggle. The whole row is the button; clicking flips
/// `checked`. Consumers react to `Changed<UiCheckbox>`.
#[derive(Component, Debug, Clone, Copy)]
pub struct UiCheckbox {
    pub checked: bool,
}

/// Marker on the filled square inside a checkbox row.
#[derive(Component)]
struct CheckboxBox;

/// Spawn a checkbox row: `[x] Label`. Returns the row entity (carries
/// [`UiCheckbox`] + `binding`).
pub fn spawn_checkbox_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    label: &str,
    checked: bool,
    binding: impl Bundle,
) -> Entity {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(20.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                ..default()
            },
            Interaction::None,
            UiCheckbox { checked },
            binding,
        ))
        .with_children(|row| {
            row.spawn((
                Node {
                    width: Val::Px(13.0),
                    height: Val::Px(13.0),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(2.0)),
                    flex_shrink: 0.0,
                    ..default()
                },
                BackgroundColor(if checked {
                    theme.text_accent
                } else {
                    Color::NONE
                }),
                BorderColor::all(theme.panel_border),
                CheckboxBox,
            ));
            row.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
        })
        .id()
}

/// Flip a checkbox on press.
fn drive_checkboxes(
    mut boxes: Query<(&Interaction, &mut UiCheckbox), Changed<Interaction>>,
) {
    for (interaction, mut checkbox) in &mut boxes {
        if matches!(interaction, Interaction::Pressed) {
            checkbox.checked = !checkbox.checked;
        }
    }
}

/// Fill / border tint for each checkbox from its state + hover.
fn update_checkbox_visuals(
    theme: Res<HudTheme>,
    boxes: Query<(&UiCheckbox, &Interaction, &Children)>,
    mut squares: Query<(&mut BackgroundColor, &mut BorderColor), With<CheckboxBox>>,
) {
    for (checkbox, interaction, children) in &boxes {
        let Some(&child) = children.first() else {
            continue;
        };
        let Ok((mut bg, mut border)) = squares.get_mut(child) else {
            continue;
        };
        let fill = if checkbox.checked {
            theme.text_accent
        } else {
            Color::NONE
        };
        let border_color = if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            theme.text_accent
        } else {
            theme.panel_border
        };
        if bg.0 != fill {
            bg.0 = fill;
        }
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
    }
}

// ---------------------------------------------------------------------------
// Cycle (one-of-N picker)
// ---------------------------------------------------------------------------

/// A `[<] value [>]` one-of-N picker. Holds its own option labels so it renders
/// the current value without help; consumers react to `Changed<UiCycle>` and
/// map `index` onto their real value. Spawn it with the model's current index.
#[derive(Component, Debug, Clone)]
pub struct UiCycle {
    pub index: usize,
    pub options: Vec<String>,
}

impl UiCycle {
    fn len(&self) -> usize {
        self.options.len()
    }
}

/// Prev/next arrow inside a cycle. Its parent ([`ChildOf`]) is the [`UiCycle`]
/// track, so the driver needs no entity back-reference.
#[derive(Component)]
struct CycleArrow {
    delta: i32,
}

/// Value label; its parent ([`ChildOf`]) is the [`UiCycle`] track.
#[derive(Component)]
struct CycleValueText;

/// Spawn a labelled cycle row: `LABEL  [<] value [>]`. Returns the track entity
/// (carries [`UiCycle`] + `binding`). The prev/value/next nodes are its direct
/// children, so the driver/visual systems reach the track via `ChildOf`.
pub fn spawn_cycle_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    label: &str,
    options: Vec<String>,
    index: usize,
    binding: impl Bundle,
) -> Entity {
    let index = index.min(options.len().saturating_sub(1));
    let current = options.get(index).cloned().unwrap_or_default();
    let mut track_entity = Entity::PLACEHOLDER;
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Px(24.0),
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
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.text_dim),
                Node {
                    width: Val::Px(120.0),
                    flex_shrink: 0.0,
                    ..default()
                },
            ));
            track_entity = row
                .spawn((
                    Node {
                        flex_grow: 1.0,
                        height: Val::Px(22.0),
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(6.0),
                        ..default()
                    },
                    UiCycle { index, options },
                ))
                .insert(binding)
                .with_children(|track| {
                    spawn_cycle_arrow(track, theme, -1, "‹");
                    track.spawn((
                        Text::new(current),
                        TextFont {
                            font: theme.font.clone(),
                            font_size: 11.0,
                            ..default()
                        },
                        TextColor(theme.text_primary),
                        Node {
                            flex_grow: 1.0,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        CycleValueText,
                    ));
                    spawn_cycle_arrow(track, theme, 1, "›");
                })
                .id();
        });
    track_entity
}

fn spawn_cycle_arrow(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    delta: i32,
    glyph: &str,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(20.0),
                height: Val::Px(20.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                flex_shrink: 0.0,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            UiButton::default(),
            CycleArrow { delta },
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(glyph),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 13.0,
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
        });
}

/// Advance a cycle's index when its arrows are pressed (wrapping).
fn drive_cycles(
    arrows: Query<(&Interaction, &ChildOf, &CycleArrow), Changed<Interaction>>,
    mut cycles: Query<&mut UiCycle>,
) {
    for (interaction, child_of, arrow) in &arrows {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let Ok(mut cycle) = cycles.get_mut(child_of.0) else {
            continue;
        };
        let len = cycle.len();
        if len == 0 {
            continue;
        }
        let next = (cycle.index as i32 + arrow.delta).rem_euclid(len as i32) as usize;
        if next != cycle.index {
            cycle.index = next;
        }
    }
}

/// Render each cycle's current option into its value label.
fn update_cycle_visuals(
    cycles: Query<&UiCycle>,
    mut labels: Query<(&ChildOf, &mut Text), With<CycleValueText>>,
) {
    for (child_of, mut text) in &mut labels {
        let Ok(cycle) = cycles.get(child_of.0) else {
            continue;
        };
        let value = cycle.options.get(cycle.index).cloned().unwrap_or_default();
        if **text != value {
            **text = value;
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
fn scroll_scrollables(
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
// Text field
// ---------------------------------------------------------------------------

/// A minimal single-line text field. Holds its own editable string so it is
/// self-contained; consumers react to `Changed<TextField>` and read `text`.
#[derive(Component, Debug, Clone)]
pub struct TextField {
    pub text: String,
    pub max_len: usize,
}

/// Which text field currently owns the keyboard. While `Some`, the game input
/// gate disables the keyboard action source (see `crate::input`) so raw keys
/// edit the field instead of tripping flight/system bindings.
#[derive(Resource, Debug, Default)]
pub struct TextFieldFocus {
    pub field: Option<Entity>,
}

impl TextFieldFocus {
    pub fn is_focused(&self) -> bool {
        self.field.is_some()
    }
}

/// Inner text node of a [`TextField`].
#[derive(Component)]
struct TextFieldText;

/// Spawn a text field. Returns the field entity (carries [`TextField`] +
/// `binding`); width is fixed so it lays out predictably in a row.
pub fn spawn_text_field(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    initial: &str,
    width_px: f32,
    binding: impl Bundle,
) -> Entity {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(width_px),
                height: Val::Px(20.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(2.0)),
                padding: UiRect::axes(Val::Px(6.0), Val::Px(1.0)),
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.02, 0.02, 0.9)),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            TextField {
                text: initial.to_string(),
                max_len: 48,
            },
            binding,
        ))
        .with_children(|f| {
            f.spawn((
                Text::new(initial),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.text_primary),
                TextFieldText,
            ));
        })
        .id()
}

/// Focus a text field on click; defocus when clicking anything else.
fn focus_text_field_on_click(
    mut focus: ResMut<TextFieldFocus>,
    fields: Query<(Entity, &Interaction), (Changed<Interaction>, With<TextField>)>,
    other_presses: Query<(Entity, &Interaction), (Changed<Interaction>, Without<TextField>)>,
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

/// Feed raw key events into the focused field's backing string.
fn edit_focused_text_field(
    mut focus: ResMut<TextFieldFocus>,
    mut key_events: MessageReader<KeyboardInput>,
    mut fields: Query<&mut TextField>,
) {
    let Some(entity) = focus.field else {
        key_events.clear();
        return;
    };
    let Ok(mut field) = fields.get_mut(entity) else {
        focus.field = None;
        return;
    };
    let max_len = field.max_len;
    for event in key_events.read() {
        if !event.state.is_pressed() {
            continue;
        }
        match &event.logical_key {
            Key::Character(s) => {
                for c in s.chars().filter(|c| !c.is_control()) {
                    if field.text.chars().count() < max_len {
                        field.text.push(c);
                    }
                }
            }
            Key::Space if field.text.chars().count() < max_len => field.text.push(' '),
            Key::Backspace => {
                field.text.pop();
            }
            Key::Enter | Key::Escape => {
                focus.field = None;
                break;
            }
            _ => {}
        }
    }
}

/// Render each field's text (with a caret while focused) + focus border.
fn update_text_field_visuals(
    theme: Res<HudTheme>,
    focus: Res<TextFieldFocus>,
    mut fields: Query<(Entity, &TextField, &mut BorderColor, &Children)>,
    mut texts: Query<&mut Text, With<TextFieldText>>,
) {
    for (entity, field, mut border, children) in &mut fields {
        let focused = focus.field == Some(entity);
        let border_color = if focused {
            theme.text_accent
        } else {
            theme.panel_border
        };
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
        if let Some(&child) = children.first()
            && let Ok(mut text) = texts.get_mut(child)
        {
            let shown = if focused {
                format!("{}_", field.text)
            } else {
                field.text.clone()
            };
            if **text != shown {
                **text = shown;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Set border + background only when they actually change (avoids dirtying the
/// UI layout every frame).
pub fn apply_border_bg(
    border: &mut BorderColor,
    bg: &mut BackgroundColor,
    border_color: Color,
    bg_color: Color,
) {
    let new_border = BorderColor::all(border_color);
    if border.top != new_border.top {
        *border = new_border;
    }
    if bg.0 != bg_color {
        bg.0 = bg_color;
    }
}
