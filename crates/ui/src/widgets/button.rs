//! Buttons and interactive rows.
//!
//! Every interactive control carries [`UiButton`]; one shared system
//! ([`style_buttons`]) drives hover/press/latched/selected visuals from the
//! token palette, so no screen ever hand-rolls interaction colours.

use bevy::prelude::*;

use crate::tokens::*;
use crate::UiTheme;

/// Visual variant of a [`UiButton`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ButtonVariant {
    /// Soft translucent fill, no outline (visionOS-style). The default.
    #[default]
    Ghost,
    /// Accent-filled headline action (LAUNCH, PLAY).
    Primary,
    /// Ghost with danger colouring on hover (delete, quit).
    Danger,
    /// Fully transparent at rest — dense lists and inline controls.
    Bare,
}

/// Marker + state for every themed button. `latched` renders the accent
/// "toggle on" state; `selected` renders the list-row selection fill.
#[derive(Component, Default)]
pub struct UiButton {
    pub variant: ButtonVariant,
    pub latched: bool,
    pub selected: bool,
}

impl UiButton {
    pub fn new(variant: ButtonVariant) -> Self {
        Self {
            variant,
            ..Default::default()
        }
    }
}

/// Marker on a button's main label text (styled by [`style_buttons`]).
#[derive(Component)]
pub struct ButtonLabel;

/// Marker on a button's secondary/description text (kept dim).
#[derive(Component)]
pub struct ButtonDesc;

/// Spawn a compact text button. `action` is the caller's click-marker bundle.
pub fn spawn_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    action: impl Bundle,
    label: &str,
    variant: ButtonVariant,
    height: f32,
) -> Entity {
    parent
        .spawn((
            Button,
            Node {
                height: Val::Px(height),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(RADIUS_CTRL)),
                padding: UiRect::horizontal(Val::Px(SPACE_MD)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..Default::default()
            },
            BackgroundColor(FILL_REST),
            BorderColor::all(Color::NONE),
            Interaction::None,
            UiButton::new(variant),
            action,
        ))
        .with_children(|c| {
            c.spawn((
                theme.text_for_button(label, variant),
                ButtonLabel,
            ));
        })
        .id()
}

/// Spawn a full-width menu row: semibold label left, dim description right.
/// The menu-screen staple (PLAY / SETTINGS / …).
pub fn spawn_menu_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    action: impl Bundle,
    label: &str,
    desc: &str,
) -> Entity {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(CTRL_H_LG),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(RADIUS_CTRL)),
                padding: UiRect::horizontal(Val::Px(SPACE_LG)),
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                ..Default::default()
            },
            BackgroundColor(FILL_REST),
            BorderColor::all(Color::NONE),
            Interaction::None,
            UiButton::new(ButtonVariant::Ghost),
            action,
        ))
        .with_children(|c| {
            c.spawn((theme.body_strong(label), ButtonLabel));
            if !desc.is_empty() {
                c.spawn((theme.small(desc), ButtonDesc));
            }
        })
        .id()
}

impl UiTheme {
    /// Label text for a button of the given variant.
    pub fn text_for_button(&self, label: &str, variant: ButtonVariant) -> (Text, TextFont, TextColor) {
        let mut bundle = self.body_strong(label);
        bundle.1.font_size = FontSize::Px(FS_SMALL + 1.0);
        if variant == ButtonVariant::Primary {
            bundle.2 = TextColor(ON_ACCENT);
        }
        bundle
    }
}

/// Hover / press / latched / selected visuals for every [`UiButton`].
pub fn style_buttons(
    mut buttons: Query<(
        &Interaction,
        &UiButton,
        &mut BorderColor,
        &mut BackgroundColor,
        &Children,
    )>,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
) {
    for (interaction, button, mut border, mut bg, children) in &mut buttons {
        let (border_c, bg_c, label_c) = match button.variant {
            ButtonVariant::Primary => match interaction {
                Interaction::Pressed => (ACCENT_DIM, ACCENT_DIM, ON_ACCENT),
                Interaction::Hovered => (Color::srgba(1.0, 0.86, 0.55, 1.0), Color::srgba(1.0, 0.86, 0.55, 1.0), ON_ACCENT),
                Interaction::None => (ACCENT, ACCENT, ON_ACCENT),
            },
            ButtonVariant::Danger => match interaction {
                Interaction::Pressed => (Color::NONE, Color::srgba(1.0, 0.45, 0.35, 0.28), DANGER),
                Interaction::Hovered => (Color::NONE, Color::srgba(1.0, 0.45, 0.35, 0.14), DANGER),
                Interaction::None => (Color::NONE, FILL_REST, TEXT_DIM),
            },
            ButtonVariant::Ghost | ButtonVariant::Bare => {
                let rest_fill = if button.variant == ButtonVariant::Bare {
                    Color::NONE
                } else {
                    FILL_REST
                };
                match (interaction, button.latched, button.selected) {
                    (Interaction::Pressed, _, _) => (Color::NONE, FILL_ACTIVE, TEXT_PRIMARY),
                    (Interaction::Hovered, _, _) => (Color::NONE, FILL_HOVER, TEXT_PRIMARY),
                    (Interaction::None, true, _) => (ACCENT_DIM, FILL_SELECTED, ACCENT),
                    (Interaction::None, _, true) => (ACCENT_DIM, FILL_SELECTED, TEXT_PRIMARY),
                    (Interaction::None, false, false) => (Color::NONE, rest_fill, TEXT_PRIMARY),
                }
            }
        };
        let new_border = BorderColor::all(border_c);
        if border.top != new_border.top {
            *border = new_border;
        }
        if bg.0 != bg_c {
            bg.0 = bg_c;
        }
        for child in children.iter() {
            if let Ok(mut tc) = labels.get_mut(child)
                && tc.0 != label_c
            {
                tc.0 = label_c;
            }
        }
    }
}
