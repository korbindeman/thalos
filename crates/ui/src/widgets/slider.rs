//! Drag slider — Bevy UI has no built-in one.
//!
//! Styled as a *filled bar* (visionOS-like): the whole control is one rounded
//! glass bar; the label sits inside on the left, the value inside on the
//! right, a lighter fill sweeps from the left edge with a bright caret line
//! at its lip, and faint tick dots mark the run. Drag anywhere on the bar.

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use crate::tokens::*;
use crate::UiTheme;

/// How a slider's value renders in its value label.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SliderFormat {
    /// `12.50 m`
    Meters,
    /// `35.0°`
    Degrees,
    /// `0.12`
    Plain2,
    /// `1.25×`
    Scale2,
    /// `420 L` (any unit suffix)
    Amount(&'static str),
}

impl SliderFormat {
    pub fn format(self, value: f32) -> String {
        match self {
            SliderFormat::Meters => format!("{value:.2} m"),
            SliderFormat::Degrees => format!("{value:.1}°"),
            SliderFormat::Plain2 => format!("{value:.2}"),
            SliderFormat::Scale2 => format!("{value:.2}×"),
            SliderFormat::Amount(unit) => format!("{value:.0} {unit}"),
        }
    }
}

/// A horizontal drag slider. The bar node carries this plus `Interaction`
/// and `RelativeCursorPosition`; [`drive_sliders`] maps a held press to
/// `value`, and consumers react to `Changed<UiSlider>`. Writes are
/// value-guarded, so change detection means "the user moved it" — or a
/// refresh system synced it from the model.
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
    /// A continuous slider (no step quantisation).
    pub fn new(min: f32, max: f32, value: f32, format: SliderFormat) -> Self {
        Self {
            min,
            max,
            value,
            step: 0.0,
            format,
        }
    }

    pub fn fraction(&self) -> f32 {
        if self.max > self.min {
            ((self.value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Marker on the fill sheet inside a slider bar (its right border is the
/// caret line).
#[derive(Component)]
pub struct SliderFill;

/// Value label tied to a slider bar entity.
#[derive(Component)]
pub struct SliderValueText(pub Entity);

/// Bar height: taller than a plain row — the label lives inside it.
const SLIDER_H: f32 = 26.0;
/// Fill sheet: a soft lighter pane over the glass.
const FILL: Color = Color::srgba(1.0, 1.0, 1.0, 0.10);
/// The bright caret line at the fill's lip.
const CARET: Color = Color::srgba(1.0, 1.0, 1.0, 0.85);
/// Faint tick dots along the run.
const TICK: Color = Color::srgba(1.0, 1.0, 1.0, 0.12);
const TICK_COUNT: usize = 7;

/// Spawn a filled-bar slider: `[ LABEL ······ 12.5 m ]`. Returns the bar
/// entity (which carries [`UiSlider`] + the caller's binding bundle).
pub fn spawn_slider_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    label: &str,
    slider: UiSlider,
    binding: impl Bundle,
) -> Entity {
    let fraction = slider.fraction();
    let mut bar = parent.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Px(SLIDER_H),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(RADIUS_CTRL)),
            padding: UiRect::horizontal(Val::Px(SPACE_MD)),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            overflow: Overflow::clip(),
            ..Default::default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.20)),
        BorderColor::all(STROKE),
        Interaction::None,
        RelativeCursorPosition::default(),
        slider,
        binding,
    ));
    let bar_entity = bar.id();
    bar.with_children(|bar| {
        // Fill sheet (absolute, behind the texts); its 2px right border is
        // the caret line — at fraction 0 the sheet is zero-wide and only the
        // caret shows at the left lip, exactly like the reference.
        bar.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                width: Val::Percent(fraction * 100.0),
                border: UiRect::right(Val::Px(2.0)),
                ..Default::default()
            },
            BackgroundColor(FILL),
            BorderColor::all(CARET),
            SliderFill,
        ));
        // Tick dots along the run (under the texts, over the fill).
        bar.spawn(Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            right: Val::Px(0.0),
            top: Val::Px(0.0),
            bottom: Val::Px(0.0),
            justify_content: JustifyContent::SpaceEvenly,
            align_items: AlignItems::Center,
            ..Default::default()
        })
        .with_children(|ticks| {
            for _ in 0..TICK_COUNT {
                ticks.spawn((
                    Node {
                        width: Val::Px(3.0),
                        height: Val::Px(3.0),
                        border_radius: BorderRadius::all(Val::Px(2.0)),
                        ..Default::default()
                    },
                    BackgroundColor(TICK),
                ));
            }
        });
        // Label inside-left, value inside-right (flow children of the
        // space-between bar, drawn over fill + ticks).
        bar.spawn(theme.small(label));
        bar.spawn((theme.mono(""), SliderValueText(bar_entity)));
    });
    bar_entity
}

/// Map a held press on a slider bar to its value. `Interaction::Pressed`
/// latches for the whole drag, and `RelativeCursorPosition` keeps reporting
/// outside the node, so the drag keeps tracking on overshoot — clamped.
///
/// NB: Bevy 0.19's `RelativeCursorPosition::normalized` is **centre-origin**
/// ((-0.5, -0.5) top-left → (0.5, 0.5) bottom-right), not the pre-0.19
/// 0..1-from-top-left — shift by +0.5 before mapping, or the caret tracks at
/// half the drag and full deflection sits half a bar past the edge.
pub fn drive_sliders(mut sliders: Query<(&Interaction, &RelativeCursorPosition, &mut UiSlider)>) {
    for (interaction, rel, mut slider) in &mut sliders {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let Some(normalized) = rel.normalized else {
            continue;
        };
        let x = (normalized.x + 0.5).clamp(0.0, 1.0);
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

/// Keep fill width, hover border, and value label in sync with each slider.
pub fn update_slider_visuals(
    mut sliders: Query<(&UiSlider, &Interaction, &mut BorderColor, &Children)>,
    mut fills: Query<&mut Node, With<SliderFill>>,
    mut labels: Query<(&SliderValueText, &mut Text)>,
    value_sliders: Query<&UiSlider>,
) {
    for (slider, interaction, mut border, children) in &mut sliders {
        let target = Val::Percent(slider.fraction() * 100.0);
        for child in children.iter() {
            if let Ok(mut node) = fills.get_mut(child)
                && node.width != target
            {
                node.width = target;
            }
        }
        let border_c = if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            STROKE_BRIGHT
        } else {
            STROKE
        };
        let new_border = BorderColor::all(border_c);
        if border.top != new_border.top {
            *border = new_border;
        }
    }
    for (value_text, mut text) in &mut labels {
        let Ok(slider) = value_sliders.get(value_text.0) else {
            continue;
        };
        let formatted = slider.format.format(slider.value);
        if **text != formatted {
            **text = formatted;
        }
    }
}
