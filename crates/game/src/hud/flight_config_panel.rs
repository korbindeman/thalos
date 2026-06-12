//! Top-centre flight-configuration cluster: the flap lever detent and the
//! brakes latch, as two small pills under the atmosphere readout.
//!
//! **Capability-gated:** each pill appears only when the current craft
//! actually has the system. Flaps show iff the wing aero config derived any
//! flap authority from authored `Flap` windows; brakes show iff the craft has
//! landing-gear wheels (wheel brakes) or `Spoiler` windows (speedbrake). A
//! rocket/capsule shows neither; the panel also stands down outside the local
//! bubble (on rails, where neither system can act) and on EVA.

use bevy::prelude::*;
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::LocalCraftBody;

use crate::aero::ShipAero;
use crate::flight_config::FlightConfig;
use crate::hud::HudPanel;
use crate::hud::theme::{HudTheme, panel_frame, panel_node};
use crate::local_physics::{ParkingBrake, WheelSet};
use crate::rendering::SimulationState;

#[derive(Component)]
pub(super) struct FlightConfigRow;

#[derive(Component)]
pub(super) struct FlapsPill;

#[derive(Component)]
pub(super) struct FlapsText;

#[derive(Component)]
pub(super) struct BrakesPill;

#[derive(Component)]
pub(super) struct BrakesText;

pub(super) fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    // Full-width centring row just below the atmosphere panel (top ≈ 96 +
    // its height), mirroring the EVA pill idiom.
    let row = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(156.0),
                left: Val::Px(0.0),
                width: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                column_gap: Val::Px(8.0),
                ..default()
            },
            Visibility::Hidden,
            FlightConfigRow,
            HudPanel,
            Name::new("HudFlightConfig"),
        ))
        .id();

    let pill_text = |theme: &HudTheme, label: &str| {
        (
            Text::new(label),
            TextFont {
                font: theme.font.clone(),
                font_size: 13.0,
                ..default()
            },
            TextColor(theme.text_dim),
        )
    };

    commands.entity(row).with_children(|p| {
        let mut pill = panel_node();
        pill.padding = UiRect::axes(Val::Px(10.0), Val::Px(5.0));
        let (bg, border) = panel_frame(&theme);
        p.spawn((pill, bg, border, Visibility::Hidden, FlapsPill))
            .with_children(|p| {
                p.spawn((pill_text(&theme, "FLAPS UP"), FlapsText));
            });

        let mut pill = panel_node();
        pill.padding = UiRect::axes(Val::Px(10.0), Val::Px(5.0));
        let (bg, border) = panel_frame(&theme);
        p.spawn((pill, bg, border, Visibility::Hidden, BrakesPill))
            .with_children(|p| {
                p.spawn((pill_text(&theme, "BRAKES"), BrakesText));
            });
    });
}

#[allow(clippy::type_complexity)]
pub(super) fn update(
    sim: Res<SimulationState>,
    flight_config: Res<FlightConfig>,
    brake: Res<ParkingBrake>,
    theme: Res<HudTheme>,
    craft: Query<(&ShipAero, Option<&WheelSet>), With<LocalCraftBody>>,
    mut row_q: Query<
        &mut Visibility,
        (With<FlightConfigRow>, Without<FlapsPill>, Without<BrakesPill>),
    >,
    mut flaps_pill_q: Query<
        &mut Visibility,
        (With<FlapsPill>, Without<FlightConfigRow>, Without<BrakesPill>),
    >,
    mut brakes_pill_q: Query<
        &mut Visibility,
        (With<BrakesPill>, Without<FlightConfigRow>, Without<FlapsPill>),
    >,
    mut flaps_text_q: Query<(&mut Text, &mut TextColor), (With<FlapsText>, Without<BrakesText>)>,
    mut brakes_text_q: Query<(&mut Text, &mut TextColor), (With<BrakesText>, Without<FlapsText>)>,
) {
    // Capability of the *current* craft: flap/spoiler authority from the wing
    // aero config on the bubble body, wheel brakes from its wheel set. No
    // bubble (on rails) or not a ship → no panel.
    let mut has_flaps = false;
    let mut has_brakes = false;
    if sim.simulation.vessel_kind() == VesselKind::Ship
        && let Ok((aero, wheels)) = craft.single()
    {
        has_flaps = aero.config.flap_dcl > 0.0;
        has_brakes = wheels.is_some_and(|w| !w.wheels.is_empty())
            || aero.config.spoiler_dcd > 0.0;
    }

    set_visibility(&mut row_q, has_flaps || has_brakes);
    set_visibility(&mut flaps_pill_q, has_flaps);
    set_visibility(&mut brakes_pill_q, has_brakes);

    if has_flaps && let Ok((mut text, mut color)) = flaps_text_q.single_mut() {
        let target = flight_config.flap_setting as f64
            / crate::flight_config::FLAP_DETENTS.max(1) as f64;
        let in_transit = (flight_config.flap_fraction - target).abs() > 0.02;
        let label = format!("FLAPS {}", flight_config.flap_label());
        let new_color = if in_transit {
            theme.text_warn
        } else if flight_config.flap_setting > 0 {
            theme.text_accent
        } else {
            theme.text_dim
        };
        if text.0 != label {
            text.0 = label;
        }
        if color.0 != new_color {
            color.0 = new_color;
        }
    }

    if has_brakes && let Ok((mut text, mut color)) = brakes_text_q.single_mut() {
        let (label, new_color) = if brake.engaged {
            ("BRAKES ON", theme.text_warn)
        } else {
            ("BRAKES", theme.text_dim)
        };
        if text.0 != label {
            text.0 = label.to_string();
        }
        if color.0 != new_color {
            color.0 = new_color;
        }
    }
}

fn set_visibility<F: bevy::ecs::query::QueryFilter>(
    q: &mut Query<&mut Visibility, F>,
    shown: bool,
) {
    let target = if shown {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut visibility in q.iter_mut() {
        if *visibility != target {
            *visibility = target;
        }
    }
}
