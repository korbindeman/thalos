//! Top-centre flight-configuration cluster: the flap lever detent and the
//! brakes latch, as two **clickable** pills under the atmosphere readout
//! (same toggle-button treatment as the nav panel's SAS/RCS buttons).
//!
//! - Clicking **FLAPS** cycles the lever one detent (UP → T/O → LDG → UP);
//!   the `F`/`R` keys remain the stepwise non-wrapping path.
//! - Clicking **BRAKES** toggles the latch, exactly like `B`.
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
use crate::flight_config::{FLAP_DETENTS, FlightConfig};
use crate::hud::HudPanel;
use crate::hud::nav_panel::{apply_button_colors, nav_button_colors};
use crate::hud::theme::HudTheme;
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

    // A clickable pill: a flex child of the centring row (NOT absolute, or
    // the two stack at the row origin), styled like the nav-assist buttons.
    let pill_node = || Node {
        border: UiRect::all(Val::Px(1.0)),
        border_radius: BorderRadius::all(Val::Px(3.0)),
        padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        ..default()
    };
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
        p.spawn((
            Button,
            pill_node(),
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            Visibility::Hidden,
            FlapsPill,
            Name::new("FlightConfigFlapsButton"),
        ))
        .with_children(|p| {
            p.spawn((pill_text(&theme, "FLAPS UP"), FlapsText));
        });

        p.spawn((
            Button,
            pill_node(),
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            Visibility::Hidden,
            BrakesPill,
            Name::new("FlightConfigBrakesButton"),
        ))
        .with_children(|p| {
            p.spawn((pill_text(&theme, "BRAKES"), BrakesText));
        });
    });
}

/// Click handling: FLAPS cycles the lever one detent (wrapping), BRAKES
/// toggles the latch — the same state the `F`/`R` and `B` keys drive
/// ([`crate::flight_config`], `local_physics::toggle_parking_brake`).
pub(super) fn handle_clicks(
    flaps: Query<&Interaction, (Changed<Interaction>, With<FlapsPill>)>,
    brakes: Query<&Interaction, (Changed<Interaction>, With<BrakesPill>)>,
    mut config: ResMut<FlightConfig>,
    mut brake: ResMut<ParkingBrake>,
) {
    for interaction in &flaps {
        if matches!(interaction, Interaction::Pressed) {
            config.flap_setting = (config.flap_setting + 1) % (FLAP_DETENTS + 1);
        }
    }
    for interaction in &brakes {
        if matches!(interaction, Interaction::Pressed) {
            brake.engaged = !brake.engaged;
        }
    }
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
        (&mut Visibility, &Interaction, &mut BorderColor, &mut BackgroundColor),
        (With<FlapsPill>, Without<FlightConfigRow>, Without<BrakesPill>),
    >,
    mut brakes_pill_q: Query<
        (&mut Visibility, &Interaction, &mut BorderColor, &mut BackgroundColor),
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

    if let Ok(mut row_vis) = row_q.single_mut() {
        set_visibility(&mut row_vis, has_flaps || has_brakes);
    }

    if let Ok((mut vis, interaction, mut border, mut bg)) = flaps_pill_q.single_mut() {
        set_visibility(&mut vis, has_flaps);
        if has_flaps {
            let active = flight_config.flap_setting > 0;
            let (border_color, bg_color) =
                nav_button_colors(&theme, active, true, false, interaction);
            apply_button_colors(&mut border, &mut bg, border_color, bg_color);
        }
    }

    if let Ok((mut vis, interaction, mut border, mut bg)) = brakes_pill_q.single_mut() {
        set_visibility(&mut vis, has_brakes);
        if has_brakes {
            let (border_color, bg_color) =
                nav_button_colors(&theme, brake.engaged, true, false, interaction);
            apply_button_colors(&mut border, &mut bg, border_color, bg_color);
        }
    }

    if has_flaps && let Ok((mut text, mut color)) = flaps_text_q.single_mut() {
        let target = flight_config.flap_setting as f64 / FLAP_DETENTS.max(1) as f64;
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

fn set_visibility(visibility: &mut Visibility, shown: bool) {
    let target = if shown {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    if *visibility != target {
        *visibility = target;
    }
}
