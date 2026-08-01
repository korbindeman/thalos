//! Top-centre flight-configuration cluster: the flap lever and the
//! brakes latch, sitting under the atmosphere readout.
//!
//! - **FLAPS** is a *segmented gate* — three detent segments
//!   `[ UP · T/O · LDG ]` laid out like a flap lever's gate. Clicking a
//!   segment drives the lever **straight to that detent** (one click to any
//!   position, never the wrong direction); the current detent is highlighted
//!   and glows amber while the actuators are still travelling toward it. The
//!   `F`/`R` keys remain the stepwise extend/retract path. (This replaced a
//!   single one-directional cycling pill, whose only motion was
//!   `UP → T/O → LDG → UP`: it couldn't express "retract" and needed two
//!   clicks to clean up after takeoff.)
//! - **BRAKES** stays a single latched toggle pill (it *is* a binary latch),
//!   exactly like `B` — wheel brakes on the ground and spoilers in the air.
//!
//! **Capability-gated:** the flap gate appears only when the wing aero config
//! derived flap authority from authored `Flap` windows; brakes show iff the
//! craft has landing-gear wheels (wheel brakes) or `Spoiler` windows
//! (speedbrake). A rocket/capsule shows neither; the panel also stands down
//! outside the local bubble (on rails, where neither system can act) and on
//! EVA.

use bevy::prelude::*;
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::LocalCraftBody;

use thalos_game_state::flight::ShipAero;
use thalos_game_state::flight::{FLAP_DETENTS, FlightConfig};
use crate::HudPanel;
use crate::nav_panel::{apply_button_colors, nav_button_colors};
use crate::theme::HudTheme;
use thalos_game_state::flight::{GearState, ParkingBrake, WeightOnWheels, WheelSet, set_gear_down};
use thalos_game_state::SimulationState;

#[derive(Component)]
pub(super) struct FlightConfigRow;

/// Container for the flap-lever gate (the static `FLAPS` label + the detent
/// segments). Carries the whole group's capability visibility.
#[derive(Component)]
pub(super) struct FlapsGate;

/// One detent segment button. `detent` is the lever position it selects
/// (0 = UP, `FLAP_DETENTS` = LANDING).
#[derive(Component)]
pub(super) struct FlapsSegment {
    detent: u8,
}

/// The label inside a flap segment, tagged with its detent so the per-segment
/// colouring can find it.
#[derive(Component)]
pub(super) struct FlapsSegmentText {
    detent: u8,
}

#[derive(Component)]
pub(super) struct BrakesPill;

#[derive(Component)]
pub(super) struct BrakesText;

#[derive(Component)]
pub(super) struct GearPill;

#[derive(Component)]
pub(super) struct GearText;

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
                align_items: AlignItems::Center,
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
    // Segments are slightly tighter than a full pill so the three read as one
    // gate rather than three separate buttons.
    let segment_node = || Node {
        border: UiRect::all(Val::Px(1.0)),
        border_radius: BorderRadius::all(Val::Px(3.0)),
        padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        ..default()
    };
    let pill_text = |theme: &HudTheme, label: &str| {
        (
            Text::new(label),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(13.0),
                ..default()
            },
            TextColor(theme.text_dim),
        )
    };

    commands.entity(row).with_children(|p| {
        // Flap gate: a static "FLAPS" label followed by the three detent
        // segments. The gate node carries the capability visibility.
        p.spawn((
            Node {
                align_items: AlignItems::Center,
                column_gap: Val::Px(6.0),
                ..default()
            },
            Visibility::Hidden,
            FlapsGate,
            Name::new("FlightConfigFlapsGate"),
        ))
        .with_children(|p| {
            p.spawn(pill_text(&theme, "FLAPS"));
            // The detent segments, joined by a tight gap.
            p.spawn(Node {
                align_items: AlignItems::Center,
                column_gap: Val::Px(3.0),
                ..default()
            })
            .with_children(|p| {
                for detent in 0..=FLAP_DETENTS {
                    p.spawn((
                        Button,
                        segment_node(),
                        BackgroundColor(theme.panel_bg),
                        BorderColor::all(theme.panel_border),
                        Interaction::None,
                        FlapsSegment { detent },
                        Name::new("FlightConfigFlapSegment"),
                    ))
                    .with_children(|p| {
                        p.spawn((
                            pill_text(&theme, FlightConfig::detent_label(detent)),
                            FlapsSegmentText { detent },
                        ));
                    });
                }
            });
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

        p.spawn((
            Button,
            pill_node(),
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            Visibility::Hidden,
            GearPill,
            Name::new("FlightConfigGearButton"),
        ))
        .with_children(|p| {
            p.spawn((pill_text(&theme, "GEAR DN"), GearText));
        });
    });
}

/// Click handling: a FLAPS segment sets the lever straight to its detent,
/// BRAKES toggles the latch — the same state the `F`/`R` and `B` keys drive
/// (the runtime's `flight_config` module, `local_physics::toggle_parking_brake`).
pub(super) fn handle_clicks(
    flaps: Query<(&Interaction, &FlapsSegment), Changed<Interaction>>,
    brakes: Query<&Interaction, (Changed<Interaction>, With<BrakesPill>)>,
    gear_pill: Query<&Interaction, (Changed<Interaction>, With<GearPill>)>,
    mut config: ResMut<FlightConfig>,
    mut brake: ResMut<ParkingBrake>,
    mut gear: ResMut<GearState>,
    weight_on_wheels: Res<WeightOnWheels>,
) {
    for (interaction, segment) in &flaps {
        if matches!(interaction, Interaction::Pressed) {
            config.flap_setting = segment.detent.min(FLAP_DETENTS);
        }
    }
    for interaction in &brakes {
        if matches!(interaction, Interaction::Pressed) {
            brake.engaged = !brake.engaged;
        }
    }
    for interaction in &gear_pill {
        if matches!(interaction, Interaction::Pressed) {
            // Same interlock as the G key: retraction refused while grounded.
            let target = !gear.down;
            set_gear_down(&mut gear, &weight_on_wheels, target);
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn update(
    sim: Res<SimulationState>,
    flight_config: Res<FlightConfig>,
    brake: Res<ParkingBrake>,
    gear_state: Res<GearState>,
    theme: Res<HudTheme>,
    craft: Query<(&ShipAero, Option<&WheelSet>), With<LocalCraftBody>>,
    mut row_q: Query<
        &mut Visibility,
        (
            With<FlightConfigRow>,
            Without<FlapsGate>,
            Without<BrakesPill>,
        ),
    >,
    mut gate_q: Query<
        &mut Visibility,
        (
            With<FlapsGate>,
            Without<FlightConfigRow>,
            Without<BrakesPill>,
        ),
    >,
    mut segments_q: Query<
        (
            &FlapsSegment,
            &Interaction,
            &mut BorderColor,
            &mut BackgroundColor,
        ),
        Without<BrakesPill>,
    >,
    mut brakes_pill_q: Query<
        (
            &mut Visibility,
            &Interaction,
            &mut BorderColor,
            &mut BackgroundColor,
        ),
        (
            With<BrakesPill>,
            Without<FlightConfigRow>,
            Without<FlapsGate>,
            Without<FlapsSegment>,
        ),
    >,
    mut gear_pill_q: Query<
        (
            &mut Visibility,
            &Interaction,
            &mut BorderColor,
            &mut BackgroundColor,
        ),
        (
            With<GearPill>,
            Without<FlightConfigRow>,
            Without<FlapsGate>,
            Without<FlapsSegment>,
            Without<BrakesPill>,
        ),
    >,
    mut segment_text_q: Query<(&FlapsSegmentText, &mut TextColor), Without<BrakesText>>,
    mut brakes_text_q: Query<
        (&mut Text, &mut TextColor),
        (
            With<BrakesText>,
            Without<FlapsSegmentText>,
            Without<GearText>,
        ),
    >,
    mut gear_text_q: Query<
        (&mut Text, &mut TextColor),
        (
            With<GearText>,
            Without<FlapsSegmentText>,
            Without<BrakesText>,
        ),
    >,
) {
    // Capability of the *current* craft: flap/spoiler authority from the wing
    // aero config on the bubble body, wheel brakes from its wheel set. No
    // bubble (on rails) or not a ship → no panel.
    let mut has_flaps = false;
    let mut has_brakes = false;
    let mut has_gear = false;
    if sim.simulation.vessel_kind() == VesselKind::Ship
        && let Ok((aero, wheels)) = craft.single()
    {
        let has_wheels = wheels.is_some_and(|w| !w.wheels.is_empty());
        has_flaps = aero.config.flap_dcl > 0.0;
        has_brakes = has_wheels || aero.config.spoiler_dcd > 0.0;
        // Retractable gear is offered for any craft that has wheels.
        has_gear = has_wheels;
    }

    if let Ok(mut row_vis) = row_q.single_mut() {
        set_visibility(&mut row_vis, has_flaps || has_brakes || has_gear);
    }

    if let Ok(mut gate_vis) = gate_q.single_mut() {
        set_visibility(&mut gate_vis, has_flaps);
    }

    if has_flaps {
        // The commanded detent is highlighted; while the actuators are still
        // travelling toward it the highlight reads amber ("moving").
        let target = flight_config.flap_setting as f64 / FLAP_DETENTS.max(1) as f64;
        let in_transit = (flight_config.flap_fraction - target).abs() > 0.02;

        for (segment, interaction, mut border, mut bg) in &mut segments_q {
            let active = segment.detent == flight_config.flap_setting;
            let (border_color, bg_color) =
                nav_button_colors(&theme, active, true, false, interaction);
            apply_button_colors(&mut border, &mut bg, border_color, bg_color);
        }

        for (label, mut color) in &mut segment_text_q {
            let new_color = if label.detent != flight_config.flap_setting {
                theme.text_dim
            } else if in_transit {
                theme.text_warn
            } else {
                theme.text_accent
            };
            if color.0 != new_color {
                color.0 = new_color;
            }
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

    // Gear pill: amber while retracted (an in-flight reminder the gear is up),
    // dim when extended. `active` highlights the up state the same way BRAKES
    // highlights engaged.
    if let Ok((mut vis, interaction, mut border, mut bg)) = gear_pill_q.single_mut() {
        set_visibility(&mut vis, has_gear);
        if has_gear {
            let (border_color, bg_color) =
                nav_button_colors(&theme, !gear_state.down, true, false, interaction);
            apply_button_colors(&mut border, &mut bg, border_color, bg_color);
        }
    }

    if has_gear && let Ok((mut text, mut color)) = gear_text_q.single_mut() {
        let (label, new_color) = if gear_state.down {
            ("GEAR DN", theme.text_dim)
        } else {
            ("GEAR UP", theme.text_warn)
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
