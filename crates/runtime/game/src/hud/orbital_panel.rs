//! Top-middle HUD: altitude panel centred on screen with the orbital
//! info panel (AP / PE + ORBITAL subtitle) sitting just to its
//! right.
//!
//! Layout trick: the wrapper is a centred Row of three children
//! `[balancer | altitude | orbital]`. The balancer is an invisible
//! Node the same width as the orbital panel, which makes the row
//! symmetric around altitude — so altitude lands exactly at screen
//! centre while orbital floats to its right.
//!
//! Altitude datum (SEA vs GND) auto-picks from regime: GND while the
//! local bubble has a terrain collider attached over the dominant
//! body, SEA otherwise. Clicking the panel installs a sticky override
//! that survives until the next auto-state transition (i.e. when the
//! regime changes on its own, the override clears and auto takes over
//! again).

use std::f64::consts::{PI, TAU};

use bevy::prelude::*;
use thalos_physics_canonical::orbital_math::{OsculatingElements, cartesian_to_elements};
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry};
use thalos_world::StateVector;

use crate::hud::HudPanel;
use crate::hud::format;
use crate::hud::theme::{HudTheme, emphasis, label, panel_frame, panel_node};
use crate::rendering::{SimulationState, SolarSystemState};
use crate::units_settings::UnitDomain;

use crate::local_physics::PHYSICS_QUERY_TILE_LOD_M;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AltitudeDatum {
    Sea,
    Ground,
}

#[derive(Resource, Default, Debug)]
pub(super) struct AltitudeDisplay {
    /// Sticky user override; cleared when `last_auto` changes.
    override_choice: Option<AltitudeDatum>,
    /// Auto-picked datum from the previous frame, used to detect
    /// regime transitions that should clear the override.
    last_auto: Option<AltitudeDatum>,
    /// The datum + altitude (m) resolved this frame, published for other
    /// HUD consumers (the PFD altitude tape). **Sole writer:** [`update`].
    pub(super) resolved: Option<(AltitudeDatum, f64)>,
}

#[derive(Component)]
pub(super) struct AltitudePanel;

#[derive(Component)]
pub(super) struct AltitudeText;

#[derive(Component)]
pub(super) struct AnchorBodyText;

#[derive(Component)]
pub(super) struct ApoapsisAltText;

#[derive(Component)]
pub(super) struct ApoapsisTimeText;

#[derive(Component)]
pub(super) struct PeriapsisAltText;

#[derive(Component)]
pub(super) struct PeriapsisTimeText;

/// Width of the orbital-info panel. The left balancer matches this so
/// the row is symmetric around the altitude panel.
const ORBITAL_PANEL_WIDTH: f32 = 276.0;

pub fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    let wrapper = Node {
        position_type: PositionType::Absolute,
        top: Val::Px(16.0),
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Row,
        justify_content: JustifyContent::Center,
        align_items: AlignItems::FlexStart,
        column_gap: Val::Px(10.0),
        ..default()
    };

    commands
        .spawn((wrapper, Name::new("HudTopMidWrapper")))
        .with_children(|p| {
            spawn_balancer(p);
            spawn_altitude(p, &theme);
            spawn_orbital_info(p, &theme);
        });
}

fn spawn_balancer(p: &mut ChildSpawnerCommands<'_>) {
    p.spawn((
        Node {
            width: Val::Px(ORBITAL_PANEL_WIDTH),
            ..default()
        },
        Name::new("HudTopMidBalancer"),
    ));
}

fn spawn_altitude(p: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let mut root = panel_node();
    root.position_type = PositionType::Relative;
    root.min_width = Val::Px(160.0);
    root.align_items = AlignItems::Center;
    let (bg, border) = panel_frame(theme);

    p.spawn((
        Button,
        root,
        bg,
        border,
        Interaction::None,
        AltitudePanel,
        HudPanel,
        Name::new("HudAltitude"),
    ))
    .with_children(|c| {
        c.spawn((label(theme, "—"), AnchorBodyText));
        c.spawn((emphasis(theme, "—"), AltitudeText));
        c.spawn(Node {
            height: Val::Px(2.0),
            ..default()
        });
        c.spawn((
            Text::new("ALTITUDE"),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(11.0),
                ..default()
            },
            TextColor(theme.text_subtitle),
            Node {
                align_self: AlignSelf::Center,
                ..default()
            },
        ));
    });
}

fn spawn_orbital_info(p: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let mut root = panel_node();
    root.position_type = PositionType::Relative;
    root.width = Val::Px(ORBITAL_PANEL_WIDTH);
    let (bg, border) = panel_frame(theme);

    p.spawn((root, bg, border, HudPanel, Name::new("HudOrbitalInfo")))
        .with_children(|c| {
            // AP row.
            c.spawn(row_node()).with_children(|row| {
                row.spawn(label_cell(theme, "AP"));
                row.spawn((value_cell(theme), ApoapsisAltText));
                row.spawn((countdown_cell(theme), ApoapsisTimeText));
            });
            // PE row.
            c.spawn(row_node()).with_children(|row| {
                row.spawn(label_cell(theme, "PE"));
                row.spawn((value_cell(theme), PeriapsisAltText));
                row.spawn((countdown_cell(theme), PeriapsisTimeText));
            });
            c.spawn(Node {
                height: Val::Px(2.0),
                ..default()
            });
            c.spawn((
                Text::new("ORBITAL"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(11.0),
                    ..default()
                },
                TextColor(theme.text_subtitle),
                Node {
                    align_self: AlignSelf::Center,
                    ..default()
                },
            ));
        });
}

fn row_node() -> Node {
    Node {
        flex_direction: FlexDirection::Row,
        column_gap: Val::Px(8.0),
        align_items: AlignItems::Center,
        ..default()
    }
}

fn label_cell(theme: &HudTheme, label_text: &str) -> impl Bundle {
    (
        Text::new(label_text.to_string()),
        TextFont {
            font: theme.font.clone(),
            font_size: FontSize::Px(13.0),
            ..default()
        },
        TextColor(theme.text_label_alt),
        Node {
            min_width: Val::Px(23.0),
            ..default()
        },
    )
}

fn value_cell(theme: &HudTheme) -> impl Bundle {
    (
        Text::new("—"),
        TextFont {
            font: theme.font.clone(),
            font_size: FontSize::Px(13.0),
            ..default()
        },
        TextColor(theme.text_primary),
        Node {
            min_width: Val::Px(98.0),
            ..default()
        },
    )
}

fn countdown_cell(theme: &HudTheme) -> impl Bundle {
    (
        Text::new("—"),
        TextFont {
            font: theme.font.clone(),
            font_size: FontSize::Px(12.0),
            ..default()
        },
        TextColor(theme.text_dim),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn update(
    sim: Res<SimulationState>,
    solar_system: Res<SolarSystemState>,
    active: Res<ActiveLocalBubble>,
    height_sources: Res<HeightSourceRegistry>,
    theme: Res<HudTheme>,
    units: Res<crate::units_settings::UnitsSettings>,
    mut display: ResMut<AltitudeDisplay>,
    mut alt_q: Query<&mut Text, With<AltitudeText>>,
    mut anchor_q: Query<(&mut Text, &mut TextColor), (With<AnchorBodyText>, Without<AltitudeText>)>,
    mut ap_alt_q: Query<
        &mut Text,
        (
            With<ApoapsisAltText>,
            Without<AltitudeText>,
            Without<AnchorBodyText>,
        ),
    >,
    mut ap_time_q: Query<
        &mut Text,
        (
            With<ApoapsisTimeText>,
            Without<AltitudeText>,
            Without<AnchorBodyText>,
            Without<ApoapsisAltText>,
        ),
    >,
    mut pe_alt_q: Query<
        &mut Text,
        (
            With<PeriapsisAltText>,
            Without<AltitudeText>,
            Without<AnchorBodyText>,
            Without<ApoapsisAltText>,
            Without<ApoapsisTimeText>,
        ),
    >,
    mut pe_time_q: Query<
        &mut Text,
        (
            With<PeriapsisTimeText>,
            Without<AltitudeText>,
            Without<AnchorBodyText>,
            Without<ApoapsisAltText>,
            Without<ApoapsisTimeText>,
            Without<PeriapsisAltText>,
        ),
    >,
) {
    let ship = sim.simulation.ship_state();
    let anchor_id = sim.simulation.dominant_body();
    let body = &sim.simulation.bodies()[anchor_id];
    let Some(states) = solar_system.states.as_deref() else {
        return;
    };
    let Some(body_state) = states.get(anchor_id) else {
        return;
    };
    let rel = StateVector {
        position: ship.position - body_state.position,
        velocity: ship.velocity - body_state.velocity,
    };

    let asl_m = rel.position.length() - body.radius_m;
    let elements = cartesian_to_elements(rel, body.gm);

    // AP/PE are radius-relative (sea-level / datum), independent of GND/SEA toggle.
    let (ap_str, pe_str) = match elements {
        Some(el) => {
            let ap = if el.apoapsis_m.is_finite() {
                format::altitude(el.apoapsis_m - body.radius_m, units.system_for(UnitDomain::General))
            } else {
                "—".to_string()
            };
            let pe = format::altitude(el.periapsis_m - body.radius_m, units.system_for(UnitDomain::General));
            (ap, pe)
        }
        None => ("—".to_string(), "—".to_string()),
    };

    let (ap_time, pe_time) = match elements
        .as_ref()
        .and_then(|el| time_to_apsides(el, body.gm))
    {
        Some((to_apo, to_peri)) => (format::countdown(to_apo), format::countdown(to_peri)),
        None => ("—".to_string(), "—".to_string()),
    };

    // Auto-pick: GND while the local bubble has terrain attached over
    // the dominant body and we have a height source to sample. Falls
    // back to SEA otherwise (deep space, transitional gaps, missing bake).
    let height_source = height_sources.get(anchor_id);
    let bubble_grounded = active
        .bubble
        .as_ref()
        .is_some_and(|b| b.body_id == anchor_id && b.terrain_entity.is_some());
    let auto = if bubble_grounded && height_source.is_some() {
        AltitudeDatum::Ground
    } else {
        AltitudeDatum::Sea
    };

    // Clear sticky override on auto-state transition.
    if let Some(prev) = display.last_auto
        && prev != auto
    {
        display.override_choice = None;
    }
    display.last_auto = Some(auto);

    // Honor override unless GND was chosen with no height source available.
    let chosen = display.override_choice.unwrap_or(auto);
    let chosen = match chosen {
        AltitudeDatum::Ground if height_source.is_none() => AltitudeDatum::Sea,
        other => other,
    };

    let (alt_value, datum_label, datum_color) = match chosen {
        AltitudeDatum::Sea => (asl_m, "SEA", theme.text_datum_sea),
        AltitudeDatum::Ground => {
            let height_source = height_source
                .as_ref()
                .expect("height source presence checked above");
            let position_body =
                body_state.orientation.inverse() * (ship.position - body_state.position);
            let dir = position_body.try_normalize();
            let agl = match dir {
                Some(d) => {
                    let terrain_h = height_source
                        .sample_height_m(d.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
                        .unwrap_or(0.0) as f64;
                    position_body.length() - body.radius_m - terrain_h
                }
                None => asl_m,
            };
            (agl, "GND", theme.text_datum_gnd)
        }
    };
    display.resolved = Some((chosen, alt_value));

    set_text(&mut alt_q, format::altitude(alt_value, units.system_for(UnitDomain::General)));

    let anchor_str = format!("{} {}", body.name, datum_label);
    if let Ok((mut text, mut color)) = anchor_q.single_mut() {
        if text.0 != anchor_str {
            text.0 = anchor_str;
        }
        if color.0 != datum_color {
            color.0 = datum_color;
        }
    }

    set_text(&mut ap_alt_q, ap_str);
    set_text(&mut ap_time_q, ap_time);
    set_text(&mut pe_alt_q, pe_str);
    set_text(&mut pe_time_q, pe_time);
}

pub fn handle_click(
    interactions: Query<&Interaction, (With<AltitudePanel>, Changed<Interaction>)>,
    mut display: ResMut<AltitudeDisplay>,
) {
    for interaction in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        // Toggle from whatever is currently displayed (override if any,
        // otherwise the last auto pick, otherwise SEA as a safe default).
        let current = display
            .override_choice
            .or(display.last_auto)
            .unwrap_or(AltitudeDatum::Sea);
        display.override_choice = Some(match current {
            AltitudeDatum::Sea => AltitudeDatum::Ground,
            AltitudeDatum::Ground => AltitudeDatum::Sea,
        });
    }
}

fn set_text<F: bevy::ecs::query::QueryFilter>(query: &mut Query<&mut Text, F>, new_value: String) {
    if let Ok(mut t) = query.single_mut()
        && t.0 != new_value
    {
        t.0 = new_value;
    }
}

/// Time-until-next-apoapsis and time-until-next-periapsis for an
/// elliptic orbit, in seconds. Returns `None` for non-elliptic orbits.
fn time_to_apsides(elements: &OsculatingElements, mu: f64) -> Option<(f64, f64)> {
    let e = elements.eccentricity;
    if e >= 1.0 {
        return None;
    }
    let a = elements.semi_major_axis_m;
    if a <= 0.0 || !a.is_finite() {
        return None;
    }
    let nu = elements.true_anomaly_rad;

    // True anomaly → eccentric anomaly.
    let half_nu = nu * 0.5;
    let e_anomaly =
        2.0 * ((1.0 - e).sqrt() * half_nu.sin()).atan2((1.0 + e).sqrt() * half_nu.cos());

    // Eccentric → mean anomaly (Kepler's eqn).
    let m = e_anomaly - e * e_anomaly.sin();

    let n = (mu / a.powi(3)).sqrt();
    let period = TAU / n;

    // Time since last periapsis (positive, wrapped into [0, period)).
    let t_since_peri = (m / n).rem_euclid(period);

    // Next periapsis is `period - t_since_peri` from now. At t_since=0 the
    // next periapsis is one full period away.
    let t_to_peri = period - t_since_peri;
    // Apoapsis happens at t_since = period/2; wrap if we've already passed it.
    let t_to_apo = (PI / n - t_since_peri).rem_euclid(period);

    Some((t_to_apo, t_to_peri))
}
