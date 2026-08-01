//! Per-stage Δv / fuel readout, derived from decoupler position (same pure
//! `stage_summaries` the HUD staging panel uses). Read-only — stages
//! reorder by moving decouplers in the build.

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use thalos_game_state::units::format;
use thalos_shipyard::Resource;
use thalos_shipyard::flyability::FlyabilitySeverity;

use thalos_ui::{self as ui, SPACE_XS, ScrollableColumn, UiTheme, spawn_heading, tokens};

use super::EditorStatsCache;

#[derive(Component)]
pub(super) struct StagingContent;

pub(super) fn spawn(root: &mut ChildSpawnerCommands<'_>, theme: &UiTheme) {
    root.spawn((
        Node {
            right: Val::Px(320.0),
            top: Val::Px(64.0),
            width: Val::Px(176.0),
            max_height: Val::Percent(62.0),
            ..ui::floating_panel_node()
        },
        theme.glass(),
        Interaction::None,
        Name::new("ShipyardStaging"),
    ))
    .with_children(|panel| {
        spawn_heading(panel, theme, "STAGING", false);
        panel.spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(SPACE_XS + 1.0),
                overflow: Overflow::scroll_y(),
                ..default()
            },
            ScrollPosition::default(),
            RelativeCursorPosition::default(),
            Interaction::None,
            ScrollableColumn,
            StagingContent,
        ));
    });
}

/// Rebuild the stage cards when the summaries change. The digest keeps the
/// rebuild cheap: identical formatted output ⇒ no churn.
pub(super) fn rebuild_staging(
    mut commands: Commands,
    cache: Res<EditorStatsCache>,
    theme: Res<UiTheme>,
    units: Res<thalos_game_state::units::UnitsSettings>,
    content: Query<(Entity, Option<&Children>), With<StagingContent>>,
    mut shown_digest: Local<String>,
) {
    // The editor is not a flight instrument: it follows the global switch.
    // No extra change detection is needed — `shown_digest` is built from the
    // formatted strings, so a units change flips the digest and forces a
    // rebuild by itself.
    let system = units.system_for(thalos_game_state::units::UnitDomain::General);
    // (stage label, Δv label, fuel label, resource bars (name, fraction))
    let mut rows: Vec<(String, String, Vec<(String, f32)>)> = Vec::new();
    let header;
    match &cache.staging {
        Some(Ok(summaries)) if !summaries.is_empty() => {
            let total_dv: f64 = summaries
                .iter()
                .map(|s| s.delta_v_m_s)
                .sum::<f64>()
                .max(0.0);
            header = format!("TOTAL Δv {}", format::delta_v(total_dv, system));
            for s in summaries {
                let dv = if s.has_engine {
                    format::delta_v(s.delta_v_m_s, system)
                } else {
                    "drop only".into()
                };
                let mut bars = Vec::new();
                for res in Resource::MASS_BEARING {
                    let Some(totals) = s.resources.get(&res) else {
                        continue;
                    };
                    if totals.capacity <= 0.0 && totals.amount <= 0.0 {
                        continue;
                    }
                    let frac = if totals.capacity > 0.0 {
                        (totals.amount / totals.capacity).clamp(0.0, 1.0) as f32
                    } else {
                        0.0
                    };
                    bars.push((
                        format!(
                            "{} {}",
                            res.display_name(),
                            format::mass_large(totals.mass_kg, system)
                        ),
                        frac,
                    ));
                }
                rows.push((format!("STAGE {}", s.number), dv, bars));
            }
        }
        Some(Ok(_)) => header = "(no stages)".into(),
        Some(Err(e)) => header = format!("staging error: {e}"),
        None => header = "(no ship)".into(),
    }

    // Faults sit under the stage cards, in the panel the player is
    // already reading to understand staging — not behind a click on
    // LAUNCH, which would make the refusal a surprise.
    let faults: Vec<(bool, String)> = match &cache.flyability {
        Some(Ok(findings)) => findings
            .iter()
            .map(|finding| {
                (
                    finding.severity == FlyabilitySeverity::Blocking,
                    finding.message(),
                )
            })
            .collect(),
        _ => Vec::new(),
    };

    let digest = format!("{header}|{rows:?}|{faults:?}");
    if *shown_digest == digest {
        return;
    }
    *shown_digest = digest;

    let Ok((content_entity, children)) = content.single() else {
        return;
    };
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }

    let theme = theme.clone();
    commands.entity(content_entity).with_children(|c| {
        let mut head = theme.mono(header);
        head.1.font_size = FontSize::Px(10.0);
        c.spawn(head);
        for (stage, dv, bars) in rows {
            c.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(3.0),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(ui::RADIUS_CTRL)),
                    padding: UiRect::axes(Val::Px(7.0), Val::Px(5.0)),
                    ..default()
                },
                BackgroundColor(tokens::FILL_HOVER),
                BorderColor::all(tokens::STROKE),
            ))
            .with_children(|card| {
                card.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                })
                .with_children(|head| {
                    let mut stage_text = theme.mono(stage);
                    stage_text.1.font_size = FontSize::Px(10.0);
                    stage_text.2 = TextColor(tokens::ACCENT);
                    head.spawn(stage_text);
                    let mut dv_text = theme.mono(dv);
                    dv_text.1.font_size = FontSize::Px(10.0);
                    head.spawn(dv_text);
                });
                for (label, frac) in bars {
                    card.spawn((
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(4.0),
                            border_radius: BorderRadius::all(Val::Px(2.0)),
                            overflow: Overflow::clip(),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.10)),
                    ))
                    .with_children(|bar| {
                        bar.spawn((
                            Node {
                                width: Val::Percent(frac * 100.0),
                                height: Val::Percent(100.0),
                                ..default()
                            },
                            BackgroundColor(Color::srgba(0.42, 0.74, 0.88, 0.9)),
                        ));
                    });
                    let mut label_text = theme.faint(label);
                    label_text.1.font_size = FontSize::Px(9.0);
                    card.spawn(label_text);
                }
            });
        }

        // Blocking faults get the danger colour and a left rule; warnings get
        // the accent. Both wrap, because the message names the fix and a
        // truncated fix is no fix.
        for (blocking, message) in faults {
            let colour = if blocking {
                tokens::DANGER
            } else {
                tokens::ACCENT
            };
            c.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    border: UiRect::left(Val::Px(2.0)),
                    padding: UiRect::axes(Val::Px(6.0), Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(tokens::FILL_REST),
                BorderColor::all(colour),
            ))
            .with_children(|row| {
                let mut text = theme.mono(message);
                text.1.font_size = FontSize::Px(9.0);
                text.2 = TextColor(colour);
                row.spawn(text);
            });
        }
    });
}
