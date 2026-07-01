//! Per-stage Δv / fuel readout, derived from decoupler position (same pure
//! `stage_summaries` the HUD staging panel uses). Read-only — stages
//! reorder by moving decouplers in the build.

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use thalos_shipyard::Resource;
use crate::shipyard_editor::core::{format_delta_v, format_mass_kg};

use crate::hud::theme::{HudTheme, panel_frame};

use super::EditorStatsCache;
use super::widgets::ScrollableColumn;

#[derive(Component)]
pub(super) struct StagingContent;

pub(super) fn spawn(root: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let (bg, border) = panel_frame(theme);
    root.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(320.0),
            top: Val::Px(60.0),
            width: Val::Px(176.0),
            max_height: Val::Percent(62.0),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(4.0)),
            padding: UiRect::axes(Val::Px(10.0), Val::Px(8.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(5.0),
            ..default()
        },
        bg,
        border,
        Interaction::None,
        Name::new("ShipyardStaging"),
    ))
    .with_children(|panel| {
        panel.spawn((
            Text::new("STAGING"),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(12.0),
                ..default()
            },
            TextColor(theme.text_subtitle),
        ));
        panel.spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(5.0),
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
    theme: Res<HudTheme>,
    content: Query<(Entity, Option<&Children>), With<StagingContent>>,
    mut shown_digest: Local<String>,
) {
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
            header = format!("TOTAL Δv {}", format_delta_v(total_dv));
            for s in summaries {
                let dv = if s.has_engine {
                    format_delta_v(s.delta_v_m_s)
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
                        format!("{} {}", res.display_name(), format_mass_kg(totals.mass_kg)),
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

    let digest = format!("{header}|{rows:?}");
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
        c.spawn((
            Text::new(header),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(10.0),
                ..default()
            },
            TextColor(theme.text_primary),
        ));
        for (stage, dv, bars) in rows {
            c.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(3.0),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    padding: UiRect::axes(Val::Px(7.0), Val::Px(5.0)),
                    ..default()
                },
                BackgroundColor(theme.panel_bg_alt),
                BorderColor::all(theme.panel_border),
            ))
            .with_children(|card| {
                card.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                })
                .with_children(|head| {
                    head.spawn((
                        Text::new(stage),
                        TextFont {
                            font: theme.font.clone(),
                            font_size: FontSize::Px(10.0),
                            ..default()
                        },
                        TextColor(theme.text_accent),
                    ));
                    head.spawn((
                        Text::new(dv),
                        TextFont {
                            font: theme.font.clone(),
                            font_size: FontSize::Px(10.0),
                            ..default()
                        },
                        TextColor(theme.text_primary),
                    ));
                });
                for (label, frac) in bars {
                    card.spawn((
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Px(10.0),
                            border: UiRect::all(Val::Px(1.0)),
                            border_radius: BorderRadius::all(Val::Px(2.0)),
                            padding: UiRect::all(Val::Px(1.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(0.02, 0.02, 0.02, 0.9)),
                        BorderColor::all(theme.panel_border),
                    ))
                    .with_children(|bar| {
                        bar.spawn((
                            Node {
                                width: Val::Percent(frac * 100.0),
                                height: Val::Percent(100.0),
                                ..default()
                            },
                            BackgroundColor(theme.text_datum_sea.with_alpha(0.55)),
                        ));
                    });
                    card.spawn((
                        Text::new(label),
                        TextFont {
                            font: theme.font.clone(),
                            font_size: FontSize::Px(8.5),
                            ..default()
                        },
                        TextColor(theme.text_dim),
                    ));
                }
            });
        }
    });
}
