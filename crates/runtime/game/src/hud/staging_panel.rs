//! Bottom-right HUD: a KSP-style staging stack — one bordered card per
//! stage, each showing the stage number, its vacuum Δv, the propellant it
//! burns, and a per-reactant fuel bar for that stage's section.
//!
//! Data comes from [`StagingSummaries`] (sole writer
//! `staging::publish_staging_summaries`). Cards stack with the active
//! (currently-burning) stage at the bottom via [`FlexDirection::ColumnReverse`];
//! the active card is accent-highlighted, and nothing is highlighted before
//! the first stage is fired. Decoupler-only "drop" stages show no Δv. Nothing
//! renders for vessels without a staging plan (e.g. EVA).
//!
//! Cards and their reactant bars are pre-spawned (up to [`MAX_STAGE_CARDS`] ×
//! the mass-bearing resources) and shown/hidden as the live stage count and
//! per-stage reactants change — no per-frame spawn/despawn churn. The three
//! `&mut Node` queries in [`update`] carry `Without` filters so their access
//! is provably disjoint at schedule-build time.

use bevy::prelude::*;
use thalos_shipyard::Resource;

use crate::hud::HudPanel;
use crate::hud::format;
use crate::hud::theme::{HudTheme, emphasis, label};
use crate::staging::StagingSummaries;
use crate::units_settings::UnitDomain;

/// Pre-spawned stage cards. A rocket rarely exceeds this many stages.
const MAX_STAGE_CARDS: usize = 8;
const BAR_HEIGHT: f32 = 8.0;

#[derive(Component)]
pub(super) struct StageCard {
    index: usize,
}

#[derive(Component)]
pub(super) struct StageText {
    index: usize,
    field: StageField,
}

#[derive(Clone, Copy)]
enum StageField {
    Number,
    DeltaV,
    Fuel,
}

#[derive(Component)]
pub(super) struct StageResRow {
    card: usize,
    resource: Resource,
}

#[derive(Component)]
pub(super) struct StageResFill {
    card: usize,
    resource: Resource,
}

#[derive(Component)]
pub(super) struct StageResAmount {
    card: usize,
    resource: Resource,
}

fn short_label(resource: Resource) -> &'static str {
    resource.short_label()
}

pub fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(16.0),
                bottom: Val::Px(16.0),
                // Active stage (card 0) sits at the bottom, like KSP.
                flex_direction: FlexDirection::ColumnReverse,
                align_items: AlignItems::End,
                row_gap: Val::Px(5.0),
                ..default()
            },
            HudPanel,
            Name::new("HudStaging"),
        ))
        .with_children(|p| {
            for index in 0..MAX_STAGE_CARDS {
                spawn_stage_card(p, &theme, index);
            }
        });
}

fn spawn_stage_card(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, index: usize) {
    parent
        .spawn((
            Node {
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                padding: UiRect::axes(Val::Px(9.0), Val::Px(5.0)),
                min_width: Val::Px(178.0),
                row_gap: Val::Px(3.0),
                display: Display::None,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            StageCard { index },
            Name::new(format!("StageCard{index}")),
        ))
        .with_children(|card| {
            card.spawn(Node {
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Baseline,
                column_gap: Val::Px(10.0),
                ..default()
            })
            .with_children(|head| {
                head.spawn((
                    label(theme, "—"),
                    StageText {
                        index,
                        field: StageField::Number,
                    },
                ));
                head.spawn((
                    label(theme, ""),
                    StageText {
                        index,
                        field: StageField::Fuel,
                    },
                ));
            });

            card.spawn((
                emphasis(theme, "—"),
                StageText {
                    index,
                    field: StageField::DeltaV,
                },
            ));

            for &resource in &Resource::MASS_BEARING {
                spawn_resource_bar(card, theme, index, resource);
            }
        });
}

fn spawn_resource_bar(
    card: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    card_index: usize,
    resource: Resource,
) {
    card.spawn((
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(2.0),
            display: Display::None,
            ..default()
        },
        StageResRow {
            card: card_index,
            resource,
        },
        Name::new(format!("StageCard{card_index}_{}", short_label(resource))),
    ))
    .with_children(|row| {
        row.spawn(Node {
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Baseline,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|head| {
            head.spawn(label(theme, short_label(resource)));
            head.spawn((
                label(theme, "—"),
                StageResAmount {
                    card: card_index,
                    resource,
                },
            ));
        });

        row.spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(BAR_HEIGHT),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(2.0)),
                ..default()
            },
            BackgroundColor(theme.panel_bg_alt),
            BorderColor::all(theme.panel_border),
        ))
        .with_children(|bar| {
            bar.spawn((
                Node {
                    width: Val::Percent(0.0),
                    height: Val::Percent(100.0),
                    ..default()
                },
                BackgroundColor(theme.text_subtitle),
                StageResFill {
                    card: card_index,
                    resource,
                },
            ));
        });
    });
}

#[allow(clippy::type_complexity)]
pub fn update(
    summaries: Res<StagingSummaries>,
    theme: Res<HudTheme>,
    units: Res<crate::units_settings::UnitsSettings>,
    mut card_q: Query<
        (
            &StageCard,
            &mut Node,
            &mut BorderColor,
            &mut BackgroundColor,
        ),
        (Without<StageResRow>, Without<StageResFill>),
    >,
    mut text_q: Query<(&StageText, &mut Text, &mut TextColor), Without<StageResAmount>>,
    mut row_q: Query<(&StageResRow, &mut Node), (Without<StageCard>, Without<StageResFill>)>,
    mut fill_q: Query<(&StageResFill, &mut Node), (Without<StageCard>, Without<StageResRow>)>,
    mut amount_q: Query<(&StageResAmount, &mut Text), Without<StageText>>,
) {
    let stages = &summaries.0;

    for (card, mut node, mut border, mut bg) in card_q.iter_mut() {
        let Some(stage) = stages.get(card.index) else {
            if node.display != Display::None {
                node.display = Display::None;
            }
            continue;
        };
        if node.display != Display::Flex {
            node.display = Display::Flex;
        }
        let (want_border, want_bg) = if stage.active {
            (theme.text_accent, theme.panel_bg_alt)
        } else {
            (theme.panel_border, theme.panel_bg)
        };
        *border = BorderColor::all(want_border);
        if bg.0 != want_bg {
            bg.0 = want_bg;
        }
    }

    for (cell, mut text, mut color) in text_q.iter_mut() {
        let Some(stage) = stages.get(cell.index) else {
            continue;
        };
        let (s, want_color) = match cell.field {
            StageField::Number => (
                format!("STAGE {}", stage.number),
                if stage.active {
                    theme.text_accent
                } else {
                    theme.text_dim
                },
            ),
            StageField::DeltaV => {
                let s = if stage.has_engine {
                    format::delta_v(stage.delta_v_m_s, units.system_for(UnitDomain::General))
                } else {
                    "drop only".to_string()
                };
                (s, theme.text_primary)
            }
            StageField::Fuel => {
                let s = if stage.fuel_kg > 0.0 {
                    format::mass(stage.fuel_kg, units.system_for(UnitDomain::General))
                } else {
                    "—".to_string()
                };
                (s, theme.text_dim)
            }
        };
        if text.0 != s {
            text.0 = s;
        }
        if color.0 != want_color {
            color.0 = want_color;
        }
    }

    for (row, mut node) in row_q.iter_mut() {
        let present = stages
            .get(row.card)
            .and_then(|s| s.resources.get(&row.resource))
            .map(|t| t.capacity > 0.0 || t.amount > 0.0)
            .unwrap_or(false);
        let target = if present {
            Display::Flex
        } else {
            Display::None
        };
        if node.display != target {
            node.display = target;
        }
    }

    for (fill, mut node) in fill_q.iter_mut() {
        let pct = stages
            .get(fill.card)
            .and_then(|s| s.resources.get(&fill.resource))
            .map(|t| {
                if t.capacity > 0.0 {
                    ((t.amount / t.capacity).clamp(0.0, 1.0) * 100.0) as f32
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        let target = Val::Percent(pct);
        if node.width != target {
            node.width = target;
        }
    }

    for (am, mut text) in amount_q.iter_mut() {
        let s = stages
            .get(am.card)
            .and_then(|s| s.resources.get(&am.resource))
            .map(|t| {
                format::resource_ratio(
                    t.mass_kg,
                    t.capacity * am.resource.density_kg_per_unit(),
                    units.system_for(UnitDomain::General),
                )
            })
            .unwrap_or_else(|| "—".to_string());
        if text.0 != s {
            text.0 = s;
        }
    }
}
