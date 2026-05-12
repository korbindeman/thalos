//! Bottom-right HUD panel: vacuum Δv estimate + per-reactant fuel bars.
//! Staging breakdown TBD (per-stage rows).

use bevy::prelude::*;
use thalos_shipyard::{
    DeltaVEnvironment, DeltaVInputs, PartResources, Resource, aggregate_resource_totals,
    estimate_delta_v,
};

use crate::fuel::ActivePropulsion;
use crate::hud::HudPanel;
use crate::hud::format;
use crate::hud::theme::{HudTheme, emphasis, label, panel_frame, panel_node};

#[derive(Component)]
pub(super) struct DeltaVText;

#[derive(Component)]
pub(super) struct ResourceRow {
    resource: Resource,
}

#[derive(Component)]
pub(super) struct ResourceBarFill {
    resource: Resource,
}

#[derive(Component)]
pub(super) struct ResourceAmountText {
    resource: Resource,
}

const BAR_HEIGHT: f32 = 8.0;

fn short_label(resource: Resource) -> &'static str {
    match resource {
        Resource::Methane => "CH4",
        Resource::Lox => "Ox",
        Resource::Hydrogen => "LH2",
        Resource::Electricity => "EC",
    }
}

pub fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    let mut root = panel_node();
    root.right = Val::Px(20.0);
    root.bottom = Val::Px(20.0);
    root.min_width = Val::Px(260.0);
    root.align_items = AlignItems::Stretch;
    root.row_gap = Val::Px(6.0);

    let (bg, border) = panel_frame(&theme);
    commands
        .spawn((root, bg, border, HudPanel, Name::new("HudDeltaV")))
        .with_children(|p| {
            p.spawn(Node {
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::End,
                row_gap: Val::Px(2.0),
                ..default()
            })
            .with_children(|head| {
                head.spawn(label(&theme, "VACUUM Δv"));
                head.spawn((emphasis(&theme, "—"), DeltaVText));
            });

            for &resource in &Resource::MASS_BEARING {
                spawn_resource_row(p, &theme, resource);
            }
        });
}

fn spawn_resource_row(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, resource: Resource) {
    parent
        .spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                display: Display::None,
                ..default()
            },
            ResourceRow { resource },
            Name::new(format!("ResourceRow_{}", short_label(resource))),
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
                head.spawn((label(theme, "—"), ResourceAmountText { resource }));
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
                    ResourceBarFill { resource },
                ));
            });
        });
}

#[allow(clippy::type_complexity)]
pub fn update(
    active_propulsion: Res<ActivePropulsion>,
    resources_q: Query<&PartResources>,
    theme: Res<HudTheme>,
    mut dv_q: Query<(&mut Text, &mut TextColor), (With<DeltaVText>, Without<ResourceAmountText>)>,
    mut row_q: Query<(&ResourceRow, &mut Node), Without<ResourceBarFill>>,
    mut fill_q: Query<(&ResourceBarFill, &mut Node), Without<ResourceRow>>,
    mut amount_q: Query<(&ResourceAmountText, &mut Text), Without<DeltaVText>>,
) {
    let resource_totals = aggregate_resource_totals(resources_q.iter());
    let dv = estimate_delta_v(
        DeltaVEnvironment::Vacuum,
        DeltaVInputs {
            dry_mass_kg: active_propulsion.dry_mass_kg,
            wet_mass_kg: active_propulsion.wet_mass_kg,
            total_thrust_n: active_propulsion.total_thrust_n,
            mass_flow_kg_per_s: active_propulsion.mass_flow_kg_per_s,
            power_draw_kw: active_propulsion.power_draw_kw,
            reactant_fractions: &active_propulsion.reactant_fractions,
            resources: &resource_totals,
        },
    );

    if let Ok((mut t, mut color)) = dv_q.single_mut() {
        let s = format::delta_v(dv.delta_v_m_per_s);
        if t.0 != s {
            t.0 = s;
        }
        let warn = active_propulsion.total_thrust_n > 0.0 && dv.delta_v_m_per_s <= 1.0;
        let want = if warn {
            theme.text_warn
        } else {
            theme.text_primary
        };
        if color.0 != want {
            color.0 = want;
        }
    }

    for (row, mut node) in row_q.iter_mut() {
        let used = active_propulsion
            .reactant_fractions
            .get(&row.resource)
            .copied()
            .unwrap_or(0.0)
            > 0.0;
        let target = if used { Display::Flex } else { Display::None };
        if node.display != target {
            node.display = target;
        }
    }

    for (fill, mut node) in fill_q.iter_mut() {
        let totals = resource_totals
            .get(&fill.resource)
            .copied()
            .unwrap_or_default();
        let pct = if totals.capacity > 0.0 {
            ((totals.amount / totals.capacity).clamp(0.0, 1.0) * 100.0) as f32
        } else {
            0.0
        };
        let target = Val::Percent(pct);
        if node.width != target {
            node.width = target;
        }
    }

    for (am, mut t) in amount_q.iter_mut() {
        let totals = resource_totals
            .get(&am.resource)
            .copied()
            .unwrap_or_default();
        let current_kg = totals.mass_kg;
        let max_kg = totals.capacity * am.resource.density_kg_per_unit();
        let s = format::resource_ratio(current_kg, max_kg);
        if t.0 != s {
            t.0 = s;
        }
    }
}
