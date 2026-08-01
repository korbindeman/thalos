//! Editor top bar: title + ship-name field on the left, live build stats in
//! the centre, and the action rail on the right — build toggles
//! (mirror / snap / horizontal layout), HANGAR (craft load/save overlay),
//! SAVE / NEW housekeeping, the headline ▶ LAUNCH, and EXIT.

use bevy::prelude::*;

use thalos_shipyard::Ship;
use thalos_ui::{
    self as ui, ButtonVariant, SPACE_SM, TextFieldFocus, UiButton, UiTextField, UiTheme,
    spawn_button, spawn_text_field,
};

use crate::core::{BuildOrientation, EditorPart, EditorState, PlacementSnap, SymmetryMode};
use thalos_game_state::context::{ContextHistory, GameContext, back_out};
use thalos_game_state::relaunch::{RelaunchRequest, RelaunchSpec};
use thalos_game_state::scenario::SpawnSituation;

use thalos_shipyard::flyability::{FlyabilitySeverity, blocks_launch};

use super::EditorStatsCache;
use super::hangar::HangarOpen;

#[derive(Component, Clone, Copy, PartialEq)]
pub(super) enum TopBarAction {
    ToggleMirror,
    ToggleSnap,
    ToggleLayout,
    Hangar,
    Launch,
    Save,
    New,
    Exit,
}

#[derive(Component)]
pub(super) struct StatsText;

/// Marker on the ship-name [`UiTextField`].
#[derive(Component)]
pub(super) struct ShipNameField;

pub(super) fn spawn(root: &mut ChildSpawnerCommands<'_>, theme: &UiTheme) {
    root.spawn((
        Node {
            left: Val::Px(12.0),
            right: Val::Px(12.0),
            top: Val::Px(12.0),
            height: Val::Px(44.0),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(SPACE_SM),
            padding: UiRect::horizontal(Val::Px(ui::SPACE_LG)),
            ..ui::floating_panel_node()
        },
        theme.glass(),
        Interaction::None,
        Name::new("ShipyardTopBar"),
    ))
    .with_children(|bar| {
        bar.spawn(theme.title("SHIPYARD"));

        spawn_text_field(
            bar,
            theme,
            UiTextField::new("", "ship name"),
            Val::Px(220.0),
            ShipNameField,
        );

        // Live stats, centred in the leftover space.
        bar.spawn((
            theme.mono_dim(""),
            Node {
                flex_grow: 1.0,
                justify_content: JustifyContent::Center,
                ..default()
            },
            StatsText,
        ));

        for (action, label) in [
            (TopBarAction::ToggleMirror, "MIRROR 2×"),
            (TopBarAction::ToggleSnap, "SNAP 15°"),
            (TopBarAction::ToggleLayout, "HORIZONTAL"),
        ] {
            spawn_button(bar, theme, action, label, ButtonVariant::Ghost, ui::CTRL_H);
        }
        bar.spawn((
            Node {
                width: Val::Px(1.0),
                height: Val::Px(20.0),
                margin: UiRect::horizontal(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(ui::tokens::STROKE),
        ));
        spawn_button(
            bar,
            theme,
            TopBarAction::Hangar,
            "HANGAR",
            ButtonVariant::Ghost,
            ui::CTRL_H,
        );
        spawn_button(
            bar,
            theme,
            TopBarAction::Save,
            "SAVE",
            ButtonVariant::Ghost,
            ui::CTRL_H,
        );
        spawn_button(
            bar,
            theme,
            TopBarAction::New,
            "NEW",
            ButtonVariant::Ghost,
            ui::CTRL_H,
        );
        // LAUNCH is the headline action — fly the current design.
        spawn_button(
            bar,
            theme,
            TopBarAction::Launch,
            "▶  LAUNCH",
            ButtonVariant::Primary,
            ui::CTRL_H,
        );
        spawn_button(
            bar,
            theme,
            TopBarAction::Exit,
            "EXIT",
            ButtonVariant::Bare,
            ui::CTRL_H,
        );
    });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_actions(
    interactions: Query<(&Interaction, &TopBarAction), Changed<Interaction>>,
    mut state: ResMut<EditorState>,
    mut symmetry: ResMut<SymmetryMode>,
    mut snap: ResMut<PlacementSnap>,
    mut orientation: ResMut<BuildOrientation>,
    mut hangar: ResMut<HangarOpen>,
    cache: Res<EditorStatsCache>,
    mut relaunch: ResMut<RelaunchRequest>,
    mut launch_req: ResMut<thalos_game_state::relaunch::SpaceportLaunchRequest>,
    mut next_ctx: Option<ResMut<NextState<GameContext>>>,
    mut history: ResMut<ContextHistory>,
) {
    for (interaction, action) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        match action {
            TopBarAction::ToggleMirror => symmetry.mirror = !symmetry.mirror,
            TopBarAction::ToggleSnap => snap.enabled = !snap.enabled,
            TopBarAction::ToggleLayout => {
                let flipped = !orientation.bypass_change_detection().horizontal;
                orientation.horizontal = flipped;
            }
            TopBarAction::Hangar => {
                hangar.0 = !hangar.0;
                if hangar.0 {
                    // Fresh listing every time the overlay opens.
                    state.refresh_list = true;
                }
            }
            TopBarAction::Launch => {
                let Some(blueprint) = cache.blueprint.clone() else {
                    state.status = "Nothing to launch — build a ship first".into();
                    continue;
                };
                // Refuse a build that cannot work. The bar is
                // "impossible", never "suboptimal" — Δv margin and TWR
                // stay the player's call and the ORBIT preflight's
                // second gate. The findings are already on the staging
                // panel, so this message names the first one rather
                // than being the player's first hint that anything is
                // wrong.
                if let Some(Ok(findings)) = cache.flyability.as_ref()
                    && blocks_launch(findings)
                {
                    let first = findings
                        .iter()
                        .find(|f| f.severity == FlyabilitySeverity::Blocking)
                        .expect("blocks_launch guarantees one");
                    state.status = format!("Cannot launch — {}", first.message());
                    continue;
                }
                // Rebuild the craft into an orbit hold, then drop into the
                // in-world launch-point picker (`base_editor::launch_select`) so
                // the player chooses a runway or a launchpad to launch from. The
                // spaceport is built lazily on the first launch. Persist too.
                state.save_requested = true;
                relaunch.0 = Some(RelaunchSpec {
                    blueprint,
                    situation: SpawnSituation::ShipOrbit,
                });
                launch_req.arm = true;
                // "Launched to fly": clear the return stack and drop to Flight.
                // The launch-select flow then re-enters BaseEditor (the picker)
                // parented to Flight, so placing / cancelling both land in flight
                // — never back in the VAB or hub (`docs/gameplay/ui_flow.md`).
                history.0.clear();
                if let Some(next) = next_ctx.as_mut() {
                    next.set(GameContext::Flight);
                }
                state.status = "Launching…".into();
            }
            TopBarAction::Save => state.save_requested = true,
            TopBarAction::New => {
                // Clear the canvas: deleting the root despawns the whole
                // build (the core's delete path).
                if state.ship_root.is_some() {
                    state.selected = state.ship_root;
                    state.delete_selected = true;
                } else {
                    state.status = "Canvas already empty".into();
                }
            }
            TopBarAction::Exit => {
                // Back out to whatever opened the VAB (the hub, or flight), or
                // Flight when the VAB is the session root (`just game shipyard`).
                if let Some(next) = next_ctx.as_mut()
                    && back_out(next, &mut history).is_none()
                {
                    next.set(GameContext::Flight);
                }
            }
        }
    }
}

/// Render the latched state of the three toggles.
pub(super) fn update_toggle_latches(
    symmetry: Res<SymmetryMode>,
    snap: Res<PlacementSnap>,
    orientation: Res<BuildOrientation>,
    mut buttons: Query<(&TopBarAction, &mut UiButton)>,
) {
    for (action, mut button) in &mut buttons {
        let latched = match action {
            TopBarAction::ToggleMirror => symmetry.mirror,
            TopBarAction::ToggleSnap => snap.enabled,
            TopBarAction::ToggleLayout => orientation.horizontal,
            _ => continue,
        };
        if button.latched != latched {
            button.latched = latched;
        }
    }
}

pub(super) fn update_stats_text(
    cache: Res<EditorStatsCache>,
    units: Res<thalos_game_state::units::UnitsSettings>,
    mut texts: Query<&mut Text, With<StatsText>>,
) {
    let Ok(mut text) = texts.single_mut() else {
        return;
    };
    // The editor is not a flight instrument, so it follows the global switch.
    let system = units.system_for(thalos_game_state::units::UnitDomain::General);
    let line = match (&cache.stats, &cache.staging) {
        (Some(Ok(stats)), staging) => {
            // `.max(0.0)` keeps an engine-less build from reading "Δv -0 m/s"
            // (tiny negative / negative-zero sums from the stage estimate).
            let total_dv: f64 = staging
                .as_ref()
                .and_then(|s| s.as_ref().ok())
                .map(|s| s.iter().map(|st| st.delta_v_m_s).sum::<f64>().max(0.0))
                .unwrap_or(0.0);
            let mut line = format!(
                "MASS {}  ·  Δv {}",
                thalos_game_state::units::format::mass_large(stats.wet_mass_kg(), system),
                thalos_game_state::units::format::delta_v(total_dv, system),
            );
            if stats.wet_mass_kg() > 0.0 && stats.total_thrust_n > 0.0 {
                line.push_str(&format!(
                    "  ·  TWR {:.2}",
                    stats.current_acceleration() / thalos_shipyard::G0
                ));
            }
            if stats.wing_area_m2 > 0.0 {
                line.push_str(&format!(
                    "  ·  WING {}",
                    thalos_game_state::units::format::area(stats.wing_area_m2, system)
                ));
            }
            line
        }
        (Some(Err(e)), _) => format!("stats error: {e}"),
        (None, _) => "EMPTY CANVAS — pick a part from the palette".into(),
    };
    if **text != line {
        **text = line;
    }
}

/// Two-way sync between the name field and the build.
///
/// - **Field → model** on user edits (`Changed<UiTextField>` while focused):
///   writes the live `Ship` entity when one exists, plus the staged
///   `EditorState::ship_name` used for the next root placement.
/// - **Model → field** while not focused, so loads / New / re-opens show the
///   real name without clobbering in-progress typing.
pub(super) fn sync_ship_name(
    focus: Res<TextFieldFocus>,
    mut state: ResMut<EditorState>,
    mut fields: Query<(Entity, &mut UiTextField), With<ShipNameField>>,
    mut ships: Query<&mut Ship, With<EditorPart>>,
) {
    let Ok((field_entity, mut field)) = fields.single_mut() else {
        return;
    };
    let focused = focus.field == Some(field_entity);

    if focused {
        // User is typing: push field → model (value-guarded).
        let name = field.value.clone();
        if let Some(ship_entity) = state.ship_entity
            && let Ok(mut ship) = ships.get_mut(ship_entity)
            && ship.name != name
        {
            ship.name = name.clone();
        }
        if state.ship_name != name {
            state.ship_name = name;
        }
    } else {
        // Idle: pull model → field.
        let name = state
            .ship_entity
            .and_then(|e| ships.get(e).ok())
            .map(|s| s.name.clone())
            .unwrap_or_else(|| state.ship_name.clone());
        if field.value != name {
            field.value = name;
        }
    }
}
