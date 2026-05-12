//! Bottom-left HUD panel (next to the navball): throttle + orbital
//! velocity relative to the current SOI body.

use bevy::prelude::*;
use thalos_physics::canonical::Epoch;

use crate::fuel::ThrottleState;
use crate::hud::HudPanel;
use crate::hud::format;
use crate::hud::theme::{HudTheme, emphasis, label, panel_frame, panel_node, text};
use crate::rendering::SimulationState;

/// The navball cluster sits at the bottom-left (navball at x=40,
/// nav panel just to its right). The flight readouts sit ABOVE the
/// navball with a small gap.
const NAVBALL_LEFT_PX: f32 = 40.0;
const NAVBALL_BOTTOM_PX: f32 = 40.0;
const NAVBALL_SIZE_PX: f32 = 256.0;

#[derive(Component)]
pub(super) struct VelocityText;

#[derive(Component)]
pub(super) struct ThrottleText;

pub fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    let mut root = panel_node();
    // Sit immediately above the navball, aligned with its left edge.
    root.left = Val::Px(NAVBALL_LEFT_PX);
    root.bottom = Val::Px(NAVBALL_BOTTOM_PX + NAVBALL_SIZE_PX + 10.0);
    root.min_width = Val::Px(NAVBALL_SIZE_PX);

    let (bg, border) = panel_frame(&theme);
    commands
        .spawn((root, bg, border, HudPanel, Name::new("HudFlight")))
        .with_children(|p| {
            p.spawn(label(&theme, "ORBITAL VELOCITY"));
            p.spawn((emphasis(&theme, "—"), VelocityText));
            p.spawn(Node {
                height: Val::Px(4.0),
                ..default()
            });
            p.spawn(label(&theme, "THROTTLE"));
            p.spawn((text(&theme, "0%"), ThrottleText));
        });
}

pub fn update(
    sim: Res<SimulationState>,
    throttle: Res<ThrottleState>,
    theme: Res<HudTheme>,
    mut vel_q: Query<&mut Text, (With<VelocityText>, Without<ThrottleText>)>,
    mut thr_q: Query<(&mut Text, &mut TextColor), (With<ThrottleText>, Without<VelocityText>)>,
) {
    let ship = sim.simulation.ship_state();
    let body = sim.simulation.dominant_body();
    let body_state = sim
        .simulation
        .ephemeris()
        .state(body, Epoch(sim.simulation.sim_time()));
    let rel_speed = (ship.velocity - body_state.velocity).length();

    if let Ok(mut t) = vel_q.single_mut() {
        let s = format::speed(rel_speed);
        if t.0 != s {
            t.0 = s;
        }
    }

    if let Ok((mut t, mut color)) = thr_q.single_mut() {
        let s = format!("{:>3.0}%", throttle.commanded * 100.0);
        if t.0 != s {
            t.0 = s;
        }
        // Engine starvation: commanded > 0 but effective thrust capped
        // below — recolor the readout so the player sees it.
        let starved = throttle.commanded > 0.0 && throttle.effective + 1e-3 < throttle.commanded;
        let want = if starved {
            theme.text_warn
        } else {
            theme.text_primary
        };
        if color.0 != want {
            color.0 = want;
        }
    }
}
