//! Top-middle atmospheric-flight readout: true airspeed, dynamic pressure, and
//! Mach, shown only while the ship is inside an atmosphere.
//!
//! Sources the read-only [`AeroReadout`] that the native aero model writes onto
//! the ship root each physics step (see [`crate::aero`]). Hidden in vacuum (and
//! for EVA, which has no aero root).

use bevy::prelude::*;
use thalos_physics_local::LocalCraftBody;

use crate::aero::AeroReadout;

use crate::hud::HudPanel;
use crate::hud::theme::{HudTheme, emphasis, label, panel_frame, panel_node};

/// Density (kg/m³) above which the panel is shown — i.e. "in atmosphere".
const IN_ATMOSPHERE_DENSITY: f64 = 1.0e-6;

#[derive(Component)]
pub(super) struct AtmoPanel;

#[derive(Component)]
pub(super) struct AtmoText;

pub fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    // Full-width centring row (mirrors the orbital panel idiom) so the panel
    // sits centred just below the top-middle altitude/orbital cluster.
    let wrapper = Node {
        position_type: PositionType::Absolute,
        top: Val::Px(96.0),
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Row,
        justify_content: JustifyContent::Center,
        align_items: AlignItems::FlexStart,
        ..default()
    };

    commands
        .spawn((wrapper, Name::new("HudAtmoWrapper")))
        .with_children(|p| {
            let mut root = panel_node();
            root.position_type = PositionType::Relative;
            root.align_items = AlignItems::Center;
            let (bg, border) = panel_frame(&theme);
            // Start hidden; `update` reveals it only inside an atmosphere.
            p.spawn((
                root,
                bg,
                border,
                Visibility::Hidden,
                AtmoPanel,
                HudPanel,
                Name::new("HudAtmosphere"),
            ))
            .with_children(|c| {
                c.spawn((emphasis(&theme, "—"), AtmoText));
                c.spawn(label(&theme, "ATMOSPHERE"));
            });
        });
}

pub fn update(
    ship_q: Query<&AeroReadout, With<LocalCraftBody>>,
    units: Res<crate::units_settings::UnitsSettings>,
    mut panel_q: Query<&mut Visibility, With<AtmoPanel>>,
    mut text_q: Query<&mut Text, With<AtmoText>>,
) {
    let in_atmosphere = ship_q
        .single()
        .map(|r| r.density_kgm3 > IN_ATMOSPHERE_DENSITY)
        .unwrap_or(false);

    if let Ok(mut visibility) = panel_q.single_mut() {
        let target = if in_atmosphere {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != target {
            *visibility = target;
        }
    }

    if in_atmosphere
        && let Ok(flight) = ship_q.single()
        && let Ok(mut text) = text_q.single_mut()
    {
        // TAS reuses the shared speed formatter (m/s or knots); dynamic
        // pressure is kPa metric / psf imperial; Mach is dimensionless.
        let tas = crate::hud::format::speed(flight.airspeed_ms, units.system);
        let value = if units.system.is_imperial() {
            let q_psf = flight.dynamic_pressure_pa * 0.020_885_434;
            format!("TAS {tas}  ·  q {q_psf:.0} psf  ·  M {:.2}", flight.mach)
        } else {
            let q_kpa = flight.dynamic_pressure_pa / 1000.0;
            format!("TAS {tas}  ·  q {q_kpa:.1} kPa  ·  M {:.2}", flight.mach)
        };
        if text.0 != value {
            text.0 = value;
        }
    }
}
