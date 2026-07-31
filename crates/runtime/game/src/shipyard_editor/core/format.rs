//! Display-string helpers for the editor: the parts-palette
//! category/ordering/summary scheme. Pure functions — no ECS access — so both
//! the egui binary and the in-game Bevy UI render the same catalog the same
//! way.
//!
//! **Unit conversion is not done here.** Every quantity routes through
//! [`crate::hud::format`], the one place SI becomes a display string, so the
//! shipyard obeys the same measurement preference as the HUD instead of being
//! hardcoded metric. The editor is not a flight instrument, so it resolves at
//! [`crate::units_settings::UnitDomain::General`] — it follows the global switch and never the
//! aeronautical override.

use thalos_shipyard::blueprint::default_params_for;
use thalos_shipyard::{CatalogEntry, PartParams};

use crate::hud::format;
use crate::units_settings::UnitSystem;

/// Stable ordering inside each palette category. Within each kind, callers
/// sort by display name.
pub fn kind_order(entry: &CatalogEntry) -> u8 {
    match entry {
        CatalogEntry::Pod(_) => 0,
        CatalogEntry::Engine(_) => 1,
        CatalogEntry::Intake(_) => 2,
        CatalogEntry::Decoupler(_) => 3,
        CatalogEntry::Adapter(_) => 4,
        CatalogEntry::Tank(_) => 5,
        CatalogEntry::Fuselage(_) => 6,
        CatalogEntry::Wing(_) => 7,
        CatalogEntry::Gear(_) => 8,
    }
}

pub fn palette_category_order(entry: &CatalogEntry) -> u8 {
    match entry {
        CatalogEntry::Pod(_) => 0,
        CatalogEntry::Engine(_) => 1,
        CatalogEntry::Intake(_) => 2,
        CatalogEntry::Tank(_) => 3,
        CatalogEntry::Adapter(_) | CatalogEntry::Decoupler(_) => 4,
        CatalogEntry::Fuselage(_) => 4,
        CatalogEntry::Wing(_) => 4,
        CatalogEntry::Gear(_) => 5,
    }
}

pub fn palette_category_label(entry: &CatalogEntry) -> &'static str {
    match entry {
        CatalogEntry::Pod(_) => "Command Pods",
        CatalogEntry::Engine(_) => "Engines",
        CatalogEntry::Intake(_) => "Intakes",
        CatalogEntry::Tank(_) => "Propellant Tanks",
        CatalogEntry::Adapter(_) | CatalogEntry::Decoupler(_) | CatalogEntry::Fuselage(_) => {
            "Structure"
        }
        CatalogEntry::Wing(_) => "Aerodynamics",
        CatalogEntry::Gear(_) => "Landing Gear",
    }
}

/// A part dimension at palette precision (one decimal).
pub fn meters_label(value: f32, system: UnitSystem) -> String {
    format::length(value as f64, 1, system)
}

/// One-line spec summary under each palette entry's name.
pub fn palette_part_summary(entry: &CatalogEntry, system: UnitSystem) -> String {
    let m = |v: f32| meters_label(v, system);
    match entry {
        CatalogEntry::Pod(p) => {
            format!(
                "{} · Diameter {} · {} dry",
                p.geometry.label(),
                m(p.diameter),
                format::mass_large(p.dry_mass as f64, system)
            )
        }
        CatalogEntry::Engine(e) => {
            // Specific impulse is seconds in both systems — the one figure here
            // that must NOT be converted.
            format!(
                "{} · {} · Diameter {} · {} · {:.0} s",
                e.optimized_for.label(),
                e.geometry.label(),
                m(e.diameter),
                format::thrust(e.thrust as f64, system),
                e.isp
            )
        }
        CatalogEntry::Intake(i) => format!(
            "Diameter {} · area {} · {}",
            m(i.diameter),
            format::area((i.capture.area_m2 * i.capture.efficiency) as f64, system),
            i.capture.kind.label()
        ),
        CatalogEntry::Decoupler(_) => match default_params_for(entry) {
            PartParams::Decoupler { diameter } => {
                format!("Default diameter {} · staging", m(diameter))
            }
            _ => "Parametric diameter".into(),
        },
        CatalogEntry::Adapter(_) => match default_params_for(entry) {
            PartParams::Adapter {
                diameter,
                target_diameter,
            } => format!("Default {} to {} diameter", m(diameter), m(target_diameter)),
            _ => "Parametric diameter".into(),
        },
        CatalogEntry::Tank(_) => match default_params_for(entry) {
            PartParams::Tank { diameter, length } => {
                format!("Default diameter {} · length {}", m(diameter), m(length))
            }
            _ => "Parametric diameter".into(),
        },
        CatalogEntry::Fuselage(_) => match default_params_for(entry) {
            PartParams::Fuselage {
                length, max_width, ..
            } => format!(
                "Loft body · default Ø{} · length {} · upswept tail",
                m(max_width),
                m(length)
            ),
            _ => "Stationed-loft fuselage".into(),
        },
        CatalogEntry::Wing(_) => match default_params_for(entry) {
            PartParams::Wing {
                span,
                root_chord,
                tip_chord,
                ..
            } => format!(
                "Span {} · chord {}→{} · click a hull to mount",
                m(span),
                m(root_chord),
                m(tip_chord)
            ),
            _ => "Parametric wing".into(),
        },
        CatalogEntry::Gear(g) => match default_params_for(entry) {
            PartParams::Gear {
                strut_length,
                wheel_radius,
            } => format!(
                "{} · strut {} · wheel Ø{} · click a belly to mount",
                if g.track_fraction > 0.0 {
                    "Main (L/R)"
                } else {
                    "Nose"
                },
                m(strut_length),
                m(wheel_radius * 2.0)
            ),
            _ => "Parametric gear".into(),
        },
    }
}
