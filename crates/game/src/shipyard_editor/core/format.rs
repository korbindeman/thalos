//! Display-string helpers shared by editor front-ends: unit formatting and
//! the parts-palette category/ordering/summary scheme. Pure functions — no
//! ECS access — so both the egui binary and the in-game Bevy UI render the
//! same catalog the same way.

use thalos_shipyard::blueprint::default_params_for;
use thalos_shipyard::{CatalogEntry, PartParams};

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

pub fn meters_label(value: f32) -> String {
    format!("{value:.1} m")
}

/// One-line spec summary under each palette entry's name.
pub fn palette_part_summary(entry: &CatalogEntry) -> String {
    match entry {
        CatalogEntry::Pod(p) => {
            format!(
                "{} · Diameter {} · {:.1} t dry",
                p.geometry.label(),
                meters_label(p.diameter),
                p.dry_mass / 1000.0
            )
        }
        CatalogEntry::Engine(e) => {
            format!(
                "{} · {} · Diameter {} · {:.0} kN · {:.0} s",
                e.optimized_for.label(),
                e.geometry.label(),
                meters_label(e.diameter),
                e.thrust / 1000.0,
                e.isp
            )
        }
        CatalogEntry::Intake(i) => format!(
            "Diameter {} · area {:.2} m² · {}",
            meters_label(i.diameter),
            i.capture.area_m2 * i.capture.efficiency,
            i.capture.kind.label()
        ),
        CatalogEntry::Decoupler(_) => match default_params_for(entry) {
            PartParams::Decoupler { diameter } => {
                format!("Default diameter {} · staging", meters_label(diameter))
            }
            _ => "Parametric diameter".into(),
        },
        CatalogEntry::Adapter(_) => match default_params_for(entry) {
            PartParams::Adapter {
                diameter,
                target_diameter,
            } => format!(
                "Default {} to {} diameter",
                meters_label(diameter),
                meters_label(target_diameter)
            ),
            _ => "Parametric diameter".into(),
        },
        CatalogEntry::Tank(_) => match default_params_for(entry) {
            PartParams::Tank { diameter, length } => format!(
                "Default diameter {} · length {}",
                meters_label(diameter),
                meters_label(length)
            ),
            _ => "Parametric diameter".into(),
        },
        CatalogEntry::Fuselage(_) => match default_params_for(entry) {
            PartParams::Fuselage {
                length, max_width, ..
            } => format!(
                "Loft body · default Ø{} · length {} · upswept tail",
                meters_label(max_width),
                meters_label(length)
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
                meters_label(span),
                meters_label(root_chord),
                meters_label(tip_chord)
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
                meters_label(strut_length),
                meters_label(wheel_radius * 2.0)
            ),
            _ => "Parametric gear".into(),
        },
    }
}

pub fn format_delta_v(meters_per_second: f64) -> String {
    if meters_per_second.abs() >= 9_999.5 {
        format!("{:.2} km/s", meters_per_second / 1_000.0)
    } else {
        format!("{:.0} m/s", meters_per_second)
    }
}

pub fn format_mass_kg(kg: f64) -> String {
    if kg.abs() >= 999_500.0 {
        format!("{:.2} kt", kg / 1_000_000.0)
    } else if kg.abs() >= 9_999.5 {
        format!("{:.1} t", kg / 1_000.0)
    } else {
        format!("{:.0} kg", kg)
    }
}
