//! Formatters shared by HUD panels.
//!
//! Every value the player sees is stored internally in SI; these formatters are
//! the single point where it is converted for display. Each takes the active
//! [`UnitSystem`] (from the persisted [`crate::units_settings::UnitsSettings`])
//! and renders either the metric or the aviation-flavoured imperial unit.

use crate::units_settings::UnitSystem;

// Exact SI → imperial conversion factors.
const M_TO_FT: f64 = 3.280_839_895;
const M_TO_NMI: f64 = 1.0 / 1852.0;
const MPS_TO_KN: f64 = 1.943_844_492;
const MPS_TO_FPM: f64 = 196.850_393_7;
const KG_TO_LB: f64 = 2.204_622_622;

/// Compact altitude string.
///
/// Metric: `m` below 10 km, then `km` up to 9999, then `Mm`, then `Gm`.
/// Imperial: `ft` up to ~30 km, then nautical miles.
pub fn altitude(meters: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        let ft = meters * M_TO_FT;
        if ft.abs() < 99_999.5 {
            format!("{:.0} ft", ft)
        } else {
            let nmi = meters * M_TO_NMI;
            if nmi.abs() < 9_999.5 {
                format!("{:.1} nmi", nmi)
            } else {
                format!("{:.0} nmi", nmi)
            }
        }
    } else {
        let abs = meters.abs();
        if abs < 9_999.5 {
            format!("{:.0} m", meters)
        } else if abs < 9_999_500.0 {
            format!("{:.1} km", meters / 1_000.0)
        } else if abs < 9_999_500_000.0 {
            format!("{:.1} Mm", meters / 1_000_000.0)
        } else {
            format!("{:.2} Gm", meters / 1_000_000_000.0)
        }
    }
}

/// Δv. Metric: m/s or km/s by magnitude. Imperial: ft/s (the aerospace unit).
pub fn delta_v(meters_per_second: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        let fps = meters_per_second * M_TO_FT;
        if fps.abs() >= 99_999.5 {
            format!("{:.1}k ft/s", fps / 1_000.0)
        } else {
            format!("{:.0} ft/s", fps)
        }
    } else if meters_per_second.abs() >= 9_999.5 {
        format!("{:.2} km/s", meters_per_second / 1_000.0)
    } else {
        format!("{:.0} m/s", meters_per_second)
    }
}

/// Fine-grained Δv for the maneuver-node editor, kept to one decimal so small
/// node tweaks stay legible. Metric: m/s. Imperial: ft/s.
pub fn delta_v_fine(meters_per_second: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        format!("{:.1} ft/s", meters_per_second * M_TO_FT)
    } else {
        format!("{:.1} m/s", meters_per_second)
    }
}

/// Speed. Metric: compact m/s or km/s. Imperial: knots.
pub fn speed(meters_per_second: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        let kn = meters_per_second * MPS_TO_KN;
        if kn.abs() >= 99_999.5 {
            format!("{:.0}k kn", kn / 1_000.0)
        } else {
            format!("{:.0} kn", kn)
        }
    } else if meters_per_second.abs() >= 9_999.5 {
        format!("{:.2} km/s", meters_per_second / 1_000.0)
    } else {
        format!("{:.0} m/s", meters_per_second)
    }
}

/// Altitude tape value in the active instrument's base unit (m or ft). Used by
/// the PFD tape so its ticks match the [`altitude`] readout.
pub fn altitude_tape_value(meters: f64, system: UnitSystem) -> f64 {
    if system.is_imperial() {
        meters * M_TO_FT
    } else {
        meters
    }
}

/// Speed tape value in the active instrument's base unit (m/s or kn). Used by
/// the PFD speed tape so its ticks match the [`speed`] readout.
pub fn speed_tape_value(meters_per_second: f64, system: UnitSystem) -> f64 {
    if system.is_imperial() {
        meters_per_second * MPS_TO_KN
    } else {
        meters_per_second
    }
}

/// Vertical-speed value in the active instrument's base unit (m/s or ft/min).
pub fn vertical_speed_value(meters_per_second: f64, system: UnitSystem) -> f64 {
    if system.is_imperial() {
        meters_per_second * MPS_TO_FPM
    } else {
        meters_per_second
    }
}

/// Countdown formatted as `T-HH:MM:SS`, or `T-Nd HH:MM:SS` for very long
/// intervals. Always positive; pass the absolute seconds-to-event.
pub fn countdown(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let days = total / 86_400;
    let hours = (total % 86_400) / 3_600;
    let minutes = (total % 3_600) / 60;
    let secs = total % 60;
    if days > 0 {
        format!("T-{}d {:02}:{:02}:{:02}", days, hours, minutes, secs)
    } else {
        format!("T-{:02}:{:02}:{:02}", hours, minutes, secs)
    }
}

/// Numeric pieces of the warp-panel clock. Years/days are unpadded so
/// long missions stretch the readout naturally; the clock keeps two-digit
/// `HH:MM:SS` so it doesn't visually jitter every second.
pub struct WarpClockParts {
    pub years: String,
    pub days: String,
    pub clock: String,
}

/// Decompose `seconds` into the year / day / clock fragments rendered by
/// the warp panel. Year = 365 days.
pub fn warp_panel_time(seconds: f64) -> WarpClockParts {
    const YEAR_S: u64 = 365 * 86_400;
    let total = seconds.max(0.0) as u64;
    let years = total / YEAR_S;
    let rem_y = total % YEAR_S;
    let days = rem_y / 86_400;
    let rem_d = rem_y % 86_400;
    let hours = rem_d / 3_600;
    let minutes = (rem_d % 3_600) / 60;
    let secs = rem_d % 60;
    WarpClockParts {
        years: years.to_string(),
        days: days.to_string(),
        clock: format!("{:02}:{:02}:{:02}", hours, minutes, secs),
    }
}

/// Single mass figure. Metric: tonnes at/above 1 t, else kg. Imperial: pounds,
/// switching to thousands of pounds (`k lb`) above 100,000 lb.
pub fn mass(kg: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        let lb = kg * KG_TO_LB;
        if lb.abs() >= 100_000.0 {
            format!("{:.1}k lb", lb / 1_000.0)
        } else {
            format!("{:.0} lb", lb)
        }
    } else if kg.abs() >= 1_000.0 {
        format!("{:.1} t", kg / 1_000.0)
    } else {
        format!("{:.0} kg", kg)
    }
}

/// "current / max" mass readout, in the same unit on both sides. Metric: tonnes
/// once either side hits 1 t, else kg. Imperial: pounds (or `k lb`).
pub fn resource_ratio(current_kg: f64, max_kg: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        let current = current_kg * KG_TO_LB;
        let max = max_kg * KG_TO_LB;
        if current.abs().max(max.abs()) >= 100_000.0 {
            format!("{:.1} / {:.1}k lb", current / 1_000.0, max / 1_000.0)
        } else {
            format!("{:.0} / {:.0} lb", current, max)
        }
    } else {
        let scale = current_kg.abs().max(max_kg.abs());
        if scale >= 1_000.0 {
            format!("{:.1} / {:.1} t", current_kg / 1_000.0, max_kg / 1_000.0)
        } else {
            format!("{:.0} / {:.0} kg", current_kg, max_kg)
        }
    }
}
