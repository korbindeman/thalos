//! Formatters shared by HUD panels.
//!
//! Every value the player sees is stored internally in SI; these formatters are
//! the single point where it is converted for display. Each takes a
//! [`UnitSystem`] and renders either the metric or the aviation-flavoured
//! imperial unit.
//!
//! Callers do **not** pass `UnitsSettings::system`. They pass
//! `units.system_for(UnitDomain::…)`, so an aviation instrument can read feet
//! and knots while the global preference stays metric — see
//! [`crate::units_settings`].

use crate::units::UnitSystem;

// Exact SI → imperial conversion factors.
const M_TO_FT: f64 = 3.280_839_895;
const M_TO_NMI: f64 = 1.0 / 1852.0;
const MPS_TO_KN: f64 = 1.943_844_492;
const MPS_TO_FPM: f64 = 196.850_393_7;
const KG_TO_LB: f64 = 2.204_622_622;
const PA_TO_PSF: f64 = 0.020_885_434;
const N_TO_LBF: f64 = 0.224_808_943;
const M2_TO_FT2: f64 = 10.763_910_417;

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

/// Horizontal ground distance, to one decimal — distance-to-go, cross-track,
/// runway range. Imperial: nautical miles, the aviation distance unit.
/// Metric: metres below 1 km, else kilometres.
pub fn ground_distance(meters: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        format!("{:.1} nmi", meters * M_TO_NMI)
    } else if meters.abs() < 1_000.0 {
        format!("{:.0} m", meters)
    } else {
        format!("{:.1} km", meters / 1_000.0)
    }
}

/// Coarse horizontal distance for a range-ring or scale label, where a decimal
/// place is noise. Same units as [`ground_distance`].
pub fn ground_range(meters: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        let nmi = meters * M_TO_NMI;
        // Sub-mile rings would read "0 nmi" without a decimal.
        if nmi.abs() < 9.95 {
            format!("{:.1} nmi", nmi)
        } else {
            format!("{:.0} nmi", nmi)
        }
    } else if meters.abs() < 1_000.0 {
        format!("{:.0} m", meters)
    } else {
        format!("{:.0} km", meters / 1_000.0)
    }
}

/// Signed altitude deviation, as a pilot reads it (`+` = high). Imperial: feet.
/// Metric: metres. Clamped so a nonsense guidance value can't stretch the line.
pub fn altitude_delta(meters: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        format!("{:+.0} ft", (meters * M_TO_FT).clamp(-99_999.0, 99_999.0))
    } else {
        format!("{:+.0} m", meters.clamp(-9_999.0, 9_999.0))
    }
}

/// Dynamic pressure. Metric: kilopascals. Imperial: pounds per square foot,
/// the unit `q` is quoted in on US flight-test cards.
pub fn dynamic_pressure(pascals: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        format!("{:.0} psf", pascals * PA_TO_PSF)
    } else {
        format!("{:.1} kPa", pascals / 1_000.0)
    }
}

/// Unit suffix for the vertical-speed readout, so a bare signed number on the
/// PFD's V/S tape can be labelled with what it actually means.
pub fn vertical_speed_unit(system: UnitSystem) -> &'static str {
    if system.is_imperial() {
        "ft/min"
    } else {
        "m/s"
    }
}

// ── Construction-scale quantities (the shipyard) ───────────────────────────────
//
// Part dimensions, thrust, and areas never appear on a flight instrument, so
// they have no aviation convention of their own — they follow the global system
// via `UnitDomain::General`. They live here anyway, because the conversion
// factors must have exactly one home.

/// A part dimension — diameter, span, chord, strut length. Metric: metres.
/// Imperial: feet. `decimals` is the caller's precision (the palette summarises
/// at 1, the inspector and its sliders read at 2).
pub fn length(meters: f64, decimals: usize, system: UnitSystem) -> String {
    if system.is_imperial() {
        let ft = meters * M_TO_FT;
        format!("{ft:.decimals$} ft")
    } else {
        format!("{meters:.decimals$} m")
    }
}

/// Scale factor and suffix for a length, for callers that must render the
/// number themselves — the shipyard's edit sliders, whose widget owns its own
/// formatting. Keeps the conversion factor from being copied into the UI crate.
pub fn length_display(system: UnitSystem) -> (f64, &'static str) {
    if system.is_imperial() {
        (M_TO_FT, "ft")
    } else {
        (1.0, "m")
    }
}

/// Engine thrust. Metric: kilonewtons. Imperial: pounds-force, in thousands
/// once it passes 100,000 lbf so a heavy first stage stays readable.
pub fn thrust(newtons: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        let lbf = newtons * N_TO_LBF;
        if lbf.abs() >= 100_000.0 {
            format!("{:.1}k lbf", lbf / 1_000.0)
        } else {
            format!("{lbf:.0} lbf")
        }
    } else {
        format!("{:.0} kN", newtons / 1_000.0)
    }
}

/// A surface area — wing planform, intake capture. Metric: m². Imperial: ft².
pub fn area(square_meters: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        format!("{:.2} ft²", square_meters * M2_TO_FT2)
    } else {
        format!("{square_meters:.2} m²")
    }
}

/// Impulse, as quoted for a decoupler's separation charge. Metric: newton-
/// seconds. Imperial: pound-force-seconds.
pub fn impulse(newton_seconds: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        format!("{:.0} lbf·s", newton_seconds * N_TO_LBF)
    } else {
        format!("{newton_seconds:.0} N·s")
    }
}

/// Mass at *vehicle* scale, for the shipyard's totals.
///
/// Deliberately coarser than [`mass`]: a HUD stage readout wants kilograms to
/// resolve a nearly-dry tank, while a VAB total is a whole launch vehicle and
/// would be unreadable in them. Metric adds a kilotonne tier; imperial adds
/// millions of pounds.
pub fn mass_large(kg: f64, system: UnitSystem) -> String {
    if system.is_imperial() {
        let lb = kg * KG_TO_LB;
        if lb.abs() >= 999_500.0 {
            format!("{:.2}M lb", lb / 1_000_000.0)
        } else if lb.abs() >= 9_999.5 {
            format!("{:.1}k lb", lb / 1_000.0)
        } else {
            format!("{lb:.0} lb")
        }
    } else if kg.abs() >= 999_500.0 {
        format!("{:.2} kt", kg / 1_000_000.0)
    } else if kg.abs() >= 9_999.5 {
        format!("{:.1} t", kg / 1_000.0)
    } else {
        format!("{kg:.0} kg")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::UnitSystem::{Imperial, Metric};

    /// The conversion factors, pinned against textbook figures. A silently
    /// wrong factor produces a plausible-looking instrument, which is the worst
    /// possible failure for a readout a player trusts to land with.
    #[test]
    fn conversion_factors_match_the_definitions() {
        // 1 nautical mile is exactly 1852 m; FL350 is 35 000 ft.
        assert_eq!(altitude(1852.0, Imperial), "6076 ft");
        assert_eq!(altitude(10_668.0, Imperial), "35000 ft");
        // 100 kn is 51.444 m/s.
        assert_eq!(speed(51.444, Imperial), "100 kn");
        // 1 m/s is 196.85 ft/min; a 500 ft/min descent is 2.54 m/s.
        assert!((vertical_speed_value(-2.54, Imperial) + 500.0).abs() < 0.5);
        // 1 kg is 2.2046 lb.
        assert_eq!(mass(453.592, Imperial), "1000 lb");
    }

    /// Metric and imperial must describe the *same* physical quantity — a
    /// mis-signed or mis-scaled branch would only show up in one of them.
    #[test]
    fn the_two_systems_agree_on_magnitude() {
        for m in [-8_000.0, -12.5, 0.0, 137.0, 9_500.0] {
            let ft = altitude_tape_value(m, Imperial);
            assert!(
                (ft / M_TO_FT - m).abs() < 1e-6,
                "altitude tape disagrees at {m} m"
            );
            let vs = vertical_speed_value(m, Imperial);
            assert_eq!(vs.signum(), m.signum(), "V/S sign flipped at {m} m/s");
        }
    }

    #[test]
    fn metric_altitude_climbs_through_its_units() {
        assert_eq!(altitude(500.0, Metric), "500 m");
        assert_eq!(altitude(120_000.0, Metric), "120.0 km");
        assert_eq!(altitude(4.0e8, Metric), "400.0 Mm");
        assert_eq!(altitude(1.5e11, Metric), "150.00 Gm");
    }

    #[test]
    fn ground_distance_uses_nautical_miles_when_imperial() {
        assert_eq!(ground_distance(18_520.0, Imperial), "10.0 nmi");
        assert_eq!(ground_distance(18_520.0, Metric), "18.5 km");
        assert_eq!(ground_distance(400.0, Metric), "400 m");
        // A range ring inside a mile must not collapse to "0 nmi".
        assert_eq!(ground_range(1852.0, Imperial), "1.0 nmi");
        assert_eq!(ground_range(92_600.0, Imperial), "50 nmi");
    }

    #[test]
    fn altitude_delta_keeps_its_sign_and_clamps() {
        assert_eq!(altitude_delta(30.48, Imperial), "+100 ft");
        assert_eq!(altitude_delta(-30.48, Imperial), "-100 ft");
        assert_eq!(altitude_delta(76.0, Metric), "+76 m");
        assert_eq!(altitude_delta(1.0e9, Metric), "+9999 m");
    }

    #[test]
    fn dynamic_pressure_switches_unit() {
        assert_eq!(dynamic_pressure(10_000.0, Metric), "10.0 kPa");
        assert_eq!(dynamic_pressure(10_000.0, Imperial), "209 psf");
    }

    /// Construction-scale factors, pinned against their definitions.
    #[test]
    fn construction_factors_match_the_definitions() {
        // 1 ft is exactly 0.3048 m; 1 lbf is 4.448222 N; 1 m² is 10.7639 ft².
        assert_eq!(length(0.3048, 2, Imperial), "1.00 ft");
        assert_eq!(length(2.5, 1, Metric), "2.5 m");
        assert_eq!(thrust(4448.222, Imperial), "1000 lbf");
        assert_eq!(thrust(4448.222, Metric), "4 kN");
        assert_eq!(area(1.0, Imperial), "10.76 ft²");
        assert_eq!(impulse(4448.222, Imperial), "1000 lbf·s");
    }

    /// The slider path must scale by the same factor the string path uses, or
    /// an edit control and the info text beside it would disagree.
    #[test]
    fn length_display_agrees_with_the_length_formatter() {
        for system in UnitSystem::ALL {
            let (factor, suffix) = length_display(system);
            assert_eq!(length(1.0 / factor, 2, system), format!("1.00 {suffix}"));
        }
    }

    /// Vehicle-scale mass is deliberately coarser than the HUD's stage readout.
    #[test]
    fn mass_large_tiers_at_vehicle_scale() {
        assert_eq!(mass_large(800.0, Metric), "800 kg");
        assert_eq!(mass_large(45_000.0, Metric), "45.0 t");
        assert_eq!(mass_large(2.5e6, Metric), "2.50 kt");
        // The same 45 t in pounds is ~99.2k lb.
        assert_eq!(mass_large(45_000.0, Imperial), "99.2k lb");
        // ...where the HUD's finer `mass` would still be counting kilograms.
        assert_eq!(mass(45_000.0, Metric), "45.0 t");
        assert_eq!(mass(800.0, Metric), "800 kg");
    }

    #[test]
    fn vertical_speed_unit_labels_the_bare_number() {
        assert_eq!(vertical_speed_unit(Metric), "m/s");
        assert_eq!(vertical_speed_unit(Imperial), "ft/min");
    }
}
