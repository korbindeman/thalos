//! Formatters shared by HUD panels.

/// Compact altitude string. `km` up to 9999, then `Mm`, then `Gm`.
pub fn altitude(meters: f64) -> String {
    let abs = meters.abs();
    if abs < 9_999_500.0 {
        format!("{:.1} km", meters / 1_000.0)
    } else if abs < 9_999_500_000.0 {
        format!("{:.1} Mm", meters / 1_000_000.0)
    } else {
        format!("{:.2} Gm", meters / 1_000_000_000.0)
    }
}

/// Δv in m/s or km/s depending on magnitude.
pub fn delta_v(meters_per_second: f64) -> String {
    if meters_per_second.abs() >= 9_999.5 {
        format!("{:.2} km/s", meters_per_second / 1_000.0)
    } else {
        format!("{:.0} m/s", meters_per_second)
    }
}

/// Orbital speed. Compact m/s or km/s.
pub fn speed(meters_per_second: f64) -> String {
    if meters_per_second.abs() >= 9_999.5 {
        format!("{:.2} km/s", meters_per_second / 1_000.0)
    } else {
        format!("{:.0} m/s", meters_per_second)
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

/// "current / max" mass readout, in the same unit on both sides.
/// Tonnes once either side hits 1 t, otherwise kg.
pub fn resource_ratio(current_kg: f64, max_kg: f64) -> String {
    let scale = current_kg.abs().max(max_kg.abs());
    if scale >= 1_000.0 {
        format!("{:.1} / {:.1} t", current_kg / 1_000.0, max_kg / 1_000.0)
    } else {
        format!("{:.0} / {:.0} kg", current_kg, max_kg)
    }
}
