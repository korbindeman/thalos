//! The vertical half of a route: what altitude and speed you should be at, as a
//! function of **distance still to fly**.
//!
//! Parameterising by distance-to-go (not by time, and not by waypoint index) is
//! what makes the profile a single continuous function the whole stack can
//! agree on: the ND draws it, the PFD's glideslope scale deviates against it,
//! and the autopilot's vertical-speed command is its derivative. It also means
//! there is no state to keep in sync — recomputing the profile every frame gives
//! the same answer.
//!
//! The shape, from far out to touchdown:
//!
//! ```text
//!  alt
//!   │────────────────╮  cruise (level, whatever you were at when planned)
//!   │                 ╲  descent at `cruise_descent_rad`
//!   │                  ╰──────────╮  capture / platform altitude (level)
//!   │                              ╲  glideslope at `glideslope_rad`
//!   └───────────────────────────────╲──▶  dtg → 0 at the aim point
//! ```
//!
//! Everything is **altitude above the body reference radius** (Thalos has no sea
//! level layer; sea level *is* 0 m), never AGL — an AGL profile would step every
//! time the terrain under it changed.

/// A speed constraint at a distance-to-go, with a label for the display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedGate {
    /// Distance-to-go at which this speed should be reached (m).
    pub dtg_m: f64,
    pub speed_m_s: f64,
    /// Short annunciation, e.g. `"FLAPS"` / `"GEAR"` / `"VAPP"`.
    pub label: &'static str,
}

/// Inputs for building a [`VerticalProfile`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VnavParams {
    /// Glideslope angle (rad) flown on final. 3° is the civil standard and the
    /// default; steeper is legal and shows up directly in the deviation scale.
    pub glideslope_rad: f64,
    /// Descent gradient (rad) flown between cruise and the capture altitude.
    pub cruise_descent_rad: f64,
    /// Distance-to-go where the final approach segment begins — the glideslope
    /// runs from here down to the aim point, so this is also where the capture
    /// altitude is defined.
    pub final_dtg_m: f64,
    /// Approach speed (m/s) to be established by the start of final.
    pub approach_speed_m_s: f64,
    /// Distance-to-go before the final approach point at which the craft should
    /// already be configured (gear/flaps) and slowing.
    pub configure_lead_m: f64,
}

impl Default for VnavParams {
    fn default() -> Self {
        Self {
            glideslope_rad: 3.0_f64.to_radians(),
            cruise_descent_rad: 3.0_f64.to_radians(),
            final_dtg_m: 9_000.0,
            approach_speed_m_s: 80.0,
            configure_lead_m: 6_000.0,
        }
    }
}

/// The planned vertical profile.
#[derive(Debug, Clone, PartialEq)]
pub struct VerticalProfile {
    /// Altitude of the aim point (m above the body reference radius) — where
    /// `dtg = 0`.
    pub aim_altitude_m: f64,
    pub glideslope_rad: f64,
    /// Distance-to-go where the glideslope starts (= the final approach point).
    pub capture_dtg_m: f64,
    /// Altitude the glideslope reaches at `capture_dtg_m` — the platform the
    /// craft levels at while intercepting.
    pub capture_altitude_m: f64,
    /// Level cruise altitude the profile starts from (never below capture).
    pub cruise_altitude_m: f64,
    pub cruise_descent_rad: f64,
    /// Speed gates, ordered by **descending** `dtg_m` (furthest out first).
    pub speed_gates: Vec<SpeedGate>,
}

impl VerticalProfile {
    /// Build the profile for an approach whose aim point sits at
    /// `aim_altitude_m`, planned while the craft is at `craft_altitude_m`.
    ///
    /// The cruise altitude is the craft's own altitude (floored at the capture
    /// altitude), so the profile never asks a craft that is already low to climb
    /// back up to a nominal cruise level just to descend again.
    pub fn plan(params: &VnavParams, aim_altitude_m: f64, craft_altitude_m: f64) -> Self {
        let capture_dtg_m = params.final_dtg_m.max(0.0);
        let capture_altitude_m = aim_altitude_m + capture_dtg_m * params.glideslope_rad.tan();
        let cruise_altitude_m = craft_altitude_m.max(capture_altitude_m);
        let va = params.approach_speed_m_s.max(1.0);
        // Gates furthest-out first: configure and slow to 1.3 Vapp before the
        // final approach point, be at Vapp by the time the glideslope starts.
        let speed_gates = vec![
            SpeedGate {
                dtg_m: capture_dtg_m + params.configure_lead_m.max(0.0),
                speed_m_s: va * 1.3,
                label: "FLAPS",
            },
            SpeedGate {
                dtg_m: capture_dtg_m + params.configure_lead_m.max(0.0) * 0.4,
                speed_m_s: va * 1.12,
                label: "GEAR",
            },
            SpeedGate {
                dtg_m: capture_dtg_m,
                speed_m_s: va,
                label: "VAPP",
            },
            SpeedGate {
                dtg_m: 0.0,
                speed_m_s: va,
                label: "VAPP",
            },
        ];
        Self {
            aim_altitude_m,
            glideslope_rad: params.glideslope_rad,
            capture_dtg_m,
            capture_altitude_m,
            cruise_altitude_m,
            cruise_descent_rad: params.cruise_descent_rad,
            speed_gates,
        }
    }

    /// Distance-to-go at which the descent from cruise to the capture altitude
    /// should begin (the top of descent). Equals [`Self::capture_dtg_m`] when
    /// the craft is already at or below the capture altitude.
    pub fn top_of_descent_dtg_m(&self) -> f64 {
        let drop = (self.cruise_altitude_m - self.capture_altitude_m).max(0.0);
        let gradient = self.cruise_descent_rad.tan();
        if gradient <= 1e-9 {
            self.capture_dtg_m
        } else {
            self.capture_dtg_m + drop / gradient
        }
    }

    /// Target altitude (m above the reference radius) at a distance-to-go.
    ///
    /// Monotone non-decreasing in `dtg`, which is what lets the deviation be
    /// read as a simple high/low: on the glideslope inside the final segment,
    /// then the descent ramp, then level cruise.
    pub fn target_altitude_m(&self, dtg_m: f64) -> f64 {
        let dtg = dtg_m.max(0.0);
        if dtg <= self.capture_dtg_m {
            self.aim_altitude_m + dtg * self.glideslope_rad.tan()
        } else {
            let ramp = self.capture_altitude_m
                + (dtg - self.capture_dtg_m) * self.cruise_descent_rad.tan();
            ramp.min(self.cruise_altitude_m)
        }
    }

    /// Target speed (m/s) at a distance-to-go: linear between gates, held
    /// constant outside them. `None` if the profile has no gates.
    pub fn target_speed_m_s(&self, dtg_m: f64) -> Option<f64> {
        let gates = &self.speed_gates;
        let first = gates.first()?;
        if dtg_m >= first.dtg_m {
            return Some(first.speed_m_s);
        }
        let last = gates.last()?;
        if dtg_m <= last.dtg_m {
            return Some(last.speed_m_s);
        }
        for w in gates.windows(2) {
            let (a, b) = (w[0], w[1]);
            if dtg_m <= a.dtg_m && dtg_m >= b.dtg_m {
                let span = a.dtg_m - b.dtg_m;
                let t = if span > 1e-9 {
                    (a.dtg_m - dtg_m) / span
                } else {
                    1.0
                };
                return Some(a.speed_m_s + (b.speed_m_s - a.speed_m_s) * t);
            }
        }
        Some(last.speed_m_s)
    }

    /// The next gate still ahead at this distance-to-go (the one to annunciate).
    pub fn next_gate(&self, dtg_m: f64) -> Option<SpeedGate> {
        self.speed_gates
            .iter()
            .filter(|g| g.dtg_m <= dtg_m)
            .max_by(|a, b| a.dtg_m.total_cmp(&b.dtg_m))
            .copied()
    }

    /// Angular glideslope deviation (rad, **positive = above the slope**) for a
    /// craft at `altitude_m` with `horizontal_distance_m` to the aim point.
    ///
    /// Angular (not linear) is what an ILS-style scale wants: the same 30 m high
    /// is a full-scale error on short final and nothing at all 15 km out, which
    /// is exactly the sensitivity a pilot needs.
    pub fn glideslope_deviation_rad(&self, horizontal_distance_m: f64, altitude_m: f64) -> f64 {
        let d = horizontal_distance_m.max(1.0);
        let elevation = ((altitude_m - self.aim_altitude_m) / d).atan();
        elevation - self.glideslope_rad
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn profile() -> VerticalProfile {
        // Aim point at 700 m elevation, craft cruising at 3,000 m.
        VerticalProfile::plan(&VnavParams::default(), 700.0, 3_000.0)
    }

    #[test]
    fn glideslope_holds_three_degrees_inside_the_final_segment() {
        let p = profile();
        let tan3 = 3.0_f64.to_radians().tan();
        assert_abs_diff_eq!(p.target_altitude_m(0.0), 700.0, epsilon = 1e-9);
        assert_abs_diff_eq!(
            p.target_altitude_m(1_000.0),
            700.0 + 1_000.0 * tan3,
            epsilon = 1e-9
        );
        // The capture altitude is the glideslope evaluated at the final point.
        assert_abs_diff_eq!(
            p.capture_altitude_m,
            700.0 + 9_000.0 * tan3,
            epsilon = 1e-9
        );
    }

    #[test]
    fn profile_is_level_at_cruise_then_ramps_then_slopes() {
        let p = profile();
        let tod = p.top_of_descent_dtg_m();
        assert!(tod > p.capture_dtg_m, "descent must start before final");
        // Beyond the top of descent: level cruise.
        assert_abs_diff_eq!(p.target_altitude_m(tod + 5_000.0), 3_000.0, epsilon = 1e-9);
        assert_abs_diff_eq!(p.target_altitude_m(tod), 3_000.0, epsilon = 1e-6);
        // Between TOD and final: strictly descending.
        let mid = 0.5 * (tod + p.capture_dtg_m);
        assert!(p.target_altitude_m(mid) < 3_000.0);
        assert!(p.target_altitude_m(mid) > p.capture_altitude_m);
    }

    #[test]
    fn target_altitude_is_monotone_in_distance_to_go() {
        let p = profile();
        let mut prev = f64::NEG_INFINITY;
        for i in 0..400 {
            let dtg = i as f64 * 200.0;
            let alt = p.target_altitude_m(dtg);
            assert!(
                alt >= prev - 1e-9,
                "altitude dipped at dtg {dtg}: {alt} < {prev}"
            );
            prev = alt;
        }
    }

    #[test]
    fn a_low_craft_is_never_told_to_climb_to_cruise() {
        // Planned while already below the capture altitude.
        let p = VerticalProfile::plan(&VnavParams::default(), 700.0, 300.0);
        assert_abs_diff_eq!(p.cruise_altitude_m, p.capture_altitude_m, epsilon = 1e-9);
        assert_abs_diff_eq!(p.top_of_descent_dtg_m(), p.capture_dtg_m, epsilon = 1e-9);
        // Far out, the target is the platform, not a cruise level above it.
        assert_abs_diff_eq!(
            p.target_altitude_m(50_000.0),
            p.capture_altitude_m,
            epsilon = 1e-9
        );
    }

    #[test]
    fn glideslope_deviation_signs_read_high_and_low() {
        let p = profile();
        let d = 5_000.0;
        let on_slope = p.aim_altitude_m + d * p.glideslope_rad.tan();
        assert_abs_diff_eq!(p.glideslope_deviation_rad(d, on_slope), 0.0, epsilon = 1e-12);
        assert!(p.glideslope_deviation_rad(d, on_slope + 200.0) > 0.0, "high");
        assert!(p.glideslope_deviation_rad(d, on_slope - 200.0) < 0.0, "low");
    }

    #[test]
    fn deviation_sensitivity_grows_as_range_closes() {
        let p = profile();
        // The same 40 m high is a much larger angular error up close.
        let near = p
            .glideslope_deviation_rad(600.0, p.aim_altitude_m + 600.0 * p.glideslope_rad.tan() + 40.0)
            .abs();
        let far = p
            .glideslope_deviation_rad(
                12_000.0,
                p.aim_altitude_m + 12_000.0 * p.glideslope_rad.tan() + 40.0,
            )
            .abs();
        assert!(near > far * 10.0, "near {near} vs far {far}");
    }

    #[test]
    fn speed_gates_step_down_toward_the_approach_speed() {
        let p = profile();
        let va = VnavParams::default().approach_speed_m_s;
        // Far out: the first (fastest) gate.
        assert_abs_diff_eq!(
            p.target_speed_m_s(100_000.0).expect("gates exist"),
            va * 1.3,
            epsilon = 1e-9
        );
        // On final: approach speed.
        assert_abs_diff_eq!(
            p.target_speed_m_s(1_000.0).expect("gates exist"),
            va,
            epsilon = 1e-9
        );
        // Monotone non-increasing as distance closes.
        let mut prev = f64::INFINITY;
        for i in (0..300).rev() {
            let v = p.target_speed_m_s(i as f64 * 200.0).expect("gates exist");
            assert!(v <= prev + 1e-9, "speed rose closing in: {v} > {prev}");
            prev = v;
        }
    }

    #[test]
    fn next_gate_is_the_nearest_one_still_ahead() {
        let p = profile();
        let g = p.next_gate(20_000.0).expect("a gate is ahead");
        assert_eq!(g.label, "FLAPS");
        // At 12 km to go the GEAR gate (11.4 km) is the next one ahead...
        let g = p.next_gate(12_000.0).expect("a gate is ahead");
        assert_eq!(g.label, "GEAR");
        // ...and by 9.5 km it is already behind, so VAPP is next.
        let g = p.next_gate(9_500.0).expect("a gate is ahead");
        assert_eq!(g.label, "VAPP");
        let g = p.next_gate(4_000.0).expect("a gate is ahead");
        assert_eq!(g.label, "VAPP");
    }

    #[test]
    fn degenerate_gradient_does_not_divide_by_zero() {
        let params = VnavParams {
            cruise_descent_rad: 0.0,
            ..VnavParams::default()
        };
        let p = VerticalProfile::plan(&params, 700.0, 5_000.0);
        assert!(p.top_of_descent_dtg_m().is_finite());
    }
}
