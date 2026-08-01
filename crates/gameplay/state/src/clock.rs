//! The simulation clock's shared vocabulary.
//!
//! Bevy's default `Time`/`Time<Virtual>` is intentionally left as an app
//! clock. Presentation systems that need animation while the simulation is
//! paused should use `Time<Real>` directly. Systems that advance canonical
//! state, local physics ownership, resource burn, or surface walking consume
//! [`SimClock`] instead.
//!
//! **Sole writer:** the runtime's `sync_sim_clock`. Every pause source that
//! should halt canonical/local simulation folds into that one predicate. The
//! wall/driven mechanics ([`SimClockDrive`] → Bevy's `TimeUpdateStrategy`)
//! also live with the runtime (`apply_clock_drive`); see that module's docs
//! for why the driven mode goes through Bevy's own clock.

use bevy::prelude::*;

/// Fastest driven step accepted (10 000 fps). Below this the `Duration`
/// conversion starts losing resolution and no output format wants it.
pub const MIN_DRIVEN_DT_S: f64 = 1.0e-4;
/// Slowest driven step accepted (1 fps). A larger step would push local physics
/// past what its integrator resolves in one frame.
pub const MAX_DRIVEN_DT_S: f64 = 1.0;

/// How the app's clock advances.
///
/// Input to the runtime's `sync_sim_clock` and `apply_clock_drive` — **not** a
/// second clock. The driven mode exists for offline render: a frame lasts
/// exactly `dt_s` no matter how long it took, so frame *n* lands at *n · dt*
/// (the foundation of deterministic capture, ADR-20260730T212556Z).
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
pub enum SimClockDrive {
    /// Interactive: a frame lasts as long as it took.
    #[default]
    Wall,
    /// Offline render: a frame lasts exactly `dt_s`.
    Driven { dt_s: f64 },
}

impl SimClockDrive {
    /// Driven mode at a frame rate, e.g. `60.0` → a 1/60 s step.
    pub fn driven_from_fps(fps: f64) -> Result<Self, String> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err(format!(
                "driven clock fps {fps} must be finite and positive"
            ));
        }
        Self::driven_from_dt_s(1.0 / fps)
    }

    /// Driven mode at an explicit step.
    pub fn driven_from_dt_s(dt_s: f64) -> Result<Self, String> {
        if !dt_s.is_finite() || !(MIN_DRIVEN_DT_S..=MAX_DRIVEN_DT_S).contains(&dt_s) {
            return Err(format!(
                "driven clock step {dt_s} s is outside {MIN_DRIVEN_DT_S}..={MAX_DRIVEN_DT_S}"
            ));
        }
        Ok(Self::Driven { dt_s })
    }

    /// Parse the CLI/environment adapter form: `wall`, `driven` (60 fps),
    /// `driven:<fps>`, or a bare frame rate.
    pub fn parse(raw: &str) -> Result<Self, String> {
        const DEFAULT_DRIVEN_FPS: f64 = 60.0;
        let raw = raw.trim();
        match raw.to_ascii_lowercase().as_str() {
            "" => Err("clock mode cannot be empty".into()),
            "wall" | "real" => Ok(Self::Wall),
            "driven" | "fixed" => Self::driven_from_fps(DEFAULT_DRIVEN_FPS),
            lowered => {
                let fps = lowered
                    .strip_prefix("driven:")
                    .or_else(|| lowered.strip_prefix("fixed:"))
                    .unwrap_or(lowered);
                let fps = fps.parse::<f64>().map_err(|_| {
                    format!("clock mode {raw:?} must be wall, driven, driven:<fps>, or <fps>")
                })?;
                Self::driven_from_fps(fps)
            }
        }
    }

    /// The fixed step, or `None` on the wall clock.
    pub fn dt_s(self) -> Option<f64> {
        match self {
            Self::Wall => None,
            Self::Driven { dt_s } => Some(dt_s),
        }
    }
}

/// Frame-local simulation clock.
///
/// `delta_s` is the frame's step while simulation is running and zero while any
/// sim pause source is active. Consumers should use `delta_secs_f64()` for
/// physical integration rather than reaching for Bevy's global `Time` resource.
/// The mode itself is **not** mirrored here — [`SimClockDrive`] is the one
/// place it lives, and consumers that need to report it (the capture receipt)
/// read that resource. This carries the delta; that carries the mode.
///
/// **Sole writer:** the runtime's `sync_sim_clock`, which also owns the
/// canonical list of pause sources.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct SimClock {
    delta_s: f64,
    paused: bool,
}

impl SimClock {
    /// Construct a clock value — for the sole writer (and tests). Reading
    /// consumers never need this.
    pub fn from_writer(delta_s: f64, paused: bool) -> Self {
        Self { delta_s, paused }
    }

    pub fn delta_secs_f64(self) -> f64 {
        self.delta_s
    }

    pub fn is_paused(self) -> bool {
        self.paused
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drive_parses_the_adapter_forms() {
        assert_eq!(SimClockDrive::parse("wall").unwrap(), SimClockDrive::Wall);
        assert_eq!(SimClockDrive::parse(" Real ").unwrap(), SimClockDrive::Wall);
        assert_eq!(
            SimClockDrive::parse("driven").unwrap(),
            SimClockDrive::Driven { dt_s: 1.0 / 60.0 }
        );
        assert_eq!(
            SimClockDrive::parse("driven:30").unwrap(),
            SimClockDrive::Driven { dt_s: 1.0 / 30.0 }
        );
        assert_eq!(
            SimClockDrive::parse("24").unwrap(),
            SimClockDrive::Driven { dt_s: 1.0 / 24.0 }
        );
    }

    #[test]
    fn drive_rejects_unusable_steps() {
        // Out of range in both directions, and non-numeric.
        assert!(SimClockDrive::driven_from_fps(0.0).is_err());
        assert!(SimClockDrive::driven_from_fps(-60.0).is_err());
        assert!(SimClockDrive::driven_from_fps(f64::NAN).is_err());
        assert!(SimClockDrive::driven_from_fps(0.5).is_err()); // 2 s step
        assert!(SimClockDrive::driven_from_fps(1.0e6).is_err()); // 1 µs step
        assert!(SimClockDrive::parse("").is_err());
        assert!(SimClockDrive::parse("sometimes").is_err());
        assert!(SimClockDrive::parse("driven:fast").is_err());
    }
}
