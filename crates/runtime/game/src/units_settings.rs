//! Measurement-unit preference (metric vs imperial), resolved per display domain.
//!
//! [`UnitsSettings`] is the unit-system preference. It is persisted (alongside
//! window + graphics) by [`crate::settings`] as the `units` section of the
//! unified `settings.ron`; this module owns only the resource + its `Reflect`
//! registration, not the file IO.
//!
//! SI is always the internal/simulation unit; this preference only affects how
//! the HUD *displays* distances, speeds, and masses. The settings menu's Units
//! tab is the sole writer; the HUD formatters in [`crate::hud::format`] read it
//! and dispatch on [`UnitSystem`].
//!
//! # Why a domain, not just a global switch
//!
//! Aviation is unit-conservative in a way the rest of the world is not: feet,
//! knots, feet-per-minute, and nautical miles are the instrument units in
//! metric countries too. So the preference is resolved **per
//! [`UnitDomain`]** — a panel asks for the system appropriate to the kind of
//! instrument it is, via [`UnitsSettings::system_for`], instead of reading
//! `system` directly. An *instrument* decides its domain, not the craft or the
//! flight regime: a tape whose unit changed mid-climb would be worse than one
//! that is consistently in the wrong system (ADR-free judgment call; the
//! alternatives considered were regime-driven and craft-class-driven
//! resolution).

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

// ── Unit system ─────────────────────────────────────────────────────────────────

/// Which measurement system the HUD formats values in.
///
/// `Imperial` is the aviation-flavoured set: feet for altitude, knots for speed,
/// feet-per-minute for vertical speed, nautical miles for long distances, and
/// pounds for mass.
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitSystem {
    Metric,
    Imperial,
}

impl UnitSystem {
    pub const ALL: [UnitSystem; 2] = [UnitSystem::Metric, UnitSystem::Imperial];

    pub fn label(self) -> &'static str {
        match self {
            UnitSystem::Metric => "Metric (m, km, m/s)",
            UnitSystem::Imperial => "Imperial (ft, kn)",
        }
    }

    pub fn is_imperial(self) -> bool {
        matches!(self, UnitSystem::Imperial)
    }
}

// ── Display domains ────────────────────────────────────────────────────────────

/// The kind of readout a formatter is rendering, which selects the unit
/// convention it obeys.
///
/// Add a variant only alongside a matching field on [`UnitsSettings`] and a
/// `system_for` arm — a domain nothing resolves is dead weight. (A `Marine`
/// domain, knots + nautical miles, is the obvious next one; it is deliberately
/// absent until there is a ship instrument to attach it to.)
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitDomain {
    /// Spaceflight and everything else: orbital altitude and apsides, Δv,
    /// staging masses, map scales. Follows the global [`UnitSystem`] exactly.
    General,
    /// Aviation instruments: the PFD tapes and readouts, the atmospheric
    /// TAS/q/Mach pill, and the MFD navigation display.
    Aviation,
}

/// Unit convention for the aviation instruments, independent of the global
/// measurement system.
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AviationUnits {
    /// Feet, knots, feet-per-minute, nautical miles — whatever the global
    /// system says. The real-world instrument convention, and the default.
    Aeronautical,
    /// Use the global measurement system on the flight instruments too.
    FollowGlobal,
}

impl AviationUnits {
    pub const ALL: [AviationUnits; 2] = [AviationUnits::Aeronautical, AviationUnits::FollowGlobal];

    pub fn label(self) -> &'static str {
        match self {
            AviationUnits::Aeronautical => "Aeronautical (ft, kn, ft/min)",
            AviationUnits::FollowGlobal => "Follow global",
        }
    }
}

// ── Resource ───────────────────────────────────────────────────────────────────

/// User measurement-unit preference, persisted as the `units` section of the
/// unified `settings.ron` (see [`crate::settings`]).
///
/// Writer: the settings menu's Units tab. The HUD formatters read — through
/// [`Self::system_for`], never `self.system` directly.
/// `Reflect`-registered (for a future in-game debug UI).
#[derive(Resource, Reflect, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[reflect(Resource)]
#[serde(default)]
pub struct UnitsSettings {
    /// Measurement system the HUD formats displayed values in.
    pub system: UnitSystem,
    /// Whether the aviation instruments override `system` with the aeronautical
    /// convention.
    pub aviation: AviationUnits,
}

impl Default for UnitsSettings {
    fn default() -> Self {
        Self {
            system: UnitSystem::Metric,
            aviation: AviationUnits::Aeronautical,
        }
    }
}

impl UnitsSettings {
    /// The measurement system a `domain`'s readouts should use.
    ///
    /// This is the only supported way to reach a display unit; reading
    /// [`Self::system`] from a panel bypasses the per-domain conventions.
    pub fn system_for(self, domain: UnitDomain) -> UnitSystem {
        match domain {
            UnitDomain::General => self.system,
            UnitDomain::Aviation => match self.aviation {
                AviationUnits::Aeronautical => UnitSystem::Imperial,
                AviationUnits::FollowGlobal => self.system,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the feature: metric everywhere, feet and knots on the
    /// flight instruments.
    #[test]
    fn metric_global_still_flies_on_feet_and_knots() {
        let s = UnitsSettings::default();
        assert_eq!(s.system_for(UnitDomain::General), UnitSystem::Metric);
        assert_eq!(s.system_for(UnitDomain::Aviation), UnitSystem::Imperial);
    }

    #[test]
    fn follow_global_makes_the_domains_agree() {
        for system in UnitSystem::ALL {
            let s = UnitsSettings {
                system,
                aviation: AviationUnits::FollowGlobal,
            };
            assert_eq!(s.system_for(UnitDomain::General), system);
            assert_eq!(s.system_for(UnitDomain::Aviation), system);
        }
    }

    /// An imperial global must not be *undone* by the aviation override.
    #[test]
    fn imperial_global_stays_imperial_on_the_instruments() {
        for aviation in AviationUnits::ALL {
            let s = UnitsSettings {
                system: UnitSystem::Imperial,
                aviation,
            };
            assert_eq!(s.system_for(UnitDomain::Aviation), UnitSystem::Imperial);
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────────

pub struct UnitsSettingsPlugin;

impl Plugin for UnitsSettingsPlugin {
    fn build(&self, app: &mut App) {
        // The resource is inserted in `main()` from the unified `settings.ron`
        // and persisted by `crate::settings::AppSettingsPlugin`; this plugin
        // only registers the type for the reflection / debug-UI path.
        app.register_type::<UnitsSettings>();
    }
}
