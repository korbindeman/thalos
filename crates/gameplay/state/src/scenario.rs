//! Spawn-scenario vocabulary: which start the session was asked for.
//! Placement itself (deferred descent/runway finishers, site search, descent
//! tuning) stays with the runtime's `spawn` / `runway` modules.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Which start scenario the session was asked for (`just game [mode]`,
/// `THALOS_SPAWN`, the start screen, or a respawn/relaunch picker), so the
/// deferred placement systems (the runtime's `runway` module and
/// `refine_descent_spawn`) can tell which path they belong to.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpawnSituation {
    /// Ship in the low equatorial parking orbit (default).
    ShipOrbit,
    /// Ship in a low polar parking orbit (same altitude as [`Self::ShipOrbit`],
    /// inclination ≈ 90°).
    PolarOrbit,
    /// Player on foot at the sub-stellar point.
    Eva,
    /// Ship descending toward a landing site over land.
    Landing,
    /// Ship already low and slow over a flat dry patch.
    FinalApproach,
    /// Aircraft parked at rest on the Thalos surface runway, lined up on the
    /// centerline ready for a takeoff roll. Placed by the runtime's `runway` module.
    Runway,
    /// Aircraft airborne on short final, lined up with the runway centerline
    /// and descending toward it. Placed by the runtime's `runway` module.
    RunwayApproach,
    /// Saturn rocket standing vertically on a default-spaceport launchpad.
    /// Placed by the runtime's `runway` module through the shared launchpad placement core.
    Launch,
    /// Meridian aircraft at ~15,000 ft (~4,600 m AGL), flying level at cruise
    /// speed over dry land. Placed by the runtime's `refine_descent_spawn`.
    Cruise,
}

impl SpawnSituation {
    /// Parse the `just game [mode]` argument / `THALOS_SPAWN` value. Unknown
    /// values warn and fall back to the ship orbit.
    pub fn from_request(request: &str) -> Self {
        match request.trim().to_ascii_lowercase().as_str() {
            "eva" => Self::Eva,
            "land" | "landing" | "descent" => Self::Landing,
            "final" | "final-approach" | "final_approach" | "approach" => Self::FinalApproach,
            "runway" | "rwy" => Self::Runway,
            "runway-approach" | "runway_approach" | "rwy-approach" | "approach-runway" => {
                Self::RunwayApproach
            }
            "launch" | "launchpad" | "pad" => Self::Launch,
            "cruise" | "cruising" => Self::Cruise,
            "polar" | "polar-orbit" | "polar_orbit" => Self::PolarOrbit,
            "" | "orbit" | "ship" => Self::ShipOrbit,
            other => {
                eprintln!("  Unknown spawn mode '{other}'; defaulting to ship orbit.");
                Self::ShipOrbit
            }
        }
    }

    /// True for the two runway scenarios, which the runtime's `runway` module finishes once
    /// terrain is resident (and which load the aircraft blueprint instead of
    /// the default rocket).
    pub fn is_runway(self) -> bool {
        matches!(self, Self::Runway | Self::RunwayApproach)
    }

    /// True for starts that build the canonical spaceport before placing the
    /// craft: the two runway starts and the launchpad rocket start.
    pub fn is_spaceport(self) -> bool {
        self.is_runway() || matches!(self, Self::Launch)
    }

    /// True for scenarios that fly the Meridian aircraft (runway + cruise).
    pub fn is_aircraft(self) -> bool {
        matches!(self, Self::Runway | Self::RunwayApproach | Self::Cruise)
    }

    /// True when the surface state is installed by a *deferred*, terrain-aware
    /// placement system (the runtime's `runway` module for the runway scenarios,
    /// the runtime's `refine_descent_spawn` for the descents and cruise) rather than seeded
    /// directly in `main.rs`. The settle gate must wait for that placement
    /// before judging whether tiles at the (then-known) site have settled.
    pub fn has_deferred_placement(self) -> bool {
        self.is_spaceport() || self.is_descent()
    }

    /// Ship blueprint to load for this scenario. Aircraft scenarios fly the
    /// Meridian jetliner; everything else flies the default rocket.
    pub fn ship_blueprint_path(self) -> &'static str {
        match self {
            Self::Launch => "ships/saturn.ron",
            _ if self.is_aircraft() => "ships/meridian.ron",
            _ => "ships/apollo.ron",
        }
    }

    /// Human label for the descent-style scenarios (used by boot logging and
    /// the scenario menu). `const` so the runtime's descent profiles can cite
    /// it as the one label source.
    pub const fn descent_label(self) -> Option<&'static str> {
        match self {
            Self::Landing => Some("landing approach"),
            Self::FinalApproach => Some("final approach"),
            Self::Cruise => Some("cruise"),
            _ => None,
        }
    }

    /// True for the scenarios placed by the deferred descent finisher (the
    /// runtime's `refine_descent_spawn`): the two descents and cruise.
    pub fn is_descent(self) -> bool {
        matches!(self, Self::Landing | Self::FinalApproach | Self::Cruise)
    }
}
