use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

// Authored world-definition types now live in `thalos_world`. Re-exported here
// **crate-internal only** so physics-internal `crate::types::…` references keep
// resolving; this is no longer public API. External code must import these from
// `thalos_world` directly.
// `AU_TO_METERS` is referenced only from `#[cfg(test)]` modules, so the
// re-export reads as unused in a plain (non-test) build — allow that.
#[allow(unused_imports)]
pub(crate) use thalos_world::{
    AU_TO_METERS, BodyDefinition, BodyId, BodyKind, G, OrbitalElements, SolarSystemDefinition,
    StateVector, keplerian_basis, orbital_elements_to_cartesian,
};

/// Ship attitude state. Kept separate from [`StateVector`] so trajectory
/// prediction (which doesn't care about orientation) stays cheap.
///
/// `orientation` is the body→world quaternion; `angular_velocity` is
/// expressed in the **body frame** (rad/s) — convention `Iω̇ = τ` plays
/// out cleanly when both `ω` and `τ` are in body coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttitudeState {
    pub orientation: DQuat,
    pub angular_velocity: DVec3,
}

impl Default for AttitudeState {
    fn default() -> Self {
        Self {
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
        }
    }
}

/// Static physical properties needed to integrate ship attitude and thrust.
///
/// `moment_of_inertia` is the principal-axis MOI tensor's diagonal in
/// kg·m², expressed in the body frame. Off-diagonal terms are assumed
/// zero — adequate for axially-symmetric ship stacks. `max_torque` is
/// the per-axis torque cap from all reaction-wheel-providing parts
/// summed, in N·m.
///
/// `thrust_n`, `mass_flow_kg_per_s`, and `dry_mass_kg` are the current
/// aggregate values for whatever ship configuration the game layer has
/// made active. Current ship mass is tracked separately on
/// [`crate::Simulation`] because it changes as fuel burns; once it
/// reaches `dry_mass_kg` thrust cuts off cleanly.
#[derive(Debug, Clone, Copy)]
pub struct ShipParameters {
    pub moment_of_inertia: DVec3,
    /// Center of mass in the craft body frame (`X=right, Y=nose, Z=dorsal`),
    /// metres from the root-part origin. Aircraft land on gear that straddle
    /// this point, so the ground-physics rigid body must rotate about it
    /// (Avian `CenterOfMass`) rather than the nose origin, or it tips over.
    /// `moment_of_inertia` is already expressed about this point.
    pub center_of_mass: DVec3,
    pub max_torque: DVec3,
    /// Attitude torque (N·m) the gimballed engines produce **at full thrust**,
    /// per body axis (`x` pitch, `z` yaw; `y` roll is ~0 — a centred bell
    /// can't roll). The *effective* authority scales with the current thrust
    /// fraction (throttle), so gimbal steering vanishes at zero throttle and
    /// during coast — the game applies `gimbal_torque_full · throttle` on top
    /// of `max_torque` in both the controller's authority sum and the realized
    /// torque. Aggregated from each gimballed engine's `thrust · sin(range) ·
    /// CoM arm` (`thalos_runtime::staging`). Zero for aircraft / fixed-bell
    /// rockets. See `docs/simulation/aerodynamics.md` *Thrust vectoring*.
    pub gimbal_torque_full: DVec3,
    pub thrust_n: f64,
    pub mass_flow_kg_per_s: f64,
    /// Dry mass — the floor under which `Simulation::ship_mass_kg` cannot
    /// fall, and the threshold below which thrust stops being applied.
    /// "Out of fuel" in physical units rather than an arbitrary numerical
    /// safety floor.
    pub dry_mass_kg: f64,
    /// KSP-style crash tolerance: the surface-relative approach speed
    /// (m/s) above which a terrain contact destroys the whole craft. The
    /// game's `detect_terrain_impact` compares the pre-contact approach
    /// speed against this. `f64::INFINITY` means "indestructible" — the
    /// sentinel for a craft whose real stats haven't been pushed yet, and
    /// for EVA (no Avian contact damage). See `docs/simulation/surface.md`.
    pub impact_tolerance_m_s: f64,
    /// Aerodynamic reference (frontal) area in m², for the aggregate
    /// bluff-body drag of this craft. Derived from the ship's geometry
    /// (`ShipStats::frontal_area_m2`); 0 means "no aerodynamic drag" (sentinel
    /// default and EVA). See `docs/simulation/aerodynamics.md`.
    pub reference_area_m2: f64,
    /// Aggregate bluff-body drag coefficient (dimensionless). Blunt capsule
    /// ~1.0–1.4, streamlined rocket ~0.3–0.5. 0 means "no aerodynamic drag".
    pub drag_coefficient: f64,
}

impl Default for ShipParameters {
    fn default() -> Self {
        // Sentinel values: nonzero MOI to avoid divide-by-zero, zero
        // torque so a ship with no parameters set can't accidentally
        // accept attitude commands. Zero thrust = drifting until a real
        // ship is spawned and pushes its blueprint stats in. Dry mass
        // sits at the safety floor so the integrator's mass never
        // divides by zero before a real ship has been pushed in.
        Self {
            moment_of_inertia: DVec3::ONE,
            center_of_mass: DVec3::ZERO,
            max_torque: DVec3::ZERO,
            // No thrust vectoring until a real ship pushes its gimbal geometry.
            gimbal_torque_full: DVec3::ZERO,
            thrust_n: 0.0,
            mass_flow_kg_per_s: 0.0,
            dry_mass_kg: MIN_SHIP_MASS_KG,
            // Indestructible until a real ship pushes its stats — same
            // sentinel philosophy as the zero torque above.
            impact_tolerance_m_s: f64::INFINITY,
            // No drag until a real ship pushes its frontal area / Cd.
            reference_area_m2: 0.0,
            drag_coefficient: 0.0,
        }
    }
}

impl ShipParameters {
    /// EVA "vessel" parameters — the player on foot is treated as a
    /// single-part vessel à la KSP. 90 kg dry mass, no thrust, no
    /// reaction-wheel torque (orientation is driven by walking input,
    /// not by `ControlInput::torque_command`). MOI is a back-of-the-
    /// envelope value for a 1.8 m × 0.6 m capsule of uniform density
    /// and is only used to keep the integrator from dividing by zero.
    pub fn eva() -> Self {
        Self {
            moment_of_inertia: DVec3::new(15.0, 1.5, 15.0),
            center_of_mass: DVec3::ZERO,
            max_torque: DVec3::ZERO,
            gimbal_torque_full: DVec3::ZERO,
            thrust_n: 0.0,
            mass_flow_kg_per_s: 0.0,
            dry_mass_kg: 90.0,
            // On-foot contact damage is out of scope (EVA doesn't use Avian
            // contact resolution); never destroyed by terrain impact.
            impact_tolerance_m_s: f64::INFINITY,
            // EVA has no aerodynamic model (a jetpack/suit-drag pass is future
            // work); aero systems skip EVA entirely.
            reference_area_m2: 0.0,
            drag_coefficient: 0.0,
        }
    }
}

/// What kind of vessel the player is currently controlling.
///
/// Modelled after KSP's `VesselType`: an EVA Kerbal is just another
/// vessel, with its own `CraftState`, `ShipParameters`, orbit, and
/// authority — distinguished from a rocket only by this tag and by the
/// shape of its parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VesselKind {
    #[default]
    Ship,
    Eva,
}

/// Hard numerical floor on ship mass — keeps the integrator from dividing
/// by zero before a real ship has been spawned and `dry_mass_kg` set. Once
/// a ship is spawned, its actual `dry_mass_kg` is the operative floor.
pub(crate) const MIN_SHIP_MASS_KG: f64 = 1.0;

/// Player attitude + thrust command sampled each frame and pushed into
/// the simulation via [`crate::simulation::Simulation::set_control`].
///
/// `torque_command` is in body frame, components in `[-1, 1]`. Each
/// axis is multiplied by the matching [`ShipParameters::max_torque`]
/// component to produce the actual torque applied. `throttle` is the
/// player's commanded engine throttle, in `[0, 1]`. The bridge gates
/// this on fuel availability before sending; the simulation trusts the
/// value it receives and applies thrust along the body nose direction.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControlInput {
    pub torque_command: DVec3,
    pub sas_enabled: bool,
    pub throttle: f64,
}

/// A timestamped state for a body evaluated from the active body trajectory
/// provider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyState {
    pub id: BodyId,
    pub epoch: crate::canonical::Epoch,
    pub position: DVec3,
    pub velocity: DVec3,
    pub orientation: DQuat,
    pub angular_velocity: DVec3,
    pub mass_kg: f64,
    pub gm: f64,
    pub radius_m: f64,
}

/// Snapshot of all body states at a given time.
pub type BodyStates = Vec<BodyState>;

/// A single sample of the ship's propagated trajectory.
///
/// Under the analytical patched-conics propagator there is one gravitational
/// source per sample — the SOI body — so rendering, colouring, and encounter
/// detection all share the single `anchor_body` field. `ref_pos` is the
/// anchor body's heliocentric position at `time`, cached on the sample so
/// the renderer can compute the anchor-relative position without an
/// ephemeris query per sample per frame.
#[derive(Debug, Clone, Copy)]
pub struct TrajectorySample {
    pub time: f64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub anchor_body: BodyId,
    /// `anchor_body`'s position at `time`, cached for cheap rendering.
    pub ref_pos: DVec3,
}
