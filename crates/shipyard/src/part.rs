use crate::catalog::{EngineGeometry, IntakeCapture, IntakeRequirement};
use crate::resource::Resource;
use bevy::prelude::*;

/// Marker — entity is a ship part.
#[derive(Component, Debug, Clone, Copy)]
#[require(Transform, Visibility)]
pub struct Part;

/// Surface finish for a ship part. Drives which procedural shader /
/// parameter set the rendering layer picks. Only one variant today; the
/// enum is here so call sites (editor palette, blueprint round-trip) can
/// be extended additively.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MaterialKind {
    #[default]
    StainlessSteel,
}

/// Attached to any part whose surface should be driven by the ship
/// rendering layer (as opposed to a plain `StandardMaterial`). The
/// rendering layer reacts to the `kind` field; parts without this
/// component keep whatever material the editor / game assigned.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PartMaterial {
    pub kind: MaterialKind,
}

#[derive(Component, Debug, Clone)]
pub struct CommandPod {
    pub model: String,
    pub diameter: f32,
    pub dry_mass: f32,
    /// Torque this pod's built-in reaction wheel can produce per body
    /// axis, N·m. The blueprint loader auto-attaches a matching
    /// [`ReactionWheel`] component so the runtime aggregator only has
    /// to query for the capability.
    pub reaction_wheel_torque: f32,
}

/// Capability component: this part contributes reaction-wheel torque to
/// the ship's attitude control budget. Built-in to every [`CommandPod`];
/// future dedicated reaction-wheel parts attach the same component.
#[derive(Component, Debug, Clone, Copy)]
pub struct ReactionWheel {
    /// Maximum torque per body axis, N·m. Symmetric — reaction wheels
    /// are isotropic. Per-axis-asymmetric torque (RCS arrangements)
    /// belongs on a separate component.
    pub max_torque: f32,
}

/// Runtime engine activation gate. Disabled engines do not contribute
/// thrust, mass flow, fuel demand, burn-duration estimates, or plume
/// state. This is deliberately independent of staging: a later staging
/// system can mutate this component, but manual toggles, failures, and
/// editor test fire controls can use the same surface.
#[derive(Component, Debug, Clone, Copy)]
pub struct EngineActivation {
    pub enabled: bool,
}

impl Default for EngineActivation {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Per-engine runtime thrust output, N. Updated each frame by the game
/// crate from the gated effective throttle. Stays at zero while the
/// engine isn't firing. Plumbing for visual feedback (current temporary
/// red mesh tint, future particle/plume effects) so consumers don't
/// have to rederive `engine.thrust * throttle.effective` themselves and
/// stay in sync with whatever gating the bridge applies (fuel-out,
/// auto-burn vs. manual, warp-disabled, etc.).
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EngineThrust {
    pub current_n: f32,
}

/// Fuel crossfeed capability for the attach graph. When disabled, fuel
/// routing does not traverse through this part. Decouplers default to
/// `enabled = false`; ordinary structural parts, tanks, pods, and engines
/// default to `enabled = true`.
#[derive(Component, Debug, Clone, Copy)]
pub struct FuelCrossfeed {
    pub enabled: bool,
}

impl Default for FuelCrossfeed {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Parametric in radius: `diameter` drives this part's `top` node when it
/// is a ship root; when attached to a parent, the parent's node diameter
/// overrides via `sizing::propagate_node_sizes`.
#[derive(Component, Debug, Clone)]
pub struct Decoupler {
    pub diameter: f32,
    pub ejection_impulse: f32,
    pub dry_mass: f32,
}

/// Marker: this part has a silhouette that a neighboring [`ShroudProvider`]
/// can wrap with an auto-generated shroud. Inserted at spawn on parts
/// that are shroudable (currently: engines).
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Shroudable;

/// Marker: when this part's `top` node is attached to a [`Shroudable`]'s
/// `bottom` node, the editor spawns a shroud entity as its child, sized
/// to cover the shrouded silhouette. The shroud stays with the provider
/// on staging, matching the KSP-style "interstage" convention.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct ShroudProvider;

/// `diameter` is the `top` diameter (used when this part is the root);
/// `target_diameter` is always the `bottom` diameter. Child-attached
/// adapters get their `top` overridden from the parent.
#[derive(Component, Debug, Clone)]
pub struct Adapter {
    pub diameter: f32,
    pub target_diameter: f32,
    pub dry_mass: f32,
}

/// A parametric lifting surface — main wing, tailplane (horizontal
/// stabiliser), or fin (vertical stabiliser), distinguished only by its
/// parameters and its [`crate::SurfaceMount::side`]. A single tapered,
/// swept, dihedral panel per side; a `MountSide::Pair` part renders the
/// mirrored left+right pair from these same parameters.
///
/// Geometry is authored in the host's local frame at mount time (see
/// [`crate::wing_mesh`]): span is the half-span (root→tip of one panel),
/// chord runs fore/aft along the host body axis, thickness is the airfoil
/// depth. `dry_mass` is catalog-derived from planform area.
///
/// **Extension point (control surfaces).** Per `docs/construction.md`, a
/// wing's flaps / ailerons / elevators / rudder are planned as *parameters
/// of the wing*, not separate parts — a trailing-edge chord fraction +
/// spanwise window + a hinge/deflection descriptor will be added here as
/// optional fields, so the wing stays one authored unit. Landing gear, by
/// contrast, will be its own footprint part kind.
#[derive(Component, Debug, Clone)]
pub struct Wing {
    /// Half-span of one panel (host skin → tip), metres.
    pub span: f32,
    /// Chord at the root (host skin), metres.
    pub root_chord: f32,
    /// Chord at the tip, metres. `< root_chord` for a tapered wing.
    pub tip_chord: f32,
    /// Leading-edge sweep, radians. Positive sweeps the tip aft.
    pub sweep: f32,
    /// Dihedral, radians. Positive raises the tip above the root.
    pub dihedral: f32,
    /// Maximum airfoil thickness as a fraction of local chord (t/c).
    pub thickness: f32,
    /// Mounting incidence, radians. Positive pitches the leading edge up.
    pub incidence: f32,
    /// Catalog-derived structural mass, kg (= `mass_per_m2` × planform
    /// area, doubled for a mirrored pair).
    pub dry_mass: f32,
}

/// Pure geometry — contents live in [`crate::PartResources`]. A tank can
/// hold any resource; this part does not restrict which. `diameter`
/// drives node sizing when root; overridden by parent when attached.
#[derive(Component, Debug, Clone)]
pub struct FuelTank {
    pub diameter: f32,
    pub length: f32,
    pub dry_mass: f32,
}

/// Ambient-flow capture capability. This is not a stored resource: it is
/// external flow supplied by the current atmosphere and consumed by engines
/// that declare an [`IntakeRequirement`].
#[derive(Component, Debug, Clone)]
pub struct AirIntake {
    pub model: String,
    pub diameter: f32,
    pub length: f32,
    pub dry_mass: f32,
    pub capture: IntakeCapture,
}

/// A mass fraction of a single reactant relative to the engine's total
/// mass flow. For a methalox engine at O/F = 3.6:
/// `[(Methane, 0.217), (Lox, 0.783)]`.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReactantRatio {
    pub resource: Resource,
    pub mass_fraction: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Engine {
    pub model: String,
    pub geometry: EngineGeometry,
    pub requires_atmosphere: bool,
    pub intake_requirement: Option<IntakeRequirement>,
    pub builtin_intake: Option<IntakeCapture>,
    pub diameter: f32,
    /// Thrust in vacuum, N.
    pub thrust: f32,
    /// Specific impulse in vacuum, s.
    pub isp: f32,
    pub dry_mass: f32,
    /// Which reactants this engine consumes, as mass fractions of its
    /// total mass flow. Fractions must sum to 1.0; resources must all be
    /// mass-bearing. Non-mass-bearing resources (like electricity) belong
    /// in `power_draw_kw`, not here.
    pub reactants: Vec<ReactantRatio>,
    /// Continuous electrical draw while the engine is firing, kW.
    /// 0 for chemical engines.
    pub power_draw_kw: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineValidationError {
    /// Reactant mass fractions don't sum to 1.0 within tolerance.
    ReactantFractionsNotNormalized,
    /// A reactant references a non-mass-bearing resource (e.g. Electricity).
    ReactantNotMassBearing,
    /// Reactants list is empty — every engine must have at least one.
    NoReactants,
    /// A fraction is zero or negative.
    NonPositiveFraction,
}

impl Engine {
    /// Check invariants that the stats aggregator relies on. Call from
    /// tests, editors, or loaders; stats computation assumes these hold
    /// and will produce meaningless numbers otherwise.
    pub fn validate(&self) -> Result<(), EngineValidationError> {
        if self.reactants.is_empty() {
            return Err(EngineValidationError::NoReactants);
        }
        let mut sum = 0.0_f32;
        for r in &self.reactants {
            if r.mass_fraction <= 0.0 {
                return Err(EngineValidationError::NonPositiveFraction);
            }
            if !r.resource.is_mass_bearing() {
                return Err(EngineValidationError::ReactantNotMassBearing);
            }
            sum += r.mass_fraction;
        }
        if (sum - 1.0).abs() > 1e-4 {
            return Err(EngineValidationError::ReactantFractionsNotNormalized);
        }
        Ok(())
    }
}
