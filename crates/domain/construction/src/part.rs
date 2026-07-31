use crate::catalog::{
    EngineGeometry, EngineOptimization, IntakeCapture, IntakeRequirement, PodGeometry,
};
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
    /// Silhouette + length-to-diameter ratio (capsule vs aircraft cockpit
    /// cone). Copied from the catalog [`crate::PodSpec`] at spawn.
    pub geometry: PodGeometry,
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
/// parameters and its mount. A single tapered, swept, dihedral panel; a
/// mirrored pair is two of these as separate entities linked by a
/// [`crate::SymmetryGroup`] (KSP-style), not one part drawing both sides.
///
/// Geometry is authored in the host's local frame at mount time (see
/// [`crate::wing_mesh`]): span is the half-span (root→tip of one panel),
/// chord runs fore/aft along the host body axis, thickness is the airfoil
/// depth. `dry_mass` is catalog-derived from planform area.
///
/// Control surfaces are authored as *parameters of the wing*
/// ([`Wing::control_surfaces`]), not separate parts, so the wing stays one
/// authored unit. Landing gear, by contrast, is its own footprint part kind.
#[derive(Component, Debug, Clone, PartialEq)]
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
    /// area, per panel; a mirrored pair is two entities, each one panel).
    pub dry_mass: f32,
    /// Trailing-edge control surfaces hinged into this panel (ailerons,
    /// elevator, rudder). Empty for a plain lifting surface. The mesh
    /// builder notches these out of the loft and meshes each as a separate
    /// hinged sub-mesh; the game animates them from the fly-by-wire command.
    pub control_surfaces: Vec<ControlSurface>,
}

/// What a [`ControlSurface`] controls. This selects which arbitrated
/// command axis drives the surface's deflection and how it reflects across
/// a mirrored pair (see [`crate::SymmetryGroup`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ControlSurfaceRole {
    /// Roll control. **Anti-symmetric** across a mirror: the left and right
    /// ailerons deflect in opposite senses, so they take the roll command
    /// with a per-side sign (keyed on the panel's mount azimuth).
    Aileron,
    /// Pitch control. **Symmetric**: both halves of a tailplane deflect
    /// together; takes the pitch command.
    Elevator,
    /// Yaw control. Lives on a near-vertical fin; takes the yaw command.
    Rudder,
    /// High-lift device. **Symmetric**, driven by the flap lever (a craft
    /// *configuration*, not an attitude command): deflects trailing-edge-down
    /// with the lever setting. Its authored window area/chord also derives the
    /// craft's flap ΔCL/ΔCD (see `thalos_runtime::aero::build_ship_aero_config`),
    /// so sizing the flaps in the shipyard changes the landing performance.
    Flap,
    /// Spoiler / speedbrake panel. **Symmetric**, driven by the brakes toggle:
    /// raises trailing-edge-up when the brakes are engaged, dumping lift and
    /// adding drag. Window area derives the craft's speedbrake ΔCD/ΔCL.
    Spoiler,
}

/// A trailing-edge control surface hinged into a [`Wing`] panel.
///
/// Authored against the panel's own geometry so it scales with the wing:
/// `span_start`/`span_end` are fractions of the half-span (root = 0, tip =
/// 1) and `chord_fraction` is the fraction of the *local* chord, measured
/// from the trailing edge forward, that the surface occupies. The hinge
/// line runs spanwise at that chord station.
///
/// This is the single authored record both the visual layer and a future
/// per-surface force model read: the visual layer takes the hinge line and
/// `max_deflection`; the force model takes the area/arm derived from the
/// same window (see [`crate::wing_mesh::control_surface_geometry`]).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ControlSurface {
    pub role: ControlSurfaceRole,
    /// Inboard edge of the surface, as a fraction of the half-span (0 = root).
    pub span_start: f32,
    /// Outboard edge of the surface, as a fraction of the half-span (1 = tip).
    pub span_end: f32,
    /// Fraction of the local chord the surface occupies, from the trailing
    /// edge forward. The hinge sits at this chord station.
    pub chord_fraction: f32,
    /// Maximum deflection magnitude, radians (trailing edge down positive).
    pub max_deflection: f32,
}

/// A self-contained landing-gear assembly — a "gearbox" footprint part that
/// draws *all* of its legs in one mesh, like the wing model drew both panels
/// in one mesh. This deliberately does **not** use [`crate::SymmetryGroup`]:
/// `gear_main` houses a left/right main pair internally; `gear_nose` houses a
/// single centred leg. The editor special-cases the [`crate::CatalogEntry`]
/// kind so a gear is always placed as a single mount regardless of the Mirror
/// toggle.
///
/// Geometry is authored in the host's local frame (see [`crate::gear_mesh`]):
/// the strut runs out along the mount radial (toward the belly) from the host
/// skin, and the wheel hangs at the strut's end with a lateral axle so it rolls
/// fore/aft. `track_fraction` is the lateral leg spacing as a fraction of the
/// host radius (0 → a single centred leg); the leg count derives from it.
/// `dry_mass` is catalog-derived from strut length, wheel mass, and leg count.
///
/// **Future** (`docs/gameplay/construction.md` §4.4): the fuselage will recess-morph to
/// house the gearbox inside the belly. For now it sits at/below the belly with
/// no skin deformation.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Gear {
    /// Length of each strut from the host skin to the wheel, metres.
    pub strut_length: f32,
    /// Wheel radius, metres.
    pub wheel_radius: f32,
    /// Lateral offset of each main leg as a fraction of the host radius. `0.0`
    /// means a single centred leg (nose gear); `> 0.0` means a left/right main
    /// pair at `±track_fraction × host_radius`. Catalog-derived (copied from
    /// [`crate::GearSpec`] at spawn), so it is fixed per part kind.
    pub track_fraction: f32,
    /// Wheels per leg, fore/aft (a tandem **bogie** when `> 1`). Catalog-derived
    /// like `track_fraction`. The mesh draws this many wheels per leg.
    pub wheels_per_leg: u8,
    /// Catalog-derived structural mass, kg (struts + wheels for every leg).
    pub dry_mass: f32,
}

impl Gear {
    /// Number of legs this gearbox draws: a left/right main pair when a track
    /// is set, otherwise a single centred leg. The mesh, mass, and editor
    /// placement all read leg count from here so they never disagree.
    pub fn legs(&self) -> u8 {
        if self.track_fraction > 0.0 { 2 } else { 1 }
    }
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

/// A **stationed-loft fuselage** (`docs/gameplay/construction.md` §4.2): the advanced
/// airframe body that replaces a straight tank-cylinder + cone tailcone with
/// one continuous, upswept superellipse loft. Parameterised by high-level
/// airliner numbers; [`crate::fuselage_mesh`] generates the cross-section
/// stations and the skin, and exposes the host-skin query
/// ([`crate::fuselage_mesh::skin_radius`] / [`crate::fuselage_mesh::v_offset_at`])
/// that surface mounts (wings, gear, nacelles) ride.
///
/// `max_width` is the declared barrel diameter; like [`FuelTank::diameter`]
/// it is **overridden by the parent's mating-node diameter** when this part
/// is node-stacked under another (`sizing::propagate_node_sizes`), and every
/// other extent scales with it so the authored proportions are preserved.
///
/// **Future** (`docs/gameplay/construction.md` §5): a wet/role-filled fuselage carries
/// fuel/crew/cargo in its integrated volume; today the airliner body is
/// structure-only (empty storage whitelist) and fuel lives in the wet wings.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Fuselage {
    /// Overall body length along the axis, metres.
    pub length: f32,
    /// Barrel cross-section width (X) and height (Z), metres. `max_width` is
    /// the declared/overridable diameter; `max_height` sets the section
    /// aspect (equal → circular).
    pub max_width: f32,
    pub max_height: f32,
    /// Superellipse roundness `∈ [0, 1]`: `1` → round, `0` → boxy belly.
    pub roundness: f32,
    /// Fraction of length spent on the parametric nose taper (`0` → no nose;
    /// the barrel starts at full diameter). The fuselage owns its nose — there
    /// is no separate nose-cone pod.
    pub nose_fraction: f32,
    /// Nose profile shape `∈ [0, 1]`: `0` → a straight cone (pointed), `1` → a
    /// rounded radome (convex ellipsoidal, the airliner look). Blends between
    /// the two. Tune directly alongside `nose_fraction`/`nose_droop` to dial in
    /// any nose without presets.
    pub nose_bluntness: f32,
    /// Fraction of length spent on the tailcone neck.
    pub tail_fraction: f32,
    /// Nose centerline droop, metres (lowers the nose tip).
    pub nose_droop: f32,
    /// Tail centerline upsweep, metres (raises the tail — the airliner look).
    pub tail_upsweep: f32,
    /// Diameter the tailcone necks down to at the tip, metres. `0` → the
    /// tailcone closes to a point (a clean classic boat-tail); `> 0` → it is
    /// truncated with a flat cap (the APU-style blunt tailcone).
    pub tail_tip_diameter: f32,
    /// Tail tip profile `∈ [0, 1]`: `0` → a straight cone (sharply pointed),
    /// `1` → a rounded ogive dome. Symmetric to `nose_bluntness` for the aft
    /// end, so each end is shaped independently.
    pub tail_bluntness: f32,
    /// Catalog-derived structural mass, kg.
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
    /// Design point the nozzle is sized for. Sets the expansion ratio, and with
    /// it the exit pressure — which is what decides whether the exhaust is over-
    /// or underexpanded at a given altitude. Read by the plume renderer
    /// (`docs/rendering/plume.md`) to shape the exhaust from the pad to vacuum.
    pub optimized_for: EngineOptimization,
    pub requires_atmosphere: bool,
    pub intake_requirement: Option<IntakeRequirement>,
    pub builtin_intake: Option<IntakeCapture>,
    pub diameter: f32,
    /// Thrust in vacuum, N.
    pub thrust: f32,
    /// Specific impulse in vacuum, s.
    pub isp: f32,
    /// Specific impulse at the 1-atm reference pressure
    /// ([`ISP_REFERENCE_PRESSURE_PA`]), s. `None` = pressure-insensitive
    /// (air-breathers, which lapse with density instead). For a rocket this
    /// implicitly authors the nozzle exit area: thrust falls linearly with
    /// ambient pressure (`F = F_vac − p·A_e`) while mass flow stays fixed,
    /// so a vacuum-optimized bell is heavily penalized at sea level. See
    /// [`Engine::pressure_thrust_factor`].
    pub sea_level_isp: Option<f32>,
    pub dry_mass: f32,
    /// Which reactants this engine consumes, as mass fractions of its
    /// total mass flow. Fractions must sum to 1.0; resources must all be
    /// mass-bearing. Non-mass-bearing resources (like electricity) belong
    /// in `power_draw_kw`, not here.
    pub reactants: Vec<ReactantRatio>,
    /// Continuous electrical draw while the engine is firing, kW.
    /// 0 for chemical engines.
    pub power_draw_kw: f32,
    /// Peak thrust-vector deflection (degrees) from the nose axis; `0` = fixed
    /// bell (no gimbal). Aggregated into `ShipParameters::gimbal_torque_full`
    /// (thrust × sin(range) × CoM arm) so a gimballed engine steers the craft
    /// in pitch/yaw under power. See `docs/simulation/aerodynamics.md` *Thrust vectoring*.
    pub gimbal_range_deg: f32,
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

    /// Fraction of vacuum thrust deliverable at `ambient_pressure_pa`.
    ///
    /// Nozzle back-pressure: `F(p) = F_vac − p·A_e` is exactly linear in
    /// ambient pressure, so the authored `sea_level_isp` pins the line at the
    /// 1-atm reference and vacuum pins it at 1 — no explicit exit area needed
    /// (`A_e = (1 − Isp_sl/Isp_vac)·F_vac / p_ref` falls out). Mass flow is a
    /// pump property and does **not** change with altitude; callers keep
    /// `mdot = F_vac/(Isp_vac·g0)` and let effective Isp follow the thrust.
    /// Clamped to `[0, 1]`: pressures beyond the separation point simply kill
    /// thrust rather than going negative. `None` / non-positive inputs → 1.
    pub fn pressure_thrust_factor(&self, ambient_pressure_pa: f64) -> f64 {
        let Some(sea_level_isp) = self.sea_level_isp else {
            return 1.0;
        };
        if ambient_pressure_pa <= 0.0 || self.isp <= 0.0 || sea_level_isp >= self.isp {
            return 1.0;
        }
        let loss_at_ref = 1.0 - sea_level_isp as f64 / self.isp as f64;
        (1.0 - loss_at_ref * ambient_pressure_pa / ISP_REFERENCE_PRESSURE_PA).clamp(0.0, 1.0)
    }
}

/// Reference ambient pressure (Pa) at which [`Engine::sea_level_isp`] is
/// authored — one standard atmosphere.
pub const ISP_REFERENCE_PRESSURE_PA: f64 = 101_325.0;

#[cfg(test)]
mod pressure_thrust_tests {
    use super::*;
    use crate::catalog::{EngineGeometry, EngineOptimization};
    use crate::resource::Resource;

    fn rocket(isp: f32, sea_level_isp: Option<f32>) -> Engine {
        Engine {
            model: "test".into(),
            geometry: EngineGeometry::default(),
            optimized_for: EngineOptimization::Atmosphere,
            requires_atmosphere: false,
            intake_requirement: None,
            builtin_intake: None,
            diameter: 2.5,
            thrust: 500_000.0,
            isp,
            sea_level_isp,
            dry_mass: 450.0,
            reactants: vec![ReactantRatio {
                resource: Resource::Methane,
                mass_fraction: 1.0,
            }],
            power_draw_kw: 0.0,
            gimbal_range_deg: 0.0,
        }
    }

    #[test]
    fn vacuum_gives_full_thrust() {
        let e = rocket(355.0, Some(330.0));
        assert_eq!(e.pressure_thrust_factor(0.0), 1.0);
    }

    #[test]
    fn one_atm_matches_authored_sea_level_isp() {
        // At the reference pressure the thrust (and, with fixed mdot, the
        // effective Isp) must land exactly on the authored sea-level rating.
        let e = rocket(355.0, Some(330.0));
        let f = e.pressure_thrust_factor(ISP_REFERENCE_PRESSURE_PA);
        assert!((f - 330.0 / 355.0).abs() < 1e-12);
        assert!((e.isp as f64 * f - 330.0).abs() < 1e-9);
    }

    #[test]
    fn loss_is_linear_in_pressure() {
        let e = rocket(355.0, Some(330.0));
        let half = e.pressure_thrust_factor(ISP_REFERENCE_PRESSURE_PA * 0.5);
        let full = e.pressure_thrust_factor(ISP_REFERENCE_PRESSURE_PA);
        assert!(((1.0 - half) * 2.0 - (1.0 - full)).abs() < 1e-12);
    }

    #[test]
    fn vacuum_bell_is_crippled_but_never_negative() {
        // A big vacuum nozzle loses most of its thrust at 1 atm and clamps to
        // zero (flow separation) instead of going negative at higher pressure.
        let e = rocket(380.0, Some(120.0));
        let f = e.pressure_thrust_factor(ISP_REFERENCE_PRESSURE_PA);
        assert!((f - 120.0 / 380.0).abs() < 1e-12);
        assert_eq!(
            e.pressure_thrust_factor(ISP_REFERENCE_PRESSURE_PA * 3.0),
            0.0
        );
    }

    #[test]
    fn unauthored_or_degenerate_is_pressure_insensitive() {
        assert_eq!(
            rocket(355.0, None).pressure_thrust_factor(ISP_REFERENCE_PRESSURE_PA),
            1.0
        );
        // sea-level >= vacuum is a nonsense authoring; fail safe to 1.
        assert_eq!(
            rocket(355.0, Some(400.0)).pressure_thrust_factor(ISP_REFERENCE_PRESSURE_PA),
            1.0
        );
    }
}
