//! Parts catalog. Single source of truth for part stats.
//!
//! Map keys are stable [`CatalogId`]s assigned once and never changed.
//! `display_name` is the mutable, user-facing label. Saved blueprints
//! reference parts by ID, not by name — renaming "Boreas" to "Aestus" is
//! a one-line catalog edit and old saves keep loading.
//!
//! Catalog entries fall into two broad shapes:
//! - Fixed-geometry parts (Pod, Engine, Intake): full stats, no per-instance
//!   geometry parameters.
//! - Parametric parts (Decoupler, Adapter, Tank, Wing): the entry holds
//!   recipes (mass per area, storage per volume) and the blueprint provides
//!   variable geometry.

use crate::part::{ControlSurface, ControlSurfaceRole, ReactantRatio};
use crate::resource::Resource;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub type CatalogId = String;

/// Marker component carrying the [`CatalogId`] that this part was spawned
/// from. Consumed at save time to round-trip blueprints back to their
/// catalog references.
#[derive(Component, Debug, Clone)]
pub struct CatalogRef {
    pub id: CatalogId,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct PartCatalog {
    pub parts: HashMap<CatalogId, CatalogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CatalogEntry {
    Pod(PodSpec),
    Engine(EngineSpec),
    Intake(IntakeSpec),
    Decoupler(DecouplerSpec),
    Adapter(AdapterSpec),
    Tank(TankSpec),
    Fuselage(FuselageSpec),
    Wing(WingSpec),
    Gear(GearSpec),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSpec {
    pub display_name: String,
    /// Silhouette: a blunt crew capsule (default) or a slender aircraft
    /// cockpit cone. Pre-cockpit saves omit it and load as `Capsule`, so
    /// existing pods are byte-identical.
    #[serde(default)]
    pub geometry: PodGeometry,
    pub diameter: f32,
    pub dry_mass: f32,
    pub reaction_wheel_torque: f32,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

/// Command-pod silhouette. Selects the editor / in-game mesh **and** the
/// body's length-to-diameter ratio. Mirrors [`EngineGeometry`]: a backend
/// picks geometry, never its own physics. Node layout, MOI cylinder,
/// collider height, and visual height all read
/// [`PodGeometry::length_factor`] so rendering and physics never disagree
/// on the pod's extent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PodGeometry {
    /// Blunt crew capsule — a truncated cone, slightly taller than wide.
    #[default]
    Capsule,
    /// Aircraft nose: a rounded ogive radome (see [`crate::build_cockpit_mesh`])
    /// about as long as it is wide, blunt and convex rather than a needle
    /// spike. A simple symmetric placeholder; a real nose-low model can replace
    /// it later.
    AircraftCockpit,
    /// **Inline cockpit**: an internal command module with *no body mesh of its
    /// own*. It surface-mounts inside a parametric fuselage near the nose and
    /// supplies command / crew / reaction-wheel capability; the fuselage's own
    /// parametric nose is the visible nose, and a windshield region is morphed
    /// into the skin (see `docs/construction.md` §5.6). Used in place of a
    /// nose-cone pod on loft-bodied aircraft.
    Inline,
}

impl PodGeometry {
    pub fn label(self) -> &'static str {
        match self {
            PodGeometry::Capsule => "Capsule",
            PodGeometry::AircraftCockpit => "Aircraft cockpit",
            PodGeometry::Inline => "Inline cockpit",
        }
    }

    /// Body length as a multiple of the pod diameter. The single knob that
    /// keeps a cockpit cone longer than a capsule while every consumer
    /// (node offset, inertia, collider, mesh) stays in agreement. An inline
    /// cockpit has no body, so it contributes no length.
    pub fn length_factor(self) -> f32 {
        match self {
            PodGeometry::Capsule => 0.9,
            PodGeometry::AircraftCockpit => 1.0,
            PodGeometry::Inline => 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSpec {
    pub display_name: String,
    #[serde(default)]
    pub optimized_for: EngineOptimization,
    #[serde(default)]
    pub geometry: EngineGeometry,
    /// Air-breathing engines need a body atmosphere. This is a runtime gate;
    /// design-time stats still report their nominal thrust until the aero
    /// model grows environment-aware estimates.
    #[serde(default)]
    pub requires_atmosphere: bool,
    /// Ambient flow the engine needs at full rated thrust.
    #[serde(default)]
    pub intake_requirement: Option<IntakeRequirement>,
    /// Intake capture built into this engine's own housing. Jet nacelles use
    /// this; later buried/standalone engine cores can omit it and depend on
    /// separate intake parts.
    #[serde(default)]
    pub builtin_intake: Option<IntakeCapture>,
    pub diameter: f32,
    pub thrust: f32,
    pub isp: f32,
    pub dry_mass: f32,
    pub reactants: Vec<ReactantRatio>,
    pub power_draw_kw: f32,
    /// Peak thrust-vector deflection (degrees) the engine can gimbal from the
    /// nose axis. `0` (the default, e.g. jets and fixed bells) means no
    /// thrust vectoring. A gimballed engine steers the craft in pitch/yaw
    /// under power: the effective attitude torque is `thrust · sin(range) ·
    /// arm` about the CoM, wired into the fly-by-wire allocator alongside the
    /// reaction wheels. See `docs/aerodynamics.md` *Thrust vectoring*.
    #[serde(default)]
    pub gimbal_range_deg: f32,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntakeSpec {
    pub display_name: String,
    pub diameter: f32,
    pub length: f32,
    pub dry_mass: f32,
    pub capture: IntakeCapture,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AmbientIntakeKind {
    /// Oxygen-bearing atmospheric air for gas-turbine style engines.
    #[default]
    OxygenatedAir,
}

impl AmbientIntakeKind {
    pub fn label(self) -> &'static str {
        match self {
            AmbientIntakeKind::OxygenatedAir => "Oxygenated air",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IntakeRequirement {
    #[serde(default)]
    pub kind: AmbientIntakeKind,
    /// Effective intake capture area required at full rated thrust, m².
    pub area_m2: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IntakeCapture {
    #[serde(default)]
    pub kind: AmbientIntakeKind,
    /// Geometric/capture area, m².
    pub area_m2: f32,
    /// Scalar efficiency applied to area until a real aero/ram model exists.
    #[serde(default = "default_intake_efficiency")]
    pub efficiency: f32,
}

fn default_intake_efficiency() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineOptimization {
    Atmosphere,
    Vacuum,
    #[default]
    Balanced,
}

impl EngineOptimization {
    pub fn label(self) -> &'static str {
        match self {
            EngineOptimization::Atmosphere => "Atmosphere",
            EngineOptimization::Vacuum => "Vacuum",
            EngineOptimization::Balanced => "Balanced",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineGeometry {
    /// Bell/nozzle engine for stack-mounted rockets.
    #[default]
    RocketBell,
    /// Cylindrical jet/turbofan nacelle, optionally carried under a wing by
    /// an auto-generated pylon.
    JetNacelle,
}

impl EngineGeometry {
    pub fn label(self) -> &'static str {
        match self {
            EngineGeometry::RocketBell => "Rocket bell",
            EngineGeometry::JetNacelle => "Jet nacelle",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecouplerSpec {
    pub display_name: String,
    /// Linear scaling: dry_mass = mass_per_diameter × diameter (kg/m).
    pub mass_per_diameter: f32,
    /// Linear scaling: ejection impulse = factor × diameter (N·s/m).
    pub ejection_impulse_per_diameter: f32,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterSpec {
    pub display_name: String,
    /// dry_mass = wall_mass_per_m2 × frustum lateral surface area.
    pub wall_mass_per_m2: f32,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TankSpec {
    pub display_name: String,
    /// dry_mass = wall_mass_per_m2 × cylinder surface area (sides + caps).
    pub wall_mass_per_m2: f32,
    /// Whitelisted resources this tank can carry. Capacity is normally
    /// volume-scaled (`units_per_m3`) for tanks, but fixed capacity is also
    /// supported so the same schema works for small batteries, service bays,
    /// and future internal compartments.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
    /// Mass-fraction reactant ratios for stats aggregation.
    pub reactants: Vec<ReactantRatio>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuselageSpec {
    pub display_name: String,
    /// dry_mass = wall_mass_per_m2 × lofted skin surface area (see
    /// [`fuselage_surface_area`]). Same stainless-skin recipe as a tank.
    pub wall_mass_per_m2: f32,
    /// Whitelisted resource storage. The airliner body is structure-only
    /// (empty), but the schema supports a future wet/role-filled fuselage
    /// whose capacity scales with the integrated loft volume
    /// ([`fuselage_volume`]) — mirroring the wet-wing path.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

/// Placement-time default preset for a [`WingSpec`] catalog entry. Both
/// presets spawn the *same* [`crate::Wing`] part kind — this only selects the
/// initial geometry a freshly-placed instance starts at, so the user almost
/// always gets a sane shape without hand-tuning. It is **not** stored on the
/// part or blueprint (a stabilizer is a plain `Wing` underneath); reloading a
/// saved craft restores the user's tuned params, not the preset.
///
/// Orientation (horizontal tailplane vs vertical fin) is *not* encoded here —
/// it falls out of the mount azimuth at placement: a top-of-fuselage hit
/// extends the surface vertically (fin), a side hit horizontally (tailplane /
/// canard). Twin fins are just a `Stabilizer` placed under Mirror mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WingRole {
    /// Primary lifting surface: full span, positive default incidence, swept.
    #[default]
    Lift,
    /// Trim / control surface (tailplane, fin, canard): small span, ~0°
    /// incidence, low sweep, no dihedral. The empennage default.
    Stabilizer,
}

/// Placement-time default control surfaces for a freshly-placed wing, so a
/// new part flies with sensible ailerons / elevator / rudder without
/// hand-authoring. Like the rest of the catalog presets these are only the
/// *initial* values written into the blueprint; a saved craft restores the
/// user's authored set, not this default.
///
/// `role` picks the surface kind; for a [`WingRole::Stabilizer`] the
/// horizontal-tailplane (elevator) vs vertical-fin (rudder) split falls out
/// of the mount azimuth, exactly like the geometry preset — a near-side
/// (±X) mount is a tailplane, a near-dorsal (±Z) mount is a fin.
pub fn default_control_surfaces(role: WingRole, mount_angle: f32) -> Vec<ControlSurface> {
    match role {
        // Inboard flap + outboard aileron on the back quarter-chord, the
        // classic main-wing split. The flap window also sizes the craft's
        // high-lift ΔCL/ΔCD, so a default wing lands slow out of the box.
        WingRole::Lift => vec![
            ControlSurface {
                role: ControlSurfaceRole::Flap,
                span_start: 0.08,
                span_end: 0.50,
                chord_fraction: 0.30,
                max_deflection: 35.0_f32.to_radians(),
            },
            ControlSurface {
                role: ControlSurfaceRole::Aileron,
                span_start: 0.55,
                span_end: 0.95,
                chord_fraction: 0.25,
                max_deflection: 25.0_f32.to_radians(),
            },
        ],
        WingRole::Stabilizer => {
            // Horizontal when the panel extends sideways (|sin θ| large),
            // vertical when it extends up/down toward the dorsal axis.
            let horizontal = mount_angle.sin().abs() >= std::f32::consts::FRAC_1_SQRT_2;
            let role = if horizontal {
                ControlSurfaceRole::Elevator
            } else {
                ControlSurfaceRole::Rudder
            };
            vec![ControlSurface {
                role,
                span_start: 0.05,
                span_end: 0.95,
                chord_fraction: 0.35,
                max_deflection: 25.0_f32.to_radians(),
            }]
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WingSpec {
    pub display_name: String,
    /// dry_mass = wall_mass_per_m2 × planform area (per panel; doubled for
    /// a mirrored pair). Lifting surfaces are lighter per area than
    /// pressure tanks — spar + skin, not a sealed vessel.
    pub mass_per_m2: f32,
    /// Default geometry preset this catalog entry arms with. Pre-stabilizer
    /// saves omit it and load as `Lift`, so existing wing entries are
    /// unchanged.
    #[serde(default)]
    pub role: WingRole,
    /// Whitelisted resource storage this part may carry. Dry wings leave it
    /// empty; wet wings (`wing_wet`) whitelist kerosene, whose capacity scales
    /// with the panel's internal volume (see [`wing_volume`]).
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GearSpec {
    pub display_name: String,
    /// dry_mass contribution per leg = `strut_mass_per_m × strut_length`.
    pub strut_mass_per_m: f32,
    /// Fixed mass per wheel, kg (added once per leg).
    pub wheel_mass: f32,
    /// Lateral spacing of the main legs as a fraction of the host radius
    /// (`±track_fraction × host_radius`). `0.0` means a single centred leg
    /// (nose gear); `> 0.0` a left/right main pair. This is what distinguishes
    /// `gear_main` from `gear_nose` — the leg count derives from it
    /// ([`crate::Gear::legs`]).
    pub track_fraction: f32,
    /// Strut length (host skin → wheel hub) a freshly-placed gear starts at,
    /// metres. Per-spec rather than a kind-wide constant so each **size
    /// class** authors its own ride height, and a nose/main pair ships as a
    /// **matched set**: equal `default_strut_length + default_wheel_radius`
    /// on a shared fuselage makes the aircraft sit level out of the box,
    /// instead of the user hand-tuning two struts to agree.
    pub default_strut_length: f32,
    /// Wheel radius a freshly-placed gear starts at, metres. See
    /// [`GearSpec::default_strut_length`] for the matched-set contract.
    pub default_wheel_radius: f32,
    /// Wheels per leg, fore/aft along the strut end. `1` is a single wheel;
    /// `2`+ is a tandem **bogie** (heavier main gear). Like `track_fraction`,
    /// this defines the gear *kind*, so it lives on the catalog and is copied
    /// to [`crate::Gear`] at spawn. Defaults to `1` so existing gear loads
    /// unchanged.
    #[serde(default = "default_wheels_per_leg")]
    pub wheels_per_leg: u8,
}

fn default_wheels_per_leg() -> u8 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStorageSpec {
    pub resource: Resource,
    /// Fixed capacity in the resource's native unit.
    #[serde(default)]
    pub units: f32,
    /// Capacity density in native units per cubic metre of this part's
    /// geometric storage volume.
    #[serde(default)]
    pub units_per_m3: f32,
    /// Whether a new/auto-loaded part starts with this resource pool active.
    /// Inactive whitelisted resources can be added later by the editor.
    #[serde(default = "default_storage_enabled")]
    pub default_enabled: bool,
    /// Initial fill ratio for auto/default pools.
    #[serde(default = "default_fill_fraction")]
    pub default_fill_fraction: f32,
}

fn default_storage_enabled() -> bool {
    true
}

fn default_fill_fraction() -> f32 {
    1.0
}

impl CatalogEntry {
    pub fn display_name(&self) -> &str {
        match self {
            CatalogEntry::Pod(p) => &p.display_name,
            CatalogEntry::Engine(e) => &e.display_name,
            CatalogEntry::Intake(i) => &i.display_name,
            CatalogEntry::Decoupler(d) => &d.display_name,
            CatalogEntry::Adapter(a) => &a.display_name,
            CatalogEntry::Tank(t) => &t.display_name,
            CatalogEntry::Fuselage(f) => &f.display_name,
            CatalogEntry::Wing(w) => &w.display_name,
            CatalogEntry::Gear(g) => &g.display_name,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            CatalogEntry::Pod(_) => "Pod",
            CatalogEntry::Engine(_) => "Engine",
            CatalogEntry::Intake(_) => "Intake",
            CatalogEntry::Decoupler(_) => "Decoupler",
            CatalogEntry::Adapter(_) => "Adapter",
            CatalogEntry::Tank(_) => "Tank",
            CatalogEntry::Fuselage(_) => "Fuselage",
            CatalogEntry::Wing(_) => "Wing",
            CatalogEntry::Gear(_) => "Gear",
        }
    }

    pub fn storage_options(&self) -> &[ResourceStorageSpec] {
        match self {
            CatalogEntry::Pod(p) => &p.storage,
            CatalogEntry::Engine(e) => &e.storage,
            CatalogEntry::Intake(i) => &i.storage,
            CatalogEntry::Decoupler(d) => &d.storage,
            CatalogEntry::Adapter(a) => &a.storage,
            CatalogEntry::Tank(t) => &t.storage,
            CatalogEntry::Fuselage(f) => &f.storage,
            CatalogEntry::Wing(w) => &w.storage,
            // Landing gear stores nothing.
            CatalogEntry::Gear(_) => &[],
        }
    }
}

impl PartCatalog {
    /// Look up a catalog entry by ID. Returns [`CatalogError::UnknownId`]
    /// when the ID isn't present — load-time blueprint resolution should
    /// fail fast on this.
    pub fn resolve(&self, id: &str) -> Result<&CatalogEntry, CatalogError> {
        self.parts
            .get(id)
            .ok_or_else(|| CatalogError::UnknownId(id.to_string()))
    }

    /// Parse a catalog from a RON string.
    pub fn load_from_str(s: &str) -> Result<Self, CatalogError> {
        ron::from_str(s).map_err(|e| CatalogError::Parse(e.to_string()))
    }

    /// Read and parse a catalog file.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, CatalogError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| CatalogError::Io {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
        Self::load_from_str(&text)
    }
}

#[derive(Debug, Clone)]
pub enum CatalogError {
    UnknownId(String),
    KindMismatch {
        id: String,
        expected: &'static str,
        got: &'static str,
    },
    /// Parametric catalog kind (Tank/Adapter/Decoupler) referenced with
    /// `PartParams::None`, or pure kind (Pod/Engine) referenced with
    /// non-None params.
    ParamMismatch {
        id: String,
        kind: &'static str,
    },
    ResourceNotAllowed {
        id: String,
        resource: Resource,
    },
    Io {
        path: PathBuf,
        source: String,
    },
    Parse(String),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::UnknownId(id) => write!(f, "unknown catalog id: {id}"),
            CatalogError::KindMismatch { id, expected, got } => {
                write!(
                    f,
                    "catalog kind mismatch for {id}: expected {expected}, got {got}"
                )
            }
            CatalogError::ParamMismatch { id, kind } => {
                write!(
                    f,
                    "blueprint params do not match catalog kind {kind} for id {id}"
                )
            }
            CatalogError::ResourceNotAllowed { id, resource } => {
                write!(f, "part {id} cannot store {}", resource.display_name())
            }
            CatalogError::Io { path, source } => {
                write!(f, "failed to read catalog at {}: {source}", path.display())
            }
            CatalogError::Parse(msg) => write!(f, "failed to parse catalog: {msg}"),
        }
    }
}

impl std::error::Error for CatalogError {}

// ---- geometry helpers used by catalog→part composition -------------------

/// Cone silhouette for a command pod: `(radius_top, radius_bottom, height)`
/// in metres for a given diameter and geometry. Height is
/// `diameter × geometry.length_factor()`, matching the node offset / MOI /
/// collider so physics and rendering agree on the pod's extent.
///
/// The **capsule** renders directly from this truncated cone. The **aircraft
/// cockpit** instead renders a rounded ogive ([`crate::build_cockpit_mesh`]);
/// its radii here are only a coarse cone approximation used for the terrain
/// shadow silhouette and camera-framing extent, not the rendered nose.
pub fn pod_visual_profile(diameter: f32, geometry: PodGeometry) -> (f32, f32, f32) {
    let (top_frac, bottom_frac) = match geometry {
        PodGeometry::Capsule => (0.3, 0.5),
        // Coarse cone proxy for the ogive (shadow / extent only).
        PodGeometry::AircraftCockpit => (0.15, 0.5),
        // No body — zero extent (length_factor is also 0).
        PodGeometry::Inline => (0.0, 0.0),
    };
    (
        diameter * top_frac,
        diameter * bottom_frac,
        diameter * geometry.length_factor(),
    )
}

/// Cylindrical tank surface area: lateral wall (2πrL) plus two flat caps
/// (2 × πr²). Hemispherical caps would be a refinement; flat is a fine
/// proxy for mass scaling.
pub fn tank_surface_area(diameter: f32, length: f32) -> f32 {
    let r = diameter * 0.5;
    let lateral = std::f32::consts::PI * diameter * length;
    let caps = 2.0 * std::f32::consts::PI * r * r;
    lateral + caps
}

/// Tank cylindrical volume in m³.
pub fn tank_volume(diameter: f32, length: f32) -> f32 {
    let r = diameter * 0.5;
    std::f32::consts::PI * r * r * length
}

/// Approximate lofted skin surface area of a fuselage, m², for mass scaling.
/// Treats the body as a cylinder of the mean barrel diameter over its length
/// (the nose/tail tapers roughly cancel a true integral at this fidelity),
/// plus two end caps. The barrel diameter is the effective (possibly
/// inherited) one, with height folded in via the mean of width and height.
pub fn fuselage_surface_area(length: f32, width: f32, height: f32) -> f32 {
    let mean_d = (width + height) * 0.5;
    let r = mean_d * 0.5;
    let lateral = std::f32::consts::PI * mean_d * length;
    let caps = 2.0 * std::f32::consts::PI * r * r;
    lateral + caps
}

/// Approximate enclosed volume of a fuselage loft, m³ — the capacity basis
/// for a future wet/role-filled fuselage's `units_per_m3` storage. A tapered
/// body encloses less than its barrel cylinder; `FUSELAGE_FILL` captures the
/// nose/tail taper plus unusable structure.
pub fn fuselage_volume(length: f32, width: f32, height: f32) -> f32 {
    /// Usable share of the barrel cylinder after nose/tail taper + structure.
    const FUSELAGE_FILL: f32 = 0.75;
    let a = width * 0.5;
    let b = height * 0.5;
    std::f32::consts::PI * a * b * length * FUSELAGE_FILL
}

/// Planform (top-down projected) area of one trapezoidal wing panel, m².
/// A tapered panel is a trapezoid of height `span` and parallel sides
/// `root_chord`, `tip_chord`. Sweep shears the trapezoid, which leaves the
/// area unchanged, so it does not enter here.
pub fn wing_panel_area(span: f32, root_chord: f32, tip_chord: f32) -> f32 {
    span * (root_chord + tip_chord) * 0.5
}

/// Internal wet-tank volume of one wing panel, m³ — the capacity basis for
/// a wet wing's `units_per_m3` storage. The airfoil's max-thickness envelope
/// is `planform_area × (t/c · MAC)`; the integral wing box that actually
/// holds fuel is a fraction of that (spars, ribs, and the leading/trailing
/// edges aren't wet), captured by `WING_BOX_FILL`. Mirrors how
/// [`tank_volume`] is the capacity basis for a cylindrical tank.
pub fn wing_volume(span: f32, root_chord: f32, tip_chord: f32, thickness: f32) -> f32 {
    /// Usable share of the airfoil envelope that forms the sealed wing box.
    const WING_BOX_FILL: f32 = 0.5;
    let area = wing_panel_area(span, root_chord, tip_chord);
    let mean_thickness = thickness * wing_mean_aerodynamic_chord(root_chord, tip_chord);
    area * mean_thickness * WING_BOX_FILL
}

/// Mean aerodynamic chord of one trapezoidal panel, m. For a linear taper
/// with taper ratio λ = tip/root: MAC = (2/3)·root·(1 + λ + λ²)/(1 + λ).
/// Degenerates to the chord for an untapered panel.
pub fn wing_mean_aerodynamic_chord(root_chord: f32, tip_chord: f32) -> f32 {
    if root_chord <= 0.0 {
        return 0.0;
    }
    let lambda = (tip_chord / root_chord).clamp(0.0, 1.0);
    (2.0 / 3.0) * root_chord * (1.0 + lambda + lambda * lambda) / (1.0 + lambda)
}

/// Structural mass of a landing-gear assembly, kg: every leg contributes a
/// strut (`strut_mass_per_m × strut_length`) plus a wheel (`wheel_mass`). The
/// leg count derives from the spec's `track_fraction` (a track ⇒ a main pair),
/// so `gear_main` counts both legs and `gear_nose` counts one. Single home for
/// the formula shared by blueprint composition, the recompute system, and the
/// stats aggregator.
pub fn gear_dry_mass(spec: &GearSpec, strut_length: f32) -> f32 {
    let legs = if spec.track_fraction > 0.0 { 2.0 } else { 1.0 };
    let wheels = spec.wheels_per_leg.max(1) as f32;
    legs * (spec.strut_mass_per_m * strut_length + wheels * spec.wheel_mass)
}

/// Lateral surface area of a frustum with the two given diameters and a
/// height inferred the same way the editor's adapter mesh does it
/// (`((d_top + d_bot) / 2).max(0.4)`). Keeps mass and visual geometry
/// consistent with each other.
pub fn adapter_surface_area(top_diameter: f32, bottom_diameter: f32) -> f32 {
    let h = ((top_diameter + bottom_diameter) * 0.5).max(0.4);
    let r1 = top_diameter * 0.5;
    let r2 = bottom_diameter * 0.5;
    let dr = r1 - r2;
    let slant = (h * h + dr * dr).sqrt();
    std::f32::consts::PI * (r1 + r2) * slant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_catalog() {
        let text = include_str!("../../../assets/parts.ron");
        let cat = PartCatalog::load_from_str(text).expect("parse parts.ron");
        assert!(cat.resolve("argos").is_ok());
        assert!(cat.resolve("hyperion").is_ok());
        assert!(cat.resolve("zephyr").is_ok());
        assert!(cat.resolve("boreas").is_ok());
        assert!(cat.resolve("tank_methalox").is_ok());
        assert!(cat.resolve("adapter_std").is_ok());
        assert!(cat.resolve("decoupler_std").is_ok());
        assert!(cat.resolve("nope").is_err());

        let CatalogEntry::Engine(zephyr) = cat.resolve("zephyr").unwrap() else {
            panic!("zephyr should be an engine");
        };
        assert_eq!(zephyr.optimized_for, EngineOptimization::Atmosphere);
        assert_eq!(zephyr.geometry, EngineGeometry::RocketBell);

        let CatalogEntry::Engine(mistral) = cat.resolve("mistral_jet").unwrap() else {
            panic!("mistral_jet should be an engine");
        };
        assert_eq!(mistral.geometry, EngineGeometry::JetNacelle);
        assert!(mistral.requires_atmosphere);
        assert!(mistral.intake_requirement.is_some());
        assert!(mistral.builtin_intake.is_some());
        assert_eq!(mistral.reactants[0].resource, Resource::Kerosene);

        let CatalogEntry::Intake(intake) = cat.resolve("intake_cone").unwrap() else {
            panic!("intake_cone should be an intake");
        };
        assert_eq!(intake.capture.kind, AmbientIntakeKind::OxygenatedAir);

        // The cockpit is a command pod with the aircraft-cockpit silhouette;
        // existing pods omit `geometry` and default to a capsule.
        let CatalogEntry::Pod(cockpit) = cat.resolve("cockpit").unwrap() else {
            panic!("cockpit should be a pod");
        };
        assert_eq!(cockpit.geometry, PodGeometry::AircraftCockpit);
        let CatalogEntry::Pod(argos) = cat.resolve("argos").unwrap() else {
            panic!("argos should be a pod");
        };
        assert_eq!(argos.geometry, PodGeometry::Capsule);

        let CatalogEntry::Tank(jet_tank) = cat.resolve("tank_kerosene").unwrap() else {
            panic!("tank_kerosene should be a tank");
        };
        assert!(
            jet_tank
                .storage
                .iter()
                .any(|s| s.resource == Resource::Kerosene)
        );
    }

    #[test]
    fn frustum_surface_area_is_finite_and_positive() {
        assert!(adapter_surface_area(2.5, 4.0) > 0.0);
        assert!(adapter_surface_area(2.5, 2.5) > 0.0); // degenerate cylinder still > 0
    }

    #[test]
    fn tank_volume_scales_with_length() {
        let small = tank_volume(2.5, 1.0);
        let large = tank_volume(2.5, 4.0);
        assert!((large - 4.0 * small).abs() < 1e-3);
    }
}
