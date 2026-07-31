//! Rocket-engine exhaust plumes — a data-driven, pressure-responsive renderer
//! (design: `docs/rendering/plume.md`), driven from a typed [`PlumeSignals`] boundary.
//!
//! The plume is modelled as an axisymmetric emitting gas column whose *shape* is
//! set by the nozzle/ambient pressure ratio and whose *brightness follows from
//! that shape* rather than from an authored fade curve:
//!
//! ```text
//!   R(s) = R0·lip + tan(theta)·s      free expansion off the nozzle lip
//!   rho  ∝ (R0/R)²                    mass conservation along the column
//!   T    ∝ T_exit · (R0/R)^(2(g-1))   adiabatic (expansion) cooling
//!   T   ×= exp(-e·s/R0)               entrainment cooling (atmosphere only)
//!   S    = exp(-W·(1/T − 1))          visible-band emission, Wien side
//!   L    = S · (1 − exp(−rho·chord))  emission through an absorbing column
//! ```
//!
//! That one law spans the whole flight envelope with no regime switch. In vacuum
//! the column cools by expanding and the exponential emission term collapses, so
//! the cone dissolves on its own; at sea level it barely expands, so entrainment
//! of ambient air is what cools it, and it reads as a dense, shock-celled,
//! afterburning column. This module resolves the parameters; `plume.wgsl`
//! renders the same model. See `docs/rendering/plume.md`, and INC-0020 before repurposing
//! any packed lane of [`PlumeParams`].
//!
//! **One length authority.** [`visible_length_m`] solves that same chain for the
//! station where the rendered radiance vanishes, and the billboard is cut
//! exactly there. Nothing else may shorten it — a cap the fragment stage cannot
//! see ends the geometry mid-column and renders as a lit rim hanging in mid-air
//! (INC-20260724T235437Z-plume-ended-on-a-lit-rim). Every input that should shorten a plume (throttle,
//! back-pressure, propellant opacity, the ignition transient) does so by feeding
//! the chain, never by trimming its result.
//!
//! Pipeline:
//!
//! 1. [`update_plume_signals`] reads each firing [`Engine`]'s runtime state
//!    ([`EngineThrust`] + [`ThrottleState`]) plus the craft's local ambient
//!    pressure and publishes a compact, render-facing [`PlumeSignals`] on the
//!    engine entity (the single typed boundary — visual code never reaches back
//!    into gameplay components). A [`PlumeDebugOverride`] resource can drive the
//!    signals directly for authoring / headless capture.
//! 2. [`update_plume_visuals`] resolves those signals through the propellant
//!    preset + the pressure response into [`PlumeParams`] (the flat uniform the
//!    shader renders) and toggles plume visibility.

use std::f32::consts::FRAC_PI_2;

use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};

use thalos_shipyard::{
    Engine, EngineActivation, EngineGeometry, EngineOptimization, EngineThrust, ReactantRatio,
    Resource,
};

use crate::SimStage;
use crate::rendering::flow::{
    FlowProxyMesh, FlowSignals, axial_proxy_prism_mesh, update_flow_signals,
};
use crate::shipyard_editor::core::EditorPart;

/// Bounds on `p_exit / p_ambient`. The low end is a deeply overexpanded nozzle in
/// dense air; the high end stands in for a true vacuum, where the ratio is
/// unbounded but the *look* has long since saturated.
const PRESSURE_RATIO_MIN: f32 = 0.015;
const PRESSURE_RATIO_MAX: f32 = 64.0;

/// Proxy-hull resolution. Only has to bound a smooth, monotonically widening
/// envelope, so it is deliberately coarse — the silhouette comes from the
/// density integral, not from these rings.
const PROXY_SIDES: usize = 12;
const PROXY_RINGS: usize = 16;

/// Shared plume proxy-hull handle — the flow-effect prism from
/// [`axial_proxy_prism_mesh`], which the vertex shader scales per ring by the
/// envelope bound. Rings only have to track a smooth monotonic bound, so this is
/// far coarser than the strip it replaced.
#[derive(Resource)]
struct PlumeMesh(Handle<Mesh>);

pub struct PlumePlugin;

impl Plugin for PlumePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<PlumeMaterial>::default())
            .init_resource::<PlumeDebugOverride>()
            .add_systems(Startup, insert_plume_mesh)
            .add_systems(
                Update,
                (
                    spawn_engine_plumes,
                    update_plume_signals,
                    update_plume_visuals,
                )
                    .chain()
                    .in_set(SimStage::Sync)
                    // After the engine-thrust plumbing writes `EngineThrust`,
                    // and after the shared flow boundary publishes the air.
                    .after(crate::engine::update_engine_thrust)
                    .after(update_flow_signals),
            );
    }
}

fn insert_plume_mesh(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let handle = meshes.add(axial_proxy_prism_mesh(PROXY_SIDES, PROXY_RINGS));
    commands.insert_resource(PlumeMesh(handle));
}

// ---------------------------------------------------------------------------
// Typed signal boundary
// ---------------------------------------------------------------------------

/// Compact, render-facing engine state the plume (and future particle/light
/// layers) consume. Published on the engine entity by [`update_plume_signals`];
/// visual code reads this, never the gameplay components directly.
#[derive(Component, Debug, Clone, Copy)]
pub struct PlumeSignals {
    /// Commanded throttle 0..1 (what the plume shape/brightness tracks).
    pub throttle: f32,
    /// Smoothed ignition state 0..1 — retains spool-up / shutdown transients
    /// even when the commanded throttle steps instantly.
    pub ignition: f32,
    /// Local ambient pressure at the craft, Pa.
    pub ambient_pressure_pa: f32,
    /// Nozzle-exit / ambient pressure ratio `r = p_exit / p_ambient`. `>1`
    /// underexpanded (toward vacuum), `<1` overexpanded (dense atmosphere),
    /// `≈1` perfectly expanded. Saturates at [`PRESSURE_RATIO_MAX`] in vacuum.
    pub pressure_ratio: f32,
}

impl Default for PlumeSignals {
    fn default() -> Self {
        Self {
            throttle: 0.0,
            ignition: 0.0,
            ambient_pressure_pa: 0.0,
            pressure_ratio: 1.0,
        }
    }
}

/// Authoring / capture override. When any field is `Some`, [`update_plume_signals`]
/// uses it instead of live engine state — the "scrub with frozen controller
/// values" workflow, and how the headless `plume` screenshot preset lights an
/// engine without fighting the fuel/warp gating.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct PlumeDebugOverride {
    pub throttle: Option<f32>,
    pub ambient_pressure_pa: Option<f32>,
    pub ignition: Option<f32>,
}

// ---------------------------------------------------------------------------
// Propellant families
// ---------------------------------------------------------------------------

/// Visual propellant family — the starting palette an engine profile then tunes.
/// Derived from the engine's reactants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropellantFamily {
    Methalox,
    Kerolox,
    Hydrolox,
    /// Fallback for anything else (hypergolic / mono / unknown) — a restrained
    /// translucent jet until it gets its own preset.
    Generic,
}

impl PropellantFamily {
    fn from_reactants(reactants: &[ReactantRatio]) -> Self {
        let has = |res: Resource| reactants.iter().any(|r| r.resource == res);
        if has(Resource::Kerosene) {
            Self::Kerolox
        } else if has(Resource::Hydrogen) {
            Self::Hydrolox
        } else if has(Resource::Methane) && has(Resource::Lox) {
            Self::Methalox
        } else {
            Self::Generic
        }
    }

    /// Linear-RGB palette + radiance/opacity scales. The expanded-plume and
    /// sheath colours come in a *pair*: what the exhaust looks like on its own
    /// (vacuum), and what it looks like once entrained air afterburns the
    /// fuel-rich products (atmosphere). [`resolve_params`] blends between them on
    /// ambient pressure — which is why one methalox preset reads blue-violet in
    /// orbit and orange-white on the pad.
    fn visuals(self) -> Visuals {
        match self {
            // White-hot at the throat, pale blue-violet expanding in vacuum,
            // bright orange once it afterburns at sea level.
            Self::Methalox => Visuals {
                hot: Vec3::new(1.00, 0.94, 0.86),
                mid_vacuum: Vec3::new(0.44, 0.60, 1.05),
                mid_air: Vec3::new(1.00, 0.62, 0.25),
                sheath_vacuum: Vec3::new(0.34, 0.30, 0.95),
                sheath_air: Vec3::new(1.00, 0.38, 0.10),
                radiance: 5.5,
                opacity: 1.0,
            },
            // Sooty: already orange in vacuum, and the sheath goes dark and
            // smoky rather than bright.
            Self::Kerolox => Visuals {
                hot: Vec3::new(1.00, 0.90, 0.72),
                mid_vacuum: Vec3::new(1.00, 0.72, 0.42),
                mid_air: Vec3::new(1.00, 0.55, 0.18),
                sheath_vacuum: Vec3::new(0.70, 0.38, 0.16),
                sheath_air: Vec3::new(0.85, 0.28, 0.06),
                radiance: 5.0,
                opacity: 1.25,
            },
            // Nearly transparent — legibility comes from the core glow and the
            // shock nodes, not a saturated flame.
            Self::Hydrolox => Visuals {
                hot: Vec3::new(0.92, 0.95, 1.00),
                mid_vacuum: Vec3::new(0.55, 0.68, 1.00),
                mid_air: Vec3::new(0.78, 0.82, 1.00),
                sheath_vacuum: Vec3::new(0.40, 0.46, 0.85),
                sheath_air: Vec3::new(0.62, 0.60, 0.90),
                radiance: 3.3,
                opacity: 0.42,
            },
            Self::Generic => Visuals {
                hot: Vec3::new(0.98, 0.92, 0.84),
                mid_vacuum: Vec3::new(0.85, 0.66, 0.55),
                mid_air: Vec3::new(0.95, 0.60, 0.32),
                sheath_vacuum: Vec3::new(0.48, 0.36, 0.38),
                sheath_air: Vec3::new(0.72, 0.36, 0.20),
                radiance: 4.3,
                opacity: 0.80,
            },
        }
    }
}

struct Visuals {
    hot: Vec3,
    mid_vacuum: Vec3,
    mid_air: Vec3,
    sheath_vacuum: Vec3,
    sheath_air: Vec3,
    radiance: f32,
    opacity: f32,
}

// ---------------------------------------------------------------------------
// Material + resolved uniform
// ---------------------------------------------------------------------------

/// Resolved per-engine plume profile — the flat uniform the shader renders.
/// Packed as vec4s so the std140 layout matches `plume.wgsl` unambiguously.
#[derive(Clone, Copy, ShaderType, Debug)]
pub struct PlumeParams {
    /// rgb = hot core colour, a = HDR radiance scale.
    pub core_color: Vec4,
    /// rgb = expanded-plume colour, a = nozzle exit radius R0 (m).
    pub mid_color: Vec4,
    /// rgb = shear-layer colour, a = visible axial length L (m).
    pub edge_color: Vec4,
    /// x = lip radius scale, y = tan(core half-angle),
    /// z = shear-layer spread rate, w = adiabatic exponent `2(γ−1)`.
    pub shape: Vec4,
    /// x = core opacity κ, y = sheath opacity κ,
    /// z = shock-cell wavenumber (rad/m), w = shock strength 0..1.
    pub shock: Vec4,
    /// x = shock decay length (m), y = afterburn 0..1,
    /// z = turbulence amplitude, w = **reserved** (throttle; the shader no
    /// longer reads it — throttle acts through κ, entrainment and flicker).
    pub mixing: Vec4,
    /// x = time (s), y = seed, z = ignition, w = entrainment cooling rate
    /// (per nozzle radius of axial distance; 0 in vacuum).
    pub anim: Vec4,
    /// Turbulent motion. x = eddy growth per axial metre,
    /// y = convection rate (eddies/s), z = azimuthal swirl (rad/s),
    /// w = radial wobble amplitude.
    pub flow: Vec4,
    /// x = tail dispersal growth, y = potential-core length (m),
    /// z = flicker amplitude, w = flicker rate (Hz).
    pub tail: Vec4,
    /// x = exit temperature (normalized; the ignition transient),
    /// y = core turbulence weight, z = tail turbulence boost,
    /// w = shock-cell lengthening per axial metre.
    pub therm: Vec4,
}

impl Default for PlumeParams {
    fn default() -> Self {
        Self {
            core_color: Vec4::new(1.0, 0.94, 0.86, 30.0),
            mid_color: Vec4::new(0.44, 0.60, 1.05, 1.0),
            edge_color: Vec4::new(0.34, 0.30, 0.95, 30.0),
            shape: Vec4::new(1.0, 0.13, 0.04, 0.4),
            shock: Vec4::new(2.4, 0.5, 0.0, 0.0),
            mixing: Vec4::new(1.0, 0.0, 0.25, 0.0),
            anim: Vec4::new(0.0, 0.0, 0.0, 0.0),
            flow: Vec4::new(0.10, 6.0, 0.1, 0.06),
            tail: Vec4::new(1.0, 30.0, 0.05, 8.0),
            therm: Vec4::new(1.0, 0.2, 1.0, 0.0),
        }
    }
}

/// Additive, unlit emissive plume material. One instance per engine so each
/// carries its own resolved [`PlumeParams`].
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct PlumeMaterial {
    #[uniform(0)]
    pub params: PlumeParams,
}

impl Material for PlumeMaterial {
    fn vertex_shader() -> bevy::shader::ShaderRef {
        "shaders/plume.wgsl".into()
    }

    fn fragment_shader() -> bevy::shader::ShaderRef {
        "shaders/plume.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        // src.rgb * src.a + dst — the classic additive HDR plume.
        AlphaMode::Add
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The proxy prism is closed, so a ray crosses it twice. Culling one side
        // is what keeps the raymarch from running — and additively blending —
        // twice per pixel.
        //
        // Cull the *back* faces, keeping the near surface: its depth is where
        // the column starts, so the plume keeps depth-testing against the hull
        // and the terrain roughly where it used to. Drawing the far side instead
        // would be more robust to the camera entering the exhaust, but would
        // depth-test at a point behind the craft and let the hull erase the
        // whole plume. The cost of this choice is that a camera *inside* the
        // bounding prism loses the part of it that falls behind the near plane;
        // fixing that properly means clamping the march against scene depth.
        descriptor.primitive.cull_mode = Some(bevy::render::render_resource::Face::Back);
        Ok(())
    }
}

/// Marker + wiring on the plume child entity: which engine it belongs to, its
/// material, nozzle geometry, design point, and propellant family.
#[derive(Component)]
pub struct PlumeVisual {
    engine: Entity,
    material: Handle<PlumeMaterial>,
    nozzle_radius_m: f32,
    family: PropellantFamily,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Spawn one plume child under each newly-added rocket-bell engine (flight
/// craft only — editor parts opt out). Jet nacelles get their own afterburner
/// treatment later; for now only bell engines plume.
fn spawn_engine_plumes(
    mut commands: Commands,
    mesh: Option<Res<PlumeMesh>>,
    mut materials: ResMut<Assets<PlumeMaterial>>,
    engines: Query<(Entity, &Engine), (Added<Engine>, Without<EditorPart>)>,
) {
    let Some(mesh) = mesh else {
        return;
    };
    for (entity, engine) in engines.iter() {
        if engine.geometry != EngineGeometry::RocketBell {
            continue;
        }
        // Nozzle exit geometry mirrors `ship_view::visual_spec`'s rocket bell:
        // height h = 0.9·d, exit radius = 0.5·d, exit at part-local y = -h.
        let exit_offset = -0.9 * engine.diameter;
        let nozzle_radius = 0.5 * engine.diameter;
        let family = PropellantFamily::from_reactants(&engine.reactants);

        let material = materials.add(PlumeMaterial {
            params: PlumeParams::default(),
        });
        let plume = commands
            .spawn((
                Mesh3d(mesh.0.clone()),
                MeshMaterial3d(material.clone()),
                // Plume-local -Y is the exhaust axis (= the engine's -Y), so an
                // identity-rotation child inherits the thrust axis directly.
                Transform::from_xyz(0.0, exit_offset, 0.0),
                Visibility::Hidden,
                NoFrustumCulling,
                NotShadowCaster,
                // Not vehicle structure: keep it out of the craft-bounds sweep.
                FlowProxyMesh,
                PlumeVisual {
                    engine: entity,
                    material,
                    nozzle_radius_m: nozzle_radius,
                    family,
                },
            ))
            .id();
        commands
            .entity(entity)
            .insert(PlumeSignals::default())
            .add_child(plume);
    }
}

/// Publish [`PlumeSignals`] on each engine from its live runtime state (or the
/// [`PlumeDebugOverride`]). Ambient pressure is read from
/// [`FlowSignals`] — the shared per-vehicle aerothermal boundary — so the plume
/// and the reentry shock layer can never disagree about the air they are in.
fn update_plume_signals(
    time: Res<Time>,
    flow: Res<FlowSignals>,
    over: Res<PlumeDebugOverride>,
    mut engines: Query<
        (
            &Engine,
            Option<&EngineActivation>,
            &EngineThrust,
            &mut PlumeSignals,
        ),
        Without<EditorPart>,
    >,
) {
    let dt = time.delta_secs();
    // Ambient pressure comes from the one aerothermal boundary
    // (`rendering::flow`), not from a private atmosphere lookup. The per-engine
    // authoring override still wins, because `THALOS_PLUME_PRESSURE` scrubs a
    // single engine's back-pressure without pretending the vehicle moved.
    let ambient = over.ambient_pressure_pa.unwrap_or(flow.ambient_pressure_pa);

    for (engine, activation, thrust, mut sig) in engines.iter_mut() {
        let enabled = activation.map(|a| a.enabled).unwrap_or(true);
        // Commanded throttle for this engine: the fraction of its rated thrust
        // currently produced, so fuel-out / gating fold in for free.
        let live_throttle = if enabled && engine.thrust > 0.0 {
            (thrust.current_n / engine.thrust).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let throttle = over.throttle.unwrap_or(live_throttle);

        // Ignition transient: spool toward the commanded state (fast up, slower
        // down) so start/stop reads as a flare rather than a hard cut.
        let target = if throttle > 0.02 { 1.0 } else { 0.0 };
        let tau = if target > sig.ignition { 0.10 } else { 0.22 };
        let ignition = match over.ignition {
            Some(v) => v,
            None => {
                let alpha = if tau > 0.0 {
                    1.0 - (-dt / tau).exp()
                } else {
                    1.0
                };
                sig.ignition + (target - sig.ignition) * alpha.clamp(0.0, 1.0)
            }
        };

        sig.throttle = throttle;
        sig.ignition = ignition;
        sig.ambient_pressure_pa = ambient;
        sig.pressure_ratio = pressure_ratio(engine, throttle, ambient);
    }
}

/// Resolve each plume's signals through its propellant preset + the pressure
/// response into the shader uniform, and toggle visibility.
fn update_plume_visuals(
    time: Res<Time>,
    mut materials: ResMut<Assets<PlumeMaterial>>,
    signals: Query<&PlumeSignals>,
    mut plumes: Query<(&PlumeVisual, &mut Visibility)>,
) {
    let elapsed = time.elapsed_secs();
    for (visual, mut vis) in plumes.iter_mut() {
        let Ok(sig) = signals.get(visual.engine) else {
            continue;
        };
        let firing = sig.ignition > 0.01 && sig.throttle > 0.0;
        let target_vis = if firing {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != target_vis {
            *vis = target_vis;
        }
        if !firing {
            continue;
        }
        let Some(mut mat) = materials.get_mut(&visual.material) else {
            continue;
        };
        let params = resolve_params(
            sig,
            visual.nozzle_radius_m,
            visual.family,
            elapsed,
            seed_of(visual.engine),
        );
        mat.params = params;
    }
}

// ---------------------------------------------------------------------------
// Curve evaluation (signals + preset -> uniform)
// ---------------------------------------------------------------------------

/// HDR radiance below which the plume's *additive* contribution no longer
/// survives exposure and tonemapping — i.e. where the column is genuinely gone
/// from the image, not merely faint relative to its own peak.
///
/// It has to be absolute. A fraction-of-peak floor looks safe and is not: the
/// core saturates at a radiance of order 10, so "0.3 % of peak" is still ~0.04
/// linear, which the tonemapper lifts back to a clearly visible brown. That is
/// what left a lit rim across the end of the tail after the first attempt at
/// this fix.
const VISIBLE_RADIANCE: f32 = 0.0025;

/// Dimensionless Wien parameter for the visible band — **mirrors `WIEN` in
/// `plume.wgsl`**. The CPU solves the shader's own emission curve to place the
/// end of the mesh, so the two constants must stay equal.
const WIEN: f32 = 3.0;

/// Visible-band emission from a cooling gas, normalised to 1 at chamber
/// temperature. Mirrors `band_emission` in `plume.wgsl`.
fn band_emission(temp_norm: f32) -> f32 {
    (-WIEN * (1.0 / temp_norm.max(1e-3) - 1.0)).exp()
}

/// The resolved column, evaluated on its own axis — the CPU-side twin of what
/// `plume.wgsl` renders, minus the shock ripple (a ±10 % modulation, not a
/// length authority).
struct Column {
    r0: f32,
    lip: f32,
    tan_theta: f32,
    spread: f32,
    gamma_exp: f32,
    entrainment: f32,
    temp_exit: f32,
    kappa_core: f32,
    kappa_sheath: f32,
    afterburn: f32,
    gain: f32,
}

impl Column {
    /// On-axis rendered radiance at axial station `s` (metres from the exit
    /// plane), in the shader's own HDR units: `gain · Σ emission · (1 − e^−τ)`.
    ///
    /// **Both layers are here on purpose.** The sheath outlives the core — it is
    /// wider, so its optical depth saturates long after the core has thinned out,
    /// and in atmosphere afterburning makes it the brighter of the two by the
    /// tail. A criterion that watched only the core cut the mesh while the shear
    /// layer was still glowing, which is a lit rim by another route.
    ///
    /// The sheath is evaluated *un-dispersed* (linear spread, `breakup` = 0).
    /// The dispersal flare only widens a layer whose `1 − e^−τ` is already
    /// saturated on the axis, so it cannot move the vanishing point — and
    /// leaving it out keeps this free of the circular dependency on the length
    /// it is being used to compute.
    fn radiance(&self, s: f32) -> f32 {
        let r = (self.r0 * self.lip + self.tan_theta * s).max(1e-6);
        let er = self.r0 / r;
        let temp =
            self.temp_exit * er.powf(self.gamma_exp) * (-self.entrainment * s / self.r0).exp();
        let density = er * er;
        // On-axis line integrals of the two radial kernels, matching the
        // fragment stage exactly: (pi/2)·R for the core, (16/15)·R for the
        // shear layer, both normalised by 2·R0 as tau is there.
        let tau_core = self.kappa_core * density * (FRAC_PI_2 * r / (2.0 * self.r0));
        let rs = r + self.spread * s + 0.05 * self.r0;
        let tau_sheath = self.kappa_sheath * density * ((16.0 / 15.0) * rs / (2.0 * self.r0));
        let temp_sheath = temp * lerp(0.62, 0.88, self.afterburn);
        let core = band_emission(temp) * lerp(1.0, 1.8, self.afterburn) * (1.0 - (-tau_core).exp());
        let sheath = band_emission(temp_sheath)
            * lerp(1.0, 2.4, self.afterburn)
            * (1.0 - (-tau_sheath).exp());
        self.gain * (core + sheath)
    }
}

/// Axial distance at which the column has gone dark: where [`Column::radiance`]
/// has fallen below [`VISIBLE_RADIANCE`].
///
/// **This is the only thing that sets plume length.** Both cooling mechanisms,
/// the optical depth, and therefore every input that feeds them (throttle,
/// ambient pressure, propellant opacity, the ignition transient) reach the
/// length through this one function. Do not cap it afterwards: a limit the
/// fragment stage cannot see ends the geometry while the column is still
/// incandescent, which renders as a flat lit rim hanging in mid-air — that is
/// exactly what the old `len_mixing` cap did (INC-20260724T235437Z-plume-ended-on-a-lit-rim).
///
/// The exponential Wien term dominates the far field, so `radiance` is
/// decreasing wherever it matters and a bisection lands on the vanishing point.
fn visible_length_m(col: &Column) -> f32 {
    let target = VISIBLE_RADIANCE;
    // A column that never reaches the floor (a very dim engine at idle) has no
    // visible plume at all.
    if col.radiance(0.0) <= target {
        return 0.0;
    }
    let mut hi = col.r0 * 4.0;
    // Grow the bracket until the column is dark. Bounded: a perfectly collimated
    // jet with no entrainment would otherwise run forever.
    for _ in 0..12 {
        if col.radiance(hi) <= target {
            break;
        }
        hi *= 2.0;
    }
    let mut lo = 0.0_f32;
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if col.radiance(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Map [`PlumeSignals`] + propellant preset into the flat [`PlumeParams`] the
/// shader renders. Pressure response is authored over `log2(pressure_ratio)`
/// since the ratio spans orders of magnitude.
fn resolve_params(
    sig: &PlumeSignals,
    nozzle_radius_m: f32,
    family: PropellantFamily,
    time: f32,
    seed: f32,
) -> PlumeParams {
    let vis = family.visuals();
    let r0 = nozzle_radius_m.max(0.01);
    let pa = sig.ambient_pressure_pa.max(0.0);
    let ratio = sig
        .pressure_ratio
        .clamp(PRESSURE_RATIO_MIN, PRESSURE_RATIO_MAX);

    // Compressed pressure coordinate: 0 at perfect expansion, negative
    // overexpanded (dense air), positive underexpanded (toward vacuum).
    let u = ratio.log2();
    // Master lever: 0 = sea-level dense, 1 = free vacuum expansion.
    let vac = smoothstep(-0.5, 4.5, u);
    // How much ambient gas there is to shock against and to entrain.
    let atmo = smoothstep(600.0, 30_000.0, pa);
    let afterburn = smoothstep(3_000.0, 45_000.0, pa);

    // -- geometry --------------------------------------------------------
    // Overexpanded flow turns *inward* at the lip and forms a compressed waist.
    let overexpanded = smoothstep(0.0, -2.5, u);
    let lip = lerp(1.0, 0.74, overexpanded);
    // Half-angle of the *luminous* core — much narrower than the rarefied outer
    // flow, which is invisible.
    let tan_theta = lerp(0.020, 0.190, vac);
    // Turbulent shear-layer growth *while the potential core survives*: strong
    // entrainment in atmosphere, almost none in vacuum (there is nothing to
    // entrain). Past the core it accelerates — see `dispersal` below.
    let spread = lerp(0.155, 0.040, vac);
    // Adiabatic exponent 2(γ−1) for γ ≈ 1.2 combustion products.
    let gamma_exp = 0.40;

    // -- shock cells -----------------------------------------------------
    // Diamonds need back-pressure and a mismatched nozzle; they wash out once
    // the flow is so underexpanded that the shocks leave the luminous core.
    let mismatch = u.abs();
    let shock_strength =
        atmo * smoothstep(0.12, 0.65, mismatch) * (1.0 - smoothstep(3.5, 7.0, mismatch));
    // First-cell length scales with the square root of the pressure mismatch.
    let q = ratio.max(1.0 / ratio).clamp(1.0, 40.0);
    let cell_len_m = (r0 * (0.6 + 0.35 * q.sqrt())).clamp(1.0 * r0, 3.0 * r0);
    let shock_k = std::f32::consts::TAU / cell_len_m;
    let shock_decay_m = cell_len_m * 3.2;
    // Cells lengthen as the train weakens: roughly a doubling over the decay
    // length, so the diamonds spread out downstream instead of marching at a
    // fixed pitch like a ladder.
    let cell_growth = 1.0 / (3.0 * shock_decay_m);

    // -- colour ----------------------------------------------------------
    let mid = vis.mid_vacuum.lerp(vis.mid_air, afterburn);
    let sheath = vis.sheath_vacuum.lerp(vis.sheath_air, afterburn);

    // -- density, temperature, mixing ------------------------------------
    // Throttle is chamber *pressure*, not chamber temperature: a deep-throttled
    // engine burns the same mixture just as hot, it simply flows less of it. So
    // throttle acts on the optical depth (mass flow) and on how quickly the
    // weaker jet is destroyed by mixing — never as a brightness or length trim,
    // which would be a second authority over both.
    let mass_flow = lerp(0.45, 1.0, sig.throttle);
    let kappa_core = 2.4 * vis.opacity * mass_flow;
    let kappa_sheath = 0.55 * vis.opacity * mass_flow;
    // Chamber temperature *does* ramp during ignition — that transient is what
    // makes a start read as a flare rather than a pop, and it shortens the
    // column through the same law that sizes the mesh.
    let temp_exit = lerp(0.55, 1.0, sig.ignition);

    // Entrainment cooling is expressed through the **mixing length**: the axial
    // distance, in nozzle radii, over which a turbulent jet is torn apart by the
    // ambient air it drags in. Deriving the cooling rate from that — instead of
    // capping the mesh at it — is what removed the hard lit rim: the column now
    // actually goes dark where it used to simply stop (INC-20260724T235437Z-plume-ended-on-a-lit-rim).
    let mix_len_radii = 26.0 * lerp(0.72, 1.0, sig.throttle);
    // Temperature at which emission has fallen to the visibility floor, given
    // that the column starts out saturated at a radiance of order `vis.radiance`.
    let temp_death = 1.0 / (1.0 + (vis.radiance / VISIBLE_RADIANCE).ln() / WIEN);
    let entrainment = atmo * (1.0 / temp_death).ln() / mix_len_radii;

    // -- length ----------------------------------------------------------
    // One authority: solve the rendered radiance for its own vanishing point.
    let length_m = visible_length_m(&Column {
        r0,
        lip,
        tan_theta,
        spread,
        gamma_exp,
        entrainment,
        temp_exit,
        kappa_core,
        kappa_sheath,
        afterburn,
        gain: vis.radiance * sig.ignition.clamp(0.0, 1.0),
    });

    // -- turbulent structure ---------------------------------------------
    let turbulence = lerp(0.22, 0.80, atmo);
    // Eddies grow linearly along the shear layer; the shader samples noise on
    // the resulting eddy coordinate, so structures coarsen as they travel.
    let eddy_growth = lerp(0.09, 0.16, atmo);
    // Convection rate in eddies/second. Real structures convect at a large
    // fraction of the exhaust velocity, which at these eddy sizes is far too
    // fast to read as anything but a blur; this is the legible fraction of it.
    // It is a *rate*, so — unlike advecting in normalized axial coordinates — a
    // longer plume does not slow its own motion down.
    let advect = lerp(4.5, 11.0, sig.throttle) * lerp(1.0, 1.35, atmo);
    let swirl = lerp(0.05, 0.22, atmo);
    let wobble = lerp(0.05, 0.18, atmo);
    // Potential core: the un-mixed cone that survives until the shear layer
    // reaches the axis — a few exit diameters in air, far longer in vacuum where
    // there is nothing to mix with. The column is laminar and coherent inside
    // it and breaks down beyond it.
    let core_len_m = r0 * lerp(10.0, 30.0, vac);
    // Past the core the jet disperses instead of continuing as a cone, so the
    // sheath growth accelerates and the tail opens out. Kept modest: the shear
    // layer's Gaussian profile is what should read as billowing, and a large
    // envelope multiplier just inflates a soft halo into a hard-edged trumpet.
    let dispersal = lerp(0.35, 1.1, atmo);
    // Combustion roughness: low-frequency brightness/length flicker, worse at
    // low throttle where the chamber runs rough, and damped in vacuum.
    let flicker_amp = lerp(0.15, 0.05, sig.throttle) * lerp(0.6, 1.0, atmo);
    let flicker_rate = lerp(5.0, 13.0, sig.throttle);
    let core_turbulence = lerp(0.15, 0.45, atmo);
    let tail_turbulence = lerp(0.6, 1.6, atmo);

    PlumeParams {
        core_color: vis.hot.extend(vis.radiance),
        mid_color: mid.extend(r0),
        edge_color: sheath.extend(length_m),
        shape: Vec4::new(lip, tan_theta, spread, gamma_exp),
        shock: Vec4::new(kappa_core, kappa_sheath, shock_k, shock_strength),
        mixing: Vec4::new(shock_decay_m, afterburn, turbulence, sig.throttle),
        anim: Vec4::new(time, seed, sig.ignition.clamp(0.0, 1.0), entrainment),
        flow: Vec4::new(eddy_growth, advect, swirl, wobble),
        tail: Vec4::new(dispersal, core_len_m, flicker_amp, flicker_rate),
        therm: Vec4::new(temp_exit, core_turbulence, tail_turbulence, cell_growth),
    }
}

/// Design nozzle-exit pressure (Pa) for an engine. Exit pressure is set by the
/// nozzle area ratio, which the catalog expresses as a design point: a sea-level
/// bell is a low-expansion nozzle that stays near ambient on the pad, a vacuum
/// bell expands much further and is badly overexpanded down low.
fn design_exit_pressure_pa(engine: &Engine) -> f32 {
    match engine.optimized_for {
        EngineOptimization::Atmosphere => 55_000.0,
        EngineOptimization::Balanced => 25_000.0,
        EngineOptimization::Vacuum => 7_000.0,
    }
}

/// `r = p_exit / p_ambient`. Chamber pressure tracks throttle and exit pressure
/// with it, so a throttled-down engine near the pad is *more* overexpanded —
/// shorter, with a harder shock train — which is the real behaviour.
fn pressure_ratio(engine: &Engine, throttle: f32, ambient_pressure_pa: f32) -> f32 {
    let p_exit = design_exit_pressure_pa(engine) * lerp(0.35, 1.0, throttle);
    if ambient_pressure_pa <= 1.0 {
        PRESSURE_RATIO_MAX
    } else {
        (p_exit / ambient_pressure_pa).clamp(PRESSURE_RATIO_MIN, PRESSURE_RATIO_MAX)
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Stable per-engine noise seed from its entity id, so turbulence differs
/// between clustered engines but is deterministic across frames.
fn seed_of(entity: Entity) -> f32 {
    (entity.to_bits() % 997) as f32 / 997.0
}
