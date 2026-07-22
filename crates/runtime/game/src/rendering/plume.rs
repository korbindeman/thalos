//! Rocket-engine exhaust plumes — the immediate nozzle plume as a data-driven,
//! pressure-responsive billboard effect (design: the Thalos plume system,
//! `docs/plume.md`). This is Phase 1: one mesh-based emissive layer per liquid
//! engine, driven from a typed [`PlumeSignals`] boundary, with propellant-family
//! presets and a pressure-ratio → shape response. Secondary particles, distortion,
//! clustered lights, and the solid-motor cloud path are later phases.
//!
//! Pipeline (mirrors the design doc's flow):
//!
//! 1. [`update_plume_signals`] reads each firing [`Engine`]'s runtime state
//!    ([`EngineThrust`] + [`ThrottleState`]) plus the craft's local ambient
//!    pressure and publishes a compact, render-facing [`PlumeSignals`] on the
//!    engine entity (the single typed boundary — visual code never reaches back
//!    into gameplay components). A [`PlumeDebugOverride`] resource can drive the
//!    signals directly for authoring / headless capture (frozen controller
//!    values, the doc's authoring workflow).
//! 2. [`update_plume_visuals`] resolves those signals through the propellant
//!    preset + pressure-response curves into [`PlumeParams`] (the flat uniform the
//!    shader renders) and toggles plume visibility.
//!
//! The look itself lives in `assets/shaders/plume.wgsl`.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};

use thalos_physics_canonical::canonical::Epoch;
use thalos_shipyard::{Engine, EngineActivation, EngineGeometry, EngineThrust, ReactantRatio, Resource};

use crate::SimStage;
use crate::rendering::SimulationState;
use crate::shipyard_editor::core::EditorPart;

/// Shared unit-quad plume mesh handle. The quad lives in plume-local space
/// (x = lateral in [-1, 1], y = axial in [0, 1], 0 at the nozzle exit); the
/// vertex shader rebuilds it as a camera-facing, axis-locked billboard.
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
                    // After the engine-thrust plumbing writes `EngineThrust`.
                    .after(crate::engine::update_engine_thrust),
            );
    }
}

fn insert_plume_mesh(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let handle = meshes.add(plume_billboard_mesh());
    commands.insert_resource(PlumeMesh(handle));
}

/// A subdivided unit quad, x ∈ [-1, 1] (lateral), y ∈ [0, 1] (axial, 0 = exit).
/// Positions are normalized; the vertex shader scales and orients them. A few
/// axial rows keep the (cheap) linear interpolation of the profile smooth.
fn plume_billboard_mesh() -> Mesh {
    const ROWS: usize = 24;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((ROWS + 1) * 2);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((ROWS + 1) * 2);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity((ROWS + 1) * 2);
    let mut indices: Vec<u32> = Vec::with_capacity(ROWS * 6);
    for row in 0..=ROWS {
        let t = row as f32 / ROWS as f32;
        for (col, x) in [-1.0_f32, 1.0].into_iter().enumerate() {
            positions.push([x, t, 0.0]);
            normals.push([0.0, 0.0, 1.0]);
            uvs.push([col as f32, t]);
        }
    }
    for row in 0..ROWS {
        let a = (row * 2) as u32;
        let b = a + 1;
        let c = a + 2;
        let d = a + 3;
        indices.extend_from_slice(&[a, c, b, b, c, d]);
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
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
    /// `≈1` perfectly expanded. Large sentinel in a true vacuum.
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
/// uses it instead of live engine state — the doc's "scrub with frozen controller
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

/// Visual propellant family — the starting colour/opacity defaults an engine
/// profile then tunes. Derived from the engine's reactants (the doc's "propellant
/// preset supplies a visual family").
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

    /// `(core, mid, edge)` linear-RGB palette, HDR core intensity, and base
    /// opacity. Starting points — iterated by screenshot.
    fn palette(self) -> Palette {
        match self {
            // Pale blue-white core, blue plume, blue-violet sheath.
            Self::Methalox => Palette {
                core: Vec3::new(0.72, 0.84, 1.0),
                mid: Vec3::new(0.30, 0.52, 1.0),
                edge: Vec3::new(0.24, 0.28, 0.72),
                intensity: 24.0,
                density: 0.85,
            },
            // Warm white core, orange plume, sooty amber sheath.
            Self::Kerolox => Palette {
                core: Vec3::new(1.0, 0.86, 0.58),
                mid: Vec3::new(1.0, 0.48, 0.16),
                edge: Vec3::new(0.55, 0.20, 0.06),
                intensity: 19.0,
                density: 1.05,
            },
            // Faint blue-white, near-invisible — legibility comes from the core
            // glow, not a saturated flame.
            Self::Hydrolox => Palette {
                core: Vec3::new(0.72, 0.80, 1.0),
                mid: Vec3::new(0.42, 0.50, 0.92),
                edge: Vec3::new(0.36, 0.40, 0.70),
                intensity: 13.0,
                density: 0.50,
            },
            Self::Generic => Palette {
                core: Vec3::new(0.95, 0.85, 0.75),
                mid: Vec3::new(0.85, 0.55, 0.40),
                edge: Vec3::new(0.45, 0.30, 0.28),
                intensity: 15.0,
                density: 0.75,
            },
        }
    }
}

struct Palette {
    core: Vec3,
    mid: Vec3,
    edge: Vec3,
    intensity: f32,
    density: f32,
}

// ---------------------------------------------------------------------------
// Material + resolved uniform
// ---------------------------------------------------------------------------

/// Resolved per-engine plume profile — the flat uniform the shader renders.
/// Packed as vec4s so the std140 layout matches `plume.wgsl` unambiguously.
#[derive(Clone, Copy, ShaderType, Debug)]
pub struct PlumeParams {
    /// rgb = hot core colour, a = HDR emission scale.
    pub core_color: Vec4,
    /// rgb = mid plume colour, a = nozzle exit radius (m).
    pub mid_color: Vec4,
    /// rgb = cool sheath / tip colour, a = billboard half-width R_max (m).
    pub edge_color: Vec4,
    /// x = visible axial length (m), y = radial expansion factor,
    /// z = shock-cell count, w = shock-cell contrast.
    pub shape: Vec4,
    /// x = axial core decay, y = shock fade, z = edge softness, w = throttle.
    pub response: Vec4,
    /// x = time (s), y = seed, z = ignition, w = density scale.
    pub anim: Vec4,
}

impl Default for PlumeParams {
    fn default() -> Self {
        Self {
            core_color: Vec4::new(0.72, 0.84, 1.0, 24.0),
            mid_color: Vec4::new(0.30, 0.52, 1.0, 1.0),
            edge_color: Vec4::new(0.24, 0.28, 0.72, 1.2),
            shape: Vec4::new(8.0, 1.4, 5.0, 1.0),
            response: Vec4::new(2.6, 4.0, 0.45, 0.0),
            anim: Vec4::new(0.0, 0.0, 0.0, 0.85),
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
        // The cylindrical billboard rebuilds the strip in the vertex shader.
        // Mesh winding was authored in local XY and does not reliably face the
        // camera after the axis-lock rotation, so default back-face culling
        // drops the whole plume. Disable culling (same pattern as rings).
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Marker + wiring on the plume child entity: which engine it belongs to, its
/// material, nozzle radius, and propellant family.
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
/// [`PlumeDebugOverride`]). Ambient pressure is resolved once per frame from the
/// craft's altitude over its dominant body's atmosphere.
fn update_plume_signals(
    time: Res<Time>,
    sim: Res<SimulationState>,
    over: Res<PlumeDebugOverride>,
    mut engines: Query<
        (&Engine, Option<&EngineActivation>, &EngineThrust, &mut PlumeSignals),
        Without<EditorPart>,
    >,
) {
    let dt = time.delta_secs();
    let ambient = over
        .ambient_pressure_pa
        .unwrap_or_else(|| craft_ambient_pressure_pa(&sim) as f32);

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
        sig.pressure_ratio = pressure_ratio(engine, ambient);
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
        mat.params = resolve_params(sig, visual.nozzle_radius_m, visual.family, elapsed, seed_of(visual.engine));
    }
}

// ---------------------------------------------------------------------------
// Curve evaluation (signals + preset -> uniform)
// ---------------------------------------------------------------------------

/// The "curve evaluator": map [`PlumeSignals`] + propellant preset into the flat
/// [`PlumeParams`] the shader renders. Pressure response is authored over
/// `log2(pressure_ratio)` since the ratio spans orders of magnitude.
fn resolve_params(
    sig: &PlumeSignals,
    nozzle_radius_m: f32,
    family: PropellantFamily,
    time: f32,
    seed: f32,
) -> PlumeParams {
    let pal = family.palette();

    // Compressed pressure coordinate: 0 at perfect expansion, negative dense /
    // overexpanded (sea level), positive toward vacuum.
    let lr = sig.pressure_ratio.clamp(0.03125, 64.0).log2();
    // 0 (dense atmosphere) → 1 (vacuum), the master expansion lever.
    let vac = smoothstep(-1.0, 4.0, lr);

    let expansion = lerp(1.15, 3.4, vac);
    let length_factor = lerp(6.5, 16.0, vac);
    let core_decay = lerp(3.4, 1.7, vac);
    let edge_softness = lerp(0.35, 0.62, vac);

    // Shock diamonds need back-pressure: gate on real ambient pressure and fade
    // as the plume goes underexpanded.
    let atmo_gate = smoothstep(1_500.0, 25_000.0, sig.ambient_pressure_pa);
    let cell_contrast = (1.0 - smoothstep(0.0, 3.0, lr.max(0.0))) * atmo_gate;
    let cell_count = lerp(3.0, 7.0, cell_contrast);

    // Ignition shortens/dims the plume during the transient.
    let ignition = sig.ignition.clamp(0.0, 1.0);
    let length_m = nozzle_radius_m * length_factor * lerp(0.35, 1.0, ignition);
    let max_radius_m = nozzle_radius_m * expansion * 1.18;

    PlumeParams {
        core_color: pal.core.extend(pal.intensity),
        mid_color: pal.mid.extend(nozzle_radius_m),
        edge_color: pal.edge.extend(max_radius_m),
        shape: Vec4::new(length_m, expansion, cell_count, cell_contrast),
        response: Vec4::new(core_decay, 4.0, edge_softness, sig.throttle),
        anim: Vec4::new(time, seed, ignition, pal.density),
    }
}

/// Ambient pressure (Pa) at the craft, from its altitude over the dominant
/// body's atmosphere. Zero in a vacuum / airless-body SOI.
fn craft_ambient_pressure_pa(sim: &SimulationState) -> f64 {
    let body_id = sim.simulation.dominant_body();
    let Some(body) = sim.system.bodies.get(body_id) else {
        return 0.0;
    };
    let Some(atmosphere) = body.terrestrial_atmosphere.as_ref() else {
        return 0.0;
    };
    let body_pos = sim
        .ephemeris
        .state(body_id, Epoch(sim.simulation.sim_time()))
        .position;
    let ship_pos = sim.simulation.ship_state().position;
    let altitude_m = (ship_pos - body_pos).length() - body.radius_m;
    atmosphere
        .sample_at_altitude_m(
            altitude_m,
            body.surface_pressure_pa(),
            body.surface_gravity_m_s2(),
        )
        .pressure_pa
}

/// `r = p_exit / p_ambient`. Nozzle-exit pressure is approximated from the
/// engine's design point (a first-slice constant until the propulsion layer
/// exposes a real `p_exit`); a true vacuum yields a large sentinel ratio.
fn pressure_ratio(_engine: &Engine, ambient_pressure_pa: f32) -> f32 {
    // Design exit pressure ~ 45 kPa: overexpanded near sea level (visible shock
    // diamonds), strongly underexpanded (broad, feathered) toward vacuum.
    const DESIGN_EXIT_PRESSURE_PA: f32 = 45_000.0;
    if ambient_pressure_pa <= 1.0 {
        64.0
    } else {
        (DESIGN_EXIT_PRESSURE_PA / ambient_pressure_pa).clamp(0.03125, 64.0)
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
