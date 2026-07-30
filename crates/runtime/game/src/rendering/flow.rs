//! One aerothermal signal boundary for every vehicle flow effect.
//!
//! Engine plumes, reentry shock layers, vapour/sonic cones, contrails, heat haze
//! and afterburners are all functions of the *same* handful of freestream
//! numbers. Before this module each of them would have had to re-derive ambient
//! conditions its own way, and they would then disagree at exactly the moment
//! several are on screen at once (reentry with the engine lit). [`FlowSignals`]
//! is the single typed answer to "what air is this vehicle flying through, how
//! fast, and how hard is it being heated" — visual code reads this, never the
//! simulation components directly.
//!
//! This is the generalisation of `plume::PlumeSignals`, which stays: throttle,
//! ignition and nozzle pressure ratio are genuinely *per engine*, while
//! everything here is per vehicle. The plume now takes its ambient half from
//! here instead of resolving atmosphere itself.
//!
//! **Not derived from `AeroReadout`.** That readout only exists inside the Avian
//! bubble (`aero::apply_aero_forces` requires an active bubble and
//! `LocalCraftBody`), which is precisely where reentry *is not*: a vehicle
//! entering from orbit is on the canonical propagator, far above any bubble. So
//! the flow state is resolved from canonical ship state against the dominant
//! body — the same path `plume` already used for ambient pressure — and is
//! therefore valid from orbit to touchdown with no regime switch.

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::{Aabb, MeshAabb};
use bevy::math::{Affine3A, Vec3A};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use super::types::{PlayerShip, SimulationState, SolarSystemState};
use crate::SimStage;

/// Sutton–Graves cold-wall stagnation-point heating coefficient for air, SI:
/// `q = K·sqrt(rho / R_n)·v³` in W/m² with rho in kg/m³, `R_n` in m, v in m/s.
///
/// This is the standard engineering correlation, and it is the *only* quantity
/// that separates "fast in thin air" from "fast in thick air" — which is the
/// whole difference between a benign high orbit pass and a fireball. Brightness
/// must ride on this, not on speed alone.
const SUTTON_GRAVES_K: f64 = 1.7415e-4;

/// Ratio of specific heats used for the stagnation-temperature rise. The
/// atmosphere model carries a per-body `gamma` in its optional profile, but not
/// on the sample, so this is the air-like default; it only sets how fast
/// `T_stagnation` climbs with Mach, never whether the effect exists.
const GAMMA: f64 = 1.4;

/// Airspeed below which the flow is treated as still. Below this the direction
/// of `flow_from_dir` is numerical noise, and every consumer wants "off" rather
/// than a jittering axis.
const AIRSPEED_FLOOR_M_S: f64 = 1.0;

/// Fraction of the craft's cross-flow bounding radius used as the effective
/// nose radius in the heating correlation.
///
/// **This is a stand-in, and it is the seam where a real value belongs.** A true
/// `R_n` is a property of the windward geometry — a capsule's heat shield or a
/// wing leading edge — and the shipyard does not publish one today. A single
/// fraction of the bounding radius at least scales with the vehicle and keeps
/// blunt bodies cooler than slender ones for the right reason (bigger `R_n` =
/// lower heat flux). When parts gain an authored leading-edge radius, this
/// constant is what it replaces.
const NOSE_RADIUS_FRACTION: f64 = 0.35;

/// Aerothermal state of the player vehicle's freestream, for visual consumers.
///
/// **Single writer: [`update_flow_signals`].** Everything here is a projection
/// of canonical ship state plus the dominant body's atmosphere.
///
/// Single-vehicle by construction, because `SimulationState::ship_state()` is:
/// there is one canonical player craft. When a second controllable vehicle
/// exists this becomes a per-vehicle component published by the same system, and
/// consumers already read it through a resource lookup rather than reaching into
/// the simulation, so that change stays local to this file.
#[derive(Resource, Debug, Clone, Copy)]
pub struct FlowSignals {
    /// False in vacuum / airless SOI / above the Kármán line. Every effect that
    /// needs air should gate on this rather than testing a density epsilon.
    pub in_atmosphere: bool,
    /// Altitude above the dominant body's reference radius, m.
    pub altitude_m: f32,
    /// Static (ambient) pressure, Pa.
    pub ambient_pressure_pa: f32,
    /// Air density, kg/m³.
    pub density_kg_m3: f32,
    /// Static air temperature, K.
    pub static_temp_k: f32,
    /// Speed of sound, m/s.
    pub speed_of_sound_m_s: f32,
    /// Speed relative to the **co-rotating** airmass, m/s. The body's spin is
    /// included: at orbital entry speeds it is a few percent of airspeed, and
    /// heat flux goes as v³, so dropping it would be a ~15 % brightness error.
    pub airspeed_m_s: f32,
    /// Freestream Mach number. Zero in vacuum.
    pub mach: f32,
    /// Dynamic pressure `½ρv²`, Pa.
    pub dynamic_pressure_pa: f32,
    /// Stagnation temperature `T·(1 + (γ−1)/2·M²)`, K — how hot the air gets
    /// when brought to rest on the vehicle. This is the colour-temperature
    /// driver for shock-heated air.
    pub stagnation_temp_k: f32,
    /// Sutton–Graves cold-wall stagnation-point heat flux, W/m².
    pub heat_flux_w_m2: f32,
    /// Unit direction the freestream arrives **from**, in render space. Zero
    /// when `airspeed_m_s` is below the floor. Render space is canonical space
    /// scaled and translated with no rotation, so this is the canonical
    /// direction unchanged.
    pub flow_from_dir: Vec3,
    /// Effective nose radius used in the heating correlation, m. See
    /// [`NOSE_RADIUS_FRACTION`].
    pub nose_radius_m: f32,
    /// Descendant meshes the bounds sweep actually measured. Zero means the sweep
    /// found nothing and every attached effect is running on a default size.
    pub measured_mesh_count: u32,
    /// Radius of the craft's visual bounding sphere about its origin, m.
    pub craft_radius_m: f32,
    /// Half-extents of the craft's visual bounding box about its origin, in
    /// **craft-local axes**, m.
    ///
    /// Attached effects must size themselves against this, not against
    /// [`Self::craft_radius_m`]. A bounding *sphere* is a bad stand-in for an
    /// elongated vehicle: on a 40 m rocket it is a 20 m ball, so a shock layer
    /// hugging that sphere hangs metres out in empty space along every axis
    /// instead of wrapping the hull. The box collapses to the sphere for a
    /// capsule and stays tight for a rocket.
    pub craft_half_extents_m: Vec3,
    /// [`Self::flow_from_dir`] expressed in craft-local axes, so a shader working
    /// in the craft's own frame (as any hull-fitted effect must) does not have to
    /// reconstruct it.
    pub flow_from_local: Vec3,
}

impl Default for FlowSignals {
    fn default() -> Self {
        Self {
            in_atmosphere: false,
            altitude_m: 0.0,
            ambient_pressure_pa: 0.0,
            density_kg_m3: 0.0,
            static_temp_k: 0.0,
            speed_of_sound_m_s: 0.0,
            airspeed_m_s: 0.0,
            mach: 0.0,
            dynamic_pressure_pa: 0.0,
            stagnation_temp_k: 0.0,
            heat_flux_w_m2: 0.0,
            flow_from_dir: Vec3::ZERO,
            nose_radius_m: 1.0,
            measured_mesh_count: 0,
            craft_radius_m: 1.0,
            craft_half_extents_m: Vec3::ONE,
            flow_from_local: Vec3::ZERO,
        }
    }
}

/// Authoring / capture override. Any `Some` field replaces the resolved value,
/// so a headless preset can park a vehicle in a chosen flow state without
/// flying it there — the same pattern as `plume::PlumeDebugOverride`, and the
/// only reason reentry is screenshot-verifiable at all.
///
/// Overrides are applied to the *inputs* where possible (density, airspeed) so
/// the derived quantities stay mutually consistent; a test that forced
/// `heat_flux` directly while leaving airspeed at zero would render a state no
/// atmosphere can produce.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct FlowDebugOverride {
    pub density_kg_m3: Option<f32>,
    pub static_temp_k: Option<f32>,
    pub speed_of_sound_m_s: Option<f32>,
    pub airspeed_m_s: Option<f32>,
    /// Freestream arrival direction in **craft-local** axes, normalized by the
    /// consumer. Lets a preset put the shock cap on a chosen face without
    /// orienting the whole vehicle.
    pub flow_from_local: Option<Vec3>,
}

impl FlowDebugOverride {
    /// True when any field is set — the tell that the resolved flow is being
    /// driven by an authoring surface rather than by flight.
    pub fn active(&self) -> bool {
        self.density_kg_m3.is_some()
            || self.static_temp_k.is_some()
            || self.speed_of_sound_m_s.is_some()
            || self.airspeed_m_s.is_some()
            || self.flow_from_local.is_some()
    }
}

/// Marker excluding a mesh from the craft-bounds sweep.
///
/// **Every flow effect's proxy hull must carry this.** The proxies are descendants
/// of the craft, so without it the sweep measures the effects' own bounding
/// geometry as if it were vehicle structure — and the reentry shell, whose size is
/// *derived* from the result, ends up measuring itself. Their mesh AABBs are also
/// meaningless: a proxy's real extent is applied in its vertex shader, so the
/// asset's AABB is the unscaled template.
#[derive(Component)]
pub struct FlowProxyMesh;

/// How many consecutive stable measurements end the re-measure loop.
///
/// Gating on descendant mesh *count* alone is not enough. Every part entity exists
/// from the first frame, but `Mesh::compute_aabb` returns `None` until the asset is
/// loaded, so an early sweep sees the right number of meshes and the wrong size —
/// and then never looks again. Measured: a ~10 m craft cached as a 4 m box, which
/// scaled every attached effect wrong. So the sweep repeats until the answer stops
/// growing.
const BOUNDS_STABLE_TICKS: u32 = 8;

/// Relative growth below which a new measurement counts as unchanged.
const BOUNDS_STABLE_EPS: f32 = 0.01;

/// Cached craft geometry for flow effects.
///
/// `Mesh::compute_aabb` walks every vertex, so the descendant sweep cannot run
/// unconditionally per frame. It re-measures until the result is stable, and again
/// whenever the craft entity or its descendant mesh count changes — which covers
/// staging separation, the one event that genuinely changes the vehicle's size
/// mid-flight.
#[derive(Resource, Debug, Default)]
pub struct CraftFlowGeometry {
    measured_for: Option<Entity>,
    mesh_count: usize,
    stable_ticks: u32,
    radius_m: f32,
    half_extents_m: Vec3,
}

pub struct FlowSignalsPlugin;

impl Plugin for FlowSignalsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FlowSignals>()
            .init_resource::<FlowDebugOverride>()
            .init_resource::<CraftFlowGeometry>()
            .add_systems(Update, update_flow_signals.in_set(SimStage::Sync));
    }
}

/// Publish [`FlowSignals`] from canonical ship state against the dominant
/// body's atmosphere.
pub fn update_flow_signals(
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    over: Res<FlowDebugOverride>,
    mut geometry: ResMut<CraftFlowGeometry>,
    mut signals: ResMut<FlowSignals>,
    ships: Query<(Entity, &GlobalTransform), With<PlayerShip>>,
    children_q: Query<&Children>,
    mesh_q: Query<(&GlobalTransform, &Mesh3d), Without<FlowProxyMesh>>,
    meshes: Res<Assets<Mesh>>,
) {
    let ship = ships.iter().next();
    if let Some((root, root_gt)) = ship {
        refresh_craft_geometry(
            &mut geometry,
            root,
            root_gt,
            &children_q,
            &mesh_q,
            &meshes,
        );
    }
    let craft_radius_m = geometry.radius_m.max(0.1);
    let craft_half_extents_m = geometry.half_extents_m.max(Vec3::splat(0.05));
    // The nose radius is a *windward* curvature, so it scales with the smallest
    // cross-section, not with the vehicle's length. Using the bounding radius
    // here would make a long rocket read as a very blunt body and under-heat it.
    let nose_radius_m = (craft_half_extents_m.min_element() as f64 * NOSE_RADIUS_FRACTION)
        .max(0.05);

    let resolved = resolve_flow(&sim, &solar);

    let density = over
        .density_kg_m3
        .map(|v| v as f64)
        .unwrap_or(resolved.density_kg_m3);
    let static_temp = over
        .static_temp_k
        .map(|v| v as f64)
        .unwrap_or(resolved.temperature_k);
    let speed_of_sound = over
        .speed_of_sound_m_s
        .map(|v| v as f64)
        .unwrap_or(resolved.speed_of_sound_m_s);
    let airspeed = over
        .airspeed_m_s
        .map(|v| v as f64)
        .unwrap_or(resolved.airspeed_m_s);

    // Direction: a local-axis override is resolved through the craft's current
    // orientation, so a preset says "wind on the nose" and gets that whatever
    // attitude the vehicle is parked in.
    let flow_from_dir = match (over.flow_from_local, ship) {
        (Some(local), Some((_, root_gt))) => {
            (root_gt.rotation() * local).normalize_or_zero()
        }
        (Some(local), None) => local.normalize_or_zero(),
        (None, _) => resolved.flow_from_dir,
    };

    let mach = if speed_of_sound > 0.0 {
        airspeed / speed_of_sound
    } else {
        0.0
    };
    // Only claim air when there is measurable density AND the flow direction is
    // meaningful — the two conditions every consumer would otherwise re-test.
    let in_atmosphere =
        density > 0.0 && airspeed >= AIRSPEED_FLOOR_M_S && flow_from_dir != Vec3::ZERO;

    let stagnation_temp = if static_temp > 0.0 {
        static_temp * (1.0 + 0.5 * (GAMMA - 1.0) * mach * mach)
    } else {
        0.0
    };
    let heat_flux = if density > 0.0 {
        SUTTON_GRAVES_K * (density / nose_radius_m).sqrt() * airspeed.powi(3)
    } else {
        0.0
    };

    *signals = FlowSignals {
        in_atmosphere,
        altitude_m: resolved.altitude_m as f32,
        ambient_pressure_pa: resolved.pressure_pa as f32,
        density_kg_m3: density as f32,
        static_temp_k: static_temp as f32,
        speed_of_sound_m_s: speed_of_sound as f32,
        airspeed_m_s: airspeed as f32,
        mach: mach as f32,
        dynamic_pressure_pa: (0.5 * density * airspeed * airspeed) as f32,
        stagnation_temp_k: stagnation_temp as f32,
        heat_flux_w_m2: heat_flux as f32,
        flow_from_dir,
        nose_radius_m: nose_radius_m as f32,
        measured_mesh_count: geometry.mesh_count as u32,
        craft_radius_m,
        craft_half_extents_m,
        flow_from_local: match ship {
            Some((_, root_gt)) => {
                (root_gt.rotation().inverse() * flow_from_dir).normalize_or_zero()
            }
            None => flow_from_dir,
        },
    };
}

/// Freestream state resolved from the simulation, before any override.
struct ResolvedFlow {
    altitude_m: f64,
    pressure_pa: f64,
    density_kg_m3: f64,
    temperature_k: f64,
    speed_of_sound_m_s: f64,
    airspeed_m_s: f64,
    flow_from_dir: Vec3,
}

impl ResolvedFlow {
    const VACUUM: Self = Self {
        altitude_m: 0.0,
        pressure_pa: 0.0,
        density_kg_m3: 0.0,
        temperature_k: 0.0,
        speed_of_sound_m_s: 0.0,
        airspeed_m_s: 0.0,
        flow_from_dir: Vec3::ZERO,
    };
}

fn resolve_flow(sim: &SimulationState, solar: &SolarSystemState) -> ResolvedFlow {
    let body_id = sim.simulation.dominant_body();
    let Some(catalog) = sim.system.bodies.get(body_id) else {
        return ResolvedFlow::VACUUM;
    };
    // `SolarSystemState` is the frame-local authority for body state, and the
    // only source carrying spin — which the co-rotating airmass needs.
    let Some(state) = solar.states.as_ref().and_then(|s| s.get(body_id)) else {
        return ResolvedFlow::VACUUM;
    };

    let ship = sim.simulation.ship_state();
    let rel = ship.position - state.position;
    let radius = rel.length();
    let altitude_m = radius - catalog.radius_m;

    // Velocity of the co-rotating airmass at the craft, then the craft's
    // velocity relative to it.
    let air_velocity = state.velocity + state.angular_velocity.cross(rel);
    let air_relative = ship.velocity - air_velocity;
    let airspeed_m_s = air_relative.length();
    let flow_from_dir = if airspeed_m_s >= AIRSPEED_FLOOR_M_S {
        (-air_relative / airspeed_m_s).as_vec3()
    } else {
        Vec3::ZERO
    };

    let Some(atmosphere) = catalog.terrestrial_atmosphere.as_ref() else {
        return ResolvedFlow {
            altitude_m,
            airspeed_m_s,
            flow_from_dir,
            ..ResolvedFlow::VACUUM
        };
    };
    let sample = atmosphere.sample_at_altitude_m(
        altitude_m,
        catalog.surface_pressure_pa(),
        catalog.surface_gravity_m_s2(),
    );
    ResolvedFlow {
        altitude_m,
        pressure_pa: sample.pressure_pa,
        density_kg_m3: sample.density_kg_m3,
        temperature_k: sample.temperature_k,
        speed_of_sound_m_s: sample.speed_of_sound_m_s,
        airspeed_m_s,
        flow_from_dir,
    }
}

/// Re-measure the craft's bounding radius when the vehicle changes.
fn refresh_craft_geometry(
    geometry: &mut CraftFlowGeometry,
    root: Entity,
    root_gt: &GlobalTransform,
    children_q: &Query<&Children>,
    mesh_q: &Query<(&GlobalTransform, &Mesh3d), Without<FlowProxyMesh>>,
    meshes: &Assets<Mesh>,
) {
    let mut count = 0usize;
    let mut stack: Vec<Entity> = Vec::new();
    if let Ok(c) = children_q.get(root) {
        stack.extend(c.iter());
    }
    let mut pending: Vec<Entity> = Vec::new();
    while let Some(e) = stack.pop() {
        if mesh_q.get(e).is_ok() {
            count += 1;
            pending.push(e);
        }
        if let Ok(c) = children_q.get(e) {
            stack.extend(c.iter());
        }
    }
    let changed = geometry.measured_for != Some(root) || geometry.mesh_count != count;
    if changed {
        geometry.stable_ticks = 0;
    } else if geometry.stable_ticks >= BOUNDS_STABLE_TICKS {
        return;
    }

    let root_inv = root_gt.affine().inverse();
    let mut radius = 0.0f32;
    let mut extents = Vec3::ZERO;
    for e in pending {
        let Ok((gt, mesh3d)) = mesh_q.get(e) else {
            continue;
        };
        let Some(aabb) = meshes.get(&mesh3d.0).and_then(Mesh::compute_aabb) else {
            continue;
        };
        accumulate_bounds(&aabb, root_inv * gt.affine(), &mut radius, &mut extents);
    }
    geometry.measured_for = Some(root);
    geometry.mesh_count = count;
    if radius <= 0.0 {
        // Nothing measurable yet (assets still loading). Keep looking.
        geometry.stable_ticks = 0;
        return;
    }
    let grew = radius > geometry.radius_m * (1.0 + BOUNDS_STABLE_EPS);
    geometry.radius_m = radius;
    geometry.half_extents_m = extents;
    if grew {
        geometry.stable_ticks = 0;
    } else {
        geometry.stable_ticks = geometry.stable_ticks.saturating_add(1);
    }
}

/// Fold one mesh AABB into the craft's bounding radius and per-axis half-extents,
/// both measured about the craft origin in craft-local axes.
fn accumulate_bounds(aabb: &Aabb, to_root: Affine3A, radius: &mut f32, extents: &mut Vec3) {
    let c = aabb.center;
    let h = aabb.half_extents;
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let corner = Vec3A::new(c.x + sx * h.x, c.y + sy * h.y, c.z + sz * h.z);
                let local = Vec3::from(to_root.transform_point3a(corner));
                *radius = radius.max(local.length());
                *extents = extents.max(local.abs());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared proxy geometry
// ---------------------------------------------------------------------------

/// Closed prism template shared by every ray-marched flow effect: `xy` = a
/// unit-circle direction, `z` = axial fraction 0..1, cap centres at `xy = 0`.
///
/// Each effect's vertex shader maps the rings onto its own bounding surface (the
/// plume scales them by its envelope bound, the reentry cap wraps them onto a
/// sphere), so one mesh serves both. **It is never seen** — it exists so the
/// rasterizer visits every pixel the volume can touch, and the silhouette comes
/// from the density integral.
///
/// Wound outward, so culling back faces leaves exactly one fragment per ray.
pub fn axial_proxy_prism_mesh(sides: usize, rings: usize) -> Mesh {
    let ring_count = rings + 1;
    let tube_verts = ring_count * sides;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(tube_verts + 2);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(tube_verts + 2);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(tube_verts + 2);
    let mut indices: Vec<u32> = Vec::with_capacity(rings * sides * 6 + sides * 6);

    for ring in 0..ring_count {
        let t = ring as f32 / rings as f32;
        for k in 0..sides {
            let theta = std::f32::consts::TAU * k as f32 / sides as f32;
            let (sin, cos) = theta.sin_cos();
            positions.push([cos, sin, t]);
            normals.push([cos, sin, 0.0]);
            uvs.push([k as f32 / sides as f32, t]);
        }
    }
    let near_centre = tube_verts as u32;
    let far_centre = near_centre + 1;
    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, 0.0, -1.0]);
    uvs.push([0.5, 0.0]);
    positions.push([0.0, 0.0, 1.0]);
    normals.push([0.0, 0.0, 1.0]);
    uvs.push([0.5, 1.0]);

    let idx = |ring: usize, k: usize| (ring * sides + k % sides) as u32;
    for ring in 0..rings {
        for k in 0..sides {
            let a = idx(ring, k);
            let b = idx(ring, k + 1);
            let c = idx(ring + 1, k);
            let d = idx(ring + 1, k + 1);
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    for k in 0..sides {
        // Near cap faces −axis, far cap faces +axis; both wound outward.
        indices.extend_from_slice(&[near_centre, idx(0, k + 1), idx(0, k)]);
        indices.extend_from_slice(&[far_centre, idx(rings, k), idx(rings, k + 1)]);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Heat flux must be driven by air *and* speed together. The standing trap
    /// is a brightness term that rides on speed alone: that lights a fireball
    /// in high orbit, where a craft is at full orbital velocity in effectively
    /// no air.
    #[test]
    fn heating_needs_both_speed_and_density() {
        let fast_thin = SUTTON_GRAVES_K * (1.0e-9f64 / 1.0).sqrt() * 7800.0f64.powi(3);
        let slow_thick = SUTTON_GRAVES_K * (1.225f64 / 1.0).sqrt() * 250.0f64.powi(3);
        let fast_thick = SUTTON_GRAVES_K * (1.0e-4f64 / 1.0).sqrt() * 7800.0f64.powi(3);
        assert!(
            fast_thin < slow_thick,
            "orbital speed in vacuum ({fast_thin}) must not out-heat a subsonic sea-level pass ({slow_thick})"
        );
        assert!(
            fast_thick > slow_thick * 100.0,
            "real entry ({fast_thick}) must dominate ordinary flight ({slow_thick})"
        );
    }

    /// A blunter nose is *cooler*, not hotter — the sign of the `R_n` dependence
    /// is the reason heat shields are round, and getting it backwards would make
    /// capsules the brightest thing in the sky.
    #[test]
    fn blunter_nose_lowers_heat_flux() {
        let q = |r: f64| SUTTON_GRAVES_K * (1.0e-4f64 / r).sqrt() * 7000.0f64.powi(3);
        assert!(q(2.0) < q(0.2));
    }

    /// The shared proxy prism must be closed and outward-wound, or culling one
    /// face leaves holes the raymarch never gets asked about.
    #[test]
    fn proxy_prism_is_closed() {
        let sides = 12;
        let rings = 8;
        let mesh = axial_proxy_prism_mesh(sides, rings);
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("expected u32 indices");
        };
        // Tube quads (2 tris each) + 2 cap fans.
        assert_eq!(indices.len(), (rings * sides * 2 + sides * 2) * 3);

        // Every undirected edge is shared by exactly two triangles in a closed
        // manifold.
        use std::collections::HashMap;
        let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
        for tri in indices.chunks(3) {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        assert!(
            edges.values().all(|&n| n == 2),
            "proxy hull is not closed: {} edges are not shared by two faces",
            edges.values().filter(|&&n| n != 2).count()
        );
    }
}
