//! Bevy/Avian boundary for near-surface local rigidbody physics.
//!
//! This crate deliberately contains the Avian dependency so `thalos_physics_canonical`
//! remains pure Rust and rails/canonical code does not learn Avian APIs.

use std::{collections::HashMap, sync::Arc};

use avian3d::prelude::*;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use thalos_body_render::{
    GpuAtlasMirrorHandle, GpuAtlasMirrorHeightSource, HeightSource, TerrainPatchBasis,
    TerrainPatchConfig,
};

/// `tile_lod_m` hint for the finest CPU-synthesizable terrain detail when a
/// height source falls back to its procedural pipeline (mirrors the game's
/// `PHYSICS_QUERY_TILE_LOD_M`).
const PHYSICS_QUERY_TILE_LOD_M: f32 = 0.5;
use thalos_physics_canonical::canonical::CraftId;
use thalos_physics_canonical::surface_local::SurfaceLocalFrame;
use thalos_terrain::PlanetSurface;
use thalos_world::BodyId;

pub mod avian {
    pub use avian3d::prelude::{
        AngularInertia, AngularVelocity, CenterOfMass, CoefficientCombine, Collider,
        CollisionLayers, ConstantAngularAcceleration, ConstantForce, ConstantLinearAcceleration,
        ConstantTorque, ContactGraph, CustomPositionIntegration, Friction, LayerMask,
        LinearVelocity, LockedAxes, Mass, NoAutoAngularInertia, NoAutoCenterOfMass, NoAutoMass,
        Physics, PhysicsDebugPlugin, PhysicsGizmos, PhysicsSchedule, PhysicsStepSystems,
        PhysicsTime, Position, RayHitData, Restitution, RigidBody, Rotation, SleepingDisabled,
        SpatialQuery, SpatialQueryFilter, SweptCcd,
    };
}

/// Collision-layer bit for ground colliders (terrain heightfield, runway slab).
pub const GROUND_LAYER: u32 = 1 << 0;
/// Collision-layer bit for player/craft rigid bodies.
pub const CRAFT_LAYER: u32 = 1 << 1;

/// Collision layers for a ground collider: it is a member of `GROUND` and, by
/// default, collides with everything (so gearless craft hulls still rest on it).
pub fn ground_collision_layers() -> avian::CollisionLayers {
    avian::CollisionLayers::new(avian::LayerMask(GROUND_LAYER), avian::LayerMask::ALL)
}

/// Collision layers for a **wheeled** craft hull: member of `CRAFT`, collides
/// with everything **except `GROUND`**. The raycast spring-damper landing gear
/// is the sole ground interface for such craft; the hull never produces solver
/// contact against the ground (which otherwise fought the gear and flung the
/// craft on its gear). Crash detection switches to the gear's weight-on-wheels
/// signal. Gearless craft (landers/rockets) keep the default all-vs-all layers
/// so their hull/legs rest on the ground directly. See `docs/surface_local.md`.
pub fn wheeled_craft_collision_layers() -> avian::CollisionLayers {
    avian::CollisionLayers::new(
        avian::LayerMask(CRAFT_LAYER),
        avian::LayerMask(!GROUND_LAYER),
    )
}

#[derive(Resource, Debug, Clone)]
pub struct LocalBubbleConfig {
    pub handoff_agl_m: f64,
    pub patch_half_extent_m: f64,
    pub patch_resolution: u32,
    pub patch_rebuild_distance_m: f64,
    pub stable_contact_time_s: f64,
    pub max_stable_speed_m_s: f64,
    pub max_stable_angular_speed_rad_s: f64,
    pub debug_drop_height_m: f64,
    pub debug_drop_speed_m_s: f64,
}

impl Default for LocalBubbleConfig {
    fn default() -> Self {
        Self {
            handoff_agl_m: 20_000.0,
            patch_half_extent_m: 4096.0,
            // Collider window resolution. The terrain collider is a static
            // trimesh whose pose is re-synced every frame (it co-rotates with
            // the planet in the body-centered bubble), so Avian re-evaluates
            // broad/narrow phase against all its triangles each frame — the
            // dominant surface-frame CPU cost. 129² (~32k tris) covered far
            // more ground than any resting craft contacts; 65² (~8k tris)
            // keeps native-texel density in a still-generous window around the
            // craft at a quarter of the collision cost. See docs/surface.md.
            patch_resolution: 65,
            patch_rebuild_distance_m: 1024.0,
            stable_contact_time_s: 2.0,
            max_stable_speed_m_s: 0.5,
            max_stable_angular_speed_rad_s: 0.05,
            debug_drop_height_m: 250.0,
            debug_drop_speed_m_s: 5.0,
        }
    }
}

#[derive(Resource, Default, Clone)]
pub struct TerrainSurfaceRegistry {
    surfaces: HashMap<BodyId, Arc<PlanetSurface>>,
}

#[derive(Resource, Default, Clone)]
pub struct HeightSourceRegistry {
    sources: HashMap<BodyId, Arc<dyn HeightSource>>,
    gpu_mirrors: HashMap<BodyId, GpuAtlasMirrorHandle>,
}

impl HeightSourceRegistry {
    pub fn insert(&mut self, body_id: BodyId, source: Arc<dyn HeightSource>) {
        self.gpu_mirrors.remove(&body_id);
        self.sources.insert(body_id, source);
    }

    pub fn insert_gpu_mirror_source(
        &mut self,
        body_id: BodyId,
        source: GpuAtlasMirrorHeightSource,
    ) {
        let mirror = source.mirror();
        self.gpu_mirrors.insert(body_id, mirror);
        self.sources.insert(body_id, Arc::new(source));
    }

    pub fn get(&self, body_id: BodyId) -> Option<Arc<dyn HeightSource>> {
        self.sources.get(&body_id).cloned()
    }

    pub fn contains(&self, body_id: BodyId) -> bool {
        self.sources.contains_key(&body_id)
    }

    pub fn gpu_mirror(&self, body_id: BodyId) -> Option<GpuAtlasMirrorHandle> {
        self.gpu_mirrors.get(&body_id).cloned()
    }
}

impl TerrainSurfaceRegistry {
    pub fn insert(&mut self, body_id: BodyId, surface: Arc<PlanetSurface>) {
        self.surfaces.insert(body_id, surface);
    }

    pub fn get(&self, body_id: BodyId) -> Option<Arc<PlanetSurface>> {
        self.surfaces.get(&body_id).cloned()
    }

    pub fn contains(&self, body_id: BodyId) -> bool {
        self.surfaces.contains_key(&body_id)
    }
}

/// Persistent state for the player ship's Avian rigid body and any
/// proximity-attached terrain collider patch.
///
/// The bubble exists as long as the ship does — the rigid body, collider
/// geometry, and contact graph stay alive across every regime. Whether
/// Avian's *integrator* owns translation each frame, however, is decided
/// by the `AvianOwnership` predicate in `crates/game/src/local_physics.rs`:
/// canonical Kepler propagation owns coasting flight (drift-free), and
/// Avian only takes over when there is a non-gravity force to integrate
/// (thrust, terrain contact). This avoids the orbital drift that any
/// time-stepped integrator accumulates against analytical Kepler when
/// asked to integrate `−μr/r³` over many frames. `terrain_entity` is only
/// `Some` while a body's surface is close enough to merit collider
/// geometry; in deep space the bubble runs without a collider with
/// `basis = identity` and `center_surface_body_m = ZERO`.
///
/// Ship Avian bodies live in the **surface-local frame (SLF)** — a
/// body-fixed tangent frame anchored at a surface point, Y-up, with small
/// (meters–km) coordinates near the anchor; see
/// `thalos_physics_canonical::surface_local` and `docs/surface_local.md`.
/// Gravity + the rotating-frame centrifugal/Coriolis terms come from
/// `surface_local_acceleration`. Ground colliders are static in this frame:
/// posed once from their body-fixed geometry via
/// `frame.rotation_body_to_frame`, never per-frame. The frame is rebuilt
/// (re-anchored) when the craft drifts too far from the anchor; the EVA
/// capsule still uses the body-centered inertial seam until its fold-in.
#[derive(Resource, Debug, Clone)]
pub struct LocalBubble {
    pub id: u64,
    pub body_id: BodyId,
    pub craft_entity: Entity,
    /// The surface-local frame ship physics integrates in. Meaningful for
    /// ships only this slice (EVA keeps the body-centered seam).
    pub frame: SurfaceLocalFrame,
    pub terrain_entity: Option<Entity>,
    pub center_dir_body: DVec3,
    pub center_surface_body_m: DVec3,
    pub basis: TerrainPatchBasis,
    /// Metric lateral half-extent of the attached collider patch. Drives a
    /// window-relative rebuild so the small tile-based collider window
    /// (`docs/surface.md` §3.6, only tens of metres) re-centers before the
    /// craft drifts off its edge — the global `patch_rebuild_distance_m` is
    /// too coarse for it. Zero when no patch is attached, and left at the
    /// fallback patch's `half_extent_m` for the coarse tangent-grid path.
    pub patch_half_extent_m: f64,
    pub stable_contact_s: f64,
    pub stable_landed: bool,
    /// `HeightSource::revision()` snapshot taken when `terrain_entity`
    /// was last (re)built. Drives the rebuild-on-mirror-update path in
    /// `maintain_terrain_patch`: when the source's revision advances
    /// past this snapshot, the patch is stale relative to the surface
    /// the renderer is now showing and must be rebuilt.
    pub terrain_built_at_revision: u64,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct ActiveLocalBubble {
    pub bubble: Option<LocalBubble>,
    pub next_id: u64,
}

impl ActiveLocalBubble {
    pub fn allocate_id(&mut self) -> u64 {
        self.next_id = self.next_id.max(1);
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalCraftBody {
    pub craft_id: CraftId,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct TerrainColliderPatch {
    pub body_id: BodyId,
    pub center_dir: DVec3,
    pub half_extent_m: f64,
    pub resolution: u32,
}

#[derive(Component, Debug, Clone)]
pub struct LocalCraftColliderPrimitives(pub Vec<LocalPrimitiveCollider>);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalPrimitiveShape {
    Cuboid { x: f64, y: f64, z: f64 },
    Cylinder { radius: f64, height: f64 },
    Cone { radius: f64, height: f64 },
    Sphere { radius: f64 },
    Capsule { radius: f64, length: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalPrimitiveCollider {
    pub offset_m: DVec3,
    pub rotation: DQuat,
    pub shape: LocalPrimitiveShape,
}

#[derive(Debug, Clone)]
pub struct LocalCraftSpawn {
    pub craft_id: CraftId,
    pub position_m: DVec3,
    pub rotation: DQuat,
    pub linear_velocity_m_s: DVec3,
    pub angular_velocity_rad_s: DVec3,
    pub mass_kg: f64,
    pub angular_inertia_kg_m2: DVec3,
    pub collider_primitives: Vec<LocalPrimitiveCollider>,
}

#[derive(Debug, Clone)]
pub struct SpawnedTerrainPatch {
    pub entity: Entity,
    /// Body-fixed surface point at the patch centre (drives drift-rebuild and
    /// the collider's SLF pose via [`patch_basis_rotation`]).
    pub center_surface_body_m: DVec3,
    /// Patch-tangent basis (`tangent_x`, `normal`/up, `tangent_z`) the
    /// heightfield is authored in.
    pub basis: TerrainPatchBasis,
    /// Metric lateral half-extent of the heightfield window.
    pub half_extent_m: f64,
}

pub struct LocalPhysicsPlugin;

impl Plugin for LocalPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsPlugins::default())
            // Canonical state owns each craft body's `Position`/`Rotation`: the
            // game writes them directly every frame (`snap_avian_from_canonical`,
            // the EVA walk controller, terrain-patch pose). Nothing positions a
            // physics body via `Transform`. Avian's reverse sync
            // (`transform_to_position`) is therefore unused — and harmful: while
            // the physics clock is warp-paused, `position_to_transform` doesn't
            // run, so a body's `Transform` goes stale relative to a freshly
            // snapped `Rotation` (e.g. the landing spawn re-poses canonical after
            // the bubble already spawned with the parking-orbit placeholder). On
            // unpause `transform_to_position` would clobber the snapped rotation
            // with that stale `Transform`, snapping the ship off retrograde. Keep
            // the one-way `position_to_transform` so renderers/debug still see the
            // pose; drop the reverse direction. See `docs/surface.md`.
            .insert_resource(avian3d::physics_transform::PhysicsTransformConfig {
                transform_to_position: false,
                ..default()
            })
            .insert_resource(Gravity::ZERO)
            .init_resource::<TerrainSurfaceRegistry>()
            .init_resource::<HeightSourceRegistry>()
            .init_resource::<ActiveLocalBubble>()
            .init_resource::<LocalBubbleConfig>();
    }
}

pub fn terrain_patch_config(config: &LocalBubbleConfig) -> TerrainPatchConfig {
    TerrainPatchConfig {
        half_extent_m: config.patch_half_extent_m,
        resolution: config.patch_resolution,
    }
}

/// Rotation taking a patch-tangent basis (`X = tangent_x`, `Y = normal/up`,
/// `Z = tangent_z`) into body-fixed axes. The heightfield/runway colliders are
/// authored in this tangent frame (height along local `Y`), so composing this
/// with the body-fixed→SLF rotation gives their SLF pose.
pub fn patch_basis_rotation(basis: &TerrainPatchBasis) -> DQuat {
    DQuat::from_mat3(&DMat3::from_cols(basis.tangent_x, basis.normal, basis.tangent_z))
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_terrain_collider_patch(
    commands: &mut Commands,
    body_id: BodyId,
    height_source: &dyn HeightSource,
    body_radius_m: f64,
    center_dir_body: DVec3,
    config: &LocalBubbleConfig,
    frame: &SurfaceLocalFrame,
) -> SpawnedTerrainPatch {
    // A **solid heightfield**, not a one-sided trimesh: parry's heightfield has
    // a defined interior, so resting contact resolves gently and a craft that
    // dips a little into it is pushed straight back out — instead of the violent
    // one-step penetration-recovery a one-sided trimesh applies (which launched
    // the craft off its gear). See `docs/surface_local.md` §3.
    //
    // Authored in the patch-tangent frame (`X = tangent_x`, `Y = up = normal`,
    // `Z = tangent_z`) with heights along local `Y`. The grid is sampled in
    // body-fixed coordinates, so the baked heights are independent of the SLF
    // frame — a re-anchor only re-poses the collider (no rebuild). The craft's
    // Avian body lives in the SLF, so [`sync_terrain_collider_pose`] poses this
    // at `Position = R·(center − anchor)`, `Rotation = R · patch_basis_rotation`.
    let center_dir = center_dir_body.normalize_or_zero();
    let basis = TerrainPatchBasis::from_normal(center_dir);
    let h_center = height_source
        .sample_height_m(center_dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(0.0) as f64;
    let center_surface_body_m = center_dir * (body_radius_m + h_center);

    let n = config.patch_resolution.max(2) as usize;
    let half = config.patch_half_extent_m;
    // heights[i][j]: i advances along local Z (tangent_z), j along local X
    // (tangent_x) — parry's heightfield convention. Height = offset from the
    // patch-centre surface along the up normal.
    let mut heights = vec![vec![0.0f64; n]; n];
    for (i, row) in heights.iter_mut().enumerate() {
        let z = (i as f64 / (n as f64 - 1.0) - 0.5) * 2.0 * half;
        for (j, cell) in row.iter_mut().enumerate() {
            let x = (j as f64 / (n as f64 - 1.0) - 0.5) * 2.0 * half;
            let point = center_surface_body_m + basis.tangent_x * x + basis.tangent_z * z;
            let dir = point.normalize_or_zero();
            let h = height_source
                .sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
                .map(|h| h as f64)
                .unwrap_or(h_center);
            let surface = dir * (body_radius_m + h);
            *cell = (surface - center_surface_body_m).dot(basis.normal);
        }
    }
    let collider = Collider::heightfield(heights, DVec3::new(2.0 * half, 1.0, 2.0 * half));
    let entity = commands
        .spawn((
            // Kinematic, not Static: it is re-posed by `sync_terrain_collider_pose`
            // when the SLF frame re-anchors, and Avian only refreshes the
            // broadphase/collider transform for moved *kinematic* bodies. Zero
            // velocity — the pose is written directly, never integrated.
            RigidBody::Kinematic,
            collider,
            Position(
                frame.rotation_body_to_frame
                    * (center_surface_body_m - frame.anchor_point_body_m),
            ),
            Rotation(frame.rotation_body_to_frame * patch_basis_rotation(&basis)),
            LinearVelocity(DVec3::ZERO),
            AngularVelocity(DVec3::ZERO),
            ground_collision_layers(),
            TerrainColliderPatch {
                body_id,
                center_dir,
                half_extent_m: half,
                resolution: config.patch_resolution,
            },
            Name::new("Local terrain heightfield collider"),
        ))
        .id();
    SpawnedTerrainPatch {
        entity,
        center_surface_body_m,
        basis,
        half_extent_m: half,
    }
}

pub fn spawn_local_craft_body(commands: &mut Commands, spawn: LocalCraftSpawn) -> Entity {
    let collider = compound_collider(&spawn.collider_primitives);
    commands
        .spawn((
            RigidBody::Dynamic,
            collider,
            Position(spawn.position_m),
            Rotation(spawn.rotation),
            LinearVelocity(spawn.linear_velocity_m_s),
            AngularVelocity(spawn.angular_velocity_rad_s),
            Mass(spawn.mass_kg.max(1.0) as f32),
            AngularInertia::new(spawn.angular_inertia_kg_m2.max(DVec3::ZERO).as_vec3()),
            NoAutoMass,
            NoAutoAngularInertia,
            ConstantLinearAcceleration(DVec3::ZERO),
            ConstantAngularAcceleration(DVec3::ZERO),
            SleepingDisabled,
            (
                LocalCraftBody {
                    craft_id: spawn.craft_id,
                },
                LocalCraftColliderPrimitives(spawn.collider_primitives),
            ),
            // Continuous Collision Detection so a fast descent stops at the
            // terrain trimesh instead of tunneling through it. Speculative
            // collision (Avian default) treats surfaces as infinite planes
            // and misses thin meshes at speed; the geometric swept sweep is
            // the documented backstop. `NonLinear` (the default) also covers
            // a tumbling craft. `Restitution(0)` keeps touchdowns from
            // bouncing. (The EVA path removes the collider after spawn, so
            // these are inert there.) Grouped in a nested bundle to keep the
            // outer tuple within Bevy's 15-element `Bundle` impl.
            (
                SweptCcd::default(),
                Restitution::new(0.0),
                Name::new("Local aggregate craft rigidbody"),
            ),
        ))
        .id()
}

pub fn compound_collider(primitives: &[LocalPrimitiveCollider]) -> Collider {
    if primitives.is_empty() {
        return Collider::cuboid(1.0, 1.0, 1.0);
    }
    let shapes = primitives
        .iter()
        .map(|primitive| {
            (
                Position(primitive.offset_m),
                Rotation(primitive.rotation),
                primitive_collider(primitive.shape),
            )
        })
        .collect();
    Collider::compound(shapes)
}

pub fn primitive_collider(shape: LocalPrimitiveShape) -> Collider {
    match shape {
        LocalPrimitiveShape::Cuboid { x, y, z } => Collider::cuboid(x, y, z),
        LocalPrimitiveShape::Cylinder { radius, height } => Collider::cylinder(radius, height),
        LocalPrimitiveShape::Cone { radius, height } => Collider::cone(radius, height),
        LocalPrimitiveShape::Sphere { radius } => Collider::sphere(radius),
        LocalPrimitiveShape::Capsule { radius, length } => Collider::capsule(radius, length),
    }
}

pub fn stable_contact_reached(
    timer_s: &mut f64,
    dt_s: f64,
    has_contact: bool,
    linear_speed_m_s: f64,
    angular_speed_rad_s: f64,
    throttle: f64,
    config: &LocalBubbleConfig,
) -> bool {
    let stable = has_contact
        && linear_speed_m_s < config.max_stable_speed_m_s
        && angular_speed_rad_s < config.max_stable_angular_speed_rad_s
        && throttle <= 1.0e-3;
    if stable {
        *timer_s += dt_s.max(0.0);
    } else {
        *timer_s = 0.0;
    }
    *timer_s >= config.stable_contact_time_s
}

pub fn craft_contacts_terrain(
    contact_graph: &ContactGraph,
    craft_entity: Entity,
    terrain_entity: Entity,
) -> bool {
    contact_graph
        .get(craft_entity, terrain_entity)
        .map(|(_, pair)| pair.is_touching())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_contact_requires_duration_speed_and_zero_throttle() {
        let config = LocalBubbleConfig {
            stable_contact_time_s: 2.0,
            ..Default::default()
        };
        let mut timer = 0.0;

        assert!(!stable_contact_reached(
            &mut timer, 1.0, true, 0.25, 0.01, 0.0, &config
        ));
        assert!(stable_contact_reached(
            &mut timer, 1.0, true, 0.25, 0.01, 0.0, &config
        ));

        assert!(!stable_contact_reached(
            &mut timer, 1.0, true, 0.75, 0.01, 0.0, &config
        ));
        assert_eq!(timer, 0.0);
        assert!(!stable_contact_reached(
            &mut timer, 1.0, true, 0.25, 0.01, 0.5, &config
        ));
        assert_eq!(timer, 0.0);
    }

    #[test]
    fn compound_builder_accepts_blueprint_primitives() {
        let collider = compound_collider(&[
            LocalPrimitiveCollider {
                offset_m: DVec3::ZERO,
                rotation: DQuat::IDENTITY,
                shape: LocalPrimitiveShape::Cylinder {
                    radius: 0.5,
                    height: 2.0,
                },
            },
            LocalPrimitiveCollider {
                offset_m: DVec3::Y,
                rotation: DQuat::IDENTITY,
                shape: LocalPrimitiveShape::Cone {
                    radius: 0.5,
                    height: 1.0,
                },
            },
        ]);
        let _ = collider;
    }

}
