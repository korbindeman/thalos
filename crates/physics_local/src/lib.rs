//! Bevy/Avian boundary for near-surface local rigidbody physics.
//!
//! This crate deliberately contains the Avian dependency so `thalos_physics_canonical`
//! remains pure Rust and rails/canonical code does not learn Avian APIs.

use std::{collections::HashMap, sync::Arc};

use avian3d::prelude::*;
use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use thalos_body_render::{
    GpuAtlasMirrorHandle, GpuAtlasMirrorHeightSource, HeightSource, TerrainPatchBasis,
    TerrainPatchConfig, TerrainPatchMesh, build_rendered_terrain_patch_from_source,
};
use thalos_physics_canonical::canonical::CraftId;
use thalos_terrain::PlanetSurface;
use thalos_world::BodyId;

pub mod avian {
    pub use avian3d::prelude::{
        AngularInertia, AngularVelocity, CenterOfMass, CoefficientCombine, Collider,
        ConstantAngularAcceleration, ConstantForce, ConstantLinearAcceleration, ConstantTorque,
        ContactGraph, CustomPositionIntegration, Friction, LinearVelocity, LockedAxes, Mass,
        NoAutoAngularInertia, NoAutoCenterOfMass, NoAutoMass, Physics, PhysicsDebugPlugin,
        PhysicsGizmos, PhysicsSchedule, PhysicsTime, Position, RayHitData, Restitution, RigidBody,
        Rotation, SleepingDisabled, SpatialQuery, SpatialQueryFilter, SweptCcd,
    };
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
/// The Avian body lives in **body-centered inertial** coordinates — the
/// origin tracks the dominant body's centre but the axes do not rotate.
/// Gravity in `apply_local_forces` is the textbook two-body `−μr/r³`;
/// no fictitious forces. The terrain collider is `Kinematic`, centered on
/// the local patch with body-fixed vertex offsets and `Rotation =
/// body.orientation`, so `Position + Rotation * local_vertex` evaluates to
/// the rendered body-centered inertial surface as the body spins.
#[derive(Resource, Debug, Clone)]
pub struct LocalBubble {
    pub id: u64,
    pub body_id: BodyId,
    pub craft_entity: Entity,
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
    pub mesh: TerrainPatchMesh,
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

#[allow(clippy::too_many_arguments)]
pub fn spawn_terrain_collider_patch(
    commands: &mut Commands,
    body_id: BodyId,
    height_source: &dyn HeightSource,
    body_radius_m: f64,
    center_dir_body: DVec3,
    config: &LocalBubbleConfig,
) -> SpawnedTerrainPatch {
    // Prefer a collider built from the source's native tile geometry — the GPU
    // atlas tiles the renderer meshes from — so it lines up with the drawn
    // surface by construction. Sources with no resident tile geometry (CPU
    // pipeline, flat, baked cubemap) return `None`, as does the GPU mirror
    // before a tile is resident, and we fall back to the coarser tangent-grid
    // resample. See `docs/surface.md`.
    let patch = height_source
        .build_collider_patch(center_dir_body.as_vec3(), config.patch_resolution)
        .unwrap_or_else(|| {
            let basis = TerrainPatchBasis::from_normal(center_dir_body);
            build_rendered_terrain_patch_from_source(
                height_source,
                body_radius_m,
                center_dir_body,
                basis,
                terrain_patch_config(config),
            )
        });
    // Keep the trimesh near its own origin. The source vertices are absolute
    // body-fixed positions at planet radius; feeding those directly to the
    // narrow phase makes every contact solve against million-metre local
    // coordinates. Instead, put the kinematic body at the patch centre and
    // store vertex offsets in body-fixed axes.
    //
    // The craft (ship) Avian body lives in the **body-fixed (rotating) frame**,
    // so the collider is held *static* in that frame: `Position =
    // center_surface_body_m`, identity rotation, zero velocity. Each local
    // offset then lands exactly on the rotating surface with no co-rotation
    // speed, which is what keeps ground contact stable.
    // `sync_terrain_collider_pose` maintains this each frame (cheap; only the
    // patch-recenter changes it).
    let local_vertices = terrain_patch_local_vertices(&patch);
    let collider = Collider::trimesh(local_vertices, patch.indices.clone());
    let entity = commands
        .spawn((
            RigidBody::Kinematic,
            collider,
            Position(patch.center_surface_body_m),
            Rotation(DQuat::IDENTITY),
            LinearVelocity(DVec3::ZERO),
            AngularVelocity(DVec3::ZERO),
            TerrainColliderPatch {
                body_id,
                center_dir: center_dir_body.normalize(),
                half_extent_m: config.patch_half_extent_m,
                resolution: config.patch_resolution,
            },
            Name::new("Local terrain collider patch"),
        ))
        .id();
    SpawnedTerrainPatch {
        entity,
        mesh: patch,
    }
}

fn terrain_patch_local_vertices(patch: &TerrainPatchMesh) -> Vec<DVec3> {
    patch
        .vertices_body_m
        .iter()
        .map(|vertex| *vertex - patch.center_surface_body_m)
        .collect()
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

    #[test]
    fn terrain_patch_vertices_are_local_to_patch_origin() {
        let patch = TerrainPatchMesh {
            vertices_body_m: vec![
                DVec3::new(10.0, 1000.0, -2.0),
                DVec3::new(11.0, 1001.0, -4.0),
            ],
            indices: vec![[0, 1, 1]],
            center_surface_body_m: DVec3::new(10.0, 1000.0, -2.0),
            basis: TerrainPatchBasis::from_normal(DVec3::Y),
            half_extent_m: 1.0,
        };

        let local = terrain_patch_local_vertices(&patch);

        assert_eq!(local[0], DVec3::ZERO);
        assert_eq!(local[1], DVec3::new(1.0, 1.0, -2.0));
    }

    #[test]
    fn terrain_patch_pose_tracks_rotating_surface_velocity() {
        let center_body = DVec3::new(0.0, 1000.0, 0.0);
        let orientation = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let angular_velocity = DVec3::Z * 0.1;

        let (position, velocity) = terrain_patch_pose(center_body, orientation, angular_velocity);

        assert!((position - DVec3::new(-1000.0, 0.0, 0.0)).length() < 1.0e-9);
        assert!((velocity - angular_velocity.cross(position)).length() < 1.0e-9);
    }
}
