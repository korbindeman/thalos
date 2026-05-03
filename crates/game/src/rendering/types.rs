//! Resources and components shared across the rendering submodules.
//!
//! Per-submodule resources (e.g. trail caches, click state) live in
//! their respective modules; this file holds only types touched by more
//! than one rendering concern.

use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::Task;
use thalos_physics::{
    body_state_provider::BodyStateProvider,
    simulation::Simulation,
    types::{BodyStates, SolarSystemDefinition},
};
use thalos_planet_rendering::{
    CLOUD_BAND_COUNT, GasGiantMaterial, PlanetHaloMaterial, PlanetMaterial, RingMaterial,
    SolidPlanetMaterial,
};
use thalos_terrain_gen::BodyData;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Central simulation state.
#[derive(Resource)]
pub struct SimulationState {
    pub simulation: Simulation,
    pub system: SolarSystemDefinition,
    pub ephemeris: Arc<dyn BodyStateProvider>,
}

/// Per-frame cache of all body states at the current sim time. Populated once
/// per frame by `cache_body_states` and read by multiple rendering systems to
/// avoid redundant ephemeris queries.
#[derive(Resource, Default)]
pub struct FrameBodyStates {
    pub states: Option<BodyStates>,
    pub time: f64,
}

/// Camera exposure model. Acts as the semantic "sensor" of the game camera:
/// it owns how focus distance maps to display brightness and how much grain
/// is added in consequence. Every system that cares about "how much flux
/// does the shader see" or "how much noise should the post stack add" reads
/// this resource rather than recomputing from focus distance.
///
/// Linear-in-distance compensation: outer-system focus pulls distant bodies
/// out of black without erasing the distance cue. Concretely, the display
/// flux at the focus body scales as `LIGHT_AT_1AU * (1 AU / focus_d)`,
/// so a body at 40 AU remains roughly 40x dimmer than the same body at
/// 1 AU even when focused.
///
/// The gain applied to each body's raw inverse-square flux in the impostor
/// shader is `exposure.gain = focus_d / 1 AU`. Combined with the raw
/// `(AU/body_d)^2` falloff baked into `update_planet_light_dirs`, this
/// yields the focus-relative display flux above.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct CameraExposure {
    /// Camera focus body's distance from the star, in meters.
    pub focus_dist_m: f64,
    /// Multiplicative gain applied to per-body raw inverse-square flux.
    pub gain: f32,
    /// Log2(gain). Positive = we're pushing dark outer-system scenes;
    /// negative = we're pulling down bright inner-system scenes. Drives
    /// film grain strength (and, later, lens flare intensity).
    pub ev: f32,
}

/// Shared meshes reused across every procedural planet, cached once at
/// startup so `finalize_planet_generation` doesn't need to re-add them.
#[derive(Resource)]
pub(super) struct SharedPlanetMeshes {
    pub(super) billboard: Handle<Mesh>,
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct CelestialBody {
    pub body_id: usize,
    pub is_star: bool,
    pub render_radius: f32,
    /// True physical radius in metres (not clamped like render_radius).
    pub radius_m: f64,
}

/// Marks a body whose baked surface is tidally locked to its parent. Each
/// frame the orientation uniform is recomputed so the baked near-side (local
/// +Z, where the mare/tidal asymmetry lives) keeps facing the parent body.
#[derive(Component)]
pub(super) struct TidallyLocked {
    pub(super) parent_id: usize,
}

#[derive(Component)]
pub struct ShipMarker;

/// Root of the player's ship in 3D space. Its children are the ship parts
/// rendered at 1:1 meter scale in the entity's local frame; the entity's
/// `Transform::scale` compensates so the ship renders at real size in the
/// solar-system-wide render-units coordinate space (see [`WorldScale`]).
///
/// Present in both views. In map view it's hidden (the flat `ShipMarker`
/// billboard stands in for it); in ship view it becomes visible and the
/// camera orbits it.
#[derive(Component)]
pub struct PlayerShip;

/// Marker for the directional light that simulates sunlight toward the focus body.
#[derive(Component)]
pub(super) struct SunLight;

/// Marker for the map-view mesh child of a celestial body. Inherits the
/// parent's transform, which is updated at [`MAP_SCALE`](crate::coords::MAP_SCALE)
/// each frame.
#[derive(Component)]
pub(super) struct BodyMesh;

/// Marker for the ship-view mesh child of a celestial body. Carries an
/// absolute world transform (its local translation compensates for the
/// parent's map-scale translation) so the body renders at
/// [`SHIP_SCALE`](crate::coords::SHIP_SCALE) when the ship camera draws
/// it. Updated each frame by `update_ship_body_meshes`.
#[derive(Component)]
pub(super) struct ShipBodyMesh;

/// Marker for the flat circle icon child of a celestial body.
#[derive(Component)]
pub(super) struct BodyIcon;

/// Material handles a procedural body owns: a depth-writing body pass
/// plus a depth-testing/no-depth-write halo pass, each baked at
/// [`MAP_SCALE`](crate::coords::MAP_SCALE) for [`BodyMesh`] and
/// [`SHIP_SCALE`](crate::coords::SHIP_SCALE) for [`ShipBodyMesh`].
///
/// Heavy assets (cubemaps, SSBOs) are shared via `Handle<…>`; only the
/// scale-dependent uniforms (radius, atmosphere `rim_shape`, occluder
/// positions in `params.scene`) differ between the two instances.
#[derive(Component)]
pub struct PlanetMaterials {
    pub map: Handle<PlanetMaterial>,
    pub ship: Handle<PlanetMaterial>,
    pub map_halo: Handle<PlanetHaloMaterial>,
    pub ship_halo: Handle<PlanetHaloMaterial>,
}

/// Same idea as [`PlanetMaterials`] but for [`SolidPlanetMaterial`] —
/// the placeholder used by bodies that don't have a terrain pipeline.
#[derive(Component)]
pub struct SolidPlanetMaterials {
    pub map: Handle<SolidPlanetMaterial>,
    pub ship: Handle<SolidPlanetMaterial>,
}

/// Same idea as [`PlanetMaterials`] but for [`GasGiantMaterial`].
#[derive(Component)]
pub struct GasGiantMaterials {
    pub map: Handle<GasGiantMaterial>,
    pub ship: Handle<GasGiantMaterial>,
}

/// Per-ring-entity marker for the map-layer ring child. Carries its own
/// [`RingMaterial`] handle so per-frame updates can find it.
#[derive(Component)]
pub(super) struct MapRingMaterial(pub(super) Handle<RingMaterial>);

/// Per-ring-entity marker for the ship-layer ring child. Mirror of
/// [`MapRingMaterial`] for the ship-scale instance.
#[derive(Component)]
pub(super) struct ShipRingMaterial(pub(super) Handle<RingMaterial>);

/// In-flight terrain generation task for a procedural body.
///
/// While this component is attached to the parent `CelestialBody` entity, the
/// body renders with a plain placeholder sphere. Once the background task
/// completes, `finalize_planet_generation` bakes the result into GPU textures,
/// swaps the child mesh to the impostor billboard with a `PlanetMaterial`, and
/// removes this component.
#[derive(Component)]
pub(super) struct PendingPlanetGeneration {
    pub(super) task: Task<BodyData>,
    pub(super) body_id: usize,
    pub(super) render_radius: f32,
    /// Map-view child holding the placeholder mesh; gets swapped to the
    /// impostor billboard when the task finishes.
    pub(super) mesh_entity: Entity,
    /// Ship-view child holding the placeholder mesh; mirrored swap.
    pub(super) ship_mesh_entity: Entity,
}

/// Per-body cloud-rotation state. Advanced by `update_cloud_bands` each
/// frame and uploaded into the material's `cloud_band_phases_*` fields.
/// Attached to any body whose `terrestrial_atmosphere.clouds` is `Some`.
#[derive(Component, Default, Clone)]
pub struct CloudBandState {
    pub phases: [f64; CLOUD_BAND_COUNT],
}
