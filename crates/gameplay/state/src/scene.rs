//! The shared scene vocabulary: the components and camera-level resources
//! more than one gameplay/rendering concern reads. Renderer-specific material
//! handles stay with the runtime's `rendering::types`.

use bevy::ecs::{component::Mutable, system::SystemParam};
use bevy::prelude::*;
use std::collections::HashMap;
use thalos_physics_canonical::canonical::CraftId;

/// Linear-RGB tint to use as a body's planetshine emission. Populated when
/// the body's surface info first becomes known: at bake completion for
/// terrain bodies (from `StaticSurfaceData::mean_albedo`), at spawn for gas
/// giants (from cloud albedo). Bodies without an entry contribute no
/// planetshine to their moons.
#[derive(Resource, Default)]
pub struct PlanetshineTints {
    pub by_body: HashMap<usize, [f32; 3]>,
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
#[derive(Resource, Reflect, Default, Clone, Copy, Debug)]
#[reflect(Resource)]
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

#[derive(Component)]
pub struct CelestialBody {
    pub body_id: usize,
    pub is_star: bool,
    pub render_radius: f32,
    /// True physical radius in metres (not clamped like render_radius).
    pub radius_m: f64,
}

/// Marks a body whose baked surface is tidally locked to its parent. Each
/// frame the shared surface orientation keeps the baked near-side (local +Z,
/// where the mare/tidal asymmetry lives) facing the parent body; impostors use
/// the world→body form and real-space terrain uses the inverse body→world form.
/// The authored-data rule that decides which bodies carry this tag is
/// [`crate::surface_frame::authored_lock_parent`].
#[derive(Component)]
pub struct TidallyLocked {
    pub parent_id: usize,
}

#[derive(Component)]
pub struct ShipMarker;

/// Root of one runtime projection of a canonical vessel.
///
/// A ship uses its rendered part-tree root. EVA uses its local controller
/// body. Every root carries [`CraftIdentity`] and the per-craft runtime state
/// components; systems that need the selected vessel resolve it through
/// [`ActiveCraft`] instead of assuming one root exists.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct CraftRoot;

/// Stable link from a runtime craft root, part, or map marker to canonical
/// fleet state.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct CraftIdentity(pub CraftId);

/// Ownership of a flight part. Aggregations must filter by this id so a
/// detached stage cannot contribute fuel, inertia, engines, or staging state
/// to the selected craft.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct CraftPart(pub CraftId);

/// Transitional marker for the selected rendered ship root.
///
/// New code must use [`ActiveCraft`]. This marker remains only while the
/// existing camera/view systems are migrated; it no longer means that a craft
/// exists, because every vessel is represented by [`CraftRoot`].
#[derive(Component)]
pub struct PlayerShip;

/// The craft the player is currently controlling — the **N-craft accessor seam**.
///
/// The runtime resolves the canonical active [`CraftId`] to its current
/// [`CraftRoot`]. `None` is expected during respawn/relaunch and before the EVA
/// local body or rendered ship root has materialized. New per-craft state is a
/// component on this entity, never a global resource.
///
/// **Sole writer:** the runtime's `track_active_craft`.
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveCraft(pub Option<Entity>);

/// Read-only access to component `T` on the selected craft root.
///
/// Keeping this lookup in the blackboard makes the absence semantics and the
/// `CraftRoot` filter identical across runtime, HUD, and map crates.
#[derive(SystemParam)]
pub struct ActiveCraftRef<'w, 's, T: Component> {
    active: Res<'w, ActiveCraft>,
    components: Query<'w, 's, &'static T, With<CraftRoot>>,
}

impl<'w, 's, T: Component> ActiveCraftRef<'w, 's, T> {
    pub fn get(&self) -> Option<&T> {
        self.components.get(self.active.0?).ok()
    }

    pub fn entity(&self) -> Option<Entity> {
        self.active.0
    }
}

/// Mutable access to component `T` on the selected craft root.
#[derive(SystemParam)]
pub struct ActiveCraftMut<'w, 's, T: Component<Mutability = Mutable>> {
    active: Res<'w, ActiveCraft>,
    components: Query<'w, 's, &'static mut T, With<CraftRoot>>,
}

impl<'w, 's, T: Component<Mutability = Mutable>> ActiveCraftMut<'w, 's, T> {
    pub fn get(&self) -> Option<&T> {
        self.components.get(self.active.0?).ok()
    }

    pub fn get_mut(&mut self) -> Option<Mut<'_, T>> {
        self.components.get_mut(self.active.0?).ok()
    }

    pub fn entity(&self) -> Option<Entity> {
        self.active.0
    }
}

/// A procedural interstage/fairing shroud hull. Present in both the editor
/// world (interactive: hover transparency, pick-through) and the flight
/// craft (opaque hull); the runtime's `shrouds` module owns the reconcile
/// pass that derives them.
#[derive(Component, Debug, Clone, Copy)]
pub struct ShroudBody;

/// Real-space (BigSpace) instance of a celestial body.
#[derive(Component)]
pub struct RealSpaceBody {
    pub body_id: usize,
}

/// Marker on **every entity the editor owns**: the parts being built, the
/// editor's `Ship` entity. (Mesh children are reachable through their part
/// parent and carry the visual markers below instead.)
///
/// This is the partition between the editor's build world and any other
/// ship assembled from the same part components in the same `World` — the
/// game's flight ship in particular. Editor-core systems filter
/// `With<EditorPart>`; game systems that aggregate over part components
/// (fuel, staging, gear, ship visuals) filter `Without<EditorPart>`.
#[derive(Component, Debug, Clone, Copy)]
pub struct EditorPart;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Component, Debug, PartialEq, Eq)]
    struct RootState(u32);

    fn increment_active(mut state: ActiveCraftMut<RootState>) {
        if let Some(mut state) = state.get_mut() {
            state.0 += 1;
        }
    }

    #[test]
    fn active_component_access_isolates_two_roots() {
        let mut app = App::new();
        app.init_resource::<ActiveCraft>()
            .add_systems(Update, increment_active);
        let first = app.world_mut().spawn((CraftRoot, RootState(10))).id();
        let second = app.world_mut().spawn((CraftRoot, RootState(20))).id();

        app.world_mut().resource_mut::<ActiveCraft>().0 = Some(second);
        app.update();

        assert_eq!(app.world().get::<RootState>(first), Some(&RootState(10)));
        assert_eq!(app.world().get::<RootState>(second), Some(&RootState(21)));
    }

    #[test]
    fn missing_active_root_does_not_fall_back_to_another_root() {
        let mut app = App::new();
        app.init_resource::<ActiveCraft>()
            .add_systems(Update, increment_active);
        let root = app.world_mut().spawn((CraftRoot, RootState(10))).id();

        app.update();

        assert_eq!(app.world().get::<RootState>(root), Some(&RootState(10)));
    }
}
