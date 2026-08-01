//! The shared scene vocabulary: the components and camera-level resources
//! more than one gameplay/rendering concern reads. Renderer-specific material
//! handles stay with the runtime's `rendering::types`.

use bevy::prelude::*;
use std::collections::HashMap;

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

/// Root of the player's ship in 3D space. Its children are the ship parts
/// rendered at 1:1 meter scale in the entity's local frame; the entity's
/// `Transform::scale` compensates so the ship renders at real size in the
/// solar-system-wide render-units coordinate space (see
/// [`WorldScale`](crate::coords::WorldScale)).
///
/// Present in both views. In map view it's hidden (the flat [`ShipMarker`]
/// billboard stands in for it); in ship view it becomes visible and the
/// camera orbits it.
#[derive(Component)]
pub struct PlayerShip;

/// The craft the player is currently controlling — the **N-craft accessor seam**.
///
/// Today the game has exactly one [`PlayerShip`]; this resource simply names it
/// by entity. Its purpose is architectural: it is the single sanctioned answer to
/// "which craft is active", so consumers read `active.0` (an `Option<Entity>`,
/// `None` during the respawn/relaunch rebuild window) instead of assuming exactly
/// one craft via `q.single()` — a call that *panics* the moment a second craft
/// entity exists. When N craft land, this picks the active one and nothing else
/// changes. New per-craft state should be a **component on this entity**, not a
/// new global resource (see `docs/roadmap/architecture_cleanup.md` §E).
///
/// **Sole writer:** the runtime's `track_active_craft`.
#[derive(Resource, Default)]
pub struct ActiveCraft(pub Option<Entity>);

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
