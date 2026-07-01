//! Editor-core resources, components, and markers.
//!
//! Everything here is UI-framework-agnostic: a front-end (the standalone
//! egui binary, the in-game Bevy-UI editor) drives the editor by writing
//! [`EditorState`] fields and reading them back — the core systems in this
//! module's siblings do the actual work.

use bevy::prelude::*;

use thalos_shipyard::material::ShipPartMaterial;
use thalos_shipyard::{CatalogId, NodeId, PartParams};

use super::files::SavedShip;

/// Cursor-distance threshold (in pixels) separating a click from a
/// drag. Used by deselect-on-empty-click and by orbit-while-pending so
/// the click/drag boundary is consistent — strict-less is "click",
/// `>=` is "drag".
pub const CLICK_THRESHOLD_PX: f32 = 4.0;

/// Radial segment count for cylindrical/frustum part meshes. Bevy's
/// default is 32, which leaves a visibly faceted silhouette at editor
/// zoom levels. Cost is negligible at the part counts we render.
pub const PART_RESOLUTION: u32 = 128;

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

/// Front-end-supplied pointer gate. `pointer_busy` must be true whenever
/// the cursor is over (or captured by) the front-end's 2D UI, so scene
/// interactions (picking, deselect, placement preview) stand down.
///
/// **Sole writer:** the active front-end (egui binary:
/// `gate_shipyard_input_sources`; game: `shipyard_editor::sync_editor_ui_gate`).
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct EditorUiGate {
    pub pointer_busy: bool,
}

/// Marker for the camera the editor view looks through. Core systems that
/// need a viewpoint (tank-resize screen-space drag, arrow billboard
/// placement) query this instead of assuming a single camera exists —
/// the game world has several.
#[derive(Component)]
pub struct EditorViewCamera;

/// A part the user has armed in the palette but not yet placed. Held by
/// [`EditorState::pending`] until the user clicks an attach node or
/// drops it on an empty canvas as the new root.
#[derive(Clone, Debug)]
pub struct PendingPart {
    pub catalog_id: CatalogId,
    pub params: PartParams,
}

/// Editor command/state hub. Front-ends write the command fields
/// (`save_requested`, `load_target`, `place_at`, …); `process_commands`
/// consumes them each frame. Selection and status flow the other way.
#[derive(Resource, Default)]
pub struct EditorState {
    pub ship_root: Option<Entity>,
    pub ship_entity: Option<Entity>,
    pub ship_name: String,
    pub selected: Option<Entity>,
    pub pending: Option<PendingPart>,
    pub place_at: Option<(Entity, String)>,
    /// A pending surface-mount placement: `(host part, world hit point,
    /// mount kind)`. Consumed by `process_commands` which derives the
    /// mount-kind-specific `(station, angle)` pair.
    pub place_surface_at: Option<(Entity, Vec3, thalos_shipyard::SurfaceMountKind)>,
    pub delete_selected: bool,
    pub set_as_root: bool,
    pub save_requested: bool,
    pub load_target: Option<String>,
    pub delete_file: Option<String>,
    pub refresh_list: bool,
    pub ship_list: Vec<SavedShip>,
    pub status: String,
}

/// KSP-style editor symmetry mode. When `mirror` is on, placing a footprint
/// part stamps a linked mirror counterpart across the host X = 0 plane.
#[derive(Resource, Default)]
pub struct SymmetryMode {
    pub mirror: bool,
}

/// Magnetic angle snap for body-skin (cylinder) mounts. On by default — the
/// mount azimuth rounds to [`BODY_SKIN_SNAP_STEP`](super::placement::BODY_SKIN_SNAP_STEP)
/// increments so gear/wings land dead-on the belly / sides as the cursor
/// sweeps around the fuselage.
#[derive(Resource)]
pub struct PlacementSnap {
    pub enabled: bool,
}

impl Default for PlacementSnap {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Live placement-preview state. Holds the one reused ghost entity plus the
/// signature of the mesh currently on it, so the (small) ghost mesh is rebuilt
/// only when the host / snapped angle / part params actually change, not every
/// frame the cursor moves.
#[derive(Resource, Default)]
pub struct PlacementPreview {
    pub entity: Option<Entity>,
    pub sig: Option<PreviewSig>,
}

/// What the preview ghost mesh depends on. Station is excluded — it only moves
/// the ghost along the body axis (the transform), it doesn't reshape the mesh.
#[derive(Clone, PartialEq)]
pub struct PreviewSig {
    pub host: Entity,
    pub angle: f32,
    pub parent_radius: f32,
    pub params: PartParams,
}

/// Monotonic source of [`thalos_shipyard::SymmetryGroup`] ids for newly stamped groups.
#[derive(Resource, Default)]
pub struct NextSymmetryId(pub u32);

impl NextSymmetryId {
    /// Allocate a fresh group id. (Named distinctly from `Iterator::next`.)
    pub fn allocate(&mut self) -> u32 {
        let id = self.0;
        self.0 += 1;
        id
    }
}

/// Vertical (rocket / VAB) vs horizontal (aircraft / SPH) build layout.
/// Purely a display + interaction frame: parts are always authored along
/// the body +Y axis; horizontal lays the whole assembly down so the body
/// axis runs fore/aft and the dorsal (+Z) side faces up, like KSP's
/// Spaceplane Hangar. The rotation is applied rigidly to every part in
/// `update_part_transforms`; placement / resize convert pointer hits back
/// through its inverse so building stays correct in either layout.
#[derive(Resource, Default)]
pub struct BuildOrientation {
    pub horizontal: bool,
}

impl BuildOrientation {
    pub fn rotation(&self) -> Quat {
        if self.horizontal {
            // Nose (+Y) → −Z (forward), dorsal (+Z) → +Y (up), span (X) stays
            // level — the craft lies down facing away from the camera.
            Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)
        } else {
            Quat::IDENTITY
        }
    }
}

/// Tracks the cursor at mouse-down when the press landed on empty space,
/// so a release at near-the-same position clears the selection but a
/// press→drag→release (camera orbit) does not.
#[derive(Resource, Default)]
pub struct DeselectTracker {
    pub press_cursor: Option<Vec2>,
}

/// Shared materials and meshes for editor visuals.
#[derive(Resource)]
pub struct EditorAssets {
    pub part_material: Handle<StandardMaterial>,
    /// Matte dark finish for landing gear bodies — distinct from the stainless
    /// hull. The selection-highlight system falls back to this (not
    /// `part_material`) for gear visuals so wheels never read as steel.
    pub gear_material: Handle<StandardMaterial>,
    pub hover_material: Handle<StandardMaterial>,
    pub selected_material: Handle<StandardMaterial>,
    pub pending_node_material: Handle<StandardMaterial>,
    pub node_mesh: Handle<Mesh>,
    pub resize_arrow_mesh: Handle<Mesh>,
    pub resize_arrow_material: Handle<StandardMaterial>,
    /// Translucent green ghost for the live placement preview.
    pub preview_material: Handle<StandardMaterial>,
    /// Translucent cyan x-ray ghost for the gear stow-bay box. High depth bias
    /// so the reserved volume reads *through* the opaque fuselage skin.
    pub gear_bay_material: Handle<StandardMaterial>,
}

#[derive(Component)]
pub struct PartVisual;

/// Marker on a wing's mesh child. Distinct from [`PartVisual`] so the
/// body-of-revolution rebuild (`rebuild_visuals`) never despawns wing
/// geometry — `rebuild_wing_visuals` owns it.
#[derive(Component)]
pub struct WingVisual;

#[derive(Component)]
pub struct NacelleVisual;

/// Marker on a gearbox's mesh child. Distinct from [`PartVisual`] so
/// `rebuild_visuals` (the body-of-revolution rebuild) never touches gear
/// geometry — `rebuild_gear_visuals` owns it, like wings/nacelles.
#[derive(Component)]
pub struct GearVisual;

/// Marker on a gearbox's **stow-bay** ghost child — the x-ray box showing the
/// volume inside the fuselage that will house the gear when retracted. Rendered
/// translucent and non-pickable; `rebuild_gear_visuals` owns it alongside
/// [`GearVisual`].
#[derive(Component)]
pub struct GearBayVisual;

/// Marker on the live placement-preview ghost — the translucent silhouette of
/// the pending footprint part following the cursor across a host surface. One
/// reused entity, tracked by [`PlacementPreview`].
#[derive(Component)]
pub struct PreviewGhost;

/// Back-pointer from a part's mesh child to the owning part entity.
#[derive(Component)]
pub struct PartBody(pub Entity);

/// Per-part `ShipPartMaterial` asset handle, cached on the part entity
/// so it survives child rebuilds (e.g. resizing a tank despawns and
/// respawns the body, but the material asset — and its tint state — is
/// stable). Used by any part that carries [`thalos_shipyard::PartMaterial`] — tanks and
/// decouplers today.
#[derive(Component, Clone)]
pub struct PartShaderHandle(pub Handle<ShipPartMaterial>);

#[derive(Component)]
pub struct AttachNodePin {
    pub part: Entity,
    pub node_id: NodeId,
}

#[derive(Component)]
pub struct TankResizeArrow {
    pub tank: Entity,
}

#[derive(Resource, Default)]
pub struct TankResizeDrag {
    pub active: Option<TankDragState>,
}

pub struct TankDragState {
    pub tank: Entity,
    pub start_length: f32,
    pub start_cursor: Vec2,
    pub screen_axis: Vec2,
    pub world_per_pixel: f32,
}
