//! The maneuver-plan data model — moved wholesale to
//! `thalos_game_state::maneuver_plan` (Phase 5b); re-exported for path
//! stability (`crate::maneuver::*` keeps working through `mod.rs`).

pub use thalos_game_state::maneuver_plan::*;

// --- View-layer internals (markers, gizmo constants) stay with the
// runtime's maneuver interaction/view systems; only the plan data model
// moved to the blackboard. ---
use bevy::prelude::*;

/// Flat circle marker for an unselected maneuver node.
#[derive(Component)]
pub struct NodeMarkerDisc {
    pub node_id: NodeId,
}

/// Flat circle for the snap indicator when placing a node (N key).
#[derive(Component)]
pub struct SnapIndicator;

/// Invisible picking hitbox around each arrow.
#[derive(Component)]
pub struct ArrowHitbox;

/// Draggable sphere at the node center for sliding along the trajectory.
#[derive(Component)]
pub struct NodeSlideSphere;

pub const SELECT_THRESHOLD_PX: f32 = 20.0;

/// Live world position and orbital frame for a node currently being slid.
///
/// During a slide, [`ManeuverPlan::last_slide_apply_secs`] throttles flight-
/// plan rebuilds to ~10 Hz, so the cached prediction can be up to 100 ms
/// behind `node.time` for the dragged node. Sampling that prediction at the
/// fresh `node.time` may land on the wrong leg and snap the visual marker off
/// the branch the user is dragging along.
///
/// [`super::interaction::maneuver_input`] writes this resource directly from
/// the chosen [`super::helpers::ClosestTrailPoint`], bypassing the stale
/// prediction; [`super::update_selected_node_view`] then prefers it whenever
/// [`InteractionMode::SlidingNode`] is active.
#[derive(Resource, Default)]
pub struct SlidePreview {
    pub world_pos: Option<Vec3>,
    pub frame: Option<Mat3>,
}

#[derive(Component)]
pub struct ArrowHandle {
    pub axis: usize,
    pub positive: bool,
}

pub const BASE_ARROW_LEN: f32 = 0.04;

/// Hitbox capsule radius — generous for easy grabbing.
pub const HITBOX_CAPSULE_RADIUS: f32 = 0.008;

/// Slide sphere radius.
pub const SLIDE_SPHERE_RADIUS: f32 = 0.012;

/// Cached world position and orbital frame for the selected node.
/// Recomputed each frame from the prediction.
#[derive(Resource, Default)]
pub struct SelectedNodeView {
    pub world_pos: Option<Vec3>,
    pub frame: Option<Mat3>,
}

/// Arrow colors: [prograde green, normal magenta, radial cyan].
pub const ARROW_COLORS: [Color; 3] = [
    Color::srgb(0.0, 1.0, 0.0),
    Color::srgb(0.7, 0.0, 1.0),
    Color::srgb(0.0, 1.0, 1.0),
];

pub const ARROW_STRETCH: f32 = 0.0075;

#[derive(Component)]
pub struct ArrowCone;

#[derive(Component)]
pub struct ArrowShaft;

/// Animated stretch state per arrow: [axis][positive=0/negative=1].
#[derive(Resource, Default)]
pub struct ArrowStretchState {
    pub current: [[f32; 2]; 3],
}

/// Per-arrow material handle and base color for dynamic opacity adjustment.
#[derive(Component)]
pub struct ArrowVisual {
    pub material: Handle<StandardMaterial>,
    pub base_color: LinearRgba,
}

pub const CONE_HEIGHT: f32 = 0.008;

pub const CONE_RADIUS: f32 = 0.005;

pub const HOVER_BRIGHTNESS: f32 = 1.8;

/// Arrow dimensions in screen-stable units (scaled by camera distance each frame).
pub const SHAFT_RADIUS: f32 = 0.002;

pub const STRETCH_LERP_SPEED: f32 = 12.0;

/// Material handle for the slide sphere (for hover highlight).
#[derive(Component)]
pub struct SphereVisual {
    pub material: Handle<StandardMaterial>,
}
