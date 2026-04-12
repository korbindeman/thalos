use bevy::math::DVec3;
use bevy::prelude::*;

/// Unique identifier for a game-side maneuver node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// Game-side representation of a maneuver node (owned by the UI, synced to physics).
#[derive(Clone, Debug)]
pub struct GameNode {
    pub id: NodeId,
    /// Simulation time (seconds) of the burn.
    pub time: f64,
    /// Delta-v in local orbital frame: [prograde, normal, radial] m/s.
    pub delta_v: DVec3,
    /// Body used as the local reference frame (dominant body at placement time).
    pub reference_body: usize,
}

/// UI-side maneuver plan. Synced to `ManeuverSequence` in physics when dirty.
#[derive(Resource, Default)]
pub struct ManeuverPlan {
    pub nodes: Vec<GameNode>,
    pub dirty: bool,
    next_id: u64,
}

impl ManeuverPlan {
    pub fn next_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }
}

/// Cached orbital-frame delta-v for the currently selected node.
#[derive(Resource, Default)]
pub struct NodeDeltaV {
    pub prograde: f64,
    pub normal: f64,
    pub radial: f64,
}

/// Currently selected maneuver node.
#[derive(Resource, Default)]
pub struct SelectedNode {
    pub id: Option<NodeId>,
}

/// Cached world position and orbital frame for the selected node.
/// Recomputed each frame from the prediction.
#[derive(Resource, Default)]
pub(super) struct SelectedNodeView {
    pub world_pos: Option<Vec3>,
    pub frame: Option<Mat3>,
}

/// Mutually exclusive interaction modes for the maneuver system.
///
/// Only one mode can be active at a time. Camera rotation is blocked whenever
/// the mode is not `Idle`.
#[derive(Resource, Default)]
pub enum InteractionMode {
    /// No maneuver interaction in progress.
    #[default]
    Idle,
    /// Placing a new node: cursor snaps to the closest trajectory sample.
    PlacingNode {
        snap_time: Option<f64>,
        snap_world_pos: Option<Vec3>,
        snap_dominant_body: Option<usize>,
    },
    /// Dragging an arrow handle to adjust delta-v.
    DraggingArrow {
        /// Which axis: 0=prograde, 1=normal, 2=radial.
        axis: usize,
        /// Which polarity (true=positive, false=negative).
        positive: bool,
        /// Screen-space direction of the axis (for projecting mouse delta).
        axis_screen_dir: Vec2,
        /// Screen position at drag start.
        drag_origin: Vec2,
        /// Sign of current drag rate (+1 / -1 / 0).
        rate_sign: f32,
    },
    /// Dragging the center sphere to slide a node along the trajectory.
    SlidingNode,
}

/// Events for maneuver node operations.
#[derive(Clone)]
pub enum ManeuverEvent {
    PlaceNode {
        trail_time: f64,
        reference_body: usize,
    },
    AdjustNode {
        id: NodeId,
        delta_v: DVec3,
    },
    SlideNode {
        id: NodeId,
        new_time: f64,
    },
    DeleteNode {
        id: NodeId,
    },
}

impl bevy::ecs::message::Message for ManeuverEvent {}


// ---------------------------------------------------------------------------
// Arrow components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub(super) struct ArrowHandle {
    pub axis: usize,
    pub positive: bool,
}

#[derive(Component)]
pub(super) struct ArrowShaft;

#[derive(Component)]
pub(super) struct ArrowCone;

/// Invisible picking hitbox around each arrow.
#[derive(Component)]
pub(super) struct ArrowHitbox;

/// Per-arrow material handle and base color for dynamic opacity adjustment.
#[derive(Component)]
pub(super) struct ArrowVisual {
    pub material: Handle<StandardMaterial>,
    pub base_color: LinearRgba,
}

/// Material handle for the slide sphere (for hover highlight).
#[derive(Component)]
pub(super) struct SphereVisual {
    pub material: Handle<StandardMaterial>,
}

/// Animated stretch state per arrow: [axis][positive=0/negative=1].
#[derive(Resource, Default)]
pub(super) struct ArrowStretchState {
    pub current: [[f32; 2]; 3],
}

/// Draggable sphere at the node center for sliding along the trajectory.
#[derive(Component)]
pub(super) struct NodeSlideSphere;

/// Flat circle marker for an unselected maneuver node.
#[derive(Component)]
pub(super) struct NodeMarkerDisc {
    pub node_id: NodeId,
}

/// Flat circle for the snap indicator when placing a node (N key).
#[derive(Component)]
pub(super) struct SnapIndicator;

// ---------------------------------------------------------------------------
// Visual constants
// ---------------------------------------------------------------------------

pub(super) const SELECT_THRESHOLD_PX: f32 = 20.0;

/// Arrow dimensions in screen-stable units (scaled by camera distance each frame).
pub(super) const SHAFT_RADIUS: f32 = 0.002;
pub(super) const CONE_RADIUS: f32 = 0.005;
pub(super) const CONE_HEIGHT: f32 = 0.008;
pub(super) const BASE_ARROW_LEN: f32 = 0.04;
/// Hitbox capsule radius — generous for easy grabbing.
pub(super) const HITBOX_CAPSULE_RADIUS: f32 = 0.008;
/// Slide sphere radius.
pub(super) const SLIDE_SPHERE_RADIUS: f32 = 0.012;

pub(super) const ARROW_STRETCH: f32 = 0.0075;
pub(super) const STRETCH_LERP_SPEED: f32 = 12.0;
pub(super) const HOVER_BRIGHTNESS: f32 = 1.8;

/// Arrow colors: [prograde green, normal magenta, radial cyan].
pub(super) const ARROW_COLORS: [Color; 3] = [
    Color::srgb(0.0, 1.0, 0.0),
    Color::srgb(0.7, 0.0, 1.0),
    Color::srgb(0.0, 1.0, 1.0),
];
