use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::types::{BodyId, StateVector, TrajectorySample};

/// Unique identifier for a game-side maneuver node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// Immutable trajectory section captured when a maneuver node is placed.
///
/// The physics maneuver sequence still only consumes `(time, delta_v,
/// reference_body)`. This rail is game-side UX state: node markers, drag
/// handles, and slide interaction stay attached to the exact visible path the
/// player clicked, even after later prediction rebuilds bend the planned path.
#[derive(Clone, Debug)]
pub struct TrajectoryRail {
    pub frame: RailFrame,
    pub reference_body: BodyId,
    pub samples: Vec<TrajectorySample>,
}

#[derive(Clone, Debug)]
pub enum RailFrame {
    /// A normal body-centered path in the current patched-conics frame.
    Body { body_id: BodyId },
    /// A future encounter frame pinned to a ghost body.
    Ghost {
        body_id: BodyId,
        parent_id: BodyId,
        relative_position: DVec3,
        projection_epoch: f64,
        encounter_epoch: f64,
        soi_radius: f64,
    },
}

impl TrajectoryRail {
    pub fn epoch_range(&self) -> Option<(f64, f64)> {
        Some((self.samples.first()?.time, self.samples.last()?.time))
    }

    pub fn state_at(&self, time: f64) -> Option<StateVector> {
        state_at_samples(&self.samples, time)
    }
}

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
    /// Exact visible path section this node was placed on.
    pub rail: Option<TrajectoryRail>,
}

/// UI-side maneuver plan. Synced to `ManeuverSequence` in physics when dirty.
#[derive(Resource, Default)]
pub struct ManeuverPlan {
    pub nodes: Vec<GameNode>,
    pub dirty: bool,
    next_id: u64,
    /// Bevy elapsed-seconds reading at the most recent slide-driven
    /// `dirty = true` flip. The slide handler reads this to throttle the
    /// (expensive) flight-plan rebuild during a drag — see
    /// [`super::interaction::handle_maneuver_events`].
    pub(super) last_slide_apply_secs: f64,
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

/// Live world position and orbital frame for a node currently being slid.
///
/// During a slide, [`ManeuverPlan::last_slide_apply_secs`] throttles flight-
/// plan rebuilds to ~10 Hz, so the cached prediction can be up to 100 ms
/// behind `node.time` for the dragged node. Sampling that prediction at the
/// fresh `node.time` may land on the wrong leg (e.g. the post-burn coast
/// instead of the unperturbed baseline) and snap the visual marker off the
/// orbit the user is dragging along.
///
/// [`super::interaction::maneuver_input`] writes this resource directly from
/// the chosen [`super::helpers::ClosestTrailPoint`], bypassing the stale
/// prediction; [`super::update_selected_node_view`] then prefers it whenever
/// [`InteractionMode::SlidingNode`] is active.
#[derive(Resource, Default)]
pub(super) struct SlidePreview {
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
        snap_anchor_body: Option<usize>,
        snap_rail: Option<TrajectoryRail>,
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
        rail: Option<TrajectoryRail>,
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

fn state_at_samples(samples: &[TrajectorySample], time: f64) -> Option<StateVector> {
    let n = samples.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        let s = samples[0];
        if (time - s.time).abs() <= 1e-6 {
            return Some(StateVector {
                position: s.position,
                velocity: s.velocity,
            });
        }
        return None;
    }

    let start = samples[0].time;
    let end = samples[n - 1].time;
    if time < start - 1e-6 || time > end + 1e-6 {
        return None;
    }

    let mut lo = 0usize;
    let mut hi = n - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if samples[mid].time <= time {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let a = samples[lo];
    let b = samples[(lo + 1).min(n - 1)];
    let h = b.time - a.time;
    if h <= 0.0 {
        return Some(StateVector {
            position: a.position,
            velocity: a.velocity,
        });
    }

    let tau = ((time - a.time) / h).clamp(0.0, 1.0);
    let tau2 = tau * tau;
    let tau3 = tau2 * tau;

    let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
    let h10 = tau3 - 2.0 * tau2 + tau;
    let h01 = -2.0 * tau3 + 3.0 * tau2;
    let h11 = tau3 - tau2;

    let position =
        a.position * h00 + a.velocity * (h10 * h) + b.position * h01 + b.velocity * (h11 * h);

    let dh00 = 6.0 * tau2 - 6.0 * tau;
    let dh10 = 3.0 * tau2 - 4.0 * tau + 1.0;
    let dh01 = -6.0 * tau2 + 6.0 * tau;
    let dh11 = 3.0 * tau2 - 2.0 * tau;

    let velocity =
        a.position * (dh00 / h) + a.velocity * dh10 + b.position * (dh01 / h) + b.velocity * dh11;

    Some(StateVector { position, velocity })
}
