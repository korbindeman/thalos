use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics_canonical::types::TrajectorySample;
use thalos_world::{BodyId, StateVector};

/// Unique identifier for a game-side maneuver node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// Transient trajectory section used for picking and slide interaction.
///
/// Rails are rebuilt from the current branch stack instead of being stored on
/// maneuver nodes, so upstream maneuver edits cannot leave stale copied paths
/// behind.
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

/// Burn lifecycle of a maneuver node.
///
/// A node is no longer deleted the instant its burn fires. It walks through
/// these phases so the maneuver panel and burn-progress HUD keep showing the
/// burn as it happens (KSP-style), and the spent node lingers for review until
/// the user dismisses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeBurnPhase {
    /// Not yet executed. Drives the trajectory prediction and can be armed by
    /// the autopilot.
    #[default]
    Planned,
    /// The autopilot is flying this burn right now. Excluded from the physics
    /// prediction (the live thrust already moves the ship, so re-applying the
    /// planned Δv would double-count) but still published as the active
    /// directive so the burn-progress bar fills.
    Executing,
    /// The burn has completed. Kept on screen for review until the user
    /// dismisses it; no longer drives prediction, arming, or the directive.
    Executed,
}

/// Provenance for a maneuver node.
///
/// Generated programs replace only their own unexecuted nodes. Manual nodes
/// remain untouched when an ORBIT target is replanned or cleared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeSource {
    #[default]
    Manual,
    OrbitProgram(u64),
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
    /// Where this node sits in its burn lifecycle.
    pub phase: NodeBurnPhase,
    pub source: NodeSource,
}

impl GameNode {
    /// `true` while this node should be fed into the physics `ManeuverSequence`
    /// (and therefore the trajectory prediction). Only [`NodeBurnPhase::Planned`]
    /// nodes qualify; a burning or spent node must not perturb the predicted path.
    pub fn drives_prediction(&self) -> bool {
        matches!(self.phase, NodeBurnPhase::Planned)
    }

    /// `true` while this node should be published as an autopilot burn directive
    /// — both [`NodeBurnPhase::Planned`] (so it can be armed) and
    /// [`NodeBurnPhase::Executing`] (so the burn-progress HUD keeps reading it).
    pub fn drives_directive(&self) -> bool {
        !matches!(self.phase, NodeBurnPhase::Executed)
    }

    /// `true` once the burn has been flown and the node is only kept for display.
    pub fn is_executed(&self) -> bool {
        matches!(self.phase, NodeBurnPhase::Executed)
    }
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
    pub last_slide_apply_secs: f64,
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
#[allow(clippy::enum_variant_names)]
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

// ---------------------------------------------------------------------------
// Visual constants
// ---------------------------------------------------------------------------

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
