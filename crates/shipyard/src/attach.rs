use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type NodeId = String;

#[derive(Clone, Debug)]
pub struct AttachNode {
    pub diameter: f32,
    pub offset: Vec3,
}

#[derive(Component, Default, Debug, Clone)]
pub struct AttachNodes {
    pub nodes: HashMap<NodeId, AttachNode>,
}

impl AttachNodes {
    pub fn get(&self, id: &str) -> Option<&AttachNode> {
        self.nodes.get(id)
    }

    pub fn set(&mut self, id: impl Into<NodeId>, node: AttachNode) {
        self.nodes.insert(id.into(), node);
    }
}

/// This entity is attached to `parent` — `my_node` mates with `parent_node`.
///
/// This is the **end-node stack** placement: two parts mate at named nodes
/// and diameter propagates parent→child (`sizing::propagate_node_sizes`).
/// It is the original (and only, until wings) connection mechanism — the
/// rocket path. Surface/footprint placement uses [`SurfaceMount`] instead.
#[derive(Component, Debug, Clone)]
pub struct Attachment {
    pub parent: Entity,
    pub parent_node: NodeId,
    pub my_node: NodeId,
}

/// Whether a [`SurfaceMount`] is a single footprint or a mirrored pair
/// about the host's vertical (X = 0) plane. A general footprint property —
/// wings today, landing gear / wing-mounted engines later (symmetry is
/// first-class for off-centreline mounts, `docs/construction.md` §4.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MountSymmetry {
    /// A single footprint at the mount point — a dorsal/ventral fin
    /// (vertical stabiliser), a centreline part. Not mirrored.
    Single,
    /// A mirrored left/right pair drawn from one part — main wings,
    /// tailplanes. The part renders two footprints reflected across the
    /// host's X = 0 plane, and its area / mass count double.
    #[default]
    Mirrored,
}

/// Which surface-placement frame a [`SurfaceMount`] uses.
///
/// `BodySkin` is the original wing-on-fuselage frame: `station` is a
/// fraction down the host body axis and `angle` is the azimuth around the
/// host skin. `WingPylon` reuses the same storage for a wing-hosted nacelle:
/// `station` is span fraction (root→tip) and `angle` is chord fraction
/// (`-0.5` trailing edge, `0.5` leading edge).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SurfaceMountKind {
    #[default]
    BodySkin,
    WingPylon,
}

/// **Surface / footprint** placement: this entity sits on `parent`'s skin
/// at a `(station, angle)` point rather than mating an end node. This is
/// the second placement capability from `docs/construction.md` §4.3 —
/// wings, wing-hosted nacelles, and later gear / footprint cockpits.
/// Surface mounts deliberately **opt out of diameter propagation**: a
/// footprint part's size is its own, not the host's local diameter.
///
/// Kept as a component distinct from [`Attachment`] so the existing
/// end-node stack code (sizing, shrouds, staging topology) is untouched;
/// traversals that need the *whole* part graph union both. `parent` plays
/// the same connectivity role as [`Attachment::parent`].
#[derive(Component, Debug, Clone, Copy)]
pub struct SurfaceMount {
    pub parent: Entity,
    pub kind: SurfaceMountKind,
    /// Fraction along the host body axis, 0 = top (y = 0) → 1 = bottom
    /// (y = −height) for [`SurfaceMountKind::BodySkin`], or span fraction
    /// root→tip for [`SurfaceMountKind::WingPylon`].
    pub station: f32,
    /// Angle around the host body axis, radians. The primary panel's
    /// outboard radial is `(sin angle, 0, cos angle)`; `angle = 0` is the
    /// +Z (dorsal / "up") side, `angle = π/2` the +X (right) side. For
    /// [`SurfaceMountKind::WingPylon`], this stores chord fraction instead
    /// (`-0.5` trailing edge, `0.5` leading edge).
    pub angle: f32,
    pub symmetry: MountSymmetry,
}

/// Root of a ship assembly.
#[derive(Component, Debug, Clone)]
pub struct Ship {
    pub name: String,
    pub root: Entity,
}
