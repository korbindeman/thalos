//! Parametric ship construction for Thalos.
//!
//! Ships are ECS trees of parts connected by typed attach nodes. Fixed parts
//! (CommandPod, Engine) declare static node sizes; parametric parts
//! (Decoupler, Adapter, FuelTank) have their node sizes computed from the
//! parent they are attached to, via `sizing::propagate_node_sizes`.
//!
//! Serialization goes through a flat `ShipBlueprint` struct so that the ECS
//! representation stays query-friendly while the on-disk format stays stable.

use bevy::prelude::*;

pub mod attach;
pub mod blueprint;
pub mod catalog;
pub mod part;
pub mod recompute;
pub mod resource;
pub mod sizing;
pub mod stats;

pub use attach::{AttachNode, AttachNodes, Attachment, NodeId, Ship};
pub use blueprint::{Connection, PartBlueprint, PartParams, ShipBlueprint};
pub use catalog::{
    AdapterSpec, CatalogEntry, CatalogError, CatalogId, CatalogRef, DecouplerSpec, EngineSpec,
    PartCatalog, PodSpec, TankSpec,
};
pub use part::{
    Adapter, CommandPod, Decoupler, Engine, EngineThrust, EngineValidationError, FuelTank,
    MaterialKind, Part, PartMaterial, ReactantRatio, ReactionWheel, Shroudable, ShroudProvider,
};
pub use resource::{PartResources, Resource, ResourcePool};
pub use stats::{G0, ResourceTotals, ShipStats};

pub struct ShipyardPlugin;

impl Plugin for ShipyardPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                sizing::propagate_node_sizes,
                // Catalog-driven mass + capacity recompute. Run after
                // sizing so a parent-diameter propagation that mutates
                // a child's `diameter` lands in the same frame.
                recompute::recompute_decoupler_state.after(sizing::propagate_node_sizes),
                recompute::recompute_adapter_state.after(sizing::propagate_node_sizes),
                recompute::recompute_tank_state.after(sizing::propagate_node_sizes),
            ),
        );
    }
}
