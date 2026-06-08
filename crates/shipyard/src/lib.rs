//! Parametric ship construction for Thalos.
//!
//! Ships are ECS trees of parts connected by typed attach nodes. Fixed parts
//! (CommandPod, Engine) declare static node sizes; parametric parts
//! (Decoupler, Adapter, FuelTank) have their node sizes computed from the
//! parent they are attached to, via `sizing::propagate_node_sizes`.
//!
//! Serialization goes through a flat `ShipBlueprint` struct so that the ECS
//! representation stays query-friendly while the on-disk format stays stable.

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

pub mod attach;
pub mod blueprint;
pub mod catalog;
pub mod cockpit_mesh;
pub mod engine_mesh;
pub mod fuselage_mesh;
pub mod gear_mesh;
pub mod material;
pub mod part;
pub mod recompute;
pub mod resource;
pub mod sizing;
pub mod staging;
pub mod stats;
pub mod wing_mesh;

pub use attach::{
    AttachNode, AttachNodes, Attachment, NodeId, Ship, SurfaceMount, SurfaceMountKind,
    SymmetryGroup, SymmetryRole,
};
pub use blueprint::{
    Connection, PartBlueprint, PartParams, ShipBlueprint, SurfaceConnection, resource_capacity_for,
};
pub use catalog::{
    AdapterSpec, AmbientIntakeKind, CatalogEntry, CatalogError, CatalogId, CatalogRef,
    DecouplerSpec, EngineGeometry, EngineOptimization, EngineSpec, FuselageSpec, GearSpec,
    IntakeCapture, IntakeRequirement, IntakeSpec, PartCatalog, PodGeometry, PodSpec,
    ResourceStorageSpec, TankSpec, WingSpec, fuselage_surface_area, fuselage_volume, gear_dry_mass,
    pod_visual_profile, wing_mean_aerodynamic_chord, wing_panel_area,
};
pub use engine_mesh::{
    JetNacelleMount, build_jet_nacelle_body_mesh, build_jet_nacelle_pylon_mesh,
    jet_nacelle_centers, jet_nacelle_length,
};
pub use cockpit_mesh::build_cockpit_mesh;
pub use fuselage_mesh::{
    build_fuselage_mesh, host_mount_geometry, skin_radius as fuselage_skin_radius,
    v_offset_at as fuselage_v_offset_at,
};
pub use gear_mesh::{GearLegFrame, build_gear_bay_mesh, build_gear_mesh, gear_leg_frames};
pub use material::{
    ShipPartExtension, ShipPartMaterial, ShipPartParams, landing_gear_base, stainless_steel_base,
};
pub use part::{
    Adapter, AirIntake, CommandPod, Decoupler, Engine, EngineActivation, EngineThrust,
    EngineValidationError, FuelCrossfeed, FuelTank, Fuselage, Gear, MaterialKind, Part,
    PartMaterial, ReactantRatio, ReactionWheel, ShroudProvider, Shroudable, Wing,
};
pub use resource::{PartResources, Resource, ResourcePool};
pub use staging::{
    PartRole, StageIndices, StageSummary, SummaryEngine, SummaryPart, SummaryStageInput,
    compute_stage_summaries, derive_stages,
};
pub use stats::{
    DeltaVEnvironment, DeltaVEstimate, DeltaVInputs, G0, ResourceTotals, ShipStats, WingAeroPanel,
    aggregate_resource_totals, cylinder_principal_inertia, estimate_delta_v, parallel_axis_inertia,
};
pub use wing_mesh::{WingPanelFrame, build_wing_mesh, wing_panel_frame};

pub struct ShipyardPlugin;

impl Plugin for ShipyardPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<ShipPartMaterial>::default())
            .add_systems(
                Update,
                (
                    sizing::propagate_node_sizes,
                    // Catalog-driven mass + capacity recompute. Run after
                    // sizing so a parent-diameter propagation that mutates
                    // a child's `diameter` lands in the same frame.
                    recompute::recompute_decoupler_state.after(sizing::propagate_node_sizes),
                    recompute::recompute_adapter_state.after(sizing::propagate_node_sizes),
                    recompute::recompute_tank_state.after(sizing::propagate_node_sizes),
                    recompute::recompute_fuselage_state.after(sizing::propagate_node_sizes),
                    recompute::recompute_wing_state.after(sizing::propagate_node_sizes),
                    recompute::recompute_gear_state.after(sizing::propagate_node_sizes),
                ),
            );
    }
}
