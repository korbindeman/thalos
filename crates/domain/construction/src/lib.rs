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
pub mod cockpit_mesh;
pub mod engine_mesh;
pub mod fairing_mesh;
pub mod fuselage_mesh;
pub mod gear_mesh;
pub mod part;
pub mod part_mesh;
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
    BuildLayout, Connection, PartBlueprint, PartParams, ShipBlueprint, SurfaceConnection,
    resource_capacity_for,
};
pub use catalog::{
    AdapterSpec, AmbientIntakeKind, CatalogEntry, CatalogError, CatalogId, CatalogRef,
    DecouplerSpec, EngineGeometry, EngineOptimization, EngineSpec, FuselageSpec, GearSpec,
    IntakeCapture, IntakeRequirement, IntakeSpec, PartCatalog, PodGeometry, PodSpec,
    ResourceStorageSpec, TankSpec, WingRole, WingSpec, default_control_surfaces,
    fuselage_surface_area, fuselage_volume, gear_dry_mass, pod_visual_profile,
    wing_mean_aerodynamic_chord, wing_panel_area,
};
pub use cockpit_mesh::build_cockpit_mesh;
pub use engine_mesh::{
    JetNacelleMount, build_jet_nacelle_body_mesh, build_jet_nacelle_pylon_mesh,
    jet_nacelle_centers, jet_nacelle_length,
};
pub use fairing_mesh::{build_wing_fairing_mesh, wants_wing_fairing};
pub use fuselage_mesh::{
    build_fuselage_mesh, host_mount_geometry, skin_radius as fuselage_skin_radius,
    v_offset_at as fuselage_v_offset_at,
};
pub use gear_mesh::{
    GearLegFrame, build_gear_bay_mesh, build_gear_mesh, build_gear_struct_mesh, gear_leg_frames,
};
pub use part::{
    Adapter, AirIntake, CommandPod, ControlSurface, ControlSurfaceRole, Decoupler, Engine,
    EngineActivation, EngineThrust, EngineValidationError, FuelCrossfeed, FuelTank, Fuselage, Gear,
    MaterialKind, Part, PartMaterial, ReactantRatio, ReactionWheel, ShroudProvider, Shroudable,
    Wing,
};
pub use part_mesh::add_raytracing_tangents;
pub use resource::{PartResources, Resource, ResourcePool};
pub use staging::{
    PartRole, StageIndices, StageSummary, SummaryEngine, SummaryPart, SummaryStageInput,
    compute_stage_summaries, derive_stages,
};
pub use stats::{
    AeroSurfaceWindow, DeltaVEnvironment, DeltaVEstimate, DeltaVInputs, G0, ResourceTotals,
    ShipStats, WingAeroPanel, aggregate_resource_totals, cylinder_principal_inertia,
    estimate_delta_v, live_part_centroid_offset, live_part_dry_mass_kg, live_part_self_inertia,
    live_part_total_mass_kg, parallel_axis_inertia,
};
pub use wing_mesh::{
    BuiltControlSurface, ControlSurfaceGeometry, WingPanelFrame, build_control_surface_mesh,
    build_wing_mesh, control_surface_geometry, wing_panel_frame,
};

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
                recompute::recompute_fuselage_state.after(sizing::propagate_node_sizes),
                recompute::recompute_wing_state.after(sizing::propagate_node_sizes),
                recompute::recompute_gear_state.after(sizing::propagate_node_sizes),
            ),
        );
    }
}
