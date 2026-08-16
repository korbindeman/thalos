//! Terrain-anchored structures: the data-driven generalization of the runway.
//!
//! A [`StructureSite`] is a body-fixed placement of something that sits on a
//! planetary surface — today only the runway, later buildings, pads, towers,
//! and eventually player-placed/edited structures. The single
//! [`StructureRegistry`] resource holds every site per body, and
//! [`apply_structure_flatten`] is the one path that makes a structure "stick to
//! the terrain": for a [`StructurePlacement::FlattenTo`] site it installs a
//! [`TerrainFlatten`] pad through the body's shared
//! [`crate::rendering::terrain_flatten::TerrainFlattenRegistry`] handle, so the
//! rendered ground — and, via the GPU-atlas height mirror, the collider and CPU
//! height queries — level out across the footprint and smoothstep-blend back to
//! natural terrain over the ramp. The runway populates this registry; a future
//! building is a data entry plus its own visuals, not a bespoke plugin.
//!
//! Scope note: this is the *terrain-anchoring* layer only — the full
//! part/loadout construction model is specced for M6 in `docs/gameplay/construction.md`
//! and intentionally not built here. See `docs/simulation/surface_local.md` §6.

use bevy::prelude::*;
use thalos_terrain::{FlattenRegion, TerrainFlatten};
use thalos_world::BodyId;

use crate::rendering::terrain_flatten::TerrainFlattenRegistry;

pub use thalos_game_state::structures::{
    BaseId, BaseRecord, Facility, StructureId, StructureKind, StructurePlacement,
    StructureRegistry, StructureSite,
};

/// Install a structure's terrain modification into the body's shared flatten
/// handle. The single "stick to the terrain" path: any `FlattenTo` structure
/// levels its footprint through the same machinery the runway uses, so the
/// rendered ground, the surface-local heightfield collider, and CPU height
/// queries all agree. A `Drape` structure modifies nothing. Call this before
/// the surface tiles at the site stream in so they bake flattened from the
/// start (the registry handle persists across terrain residency churn).
pub fn apply_structure_flatten(
    site: &StructureSite,
    body_radius_m: f64,
    flatten_registry: &mut TerrainFlattenRegistry,
) {
    let StructurePlacement::FlattenTo {
        elevation_m,
        half_along_m,
        half_across_m,
        ramp_m,
        rect_offset_along_m,
        rect_offset_across_m,
    } = site.placement
    else {
        return;
    };
    let across = site.anchor_dir.cross(site.heading_tangent).normalize();
    let flatten = TerrainFlatten::new(
        site.anchor_dir,
        site.heading_tangent,
        across,
        half_along_m,
        half_across_m,
        ramp_m,
        elevation_m,
        body_radius_m,
    )
    .with_rect_offset(rect_offset_along_m, rect_offset_across_m);
    if let Ok(mut guard) = flatten_registry.handle(site.body_id).write() {
        let region = FlattenRegion {
            id: site.id.0,
            flatten,
        };
        // Upsert by id so re-applying an edited site replaces its own pad
        // rather than stacking a duplicate region, and other structures'
        // pads (the runway) are left untouched.
        if let Some(existing) = guard.iter_mut().find(|r| r.id == region.id) {
            *existing = region;
        } else {
            guard.push(region);
        }
    }
}

/// Remove a structure's terrain flatten from the body's shared handle, reverting
/// its footprint to natural terrain on tiles baked afterward. The inverse of
/// [`apply_structure_flatten`]; call it when a `FlattenTo` structure is deleted
/// (and trigger a terrain rebuild so already-resident tiles re-bake unflattened).
// Inverse of `apply_structure_flatten`, ready for the base editor's
// `FlattenTo` structure-delete path; no caller wires it yet.
#[allow(dead_code)]
pub fn remove_structure_flatten(
    id: StructureId,
    body_id: BodyId,
    flatten_registry: &mut TerrainFlattenRegistry,
) {
    if let Ok(mut guard) = flatten_registry.handle(body_id).write() {
        guard.retain(|r| r.id != id.0);
    }
}

/// Registers [`StructureRegistry`] and its reflection. Structure-kind spawners
/// (the runway, future buildings) add their own systems.
pub struct StructuresPlugin;

impl Plugin for StructuresPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StructureRegistry>()
            .register_type::<BaseId>()
            .register_type::<BaseRecord>()
            .register_type::<StructureId>()
            .register_type::<StructureKind>()
            .register_type::<Facility>()
            .register_type::<StructurePlacement>();
    }
}
