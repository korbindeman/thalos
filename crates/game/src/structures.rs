//! Terrain-anchored structures: the data-driven generalization of the runway.
//!
//! A [`StructureSite`] is a body-fixed placement of something that sits on a
//! planetary surface — today only the runway, later buildings, pads, towers,
//! and eventually player-placed/edited structures. The single
//! [`StructureRegistry`] resource holds every site per body, and
//! [`apply_structure_flatten`] is the one path that makes a structure "stick to
//! the terrain": for a [`StructurePlacement::FlattenTo`] site it installs a
//! [`TerrainFlatten`] pad through the body's shared
//! [`crate::rendering::ground_terrain::TerrainFlattenRegistry`] handle, so the
//! rendered ground — and, via the GPU-atlas height mirror, the collider and CPU
//! height queries — level out across the footprint and smoothstep-blend back to
//! natural terrain over the ramp. The runway populates this registry; a future
//! building is a data entry plus its own visuals, not a bespoke plugin.
//!
//! Scope note: this is the *terrain-anchoring* layer only — the full
//! part/loadout construction model is specced for M6 in `docs/construction.md`
//! and intentionally not built here. See `docs/surface_local.md` §6.

use std::collections::HashMap;

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_terrain::TerrainFlatten;
use thalos_world::BodyId;

use crate::rendering::ground_terrain::TerrainFlattenRegistry;

/// Stable per-session identifier for a placed structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub struct StructureId(pub u64);

/// What kind of structure a site is. Drives visuals/colliders (owned by the
/// kind's own systems — e.g. `crate::runway` for `Runway`); the registry and
/// flatten path are kind-agnostic. New kinds (buildings, pads) extend this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum StructureKind {
    Runway,
}

/// How a structure meets the terrain.
#[derive(Debug, Clone, Copy, PartialEq, Reflect)]
pub enum StructurePlacement {
    /// Flatten the terrain under the footprint to a fixed elevation, blending
    /// back to natural terrain over `ramp_m`. The half-extents are along the
    /// `heading_tangent` and across it. This is how the runway gets its flat pad.
    FlattenTo {
        elevation_m: f64,
        half_along_m: f64,
        half_across_m: f64,
        ramp_m: f64,
    },
    /// Sit on the natural (un-flattened) surface — the structure's own geometry
    /// conforms to the terrain. No terrain modification.
    Drape,
}

/// A body-fixed placement of a terrain-anchored structure.
#[derive(Debug, Clone, Copy, Reflect)]
pub struct StructureSite {
    pub id: StructureId,
    pub body_id: BodyId,
    /// Unit body-fixed direction from the body centre to the site.
    pub anchor_dir: DVec3,
    /// Unit body-fixed tangent at the site (e.g. runway takeoff heading).
    pub heading_tangent: DVec3,
    pub placement: StructurePlacement,
    pub kind: StructureKind,
}

/// Every terrain-anchored structure, grouped by body.
///
/// **Sole writer:** structure spawners (today [`crate::runway`]) via
/// [`Self::register`]. Readers (collider/visual attach as the surface-local
/// bubble streams in) query [`Self::sites_on`]. Player placement will write
/// here at runtime; the data model is identical to authored sites.
#[derive(Resource, Default)]
pub struct StructureRegistry {
    sites: HashMap<BodyId, Vec<StructureSite>>,
    next_id: u64,
}

impl StructureRegistry {
    fn allocate_id(&mut self) -> StructureId {
        self.next_id += 1;
        StructureId(self.next_id)
    }

    /// Register a structure on its body, returning the assigned id. The caller
    /// supplies everything but the id.
    pub fn register(
        &mut self,
        body_id: BodyId,
        anchor_dir: DVec3,
        heading_tangent: DVec3,
        placement: StructurePlacement,
        kind: StructureKind,
    ) -> StructureId {
        let id = self.allocate_id();
        self.sites.entry(body_id).or_default().push(StructureSite {
            id,
            body_id,
            anchor_dir,
            heading_tangent,
            placement,
            kind,
        });
        id
    }

    /// All structures on a body. Read by the MFD navigation-display widget
    /// (runway projection) and, later, per-body structure-attach systems
    /// (collider/visual spawn as the surface bubble streams in).
    pub fn sites_on(&self, body_id: BodyId) -> &[StructureSite] {
        self.sites.get(&body_id).map_or(&[], |v| v.as_slice())
    }

    pub fn get(&self, id: StructureId) -> Option<&StructureSite> {
        self.sites.values().flatten().find(|s| s.id == id)
    }
}

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
    );
    if let Ok(mut guard) = flatten_registry.handle(site.body_id).write() {
        *guard = Some(flatten);
    }
}

/// Registers [`StructureRegistry`] and its reflection. Structure-kind spawners
/// (the runway, future buildings) add their own systems.
pub struct StructuresPlugin;

impl Plugin for StructuresPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StructureRegistry>()
            .register_type::<StructureId>()
            .register_type::<StructureKind>()
            .register_type::<StructurePlacement>();
    }
}
