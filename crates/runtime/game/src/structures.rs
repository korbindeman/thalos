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
//! part/loadout construction model is specced for M6 in `docs/gameplay/construction.md`
//! and intentionally not built here. See `docs/simulation/surface_local.md` §6.

use std::collections::HashMap;

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_terrain::{FlattenRegion, TerrainFlatten};
use thalos_world::BodyId;

use crate::rendering::ground_terrain::TerrainFlattenRegistry;

/// Stable per-session identifier for a placed structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub struct StructureId(pub u64);

/// What kind of structure a site is. Drives visuals/colliders (owned by the
/// kind's own systems — e.g. `crate::runway` for `Runway`); the registry and
/// flatten path are kind-agnostic. New kinds (buildings, pads) extend this.
#[derive(Debug, Clone, Copy, PartialEq, Reflect)]
pub enum StructureKind {
    /// A paved runway strip. Parametric so a base can carry several, each with
    /// its own size and heading (the takeoff heading is the site's
    /// `heading_tangent`). Half-extents are along the heading (`length`) and
    /// across it (`width`); the strip drapes flush on its parent `BaseSite`
    /// basin. Rendered + collided by [`crate::runway`].
    Runway {
        half_length_m: f32,
        half_width_m: f32,
    },
    /// A player-flattened building site. Owns the `FlattenTo` terrain pad;
    /// buildings placed on it drape on the levelled ground.
    BaseSite,
    /// A placed building — a simple parametric box for now, draped on its
    /// parent site's flattened pad. Half-extents are along the site's heading
    /// (`x`) and across it (`z`); `height_m` rises along the local vertical.
    Building {
        half_x_m: f32,
        half_z_m: f32,
        height_m: f32,
    },
    /// A launchpad — a circular slab draped on the pad that a craft can be
    /// placed on / launched from (the base editor's **L** action).
    Launchpad { radius_m: f32 },
    /// A storage tank (propellant / fluids) — a vertical cylinder draped on the
    /// pad. A stand-in for the tank-farm volume that flanks a launchpad;
    /// authored as part of the default base, and a first-class editable kind
    /// (select / move / delete) like any other structure.
    Tank { radius_m: f32, height_m: f32 },
}

/// An enterable facility a player reaches from the space-center hub, tagged onto
/// the building [`StructureSite`] that represents it. The hub's hover picker maps
/// a clicked facility building to its entry action. Only [`Facility::Vab`] is
/// wired today; the rest are the seams the picker leaves open (runway/pad launch,
/// tracking station, administration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum Facility {
    /// Vehicle Assembly Building — opens the shipyard editor.
    Vab,
}

impl Facility {
    /// Human-readable name shown in the hub's hover callout.
    pub fn label(self) -> &'static str {
        match self {
            Facility::Vab => "VEHICLE ASSEMBLY BUILDING",
        }
    }
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
        /// Rectangle-centre offset from `anchor_dir` (metres, along
        /// `heading_tangent` / `anchor_dir × heading_tangent`). The levelled
        /// **plane** stays tangent at `anchor_dir` — only the rectangle shifts —
        /// so an asymmetric footprint (the spaceport basin, pushed toward its
        /// secondary runway) shares one ground plane with everything anchored
        /// at the site centre. Anchoring the plane at the offset rect centre
        /// instead tilts the ground ~`offset/R` against the pavement: at 500 m
        /// offset on Thalos that buried the core apron's far strip under
        /// decimetres of terrain (the "dark serrated fringe" bug).
        rect_offset_along_m: f64,
        rect_offset_across_m: f64,
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
    /// For a building, the [`BaseSite`](StructureKind::BaseSite) it sits on, so
    /// deleting a site can take its buildings with it. `None` for top-level
    /// structures (runway, sites).
    pub parent_site: Option<StructureId>,
    /// If this structure is an enterable [`Facility`] (the VAB, …), which one —
    /// read by the space-center hub's click-to-enter. `None` for ordinary
    /// structures. Set post-registration via [`StructureRegistry::set_facility`].
    pub facility: Option<Facility>,
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
        parent_site: Option<StructureId>,
    ) -> StructureId {
        let id = self.allocate_id();
        self.sites.entry(body_id).or_default().push(StructureSite {
            id,
            body_id,
            anchor_dir,
            heading_tangent,
            placement,
            kind,
            parent_site,
            facility: None,
        });
        id
    }

    /// Tag a registered structure as an enterable [`Facility`] (e.g. the default
    /// base's VAB building). No-op if the id is unknown.
    pub fn set_facility(&mut self, id: StructureId, facility: Facility) {
        if let Some(site) = self.sites.values_mut().flatten().find(|s| s.id == id) {
            site.facility = Some(facility);
        }
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

    /// Mutate a registered structure in place (e.g. an inspector edit to a
    /// building's footprint). No-op if the id is unknown.
    pub fn update(&mut self, id: StructureId, f: impl FnOnce(&mut StructureSite)) {
        if let Some(site) = self.sites.values_mut().flatten().find(|s| s.id == id) {
            f(site);
        }
    }

    /// Remove a structure, returning it if found. Callers that remove a
    /// `FlattenTo` structure should also call [`remove_structure_flatten`] and
    /// trigger a terrain rebuild so the pad reverts.
    pub fn remove(&mut self, id: StructureId) -> Option<StructureSite> {
        for sites in self.sites.values_mut() {
            if let Some(pos) = sites.iter().position(|s| s.id == id) {
                return Some(sites.remove(pos));
            }
        }
        None
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
            .register_type::<StructureId>()
            .register_type::<StructureKind>()
            .register_type::<Facility>()
            .register_type::<StructurePlacement>();
    }
}
