//! Terrain-anchored structure vocabulary: what a structure is and the
//! per-body registry every placement/scatter/collider consumer reads. The
//! placement systems and per-kind visuals stay with the runtime (cleanup
//! package D turns this into the one placement layer).

use std::collections::HashMap;

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_world::BodyId;

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
    pub sites: HashMap<BodyId, Vec<StructureSite>>,
    pub next_id: u64,
}

/// Stable per-session identifier for a placed structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub struct StructureId(pub u64);

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

impl Facility {
    /// Human-readable name shown in the hub's hover callout.
    pub fn label(self) -> &'static str {
        match self {
            Facility::Vab => "VEHICLE ASSEMBLY BUILDING",
        }
    }
}

impl StructureRegistry {
    pub fn allocate_id(&mut self) -> StructureId {
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

    /// Surface elevation (m above the body reference radius) a structure
    /// actually sits at.
    ///
    /// A [`StructurePlacement::FlattenTo`] site carries its own level; a
    /// **`Drape` site does not, and its own placement says nothing** — it drapes
    /// on its parent's flattened pad, so the parent's elevation is the answer.
    /// Every runway on the spaceport is exactly that case (a `Drape` strip on a
    /// basin levelled to ~700 m), which is why reading `Drape` as "elevation 0"
    /// is wrong: harmless for a top-down plot, but a 700 m error in threshold
    /// elevation puts an approach glideslope completely off.
    pub fn site_elevation_m(&self, site: &StructureSite) -> f64 {
        match site.placement {
            StructurePlacement::FlattenTo { elevation_m, .. } => elevation_m,
            StructurePlacement::Drape => site
                .parent_site
                .and_then(|parent| self.get(parent))
                .map(|parent| self.site_elevation_m(parent))
                .unwrap_or(0.0),
        }
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
