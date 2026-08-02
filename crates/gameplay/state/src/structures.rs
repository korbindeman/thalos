//! Terrain-anchored structure vocabulary: what a structure is and the
//! per-body registry every placement/scatter/collider consumer reads. The
//! placement systems and per-kind visuals stay with the runtime (cleanup
//! package D turns this into the one placement layer).

use std::collections::HashMap;

use bevy::math::DVec3;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
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
    /// Owning base. Every structure belongs to exactly one base; this is the
    /// stable campaign identity used for save/load and projection reconcile.
    pub base_id: BaseId,
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

/// Stable campaign record for one base. Its root is the `BaseSite` that owns
/// the terrain footprint; optional roles point at ordinary child structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub struct BaseRecord {
    pub id: BaseId,
    pub body_id: BodyId,
    pub root_site: StructureId,
    pub primary_runway: Option<StructureId>,
}

/// Every terrain-anchored base and structure, grouped by body.
///
/// Base identity is the uniqueness authority. An authored loader calls
/// [`Self::ensure_base`] with a stable [`BaseId`]; requesting it twice yields
/// the existing record and cannot append a second base. Child registration
/// requires a valid parent, so an orphan structure is unrepresentable through
/// the public API.
#[derive(Resource, Default)]
pub struct StructureRegistry {
    sites: HashMap<BodyId, Vec<StructureSite>>,
    bases: HashMap<BaseId, BaseRecord>,
    next_structure_id: u64,
    next_base_id: u64,
}

/// Stable campaign identifier for a base/space center.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize, PartialOrd, Ord,
)]
pub struct BaseId(pub u64);

impl BaseId {
    const AUTHORED_NAMESPACE: u64 = 1 << 63;

    /// Stable ID for a base authored by bundled content. The high-bit namespace
    /// cannot collide with monotonically allocated player bases.
    pub const fn authored(content_id: u64) -> Self {
        assert!(content_id < Self::AUTHORED_NAMESPACE);
        Self(Self::AUTHORED_NAMESPACE | content_id)
    }

    pub const fn is_authored(self) -> bool {
        self.0 & Self::AUTHORED_NAMESPACE != 0
    }
}

/// Stable campaign identifier for a placed structure.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize, PartialOrd, Ord,
)]
pub struct StructureId(pub u64);

/// Result of reconciling an authored base identity into the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseRegistration {
    Existing(BaseRecord),
    Created(BaseRecord),
}

impl BaseRegistration {
    pub fn record(self) -> BaseRecord {
        match self {
            Self::Existing(record) | Self::Created(record) => record,
        }
    }

    pub fn was_created(self) -> bool {
        matches!(self, Self::Created(_))
    }
}

/// A stable base identity was reused for different authored data. This is
/// corrupted snapshot/fixture data, not a condition the runtime may silently
/// repair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseIdentityConflict {
    Body {
        id: BaseId,
        existing_body: BodyId,
        requested_body: BodyId,
    },
    Definition {
        id: BaseId,
    },
}

/// Child registration failed because the requested parent does not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingStructureParent(pub StructureId);

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
    fn allocate_structure_id(&mut self) -> StructureId {
        self.next_structure_id += 1;
        StructureId(self.next_structure_id)
    }

    fn allocate_base_id(&mut self) -> BaseId {
        self.next_base_id += 1;
        assert!(self.next_base_id < BaseId::AUTHORED_NAMESPACE);
        BaseId(self.next_base_id)
    }

    fn insert_base(
        &mut self,
        base_id: BaseId,
        body_id: BodyId,
        anchor_dir: DVec3,
        heading_tangent: DVec3,
        placement: StructurePlacement,
    ) -> BaseRecord {
        if !base_id.is_authored() {
            self.next_base_id = self.next_base_id.max(base_id.0);
        }
        let id = self.allocate_structure_id();
        self.sites.entry(body_id).or_default().push(StructureSite {
            id,
            base_id,
            body_id,
            anchor_dir,
            heading_tangent,
            placement,
            kind: StructureKind::BaseSite,
            parent_site: None,
            facility: None,
        });
        let record = BaseRecord {
            id: base_id,
            body_id,
            root_site: id,
            primary_runway: None,
        };
        self.bases.insert(base_id, record);
        record
    }

    /// Create a player-built base with a freshly allocated stable identity.
    pub fn create_base(
        &mut self,
        body_id: BodyId,
        anchor_dir: DVec3,
        heading_tangent: DVec3,
        placement: StructurePlacement,
    ) -> BaseRecord {
        let base_id = self.allocate_base_id();
        self.insert_base(base_id, body_id, anchor_dir, heading_tangent, placement)
    }

    /// Reconcile an authored base with an explicit campaign identity. Reusing
    /// the identity on the same body returns the original record without
    /// registering another structure; reusing it on another body is invalid.
    pub fn ensure_base(
        &mut self,
        base_id: BaseId,
        body_id: BodyId,
        anchor_dir: DVec3,
        heading_tangent: DVec3,
        placement: StructurePlacement,
    ) -> Result<BaseRegistration, BaseIdentityConflict> {
        if let Some(existing) = self.bases.get(&base_id).copied() {
            if existing.body_id != body_id {
                return Err(BaseIdentityConflict::Body {
                    id: base_id,
                    existing_body: existing.body_id,
                    requested_body: body_id,
                });
            }
            let Some(root) = self.get(existing.root_site) else {
                return Err(BaseIdentityConflict::Definition { id: base_id });
            };
            if root.anchor_dir != anchor_dir
                || root.heading_tangent != heading_tangent
                || root.placement != placement
                || root.kind != StructureKind::BaseSite
            {
                return Err(BaseIdentityConflict::Definition { id: base_id });
            }
            return Ok(BaseRegistration::Existing(existing));
        }
        Ok(BaseRegistration::Created(self.insert_base(
            base_id,
            body_id,
            anchor_dir,
            heading_tangent,
            placement,
        )))
    }

    /// Register a child on an existing base. Body and base identity are
    /// inherited from the parent, preventing mismatched/orphan records.
    pub fn register_child(
        &mut self,
        parent_site: StructureId,
        anchor_dir: DVec3,
        heading_tangent: DVec3,
        placement: StructurePlacement,
        kind: StructureKind,
    ) -> Result<StructureId, MissingStructureParent> {
        let Some(parent) = self.get(parent_site).copied() else {
            return Err(MissingStructureParent(parent_site));
        };
        let id = self.allocate_structure_id();
        self.sites
            .entry(parent.body_id)
            .or_default()
            .push(StructureSite {
                id,
                base_id: parent.base_id,
                body_id: parent.body_id,
                anchor_dir,
                heading_tangent,
                placement,
                kind,
                parent_site: Some(parent_site),
                facility: None,
            });
        Ok(id)
    }

    pub fn base(&self, id: BaseId) -> Option<&BaseRecord> {
        self.bases.get(&id)
    }

    pub fn base_for_site(&self, id: StructureId) -> Option<&BaseRecord> {
        let site = self.get(id)?;
        self.base(site.base_id)
    }

    /// Assign the runway role within a base. Returns `false` if either identity
    /// is unknown, the structure belongs to another base, or it is not a runway.
    pub fn set_primary_runway(&mut self, base_id: BaseId, runway_id: StructureId) -> bool {
        let Some(runway) = self.get(runway_id) else {
            return false;
        };
        if runway.base_id != base_id || !matches!(runway.kind, StructureKind::Runway { .. }) {
            return false;
        }
        let Some(base) = self.bases.get_mut(&base_id) else {
            return false;
        };
        base.primary_runway = Some(runway_id);
        true
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

    /// Remove a child structure, returning it if found. Base roots have a
    /// different aggregate lifecycle and cannot be removed through this API;
    /// that prevents dangling `BaseRecord`/child identities.
    pub fn remove_child(&mut self, id: StructureId) -> Option<StructureSite> {
        for sites in self.sites.values_mut() {
            if let Some(pos) = sites
                .iter()
                .position(|site| site.id == id && site.parent_site.is_some())
            {
                let removed = sites.remove(pos);
                if let Some(base) = self.bases.get_mut(&removed.base_id)
                    && base.primary_runway == Some(id)
                {
                    base.primary_runway = None;
                }
                return Some(removed);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement() -> StructurePlacement {
        StructurePlacement::FlattenTo {
            elevation_m: 10.0,
            half_along_m: 100.0,
            half_across_m: 100.0,
            ramp_m: 20.0,
            rect_offset_along_m: 0.0,
            rect_offset_across_m: 0.0,
        }
    }

    #[test]
    fn authored_base_identity_cannot_be_registered_twice() {
        let mut registry = StructureRegistry::default();
        let id = BaseId(7);
        let first = registry
            .ensure_base(id, 2, DVec3::Y, DVec3::X, placement())
            .unwrap();
        let second = registry
            .ensure_base(id, 2, DVec3::Y, DVec3::X, placement())
            .unwrap();

        assert!(first.was_created());
        assert!(!second.was_created());
        assert_eq!(first.record(), second.record());
        assert_eq!(registry.sites_on(2).len(), 1);
    }

    #[test]
    fn authored_base_identity_cannot_move_between_bodies() {
        let mut registry = StructureRegistry::default();
        registry
            .ensure_base(BaseId(3), 1, DVec3::Y, DVec3::X, placement())
            .unwrap();

        assert_eq!(
            registry.ensure_base(BaseId(3), 2, DVec3::Y, DVec3::X, placement()),
            Err(BaseIdentityConflict::Body {
                id: BaseId(3),
                existing_body: 1,
                requested_body: 2,
            })
        );
    }

    #[test]
    fn authored_base_identity_cannot_silently_change_definition() {
        let mut registry = StructureRegistry::default();
        registry
            .ensure_base(BaseId::authored(3), 1, DVec3::Y, DVec3::X, placement())
            .unwrap();

        assert_eq!(
            registry.ensure_base(BaseId::authored(3), 1, DVec3::Z, DVec3::X, placement()),
            Err(BaseIdentityConflict::Definition {
                id: BaseId::authored(3)
            })
        );
    }

    #[test]
    fn child_requires_a_real_parent_and_inherits_base() {
        let mut registry = StructureRegistry::default();
        let base = registry.create_base(4, DVec3::Y, DVec3::X, placement());
        let child = registry
            .register_child(
                base.root_site,
                DVec3::Y,
                DVec3::X,
                StructurePlacement::Drape,
                StructureKind::Launchpad { radius_m: 12.0 },
            )
            .unwrap();

        assert_eq!(registry.get(child).unwrap().base_id, base.id);
        assert!(registry.remove_child(base.root_site).is_none());
        assert!(registry.base(base.id).is_some());
        assert_eq!(
            registry.register_child(
                StructureId(999),
                DVec3::Y,
                DVec3::X,
                StructurePlacement::Drape,
                StructureKind::Launchpad { radius_m: 12.0 },
            ),
            Err(MissingStructureParent(StructureId(999)))
        );
    }

    #[test]
    fn authored_and_player_base_ids_cannot_collide() {
        let mut registry = StructureRegistry::default();
        registry
            .ensure_base(BaseId::authored(1), 4, DVec3::Y, DVec3::X, placement())
            .unwrap();
        let player = registry.create_base(4, DVec3::Y, DVec3::X, placement());

        assert_eq!(player.id, BaseId(1));
        assert_ne!(player.id, BaseId::authored(1));
    }
}
