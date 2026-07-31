use crate::attach::{
    AttachNode, AttachNodes, Attachment, NodeId, Ship, SurfaceMount, SurfaceMountKind,
    SymmetryGroup, SymmetryRole,
};
use crate::catalog::{
    CatalogEntry, CatalogError, CatalogId, CatalogRef, PartCatalog, WingRole, adapter_surface_area,
    fuselage_surface_area, fuselage_volume, gear_dry_mass, tank_surface_area, tank_volume,
    wing_panel_area, wing_volume,
};
use crate::part::{
    Adapter, AirIntake, CommandPod, ControlSurface, Decoupler, Engine, EngineActivation,
    EngineThrust, FuelCrossfeed, FuelTank, Fuselage, Gear, Part, PartMaterial, ReactionWheel,
    ShroudProvider, Shroudable, Wing,
};
use crate::resource::{PartResources, Resource, ResourcePool};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Serde default for a fuselage's `tail_bluntness` (blueprints saved before
/// the field existed): a softly rounded tail.
fn default_tail_bluntness() -> f32 {
    0.35
}

/// Per-instance parameters that the catalog cannot derive. Pure-catalog
/// kinds (Pod, Engine) carry [`PartParams::None`]. Parametric kinds carry
/// the dimensions the user picks; the catalog turns those into mass,
/// capacity, and visual geometry.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum PartParams {
    #[default]
    None,
    Decoupler {
        diameter: f32,
    },
    Adapter {
        diameter: f32,
        target_diameter: f32,
    },
    Tank {
        diameter: f32,
        length: f32,
    },
    /// Stationed-loft fuselage (`docs/gameplay/construction.md` §4.2). High-level
    /// airliner params; [`crate::fuselage_mesh`] turns them into the skin.
    Fuselage {
        length: f32,
        max_width: f32,
        max_height: f32,
        /// Superellipse roundness `∈ [0, 1]`: `1` round, `0` boxy.
        roundness: f32,
        /// Fraction of length spent on the parametric nose taper.
        nose_fraction: f32,
        /// Nose profile `∈ [0,1]`: `0` cone, `1` rounded radome.
        nose_bluntness: f32,
        /// Fraction of length spent on the tailcone neck.
        tail_fraction: f32,
        /// Nose centerline droop, metres.
        nose_droop: f32,
        /// Tail centerline upsweep, metres.
        tail_upsweep: f32,
        /// Diameter the tailcone necks to, metres (`0` → closes to a point).
        tail_tip_diameter: f32,
        /// Tail tip profile `∈ [0,1]`: `0` cone, `1` rounded ogive. Defaults
        /// (for blueprints saved before this field existed) to a rounded tail.
        #[serde(default = "default_tail_bluntness")]
        tail_bluntness: f32,
    },
    Wing {
        span: f32,
        root_chord: f32,
        tip_chord: f32,
        /// Leading-edge sweep, radians.
        sweep: f32,
        /// Dihedral, radians.
        dihedral: f32,
        /// Max airfoil thickness as a fraction of local chord.
        thickness: f32,
        /// Mounting incidence, radians.
        incidence: f32,
        /// Trailing-edge control surfaces (ailerons / elevator / rudder).
        /// Defaults empty so blueprints saved before control surfaces existed
        /// load as plain lifting surfaces.
        #[serde(default)]
        control_surfaces: Vec<ControlSurface>,
    },
    Gear {
        /// Length of each strut from the host skin to the wheel, metres.
        strut_length: f32,
        /// Wheel radius, metres.
        wheel_radius: f32,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PartBlueprint {
    pub catalog_id: CatalogId,
    #[serde(default)]
    pub params: PartParams,
    /// Per-instance selected resource pools in each resource's native unit
    /// (litres for fluids, kWh for electricity). `None` means "use the
    /// catalog default storage loadout"; `Some(map)` means exactly the
    /// selected resources in the map are active.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_resource_amounts",
        deserialize_with = "deserialize_resource_amounts"
    )]
    pub resources: Option<HashMap<Resource, f32>>,
}

fn serialize_resource_amounts<S>(
    amounts: &Option<HashMap<Resource, f32>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match amounts {
        Some(amounts) => amounts.serialize(serializer),
        None => serializer.serialize_none(),
    }
}

fn deserialize_resource_amounts<'de, D>(
    deserializer: D,
) -> Result<Option<HashMap<Resource, f32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    HashMap::<Resource, f32>::deserialize(deserializer).map(Some)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Connection {
    pub parent: usize,
    pub parent_node: String,
    pub child: usize,
    pub child_node: String,
}

/// A surface / footprint placement in the serialized blueprint: `child`
/// sits on `parent`'s skin at `(station, angle)` (see [`SurfaceMount`]).
/// Kept separate from [`Connection`] so the end-node stack format is
/// untouched and old saves load unchanged.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SurfaceConnection {
    pub parent: usize,
    pub child: usize,
    #[serde(default)]
    pub kind: SurfaceMountKind,
    pub station: f32,
    pub angle: f32,
    /// KSP symmetry group id. All surface mounts sharing an id form one
    /// linked group; the member with the lowest `child` index is the
    /// primary, the rest are mirror counterparts. `None` = a standalone
    /// (1×) mount. The game ignores this; only the editor re-links groups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symmetry_group: Option<u32>,
}

/// Default editor build layout persisted with a ship. `Vertical` is the
/// rocket / VAB stack; `Horizontal` lays the craft down like KSP's SPH
/// (aircraft). Purely editorial — it controls how the editor *displays* the
/// ship on load; a flight craft's attitude comes from its spawn scenario,
/// not this. Persisted so a plane reopens horizontal.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BuildLayout {
    #[default]
    Vertical,
    Horizontal,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ShipBlueprint {
    pub name: String,
    pub root: usize,
    pub parts: Vec<PartBlueprint>,
    pub connections: Vec<Connection>,
    /// Surface-mounted children (wings today). Defaults empty so saves
    /// written before wings existed — and any all-stack rocket — load with
    /// no extra keys.
    #[serde(default)]
    pub surface_mounts: Vec<SurfaceConnection>,
    /// Default editor build layout. `None` on saves written before this
    /// field existed; the editor then infers it (horizontal for a winged
    /// craft, vertical otherwise) on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<BuildLayout>,
}

impl ShipBlueprint {
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }

    pub fn from_ron(s: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(s)
    }

    /// Spawn a single part into the world without attaching it or
    /// creating a `Ship` entity. Used by the editor to add parts
    /// incrementally.
    pub fn spawn_part(
        commands: &mut Commands,
        catalog: &PartCatalog,
        catalog_id: &str,
        params: PartParams,
        resource_amounts: Option<HashMap<Resource, f32>>,
    ) -> Result<Entity, CatalogError> {
        let entry = catalog.resolve(catalog_id)?;
        check_params_match(catalog_id, entry, &params)?;
        check_resource_amounts_allowed(catalog_id, entry, &resource_amounts)?;
        let mut ec = commands.spawn_empty();
        insert_part(&mut ec, catalog_id, entry, &params, &resource_amounts);
        Ok(ec.id())
    }

    /// Spawn the blueprint into the world, returning the `Ship` entity.
    /// Validates every part against the catalog up front, so a single
    /// bad reference fails the whole load instead of half-spawning.
    pub fn spawn(
        &self,
        commands: &mut Commands,
        catalog: &PartCatalog,
    ) -> Result<Entity, CatalogError> {
        for pb in &self.parts {
            let entry = catalog.resolve(&pb.catalog_id)?;
            check_params_match(&pb.catalog_id, entry, &pb.params)?;
            check_resource_amounts_allowed(&pb.catalog_id, entry, &pb.resources)?;
        }

        let ids: Vec<Entity> = (0..self.parts.len())
            .map(|_| commands.spawn_empty().id())
            .collect();

        for (i, pb) in self.parts.iter().enumerate() {
            let entry = catalog.resolve(&pb.catalog_id).expect("validated above");
            let mut ec = commands.entity(ids[i]);
            insert_part(&mut ec, &pb.catalog_id, entry, &pb.params, &pb.resources);
        }

        for c in &self.connections {
            commands.entity(ids[c.child]).insert(Attachment {
                parent: ids[c.parent],
                parent_node: c.parent_node.clone(),
                my_node: c.child_node.clone(),
            });
        }

        // Lowest child index in each symmetry group is the primary.
        let mut group_primary: HashMap<u32, usize> = HashMap::new();
        for m in &self.surface_mounts {
            if let Some(g) = m.symmetry_group {
                let entry = group_primary.entry(g).or_insert(m.child);
                if m.child < *entry {
                    *entry = m.child;
                }
            }
        }

        for m in &self.surface_mounts {
            let mut ec = commands.entity(ids[m.child]);
            ec.insert(SurfaceMount {
                parent: ids[m.parent],
                kind: m.kind,
                station: m.station,
                angle: m.angle,
            });
            if let Some(g) = m.symmetry_group {
                let role = if group_primary.get(&g) == Some(&m.child) {
                    SymmetryRole::Primary
                } else {
                    SymmetryRole::Mirror
                };
                ec.insert(SymmetryGroup { id: g, role });
            }
        }

        let root = ids[self.root];
        Ok(commands
            .spawn(Ship {
                name: self.name.clone(),
                root,
            })
            .id())
    }
}

/// Verify that the [`PartParams`] variant matches the catalog kind. Pure
/// catalog kinds (Pod, Engine) require [`PartParams::None`]; parametric
/// kinds require their matching variant.
pub fn check_params_match(
    id: &str,
    entry: &CatalogEntry,
    params: &PartParams,
) -> Result<(), CatalogError> {
    match (entry, params) {
        (CatalogEntry::Pod(_), PartParams::None) => Ok(()),
        (CatalogEntry::Engine(_), PartParams::None) => Ok(()),
        (CatalogEntry::Intake(_), PartParams::None) => Ok(()),
        (CatalogEntry::Decoupler(_), PartParams::Decoupler { .. }) => Ok(()),
        (CatalogEntry::Adapter(_), PartParams::Adapter { .. }) => Ok(()),
        (CatalogEntry::Tank(_), PartParams::Tank { .. }) => Ok(()),
        (CatalogEntry::Fuselage(_), PartParams::Fuselage { .. }) => Ok(()),
        (CatalogEntry::Wing(_), PartParams::Wing { .. }) => Ok(()),
        (CatalogEntry::Gear(_), PartParams::Gear { .. }) => Ok(()),
        _ => Err(CatalogError::ParamMismatch {
            id: id.to_string(),
            kind: entry.kind_name(),
        }),
    }
}

pub fn check_resource_amounts_allowed(
    id: &str,
    entry: &CatalogEntry,
    resource_amounts: &Option<HashMap<Resource, f32>>,
) -> Result<(), CatalogError> {
    let Some(resource_amounts) = resource_amounts else {
        return Ok(());
    };
    for resource in resource_amounts.keys() {
        if !entry
            .storage_options()
            .iter()
            .any(|option| option.resource == *resource)
        {
            return Err(CatalogError::ResourceNotAllowed {
                id: id.to_string(),
                resource: *resource,
            });
        }
    }
    Ok(())
}

pub fn resource_capacity_for(
    entry: &CatalogEntry,
    params: &PartParams,
    resource: Resource,
) -> Option<f32> {
    entry
        .storage_options()
        .iter()
        .find(|option| option.resource == resource)
        .map(|option| storage_capacity_for(entry, params, option))
        .filter(|capacity| *capacity > 0.0)
}

fn insert_part(
    ec: &mut EntityCommands,
    catalog_id: &str,
    entry: &CatalogEntry,
    params: &PartParams,
    resource_amounts: &Option<HashMap<Resource, f32>>,
) {
    let nodes = nodes_for(entry, params);
    let pools = pools_for(entry, params, resource_amounts);

    ec.insert((
        Part,
        AttachNodes { nodes },
        PartResources { pools },
        CatalogRef {
            id: catalog_id.to_string(),
        },
    ));

    match (entry, params) {
        (CatalogEntry::Pod(p), _) => {
            ec.insert((
                CommandPod {
                    model: p.display_name.clone(),
                    geometry: p.geometry,
                    diameter: p.diameter,
                    dry_mass: p.dry_mass,
                    reaction_wheel_torque: p.reaction_wheel_torque,
                },
                ReactionWheel {
                    max_torque: p.reaction_wheel_torque,
                },
                FuelCrossfeed::default(),
            ));
        }
        (CatalogEntry::Engine(e), _) => {
            ec.insert((
                Engine {
                    model: e.display_name.clone(),
                    geometry: e.geometry,
                    optimized_for: e.optimized_for,
                    requires_atmosphere: e.requires_atmosphere,
                    intake_requirement: e.intake_requirement,
                    builtin_intake: e.builtin_intake,
                    diameter: e.diameter,
                    thrust: e.thrust,
                    isp: e.isp,
                    sea_level_isp: e.sea_level_isp,
                    dry_mass: e.dry_mass,
                    reactants: e.reactants.clone(),
                    power_draw_kw: e.power_draw_kw,
                    gimbal_range_deg: e.gimbal_range_deg,
                },
                EngineActivation::default(),
                FuelCrossfeed::default(),
                Shroudable,
                EngineThrust::default(),
            ));
        }
        (CatalogEntry::Intake(i), _) => {
            ec.insert((
                AirIntake {
                    model: i.display_name.clone(),
                    diameter: i.diameter,
                    length: i.length,
                    dry_mass: i.dry_mass,
                    capture: i.capture,
                },
                FuelCrossfeed::default(),
                PartMaterial::default(),
            ));
        }
        (CatalogEntry::Decoupler(d), PartParams::Decoupler { diameter }) => {
            let dry_mass = d.mass_per_diameter * *diameter;
            let ejection_impulse = d.ejection_impulse_per_diameter * *diameter;
            ec.insert((
                Decoupler {
                    diameter: *diameter,
                    ejection_impulse,
                    dry_mass,
                },
                FuelCrossfeed { enabled: false },
                ShroudProvider,
                PartMaterial::default(),
            ));
        }
        (
            CatalogEntry::Adapter(a),
            PartParams::Adapter {
                diameter,
                target_diameter,
            },
        ) => {
            let dry_mass = a.wall_mass_per_m2 * adapter_surface_area(*diameter, *target_diameter);
            ec.insert((
                Adapter {
                    diameter: *diameter,
                    target_diameter: *target_diameter,
                    dry_mass,
                },
                FuelCrossfeed::default(),
                PartMaterial::default(),
            ));
        }
        (CatalogEntry::Tank(t), PartParams::Tank { diameter, length }) => {
            let dry_mass = t.wall_mass_per_m2 * tank_surface_area(*diameter, *length);
            ec.insert((
                FuelTank {
                    diameter: *diameter,
                    length: *length,
                    dry_mass,
                },
                FuelCrossfeed::default(),
                PartMaterial::default(),
            ));
        }
        (
            CatalogEntry::Fuselage(f),
            PartParams::Fuselage {
                length,
                max_width,
                max_height,
                roundness,
                nose_fraction,
                nose_bluntness,
                tail_fraction,
                nose_droop,
                tail_upsweep,
                tail_tip_diameter,
                tail_bluntness,
            },
        ) => {
            let dry_mass =
                f.wall_mass_per_m2 * fuselage_surface_area(*length, *max_width, *max_height);
            // The airliner body is essentially circular, so the cylindrical
            // ship_part panel shader maps cleanly — the fuselage flows through
            // the standard body-of-revolution visual path with `PartMaterial`,
            // like a tank. (A non-circular / double-bubble loft would need
            // loft-derived UVs — `docs/gameplay/construction.md` §2 — deferred.)
            ec.insert((
                Fuselage {
                    length: *length,
                    max_width: *max_width,
                    max_height: *max_height,
                    roundness: *roundness,
                    nose_fraction: *nose_fraction,
                    nose_bluntness: *nose_bluntness,
                    tail_fraction: *tail_fraction,
                    nose_droop: *nose_droop,
                    tail_upsweep: *tail_upsweep,
                    tail_tip_diameter: *tail_tip_diameter,
                    tail_bluntness: *tail_bluntness,
                    dry_mass,
                },
                FuelCrossfeed::default(),
                PartMaterial::default(),
            ));
        }
        (
            CatalogEntry::Wing(w),
            PartParams::Wing {
                span,
                root_chord,
                tip_chord,
                sweep,
                dihedral,
                thickness,
                incidence,
                control_surfaces,
            },
        ) => {
            // Single-panel structural mass; a mirrored mount doubles it at
            // aggregation time (the symmetry lives on `SurfaceMount`, not
            // here). Wings deliberately get no `PartMaterial`: the
            // ship_part shader maps detail by cylindrical coords and would
            // smear across a lofted skin (`docs/gameplay/construction.md` §2).
            let dry_mass = w.mass_per_m2 * wing_panel_area(*span, *root_chord, *tip_chord);
            ec.insert(Wing {
                span: *span,
                root_chord: *root_chord,
                tip_chord: *tip_chord,
                sweep: *sweep,
                dihedral: *dihedral,
                thickness: *thickness,
                incidence: *incidence,
                dry_mass,
                control_surfaces: control_surfaces.clone(),
            });
        }
        (
            CatalogEntry::Gear(g),
            PartParams::Gear {
                strut_length,
                wheel_radius,
            },
        ) => {
            // A self-contained gearbox: one entity draws every leg. No
            // `PartMaterial` (the ship_part shader maps detail cylindrically and
            // would smear across the struts/wheels — same reasoning as wings).
            // `track_fraction` is carried from the catalog so the mesh + mass
            // know the leg count without a catalog lookup.
            let dry_mass = gear_dry_mass(g, *strut_length);
            ec.insert(Gear {
                strut_length: *strut_length,
                wheel_radius: *wheel_radius,
                track_fraction: g.track_fraction,
                wheels_per_leg: g.wheels_per_leg,
                dry_mass,
            });
        }
        _ => unreachable!("check_params_match guarantees variant match"),
    }
}

/// Initial attach-node layout for a part. Parametric kinds get
/// placeholder diameters that `sizing::propagate_node_sizes` overwrites
/// once the part is attached to a parent.
pub fn nodes_for(entry: &CatalogEntry, params: &PartParams) -> HashMap<NodeId, AttachNode> {
    let mut nodes = HashMap::new();
    match (entry, params) {
        (CatalogEntry::Pod(p), _) => {
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: p.diameter,
                    offset: Vec3::new(0.0, -p.diameter * p.geometry.length_factor(), 0.0),
                },
            );
        }
        (CatalogEntry::Engine(e), _) => {
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: e.diameter,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: e.diameter,
                    offset: Vec3::new(0.0, -e.diameter * 0.9, 0.0),
                },
            );
        }
        (CatalogEntry::Intake(i), _) => {
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: i.diameter,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: i.diameter,
                    offset: Vec3::new(0.0, -i.length, 0.0),
                },
            );
        }
        (CatalogEntry::Decoupler(_), PartParams::Decoupler { diameter }) => {
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: *diameter,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: *diameter,
                    offset: Vec3::new(0.0, -0.2, 0.0),
                },
            );
        }
        (
            CatalogEntry::Adapter(_),
            PartParams::Adapter {
                diameter,
                target_diameter,
            },
        ) => {
            let h = ((*diameter + *target_diameter) * 0.5).max(0.4);
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: *diameter,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: *target_diameter,
                    offset: Vec3::new(0.0, -h, 0.0),
                },
            );
        }
        (CatalogEntry::Tank(_), PartParams::Tank { diameter, length }) => {
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: *diameter,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: *diameter,
                    offset: Vec3::new(0.0, -*length, 0.0),
                },
            );
        }
        (
            CatalogEntry::Fuselage(_),
            PartParams::Fuselage {
                length,
                max_width,
                tail_tip_diameter,
                ..
            },
        ) => {
            // `top` is the barrel (overridable by the parent like a tank);
            // `bottom` is the necked tail tip at −length.
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: *max_width,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: *tail_tip_diameter,
                    offset: Vec3::new(0.0, -*length, 0.0),
                },
            );
        }
        _ => {}
    }
    nodes
}

/// Resource pools at spawn — capacities computed from catalog × params,
/// amounts taken from `resource_amounts` (or defaulted full when omitted).
pub fn pools_for(
    entry: &CatalogEntry,
    params: &PartParams,
    resource_amounts: &Option<HashMap<Resource, f32>>,
) -> HashMap<Resource, ResourcePool> {
    let mut pools = HashMap::new();
    for option in entry.storage_options() {
        let capacity = storage_capacity_for(entry, params, option);
        if capacity <= 0.0 {
            continue;
        }
        let active = match resource_amounts {
            Some(amounts) => amounts.contains_key(&option.resource),
            None => option.default_enabled,
        };
        if !active {
            continue;
        }
        let amount = resource_amounts
            .as_ref()
            .and_then(|amounts| amounts.get(&option.resource).copied())
            .unwrap_or(capacity * option.default_fill_fraction)
            .clamp(0.0, capacity);
        pools.insert(option.resource, ResourcePool { capacity, amount });
    }
    pools
}

pub fn storage_capacity_for(
    entry: &CatalogEntry,
    params: &PartParams,
    option: &crate::catalog::ResourceStorageSpec,
) -> f32 {
    option.units + option.units_per_m3 * storage_volume_for(entry, params)
}

fn storage_volume_for(entry: &CatalogEntry, params: &PartParams) -> f32 {
    match (entry, params) {
        (CatalogEntry::Tank(_), PartParams::Tank { diameter, length }) => {
            tank_volume(*diameter, *length)
        }
        // A future wet/role-filled fuselage scales capacity with its enclosed
        // loft volume, just as a tank does with its cylinder.
        (
            CatalogEntry::Fuselage(_),
            PartParams::Fuselage {
                length,
                max_width,
                max_height,
                ..
            },
        ) => fuselage_volume(*length, *max_width, *max_height),
        // Wet wings store fuel in the integral wing box; capacity scales with
        // the panel's internal volume just as a tank's does with its cylinder.
        (
            CatalogEntry::Wing(_),
            PartParams::Wing {
                span,
                root_chord,
                tip_chord,
                thickness,
                ..
            },
        ) => wing_volume(*span, *root_chord, *tip_chord, *thickness),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::PartCatalog;

    /// The shipped demo aircraft must stay loadable: parse `ships/meridian.ron`
    /// against the catalog, confirm its KSP-symmetry surface mounts survive
    /// the round-trip, and that stats derive a positive wing area. Guards the
    /// hand-authored RON (`surface_mounts` shape, `symmetry_group`) from drift —
    /// including the loft-fuselage-root + inline-cockpit layout and the
    /// `stabilizer`-part empennage.
    #[test]
    fn meridian_sample_loads_with_wings() {
        let cat = PartCatalog::load_from_str(include_str!("../../../../assets/parts.ron"))
            .expect("parse parts.ron");
        let bp = ShipBlueprint::from_ron(include_str!("../../../../ships/meridian.ron"))
            .expect("parse meridian.ron");
        // inline cockpit + main wing ×2 + tailplane ×2 + fin + nacelle ×4 +
        // nose gear + main gear = 12 surface mounts (everything rides the loft).
        assert_eq!(bp.surface_mounts.len(), 12);
        // Four linked groups: main wings, tailplanes, inboard + outboard nacelles.
        let groups: std::collections::HashSet<u32> = bp
            .surface_mounts
            .iter()
            .filter_map(|m| m.symmetry_group)
            .collect();
        assert_eq!(
            groups.len(),
            4,
            "main / tail / inboard + outboard nacelle groups"
        );
        let s = bp.stats(&cat).expect("meridian stats");
        assert!(s.wing_area_m2 > 0.0, "meridian should report wing area");
        assert!(s.mean_aerodynamic_chord_m > 0.0);
        assert!(s.dry_mass_kg > 0.0);
    }

    /// A structural fuselage carries no propellant; a wet wing's kerosene
    /// capacity scales with its internal volume. Guards the two new catalog
    /// parts and the wing-volume storage path.
    #[test]
    fn structural_fuselage_is_dry_and_wet_wing_holds_fuel() {
        let cat = PartCatalog::load_from_str(include_str!("../../../../assets/parts.ron"))
            .expect("parse parts.ron");

        // The structural fuselage resolves as a Tank with no storage options,
        // so it composes zero resource pools at any size.
        let fuselage = cat.resolve("fuselage_structural").expect("fuselage");
        assert!(fuselage.storage_options().is_empty());
        let fuselage_pools = pools_for(
            fuselage,
            &PartParams::Tank {
                diameter: 1.5,
                length: 9.0,
            },
            &None,
        );
        assert!(
            fuselage_pools.is_empty(),
            "structural fuselage stores nothing"
        );

        // A wet wing fills its integral box with kerosene; a bigger wing holds
        // proportionally more.
        let wet = cat.resolve("wing_wet").expect("wet wing");
        let wing_params = |span: f32| PartParams::Wing {
            span,
            root_chord: 2.4,
            tip_chord: 0.9,
            sweep: 0.44,
            dihedral: 0.05,
            thickness: 0.11,
            incidence: 0.0,
            control_surfaces: Vec::new(),
        };
        let small = pools_for(wet, &wing_params(5.5), &None);
        let large = pools_for(wet, &wing_params(11.0), &None);
        let small_kero = small.get(&Resource::Kerosene).expect("kerosene pool");
        let large_kero = large.get(&Resource::Kerosene).expect("kerosene pool");
        assert!(small_kero.capacity > 0.0);
        assert!(
            large_kero.capacity > small_kero.capacity,
            "doubling span should raise wet-wing capacity",
        );
    }

    /// The shipped airliner loads with fuel in its wings and none in the
    /// structural body. Guards `ships/meridian.ron`.
    #[test]
    fn meridian_carries_wing_fuel() {
        let cat = PartCatalog::load_from_str(include_str!("../../../../assets/parts.ron"))
            .expect("parse parts.ron");
        let bp = ShipBlueprint::from_ron(include_str!("../../../../ships/meridian.ron"))
            .expect("parse meridian.ron");
        let s = bp.stats(&cat).expect("aircraft stats");
        let kero = s
            .resources
            .get(&Resource::Kerosene)
            .expect("aircraft should carry kerosene");
        assert!(kero.capacity > 0.0, "wet wings provide kerosene capacity");
        assert!(kero.mass_kg > 0.0, "wet wings spawn full of kerosene");
        assert!(s.wing_area_m2 > 0.0);
    }

    /// The shipped orbital rocket must stay loadable and actually be
    /// orbital-class: parse `ships/atlas.ron` against the catalog, confirm it
    /// carries methalox propellant, lifts off (full-thrust accel > Thalos
    /// surface gravity), and has enough Δv for low Thalos orbit (~7 km/s).
    /// Also guards that its engines are gimballed — the ascent is unflyable
    /// without thrust vectoring. Numbers here are conservative single-stack
    /// proxies; the real staged Δv is higher.
    #[test]
    fn atlas_sample_is_an_orbital_rocket() {
        const G0: f64 = 9.80665;
        const THALOS_SURFACE_G: f64 = 9.06;
        const THALOS_ORBIT_DV: f64 = 7000.0;

        let cat = PartCatalog::load_from_str(include_str!("../../../../assets/parts.ron"))
            .expect("parse parts.ron");
        let bp = ShipBlueprint::from_ron(include_str!("../../../../ships/atlas.ron"))
            .expect("parse atlas.ron");
        let s = bp.stats(&cat).expect("atlas stats");

        assert!(s.dry_mass_kg > 0.0 && s.propellant_mass_kg > 0.0);
        for r in [Resource::Methane, Resource::Lox] {
            let pool = s
                .resources
                .get(&r)
                .unwrap_or_else(|| panic!("carries {r:?}"));
            assert!(pool.mass_kg > 0.0, "spawns full of {r:?}");
        }

        // Liftoff: full thrust must beat weight at the launch (wet) mass.
        assert!(
            s.current_acceleration() > THALOS_SURFACE_G,
            "atlas can't lift off ({:.1} m/s² ≤ g {THALOS_SURFACE_G})",
            s.current_acceleration()
        );

        // Pad TWR, honestly: staging ignites only the bottom booster, and at
        // 1 atm its thrust is derated by the sea-level/vacuum Isp ratio
        // (nozzle back-pressure — mass flow fixed, thrust falls linearly with
        // ambient pressure). The shipped rocket must fly a sane ascent, not
        // crawl off the pad.
        let CatalogEntry::Engine(typhon) = cat.resolve("typhon").unwrap() else {
            panic!("typhon should be an engine");
        };
        let sl_factor = typhon
            .sea_level_isp
            .map(|sl| sl as f64 / typhon.isp as f64)
            .unwrap_or(1.0);
        let pad_twr = typhon.thrust as f64 * sl_factor / (s.wet_mass_kg() * THALOS_SURFACE_G);
        assert!(
            pad_twr > 1.2,
            "atlas stage-1 pad TWR {pad_twr:.2} too low to leave the pad cleanly"
        );

        // Δv (single-stack lower bound: combined Isp over the full mass ratio —
        // staging only adds to this) must clear low Thalos orbit.
        let dv = s.combined_isp_s * G0 * (s.wet_mass_kg() / s.dry_mass_kg).ln();
        assert!(
            dv > THALOS_ORBIT_DV,
            "atlas Δv {dv:.0} m/s below Thalos orbit needs (~{THALOS_ORBIT_DV})"
        );

        // Both stage engines must gimbal, or the ascent can't be steered.
        for id in ["typhon", "boreas"] {
            let CatalogEntry::Engine(e) = cat
                .resolve(id)
                .unwrap_or_else(|e| panic!("{id} in catalog: {e:?}"))
            else {
                panic!("{id} should be an engine");
            };
            assert!(
                e.gimbal_range_deg > 0.0,
                "{id} must be gimballed for ascent steering"
            );
        }
    }
}

/// Default per-instance params for adding a fresh part of `entry`'s kind.
/// Parametric defaults assume a 2.5 m parent — `sizing.rs` overrides once
/// attached.
pub fn default_params_for(entry: &CatalogEntry) -> PartParams {
    match entry {
        CatalogEntry::Pod(_) | CatalogEntry::Engine(_) | CatalogEntry::Intake(_) => {
            PartParams::None
        }
        CatalogEntry::Decoupler(_) => PartParams::Decoupler { diameter: 2.5 },
        CatalogEntry::Adapter(_) => PartParams::Adapter {
            diameter: 2.5,
            target_diameter: 4.0,
        },
        CatalogEntry::Tank(_) => PartParams::Tank {
            diameter: 2.5,
            length: 3.0,
        },
        // A mid-size airliner barrel with an upswept tailcone — a sensible
        // starting body the inspector then tunes.
        CatalogEntry::Fuselage(_) => PartParams::Fuselage {
            length: 14.0,
            max_width: 2.5,
            max_height: 2.5,
            roundness: 1.0,
            nose_fraction: 0.14,
            nose_bluntness: 0.8,
            tail_fraction: 0.34,
            nose_droop: 0.1,
            tail_upsweep: 0.9,
            tail_tip_diameter: 0.0,
            tail_bluntness: 0.35,
        },
        // Both wing roles spawn the same `Wing` kind; the role only picks the
        // starting geometry so a freshly-placed part is sane without tuning.
        // A `Stabilizer` is a small, dry, ~0°-incidence trim surface — equally
        // a tailplane or a fin, since orientation is decided by the mount
        // azimuth, not these params.
        // Control surfaces depend on the mount azimuth (elevator vs rudder),
        // which isn't known until placement, so they start empty here and the
        // editor fills role-appropriate defaults once the wing is mounted (see
        // `default_control_surfaces`).
        CatalogEntry::Wing(spec) => match spec.role {
            WingRole::Lift => PartParams::Wing {
                span: 5.0,
                root_chord: 2.5,
                tip_chord: 1.0,
                sweep: 20.0_f32.to_radians(),
                dihedral: 3.0_f32.to_radians(),
                thickness: 0.12,
                incidence: 0.0,
                control_surfaces: Vec::new(),
            },
            WingRole::Stabilizer => PartParams::Wing {
                span: 2.0,
                root_chord: 1.4,
                tip_chord: 0.7,
                sweep: 12.0_f32.to_radians(),
                dihedral: 0.0,
                thickness: 0.10,
                incidence: 0.0,
                control_surfaces: Vec::new(),
            },
        },
        CatalogEntry::Gear(g) => PartParams::Gear {
            strut_length: g.default_strut_length,
            wheel_radius: g.default_wheel_radius,
        },
    }
}
