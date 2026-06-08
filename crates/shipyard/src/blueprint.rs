use crate::attach::{
    AttachNode, AttachNodes, Attachment, MountSymmetry, NodeId, Ship, SurfaceMount,
    SurfaceMountKind,
};
use crate::catalog::{
    CatalogEntry, CatalogError, CatalogId, CatalogRef, PartCatalog, adapter_surface_area,
    tank_surface_area, tank_volume, wing_panel_area,
};
use crate::part::{
    Adapter, AirIntake, CommandPod, Decoupler, Engine, EngineActivation, EngineThrust,
    FuelCrossfeed, FuelTank, Part, PartMaterial, ReactionWheel, ShroudProvider, Shroudable, Wing,
};
use crate::resource::{PartResources, Resource, ResourcePool};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-instance parameters that the catalog cannot derive. Pure-catalog
/// kinds (Pod, Engine) carry [`PartParams::None`]. Parametric kinds carry
/// the dimensions the user picks; the catalog turns those into mass,
/// capacity, and visual geometry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    #[serde(default)]
    pub symmetry: MountSymmetry,
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

        for m in &self.surface_mounts {
            commands.entity(ids[m.child]).insert(SurfaceMount {
                parent: ids[m.parent],
                kind: m.kind,
                station: m.station,
                angle: m.angle,
                symmetry: m.symmetry,
            });
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
        (CatalogEntry::Wing(_), PartParams::Wing { .. }) => Ok(()),
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
                    requires_atmosphere: e.requires_atmosphere,
                    intake_requirement: e.intake_requirement,
                    builtin_intake: e.builtin_intake,
                    diameter: e.diameter,
                    thrust: e.thrust,
                    isp: e.isp,
                    dry_mass: e.dry_mass,
                    reactants: e.reactants.clone(),
                    power_draw_kw: e.power_draw_kw,
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
            CatalogEntry::Wing(w),
            PartParams::Wing {
                span,
                root_chord,
                tip_chord,
                sweep,
                dihedral,
                thickness,
                incidence,
            },
        ) => {
            // Single-panel structural mass; a mirrored mount doubles it at
            // aggregation time (the symmetry lives on `SurfaceMount`, not
            // here). Wings deliberately get no `PartMaterial`: the
            // ship_part shader maps detail by cylindrical coords and would
            // smear across a lofted skin (`docs/construction.md` §2).
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
                    offset: Vec3::new(0.0, -p.diameter * 0.9, 0.0),
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
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::PartCatalog;

    /// The shipped demo aircraft must stay loadable: parse `ships/skyhawk.ron`
    /// against the catalog, confirm its three surface-mounted wings survive
    /// the round-trip, and that stats derive a positive wing area. Guards the
    /// hand-authored RON (enum spelling, `surface_mounts` shape) from drift.
    #[test]
    fn skyhawk_sample_loads_with_wings() {
        let cat = PartCatalog::load_from_str(include_str!("../../../assets/parts.ron"))
            .expect("parse parts.ron");
        let bp = ShipBlueprint::from_ron(include_str!("../../../ships/skyhawk.ron"))
            .expect("parse skyhawk.ron");
        assert_eq!(
            bp.surface_mounts.len(),
            4,
            "main wing + tailplane + fin + jet nacelle"
        );
        let s = bp.stats(&cat).expect("skyhawk stats");
        assert!(s.wing_area_m2 > 0.0, "skyhawk should report wing area");
        assert!(s.mean_aerodynamic_chord_m > 0.0);
        assert!(s.dry_mass_kg > 0.0);
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
        CatalogEntry::Wing(_) => PartParams::Wing {
            span: 5.0,
            root_chord: 2.5,
            tip_chord: 1.0,
            sweep: 20.0_f32.to_radians(),
            dihedral: 3.0_f32.to_radians(),
            thickness: 0.12,
            incidence: 0.0,
        },
    }
}
