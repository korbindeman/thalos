use crate::attach::{AttachNode, AttachNodes, Attachment, NodeId, Ship};
use crate::catalog::{
    CatalogEntry, CatalogError, CatalogId, CatalogRef, PartCatalog, adapter_surface_area,
    tank_surface_area, tank_volume,
};
use crate::part::{
    Adapter, CommandPod, Decoupler, Engine, EngineActivation, EngineThrust, FuelCrossfeed,
    FuelTank, Part, PartMaterial, ReactionWheel, ShroudProvider, Shroudable,
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
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PartBlueprint {
    pub catalog_id: CatalogId,
    #[serde(default)]
    pub params: PartParams,
    /// Per-instance resource amounts in each resource's native unit
    /// (litres for fluids, kWh for electricity). Capacities are computed
    /// from the catalog × params at spawn — a blueprint that omits a
    /// resource gets the catalog's default amount (typically full).
    #[serde(default)]
    pub resources: HashMap<Resource, f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Connection {
    pub parent: usize,
    pub parent_node: String,
    pub child: usize,
    pub child_node: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ShipBlueprint {
    pub name: String,
    pub root: usize,
    pub parts: Vec<PartBlueprint>,
    pub connections: Vec<Connection>,
}

impl ShipBlueprint {
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).map_err(Into::into)
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
        resource_amounts: HashMap<Resource, f32>,
    ) -> Result<Entity, CatalogError> {
        let entry = catalog.resolve(catalog_id)?;
        check_params_match(catalog_id, entry, &params)?;
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
        (CatalogEntry::Decoupler(_), PartParams::Decoupler { .. }) => Ok(()),
        (CatalogEntry::Adapter(_), PartParams::Adapter { .. }) => Ok(()),
        (CatalogEntry::Tank(_), PartParams::Tank { .. }) => Ok(()),
        _ => Err(CatalogError::ParamMismatch {
            id: id.to_string(),
            kind: entry.kind_name(),
        }),
    }
}

fn insert_part(
    ec: &mut EntityCommands,
    catalog_id: &str,
    entry: &CatalogEntry,
    params: &PartParams,
    resource_amounts: &HashMap<Resource, f32>,
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
    resource_amounts: &HashMap<Resource, f32>,
) -> HashMap<Resource, ResourcePool> {
    let mut pools = HashMap::new();
    match (entry, params) {
        (CatalogEntry::Pod(p), _) => {
            let cap = p.base_electricity_kwh;
            if cap > 0.0 {
                let amount = resource_amounts
                    .get(&Resource::Electricity)
                    .copied()
                    .unwrap_or(cap);
                pools.insert(
                    Resource::Electricity,
                    ResourcePool {
                        capacity: cap,
                        amount,
                    },
                );
            }
        }
        (CatalogEntry::Tank(t), PartParams::Tank { diameter, length }) => {
            let v = tank_volume(*diameter, *length);
            let cap_ch4 = t.methane_l_per_m3 * v;
            let cap_lox = t.lox_l_per_m3 * v;
            let amt_ch4 = resource_amounts
                .get(&Resource::Methane)
                .copied()
                .unwrap_or(cap_ch4);
            let amt_lox = resource_amounts
                .get(&Resource::Lox)
                .copied()
                .unwrap_or(cap_lox);
            pools.insert(
                Resource::Methane,
                ResourcePool {
                    capacity: cap_ch4,
                    amount: amt_ch4,
                },
            );
            pools.insert(
                Resource::Lox,
                ResourcePool {
                    capacity: cap_lox,
                    amount: amt_lox,
                },
            );
        }
        _ => {}
    }
    pools
}

/// Default per-instance params for adding a fresh part of `entry`'s kind.
/// Parametric defaults assume a 2.5 m parent — `sizing.rs` overrides once
/// attached.
pub fn default_params_for(entry: &CatalogEntry) -> PartParams {
    match entry {
        CatalogEntry::Pod(_) | CatalogEntry::Engine(_) => PartParams::None,
        CatalogEntry::Decoupler(_) => PartParams::Decoupler { diameter: 2.5 },
        CatalogEntry::Adapter(_) => PartParams::Adapter {
            diameter: 2.5,
            target_diameter: 4.0,
        },
        CatalogEntry::Tank(_) => PartParams::Tank {
            diameter: 2.5,
            length: 3.0,
        },
    }
}
