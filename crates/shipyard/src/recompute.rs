//! Keep parametric part state in sync with catalog formulas when the
//! per-instance dimensions mutate at runtime (inspector slider, tank
//! resize-arrow drag, parent diameter propagation).
//!
//! Without these, resizing a tank length would leave its [`FuelTank::dry_mass`]
//! and resource-pool capacities frozen at spawn-time values until the
//! next save/reload — visually consistent stage size, but mass and fuel
//! visibly out of date in the inspector.
//!
//! Each system is gated on `Changed<T>`. To prevent feedback (our
//! mutation re-firing the filter on the next frame), we only call
//! `DerefMut` when a value actually differs by more than a small epsilon.

use crate::catalog::{
    CatalogEntry, CatalogRef, PartCatalog, adapter_surface_area, tank_surface_area, tank_volume,
};
use crate::part::{Adapter, Decoupler, FuelTank};
use crate::resource::{PartResources, Resource};
use bevy::prelude::*;

/// Tolerance for "did this value actually change?" guards. Bigger than
/// `f32::EPSILON` so floating-point noise doesn't perpetually re-mutate;
/// smaller than any user-visible difference.
const EPS: f32 = 0.5;

pub fn recompute_decoupler_state(
    catalog: Option<Res<PartCatalog>>,
    mut q: Query<(&CatalogRef, &mut Decoupler), Changed<Decoupler>>,
) {
    let Some(catalog) = catalog else { return };
    for (cat_ref, mut dec) in q.iter_mut() {
        let Ok(CatalogEntry::Decoupler(d)) = catalog.resolve(&cat_ref.id) else {
            continue;
        };
        let new_mass = d.mass_per_diameter * dec.diameter;
        let new_imp = d.ejection_impulse_per_diameter * dec.diameter;
        if (dec.dry_mass - new_mass).abs() > EPS {
            dec.dry_mass = new_mass;
        }
        if (dec.ejection_impulse - new_imp).abs() > EPS {
            dec.ejection_impulse = new_imp;
        }
    }
}

pub fn recompute_adapter_state(
    catalog: Option<Res<PartCatalog>>,
    mut q: Query<(&CatalogRef, &mut Adapter), Changed<Adapter>>,
) {
    let Some(catalog) = catalog else { return };
    for (cat_ref, mut adp) in q.iter_mut() {
        let Ok(CatalogEntry::Adapter(a)) = catalog.resolve(&cat_ref.id) else {
            continue;
        };
        let new_mass = a.wall_mass_per_m2 * adapter_surface_area(adp.diameter, adp.target_diameter);
        if (adp.dry_mass - new_mass).abs() > EPS {
            adp.dry_mass = new_mass;
        }
    }
}

/// Tank state recompute: dry mass plus propellant capacities. When
/// capacity scales (length or diameter changed), each propellant amount
/// is rescaled at the same fill ratio so a full tank stays full and a
/// half-empty tank stays half-empty.
pub fn recompute_tank_state(
    catalog: Option<Res<PartCatalog>>,
    mut q: Query<(&CatalogRef, &mut FuelTank, &mut PartResources), Changed<FuelTank>>,
) {
    let Some(catalog) = catalog else { return };
    for (cat_ref, mut tank, mut res) in q.iter_mut() {
        let Ok(CatalogEntry::Tank(t)) = catalog.resolve(&cat_ref.id) else {
            continue;
        };
        let new_mass = t.wall_mass_per_m2 * tank_surface_area(tank.diameter, tank.length);
        if (tank.dry_mass - new_mass).abs() > EPS {
            tank.dry_mass = new_mass;
        }

        let v = tank_volume(tank.diameter, tank.length);
        rescale_pool(&mut res, Resource::Methane, t.methane_l_per_m3 * v);
        rescale_pool(&mut res, Resource::Lox, t.lox_l_per_m3 * v);
    }
}

fn rescale_pool(res: &mut PartResources, kind: Resource, new_capacity: f32) {
    let Some(p) = res.pools.get_mut(&kind) else {
        return;
    };
    if (p.capacity - new_capacity).abs() < EPS {
        return;
    }
    let ratio = if p.capacity > 0.0 {
        p.amount / p.capacity
    } else {
        1.0
    };
    p.capacity = new_capacity;
    p.amount = ratio * new_capacity;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourcePool;
    use std::collections::HashMap;

    #[test]
    fn rescale_pool_keeps_fill_ratio() {
        let mut res = PartResources {
            pools: HashMap::from([(
                Resource::Methane,
                ResourcePool {
                    capacity: 1000.0,
                    amount: 500.0, // 50% full
                },
            )]),
        };
        rescale_pool(&mut res, Resource::Methane, 4000.0);
        let p = res.pools.get(&Resource::Methane).unwrap();
        assert!((p.capacity - 4000.0).abs() < 1.0);
        assert!((p.amount - 2000.0).abs() < 1.0);
    }

    #[test]
    fn rescale_pool_full_stays_full() {
        let mut res = PartResources {
            pools: HashMap::from([(
                Resource::Lox,
                ResourcePool {
                    capacity: 100.0,
                    amount: 100.0,
                },
            )]),
        };
        rescale_pool(&mut res, Resource::Lox, 400.0);
        let p = res.pools.get(&Resource::Lox).unwrap();
        assert!((p.amount - 400.0).abs() < 1.0);
    }

    #[test]
    fn rescale_pool_empty_capacity_fills_full() {
        // Edge case: a pool spawned with 0 capacity (shouldn't happen
        // in practice, but the code path needs to be defined). Treat
        // as "fill it up" — better than NaN amounts.
        let mut res = PartResources {
            pools: HashMap::from([(
                Resource::Methane,
                ResourcePool {
                    capacity: 0.0,
                    amount: 0.0,
                },
            )]),
        };
        rescale_pool(&mut res, Resource::Methane, 1000.0);
        let p = res.pools.get(&Resource::Methane).unwrap();
        assert!((p.amount - 1000.0).abs() < 1.0);
    }

    #[test]
    fn rescale_pool_missing_kind_is_noop() {
        let mut res = PartResources::default();
        rescale_pool(&mut res, Resource::Methane, 1000.0);
        assert!(res.pools.is_empty());
    }
}
