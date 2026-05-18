use bevy::prelude::*;
use big_space::prelude::*;
use thalos_physics_canonical::types::BodyState;

use super::transforms::surface_body_to_world_orientation;
use super::types::{PlayerShip, RealSpaceBody, SolarSystemState, TidallyLocked};

pub const REAL_SPACE_CELL_SIZE_M: f32 = 1_000.0;

#[derive(Resource, Debug, Clone, Copy)]
pub struct RealSpaceRoot {
    pub entity: Entity,
}

pub fn real_space_grid() -> Grid {
    Grid::new(REAL_SPACE_CELL_SIZE_M, 0.0)
}

pub(super) fn setup_big_space(mut commands: Commands) {
    let root = commands
        .spawn((
            BigSpaceRootBundle {
                grid: real_space_grid(),
                ..default()
            },
            Visibility::Inherited,
            InheritedVisibility::default(),
            ViewVisibility::default(),
            Name::new("Pyros BigSpace"),
        ))
        .id();
    commands.insert_resource(RealSpaceRoot { entity: root });
}

pub(super) fn attach_ship_camera_to_big_space(
    mut commands: Commands,
    root: Res<RealSpaceRoot>,
    grid: Query<&Grid, With<BigSpace>>,
    cameras: Query<(Entity, &Transform), With<crate::camera::ShipCamera>>,
) {
    let Ok(root_grid) = grid.single() else {
        return;
    };
    for (entity, transform) in &cameras {
        let (cell, local) = root_grid.translation_to_grid(transform.translation.as_dvec3());
        commands.entity(entity).insert((
            cell,
            Transform::from_translation(local).with_rotation(transform.rotation),
            FloatingOrigin,
            ChildOf(root.entity),
        ));
    }
}

pub(super) fn attach_player_ship_to_big_space(
    mut commands: Commands,
    root: Res<RealSpaceRoot>,
    ships: Query<Entity, With<PlayerShip>>,
) {
    for entity in &ships {
        commands
            .entity(entity)
            .insert((CellCoord::ZERO, ChildOf(root.entity)));
    }
}

pub(super) fn update_real_space_body_positions(
    cache: Res<SolarSystemState>,
    grid: Query<&Grid, With<BigSpace>>,
    mut bodies: Query<(
        &RealSpaceBody,
        Option<&TidallyLocked>,
        &mut CellCoord,
        &mut Transform,
    )>,
) {
    let Some(states) = cache.states.as_deref() else {
        return;
    };
    let Ok(root_grid) = grid.single() else {
        return;
    };

    for (body, lock, mut cell, mut transform) in &mut bodies {
        let Some(state) = states.get(body.body_id) else {
            continue;
        };
        write_body_transform(state, lock, states, root_grid, &mut cell, &mut transform);
    }
}

fn write_body_transform(
    state: &BodyState,
    lock: Option<&TidallyLocked>,
    states: &[BodyState],
    grid: &Grid,
    cell: &mut CellCoord,
    transform: &mut Transform,
) {
    let (next_cell, local) = grid.translation_to_grid(state.position);
    *cell = next_cell;
    transform.translation = local;
    transform.rotation =
        surface_body_to_world_orientation(state.id, lock, states).unwrap_or_else(|| {
            warn!(
                "could not resolve surface orientation for body {}; falling back to ephemeris orientation",
                state.id,
            );
            state.orientation.as_quat().normalize()
        });
}

#[cfg(test)]
mod tests {
    use bevy::math::{DQuat, DVec3};
    use thalos_physics_canonical::body_trajectory_provider::BodyTrajectoryProvider;
    use thalos_physics_canonical::canonical::Epoch;
    use thalos_physics_canonical::parsing::load_solar_system_from_dir;
    use thalos_physics_canonical::patched_conics::PatchedConics;

    use super::*;

    #[test]
    fn real_space_body_positions_use_cells_not_large_transforms() {
        let grid = real_space_grid();
        let state = BodyState {
            id: 1,
            epoch: Epoch::ZERO,
            position: DVec3::new(1.5e11, 0.0, -2.0e9),
            velocity: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
            mass_kg: 1.0,
            gm: 1.0,
            radius_m: 1.0,
        };
        let mut cell = CellCoord::ZERO;
        let mut transform = Transform::default();

        write_body_transform(&state, None, &[state], &grid, &mut cell, &mut transform);

        assert!(cell.x != 0);
        assert!(transform.translation.length() <= REAL_SPACE_CELL_SIZE_M);
    }

    #[test]
    fn all_authored_bodies_fit_inside_big_space_cells() {
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let system = load_solar_system_from_dir(&assets).expect("load authored solar system");
        assert!(system.name_to_id.contains_key("Mira"));
        assert!(system.name_to_id.contains_key("Nereus"));

        let provider = PatchedConics::new(&system, 3.156e11);
        let states = provider.states(Epoch::ZERO);
        let grid = real_space_grid();

        for state in &states {
            let (_cell, local) = grid.translation_to_grid(state.position);
            assert!(
                local.length() <= REAL_SPACE_CELL_SIZE_M,
                "body {} produced oversized local translation {local:?}",
                state.id
            );
        }
    }
}
