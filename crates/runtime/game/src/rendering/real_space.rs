use bevy::math::DVec3;
use bevy::prelude::*;
use big_space::prelude::*;
use thalos_body_render::udlod::prelude::PreciseRotation;
use thalos_physics_canonical::types::BodyState;

use super::transforms::surface_body_to_world_orientation_f64;
use super::types::{PlayerShip, RealSpaceBody, SolarSystemState, TidallyLocked};

pub const REAL_SPACE_CELL_SIZE_M: f32 = 1_000.0;

/// The world point that renders at render-space **zero** — the one authority on
/// the big_space render frame.
///
/// big_space places every entity relative to the *grid cell origin of the
/// [`FloatingOrigin`]* (`LocalFloatingOrigin::set(origin_cell, ZERO, IDENTITY)`
/// in `big_space::grid::local_origin`), so for anything in the root grid
/// `render = world − floating_origin_cell_origin`. An entity that must sit
/// among big_space content while living *outside* the hierarchy — the
/// sun-shadow cascade cameras above all — has to measure from exactly this
/// point, or it lands somewhere the world isn't.
///
/// **This is not [`RenderOrigin`](crate::coords::RenderOrigin).** That one
/// tracks the *camera focus pivot* (the craft, in flight) and feeds the scaled
/// map / orbit projections, where a kilometre of slop is invisible. It
/// diverges from the render frame by the entire camera↔craft separation the
/// moment the view leaves the craft — freecam, god view — which is what parked
/// the shadow cascades around the ship instead of around the view
/// (INC-20260724T232104Z). Real-space placement reads this resource; scaled
/// map-space placement keeps `RenderOrigin`.
///
/// **Sole writer:** [`update_real_space_origin`].
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct RealSpaceOrigin {
    pub position: DVec3,
}

impl RealSpaceOrigin {
    /// A world-space point in render space.
    #[inline]
    pub fn to_render(&self, world: DVec3) -> Vec3 {
        (world - self.position).as_vec3()
    }
}

/// Publish the floating origin's cell origin as [`RealSpaceOrigin`].
///
/// Reads the camera's `CellCoord` as it stands at `SimStage::Sync` — the value
/// last frame's `TransformSystems::Propagate` rendered against. The camera
/// drivers rewrite it later this frame (`SimStage::Camera`), so on a frame where
/// the camera crosses a cell boundary this trails the render frame by one cell
/// (≤ `REAL_SPACE_CELL_SIZE_M`) for that frame; every other frame it is exact.
/// That residual is the ordinary one-frame camera lag every `SimStage::Sync`
/// consumer carries (see `rendering::view_anchor`), not the unbounded
/// craft-relative error it replaces.
pub(super) fn update_real_space_origin(
    grid: Query<&Grid, With<BigSpace>>,
    floating_origin: Query<&CellCoord, With<FloatingOrigin>>,
    mut origin: ResMut<RealSpaceOrigin>,
) {
    let (Ok(grid), Ok(cell)) = (grid.single(), floating_origin.single()) else {
        return;
    };
    origin.position = grid.grid_position_double(cell, &Transform::IDENTITY);
}

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

/// Fallback seat of a [`PlayerShip`] root into the BigSpace hierarchy, once per
/// frame in `Update`. The **primary** seat now happens at build time, inside
/// `ship_view::build_player_ship`'s part-reparent closure, so the root already
/// carries `CellCoord` + `ChildOf(root)` the instant its parts attach — that is
/// required for `Grid::tag_low_precision_roots` to tag the parts as
/// `LowPrecisionRoot` on their reparent frame (an un-seated root loses that race
/// permanently; see the closure comment). Without the seat the canonical→render
/// sync (`ship_view::update_player_ship_world_position`) also never matches the
/// root — the craft visually freezes in the inertial frame while the planet
/// sails away. This pass stays as a safety net for any PlayerShip spawned
/// without going through that closure; its `Without<CellCoord>` filter makes it
/// a no-op once every root is attached.
pub(super) fn attach_player_ship_to_big_space(
    mut commands: Commands,
    root: Res<RealSpaceRoot>,
    ships: Query<Entity, (With<PlayerShip>, Without<CellCoord>)>,
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
        &mut PreciseRotation,
    )>,
) {
    let Some(states) = cache.states.as_deref() else {
        return;
    };
    let Ok(root_grid) = grid.single() else {
        return;
    };

    for (body, lock, mut cell, mut transform, mut precise) in &mut bodies {
        let Some(state) = states.get(body.body_id) else {
            continue;
        };
        write_body_transform(
            state,
            lock,
            states,
            root_grid,
            &mut cell,
            &mut transform,
            &mut precise,
        );
    }
}

fn write_body_transform(
    state: &BodyState,
    lock: Option<&TidallyLocked>,
    states: &[BodyState],
    grid: &Grid,
    cell: &mut CellCoord,
    transform: &mut Transform,
    precise: &mut PreciseRotation,
) {
    let (next_cell, local) = grid.translation_to_grid(state.position);
    *cell = next_cell;
    transform.translation = local;

    // One f64 source feeds both rotations: the grid's f32 `Transform.rotation`
    // (big_space / udlod's low-precision far vertex path) and the f64
    // `PreciseRotation` (udlod's high-precision near Taylor path). Writing both
    // from the same value in the same system keeps the two precision paths from
    // slipping at the LOD swap.
    let rotation = surface_body_to_world_orientation_f64(state.id, lock, states).unwrap_or_else(|| {
        warn!(
            "could not resolve surface orientation for body {}; falling back to ephemeris orientation",
            state.id,
        );
        state.orientation.normalize()
    });
    transform.rotation = rotation.as_quat();
    precise.0 = rotation;
}

#[cfg(test)]
mod tests {
    use bevy::math::{DQuat, DVec3};
    use thalos_physics_canonical::body_trajectory_provider::BodyTrajectoryProvider;
    use thalos_physics_canonical::canonical::Epoch;
    use thalos_physics_canonical::patched_conics::PatchedConics;
    use thalos_world::parsing::load_solar_system_from_dir;

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
        let mut precise = PreciseRotation(DQuat::IDENTITY);

        write_body_transform(
            &state,
            None,
            &[state],
            &grid,
            &mut cell,
            &mut transform,
            &mut precise,
        );

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
