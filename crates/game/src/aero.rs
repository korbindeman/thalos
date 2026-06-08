//! Atmospheric aerodynamics for the local-physics bubble.
//!
//! Thalos uses the vendored [`avian_fdm`] flight-dynamics model for aerodynamic
//! forces, but only its **force pipeline** — Thalos owns mass, gravity, and the
//! body-centered-inertial bubble frame. This module is the adapter that feeds
//! `avian_fdm` Thalos's environment instead of its built-in Earth-ISA/world-Y
//! assumptions:
//!
//! - **Atmosphere.** `AircraftFdmPlugin` is added with `manage_atmosphere: false`,
//!   so its `update_atmosphere` (density from world-space Y) is *not* registered.
//!   [`sync_aero_environment`] fills [`AtmosphereState`] each physics step from
//!   the dominant body's physical atmosphere
//!   ([`thalos_world::TerrestrialAtmosphere::sample_at_altitude_m`]) at the
//!   craft's real altitude above the mean surface.
//! - **Airspeed.** The local airmass co-rotates with the planet, so true
//!   airspeed is `v − ω×r`. We publish `ω×r` into `avian_fdm`'s [`WindResource`]
//!   (read by its `update_flight_state` as `vel_world = lin_vel − wind`).
//! - **Force-only.** The ship's [`AeroZone`]s carry **no collider**, so Avian's
//!   mass/inertia model is untouched; `avian_fdm` only writes
//!   [`ConstantForce`]/[`ConstantTorque`], which sum with Thalos's
//!   gravity/thrust `ConstantLinearAcceleration` in the solver.
//!
//! Scope (first slice): a single bluff-body drag zone on the player ship —
//! enough to feel reentry deceleration and terminal velocity. Lift / control
//! surfaces / planes come later (see `docs/aerodynamics.md`). EVA is excluded
//! (no aero zones are attached), exactly like it is excluded from terrain
//! contact.

use avian_fdm::prelude::{
    AeroCoeff, AeroZone, AircraftFdmPlugin, AircraftFdmSystems, AircraftGeometry, AtmosphereState,
    ControlInputs, FlightState,
};
use avian_fdm::components::WindResource;
use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics_canonical::canonical::Epoch;
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::ActiveLocalBubble;
use thalos_physics_local::LocalCraftBody;
use thalos_physics_local::avian::{ConstantForce, ConstantTorque, PhysicsSchedule, Position};

use crate::rendering::SimulationState;

/// Wires the vendored `avian_fdm` force pipeline into the local bubble and
/// feeds it Thalos's per-body atmosphere + co-rotation airspeed.
pub struct GameAeroPlugin;

impl Plugin for GameAeroPlugin {
    fn build(&self, app: &mut App) {
        // `manage_atmosphere: false` vacates the `Atmosphere` set so we own
        // `AtmosphereState`; `validate_on_startup: false` because our roots are
        // attached at runtime (not at Startup) and carry no zone colliders.
        app.add_plugins(AircraftFdmPlugin {
            validate_on_startup: false,
            manage_atmosphere: false,
        })
        .init_resource::<WindResource>()
        .add_systems(
            PhysicsSchedule,
            sync_aero_environment.in_set(AircraftFdmSystems::Atmosphere),
        )
        .add_systems(Update, attach_ship_aero);
    }
}

/// Attach the `avian_fdm` aircraft-root components + a bluff-body drag zone to
/// the player ship's Avian body, once, after it spawns.
///
/// `Without<AircraftGeometry>` makes this idempotent — once the root components
/// are inserted the entity no longer matches. EVA bodies are skipped (only
/// `VesselKind::Ship` gets aero).
fn attach_ship_aero(
    mut commands: Commands,
    sim: Res<SimulationState>,
    ships: Query<Entity, (With<LocalCraftBody>, Without<AircraftGeometry>)>,
) {
    if sim.simulation.vessel_kind() != VesselKind::Ship {
        return;
    }
    // Per-vehicle drag from the ship's actual geometry (frontal area) + a
    // blunt-body Cd, pushed into `ShipParameters` by `ship_view`.
    let params = *sim.simulation.ship_params();
    let reference_area_m2 = params.reference_area_m2;
    let drag_coefficient = params.drag_coefficient;
    for ship in &ships {
        // Root: geometry (only `wing_area`/`span`/`chord` non-dimensionalisers;
        // a pure drag zone barely uses them), control inputs, and the output +
        // accumulator components `avian_fdm` reads/writes each step.
        commands.entity(ship).insert((
            AircraftGeometry {
                wing_area_m2: reference_area_m2.max(1.0),
                wing_span_m: 4.0,
                chord_m: 2.0,
            },
            ControlInputs::default(),
            FlightState::default(),
            AtmosphereState::default(),
            ConstantForce::default(),
            ConstantTorque::default(),
        ));

        // One collider-less bluff-body drag zone at the CoM: drag opposes the
        // airflow, ~zero lift, no pitching moment. Collider-less so Avian's
        // mass/inertia (owned by Thalos) is untouched.
        let zone = commands
            .spawn((
                AeroZone {
                    cl: AeroCoeff::Scalar(0.0),
                    cd: AeroCoeff::Scalar(drag_coefficient),
                    area_m2: reference_area_m2,
                    chord_m: 0.0,
                    ..Default::default()
                },
                Transform::default(),
                Name::new("ShipDragZone"),
            ))
            .id();
        commands.entity(ship).add_child(zone);
    }
}

/// Per-physics-step: populate the ship's [`AtmosphereState`] from the dominant
/// body's density model and publish the co-rotation [`WindResource`].
///
/// Runs in [`AircraftFdmSystems::Atmosphere`] (the slot vacated by
/// `manage_atmosphere: false`), so `avian_fdm`'s `update_flight_state` and
/// force systems read a Thalos-populated atmosphere later in the same step.
/// Altitude uses the cheap mean-radius approximation (the Kármán line sits far
/// above terrain relief, so per-pixel terrain height is unnecessary here).
fn sync_aero_environment(
    sim: Res<SimulationState>,
    active: Res<ActiveLocalBubble>,
    mut wind: ResMut<WindResource>,
    mut craft: Query<(&Position, &mut AtmosphereState), With<LocalCraftBody>>,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    if sim.simulation.vessel_kind() != VesselKind::Ship {
        return;
    }
    let Ok((position, mut atm)) = craft.get_mut(bubble.craft_entity) else {
        return;
    };

    let body = &sim.system.bodies[bubble.body_id];
    let Some(atmosphere) = body.terrestrial_atmosphere.as_ref() else {
        // Airless body: vacuum everywhere → no drag, no wind.
        *atm = AtmosphereState::default();
        wind.velocity_world_ms = DVec3::ZERO;
        return;
    };

    // Physical exponential atmosphere: ρ, P, T, and speed-of-sound derived from
    // the body's surface pressure (single-sourced from the terrain
    // `AtmosphereSpec`), its surface gravity, and the authored surface
    // temperature / gas constant. No Earth-hardcoded constants.
    let altitude_m = position.0.length() - body.radius_m;
    let sample = atmosphere.sample_at_altitude_m(
        altitude_m,
        body.surface_pressure_pa(),
        body.surface_gravity_m_s2(),
    );
    *atm = AtmosphereState {
        density_kgm3: sample.density_kg_m3,
        pressure_pa: sample.pressure_pa,
        temperature_k: sample.temperature_k,
        speed_of_sound_ms: sample.speed_of_sound_m_s,
    };

    // The airmass co-rotates with the body. In the body-centered-inertial
    // bubble the craft velocity is already relative to the body centre, so the
    // local air velocity is purely the rotational term ω × r.
    let body_state = sim
        .ephemeris
        .state(bubble.body_id, Epoch(sim.simulation.sim_time()));
    wind.velocity_world_ms = body_state.angular_velocity.cross(position.0);
}
