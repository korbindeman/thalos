//! Atmospheric aerodynamics for the local-physics bubble.
//!
//! Thalos uses the vendored [`avian_fdm`] flight-dynamics model for aerodynamic
//! forces, but only its **force pipeline** — Thalos owns mass, gravity, and the
//! body-centered-inertial bubble frame. This module is the adapter:
//!
//! - **Atmosphere.** `AircraftFdmPlugin` runs with `manage_atmosphere: false`;
//!   [`sync_aero_environment`] fills [`AtmosphereState`] from the body's physical
//!   atmosphere ([`thalos_world::TerrestrialAtmosphere::sample_at_altitude_m`]).
//! - **Airspeed.** Local air co-rotates with the planet, so true airspeed is
//!   `v − ω×r`; that wind is published into `avian_fdm`'s [`WindResource`].
//! - **Frame.** `avian_fdm` works in SAE body axes (X=nose, Y=right, Z=down);
//!   Thalos ships are Y=nose, X=right, Z=up. An [`AeroFrame`] on the aircraft
//!   root carries that fixed rotation so lift/drag/AoA are computed correctly,
//!   and all zone transforms are authored in the SAE frame (see [`entity_to_sae`]).
//! - **Force-only.** Zones carry **no collider**, so Avian's mass/inertia model
//!   is untouched; `avian_fdm` only writes [`ConstantForce`]/[`ConstantTorque`],
//!   which sum with Thalos's gravity/thrust accelerators in the solver. Engine
//!   thrust stays Thalos's (nose-forward throttle), so no `EngineZone` is used.
//!
//! Aircraft get a full set of lift/control zones derived from their wing parts
//! ([`build_ship_aero_layout`]); a fuselage bluff-body drag zone is always
//! present. EVA is excluded. **First-cut flight model — airfoil and control
//! constants need in-game tuning (see `docs/aerodynamics.md`).**

use avian_fdm::components::WindResource;
use avian_fdm::prelude::{
    AeroCoeff, AeroFrame, AeroZone, AircraftFdmPlugin, AircraftFdmSystems, AircraftGeometry,
    AtmosphereState, ControlInputs, ControlSurfaceRole, FlightState,
};
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use std::f64::consts::PI;
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::canonical::Epoch;
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::ActiveLocalBubble;
use thalos_physics_local::LocalCraftBody;
use thalos_physics_local::avian::{ConstantForce, ConstantTorque, PhysicsSchedule, Position};
use thalos_shipyard::WingAeroPanel;

use crate::rendering::SimulationState;

/// Rotation mapping a Thalos body-frame vector (X=right, Y=nose, Z=up) into
/// `avian_fdm`'s SAE aero frame (X=nose, Y=right, Z=down). A 180° rotation about
/// (1,1,0)/√2; it is its own inverse, so `sae_to_entity == entity_to_sae`.
pub fn entity_to_sae() -> DQuat {
    DQuat::from_axis_angle(DVec3::new(1.0, 1.0, 0.0).normalize(), PI)
}

// --- First-cut airfoil / control tuning constants (need in-game tuning) ------
/// Lift-curve slope, per radian (3-D, below 2-D 2π for finite aspect ratio).
const LIFT_SLOPE_PER_RAD: f64 = 5.0;
/// Camber lift at zero angle of attack for the cambered main wing.
const MAIN_WING_CL0: f64 = 0.25;
/// Stall angle (rad) where the linear pre-stall table ends; Viterna extends past.
const STALL_ANGLE_RAD: f64 = 0.26; // ~15°
/// Parasitic (zero-lift) drag coefficient of a lifting surface.
const PARASITIC_CD: f64 = 0.018;
/// Control-surface area as a fraction of its host surface area.
const CONTROL_AREA_FRACTION: f64 = 0.28;
/// Lift coefficient produced by a control surface at full deflection.
const CONTROL_AUTHORITY_CL: f64 = 1.8;
/// Stations beyond this fraction down the fuselage are the empennage (tail).
const TAIL_STATION_THRESHOLD: f64 = 0.75;
/// Mount-angle tolerance (rad) for classifying a surface as vertical (a fin).
const VERTICAL_ANGLE_TOLERANCE: f64 = 0.6;

/// One ready-to-spawn aero zone: its SAE-frame transform plus the built
/// [`AeroZone`]. Produced by [`build_ship_aero_layout`], consumed by
/// [`attach_ship_aero`].
#[derive(Clone)]
pub struct AeroZoneSpec {
    pub name: String,
    pub translation_sae: Vec3,
    pub rotation_sae: Quat,
    pub zone: AeroZone,
}

/// The aircraft's aerodynamic zone layout, computed from its blueprint by
/// `ship_view` and consumed when the Avian body spawns. Replaced on each ship
/// spawn.
#[derive(Resource, Default)]
pub struct ShipAeroLayout {
    pub zones: Vec<AeroZoneSpec>,
    pub reference_area_m2: f64,
    pub reference_chord_m: f64,
    pub reference_span_m: f64,
}

/// Wires the vendored `avian_fdm` force pipeline into the local bubble.
pub struct GameAeroPlugin;

impl Plugin for GameAeroPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(AircraftFdmPlugin {
            validate_on_startup: false,
            manage_atmosphere: false,
        })
        .init_resource::<WindResource>()
        .init_resource::<ShipAeroLayout>()
        .add_systems(
            PhysicsSchedule,
            sync_aero_environment.in_set(AircraftFdmSystems::Atmosphere),
        )
        .add_systems(Update, (attach_ship_aero, sync_flight_controls));
    }
}

/// Build a [`Table1D`](AeroCoeff::Table1D) lift curve through three breakpoints
/// (−stall, 0, +stall) with the given zero-α lift and slope.
fn lift_table(cl0: f64, slope: f64) -> AeroCoeff {
    AeroCoeff::Table1D {
        breakpoints: vec![-STALL_ANGLE_RAD, 0.0, STALL_ANGLE_RAD],
        values: vec![cl0 - slope * STALL_ANGLE_RAD, cl0, cl0 + slope * STALL_ANGLE_RAD],
    }
}

/// A base lifting-surface zone (no control role). `cambered` adds camber lift.
fn base_lifting_zone(area_m2: f64, chord_m: f64, cambered: bool) -> AeroZone {
    let cl0 = if cambered { MAIN_WING_CL0 } else { 0.0 };
    AeroZone {
        cl: lift_table(cl0, LIFT_SLOPE_PER_RAD),
        cd: AeroCoeff::Scalar(PARASITIC_CD),
        area_m2,
        chord_m,
        ..Default::default()
    }
    // Extend the tables to ±180° (Viterna) so tumbling / deep stall stay finite.
    .with_post_stall_extension()
}

/// A control-surface zone: constant full-deflection lift authority, scaled by
/// the pilot input via [`ControlSurfaceRole`]. Co-located with its host surface.
fn control_zone(area_m2: f64, chord_m: f64, role: ControlSurfaceRole) -> AeroZone {
    AeroZone {
        cl: AeroCoeff::Scalar(CONTROL_AUTHORITY_CL),
        cd: AeroCoeff::Scalar(PARASITIC_CD),
        area_m2,
        chord_m,
        control_role: Some(role),
        ..Default::default()
    }
}

/// SAE-frame transform (translation + rotation) for a wing panel from its
/// body-frame aerodynamic geometry. The zone's local frame is X = chord-forward,
/// Z = "down" (opposite the airfoil lift normal), Y = Z×X.
fn panel_transform_sae(panel: &WingAeroPanel) -> (Vec3, Quat) {
    let e2s = entity_to_sae();
    let translation = (e2s * panel.center_body_m).as_vec3();
    let zx = (e2s * panel.fore_dir).normalize();
    let mut zz = e2s * (-panel.thick_dir);
    // Orthogonalise the lift axis against the chord axis.
    zz = (zz - zx * zz.dot(zx)).normalize();
    let zy = zz.cross(zx);
    let rotation = DQuat::from_mat3(&DMat3::from_cols(zx, zy, zz)).as_quat();
    (translation, rotation)
}

/// Compute the aerodynamic zone layout for an aircraft from its wing panels.
///
/// Classifies each panel by mount geometry: vertical surfaces → fin (rudder),
/// aft horizontal → stabiliser (elevator), forward horizontal → main wing
/// (cambered, ailerons). Each surface gets a base lifting zone (stability) plus
/// a control zone (authority). A fuselage bluff-body drag zone is always added.
pub fn build_ship_aero_layout(
    panels: &[WingAeroPanel],
    frontal_area_m2: f64,
    drag_cd: f64,
) -> ShipAeroLayout {
    let mut zones = Vec::new();

    // Fuselage bluff-body drag at the CoM (zero lift).
    zones.push(AeroZoneSpec {
        name: "fuselage_drag".into(),
        translation_sae: Vec3::ZERO,
        rotation_sae: Quat::IDENTITY,
        zone: AeroZone {
            cl: AeroCoeff::Scalar(0.0),
            cd: AeroCoeff::Scalar(drag_cd),
            area_m2: frontal_area_m2,
            ..Default::default()
        },
    });

    let mut total_area = 0.0;
    let mut mac_acc = 0.0;
    let mut max_span = 0.0_f64;

    for (i, panel) in panels.iter().enumerate() {
        let (translation_sae, rotation_sae) = panel_transform_sae(panel);

        // Vertical (fin) when mounted near the dorsal/belly meridian.
        let vertical = panel.angle.abs() < VERTICAL_ANGLE_TOLERANCE
            || (PI - panel.angle.abs()).abs() < VERTICAL_ANGLE_TOLERANCE;
        let aft = panel.station > TAIL_STATION_THRESHOLD;
        let right_side = panel.angle > 0.0;
        let cambered = !vertical && !aft;

        let role = if vertical {
            ControlSurfaceRole::Rudder
        } else if aft {
            ControlSurfaceRole::Elevator
        } else if right_side {
            ControlSurfaceRole::AileronRight
        } else {
            ControlSurfaceRole::AileronLeft
        };

        // Base lifting surface (no control) — provides stability + damping.
        zones.push(AeroZoneSpec {
            name: format!("wing{i}_surface"),
            translation_sae,
            rotation_sae,
            zone: base_lifting_zone(panel.area_m2, panel.chord_m, cambered),
        });

        // Co-located control surface (a fraction of the area).
        zones.push(AeroZoneSpec {
            name: format!("wing{i}_control"),
            translation_sae,
            rotation_sae,
            zone: control_zone(
                panel.area_m2 * CONTROL_AREA_FRACTION,
                panel.chord_m,
                role,
            ),
        });

        if !vertical {
            total_area += panel.area_m2;
            mac_acc += panel.chord_m * panel.area_m2;
            max_span = max_span.max(panel.span_m);
        }
    }

    let reference_area_m2 = if total_area > 0.0 {
        total_area
    } else {
        frontal_area_m2.max(1.0)
    };
    let reference_chord_m = if total_area > 0.0 {
        mac_acc / total_area
    } else {
        1.0
    };

    ShipAeroLayout {
        zones,
        reference_area_m2,
        reference_chord_m,
        reference_span_m: max_span.max(1.0),
    }
}

/// Attach the `avian_fdm` aircraft-root components + zone children to the player
/// ship's Avian body, once, after it spawns. Idempotent via
/// `Without<AircraftGeometry>`. EVA bodies are skipped.
fn attach_ship_aero(
    mut commands: Commands,
    sim: Res<SimulationState>,
    layout: Res<ShipAeroLayout>,
    ships: Query<Entity, (With<LocalCraftBody>, Without<AircraftGeometry>)>,
) {
    if sim.simulation.vessel_kind() != VesselKind::Ship {
        return;
    }
    let params = *sim.simulation.ship_params();
    let drag_cd = params.drag_coefficient;
    let drag_area = params.reference_area_m2;

    for ship in &ships {
        // Reference geometry: wings if the layout has them, else fall back to
        // the bluff-body frontal area.
        let (ref_area, ref_chord, ref_span, specs): (f64, f64, f64, Vec<AeroZoneSpec>) =
            if layout.zones.len() > 1 {
                (
                    layout.reference_area_m2,
                    layout.reference_chord_m,
                    layout.reference_span_m,
                    layout.zones.clone(),
                )
            } else {
                // No wings (rocket/capsule): a single bluff-body drag zone.
                (
                    drag_area.max(1.0),
                    1.0,
                    4.0,
                    vec![AeroZoneSpec {
                        name: "fuselage_drag".into(),
                        translation_sae: Vec3::ZERO,
                        rotation_sae: Quat::IDENTITY,
                        zone: AeroZone {
                            cl: AeroCoeff::Scalar(0.0),
                            cd: AeroCoeff::Scalar(drag_cd),
                            area_m2: drag_area,
                            ..Default::default()
                        },
                    }],
                )
            };

        commands.entity(ship).insert((
            AircraftGeometry {
                wing_area_m2: ref_area.max(1.0),
                wing_span_m: ref_span.max(1.0),
                chord_m: ref_chord.max(0.1),
            },
            AeroFrame {
                sae_to_entity: entity_to_sae(),
            },
            ControlInputs::default(),
            FlightState::default(),
            AtmosphereState::default(),
            ConstantForce::default(),
            ConstantTorque::default(),
        ));

        for spec in &specs {
            let zone = commands
                .spawn((
                    spec.zone.clone(),
                    Transform {
                        translation: spec.translation_sae,
                        rotation: spec.rotation_sae,
                        ..Default::default()
                    },
                    Name::new(spec.name.clone()),
                ))
                .id();
            commands.entity(ship).add_child(zone);
        }
    }
}

/// Per frame: feed flight-control intent (pitch/roll/yaw) into the aircraft's
/// `avian_fdm` [`ControlInputs`]. Elevator = pitch, aileron = roll, rudder =
/// yaw. (Throttle stays Thalos's nose-forward thrust, applied separately.)
fn sync_flight_controls(
    intent: Res<GameInputIntent>,
    mut aircraft: Query<&mut ControlInputs, With<AircraftGeometry>>,
) {
    let Ok(mut ctrl) = aircraft.single_mut() else {
        return;
    };
    // `attitude` = (pitch, roll, yaw) ∈ [−1, 1]. Pulling pitch (nose up) should
    // deflect the elevator to pitch up; sign tuned alongside the airfoil.
    let elevator = intent.attitude.x as f64;
    let aileron = intent.attitude.y as f64;
    let rudder = intent.attitude.z as f64;
    if (ctrl.elevator - elevator).abs() > 1e-4
        || (ctrl.aileron - aileron).abs() > 1e-4
        || (ctrl.rudder - rudder).abs() > 1e-4
    {
        ctrl.elevator = elevator;
        ctrl.aileron = aileron;
        ctrl.rudder = rudder;
    }
}

/// Per-physics-step: populate the ship's [`AtmosphereState`] from the dominant
/// body's density model and publish the co-rotation [`WindResource`].
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
        *atm = AtmosphereState::default();
        wind.velocity_world_ms = DVec3::ZERO;
        return;
    };

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

    let body_state = sim
        .ephemeris
        .state(bubble.body_id, Epoch(sim.simulation.sim_time()));
    wind.velocity_world_ms = body_state.angular_velocity.cross(position.0);
}
