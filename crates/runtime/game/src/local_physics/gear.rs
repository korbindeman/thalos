//! Raycast spring-damper landing gear: wheels, tuning, parking brake, ground forces.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/simulation/regimes.md`).

#[allow(unused_imports)]
use super::*;
use std::collections::HashMap;

use bevy::math::{DQuat, DVec3};
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::avian::{
    AngularVelocity, ConstantAngularAcceleration, ConstantLinearAcceleration, LinearVelocity,
    Position, Rotation, SpatialQuery, SpatialQueryFilter,
};
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry, LocalCraftBody};
use thalos_shipyard::{AttachNodes, Gear, Part, SurfaceMount, SurfaceMountKind, gear_leg_frames};

use crate::rendering::SimulationState;

pub use thalos_game_state::flight::{
    GearState, ParkingBrake, WeightOnWheels, Wheel, WheelSet, set_gear_down,
};

/// The landing-gear parts of the *flight* craft (editor builds excluded) —
/// the one query shape every gear consumer uses ([`build_wheel_set`],
/// [`gear_contact_geometry`], spawn, runway/launchpad placement).
pub(crate) type GearPartQuery<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static Gear, &'static SurfaceMount),
    (
        With<Part>,
        Without<crate::shipyard_editor::core::EditorPart>,
    ),
>;

/// Landing-gear suspension/grip coefficients. Reflect-registered (for a future
/// debug UI); edit the defaults and rebuild to tune them. Forces are computed
/// per wheel and summed into the craft's acceleration accumulators.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct GearTuning {
    /// Loaded ride-height compression as a fraction of usable strut travel.
    /// Per-wheel spring stiffness is derived from this, the craft's surface
    /// weight, the wheel's real axle load, and its authored stroke. A fixed
    /// N/m constant cannot represent both a light nose leg and a main leg
    /// carrying almost half an airliner.
    pub static_sag_fraction: f64,
    /// Suspension damping ratio ζ while the strut is **compressing** (the
    /// touchdown stroke). The damper coefficient is derived per frame from the
    /// craft's real per-wheel mass — `c = 2·ζ·√(k·m_per_wheel)` — so the tuning
    /// is mass-independent. Deliberately soft (well under critical), like a real
    /// oleo strut: at contact onset the compression *rate* is the full sink
    /// speed while the spring force is still zero, so a near-critical damper
    /// here delivers its whole force as a step the instant the wheels touch —
    /// the "every landing is a slam" defect. Soft compression damping lets the
    /// stroke absorb the sink; the rebound ratio below stops the bounce.
    pub damping_ratio_compress: f64,
    /// Suspension damping ratio ζ while the strut is **extending** (rebound).
    /// At/above critical, so the energy the spring stored on the touchdown
    /// stroke is dissipated on the way back out instead of bouncing the craft —
    /// the classic asymmetric oleo schedule (soft in, firm out).
    pub damping_ratio_rebound: f64,
    /// Fraction of max travel over which the compression damper ramps in from
    /// zero. Makes the contact-onset force continuous (spring ≈ 0 *and* damper
    /// ≈ 0 at first touch) so there is no discontinuous jolt at the moment of
    /// touchdown; by this depth into the stroke the damper is at full strength.
    /// Rebound damping is never ramped — a nearly-extended strut still needs
    /// full rebound damping to not fling the wheel out.
    pub damper_engage_frac: f64,
    /// Fraction of travel where the progressive end stop begins. Normal taxi
    /// and touchdown loads stay on the linear oleo; the end stop only catches
    /// a certification-rate or badly asymmetric impact before the hard clamp.
    pub bump_stop_start_fraction: f64,
    /// Strength of the quadratic end stop relative to the linear spring.
    pub bump_stop_stiffness_factor: f64,
    /// Friction-circle limit: max horizontal force as a multiple of normal load.
    pub mu: f64,
    /// Lateral grip stiffness, N per (m/s) of sideways slip (clamped by `mu·N`).
    pub k_lat: f64,
    /// Free-rolling resistance coefficient: the Coulomb cap on fore/aft wheel
    /// force as a fraction of normal load (`μ_roll·N`). A *constant* opposing
    /// force, not viscous — so a coasting craft decelerates linearly to a true
    /// stop and then holds, instead of asymptotically creeping forever. Small,
    /// so wheels stay low-resistance under thrust and roll on gentle slopes.
    pub rolling_mu: f64,
    /// Hold stiffness for the rolling-resistance Coulomb term: N per (m/s) of
    /// roll speed, clamped to `rolling_mu·N`. High enough that the breakaway
    /// speed (`cap/stiffness`) is a few cm/s, so below it the wheel is pinned
    /// (static) and above it the force saturates to the constant Coulomb cap.
    pub rolling_hold_stiffness: f64,
    /// Parking-brake hold stiffness: N per (m/s) of fore/aft creep, clamped to
    /// the friction circle `mu·N`. High so even a tiny creep is opposed
    /// near-maximally — the craft stays put when the brake is engaged.
    pub parking_brake_stiffness: f64,
    /// Max nosewheel steer angle at full yaw input, radians. This is the
    /// *taxi* (tiller) authority; it fades with ground speed — see
    /// [`GearTuning::steer_fade_speed_m_s`].
    pub max_steer_rad: f64,
    /// Ground speed (m/s) at which nosewheel steering authority has faded to
    /// half its taxi value (`scale = 1 / (1 + (v/v_fade)²)`). Full tiller
    /// throw at taxi speed would trip the craft over its main gear at takeoff
    /// speed, so steering blends out as the aero rudder blends in — the
    /// real-world tiller→pedals split.
    pub steer_fade_speed_m_s: f64,
    /// Max suspension travel as a fraction of strut length.
    pub max_travel_fraction: f64,
    /// Extra ray length past the rest length so a wheel just off the ground or
    /// over a slope edge still finds the surface, metres.
    pub skin_margin: f64,
    /// How far **above** the strut top the suspension ray starts, metres. The
    /// strut top sits on the hull skin, and a hard landing can drive the hull
    /// below the terrain surface (the floor backstop tolerates up to its
    /// `skin_m` of penetration) — a ray started *at* a buried strut top points
    /// down from underneath the heightfield, hits nothing, and the wheel
    /// silently loses the ground exactly when it is needed most (no normal
    /// force → no brakes → the front-wheel-buried slide). Starting the ray
    /// above the skin keeps the surface in view through any penetration the
    /// backstop permits; the lift is subtracted from the hit distance so the
    /// suspension geometry is unchanged.
    pub ray_start_lift_m: f64,
}

impl Default for GearTuning {
    fn default() -> Self {
        Self {
            // Airliner oleos sit visibly inside their stroke under ramp weight;
            // leaving roughly four fifths available absorbs landing energy
            // without making taxi feel like a rigid collider.
            static_sag_fraction: 0.22,
            // Soft touchdown stroke (~1 g peak on a 2 m/s sink for the demo
            // aircraft), firm rebound: the real-oleo asymmetric schedule.
            damping_ratio_compress: 0.4,
            damping_ratio_rebound: 1.2,
            damper_engage_frac: 0.2,
            bump_stop_start_fraction: 0.72,
            bump_stop_stiffness_factor: 6.0,
            // Dry-tire grip. Deliberately below ~1.0: the lateral force a
            // skidding tire can transmit is what rolls a craft over its gear,
            // and a real tire slides at ~0.8 before it can generate a
            // tipping moment that large.
            mu: 0.8,
            k_lat: 40_000.0,
            rolling_mu: 0.02,
            rolling_hold_stiffness: 60_000.0,
            parking_brake_stiffness: 60_000.0,
            max_steer_rad: 0.5,
            steer_fade_speed_m_s: 12.0,
            max_travel_fraction: 0.8,
            skin_margin: 0.5,
            // Covers the backstop's 0.5 m penetration allowance with margin.
            ray_start_lift_m: 1.0,
        }
    }
}

/// Per-gear-part suspension state for the **visual** gear-mesh offset:
/// `(susp_dir_local, compression_m)` of the deepest-compressed wheel of that
/// gearbox. The rendered gear mesh is rigid (authored at full extension), so
/// without this every centimetre of real suspension compression rendered as
/// the wheel sinking into the pavement. `ship_view::sync_gear_compression`
/// slides each gearbox mesh *up into the hull* by this amount — the classic
/// strut-swallow cheat — so the wheels stay on the surface.
///
/// Sole writer: [`apply_landing_gear_forces`] (cleared up front each frame, so
/// airborne/retracted/stand-down paths read as "no compression").
#[derive(Resource, Default, Debug, Clone)]
pub struct GearVisualCompression(pub HashMap<Entity, (DVec3, f64)>);

/// Per-axis reaction-wheel torque mask for the current ground state, applied
/// to `ShipParameters::max_torque` by **both** the fly-by-wire controller's
/// authority normalization (`control_bus::realize_control`) and the force
/// realization (`local_physics::apply_local_forces`) — the two must agree or
/// the controller's commanded torque stops equalling the realized torque.
///
/// Real aircraft have no reaction wheels; realizing them on the ground let a
/// stick or SAS roll/yaw command torque the airframe over its own gear at
/// taxi speed (the "tips over on the runway" defect). While weight is on the
/// wheels: **pitch (body X) keeps** wheel assist so takeoff rotation stays
/// available on marginal elevator authority; **roll (body Y) and yaw (body Z)
/// are zeroed** — roll is the flip axis and has no legitimate on-ground use,
/// and yaw belongs to the rudder + nosewheel steering.
///
/// A craft on its **hull** with no wheel bearing load (tipped onto a wing, or
/// a gear-up belly slide — [`super::HullGroundContact`]) loses **all** wheel
/// torque: with the wheels unloaded the old mask handed SAS its full roll/yaw
/// authority back, and its attempt to right the tipped craft power-slid it
/// across the pavement in a constant pirouette (with the saturated commands
/// slamming the control surfaces side to side as the error axis swept
/// through the body frame). Weight-on-wheels wins when both are true, so a
/// tail-graze during takeoff rotation cannot drop the pitch assist.
/// Airborne: full torque.
pub fn wheel_torque_ground_mask(weight_on_wheels: bool, hull_grounded: bool) -> DVec3 {
    if weight_on_wheels {
        DVec3::new(1.0, 0.0, 0.0)
    } else if hull_grounded {
        DVec3::ZERO
    } else {
        DVec3::ONE
    }
}

/// Flip the parking brake on the toggle edge (B). Runs before the gear forces.
pub(crate) fn toggle_parking_brake(intent: Res<GameInputIntent>, mut brake: ResMut<ParkingBrake>) {
    if intent.parking_brake_toggle {
        brake.engaged = !brake.engaged;
    }
}

/// Flip the landing gear on the toggle edge (G), with the retract-on-ground
/// interlock: a down→up request is **ignored** while weight is on the wheels
/// (the craft can't retract the legs it is standing on). Extending is always
/// allowed. Reads the previous frame's [`WeightOnWheels`] (set by
/// [`apply_landing_gear_forces`], which runs later in the chain) — a one-frame
/// lag that is immaterial for a manual latch. The HUD `GEAR` pill drives the
/// same state through the same interlock (`hud::flight_config_panel`).
pub(crate) fn toggle_gear(
    intent: Res<GameInputIntent>,
    weight_on_wheels: Res<WeightOnWheels>,
    mut gear: ResMut<GearState>,
) {
    if intent.gear_toggle {
        let target = !gear.down;
        set_gear_down(&mut gear, &weight_on_wheels, target);
    }
}

/// Build the cached [`WheelSet`] for a craft from its landing-gear parts,
/// reusing [`gear_leg_frames`] (the same per-leg geometry the visual mesh
/// draws) so collider wheels sit exactly under the rendered ones. `positions`
/// is the part-tree translation map ([`compute_part_collider_positions`]); the
/// gear's mount point on the host axis mirrors the `BodySkin` station offset
/// that map applies.
/// Landing-gear contact geometry for parked placement: the lowest wheel-bottom
/// depth below the craft origin along the ventral (−Z) axis, plus the mean
/// authored strut length used for parked static sag. Derived from the **gear
/// contact geometry** ([`build_wheel_set`], the
/// same data the runtime suspension uses), *not* visual meshes: at parked-spawn
/// time the gear's visual meshes may not be spawned yet, so a visual-extent
/// measurement would only see the fuselage and bury the gear. Returns `None` for
/// a craft with no landing gear (the caller falls back to the visual-mesh
/// clearance and rests it on its belly).
///
/// The depth is the *zero-compression* rest height (wheels just touching). The
/// caller subtracts the static suspension sag so the craft spawns with its gear
/// already loaded — see [`crate::runway`].
pub(crate) fn gear_contact_geometry(
    parts: &PartColliderQuery,
    gear_q: &GearPartQuery,
    host_nodes: &Query<&AttachNodes>,
) -> Option<(f64, f64)> {
    let positions = compute_part_collider_positions(parts);
    let wheels = build_wheel_set(gear_q, host_nodes, &positions);
    if wheels.is_empty() {
        return None;
    }
    let lowest = wheels.iter().fold(f64::INFINITY, |acc, w| {
        let bottom = w.strut_top_local + w.susp_dir_local * (w.strut_length + w.wheel_radius);
        acc.min(bottom.z)
    });
    let mean_strut_length_m =
        wheels.iter().map(|wheel| wheel.strut_length).sum::<f64>() / wheels.len() as f64;
    (lowest.is_finite() && lowest < 0.0).then_some((-lowest, mean_strut_length_m))
}

pub(crate) fn build_wheel_set(
    gear_q: &GearPartQuery,
    host_nodes: &Query<&AttachNodes>,
    positions: &HashMap<Entity, DVec3>,
) -> Vec<Wheel> {
    let mut wheels = Vec::new();
    for (gear_entity, gear, mount) in gear_q.iter() {
        let Ok(nodes) = host_nodes.get(mount.parent) else {
            continue;
        };
        let parent_radius = nodes
            .get("top")
            .map(|n| n.diameter * 0.5)
            .unwrap_or(1.0)
            .max(0.01);
        let host_pos = positions.get(&mount.parent).copied().unwrap_or(DVec3::ZERO);
        // The gear's mount origin on the host axis at its station — mirrors the
        // `SurfaceMountKind` branch in `compute_part_collider_positions`.
        let mount_axis = match mount.kind {
            SurfaceMountKind::BodySkin => {
                let host_height = nodes.get("bottom").map(|n| -n.offset.y).unwrap_or(0.0) as f64;
                host_pos + DVec3::new(0.0, -(mount.station as f64) * host_height, 0.0)
            }
            SurfaceMountKind::WingPylon => host_pos,
        };
        for leg in gear_leg_frames(gear, mount.angle, parent_radius) {
            wheels.push(Wheel {
                source: gear_entity,
                strut_top_local: mount_axis + leg.strut_top.as_dvec3(),
                susp_dir_local: leg.susp_dir.as_dvec3(),
                roll_dir_local: leg.roll_dir.as_dvec3(),
                axle_dir_local: leg.axle_dir.as_dvec3(),
                strut_length: gear.strut_length as f64,
                wheel_radius: gear.wheel_radius as f64,
                steerable: gear.legs() == 1,
            });
        }
    }
    wheels
}

#[derive(Debug, Clone, Copy)]
struct WheelLoadModel {
    steerable_count: usize,
    main_count: usize,
    steerable_load_fraction: f64,
}

impl WheelLoadModel {
    fn static_share(self, wheel: &Wheel, wheel_count: usize) -> f64 {
        if self.steerable_count == 0 || self.main_count == 0 {
            return 1.0 / wheel_count.max(1) as f64;
        }
        if wheel.steerable {
            self.steerable_load_fraction / self.steerable_count as f64
        } else {
            (1.0 - self.steerable_load_fraction) / self.main_count as f64
        }
    }

    fn compression_mass_share(self, wheel: &Wheel, wheel_count: usize) -> f64 {
        if self.steerable_count == 0 || self.main_count == 0 {
            return 1.0 / wheel_count.max(1) as f64;
        }
        if wheel.steerable {
            self.static_share(wheel, wheel_count)
        } else {
            // Main wheels touch first and initially absorb the whole craft's
            // vertical kinetic energy, not merely their eventual static axle
            // share. Split impact mass across the main legs.
            1.0 / self.main_count as f64
        }
    }
}

/// Static nose/main reaction split from the support polygon in the craft's
/// longitudinal axis. Falls back to equal shares for unconventional layouts
/// without both a steerable and a main axle, or when the CoM lies outside the
/// axle span (an invalid parked stance that this model must not disguise).
fn wheel_load_model(wheels: &[Wheel], com_local: DVec3) -> WheelLoadModel {
    let mut steerable_count = 0usize;
    let mut main_count = 0usize;
    let mut steerable_y_sum = 0.0;
    let mut main_y_sum = 0.0;
    for wheel in wheels {
        if wheel.steerable {
            steerable_count += 1;
            steerable_y_sum += wheel.strut_top_local.y;
        } else {
            main_count += 1;
            main_y_sum += wheel.strut_top_local.y;
        }
    }
    let mut model = WheelLoadModel {
        steerable_count,
        main_count,
        steerable_load_fraction: if steerable_count + main_count > 0 {
            steerable_count as f64 / (steerable_count + main_count) as f64
        } else {
            0.0
        },
    };
    if steerable_count > 0 && main_count > 0 {
        let steerable_y = steerable_y_sum / steerable_count as f64;
        let main_y = main_y_sum / main_count as f64;
        let span = steerable_y - main_y;
        if span.abs() > 1.0e-6 {
            let fraction = (com_local.y - main_y) / span;
            if (0.0..=1.0).contains(&fraction) {
                model.steerable_load_fraction = fraction;
            }
        }
    }
    model
}

#[derive(Debug, Clone, Copy)]
struct SuspensionCoefficients {
    max_travel_m: f64,
    spring_n_per_m: f64,
    damping_compress_n_s_per_m: f64,
    damping_rebound_n_s_per_m: f64,
}

/// Loaded compression for an authored strut. Shared with runway placement so
/// parked spawn and live suspension cannot drift onto different ride heights.
pub(crate) fn nominal_static_sag_m(tuning: &GearTuning, strut_length_m: f64) -> f64 {
    let max_travel_m = (tuning.max_travel_fraction * strut_length_m).max(1.0e-3);
    (tuning.static_sag_fraction.clamp(0.02, 0.65) * max_travel_m).max(1.0e-4)
}

fn suspension_coefficients(
    tuning: &GearTuning,
    wheel: &Wheel,
    static_mass_share: f64,
    compression_mass_share: f64,
    craft_mass_kg: f64,
    gravity_m_s2: f64,
) -> SuspensionCoefficients {
    let max_travel_m = (tuning.max_travel_fraction * wheel.strut_length).max(1.0e-3);
    let static_mass_kg = (craft_mass_kg * static_mass_share).max(1.0);
    let compression_mass_kg = (craft_mass_kg * compression_mass_share).max(1.0);
    let sag_m = nominal_static_sag_m(tuning, wheel.strut_length);
    let spring_n_per_m = static_mass_kg * gravity_m_s2.max(0.1) / sag_m;
    SuspensionCoefficients {
        max_travel_m,
        spring_n_per_m,
        damping_compress_n_s_per_m: 2.0
            * tuning.damping_ratio_compress
            * (spring_n_per_m * compression_mass_kg).sqrt(),
        damping_rebound_n_s_per_m: 2.0
            * tuning.damping_ratio_rebound
            * (spring_n_per_m * static_mass_kg).sqrt(),
    }
}

fn suspension_spring_force_n(
    tuning: &GearTuning,
    coefficients: SuspensionCoefficients,
    compression_m: f64,
) -> f64 {
    let compression_m = compression_m.clamp(0.0, coefficients.max_travel_m);
    let linear_n = coefficients.spring_n_per_m * compression_m;
    let bump_start_m = tuning.bump_stop_start_fraction.clamp(0.4, 0.95) * coefficients.max_travel_m;
    let over_m = (compression_m - bump_start_m).max(0.0);
    let bump_span_m = (coefficients.max_travel_m - bump_start_m).max(1.0e-6);
    let bump_n =
        coefficients.spring_n_per_m * tuning.bump_stop_stiffness_factor.max(0.0) * over_m.powi(2)
            / bump_span_m;
    linear_n + bump_n
}

/// Carry the craft on its wheels: a raycast spring/damper per wheel, plus
/// lateral grip, rolling resistance, brake, and emergent nosewheel-steer yaw.
///
/// Forces are summed into the craft's acceleration accumulators *after*
/// [`apply_local_forces`] has written gravity + thrust + reaction-wheel torque,
/// so this is a parallel channel on top of them. Runs only when Avian owns
/// translation ([`AvianRole::Full`]) for a live Ship — exactly when there is a
/// ground collider (runway slab or terrain patch) under the craft. The
/// craft-excluded downward raycast is itself the "is there ground here" test:
/// no hit → that wheel is airborne and contributes nothing. See `docs/simulation/surface.md`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_landing_gear_forces(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    tuning: Res<GearTuning>,
    gear_state: Res<GearState>,
    parking_brake: Res<ParkingBrake>,
    ground_control: Res<crate::control_bus::ResolvedGroundControl>,
    spatial: SpatialQuery,
    sim: Res<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    mut weight_on_wheels: ResMut<WeightOnWheels>,
    mut visual_compression: ResMut<GearVisualCompression>,
    wheels_q: Query<&WheelSet>,
    mut craft_q: Query<
        (
            &Position,
            &Rotation,
            &LinearVelocity,
            &AngularVelocity,
            &mut ConstantLinearAcceleration,
            &mut ConstantAngularAcceleration,
        ),
        With<LocalCraftBody>,
    >,
    mut last_emit_sim_s: Local<f64>,
) {
    // Default to airborne; any loaded wheel below flips this true. Cleared up
    // front so every early-return path (not owning translation, no gear, etc.)
    // correctly reports "no weight on wheels" and "no compression".
    weight_on_wheels.grounded = false;
    visual_compression.0.clear();
    // Gear retracted → no ground interface at all (weight-on-wheels stays
    // false, set above). Binary: there is no partial-deploy load state.
    if !gear_state.down {
        return;
    }
    // Full only: Avian must own translation for the integrated force to mean
    // anything. Ships only — EVA has no gear — and never a destroyed wreck.
    if !authority.owns_translation()
        || sim.simulation.vessel_kind() != VesselKind::Ship
        || sim.simulation.is_destroyed()
    {
        return;
    }
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Ok(wheelset) = wheels_q.get(bubble.craft_entity) else {
        return;
    };
    if wheelset.wheels.is_empty() {
        return;
    }
    let Ok((
        position,
        rotation,
        linear_velocity,
        angular_velocity,
        mut linear_accel,
        mut angular_accel,
    )) = craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };

    let rot = rotation.0;
    // Ships integrate in the body-fixed (rotating) frame and the ground
    // collider is static there, so the surface under every wheel reads ~0
    // velocity — `v_rel` is just the contact point's body-fixed velocity, with
    // no `ω × r` co-rotation term to subtract (and no phantom slip to pump a
    // spin). This is the payoff of the body-fixed frame.
    let mass = sim.simulation.ship_mass_kg().max(1.0);
    let inertia_body = sim.simulation.ship_params().moment_of_inertia;
    // Wheel torque is about the craft CoM (the Avian rotation pivot, set to
    // this same point at spawn), not the root origin — otherwise the upward
    // forces from gear that sit aft of the nose origin have no balancing
    // torque and flip the craft.
    let com_local = sim.simulation.ship_params().center_of_mass;
    // Analytic-surface fallback inputs for wheels whose ray misses (see below).
    let height_source = height_sources.get(bubble.body_id);
    let body_radius_m = sim.system.bodies[bubble.body_id].radius_m;
    let surface_gravity_m_s2 =
        sim.system.bodies[bubble.body_id].gm / body_radius_m.max(1.0).powi(2);
    let load_model = wheel_load_model(&wheelset.wheels, com_local);

    // Nosewheel steering: full tiller throw at taxi speed, fading toward zero
    // with ground speed (the aero rudder takes over) so a hard yaw input at
    // takeoff speed can't generate the lateral grip that trips the craft over
    // its main gear.
    let ground_speed = linear_velocity.0.length();
    let steer_scale = 1.0 / (1.0 + (ground_speed / tuning.steer_fade_speed_m_s).powi(2));
    let steer = ground_control.steer.clamp(-1.0, 1.0) * tuning.max_steer_rad * steer_scale;
    let brake_command = if parking_brake.engaged {
        1.0
    } else {
        ground_control.brake.clamp(0.0, 1.0)
    };
    let filter = SpatialQueryFilter::default().with_excluded_entities([bubble.craft_entity]);

    let mut net_force = DVec3::ZERO;
    let mut net_torque = DVec3::ZERO;
    // Per-frame contact stats for the 1 Hz `gear_contact` gauge below.
    let mut wheels_loaded = 0usize;
    let mut ray_misses = 0usize;
    let mut max_compression_frac = 0.0_f64;
    let mut normal_sum_n = 0.0_f64;
    for wheel in &wheelset.wheels {
        let static_mass_share = load_model.static_share(wheel, wheelset.wheels.len());
        let compression_mass_share =
            load_model.compression_mass_share(wheel, wheelset.wheels.len());
        let suspension = suspension_coefficients(
            &tuning,
            wheel,
            static_mass_share,
            compression_mass_share,
            mass,
            surface_gravity_m_s2,
        );
        let origin = position.0 + rot * wheel.strut_top_local;
        let down = rot * wheel.susp_dir_local;
        let Ok(dir) = Dir3::new(down.as_vec3()) else {
            continue;
        };
        // Geometric rest = wheel bottom on the surface (compression 0). The
        // craft sinks by the small static sag until the spring balances its load.
        let rest_len = wheel.strut_length + wheel.wheel_radius;
        // Start the ray *above* the strut top: a hard landing can put the hull
        // skin (and thus the strut top) below the terrain surface, and a ray
        // cast from underneath the heightfield hits nothing — the wheel would
        // silently unload (no normal force, no brakes) exactly while buried.
        // The lift is subtracted back out so distances stay strut-top-relative.
        let lift = tuning.ray_start_lift_m.max(0.0);
        let ray_origin = origin - down * lift;
        let max_len = lift + rest_len + tuning.skin_margin;
        // Distance from the strut top to the ground along the suspension axis.
        // Primary: the spatial ray (sees the runway slab, structures, terrain).
        // Fallback when it misses — buried deeper than the lift, or over a
        // not-yet-streamed patch edge: the same analytic height source the
        // floor backstop reads, so the gear can never lose a surface the
        // backstop is holding the hull against.
        let dist_to_ground = match spatial.cast_ray(ray_origin, dir, max_len, true, &filter) {
            Some(hit) => hit.distance - lift,
            None => {
                // Counted whether or not the analytic fallback rescues it: a
                // grounded craft whose rays keep missing is the tell the gauge
                // below exists to expose.
                ray_misses += 1;
                let Some(hs) = height_source.as_deref() else {
                    continue;
                };
                let r_pt = bubble.frame.body_center_offset(origin);
                let Some(up_radial) = r_pt.try_normalize() else {
                    continue;
                };
                let dir_body = bubble.frame.rotation_body_to_frame.inverse() * up_radial;
                let Some(h) = hs.sample_height_m(dir_body.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
                else {
                    continue;
                };
                // Strut-top height above the local surface, projected onto the
                // suspension axis. A ray pointing near-parallel to the surface
                // (craft on its side) has no meaningful wheel contact.
                let toward_ground = down.dot(-up_radial);
                if toward_ground < 0.2 {
                    continue;
                }
                (r_pt.length() - (body_radius_m + h as f64)) / toward_ground
            }
        };
        let max_travel = suspension.max_travel_m;
        let compression = (rest_len - dist_to_ground).clamp(0.0, max_travel);
        if compression <= 0.0 {
            continue;
        }
        // A wheel is bearing load: the craft has weight on its wheels, so the
        // aero pass should treat it as grounded and suppress tipping moments.
        weight_on_wheels.grounded = true;
        // Record the gearbox's deepest compression for the visual mesh offset.
        let slot = visual_compression
            .0
            .entry(wheel.source)
            .or_insert((wheel.susp_dir_local, 0.0));
        if compression > slot.1 {
            *slot = (wheel.susp_dir_local, compression);
        }
        let up = -down;
        // Contact point relative to the craft CoM: the arm that turns wheel
        // force into torque about the rotation pivot (lets steered front-wheel
        // grip yaw the craft, and lets nose/main share the load in pitch).
        let contact_local =
            wheel.strut_top_local + wheel.susp_dir_local * dist_to_ground.clamp(0.0, rest_len);
        let r_arm = rot * (contact_local - com_local);
        // Avian's LinearVelocity is the CoM velocity; the contact moves at
        // v_com + ω_craft × arm.
        // Ground is static in the body-fixed frame, so the contact's body-fixed
        // velocity is the slip directly.
        let v_rel = linear_velocity.0 + angular_velocity.0.cross(r_arm);

        // One-way spring + damper along the suspension axis (never pulls down).
        // The compression damper ramps in over the first fraction of travel so
        // the contact-onset force is continuous (at first touch the compression
        // rate is the full sink speed — an unramped damper is a step jolt);
        // rebound damping is always at full strength so a nearly-extended strut
        // still can't fling the wheel out.
        let compress_rate = -v_rel.dot(up);
        let damper_n = if compress_rate > 0.0 {
            let engage_depth = (tuning.damper_engage_frac * max_travel).max(1.0e-6);
            suspension.damping_compress_n_s_per_m
                * compress_rate
                * (compression / engage_depth).min(1.0)
        } else {
            suspension.damping_rebound_n_s_per_m * compress_rate
        };
        let normal_n =
            (suspension_spring_force_n(&tuning, suspension, compression) + damper_n).max(0.0);
        if normal_n <= 0.0 {
            continue;
        }
        wheels_loaded += 1;
        normal_sum_n += normal_n;
        max_compression_frac = max_compression_frac.max(compression / max_travel.max(1.0e-9));
        let mu_n = tuning.mu * normal_n;

        // Steer the nose wheel by rotating its roll/axle dirs about the strut.
        let (roll_local, axle_local) = if wheel.steerable && steer.abs() > 1.0e-6 {
            let q = DQuat::from_axis_angle(wheel.susp_dir_local.normalize_or_zero(), steer);
            (q * wheel.roll_dir_local, q * wheel.axle_dir_local)
        } else {
            (wheel.roll_dir_local, wheel.axle_dir_local)
        };
        let axle_w = (rot * axle_local).normalize_or_zero();
        let roll_w = (rot * roll_local).normalize_or_zero();

        // Lateral grip resists sideways slip; longitudinal resists roll. Both
        // clamped to the friction circle so they only ever remove ground-relative
        // speed, never propel.
        let f_lat = -axle_w * (tuning.k_lat * v_rel.dot(axle_w)).clamp(-mu_n, mu_n);
        let roll_speed = v_rel.dot(roll_w);
        // Parking brake engaged → high-gain fore/aft hold (pins the craft);
        // released → free rolling resistance only.
        let f_roll = if brake_command > 0.0 {
            -roll_w
                * (tuning.parking_brake_stiffness * brake_command * roll_speed).clamp(-mu_n, mu_n)
        } else {
            // Coulomb rolling resistance: a stiff hold clamped to a small
            // `μ_roll·N` cap. The constant (non-viscous) cap means a coasting
            // craft loses speed linearly and reaches a true stop in finite time,
            // then the stiff term holds it within the cap — instead of the old
            // `∝ v` law that decayed exponentially and crept forever.
            let roll_cap = (tuning.rolling_mu * normal_n).min(mu_n);
            -roll_w * (tuning.rolling_hold_stiffness * roll_speed).clamp(-roll_cap, roll_cap)
        };

        let f = up * normal_n + f_lat + f_roll;
        net_force += f;
        net_torque += r_arm.cross(f);
    }

    // 1 Hz ground-contact gauge, emitted while any wheel bears load — an
    // airborne gear-down approach (all rays legitimately missing) stays
    // silent, so the lane isn't dominated by cruise. Answers "was the craft
    // carried by its wheels, and how hard?" from the log instead of a
    // screenshot: wheels loaded of total, rays that missed (a loaded wheel
    // with `ray_misses > 0` was rescued by the analytic fallback), the deepest
    // stroke as a fraction of max travel, and the summed normal load. The
    // zero-load-while-grounded pathology is covered by the backstop's
    // `backstop_intervention` event (`just diag` · gear_carried_by_backstop).
    if wheels_loaded > 0 {
        let sim_time_s = sim.simulation.sim_time();
        if sim_time_s - *last_emit_sim_s >= 1.0 || sim_time_s < *last_emit_sim_s {
            *last_emit_sim_s = sim_time_s;
            info!(
                target: "thalos::diagnostic::local_physics",
                event = "gear_contact",
                wheels = wheelset.wheels.len(),
                wheels_loaded,
                ray_misses,
                max_compression_frac,
                normal_sum_n,
                "landing gear ground contact"
            );
        }
    }

    if net_force == DVec3::ZERO && net_torque == DVec3::ZERO {
        return;
    }
    // Inertia-relative safety clamp, mirroring the aero force model
    // (`crate::aero`): a real undercarriage imparts at most a few g and a few
    // rad/s², so bounding the per-frame gear acceleration to the craft's own
    // mass/MOI makes a stiff-spring numerical blow-up impossible — a single bad
    // frame (or a discrete-step pumping cycle) can no longer spike the craft to
    // hundreds of rad/s and fling it off the runway — while leaving normal
    // taxi/landing loads (well under these limits) untouched. Without this the
    // gear was the one unclamped force path; see `docs/simulation/surface_local.md`.
    const GEAR_MAX_LIN_ACCEL_M_S2: f64 = 50.0; // ~5 g
    const GEAR_MAX_ANG_ACCEL_RAD_S2: f64 = 4.0;
    let lin_accel = net_force / mass;
    let lin_len = lin_accel.length();
    let lin_accel = if lin_len > GEAR_MAX_LIN_ACCEL_M_S2 {
        lin_accel * (GEAR_MAX_LIN_ACCEL_M_S2 / lin_len)
    } else {
        lin_accel
    };
    linear_accel.0 += lin_accel;

    let torque_body = rot.inverse() * net_torque;
    let inv_i = DVec3::new(
        if inertia_body.x > 0.0 {
            1.0 / inertia_body.x
        } else {
            0.0
        },
        if inertia_body.y > 0.0 {
            1.0 / inertia_body.y
        } else {
            0.0
        },
        if inertia_body.z > 0.0 {
            1.0 / inertia_body.z
        } else {
            0.0
        },
    );
    let ang_accel = torque_body * inv_i;
    let ang_len = ang_accel.length();
    let ang_accel = if ang_len > GEAR_MAX_ANG_ACCEL_RAD_S2 {
        ang_accel * (GEAR_MAX_ANG_ACCEL_RAD_S2 / ang_len)
    } else {
        ang_accel
    };
    angular_accel.0 += rot * ang_accel;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wheel(y_m: f64, strut_length_m: f64, steerable: bool) -> Wheel {
        Wheel {
            source: Entity::PLACEHOLDER,
            strut_top_local: DVec3::new(0.0, y_m, -1.0),
            susp_dir_local: -DVec3::Z,
            roll_dir_local: DVec3::Y,
            axle_dir_local: DVec3::X,
            strut_length: strut_length_m,
            wheel_radius: 0.6,
            steerable,
        }
    }

    #[test]
    fn meridian_gear_uses_its_real_axle_load_split() {
        // Geometry and CoM from `meridian_balance`; this is the shipped craft
        // that produced session 28324-1785560800045.
        let wheels = [
            wheel(-5.60, 1.3, true),
            wheel(-17.85, 1.2, false),
            wheel(-17.85, 1.2, false),
        ];
        let model = wheel_load_model(&wheels, DVec3::new(0.0, -16.363, 0.165));
        let nose = model.static_share(&wheels[0], wheels.len());
        let main = model.static_share(&wheels[1], wheels.len());
        assert!((nose - 0.1214).abs() < 0.001);
        assert!((main - 0.4393).abs() < 0.001);
        assert!((nose + 2.0 * main - 1.0).abs() < 1.0e-12);
        assert_eq!(model.compression_mass_share(&wheels[1], wheels.len()), 0.5);
    }

    #[test]
    fn suspension_static_force_matches_each_axle_load() {
        let tuning = GearTuning::default();
        let mass_kg = 40_528.8;
        let gravity_m_s2 = 8.88;
        let wheels = [
            wheel(-5.60, 1.3, true),
            wheel(-17.85, 1.2, false),
            wheel(-17.85, 1.2, false),
        ];
        let model = wheel_load_model(&wheels, DVec3::new(0.0, -16.363, 0.165));
        let mut normal_sum_n = 0.0;
        let mut damping = [0.0; 3];
        for (index, gear_wheel) in wheels.iter().enumerate() {
            let static_share = model.static_share(gear_wheel, wheels.len());
            let coefficients = suspension_coefficients(
                &tuning,
                gear_wheel,
                static_share,
                model.compression_mass_share(gear_wheel, wheels.len()),
                mass_kg,
                gravity_m_s2,
            );
            let sag_m = tuning.static_sag_fraction * coefficients.max_travel_m;
            normal_sum_n += suspension_spring_force_n(&tuning, coefficients, sag_m);
            damping[index] = coefficients.damping_compress_n_s_per_m;
        }
        assert!((normal_sum_n - mass_kg * gravity_m_s2).abs() < 1.0);
        assert!(
            damping[1] > damping[0] * 3.0,
            "a main oleo must be tuned for its much larger impact load"
        );
        assert_eq!(damping[1], damping[2]);
    }

    fn drop_response(
        tuning: &GearTuning,
        coefficients: SuspensionCoefficients,
        impact_mass_kg: f64,
        static_weight_n: f64,
        initial_sink_m_s: f64,
    ) -> (f64, f64) {
        // One main leg at 2048 Hz, deliberately much finer than runtime: this
        // tests the suspension law rather than integration error. Positive
        // velocity/compression is downward/inward.
        let dt = 1.0 / 2048.0;
        let mut compression_m: f64 = 1.0e-6;
        let mut velocity_m_s: f64 = initial_sink_m_s;
        let mut max_compression_frac = 0.0_f64;
        for _ in 0..(4 * 2048) {
            let contact_compression = compression_m.clamp(0.0, coefficients.max_travel_m);
            let engage_depth = (tuning.damper_engage_frac * coefficients.max_travel_m).max(1.0e-6);
            let damper_n = if velocity_m_s > 0.0 {
                coefficients.damping_compress_n_s_per_m
                    * velocity_m_s
                    * (contact_compression / engage_depth).min(1.0)
            } else {
                coefficients.damping_rebound_n_s_per_m * velocity_m_s
            };
            let normal_n = (suspension_spring_force_n(tuning, coefficients, contact_compression)
                + damper_n)
                .max(0.0);
            velocity_m_s += (static_weight_n - normal_n) / impact_mass_kg * dt;
            compression_m += velocity_m_s * dt;
            max_compression_frac =
                max_compression_frac.max(compression_m / coefficients.max_travel_m);
            if compression_m <= 0.0 && velocity_m_s < 0.0 {
                return (max_compression_frac, -velocity_m_s);
            }
        }
        (max_compression_frac, 0.0)
    }

    #[test]
    fn meridian_main_gear_absorbs_airliner_certification_sink_without_bottoming() {
        // 3.05 m/s is the transport-aircraft limit drop-test sink rate. The
        // regression is intentionally harsher than the autoland target so a
        // modest controller miss still stays in the oleo, not the hull floor.
        let tuning = GearTuning::default();
        let main = wheel(-17.85, 1.2, false);
        let craft_mass_kg = 40_528.8;
        let gravity_m_s2 = 8.88;
        let static_share = 0.4393;
        let impact_share = 0.5;
        let coefficients = suspension_coefficients(
            &tuning,
            &main,
            static_share,
            impact_share,
            craft_mass_kg,
            gravity_m_s2,
        );
        let (max_compression_frac, rebound_m_s) = drop_response(
            &tuning,
            coefficients,
            craft_mass_kg * impact_share,
            craft_mass_kg * static_share * gravity_m_s2,
            3.05,
        );
        assert!(
            max_compression_frac < tuning.bump_stop_start_fraction,
            "certification drop reached the end stop at {:.1}% stroke",
            max_compression_frac * 100.0
        );
        assert!(
            rebound_m_s < 0.75,
            "oleo returned {:.2} m/s instead of settling",
            rebound_m_s
        );
    }
}
