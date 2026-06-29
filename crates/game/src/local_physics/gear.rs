//! Raycast spring-damper landing gear: wheels, tuning, parking brake, ground forces.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/regimes.md`).

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
use thalos_physics_local::{ActiveLocalBubble, LocalCraftBody};
use thalos_shipyard::{AttachNodes, Gear, Part, SurfaceMount, SurfaceMountKind, gear_leg_frames};

use crate::rendering::SimulationState;

/// One landing-gear wheel as a **raycast suspension**, in the craft body frame.
///
/// All directions/points are craft-local (`X=right, Y=nose, Z=dorsal`) — the
/// same frame `gear_mesh` authors in — so `Rotation.0 * p` maps them into the
/// body-centered inertial frame the Avian rigid body lives in. Built once at
/// spawn from the gear parts ([`build_wheel_set`]) and cached so the per-frame
/// system does no part-tree walking.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct Wheel {
    /// Strut top at the host skin — the suspension ray origin.
    pub strut_top_local: DVec3,
    /// Suspension axis (belly-ward `r̂`): the ray direction and spring line.
    pub susp_dir_local: DVec3,
    /// Roll axis (`fore`): brake / rolling resistance act along this.
    pub roll_dir_local: DVec3,
    /// Axle axis (`lateral`): lateral grip resists slip along this.
    pub axle_dir_local: DVec3,
    pub strut_length: f64,
    pub wheel_radius: f64,
    /// Nose (single-leg) gear steers; main pairs do not.
    pub steerable: bool,
}

/// Every wheel on a craft, attached to its Avian rigid body so
/// [`apply_landing_gear_forces`] can find them.
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct WheelSet {
    pub wheels: Vec<Wheel>,
}

/// Landing-gear suspension/grip coefficients. Reflect-registered (for a future
/// debug UI); edit the defaults and rebuild to tune them. Forces are computed
/// per wheel and summed into the craft's acceleration accumulators.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct GearTuning {
    /// Spring stiffness, N per metre of compression.
    pub k_spring: f64,
    /// Suspension damping ratio ζ. The actual damper coefficient is derived
    /// per frame from the craft's real per-wheel mass —
    /// `c = 2·ζ·√(k·m_per_wheel)` — so a wheel is near-critically damped (no
    /// bounce, no spin-pumping) regardless of craft mass. ζ = 1 is critical;
    /// slightly above gives a dead-beat settle.
    pub damping_ratio: f64,
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
}

impl Default for GearTuning {
    fn default() -> Self {
        // Sized for the ~20–40 t demo aircraft on Thalos surface gravity. The
        // craft settles to a static squat of `(m·g/N)/k_spring`; these put that
        // in the tens-of-cm range with near-critical damping.
        Self {
            // Stiff enough that the static sag `m·g/(n·k)` is ~cm-scale for the
            // demo aircraft (so the rigid wheel meshes don't visibly clip the
            // ground), but not so stiff it rings at the 60 Hz step — the
            // near-critical `damping_ratio` keeps it dead-beat.
            k_spring: 800_000.0,
            damping_ratio: 1.2,
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
        }
    }
}

/// Latched brakes (KSP-style, the B key). When engaged,
/// [`apply_landing_gear_forces`] replaces free rolling with a high-gain
/// fore/aft hold (clamped to the tyre friction circle), so the craft stays
/// put under gravity, slopes, and the residual settle — though full takeoff
/// thrust still overpowers it — and the spoilers deploy
/// ([`crate::flight_config`]), so the same latch is the in-air speedbrake
/// and the rollout lift dump.
///
/// Defaults **off** (most spawns are airborne and must not start with
/// spoilers out); the parked runway placement engages it explicitly so a
/// freshly-spawned aircraft holds on the strip
/// (`runway::finish_runway_spawn`). Reflect-registered (for a future debug UI).
#[derive(Resource, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct ParkingBrake {
    pub engaged: bool,
}

/// Whether any landing-gear wheel is currently bearing load on the ground
/// ("weight on wheels"). Set each frame by [`apply_landing_gear_forces`] from
/// its per-wheel suspension raycast, and read in the aero pass
/// ([`crate::aero::apply_aero_forces`]) to drop all aero on a grounded craft
/// below the taxi airspeed floor, where the AoA is degenerate (the velocity is
/// suspension settle, not flow). Above that floor a grounded craft flies the
/// full aero model — rotation authority and ground-roll damping are real
/// aerodynamics. Reflect-registered (for a future debug UI).
#[derive(Resource, Default, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct WeightOnWheels {
    pub grounded: bool,
}

/// Landing-gear up/down latch (KSP-style, the G key). When `down`,
/// [`apply_landing_gear_forces`] runs the suspension; when up it stands down
/// entirely (no contact, no weight on wheels) and the gear meshes are hidden
/// (`ship_view::sync_gear_visibility`). Binary — there is no retraction
/// animation, so a half-deployed load state never exists.
///
/// Defaults **down**: every ground/approach spawn (runway, final, descent) needs
/// gear extended, and orbit/EVA craft have no wheels so the state is moot.
/// Retraction is interlocked against weight-on-wheels (see [`toggle_gear`]).
/// Reflect-registered (for a future debug UI).
#[derive(Resource, Clone, Copy, Debug, Reflect)]
#[reflect(Resource)]
pub struct GearState {
    pub down: bool,
}

impl Default for GearState {
    fn default() -> Self {
        Self { down: true }
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

/// Apply a requested gear position through the weight-on-wheels interlock.
/// Shared by the key ([`toggle_gear`]) and the HUD pill so both honour the same
/// rule: extending is always allowed; retracting is refused while grounded.
pub(crate) fn set_gear_down(gear: &mut GearState, weight_on_wheels: &WeightOnWheels, down: bool) {
    if down || !weight_on_wheels.grounded {
        gear.down = down;
    }
}

/// Build the cached [`WheelSet`] for a craft from its landing-gear parts,
/// reusing [`gear_leg_frames`] (the same per-leg geometry the visual mesh
/// draws) so collider wheels sit exactly under the rendered ones. `positions`
/// is the part-tree translation map ([`compute_part_collider_positions`]); the
/// gear's mount point on the host axis mirrors the `BodySkin` station offset
/// that map applies.
/// Landing-gear contact geometry for parked placement: the lowest wheel-bottom
/// depth below the craft origin along the ventral (−Z) axis, plus the wheel
/// count. Derived from the **gear contact geometry** ([`build_wheel_set`], the
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
    gear_q: &Query<
        (&Gear, &SurfaceMount),
        (With<Part>, Without<thalos_shipyard::editor::EditorPart>),
    >,
    host_nodes: &Query<&AttachNodes>,
) -> Option<(f64, usize)> {
    let positions = compute_part_collider_positions(parts);
    let wheels = build_wheel_set(gear_q, host_nodes, &positions);
    if wheels.is_empty() {
        return None;
    }
    let lowest = wheels.iter().fold(f64::INFINITY, |acc, w| {
        let bottom = w.strut_top_local + w.susp_dir_local * (w.strut_length + w.wheel_radius);
        acc.min(bottom.z)
    });
    (lowest.is_finite() && lowest < 0.0).then_some((-lowest, wheels.len()))
}

pub(crate) fn build_wheel_set(
    gear_q: &Query<
        (&Gear, &SurfaceMount),
        (With<Part>, Without<thalos_shipyard::editor::EditorPart>),
    >,
    host_nodes: &Query<&AttachNodes>,
    positions: &HashMap<Entity, DVec3>,
) -> Vec<Wheel> {
    let mut wheels = Vec::new();
    for (gear, mount) in gear_q.iter() {
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

/// Carry the craft on its wheels: a raycast spring/damper per wheel, plus
/// lateral grip, rolling resistance, brake, and emergent nosewheel-steer yaw.
///
/// Forces are summed into the craft's acceleration accumulators *after*
/// [`apply_local_forces`] has written gravity + thrust + reaction-wheel torque,
/// so this is a parallel channel on top of them. Runs only when Avian owns
/// translation ([`AvianRole::Full`]) for a live Ship — exactly when there is a
/// ground collider (runway slab or terrain patch) under the craft. The
/// craft-excluded downward raycast is itself the "is there ground here" test:
/// no hit → that wheel is airborne and contributes nothing. See `docs/surface.md`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_landing_gear_forces(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    tuning: Res<GearTuning>,
    gear_state: Res<GearState>,
    parking_brake: Res<ParkingBrake>,
    intent: Res<GameInputIntent>,
    spatial: SpatialQuery,
    sim: Res<SimulationState>,
    mut weight_on_wheels: ResMut<WeightOnWheels>,
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
) {
    // Default to airborne; any loaded wheel below flips this true. Cleared up
    // front so every early-return path (not owning translation, no gear, etc.)
    // correctly reports "no weight on wheels".
    weight_on_wheels.grounded = false;
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
    // Near-critical damper derived from the *actual* per-wheel mass, so the
    // suspension settles dead-beat on any craft instead of bouncing.
    let m_per_wheel = mass / wheelset.wheels.len().max(1) as f64;
    let c_damp = 2.0 * tuning.damping_ratio * (tuning.k_spring * m_per_wheel).sqrt();
    // Note on ride height: the suspension finds its own torque-balanced
    // equilibrium (loaded wheels compress more), which is what keeps the craft
    // upright — do NOT preload the spring uniformly to cancel the sag, that
    // unbalances the torque and tips the craft over. Instead `k_spring` is sized
    // stiff enough that the static sag `m·g/(n·k)` is small (a couple cm), so the
    // rigid wheel meshes barely dip below the surface.

    // Nosewheel steering: full tiller throw at taxi speed, fading toward zero
    // with ground speed (the aero rudder takes over) so a hard yaw input at
    // takeoff speed can't generate the lateral grip that trips the craft over
    // its main gear.
    let ground_speed = linear_velocity.0.length();
    let steer_scale = 1.0 / (1.0 + (ground_speed / tuning.steer_fade_speed_m_s).powi(2));
    let steer = (intent.attitude.z as f64).clamp(-1.0, 1.0) * tuning.max_steer_rad * steer_scale;
    let filter = SpatialQueryFilter::default().with_excluded_entities([bubble.craft_entity]);

    let mut net_force = DVec3::ZERO;
    let mut net_torque = DVec3::ZERO;
    for wheel in &wheelset.wheels {
        let origin = position.0 + rot * wheel.strut_top_local;
        let down = rot * wheel.susp_dir_local;
        let Ok(dir) = Dir3::new(down.as_vec3()) else {
            continue;
        };
        // Geometric rest = wheel bottom on the surface (compression 0). The
        // craft sinks by the small static sag until the spring balances its load.
        let rest_len = wheel.strut_length + wheel.wheel_radius;
        let max_len = rest_len + tuning.skin_margin;
        let Some(hit) = spatial.cast_ray(origin, dir, max_len, true, &filter) else {
            continue;
        };
        let max_travel = tuning.max_travel_fraction * wheel.strut_length;
        let compression = (rest_len - hit.distance).clamp(0.0, max_travel);
        if compression <= 0.0 {
            continue;
        }
        // A wheel is bearing load: the craft has weight on its wheels, so the
        // aero pass should treat it as grounded and suppress tipping moments.
        weight_on_wheels.grounded = true;
        let up = -down;
        // Contact point relative to the craft CoM: the arm that turns wheel
        // force into torque about the rotation pivot (lets steered front-wheel
        // grip yaw the craft, and lets nose/main share the load in pitch).
        let contact_local = wheel.strut_top_local + wheel.susp_dir_local * hit.distance;
        let r_arm = rot * (contact_local - com_local);
        // Avian's LinearVelocity is the CoM velocity; the contact moves at
        // v_com + ω_craft × arm.
        // Ground is static in the body-fixed frame, so the contact's body-fixed
        // velocity is the slip directly.
        let v_rel = linear_velocity.0 + angular_velocity.0.cross(r_arm);

        // One-way spring + damper along the suspension axis (never pulls down).
        let compress_rate = -v_rel.dot(up);
        let normal_n = (tuning.k_spring * compression + c_damp * compress_rate).max(0.0);
        if normal_n <= 0.0 {
            continue;
        }
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
        let f_roll = if parking_brake.engaged {
            -roll_w * (tuning.parking_brake_stiffness * roll_speed).clamp(-mu_n, mu_n)
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
    // gear was the one unclamped force path; see `docs/surface_local.md`.
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
