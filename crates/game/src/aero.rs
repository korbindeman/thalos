//! Atmospheric aerodynamics for the local-physics bubble.
//!
//! Thalos uses a small native whole-body flight model
//! ([`thalos_physics_canonical::aero`]): given the craft's air-relative velocity
//! and angular rate in its body frame plus an [`AeroConfig`], it returns a net
//! force + torque about the CoM. This module is the Bevy adapter:
//!
//! - **Build** the config from the blueprint's wing parts
//!   ([`build_ship_aero_config`]): wing area / chord / span → lift + stability,
//!   or a bluff-body drag config for a wingless rocket/capsule.
//! - **Drive** it each physics step ([`apply_aero_forces`]): read the Avian
//!   `LinearVelocity`/`AngularVelocity` (surface-relative in the body-fixed
//!   bubble frame, so wind = 0), sample the body's atmosphere for density, call
//!   the evaluator, and write the result into the craft's
//!   `ConstantForce`/`ConstantTorque`. Thalos still owns mass, inertia, gravity,
//!   and thrust; the aero force just *sums* into the solver.
//!
//! Frame: the body frame is X=right, Y=nose, Z=dorsal. The model's stability and
//! damping are explicit and unconditionally stable (see the evaluator docs); the
//! airfoil / control constants below are a first cut tuned in-game (see
//! `docs/aerodynamics.md`).

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics_canonical::aero::{AeroConfig, evaluate_aero};
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::ActiveLocalBubble;
use thalos_physics_local::LocalCraftBody;
use thalos_physics_local::avian::{
    AngularVelocity, ConstantForce, ConstantTorque, LinearVelocity, Physics, PhysicsDebugPlugin,
    PhysicsGizmos, PhysicsSchedule, PhysicsStepSystems, Position, Rotation,
};
use thalos_shipyard::{WingAeroPanel, WingRole};

use crate::rendering::{PlayerShip, SimulationState};

// --- Airfoil / stability / control constants ---------------------------------
/// Lift-curve slope, per radian (3-D, below the 2-D 2π for finite AR).
const LIFT_SLOPE_PER_RAD: f64 = 5.0;
/// Camber lift at zero angle of attack for a cambered main wing.
const MAIN_WING_CL0: f64 = 0.25;
/// Pitch trim moment at zero AoA: trims the statically-stable airframe at
/// `α_trim = cm0 / pitch_stability` ≈ 1.4°, the cruise attitude, so level
/// flight is hands-off instead of constant forward stick.
const MAIN_WING_CM0: f64 = 0.03;
/// Parasitic (zero-lift) drag coefficient of a winged aircraft.
const WING_PARASITIC_CD: f64 = 0.03;
/// Stall angle (rad) where |CL| is clamped.
const STALL_ANGLE_RAD: f64 = 0.26; // ~15°

// Non-dimensional moment coefficients for a winged aircraft. Restoring > 0 gives
// static stability; damping > 0 always opposes the rate; control sets pilot
// authority. Derived from transport-category stability derivatives (Cm_α ≈ −1.2,
// Cm_q ≈ −25 incl. the α̇ lag this model lacks, Cl_p ≈ −0.45, Cn_r ≈ −0.3,
// full-throw Cl_δa ≈ 0.06 / Cm_δe ≈ 0.5) mapped to this model's scaling: the
// damping term is `coeff·ρ·V·S·L²·ω` = 4× the standard `C_q·(ωL/2V)` form, so
// `coeff = C/4`, with the reference span being the **full wingspan**.
//
// What this buys is *felt inertia from real physics*: rate onset is governed by
// `τ = I / (damp·ρ·V·S·L²)`, which lands at ~1.2 s in roll for the ~37 t
// Meridian (rates build over a second-plus and coast to a stop) and a few
// tenths of a second for a fighter-sized airframe — heavy planes feel heavy and
// small ones nimble through their actual mass and geometry, not per-class
// tuning. Full deflection commands the real physical capability (an airliner
// *can* roll at ~35°/s and pull to stall AoA; its pilots just don't), so
// gentle inputs fly gently. Live-tunable at runtime via [`AeroTuning`].
const WING_PITCH_STABILITY: f64 = 1.2;
const WING_YAW_STABILITY: f64 = 0.2;
const WING_PITCH_DAMP: f64 = 8.0;
const WING_ROLL_DAMP: f64 = 0.13;
const WING_YAW_DAMP: f64 = 0.2;
const WING_PITCH_CONTROL: f64 = 0.5;
const WING_ROLL_CONTROL: f64 = 0.06;
const WING_YAW_CONTROL: f64 = 0.04;

// Bluff body (rocket/capsule): no lift, no control, but weathervane-stable.
const BLUFF_STABILITY: f64 = 0.5;
const BLUFF_DAMP: f64 = 0.5;

/// Airspeed (m/s) below which a grounded craft gets no aero at all (forces or
/// moments) — near-zero speed gives a degenerate AoA (the velocity is mostly
/// suspension settle). Above it, a grounded craft flies the full aero model.
const GROUND_AERO_AIRSPEED_FLOOR_M_S: f64 = 5.0;
/// Inertia-relative safety ceilings (see `apply_aero_forces`): a real craft
/// pulls only a few g / a few rad/s², so bound the aero force/torque by the
/// craft's own mass/MOI. Generous enough not to bind normal flight.
const MAX_LIN_ACCEL_M_S2: f64 = 100.0; // ~10 g (covers steep reentry drag)
const MAX_ANG_ACCEL_RAD_S2: f64 = 4.0;

/// The aircraft's aerodynamic config, computed from its blueprint by `ship_view`
/// and consumed when the Avian body spawns. Replaced on each spawn.
#[derive(Resource)]
pub struct ShipAeroLayout {
    pub config: AeroConfig,
}

impl Default for ShipAeroLayout {
    fn default() -> Self {
        Self { config: AeroConfig::default() }
    }
}

/// The live aero config attached to the player's Avian body.
#[derive(Component)]
pub struct ShipAero {
    pub config: AeroConfig,
}

/// Apply the live-tunable winged-aircraft moment coefficients to a base config,
/// exactly as [`apply_aero_forces`] does before evaluating. The control
/// allocator ([`crate::control_bus`]) calls this so the aero authority it splits
/// against is the same config the evaluator flies — bluff bodies (no
/// `lift_slope`) keep their authored coefficients.
pub(crate) fn resolved_aero_config(base: AeroConfig, tuning: &AeroTuning) -> AeroConfig {
    let mut config = base;
    if config.lift_slope > 0.0 {
        config.pitch_stability = tuning.pitch_stability;
        config.yaw_stability = tuning.yaw_stability;
        config.pitch_damp = tuning.pitch_damp;
        config.roll_damp = tuning.roll_damp;
        config.yaw_damp = tuning.yaw_damp;
        config.pitch_control = tuning.pitch_control;
        config.roll_control = tuning.roll_control;
        config.yaw_control = tuning.yaw_control;
    }
    config
}

/// Runtime-tunable handling coefficients for **winged** aircraft, overriding the
/// per-craft config's moment terms each frame. Reflect-registered so the feel
/// (control authority, damping, static stability) can be dialled in live over BRP
/// — e.g. `world_mutate_resources` on `thalos_game::aero::AeroTuning` — without a
/// rebuild. Defaults are the transport-derivative constants above.
#[derive(Resource, Reflect, Clone, Copy)]
#[reflect(Resource)]
pub struct AeroTuning {
    pub pitch_stability: f64,
    pub yaw_stability: f64,
    pub pitch_damp: f64,
    pub roll_damp: f64,
    pub yaw_damp: f64,
    pub pitch_control: f64,
    pub roll_control: f64,
    pub yaw_control: f64,
}

impl Default for AeroTuning {
    fn default() -> Self {
        Self {
            pitch_stability: WING_PITCH_STABILITY,
            yaw_stability: WING_YAW_STABILITY,
            pitch_damp: WING_PITCH_DAMP,
            roll_damp: WING_ROLL_DAMP,
            yaw_damp: WING_YAW_DAMP,
            pitch_control: WING_PITCH_CONTROL,
            roll_control: WING_ROLL_CONTROL,
            yaw_control: WING_YAW_CONTROL,
        }
    }
}

/// Flight readout for the HUD / BRP.
#[derive(Component, Default, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct AeroReadout {
    pub airspeed_ms: f64,
    pub dynamic_pressure_pa: f64,
    pub mach: f64,
    pub density_kgm3: f64,
    /// Net aero force magnitude (N) and the angle of attack (deg), for debug.
    pub force_n: f64,
    pub alpha_deg: f64,
}

/// Debug snapshot for the F3 overlay: net force + relative wind, body frame.
#[derive(Component, Default)]
pub struct AeroForceViz {
    pub force_body: Vec3,
    pub vel_body: Vec3,
}

/// Debug gizmo group for the aero force/wind overlay (toggled by F3).
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct AeroGizmos;

/// Wires the native aero force model into the local bubble.
pub struct GameAeroPlugin;

impl Plugin for GameAeroPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsDebugPlugin::default())
            .init_gizmo_group::<AeroGizmos>()
            .register_type::<AeroReadout>()
            .register_type::<AeroTuning>()
            .init_resource::<ShipAeroLayout>()
            .init_resource::<AeroTuning>()
            .add_systems(Startup, init_debug_overlay)
            .add_systems(
                PhysicsSchedule,
                apply_aero_forces.in_set(PhysicsStepSystems::BroadPhase),
            )
            .add_systems(
                PostUpdate,
                draw_aero_debug.after(bevy::transform::TransformSystems::Propagate),
            )
            .add_systems(Update, (attach_ship_aero, toggle_debug_overlay));
    }
}

/// Start the debug overlays disabled (toggled by F3).
fn init_debug_overlay(mut store: ResMut<GizmoConfigStore>) {
    store.config_mut::<AeroGizmos>().0.enabled = false;
    store.config_mut::<PhysicsGizmos>().0.enabled = false;
}

/// **F3** toggles the aero debug overlay: net-force / relative-wind vectors on
/// the aircraft. (The physics *hitbox* overlay — craft / gear / ground colliders
/// — shares this key but is drawn by `debug::draw_debug_hitboxes`, since Avian's
/// built-in `PhysicsGizmos` can't be placed correctly under big_space.)
fn toggle_debug_overlay(keys: Res<ButtonInput<KeyCode>>, mut store: ResMut<GizmoConfigStore>) {
    if !keys.just_pressed(KeyCode::F3) {
        return;
    }
    let on = !store.config::<AeroGizmos>().0.enabled;
    store.config_mut::<AeroGizmos>().0.enabled = on;
    info!("aero debug overlay (F3): {}", if on { "ON" } else { "off" });
}

/// Build the whole-body aero config for a craft from its wing panels.
///
/// Aircraft (panels present): reference area = total lifting (non-vertical)
/// panel area, chord = mean aerodynamic chord, span = max panel span; cambered
/// lift + control. Wingless (no panels): a bluff-body drag config sized from the
/// frontal area, no lift/control but weathervane-stable.
pub fn build_ship_aero_config(
    panels: &[WingAeroPanel],
    frontal_area_m2: f64,
    drag_cd: f64,
) -> AeroConfig {
    let mut total_area = 0.0;
    let mut mac_acc = 0.0;
    let mut max_span = 0.0_f64;
    for panel in panels {
        // Skip vertical fins (they don't contribute lifting area / chord).
        let vertical = !matches!(panel.role, WingRole::Lift)
            && (panel.angle.abs() < 0.6 || (std::f64::consts::PI - panel.angle.abs()).abs() < 0.6);
        if vertical {
            continue;
        }
        total_area += panel.area_m2;
        mac_acc += panel.chord_m * panel.area_m2;
        max_span = max_span.max(panel.span_m);
    }
    // Panels are single half-wings (mirrored pairs are separate entities), so
    // the aerodynamic reference span — the roll/yaw moment arm and the
    // aspect-ratio basis — is the full tip-to-tip wingspan, two panels.
    let full_span = 2.0 * max_span;

    if total_area > 0.0 {
        AeroConfig {
            reference_area_m2: total_area,
            reference_chord_m: mac_acc / total_area,
            reference_span_m: full_span.max(1.0),
            lift_slope: LIFT_SLOPE_PER_RAD,
            cl0: MAIN_WING_CL0,
            cm0: MAIN_WING_CM0,
            cd0: WING_PARASITIC_CD,
            stall_alpha: STALL_ANGLE_RAD,
            aspect_ratio: (full_span * full_span / total_area).clamp(1.0, 20.0),
            pitch_stability: WING_PITCH_STABILITY,
            yaw_stability: WING_YAW_STABILITY,
            pitch_damp: WING_PITCH_DAMP,
            roll_damp: WING_ROLL_DAMP,
            yaw_damp: WING_YAW_DAMP,
            pitch_control: WING_PITCH_CONTROL,
            roll_control: WING_ROLL_CONTROL,
            yaw_control: WING_YAW_CONTROL,
        }
    } else {
        // Bluff body: a body-length proxy from the frontal area as the moment arm.
        let body_len = (frontal_area_m2.max(1.0).sqrt() * 2.0).max(1.0);
        AeroConfig {
            reference_area_m2: frontal_area_m2.max(1.0),
            reference_chord_m: body_len,
            reference_span_m: body_len,
            lift_slope: 0.0,
            cl0: 0.0,
            cm0: 0.0,
            cd0: drag_cd,
            stall_alpha: STALL_ANGLE_RAD,
            aspect_ratio: 0.0,
            pitch_stability: BLUFF_STABILITY,
            yaw_stability: BLUFF_STABILITY,
            pitch_damp: BLUFF_DAMP,
            roll_damp: BLUFF_DAMP,
            yaw_damp: BLUFF_DAMP,
            pitch_control: 0.0,
            roll_control: 0.0,
            yaw_control: 0.0,
        }
    }
}

/// Attach the aero config + force accumulators to the player ship's Avian body,
/// once, after it spawns. Idempotent via `Without<ShipAero>`. EVA skipped.
fn attach_ship_aero(
    mut commands: Commands,
    sim: Res<SimulationState>,
    layout: Res<ShipAeroLayout>,
    ships: Query<Entity, (With<LocalCraftBody>, Without<ShipAero>)>,
) {
    if sim.simulation.vessel_kind() != VesselKind::Ship {
        return;
    }
    for ship in &ships {
        commands.entity(ship).insert((
            ShipAero { config: layout.config },
            AeroReadout::default(),
            AeroForceViz::default(),
            ConstantForce::default(),
            ConstantTorque::default(),
        ));
    }
}

/// Per physics step: sample the atmosphere, read the body-frame air-relative
/// velocity / rate, evaluate the native aero model, and write the resulting
/// force + torque into the craft's `ConstantForce` / `ConstantTorque`.
///
/// Lives in [`PhysicsSchedule`] so it only runs while physics is stepping —
/// never under warp/pause or the `BodyFixed` regime.
#[allow(clippy::type_complexity)]
fn apply_aero_forces(
    realized: Res<crate::control_bus::RealizedControl>,
    sim: Res<SimulationState>,
    active: Res<ActiveLocalBubble>,
    phys_time: Res<Time<Physics>>,
    tuning: Res<AeroTuning>,
    weight_on_wheels: Res<crate::local_physics::WeightOnWheels>,
    mut craft: Query<
        (
            &Position,
            &Rotation,
            &LinearVelocity,
            &AngularVelocity,
            &ShipAero,
            &mut ConstantForce,
            &mut ConstantTorque,
            &mut AeroReadout,
            &mut AeroForceViz,
        ),
        With<LocalCraftBody>,
    >,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    if sim.simulation.vessel_kind() != VesselKind::Ship {
        return;
    }
    let Ok((
        position,
        rotation,
        lin_vel,
        ang_vel,
        ship_aero,
        mut cf,
        mut ct,
        mut readout,
        mut viz,
    )) = craft.get_mut(bubble.craft_entity)
    else {
        return;
    };

    let body = &sim.system.bodies[bubble.body_id];
    let (density, speed_of_sound) = match body.terrestrial_atmosphere.as_ref() {
        Some(atmosphere) => {
            // Avian `Position` is surface-local (anchor-relative); the SLF
            // helper recovers altitude above the reference radius.
            let altitude_m =
                thalos_physics_canonical::surface_local::altitude_asl_m(&bubble.frame, position.0);
            let sample = atmosphere.sample_at_altitude_m(
                altitude_m,
                body.surface_pressure_pa(),
                body.surface_gravity_m_s2(),
            );
            (sample.density_kg_m3, sample.speed_of_sound_m_s)
        }
        None => (0.0, 0.0),
    };

    // Air-relative velocity / rate in the body frame. The bubble integrates in
    // the surface-local (co-rotating) frame, so `LinearVelocity` is already
    // surface-relative and the co-rotating airmass is static (wind = 0).
    let rot = rotation.0;
    let vel_body = rot.inverse() * lin_vel.0;
    let omega_body = rot.inverse() * ang_vel.0;
    // Control-surface deflections come from the fly-by-wire bus, not the raw
    // stick: the same allocated command the reaction wheels execute, so the
    // two effectors pull together instead of fighting (the old direct-stick
    // path was half of the SAS jitter). See `crate::control_bus`.
    let controls = realized.aero;

    // A pathological physics step (e.g. a multi-second gap behind the loading
    // screen) must not integrate aero: even a sane force over a huge dt is a huge
    // impulse. Skip and zero — the next normal step resumes flight.
    if phys_time.delta_secs_f64() > 0.25 {
        cf.0 = DVec3::ZERO;
        ct.0 = DVec3::ZERO;
        return;
    }

    // For a winged craft, take the moment coefficients from the live-tunable
    // resource (feel is iterated over BRP); bluff bodies keep their own config.
    // The same resolve feeds the control allocator's authority estimate
    // (`control_bus`), so the split matches what we actually fly here.
    let config = resolved_aero_config(ship_aero.config, &tuning);
    let out = evaluate_aero(vel_body, omega_body, density, &config, controls);

    // Parked / slow taxi: below the airspeed floor the AoA is degenerate (the
    // velocity is mostly suspension settle, not flow), so a grounded craft gets
    // no aero at all — the gear owns it outright. Above the floor the full
    // model stays live on the ground: that's where elevator authority for
    // rotation, roll damping during the takeoff run, and weathervane stability
    // come from, and at ground-roll dynamic pressures the moments are far too
    // small to fight the gear. (The old blanket weight-on-wheels torque
    // zeroing existed to protect against the previous over-damped moment
    // coefficients, which were strong enough at taxi speed to tip the craft.)
    let mut force_body = out.force;
    let mut torque_body = out.torque;
    if weight_on_wheels.grounded && vel_body.length() < GROUND_AERO_AIRSPEED_FLOOR_M_S {
        force_body = DVec3::ZERO;
        torque_body = DVec3::ZERO;
    }

    // Inertia-relative safety clamp. A real craft pulls only a few g and a few
    // rad/s²; bounding the aero force/torque to the craft's own mass/MOI makes a
    // numerical blow-up impossible (no dt/q/spawn transient can impart more than
    // this), while leaving normal flight (cruise ≈ 2 g, ≈ 0.1 rad/s²) untouched.
    let params = sim.simulation.ship_params();
    let moi = params.moment_of_inertia;
    let min_moi = moi.x.min(moi.y).min(moi.z).max(1.0);
    let mass = sim.simulation.ship_mass_kg().max(1.0);
    force_body = clamp_len(force_body, mass * MAX_LIN_ACCEL_M_S2);
    torque_body = clamp_len(torque_body, min_moi * MAX_ANG_ACCEL_RAD_S2);

    cf.0 = rot * force_body;
    ct.0 = rot * torque_body;

    let speed = vel_body.length();
    let alpha = (-vel_body.z).atan2(vel_body.y);
    *readout = AeroReadout {
        airspeed_ms: speed,
        dynamic_pressure_pa: 0.5 * density * speed * speed,
        mach: if speed_of_sound > 0.0 { speed / speed_of_sound } else { 0.0 },
        density_kgm3: density,
        force_n: force_body.length(),
        alpha_deg: alpha.to_degrees(),
    };
    viz.force_body = force_body.as_vec3();
    viz.vel_body = vel_body.as_vec3();
}

/// Clamp a vector's magnitude to `max` (no-op below it).
fn clamp_len(v: DVec3, max: f64) -> DVec3 {
    let len = v.length();
    if len > max { v * (max / len) } else { v }
}

/// **F3 force overlay, drawn at the rendered ship pose.** Aero is computed in the
/// body-centered bubble frame (~planet-radius from the floating origin), so it's
/// mapped onto the rendered [`PlayerShip`] (same rigid body, shared body frame)
/// via the body rotation. No-op while the [`AeroGizmos`] group is disabled.
fn draw_aero_debug(
    store: Res<GizmoConfigStore>,
    mut gizmos: Gizmos<AeroGizmos>,
    render_q: Query<&GlobalTransform, With<PlayerShip>>,
    viz_q: Query<&AeroForceViz>,
) {
    if !store.config::<AeroGizmos>().0.enabled {
        return;
    }
    let (Ok(render_gt), Ok(viz)) = (render_q.single(), viz_q.single()) else {
        return;
    };
    let rot = render_gt.rotation();
    let origin = render_gt.translation();
    const SCALE: f32 = 2.0e-4;

    let force = rot * viz.force_body * SCALE;
    if force.length_squared() > 1.0 {
        gizmos.arrow(origin, origin + force, Color::srgb(1.0, 0.85, 0.2));
    }
    if viz.vel_body.length() >= 1.0 {
        let wind = rot * viz.vel_body.normalize() * 14.0;
        gizmos.arrow(origin, origin - wind, Color::srgb(0.6, 0.6, 0.6));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Meridian's main-wing panels (one mirrored pair) as
    /// `build_ship_aero_config` sees them — span / chords from
    /// `ships/meridian.ron`. Tail panels are omitted: they're either vertical
    /// (skipped) or small enough not to move the reference numbers.
    fn meridian_panels() -> Vec<WingAeroPanel> {
        let span = 15.0;
        let (root, tip) = (5.2, 1.5);
        let area = span * (root + tip) * 0.5;
        let lambda: f64 = tip / root;
        let mac = (2.0 / 3.0) * root * (1.0 + lambda + lambda * lambda) / (1.0 + lambda);
        [1.5708_f64, -1.5708]
            .into_iter()
            .map(|angle| WingAeroPanel {
                center_body_m: DVec3::ZERO,
                fore_dir: DVec3::Y,
                thick_dir: DVec3::Z,
                span_dir: DVec3::new(angle.signum(), 0.0, 0.0),
                area_m2: area,
                chord_m: mac,
                span_m: span,
                station: 0.44,
                angle,
                role: WingRole::Lift,
            })
            .collect()
    }

    /// Approach-regime flight condition and Meridian-class inertia. The roll
    /// inertia is dominated by wing fuel + engines far outboard (~1.3e6 kg·m²,
    /// airliner-class); ρ/V are a Thalos sea-level approach.
    const APPROACH_SPEED_M_S: f64 = 80.0;
    const APPROACH_DENSITY: f64 = 1.2;
    const ROLL_INERTIA_KG_M2: f64 = 1.3e6;

    /// Handling-feel bands: these encode "an airliner should feel heavy and
    /// stable" as numbers, so a coefficient retune can't silently bring back
    /// the old arcade feel (instant rate onset) or a wallowing one.
    #[test]
    fn meridian_rolls_like_an_airliner() {
        let panels = meridian_panels();
        let cfg = build_ship_aero_config(&panels, 8.0, 0.5);

        // Full wingspan reference → a realistic aspect ratio (A220-class ≈ 9),
        // not the ~2 the per-panel span used to give (4× the induced drag).
        assert!(
            (7.0..12.0).contains(&cfg.aspect_ratio),
            "aspect ratio {} outside the transport band",
            cfg.aspect_ratio
        );

        // Steady full-stick roll rate: control moment balanced against roll
        // damping. Transports physically manage ~25–45°/s at full throw.
        let q = 0.5 * APPROACH_DENSITY * APPROACH_SPEED_M_S * APPROACH_SPEED_M_S;
        let control_nm = cfg.roll_control * q * cfg.reference_area_m2 * cfg.reference_span_m;
        let damp_nm_per_rad_s = cfg.roll_damp
            * APPROACH_DENSITY
            * APPROACH_SPEED_M_S
            * cfg.reference_area_m2
            * cfg.reference_span_m
            * cfg.reference_span_m;
        let p_max_deg_s = (control_nm / damp_nm_per_rad_s).to_degrees();
        assert!(
            (20.0..50.0).contains(&p_max_deg_s),
            "full-stick steady roll rate {p_max_deg_s:.1}°/s outside the airliner band"
        );

        // Roll-rate onset time constant τ = I / damping: the felt rotational
        // inertia. Sub-~0.5 s reads as an arcade snap on a 35 m airframe;
        // beyond ~2.5 s it wallows.
        let tau_s = ROLL_INERTIA_KG_M2 / damp_nm_per_rad_s;
        assert!(
            (0.5..2.5).contains(&tau_s),
            "roll onset τ {tau_s:.2}s outside the airliner band"
        );
    }

    #[test]
    fn meridian_pitch_can_rotate_but_trims_gently() {
        let panels = meridian_panels();
        let cfg = build_ship_aero_config(&panels, 8.0, 0.5);

        // Full elevator must out-muscle static stability up to a rotation /
        // flare attitude well past the approach AoA…
        let full_stick_trim_deg = (cfg.pitch_control / cfg.pitch_stability).to_degrees();
        assert!(
            full_stick_trim_deg > 12.0,
            "full-stick trim AoA {full_stick_trim_deg:.1}° too weak to rotate or flare"
        );
        // …while the hands-off trim point sits at a small positive cruise AoA.
        let trim_deg = (cfg.cm0 / cfg.pitch_stability).to_degrees();
        assert!(
            (0.5..4.0).contains(&trim_deg),
            "hands-off trim AoA {trim_deg:.1}° not a level-cruise attitude"
        );
    }
}
