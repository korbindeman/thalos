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
use thalos_shipyard::{ControlSurfaceRole, WingAeroPanel, WingRole};

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

// --- Compressibility (Korn equation) ------------------------------------------
// The drag-divergence Mach is *derived from the authored wing geometry*, not
// tuned per craft: `M_dd = κ/cosΛ − (t/c)/cos²Λ − CL/(10·cos³Λ)`. Sweep and a
// thin airfoil buy transonic margin, exactly the trade a player makes in the
// shipyard. The cruise-CL term uses the camber CL0 (a level-flight proxy).
/// Korn airfoil technology factor (0.87 = conventional, 0.95 = supercritical).
const KORN_AIRFOIL_FACTOR: f64 = 0.87;
/// Sanity band for the derived divergence Mach: even a fat straight wing
/// keeps a wall somewhere past M 0.5, and nothing subsonic-authored gets to
/// push the wall past M 0.95.
const MACH_DD_MIN: f64 = 0.5;
const MACH_DD_MAX: f64 = 0.95;

// --- High-lift devices / spoilers ----------------------------------------------
// Flap force increments are derived from the authored `Flap` windows (plain-
// flap theory): ΔCL = CLα·τ(c_f)·η·δ_max·(S_flapped/S_ref) with τ the
// chord-fraction effectiveness and η a viscous knock-down; ΔCD is Roskam's
// plain-flap form `1.7·c_f^1.38·(S_f/S)·sin²δ`. Spoiler drag is a deflected-
// plate term on the panel area; its lift dump scales with the spanned strip.
/// Viscous flap-effectiveness knock-down at large deflections.
const FLAP_VISCOUS_ETA: f64 = 0.6;
/// Roskam plain-flap profile-drag constant.
const FLAP_DRAG_FACTOR: f64 = 1.7;
/// Deflected-plate drag constant for spoiler panels.
const SPOILER_DRAG_FACTOR: f64 = 1.6;

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

// Bluff body (rocket/capsule): no lift, no control, but weathervane-stable.
const BLUFF_STABILITY: f64 = 0.5;
const BLUFF_DAMP: f64 = 0.5;

/// Flap load relief: above this dynamic pressure the effective flap
/// deployment fades as `q_relief/q`, which caps the flap force increment at
/// its value at the relief point (ΔCL·q·S stops growing). Real transports
/// auto-relieve flap loads instead of ripping the tracks off; for gameplay it
/// means slamming landing flaps at cruise speed produces a gentle balloon,
/// not a 10 g pull-up, with no placard-speed micromanagement. 10 kPa ≈ a
/// 128 m/s sea-level approach — well above any sane flap speed.
const FLAP_LOAD_RELIEF_Q_PA: f64 = 10_000.0;

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
///
/// Stability / damping are whole-body terms, so the tuning values *replace*
/// them; the control coefficients are **derived per surface** from the
/// authored geometry (see [`build_ship_aero_config`]), so the tuning values
/// only *scale* them — a live feel-tweak can't erase the difference between
/// a big and a small aileron.
pub(crate) fn resolved_aero_config(base: AeroConfig, tuning: &AeroTuning) -> AeroConfig {
    let mut config = base;
    if config.lift_slope > 0.0 {
        config.pitch_stability = tuning.pitch_stability;
        config.yaw_stability = tuning.yaw_stability;
        config.pitch_damp = tuning.pitch_damp;
        config.roll_damp = tuning.roll_damp;
        config.yaw_damp = tuning.yaw_damp;
        config.pitch_control *= tuning.pitch_control_scale;
        config.roll_control *= tuning.roll_control_scale;
        config.yaw_control *= tuning.yaw_control_scale;
    }
    config
}

/// Handling coefficients for **winged** aircraft, applied over the per-craft
/// config's moment terms each frame. Reflect-registered (for a future in-game
/// debug UI); to change the feel (control authority, damping, static stability)
/// edit the defaults and rebuild. Stability/damping defaults are the
/// transport-derivative constants above; the control *scales* default to 1 (the
/// per-surface derived authority is flown as-is).
#[derive(Resource, Reflect, Clone, Copy)]
#[reflect(Resource)]
pub struct AeroTuning {
    pub pitch_stability: f64,
    pub yaw_stability: f64,
    pub pitch_damp: f64,
    pub roll_damp: f64,
    pub yaw_damp: f64,
    /// Multipliers on the per-surface-derived control coefficients.
    pub pitch_control_scale: f64,
    pub roll_control_scale: f64,
    pub yaw_control_scale: f64,
}

impl Default for AeroTuning {
    fn default() -> Self {
        Self {
            pitch_stability: WING_PITCH_STABILITY,
            yaw_stability: WING_YAW_STABILITY,
            pitch_damp: WING_PITCH_DAMP,
            roll_damp: WING_ROLL_DAMP,
            yaw_damp: WING_YAW_DAMP,
            pitch_control_scale: 1.0,
            roll_control_scale: 1.0,
            yaw_control_scale: 1.0,
        }
    }
}

/// Flight readout for the HUD.
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
/// lift + **per-surface-derived control authority** (see
/// [`derive_control_coefficients`] — `com_body_m` is the CoM the moment arms
/// are measured about). Wingless (no panels): a bluff-body drag config sized
/// from the frontal area, no lift/control but weathervane-stable.
pub fn build_ship_aero_config(
    panels: &[WingAeroPanel],
    frontal_area_m2: f64,
    drag_cd: f64,
    com_body_m: DVec3,
) -> AeroConfig {
    let mut total_area = 0.0;
    let mut mac_acc = 0.0;
    let mut max_span = 0.0_f64;
    let mut sweep_acc = 0.0;
    let mut thickness_acc = 0.0;
    let mut lifting_panels = Vec::new();
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
        sweep_acc += panel.sweep_rad * panel.area_m2;
        thickness_acc += panel.thickness * panel.area_m2;
        lifting_panels.push(panel);
    }
    // Panels are single half-wings (mirrored pairs are separate entities), so
    // the aerodynamic reference span — the roll/yaw moment arm and the
    // aspect-ratio basis — is the full tip-to-tip wingspan, two panels.
    let full_span = 2.0 * max_span;

    if total_area > 0.0 {
        // Korn equation on the area-weighted sweep / thickness: where the
        // transonic wall stands for *this* planform. (Authored sweep is the
        // leading edge's; close enough to quarter-chord at these tapers.)
        let cos_sweep = (sweep_acc / total_area).cos().max(0.5);
        let mach_dd = (KORN_AIRFOIL_FACTOR / cos_sweep
            - (thickness_acc / total_area) / (cos_sweep * cos_sweep)
            - MAIN_WING_CL0 / (10.0 * cos_sweep.powi(3)))
        .clamp(MACH_DD_MIN, MACH_DD_MAX);

        // Flap / spoiler force increments from the authored windows.
        let mut flap_dcl = 0.0;
        let mut flap_dcd = 0.0;
        let mut spoiler_dcl = 0.0;
        let mut spoiler_dcd = 0.0;
        for panel in &lifting_panels {
            for w in &panel.surfaces {
                let spanned_frac = w.spanned_area_m2 / total_area;
                match w.role {
                    ControlSurfaceRole::Flap => {
                        flap_dcl += LIFT_SLOPE_PER_RAD
                            * flap_chord_effectiveness(w.chord_fraction)
                            * FLAP_VISCOUS_ETA
                            * w.max_deflection_rad
                            * spanned_frac;
                        flap_dcd += FLAP_DRAG_FACTOR
                            * w.chord_fraction.powf(1.38)
                            * spanned_frac
                            * w.max_deflection_rad.sin().powi(2);
                    }
                    ControlSurfaceRole::Spoiler => {
                        spoiler_dcd += SPOILER_DRAG_FACTOR * (w.area_m2 / total_area)
                            * w.max_deflection_rad.sin();
                        spoiler_dcl -= spanned_frac * w.max_deflection_rad.sin();
                    }
                    _ => {}
                }
            }
        }

        let mean_chord = mac_acc / total_area;
        let reference_span = full_span.max(1.0);
        let (pitch_control, roll_control, yaw_control) = derive_control_coefficients(
            panels,
            com_body_m,
            total_area,
            mean_chord,
            reference_span,
        );

        AeroConfig {
            reference_area_m2: total_area,
            reference_chord_m: mean_chord,
            reference_span_m: reference_span,
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
            pitch_control,
            roll_control,
            yaw_control,
            mach_drag_divergence: mach_dd,
            flap_dcl,
            flap_dcd,
            spoiler_dcl,
            spoiler_dcd,
        }
    } else {
        // Bluff body: a body-length proxy from the frontal area as the moment
        // arm. `mach_drag_divergence: 0` — a capsule's blunt-body Cd already
        // stands in for its transonic behaviour; the wall is a wing concern.
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
            ..Default::default()
        }
    }
}

/// Plain-flap chord-fraction effectiveness `τ(c_f) = 1 − (θ − sin θ)/π` with
/// `θ = acos(2·c_f − 1)` — thin-airfoil theory's lift effectiveness of a
/// trailing-edge surface occupying chord fraction `c_f`.
fn flap_chord_effectiveness(chord_fraction: f64) -> f64 {
    let theta = (2.0 * chord_fraction.clamp(0.0, 1.0) - 1.0).acos();
    1.0 - (theta - theta.sin()) / std::f64::consts::PI
}

/// Derive the per-axis control coefficients from the authored
/// aileron / elevator / rudder windows — **per-surface control authority**.
///
/// For each window, the deflection lift at full throw is the same plain-flap
/// term the flaps use (`CLα · τ(c_f) · η · δ_max` over the spanned strip), and
/// its moment about the CoM comes from the window's real geometry: the force
/// acts along the panel's lift normal (`thick_dir`) at the window centroid
/// `r`, so the torque about a body axis `â` is `(r × n̂)·â` per unit force.
/// Summing `|arm| · ΔCL_strip · S_strip` over the windows of each role and
/// non-dimensionalising by `S_ref · L_ref` yields exactly the coefficient the
/// evaluator's control term (`coeff · q̄ · S · L · input`) expects.
///
/// This is what makes shipyard surface *sizing and placement* show up in
/// handling: a bigger or further-outboard aileron rolls harder, a longer tail
/// arm pitches harder, and a craft authored with no rudder genuinely has no
/// yaw authority. Each role feeds only its own axis (an elevator's left/right
/// halves cancel in roll anyway; a rudder's roll cross-coupling is real but
/// deliberately dropped — the whole-body model keeps control moments
/// axis-diagonal so the fly-by-wire allocation stays unconditionally stable,
/// the same "explicit, not emergent" reasoning as the restoring/damping
/// terms). Vertical fins are included here even though they carry no lifting
/// area — that's where the rudder lives.
///
/// Verified against the Meridian: the derived coefficients land within ~10%
/// of the previously hand-tuned transport constants (pitch 0.48 vs 0.5, yaw
/// 0.032 vs 0.04, roll 0.037 vs the deliberately-hot 0.06), so the feel stays
/// in the airliner band while becoming a real consequence of the authored
/// geometry (pinned by the `meridian_*` tests below).
fn derive_control_coefficients(
    panels: &[WingAeroPanel],
    com_body_m: DVec3,
    reference_area_m2: f64,
    reference_chord_m: f64,
    reference_span_m: f64,
) -> (f64, f64, f64) {
    let mut pitch_qs = 0.0; // Σ torque per unit q̄, N·m / Pa
    let mut roll_qs = 0.0;
    let mut yaw_qs = 0.0;
    for panel in panels {
        for w in &panel.surfaces {
            let (axis, acc) = match w.role {
                ControlSurfaceRole::Elevator => (DVec3::X, &mut pitch_qs),
                ControlSurfaceRole::Aileron => (DVec3::Y, &mut roll_qs),
                ControlSurfaceRole::Rudder => (DVec3::Z, &mut yaw_qs),
                // Flaps / spoilers are craft configuration, not attitude
                // effectors — their force model lives in flap_dcl/spoiler_dcd.
                ControlSurfaceRole::Flap | ControlSurfaceRole::Spoiler => continue,
            };
            let strip_dcl = LIFT_SLOPE_PER_RAD
                * flap_chord_effectiveness(w.chord_fraction)
                * FLAP_VISCOUS_ETA
                * w.max_deflection_rad;
            let r = w.centroid_body_m - com_body_m;
            let arm_m = r.cross(panel.thick_dir).dot(axis).abs();
            *acc += strip_dcl * w.spanned_area_m2 * arm_m;
        }
    }
    (
        pitch_qs / (reference_area_m2 * reference_chord_m),
        roll_qs / (reference_area_m2 * reference_span_m),
        yaw_qs / (reference_area_m2 * reference_span_m),
    )
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
    flight_config: Res<crate::flight_config::FlightConfig>,
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
    // path was half of the SAS jitter). See `crate::control_bus`. Flap /
    // spoiler deployment is craft configuration, overlaid from the flight-
    // config lever state (actual actuator positions, so a moving flap's
    // forces build smoothly).
    let mut controls = realized.aero;
    controls.flap = flight_config.flap_fraction;
    controls.spoiler = flight_config.spoiler_fraction;
    // Flap load relief (see `FLAP_LOAD_RELIEF_Q_PA`).
    let q_now = 0.5 * density * vel_body.length_squared();
    if q_now > FLAP_LOAD_RELIEF_Q_PA {
        controls.flap *= FLAP_LOAD_RELIEF_Q_PA / q_now;
    }

    // A pathological physics step (e.g. a multi-second gap behind the loading
    // screen) must not integrate aero: even a sane force over a huge dt is a huge
    // impulse. Skip and zero — the next normal step resumes flight.
    if phys_time.delta_secs_f64() > 0.25 {
        cf.0 = DVec3::ZERO;
        ct.0 = DVec3::ZERO;
        return;
    }

    // For a winged craft, take the moment coefficients from the tuning
    // resource (feel is iterated by editing its defaults); bluff bodies keep
    // their own config.
    // The same resolve feeds the control allocator's authority estimate
    // (`control_bus`), so the split matches what we actually fly here.
    let config = resolved_aero_config(ship_aero.config, &tuning);
    let out = evaluate_aero(vel_body, omega_body, density, speed_of_sound, &config, controls);

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
    /// `build_ship_aero_config` sees them — span / chords / sweep / control
    /// surfaces from `ships/meridian.ron`. Tail panels are omitted: they're
    /// either vertical (skipped) or small enough not to move the reference
    /// numbers.
    /// Fuselage skin radius the panels root against (window centroids sit at
    /// `radius + mid·span` outboard) and tail moment arms aft of the CoM —
    /// the geometry the per-surface authority derivation reads.
    const FUSELAGE_RADIUS_M: f64 = 1.65;
    const TAILPLANE_ARM_M: f64 = 15.4;
    const FIN_ARM_M: f64 = 14.7;

    /// One half-wing panel with its trailing-edge windows, the way
    /// `wing_aero_panels` emits it: strip areas from the mid-window chord,
    /// window centroids in the ship body frame (CoM at the origin here).
    /// `out_dir` is the spanwise direction (±X for wings/tailplanes, +Z for
    /// the fin), `thick_dir` the lift normal, `arm_aft_m` how far the panel
    /// root sits behind the CoM.
    #[allow(clippy::too_many_arguments)]
    fn panel(
        span: f64,
        root: f64,
        tip: f64,
        sweep: f64,
        thickness: f64,
        angle: f64,
        role: WingRole,
        out_dir: DVec3,
        thick_dir: DVec3,
        arm_aft_m: f64,
        windows: &[(ControlSurfaceRole, f64, f64, f64, f64)],
    ) -> WingAeroPanel {
        let area = span * (root + tip) * 0.5;
        let lambda: f64 = tip / root;
        let mac = (2.0 / 3.0) * root * (1.0 + lambda + lambda * lambda) / (1.0 + lambda);
        let root_pos = DVec3::new(0.0, -arm_aft_m, 0.0) + out_dir * FUSELAGE_RADIUS_M;
        let surfaces = windows
            .iter()
            .map(|&(role, s0, s1, cf, dmax)| {
                let mid = 0.5 * (s0 + s1);
                let chord_mid = root + (tip - root) * mid;
                let spanned = span * (s1 - s0) * chord_mid;
                thalos_shipyard::AeroSurfaceWindow {
                    role,
                    spanned_area_m2: spanned,
                    area_m2: spanned * cf,
                    chord_fraction: cf,
                    max_deflection_rad: dmax,
                    centroid_body_m: root_pos + out_dir * (mid * span),
                }
            })
            .collect();
        WingAeroPanel {
            center_body_m: root_pos + out_dir * (0.4 * span),
            fore_dir: DVec3::Y,
            thick_dir,
            span_dir: out_dir,
            area_m2: area,
            chord_m: mac,
            span_m: span,
            sweep_rad: sweep,
            thickness,
            station: 0.5,
            angle,
            role,
            surfaces,
        }
    }

    /// The Meridian's aero panels — main wings, tailplanes, and the vertical
    /// fin, with the authored control-surface windows at their real
    /// body-frame positions (geometry from `ships/meridian.ron`; CoM at the
    /// origin). The per-surface authority derivation needs the empennage:
    /// elevators and rudder live there.
    fn meridian_panels() -> Vec<WingAeroPanel> {
        let wing_windows = [
            (ControlSurfaceRole::Flap, 0.08, 0.58, 0.30, 0.61),
            (ControlSurfaceRole::Spoiler, 0.60, 0.72, 0.30, 1.05),
            (ControlSurfaceRole::Aileron, 0.74, 0.97, 0.25, 0.35),
        ];
        let elevator = [(ControlSurfaceRole::Elevator, 0.05, 0.95, 0.32, 0.4)];
        let rudder = [(ControlSurfaceRole::Rudder, 0.05, 0.95, 0.32, 0.4)];
        vec![
            // Main wings (slightly ahead of the CoM; the lever that matters
            // for roll is spanwise, so the fore/aft arm is irrelevant here).
            panel(15.0, 5.2, 1.5, 0.52, 0.11, 1.5708, WingRole::Lift, DVec3::X, DVec3::Z, 1.0, &wing_windows),
            panel(15.0, 5.2, 1.5, 0.52, 0.11, -1.5708, WingRole::Lift, -DVec3::X, DVec3::Z, 1.0, &wing_windows),
            // Tailplanes — the pitch lever.
            panel(4.6, 2.6, 1.1, 0.55, 0.10, 1.5708, WingRole::Stabilizer, DVec3::X, DVec3::Z, TAILPLANE_ARM_M, &elevator),
            panel(4.6, 2.6, 1.1, 0.55, 0.10, -1.5708, WingRole::Stabilizer, -DVec3::X, DVec3::Z, TAILPLANE_ARM_M, &elevator),
            // Vertical fin — the yaw lever. Skipped by the lifting-area
            // aggregation (vertical), but its rudder still derives authority.
            panel(4.4, 3.4, 1.4, 0.62, 0.10, 0.0, WingRole::Stabilizer, DVec3::Z, DVec3::X, FIN_ARM_M, &rudder),
        ]
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
        let cfg = build_ship_aero_config(&panels, 8.0, 0.5, DVec3::ZERO);

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

    /// The transonic wall, end to end: the derived drag-divergence Mach sits
    /// in the early-jet band, and at M 0.9 the total drag exceeds the
    /// density-lapsed thrust of four Vega turbojets at *every* altitude — the
    /// Meridian physically cannot sustain transonic flight, in level cruise
    /// or a shallow climb-and-dash.
    #[test]
    fn meridian_cannot_sustain_transonic_flight() {
        let panels = meridian_panels();
        let cfg = build_ship_aero_config(&panels, 8.0, 0.5, DVec3::ZERO);

        // Korn on the authored sweep/thickness: a 30°-swept, 11%-thick early
        // jet wing diverges around M 0.8 (the 707 / Comet band).
        assert!(
            (0.76..0.88).contains(&cfg.mach_drag_divergence),
            "M_dd {} outside the early-jet band",
            cfg.mach_drag_divergence
        );

        // Thalos-like atmosphere: ρ₀ = 1.225, H = 9.1 km, a ≈ 337 m/s.
        let rho0 = 1.225;
        let scale_height_m = 9100.0;
        let speed_of_sound = 337.0;
        let weight_n = 37_000.0 * 9.06;
        let rated_thrust_n = 4.0 * 50_000.0;
        let mach = 0.9;
        for altitude_m in [0.0_f64, 3000.0, 6000.0, 9000.0, 12000.0] {
            let rho = rho0 * (-altitude_m / scale_height_m).exp();
            let thrust = rated_thrust_n * crate::fuel::air_breathing_thrust_factor(rho);
            let v = mach * speed_of_sound;
            let q = 0.5 * rho * v * v;
            // Level flight: CL carries the weight; CD = parasitic + induced +
            // wave (the same terms `evaluate_aero` applies).
            let cl = weight_n / (q * cfg.reference_area_m2);
            let mach_crit = cfg.mach_drag_divergence - 0.108;
            let cd = cfg.cd0
                + cl * cl / (std::f64::consts::PI * 0.8 * cfg.aspect_ratio)
                + 20.0 * (mach - mach_crit).clamp(0.0, 0.5).powi(4);
            let drag = cd * q * cfg.reference_area_m2;
            assert!(
                drag > thrust,
                "at {altitude_m} m: M 0.9 drag {:.0} kN must exceed thrust {:.0} kN",
                drag / 1000.0,
                thrust / 1000.0
            );
        }
    }

    /// Per-surface control authority: the coefficients must derive from the
    /// authored aileron / elevator / rudder windows and their real moment
    /// arms — and land in the transport band the old hand-tuned constants
    /// encoded (pitch ≈ 0.5, roll ≈ 0.04, yaw ≈ 0.04), so the feel doesn't
    /// regress while becoming a consequence of the geometry.
    #[test]
    fn meridian_control_authority_derives_from_surfaces() {
        let panels = meridian_panels();
        let cfg = build_ship_aero_config(&panels, 8.0, 0.5, DVec3::ZERO);

        assert!(
            (0.35..0.65).contains(&cfg.pitch_control),
            "derived pitch authority {} outside the transport band",
            cfg.pitch_control
        );
        assert!(
            (0.025..0.06).contains(&cfg.roll_control),
            "derived roll authority {} outside the transport band",
            cfg.roll_control
        );
        assert!(
            (0.02..0.05).contains(&cfg.yaw_control),
            "derived yaw authority {} outside the transport band",
            cfg.yaw_control
        );

        // Sizing must matter: stretching the aileron window inboard
        // (0.74–0.97 → 0.54–0.97, more strip area on a still-long arm) should
        // buy substantially more roll authority.
        let wing_windows = [
            (ControlSurfaceRole::Flap, 0.08, 0.50, 0.30, 0.61),
            (ControlSurfaceRole::Aileron, 0.54, 0.97, 0.25, 0.35),
        ];
        let big = vec![
            panel(15.0, 5.2, 1.5, 0.52, 0.11, 1.5708, WingRole::Lift, DVec3::X, DVec3::Z, 1.0, &wing_windows),
            panel(15.0, 5.2, 1.5, 0.52, 0.11, -1.5708, WingRole::Lift, -DVec3::X, DVec3::Z, 1.0, &wing_windows),
        ];
        let big_cfg = build_ship_aero_config(&big, 8.0, 0.5, DVec3::ZERO);
        assert!(
            big_cfg.roll_control > 1.5 * cfg.roll_control,
            "bigger ailerons must roll harder ({} vs {})",
            big_cfg.roll_control,
            cfg.roll_control
        );

        // A craft authored without a rudder genuinely has no yaw authority.
        let no_fin: Vec<_> = panels
            .iter()
            .filter(|p| {
                !p.surfaces
                    .iter()
                    .any(|w| matches!(w.role, ControlSurfaceRole::Rudder))
            })
            .cloned()
            .collect();
        let no_fin_cfg = build_ship_aero_config(&no_fin, 8.0, 0.5, DVec3::ZERO);
        assert_eq!(no_fin_cfg.yaw_control, 0.0, "no rudder → no yaw authority");
    }

    /// Landing flaps must buy a real approach-speed reduction (≥ 10% off the
    /// stall speed) and a real drag increment, derived purely from the
    /// authored flap windows.
    #[test]
    fn meridian_flaps_buy_a_slow_approach() {
        let panels = meridian_panels();
        let cfg = build_ship_aero_config(&panels, 8.0, 0.5, DVec3::ZERO);

        assert!(
            (0.4..0.9).contains(&cfg.flap_dcl),
            "flap ΔCL {} outside the landing-flap band",
            cfg.flap_dcl
        );
        assert!(
            (0.03..0.09).contains(&cfg.flap_dcd),
            "flap ΔCD {} outside the landing-flap band",
            cfg.flap_dcd
        );

        // Stall speed ∝ 1/√CL_max: the flap camber raises the stall ceiling.
        let cl_max_clean = cfg.cl0 + cfg.lift_slope * cfg.stall_alpha;
        let cl_max_landing = cl_max_clean + cfg.flap_dcl;
        let vs_ratio = (cl_max_clean / cl_max_landing).sqrt();
        assert!(
            vs_ratio < 0.9,
            "landing flaps should cut the stall speed by ≥10% (ratio {vs_ratio:.3})"
        );

        // Spoilers: a usable speedbrake (≈ doubles parasitic drag) that also
        // dumps lift.
        assert!(
            cfg.spoiler_dcd > 0.5 * cfg.cd0,
            "spoiler ΔCD {} too weak to brake with",
            cfg.spoiler_dcd
        );
        assert!(cfg.spoiler_dcl < 0.0, "spoilers must dump lift");
    }

    #[test]
    fn meridian_pitch_can_rotate_but_trims_gently() {
        let panels = meridian_panels();
        let cfg = build_ship_aero_config(&panels, 8.0, 0.5, DVec3::ZERO);

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
