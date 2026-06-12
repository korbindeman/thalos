//! Simple atmospheric aerodynamics: a whole-body lite-flight model.
//!
//! Pure Rust (glam f64), no Bevy. Given a craft's air-relative velocity and
//! angular rate in its body frame, an [`AeroConfig`], and air density,
//! [`evaluate_aero`] returns the net aerodynamic force and the torque about the
//! centre of mass, both in the body frame. The Bevy bubble (`thalos_game`) reads
//! the craft's Avian state, calls this, and writes the result as a
//! `ConstantForce`/`ConstantTorque` — Thalos still owns mass, inertia, gravity.
//!
//! Frame: Thalos body axes — X = right, Y = nose, Z = dorsal (up).
//!
//! **Design: stability is explicit, not emergent.** Forces are whole-body lift +
//! drag from the angle of attack / sideslip. Moments are three guaranteed-stable
//! terms: a *restoring* moment that turns the nose toward the relative wind
//! (weathervane static stability), a *damping* moment that always opposes the
//! angular rate, and a *control* moment from pilot input. This is deliberately
//! simpler — and far more robust — than a per-surface strip sum: the latter is
//! physically elegant but its rotation coupling pumps energy under the bubble's
//! explicit, per-frame-constant force integration (a spinning craft autorotates
//! to absurd rates). The restoring + damping form cannot add energy, so a craft
//! settles to trim instead of diverging. Coefficients are non-dimensional and
//! tuned in-game; a wingless craft (rocket/capsule) sets `lift_slope = 0` and
//! still weathervanes nose-into-wind from the restoring term.

use glam::DVec3;
use std::f64::consts::PI;

/// Span efficiency (Oswald) factor for the induced-drag term.
const OSWALD_E: f64 = 0.8;
/// Minimum airspeed (m/s) below which all aero is skipped — the angle of attack
/// is degenerate and dynamic pressure negligible.
const MIN_SPEED_M_S: f64 = 1.0;
/// Safety ceilings: even with stable coefficients, clamp the output so a bad
/// authored config or a transient can never inject a non-physical impulse.
const MAX_FORCE_N: f64 = 5.0e7;
const MAX_TORQUE_NM: f64 = 5.0e8;
/// Wave drag rises as `WAVE_DRAG_SCALE · (M − M_crit)⁴` past the critical Mach
/// (the canonical transonic drag-rise shape: +20 counts at drag divergence,
/// then a steep wall). The Mach excess is capped so a hypersonic entry doesn't
/// produce an absurd coefficient — past the cap the craft is simply "very
/// draggy" (ΔCD = 1.25), and the force ceiling still bounds the output.
const WAVE_DRAG_SCALE: f64 = 20.0;
const WAVE_DRAG_MACH_EXCESS_CAP: f64 = 0.5;
/// Drag divergence is conventionally defined ~0.1 above the critical Mach
/// where wave drag first appears (`ΔCD(M_dd) = 20 counts` with the quartic).
const DRAG_DIVERGENCE_ABOVE_CRITICAL: f64 = 0.108;

/// Pilot control inputs, each in [−1, 1].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ControlInputs {
    /// +1 = full nose-up (pull).
    pub pitch: f64,
    /// +1 = full roll right.
    pub roll: f64,
    /// +1 = full nose-right yaw.
    pub yaw: f64,
    /// High-lift (flap) deployment in [0, 1]: the actual actuator position,
    /// not the lever detent. Lift scales linearly with it and flap drag
    /// quadratically, so a half setting is the high-lift/low-drag takeoff
    /// configuration and full is the draggy landing one.
    pub flap: f64,
    /// Spoiler / speedbrake deployment in [0, 1]: adds drag and dumps lift.
    pub spoiler: f64,
}

/// Whole-body aerodynamic configuration for one craft. Forces use the lift/drag
/// coefficients; moments use the stability / damping / control coefficients
/// (all non-dimensional, scaled by dynamic pressure × reference area × arm).
#[derive(Clone, Copy, Debug)]
pub struct AeroConfig {
    /// Reference area (m²): wing planform for aircraft, frontal area for bluff.
    pub reference_area_m2: f64,
    /// Pitch reference length (m): mean aerodynamic chord.
    pub reference_chord_m: f64,
    /// Roll/yaw reference length (m): wingspan (or body length for bluff).
    pub reference_span_m: f64,
    /// Lift-curve slope per radian (0 → no lift; a bluff body / capsule).
    pub lift_slope: f64,
    /// Lift coefficient at zero angle of attack (camber).
    pub cl0: f64,
    /// Pitch moment coefficient at zero angle of attack (trim). A small
    /// positive value trims a statically-stable craft at a small positive AoA
    /// (`α_trim = cm0 / pitch_stability`) so it holds level flight hands-off
    /// instead of needing constant forward stick against the restoring moment.
    pub cm0: f64,
    /// Parasitic (zero-lift) drag coefficient.
    pub cd0: f64,
    /// Stall angle (rad); |CL| is clamped past it.
    pub stall_alpha: f64,
    /// Aspect ratio for induced drag `CDi = CL²/(π·e·AR)`; 0 → none.
    pub aspect_ratio: f64,
    /// Pitch static stability (≥0): restoring `Cm_α` magnitude.
    pub pitch_stability: f64,
    /// Yaw static stability (≥0): restoring `Cn_β` magnitude (weathervane).
    pub yaw_stability: f64,
    /// Pitch-rate damping (≥0).
    pub pitch_damp: f64,
    /// Roll-rate damping (≥0).
    pub roll_damp: f64,
    /// Yaw-rate damping (≥0).
    pub yaw_damp: f64,
    /// Pitch control authority.
    pub pitch_control: f64,
    /// Roll control authority.
    pub roll_control: f64,
    /// Yaw control authority.
    pub yaw_control: f64,
    /// Drag-divergence Mach number (`0` disables compressibility entirely —
    /// the bluff-body default). Above the corresponding critical Mach
    /// (`M_dd − 0.108`) wave drag rises quartically, the transonic wall that
    /// keeps a subsonic airframe subsonic. Derived from the authored wing
    /// sweep/thickness (Korn equation) by the consumer.
    pub mach_drag_divergence: f64,
    /// ΔCL at full flap deployment (high-lift camber increase). Scales
    /// linearly with `ControlInputs::flap`; also raises the stall-clamped
    /// |CL| ceiling, which is what lowers the stall speed.
    pub flap_dcl: f64,
    /// ΔCD at full flap deployment. Scales with `flap²` (drag grows with the
    /// deflection angle squared), so takeoff flap is cheap and landing flap
    /// is draggy.
    pub flap_dcd: f64,
    /// ΔCL at full spoiler deployment (negative: lift dump).
    pub spoiler_dcl: f64,
    /// ΔCD at full spoiler deployment (speedbrake drag).
    pub spoiler_dcd: f64,
}

impl Default for AeroConfig {
    fn default() -> Self {
        Self {
            reference_area_m2: 1.0,
            reference_chord_m: 1.0,
            reference_span_m: 1.0,
            lift_slope: 0.0,
            cl0: 0.0,
            cm0: 0.0,
            cd0: 0.5,
            stall_alpha: 0.26,
            aspect_ratio: 0.0,
            pitch_stability: 0.0,
            yaw_stability: 0.0,
            pitch_damp: 0.0,
            roll_damp: 0.0,
            yaw_damp: 0.0,
            pitch_control: 0.0,
            roll_control: 0.0,
            yaw_control: 0.0,
            mach_drag_divergence: 0.0,
            flap_dcl: 0.0,
            flap_dcd: 0.0,
            spoiler_dcl: 0.0,
            spoiler_dcd: 0.0,
        }
    }
}

/// Net aerodynamic force and torque (the torque is about the CoM), body frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct AeroOutput {
    pub force: DVec3,
    pub torque: DVec3,
}

/// Evaluate the net aerodynamic force + torque in the body frame.
///
/// - `vel_body`: craft CoM velocity relative to the air, body frame (m/s).
/// - `omega_body`: angular velocity, body frame (rad/s).
/// - `density`: air density (kg/m³); ≤ 0 returns zero.
/// - `speed_of_sound`: local speed of sound (m/s); ≤ 0 disables every Mach
///   effect (vacuum, or a caller that doesn't model compressibility).
pub fn evaluate_aero(
    vel_body: DVec3,
    omega_body: DVec3,
    density: f64,
    speed_of_sound: f64,
    cfg: &AeroConfig,
    controls: ControlInputs,
) -> AeroOutput {
    let speed = vel_body.length();
    if density <= 0.0 || speed < MIN_SPEED_M_S {
        return AeroOutput::default();
    }

    // Body axes: X = right, Y = nose (forward), Z = up.
    let vf = vel_body.y; // forward
    let vr = vel_body.x; // right
    let vu = vel_body.z; // up
    // Angle of attack: flow from below the belly (vu < 0) → +α → +lift.
    let alpha = (-vu).atan2(vf);
    // Sideslip: flow from the right (vr > 0) → +β.
    let beta = vr.atan2(vf);

    let q = 0.5 * density * speed * speed;
    let s = cfg.reference_area_m2;
    let c = cfg.reference_chord_m;
    let b = cfg.reference_span_m;

    // --- Forces: lift + drag. ------------------------------------------------
    let flap = controls.flap.clamp(0.0, 1.0);
    let spoiler = controls.spoiler.clamp(0.0, 1.0);
    // Flaps add camber (linear in deployment); spoilers dump lift. Folding the
    // increments into cl0 means the stall clamp below rises with the flaps —
    // the higher CL_max is exactly what lowers the stall speed.
    let cl0 = cfg.cl0 + cfg.flap_dcl * flap + cfg.spoiler_dcl * spoiler;
    let mut cl = cl0 + cfg.lift_slope * alpha;
    let cl_stall = cl0.abs() + cfg.lift_slope * cfg.stall_alpha;
    if cl_stall > 0.0 {
        cl = cl.clamp(-cl_stall, cl_stall);
    }
    let mut cd = cfg.cd0 + cfg.flap_dcd * flap * flap + cfg.spoiler_dcd * spoiler;
    if cfg.aspect_ratio > 0.0 {
        cd += cl * cl / (PI * OSWALD_E * cfg.aspect_ratio);
    }
    // Transonic wave drag: the quartic rise past the critical Mach. This is
    // the wall that keeps a subsonic airframe from casually crossing Mach 1 —
    // near M_dd the drag merely doubles, by M ≈ 1 it is several times CD0.
    if cfg.mach_drag_divergence > 0.0 && speed_of_sound > 0.0 {
        let mach = speed / speed_of_sound;
        let mach_crit = cfg.mach_drag_divergence - DRAG_DIVERGENCE_ABOVE_CRITICAL;
        let excess = (mach - mach_crit).clamp(0.0, WAVE_DRAG_MACH_EXCESS_CAP);
        cd += WAVE_DRAG_SCALE * excess.powi(4);
    }

    let drag_dir = -vel_body / speed;
    // Lift is perpendicular to the flow, toward the dorsal (+Z) side: the part
    // of +Z orthogonal to the velocity. Degenerates to zero in a vertical dive
    // (velocity along ±Z), which is correct — no lift axis is defined there.
    let up = DVec3::Z;
    let lift_dir = (up - drag_dir * up.dot(drag_dir)).normalize_or_zero();
    let force = drag_dir * (cd * q * s) + lift_dir * (cl * q * s);

    // --- Moments: restoring + damping + control. -----------------------------
    // All scaled by q·S·(arm). Damping uses `ρ·speed·S·L²·ω` (= q·S·L·(ωL/V)·2)
    // so it stays finite as V→0 and always opposes ω.
    let rho_v_s = density * speed * s;

    // Pitch about +X (nose +Y → up +Z). Restoring drives α→α_trim (= cm0 /
    // pitch_stability); +pitch = nose up.
    let pitch = (cfg.cm0 - cfg.pitch_stability * alpha) * q * s * c
        + cfg.pitch_control * q * s * c * controls.pitch
        - cfg.pitch_damp * rho_v_s * c * c * omega_body.x;

    // Roll about +Y (nose axis). No static restoring; damping + control only.
    let roll = cfg.roll_control * q * s * b * controls.roll
        - cfg.roll_damp * rho_v_s * b * b * omega_body.y;

    // Yaw about +Z. Restoring drives β→0 (weathervane); +yaw = nose right.
    let yaw = -cfg.yaw_stability * q * s * b * beta
        + cfg.yaw_control * q * s * b * controls.yaw
        - cfg.yaw_damp * rho_v_s * b * b * omega_body.z;

    let torque = DVec3::new(pitch, roll, yaw);

    let force = clamp_len(force, MAX_FORCE_N);
    let torque = clamp_len(torque, MAX_TORQUE_NM);
    if !force.is_finite() || !torque.is_finite() {
        return AeroOutput::default();
    }
    AeroOutput { force, torque }
}

/// Per-axis control-moment **authority** (N·m) at full deflection (`control =
/// ±1`) for the current dynamic pressure, body frame (`x` = pitch, `y` = roll,
/// `z` = yaw). This is exactly the control term each axis gets in
/// [`evaluate_aero`] (`coeff · q̄ · S · arm`), so a control allocator can predict
/// how much torque a unit surface deflection buys and split a desired torque
/// between the aero surfaces and other effectors without over- or
/// under-actuating. Zero in vacuum / below the airspeed floor (matching the
/// evaluator's early-out), so callers get a clean "no aero authority" signal.
pub fn control_authority(cfg: &AeroConfig, density: f64, speed: f64) -> DVec3 {
    if density <= 0.0 || speed < MIN_SPEED_M_S {
        return DVec3::ZERO;
    }
    let q = 0.5 * density * speed * speed;
    let s = cfg.reference_area_m2;
    DVec3::new(
        cfg.pitch_control * q * s * cfg.reference_chord_m,
        cfg.roll_control * q * s * cfg.reference_span_m,
        cfg.yaw_control * q * s * cfg.reference_span_m,
    )
}

fn clamp_len(v: DVec3, max: f64) -> DVec3 {
    let len = v.length();
    if len > max { v * (max / len) } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn wing() -> AeroConfig {
        AeroConfig {
            reference_area_m2: 117.0,
            reference_chord_m: 3.4,
            reference_span_m: 30.0,
            lift_slope: 5.0,
            cl0: 0.25,
            cd0: 0.03,
            aspect_ratio: 8.0,
            pitch_stability: 0.4,
            yaw_stability: 0.3,
            pitch_damp: 0.8,
            roll_damp: 0.4,
            yaw_damp: 0.6,
            pitch_control: 0.5,
            roll_control: 0.6,
            yaw_control: 0.4,
            ..Default::default()
        }
    }

    #[test]
    fn lift_up_drag_back_in_level_flight() {
        // Forward (+Y), slight descent (−Z) → positive AoA.
        let v = DVec3::new(0.0, 100.0, -3.0);
        let out = evaluate_aero(v, DVec3::ZERO, 1.225, 0.0, &wing(), ControlInputs::default());
        assert!(out.force.z > 0.0, "lift up (+Z), got {}", out.force.z);
        assert!(out.force.y < 0.0, "drag opposes +Y, got {}", out.force.y);
    }

    #[test]
    fn pitch_up_disturbance_is_restored() {
        // Nose pitched above the velocity (flow from below, +α), no rotation.
        let v = DVec3::new(0.0, 100.0, -10.0);
        let out = evaluate_aero(v, DVec3::ZERO, 1.0, 0.0, &wing(), ControlInputs::default());
        // Restoring pitch moment must be nose-down: about −X (τx < 0).
        assert!(out.torque.x < 0.0, "pitch should restore (nose down), got {}", out.torque.x);
    }

    #[test]
    fn cm0_trims_at_positive_alpha() {
        let cfg = AeroConfig { cm0: 0.03, ..wing() };
        let alpha_trim = cfg.cm0 / cfg.pitch_stability;
        // Below trim AoA the moment pitches the nose up, above it down, and at
        // trim it vanishes — a hands-off stable cruise attitude.
        let v_level = DVec3::new(0.0, 100.0, 0.0); // α = 0 < α_trim
        let up = evaluate_aero(v_level, DVec3::ZERO, 1.0, 0.0, &cfg, ControlInputs::default());
        assert!(up.torque.x > 0.0, "below trim should pitch up, got {}", up.torque.x);
        let v_trim = DVec3::new(0.0, 100.0, -100.0 * alpha_trim.tan());
        let trim = evaluate_aero(v_trim, DVec3::ZERO, 1.0, 0.0, &cfg, ControlInputs::default());
        assert_relative_eq!(trim.torque.x, 0.0, epsilon = 1.0);
        let v_high = DVec3::new(0.0, 100.0, -20.0); // α ≈ 11° > α_trim
        let down = evaluate_aero(v_high, DVec3::ZERO, 1.0, 0.0, &cfg, ControlInputs::default());
        assert!(down.torque.x < 0.0, "above trim should pitch down, got {}", down.torque.x);
    }

    #[test]
    fn weathervane_restores_sideslip() {
        // Flow from the right (+β): nose should yaw right to align (τz < 0).
        let v = DVec3::new(10.0, 100.0, 0.0);
        let out = evaluate_aero(v, DVec3::ZERO, 1.0, 0.0, &wing(), ControlInputs::default());
        assert!(out.torque.z < 0.0, "yaw should restore toward wind, got {}", out.torque.z);
    }

    #[test]
    fn rotation_is_always_damped() {
        // Spinning on every axis in steady flight: the moment must oppose ω on
        // each axis (this is what makes a spin impossible to pump).
        let v = DVec3::new(0.0, 120.0, 0.0);
        let omega = DVec3::new(2.0, 2.0, 2.0);
        let out = evaluate_aero(v, omega, 1.0, 0.0, &wing(), ControlInputs::default());
        assert!(out.torque.x < 0.0, "pitch rate damped, got {}", out.torque.x);
        assert!(out.torque.y < 0.0, "roll rate damped, got {}", out.torque.y);
        assert!(out.torque.z < 0.0, "yaw rate damped, got {}", out.torque.z);
    }

    #[test]
    fn pull_pitches_nose_up() {
        let v = DVec3::new(0.0, 120.0, 0.0);
        let ctrl = ControlInputs { pitch: 1.0, ..Default::default() };
        let out = evaluate_aero(v, DVec3::ZERO, 1.0, 0.0, &wing(), ctrl);
        assert!(out.torque.x > 0.0, "pull should pitch nose up (+X), got {}", out.torque.x);
    }

    #[test]
    fn drag_scales_with_speed_squared() {
        let body = AeroConfig { reference_area_m2: 2.0, cd0: 1.0, ..Default::default() };
        let f1 = evaluate_aero(DVec3::new(0.0, 50.0, 0.0), DVec3::ZERO, 1.0, 0.0, &body, ControlInputs::default());
        let f2 = evaluate_aero(DVec3::new(0.0, 100.0, 0.0), DVec3::ZERO, 1.0, 0.0, &body, ControlInputs::default());
        assert_relative_eq!(f2.force.length() / f1.force.length(), 4.0, epsilon = 1e-9);
    }

    #[test]
    fn wave_drag_walls_off_the_transonic_regime() {
        // M_dd = 0.82 → M_crit ≈ 0.71. Same true airspeed: drag should be
        // unchanged with compressibility disabled (speed_of_sound = 0),
        // mildly higher just past divergence, and several × CD0 near Mach 1.
        let cfg = AeroConfig { mach_drag_divergence: 0.82, ..wing() };
        let a = 320.0;
        let v = DVec3::new(0.0, 0.95 * a, 0.0);
        let no_mach = evaluate_aero(v, DVec3::ZERO, 0.4, 0.0, &cfg, ControlInputs::default());
        let transonic = evaluate_aero(v, DVec3::ZERO, 0.4, a, &cfg, ControlInputs::default());
        // Compare the drag (−Y) component directly: lift is identical.
        let d0 = -no_mach.force.y;
        let d1 = -transonic.force.y;
        assert!(
            d1 > 2.0 * d0,
            "near-sonic drag should be several × subsonic ({d1:.0} vs {d0:.0} N)"
        );
        // Below the critical Mach the term must vanish exactly.
        let slow = DVec3::new(0.0, 0.5 * a, 0.0);
        let s0 = evaluate_aero(slow, DVec3::ZERO, 0.4, 0.0, &cfg, ControlInputs::default());
        let s1 = evaluate_aero(slow, DVec3::ZERO, 0.4, a, &cfg, ControlInputs::default());
        assert_relative_eq!(s0.force.y, s1.force.y, epsilon = 1e-9);
    }

    #[test]
    fn flaps_add_lift_and_drag_and_raise_the_stall_ceiling() {
        let cfg = AeroConfig { flap_dcl: 0.6, flap_dcd: 0.05, ..wing() };
        let v = DVec3::new(0.0, 60.0, 0.0);
        let clean = evaluate_aero(v, DVec3::ZERO, 1.2, 0.0, &cfg, ControlInputs::default());
        let landing = evaluate_aero(
            v,
            DVec3::ZERO,
            1.2,
            0.0,
            &cfg,
            ControlInputs { flap: 1.0, ..Default::default() },
        );
        assert!(landing.force.z > clean.force.z, "flaps must add lift");
        assert!(landing.force.y < clean.force.y, "flaps must add drag");

        // At stall AoA the clean wing is clamped; flaps must still lift more
        // (the clamp ceiling rises with the flap camber).
        let v_stall = DVec3::new(0.0, 60.0, -60.0 * cfg.stall_alpha.tan() * 1.2);
        let clean_stall =
            evaluate_aero(v_stall, DVec3::ZERO, 1.2, 0.0, &cfg, ControlInputs::default());
        let flap_stall = evaluate_aero(
            v_stall,
            DVec3::ZERO,
            1.2,
            0.0,
            &cfg,
            ControlInputs { flap: 1.0, ..Default::default() },
        );
        assert!(
            flap_stall.force.z > clean_stall.force.z,
            "flap lift must survive the stall clamp"
        );

        // Drag grows quadratically with deployment: takeoff flap (½) costs a
        // quarter of the landing-flap drag increment.
        let takeoff = evaluate_aero(
            v,
            DVec3::ZERO,
            1.2,
            0.0,
            &cfg,
            ControlInputs { flap: 0.5, ..Default::default() },
        );
        let q_s = 0.5 * 1.2 * 60.0 * 60.0 * cfg.reference_area_m2;
        let takeoff_dcd = (clean.force.y - takeoff.force.y) / q_s;
        let landing_dcd = (clean.force.y - landing.force.y) / q_s;
        // Induced drag also moves with CL, so allow a loose band around 4×.
        assert!(
            landing_dcd / takeoff_dcd > 2.0,
            "landing flap should cost disproportionately more drag ({landing_dcd:.4} vs {takeoff_dcd:.4})"
        );
    }

    #[test]
    fn spoilers_dump_lift_and_add_drag() {
        let cfg = AeroConfig { spoiler_dcl: -0.08, spoiler_dcd: 0.035, ..wing() };
        let v = DVec3::new(0.0, 150.0, -3.0);
        let clean = evaluate_aero(v, DVec3::ZERO, 1.0, 0.0, &cfg, ControlInputs::default());
        let braked = evaluate_aero(
            v,
            DVec3::ZERO,
            1.0,
            0.0,
            &cfg,
            ControlInputs { spoiler: 1.0, ..Default::default() },
        );
        assert!(braked.force.z < clean.force.z, "spoilers must dump lift");
        assert!(braked.force.y < clean.force.y, "spoilers must add drag");
    }

    #[test]
    fn bluff_body_weathervanes_without_lift() {
        // A capsule: no lift, but yaw/pitch stability aligns it with the wind.
        let capsule = AeroConfig {
            reference_area_m2: 12.0,
            reference_chord_m: 4.0,
            reference_span_m: 4.0,
            cd0: 1.2,
            pitch_stability: 0.5,
            yaw_stability: 0.5,
            pitch_damp: 0.5,
            yaw_damp: 0.5,
            ..Default::default()
        };
        // Tumbling with sideslip: should both damp and restore, never amplify.
        let v = DVec3::new(8.0, 80.0, 0.0);
        let out = evaluate_aero(v, DVec3::new(0.0, 0.0, 1.0), 1.0, 0.0, &capsule, ControlInputs::default());
        assert!(out.torque.z < 0.0, "capsule should weathervane + damp yaw, got {}", out.torque.z);
        assert!(out.force.y < 0.0, "capsule drag opposes motion, got {}", out.force.y);
    }
}
