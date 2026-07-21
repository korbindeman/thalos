//! Surface-local frame (SLF): a body-fixed tangent frame anchored at a
//! surface point, expressed as a classical Y-up game world.
//!
//! The frame co-rotates with the body (it is a body-fixed frame with a
//! tangent-aligned basis and a surface-point origin), so a craft parked on
//! the ground is at rest (`v ≈ 0`), the terrain is static geometry, and the
//! co-rotating atmosphere makes frame velocity equal airspeed. Coordinates
//! are small (meters to kilometers from the anchor); [`reanchor`] translates
//! state exactly when the craft drifts too far from the origin.
//!
//! Axis mapping (right-handed, Bevy convention):
//! - `+X` = east  = `normalize(spin_axis_body × up)`
//! - `+Y` = up    = the anchor direction
//! - `+Z` = south = `east × up`
//!
//! Frame dynamics are exact, not flat-earth: gravity is radial per position,
//! and [`surface_local_acceleration`] includes the centrifugal and Coriolis
//! terms of the rotating frame. See `docs/surface_local.md`.

use glam::{DMat3, DQuat, DVec3};

use crate::body_fixed::{BodyFixedFrameState, body_fixed_to_inertial, inertial_to_body_fixed};
use crate::canonical::TranslationalState;
use crate::types::{AttitudeState, BodyState};

/// Below this sine-of-colatitude the anchor is considered on the spin axis
/// and the east direction is degenerate; a deterministic fallback seed is
/// used instead (same rule as `TerrainPatchBasis::from_normal`).
const POLE_DEGENERACY_EPS: f64 = 1e-6;

/// A body-fixed surface anchor: where the SLF origin sits on the body.
///
/// `elevation_m` is the terrain height above the body reference radius at
/// the anchor, supplied by the caller — canonical code never samples terrain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceAnchor {
    /// Unit body-fixed direction from the body center to the anchor.
    pub dir_body: DVec3,
    /// Terrain elevation above the body reference radius (m).
    pub elevation_m: f64,
}

/// The surface-local tangent frame at a [`SurfaceAnchor`].
///
/// Constant for a given (body, anchor) pair: both the basis and the spin
/// vector are body-fixed quantities, so the frame never needs per-frame
/// updates as the body rotates — only a new anchor invalidates it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceLocalFrame {
    pub anchor: SurfaceAnchor,
    /// Rotation taking body-fixed coordinates to SLF coordinates.
    pub rotation_body_to_frame: DQuat,
    /// Body center → anchor point, in body-fixed coordinates (m).
    pub anchor_point_body_m: DVec3,
    /// Body spin angular velocity in body-fixed coordinates (rad/s).
    /// Constant for a rigidly spinning body.
    pub spin_body: DVec3,
    /// Body reference radius (m), cached for altitude helpers.
    pub radius_m: f64,
}

/// Craft state expressed in a [`SurfaceLocalFrame`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceLocalState {
    /// Position relative to the anchor, SLF axes (m).
    pub position_m: DVec3,
    /// Velocity relative to the (co-rotating) frame, SLF axes (m/s).
    /// Equals airspeed in a co-rotating atmosphere.
    pub velocity_m_s: DVec3,
    /// Craft orientation relative to the SLF.
    pub orientation_frame: DQuat,
    /// Craft angular velocity relative to the frame, expressed in the craft
    /// body frame (same convention as [`BodyFixedFrameState`]; the SLF
    /// co-rotates with the body, so the value is identical in both frames).
    pub angular_velocity_body: DVec3,
}

impl SurfaceLocalFrame {
    pub fn new(body: &BodyState, anchor: SurfaceAnchor) -> Self {
        let up = anchor.dir_body.normalize();
        let spin_body = body.orientation.inverse() * body.angular_velocity;
        let east_raw = spin_body.normalize_or_zero().cross(up);
        let east = if east_raw.length() < POLE_DEGENERACY_EPS {
            // On (or numerically at) the spin axis east is undefined; fall
            // back to the TerrainPatchBasis::from_normal seed rule so the
            // frame basis and the collider basis can never diverge.
            let seed = if up.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
            seed.cross(up).normalize()
        } else {
            east_raw.normalize()
        };
        let south = east.cross(up).normalize();
        let frame_to_body = DMat3::from_cols(east, up, south);
        let rotation_body_to_frame = DQuat::from_mat3(&frame_to_body).normalize().inverse();
        Self {
            anchor: SurfaceAnchor {
                dir_body: up,
                elevation_m: anchor.elevation_m,
            },
            rotation_body_to_frame,
            anchor_point_body_m: up * (body.radius_m + anchor.elevation_m),
            spin_body,
            radius_m: body.radius_m,
        }
    }

    /// Body center → craft, in SLF axes.
    pub fn body_center_offset(&self, position_m: DVec3) -> DVec3 {
        self.rotation_body_to_frame * self.anchor_point_body_m + position_m
    }

    /// Body spin angular velocity in SLF axes.
    pub fn spin_frame(&self) -> DVec3 {
        self.rotation_body_to_frame * self.spin_body
    }
}

/// SLF ↔ body-fixed is a constant rotation + translation: both frames
/// co-rotate with the body, so no velocity terms appear.
pub fn body_fixed_to_surface_local(
    frame: &SurfaceLocalFrame,
    state: BodyFixedFrameState,
) -> SurfaceLocalState {
    let rot = frame.rotation_body_to_frame;
    SurfaceLocalState {
        position_m: rot * (state.translation_body.position - frame.anchor_point_body_m),
        velocity_m_s: rot * state.translation_body.velocity,
        orientation_frame: (rot * state.orientation_body).normalize(),
        angular_velocity_body: state.angular_velocity_body,
    }
}

pub fn surface_local_to_body_fixed(
    frame: &SurfaceLocalFrame,
    state: SurfaceLocalState,
) -> BodyFixedFrameState {
    let rot = frame.rotation_body_to_frame.inverse();
    BodyFixedFrameState {
        translation_body: TranslationalState {
            position: rot * state.position_m + frame.anchor_point_body_m,
            velocity: rot * state.velocity_m_s,
        },
        orientation_body: (rot * state.orientation_frame).normalize(),
        angular_velocity_body: state.angular_velocity_body,
    }
}

pub fn inertial_to_surface_local(
    body: &BodyState,
    frame: &SurfaceLocalFrame,
    translation: TranslationalState,
    attitude: AttitudeState,
) -> SurfaceLocalState {
    body_fixed_to_surface_local(frame, inertial_to_body_fixed(body, translation, attitude))
}

pub fn surface_local_to_inertial(
    body: &BodyState,
    frame: &SurfaceLocalFrame,
    state: SurfaceLocalState,
) -> (TranslationalState, AttitudeState) {
    body_fixed_to_inertial(body, surface_local_to_body_fixed(frame, state))
}

/// Total non-contact acceleration on a coasting craft in the SLF: exact
/// radial gravity plus the centrifugal and Coriolis terms of the rotating
/// frame. All inputs/outputs in SLF axes.
pub fn surface_local_acceleration(
    gm: f64,
    frame: &SurfaceLocalFrame,
    position_m: DVec3,
    velocity_m_s: DVec3,
) -> DVec3 {
    let r = frame.body_center_offset(position_m);
    let omega = frame.spin_frame();
    let r_len = r.length();
    let gravity = -gm * r / (r_len * r_len * r_len);
    let centrifugal = -omega.cross(omega.cross(r));
    let coriolis = -2.0 * omega.cross(velocity_m_s);
    gravity + centrifugal + coriolis
}

/// Altitude above the body reference radius (sea level) for an SLF position.
pub fn altitude_asl_m(frame: &SurfaceLocalFrame, position_m: DVec3) -> f64 {
    frame.body_center_offset(position_m).length() - frame.radius_m
}

/// Exact local radial up at an SLF position. Near the anchor this is ≈ +Y;
/// it curves with the planet across the frame.
pub fn radial_up(frame: &SurfaceLocalFrame, position_m: DVec3) -> DVec3 {
    frame.body_center_offset(position_m).normalize()
}

/// Translate a state from one frame to another exactly (f64, via body-fixed;
/// no inertial round trip, so no body snapshot is needed). Both frames must
/// belong to the same body.
pub fn reanchor(
    old_frame: &SurfaceLocalFrame,
    new_frame: &SurfaceLocalFrame,
    state: SurfaceLocalState,
) -> SurfaceLocalState {
    body_fixed_to_surface_local(new_frame, surface_local_to_body_fixed(old_frame, state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body_centered::{body_centered_to_inertial, inertial_to_body_centered};
    use crate::canonical::Epoch;

    /// Test spin rate: slow enough that centrifugal (ω²r ≈ 1e-3 m/s²) stays
    /// far below gravity (≈ 10 m/s²), as on any real body, while Coriolis
    /// still moves the dynamics-equivalence trajectory by ~1 m — far above
    /// its tolerance, so a sign error cannot pass.
    const SPIN_RAD_S: f64 = 1e-3;

    /// A spinning, tilted body following the patched_conics convention:
    /// `orientation = R_y(phase) * R_x(tilt)` with world angular velocity
    /// along world Y, so `spin_body = orientation⁻¹·ω` is phase-invariant.
    fn test_body(phase: f64) -> BodyState {
        let tilt = DQuat::from_rotation_x(0.4);
        let orientation = DQuat::from_rotation_y(phase) * tilt;
        BodyState {
            id: 0,
            epoch: Epoch(0.0),
            position: DVec3::new(100.0, 20.0, -50.0),
            velocity: DVec3::new(5.0, 0.0, -2.0),
            orientation,
            angular_velocity: DVec3::Y * SPIN_RAD_S,
            mass_kg: 1.0e20,
            // g = gm / r² ≈ 10 m/s² at the 1000 m reference radius.
            gm: 1.0e7,
            radius_m: 1000.0,
        }
    }

    fn test_frame(body: &BodyState) -> SurfaceLocalFrame {
        SurfaceLocalFrame::new(
            body,
            SurfaceAnchor {
                dir_body: DVec3::new(0.3, 0.8, -0.4).normalize(),
                elevation_m: 12.0,
            },
        )
    }

    #[test]
    fn basis_is_right_handed_orthonormal_y_up() {
        let body = test_body(0.7);
        let frame = test_frame(&body);
        let rot = frame.rotation_body_to_frame;

        // up maps to +Y.
        assert!((rot * frame.anchor.dir_body - DVec3::Y).length() < 1e-12);
        // Orthonormal with det +1: quaternion is unit and X×Y=Z by
        // construction; verify via the basis vectors round-tripped.
        let east = rot.inverse() * DVec3::X;
        let up = rot.inverse() * DVec3::Y;
        let south = rot.inverse() * DVec3::Z;
        assert!((east.cross(up) - south).length() < 1e-12);
        assert!(east.length() - 1.0 < 1e-12 && up.length() - 1.0 < 1e-12);
        // East is horizontal (no up component) and aligned with spin × up.
        assert!(east.dot(frame.anchor.dir_body).abs() < 1e-12);
        let expected_east = frame.spin_body.normalize().cross(frame.anchor.dir_body);
        assert!((east - expected_east.normalize()).length() < 1e-12);
    }

    #[test]
    fn pole_anchor_falls_back_deterministically() {
        let body = test_body(0.0);
        let pole_dir = body.orientation.inverse() * body.angular_velocity;
        let anchor = SurfaceAnchor {
            dir_body: pole_dir.normalize(),
            elevation_m: 0.0,
        };
        let a = SurfaceLocalFrame::new(&body, anchor);
        let b = SurfaceLocalFrame::new(&body, anchor);
        assert_eq!(a.rotation_body_to_frame, b.rotation_body_to_frame);
        let rot = a.rotation_body_to_frame;
        assert!((rot * anchor.dir_body - DVec3::Y).length() < 1e-9);
        let east = rot.inverse() * DVec3::X;
        let south = rot.inverse() * DVec3::Z;
        assert!((east.cross(rot.inverse() * DVec3::Y) - south).length() < 1e-9);
    }

    #[test]
    fn inertial_surface_local_round_trip() {
        let body = test_body(1.3);
        let frame = test_frame(&body);
        let state = SurfaceLocalState {
            position_m: DVec3::new(800.0, 40.0, -300.0),
            velocity_m_s: DVec3::new(30.0, -2.0, 4.0),
            orientation_frame: DQuat::from_rotation_x(0.2) * DQuat::from_rotation_z(-0.5),
            angular_velocity_body: DVec3::new(0.01, 0.02, 0.03),
        };

        let (translation, attitude) = surface_local_to_inertial(&body, &frame, state);
        let round_trip = inertial_to_surface_local(&body, &frame, translation, attitude);

        assert!((round_trip.position_m - state.position_m).length() < 1e-9);
        assert!((round_trip.velocity_m_s - state.velocity_m_s).length() < 1e-9);
        assert!(
            round_trip
                .orientation_frame
                .angle_between(state.orientation_frame)
                < 1e-9
        );
        assert!((round_trip.angular_velocity_body - state.angular_velocity_body).length() < 1e-9);
    }

    #[test]
    fn slf_body_fixed_round_trip_is_exact() {
        let body = test_body(0.2);
        let frame = test_frame(&body);
        let state = SurfaceLocalState {
            position_m: DVec3::new(-120.0, 5.0, 64.0),
            velocity_m_s: DVec3::new(1.0, 2.0, 3.0),
            orientation_frame: DQuat::from_rotation_y(0.8),
            angular_velocity_body: DVec3::new(0.1, 0.0, -0.2),
        };
        let bf = surface_local_to_body_fixed(&frame, state);
        let rt = body_fixed_to_surface_local(&frame, bf);
        assert!((rt.position_m - state.position_m).length() < 1e-9);
        assert!((rt.velocity_m_s - state.velocity_m_s).length() < 1e-9);
        assert!(rt.orientation_frame.angle_between(state.orientation_frame) < 1e-9);
    }

    #[test]
    fn gravity_is_down_at_anchor_and_curves_away() {
        let body = test_body(0.0);
        let frame = test_frame(&body);

        // At the anchor with zero velocity, gravity (minus the tiny
        // centrifugal term) points along -Y to within the spin flattening.
        let a = surface_local_acceleration(body.gm, &frame, DVec3::ZERO, DVec3::ZERO);
        let g_mag = body.gm / (body.radius_m + frame.anchor.elevation_m).powi(2);
        assert!(a.y < 0.0);
        assert!((a.length() - g_mag).abs() / g_mag < 0.01);

        // 200 m east, gravity gains a -X (back toward the anchor) component.
        let a_east =
            surface_local_acceleration(body.gm, &frame, DVec3::new(200.0, 0.0, 0.0), DVec3::ZERO);
        assert!(a_east.x < 0.0);

        // Altitude helper: at the anchor, ASL equals the anchor elevation.
        assert!((altitude_asl_m(&frame, DVec3::ZERO) - frame.anchor.elevation_m).abs() < 1e-9);
        // Radial up at the anchor is +Y.
        assert!((radial_up(&frame, DVec3::ZERO) - DVec3::Y).length() < 1e-9);
    }

    /// Integrate a short free fall in the body-centered inertial frame
    /// (gravity only, no fictitious forces) and in the SLF (gravity +
    /// centrifugal + Coriolis), and check both land on the same inertial
    /// state. This is the test that catches a Coriolis sign error.
    #[test]
    fn slf_dynamics_match_inertial_free_fall() {
        // Body pinned at the origin with zero velocity so the inertial-side
        // integration doesn't need an orbiting gravity center.
        let mut body = test_body(0.0);
        body.position = DVec3::ZERO;
        body.velocity = DVec3::ZERO;
        let frame = test_frame(&body);

        let body_at = |t: f64| -> BodyState {
            // Spin about world Y: orientation(t) = R_y(phase + ω t) · R_x(tilt).
            let orientation = DQuat::from_rotation_y(SPIN_RAD_S * t) * body.orientation;
            BodyState {
                orientation,
                epoch: Epoch(t),
                ..body
            }
        };

        // Initial state: 500 m above the anchor, moving east + up.
        let slf0 = SurfaceLocalState {
            position_m: DVec3::new(0.0, 500.0, 0.0),
            velocity_m_s: DVec3::new(40.0, 10.0, -5.0),
            orientation_frame: DQuat::IDENTITY,
            angular_velocity_body: DVec3::ZERO,
        };
        let (inertial0, attitude0) = surface_local_to_inertial(&body_at(0.0), &frame, slf0);

        let t_end = 5.0;
        let dt = 1e-3;
        let steps = (t_end / dt) as usize;

        // RK4 in the body-centered inertial frame: a = -gm p / |p|^3.
        let accel_inertial =
            |p: DVec3| -> DVec3 { -body.gm * p / (p.length() * p.length() * p.length()) };
        let bc0 = inertial_to_body_centered(&body_at(0.0), inertial0, attitude0);
        let mut p_i = bc0.translation_bc.position;
        let mut v_i = bc0.translation_bc.velocity;
        for _ in 0..steps {
            let (k1p, k1v) = (v_i, accel_inertial(p_i));
            let (k2p, k2v) = (v_i + 0.5 * dt * k1v, accel_inertial(p_i + 0.5 * dt * k1p));
            let (k3p, k3v) = (v_i + 0.5 * dt * k2v, accel_inertial(p_i + 0.5 * dt * k2p));
            let (k4p, k4v) = (v_i + dt * k3v, accel_inertial(p_i + dt * k3p));
            p_i += dt / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
            v_i += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        }

        // RK4 in the SLF with the full fictitious-force acceleration.
        let accel_slf =
            |p: DVec3, v: DVec3| -> DVec3 { surface_local_acceleration(body.gm, &frame, p, v) };
        let mut p_s = slf0.position_m;
        let mut v_s = slf0.velocity_m_s;
        for _ in 0..steps {
            let (k1p, k1v) = (v_s, accel_slf(p_s, v_s));
            let (k2p, k2v) = (
                v_s + 0.5 * dt * k1v,
                accel_slf(p_s + 0.5 * dt * k1p, v_s + 0.5 * dt * k1v),
            );
            let (k3p, k3v) = (
                v_s + 0.5 * dt * k2v,
                accel_slf(p_s + 0.5 * dt * k2p, v_s + 0.5 * dt * k2v),
            );
            let (k4p, k4v) = (v_s + dt * k3v, accel_slf(p_s + dt * k3p, v_s + dt * k3v));
            p_s += dt / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
            v_s += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        }

        // Convert the SLF result to inertial at t_end and compare.
        let body_end = body_at(t_end);
        let slf_end = SurfaceLocalState {
            position_m: p_s,
            velocity_m_s: v_s,
            orientation_frame: DQuat::IDENTITY,
            angular_velocity_body: DVec3::ZERO,
        };
        let (slf_inertial, _) = surface_local_to_inertial(&body_end, &frame, slf_end);
        let (ref_inertial, _) = body_centered_to_inertial(
            &body_end,
            crate::body_centered::BodyCenteredState {
                translation_bc: TranslationalState {
                    position: p_i,
                    velocity: v_i,
                },
                attitude: AttitudeState {
                    orientation: DQuat::IDENTITY,
                    angular_velocity: DVec3::ZERO,
                },
            },
        );

        assert!(
            (slf_inertial.position - ref_inertial.position).length() < 1e-6,
            "position diverged: {:?}",
            (slf_inertial.position - ref_inertial.position).length()
        );
        assert!(
            (slf_inertial.velocity - ref_inertial.velocity).length() < 1e-6,
            "velocity diverged: {:?}",
            (slf_inertial.velocity - ref_inertial.velocity).length()
        );
    }

    #[test]
    fn reanchor_preserves_inertial_state() {
        let body = test_body(0.9);
        let frame_a = test_frame(&body);
        let frame_b = SurfaceLocalFrame::new(
            &body,
            SurfaceAnchor {
                dir_body: DVec3::new(0.35, 0.78, -0.38).normalize(),
                elevation_m: -3.0,
            },
        );
        let state = SurfaceLocalState {
            position_m: DVec3::new(1500.0, 80.0, -200.0),
            velocity_m_s: DVec3::new(50.0, 1.0, -8.0),
            orientation_frame: DQuat::from_rotation_z(0.4),
            angular_velocity_body: DVec3::new(0.0, 0.1, 0.0),
        };

        let (inertial_a, attitude_a) = surface_local_to_inertial(&body, &frame_a, state);
        let moved = reanchor(&frame_a, &frame_b, state);
        let (inertial_b, attitude_b) = surface_local_to_inertial(&body, &frame_b, moved);

        assert!((inertial_a.position - inertial_b.position).length() < 1e-9);
        assert!((inertial_a.velocity - inertial_b.velocity).length() < 1e-9);
        assert!(attitude_a.orientation.angle_between(attitude_b.orientation) < 1e-9);
        assert!((attitude_a.angular_velocity - attitude_b.angular_velocity).length() < 1e-9);
    }
}
