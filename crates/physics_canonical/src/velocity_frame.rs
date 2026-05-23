//! Velocity reference frames for the navball speed mode.
//!
//! A [`VelocityReferenceFrame`] selects which reference velocity is
//! subtracted from the craft's inertial velocity before deriving the
//! navball velocity markers, the speed readout, and the SAS
//! prograde/normal/radial holds. [`nav_basis`] is the single place that
//! math lives; the game layer owns only *which* frame is active and
//! evaluates this function with its stage-correct body state (the SAS
//! control path in `Physics` reads the ephemeris, the navball/HUD path
//! after `Sync` reads the per-frame solar-system snapshot — both agree).

use glam::DVec3;

use crate::types::{BodyState, StateVector};

/// Which velocity the navball speed mode is expressed relative to.
///
/// Mirrors KSP's navball speed display:
/// - `Orbit` — relative to the dominant body's center (body-centered
///   inertial). Reference velocity = `body.velocity`.
/// - `Surface` — relative to the co-rotating surface under the craft.
///   Reference velocity = `body.velocity + ω × r`.
/// - `Target` — relative to a selected target's motion. Reference
///   velocity = `target.velocity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VelocityReferenceFrame {
    #[default]
    Orbit,
    Surface,
    Target,
}

impl VelocityReferenceFrame {
    /// Short uppercase label for the HUD readout: `ORBITAL`, `SURFACE`,
    /// `TARGET`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Orbit => "ORBITAL",
            Self::Surface => "SURFACE",
            Self::Target => "TARGET",
        }
    }
}

/// The navball velocity basis for the active frame, evaluated at the
/// craft's current state.
///
/// `reference_vel` and `speed` are always defined. The direction fields
/// are `None` when their generating vector is degenerate (zero relative
/// velocity ⇒ no prograde/normal; craft at the body center ⇒ no radial),
/// matching the navball's per-marker hide behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavBasis {
    /// Frame velocity at the craft — the vector subtracted from the
    /// craft's inertial velocity.
    pub reference_vel: DVec3,
    /// Speed relative to the frame, `|craft.velocity − reference_vel|`.
    pub speed: f64,
    /// Unit velocity-relative-to-frame direction (prograde).
    pub prograde: Option<DVec3>,
    /// Unit orbital-plane normal, `r × v_rel`.
    pub normal: Option<DVec3>,
    /// Unit radial-out from the dominant body, `r`.
    pub radial: Option<DVec3>,
}

/// Build the [`NavBasis`] for `frame`.
///
/// `body` is the dominant (SOI) body; `target` is required only for
/// [`VelocityReferenceFrame::Target`]. Returns `None` only when the
/// Target frame is requested without a target — every other frame always
/// produces a basis (with possibly-`None` direction fields).
pub fn nav_basis(
    frame: VelocityReferenceFrame,
    craft: StateVector,
    body: &BodyState,
    target: Option<&BodyState>,
) -> Option<NavBasis> {
    let rel_pos = craft.position - body.position;
    let reference_vel = match frame {
        VelocityReferenceFrame::Orbit => body.velocity,
        VelocityReferenceFrame::Surface => body.velocity + body.angular_velocity.cross(rel_pos),
        VelocityReferenceFrame::Target => target?.velocity,
    };
    let rel_vel = craft.velocity - reference_vel;
    Some(NavBasis {
        reference_vel,
        speed: rel_vel.length(),
        prograde: rel_vel.try_normalize(),
        normal: rel_pos.cross(rel_vel).try_normalize(),
        radial: rel_pos.try_normalize(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Epoch;
    use glam::DQuat;

    fn body() -> BodyState {
        BodyState {
            id: 0,
            epoch: Epoch(0.0),
            position: DVec3::new(1000.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 2000.0, 0.0),
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::Y * 1.0e-3,
            mass_kg: 1.0e22,
            gm: 1.0,
            radius_m: 100.0,
        }
    }

    #[test]
    fn orbit_frame_subtracts_body_velocity() {
        let b = body();
        let craft = StateVector {
            position: b.position + DVec3::X * 500.0,
            velocity: b.velocity + DVec3::Z * 300.0,
        };
        let basis = nav_basis(VelocityReferenceFrame::Orbit, craft, &b, None).unwrap();
        assert!((basis.reference_vel - b.velocity).length() < 1e-9);
        assert!((basis.speed - 300.0).abs() < 1e-6);
        assert!((basis.prograde.unwrap() - DVec3::Z).length() < 1e-9);
        // Radial-out is +X (craft offset from body is +X).
        assert!((basis.radial.unwrap() - DVec3::X).length() < 1e-9);
    }

    #[test]
    fn surface_frame_zeroes_speed_for_corotating_craft() {
        let b = body();
        let rel_pos = DVec3::X * 500.0;
        // A craft moving exactly with the rotating surface.
        let craft = StateVector {
            position: b.position + rel_pos,
            velocity: b.velocity + b.angular_velocity.cross(rel_pos),
        };
        let surface = nav_basis(VelocityReferenceFrame::Surface, craft, &b, None).unwrap();
        assert!(
            surface.speed < 1e-9,
            "co-rotating craft should read ~0 surface speed, got {}",
            surface.speed
        );
        assert!(surface.prograde.is_none());
        // The orbit frame still sees the rotation as motion.
        let orbit = nav_basis(VelocityReferenceFrame::Orbit, craft, &b, None).unwrap();
        assert!(orbit.speed > 1e-6);
    }

    #[test]
    fn surface_and_orbit_reference_differ_by_rotation_term() {
        let b = body();
        let rel_pos = DVec3::X * 500.0;
        let craft = StateVector {
            position: b.position + rel_pos,
            velocity: b.velocity + DVec3::Z * 50.0,
        };
        let orbit = nav_basis(VelocityReferenceFrame::Orbit, craft, &b, None).unwrap();
        let surface = nav_basis(VelocityReferenceFrame::Surface, craft, &b, None).unwrap();
        let diff = surface.reference_vel - orbit.reference_vel;
        assert!((diff - b.angular_velocity.cross(rel_pos)).length() < 1e-9);
    }

    #[test]
    fn target_frame_requires_target() {
        let b = body();
        let craft = StateVector {
            position: b.position + DVec3::X * 500.0,
            velocity: b.velocity,
        };
        assert!(nav_basis(VelocityReferenceFrame::Target, craft, &b, None).is_none());

        let mut tgt = body();
        tgt.velocity = DVec3::new(10.0, 2000.0, 0.0);
        let basis = nav_basis(VelocityReferenceFrame::Target, craft, &b, Some(&tgt)).unwrap();
        assert!((basis.reference_vel - tgt.velocity).length() < 1e-9);
    }
}
