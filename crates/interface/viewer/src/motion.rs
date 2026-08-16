//! Spatially-neutral freecam motion.

use bevy::math::{DMat3, DQuat, DVec3, Vec2, Vec3};

use crate::{CameraOptics, ViewerPreferences};

pub const VIEWER_MIN_SPEED_M_S: f64 = 1.0;
pub const VIEWER_MAX_SPEED_M_S: f64 = 1.0e7;
const SHIFT_MULTIPLIER: f64 = 5.0;
const CONTROL_MULTIPLIER: f64 = 0.2;
const LOOK_SENSITIVITY: f64 = 0.0025;
const SCROLL_LOG_STEP: f64 = 0.20;
const ROLL_RATE_RAD_S: f64 = 1.5;
const LEVEL_MAX_SIN_PITCH: f64 = 0.9998;
const LEVEL_RATE_HZ: f64 = 2.0;
const LEVEL_ROLL_SUPPRESS_AUTHORITY: f32 = 0.5;
const ZOOM_FACTOR: f32 = 4.0;
const ZOOM_LERP_RATE: f32 = 12.0;

/// Stable f64 camera pose in the coordinate frame selected by an application.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewerPose {
    pub position: DVec3,
    pub rotation: DQuat,
}

/// Semantic input for one viewer frame. `movement` is camera right/up/forward.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ViewerIntent {
    pub look_delta: Vec2,
    pub look_active: bool,
    pub movement: Vec3,
    pub roll_axis: f32,
    pub speed_scroll_lines: f32,
    pub fast: bool,
    pub slow: bool,
    pub toggle_level: bool,
    pub toggle_ground: bool,
    pub spring_zoom: bool,
}

/// Local vertical and the strength with which it owns the horizon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelLock {
    pub up: DVec3,
    pub authority: f32,
}

impl LevelLock {
    pub fn new(up: DVec3, authority: f32) -> Option<Self> {
        Some(Self {
            up: up.try_normalize()?,
            authority: authority.clamp(0.0, 1.0),
        })
    }
}

/// Apply input to a stable pose. Spatial floors and bounds are deliberately
/// left to the application adapter after this returns.
pub fn drive_motion(
    pose: &mut ViewerPose,
    preferences: &mut ViewerPreferences,
    intent: ViewerIntent,
    dt_s: f64,
    level_lock: Option<LevelLock>,
) -> bool {
    if intent.speed_scroll_lines != 0.0 {
        let log =
            preferences.base_speed_m_s.ln() + intent.speed_scroll_lines as f64 * SCROLL_LOG_STEP;
        preferences.base_speed_m_s = log.exp().clamp(VIEWER_MIN_SPEED_M_S, VIEWER_MAX_SPEED_M_S);
    }
    if intent.toggle_level {
        preferences.level_to_up = !preferences.level_to_up;
    }
    if intent.toggle_ground {
        preferences.ground_collision = !preferences.ground_collision;
    }

    let lock = preferences.level_to_up.then_some(level_lock).flatten();
    let mut changed = false;

    if intent.look_active && intent.look_delta != Vec2::ZERO {
        let yaw_axis = control_up(pose.rotation, lock);
        let right = pose.rotation * DVec3::X;
        let yaw = DQuat::from_axis_angle(yaw_axis, -intent.look_delta.x as f64 * LOOK_SENSITIVITY);
        let pitch = DQuat::from_axis_angle(right, -intent.look_delta.y as f64 * LOOK_SENSITIVITY);
        pose.rotation = (pitch * yaw * pose.rotation).normalize();
        changed = true;
    }

    if intent.roll_axis != 0.0
        && !lock.is_some_and(|lock| lock.authority >= LEVEL_ROLL_SUPPRESS_AUTHORITY)
    {
        pose.rotation = (pose.rotation
            * DQuat::from_rotation_z(intent.roll_axis as f64 * ROLL_RATE_RAD_S * dt_s))
        .normalize();
        changed = true;
    }

    if intent.movement != Vec3::ZERO && dt_s > 0.0 {
        let forward = pose.rotation * DVec3::NEG_Z;
        let right = pose.rotation * DVec3::X;
        let vertical = control_up(pose.rotation, lock);
        let direction = right * intent.movement.x as f64
            + vertical * intent.movement.y as f64
            + forward * intent.movement.z as f64;
        if let Some(direction) = direction.try_normalize() {
            let multiplier = speed_multiplier(intent.fast, intent.slow);
            pose.position += direction * preferences.base_speed_m_s * multiplier * dt_s;
            changed = true;
        }
    }

    changed
}

/// Multiplier applied to the configured cruise speed for the current modifiers.
pub fn speed_multiplier(fast: bool, slow: bool) -> f64 {
    if fast {
        SHIFT_MULTIPLIER
    } else if slow {
        CONTROL_MULTIPLIER
    } else {
        1.0
    }
}

fn control_up(rotation: DQuat, lock: Option<LevelLock>) -> DVec3 {
    let camera_up = rotation * DVec3::Y;
    let Some(lock) = lock else {
        return camera_up;
    };
    camera_up
        .lerp(lock.up, lock.authority as f64)
        .try_normalize()
        .unwrap_or(lock.up)
}

/// Ease a pose toward a roll-free, pitch-clamped horizon after the application
/// has applied its floor and bounds.
pub fn settle_level_lock(pose: &mut ViewerPose, lock: LevelLock, dt_s: f64) -> bool {
    let weight = level_lock_weight(lock.authority, dt_s);
    if weight <= 0.0 {
        return false;
    }

    let forward = pose.rotation * DVec3::NEG_Z;
    let sin_pitch = forward
        .dot(lock.up)
        .clamp(-LEVEL_MAX_SIN_PITCH, LEVEL_MAX_SIN_PITCH);
    let horizontal = (forward - lock.up * forward.dot(lock.up))
        .try_normalize()
        .or_else(|| lock.up.cross(pose.rotation * DVec3::X).try_normalize());
    let Some(horizontal) = horizontal else {
        return false;
    };
    let cos_pitch = (1.0 - sin_pitch * sin_pitch).max(0.0).sqrt();
    let level_forward = horizontal * cos_pitch + lock.up * sin_pitch;
    let right = level_forward.cross(lock.up).normalize();
    let corrected_up = right.cross(level_forward).normalize();
    let target = DQuat::from_mat3(&DMat3::from_cols(right, corrected_up, -level_forward));
    let previous = pose.rotation;
    pose.rotation = previous.slerp(target, weight).normalize();
    previous.dot(pose.rotation).abs() < 1.0 - 1.0e-12
}

fn level_lock_weight(authority: f32, dt_s: f64) -> f64 {
    if authority >= 1.0 {
        return 1.0;
    }
    if authority <= 0.0 || dt_s <= 0.0 {
        return 0.0;
    }
    let authority = authority as f64;
    let rate_hz = LEVEL_RATE_HZ * authority / (1.0 - authority);
    1.0 - (-rate_hz * dt_s).exp()
}

/// Horizon authority from the angular diameter of a spherical anchor.
pub fn level_lock_authority(radius_m: f64, body_radius_m: f64) -> f32 {
    const FULL_ANGLE_RAD: f64 = 120.0 * std::f64::consts::PI / 180.0;
    const RELEASE_ANGLE_RAD: f64 = 45.0 * std::f64::consts::PI / 180.0;
    if !radius_m.is_finite() || radius_m <= 0.0 || body_radius_m <= 0.0 {
        return 0.0;
    }
    let angular_diameter = 2.0 * (body_radius_m / radius_m).min(1.0).asin();
    let t = ((angular_diameter - RELEASE_ANGLE_RAD) / (FULL_ANGLE_RAD - RELEASE_ANGLE_RAD))
        .clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t)) as f32
}

pub fn update_spring_zoom(optics: &mut CameraOptics, active: bool, dt_s: f32) {
    let target = if active { ZOOM_FACTOR } else { 1.0 };
    let current = optics.zoom_multiplier();
    if (current - target).abs() < 1.0e-4 {
        optics.set_zoom_multiplier(target);
        return;
    }
    let smoothing = 1.0 - (-ZOOM_LERP_RATE * dt_s).exp();
    optics.set_zoom_multiplier(current + (target - current) * smoothing);
}

pub fn speed_reference(speed_m_s: f64) -> &'static str {
    match speed_m_s {
        v if v < 2.0 => "a slow walk",
        v if v < 6.0 => "a brisk walk",
        v if v < 12.0 => "a sprint",
        v if v < 35.0 => "city traffic",
        v if v < 90.0 => "highway traffic",
        v if v < 200.0 => "a race car",
        v if v < 340.0 => "an airliner",
        v if v < 1_000.0 => "supersonic",
        v if v < 3_000.0 => "a re-entry capsule",
        v if v < 12_000.0 => "orbital velocity",
        v if v < 100_000.0 => "an interplanetary transfer",
        v if v < 3.0e6 => "a solar-wind gust",
        _ => "1 % of light speed",
    }
}

pub fn format_speed(speed_m_s: f64) -> String {
    if speed_m_s < 1_000.0 {
        format!("{speed_m_s:.0} m/s")
    } else {
        format!("{:.1} km/s", speed_m_s / 1_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_motion_uses_camera_frame() {
        let mut pose = ViewerPose {
            position: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
        };
        let mut preferences = ViewerPreferences::default();
        drive_motion(
            &mut pose,
            &mut preferences,
            ViewerIntent {
                movement: Vec3::Z,
                ..Default::default()
            },
            1.0,
            None,
        );
        assert_eq!(pose.position, DVec3::new(0.0, 0.0, -100.0));
    }

    #[test]
    fn level_lock_removes_roll() {
        let mut pose = ViewerPose {
            position: DVec3::ZERO,
            rotation: DQuat::from_rotation_z(0.7),
        };
        settle_level_lock(
            &mut pose,
            LevelLock::new(DVec3::Y, 1.0).unwrap(),
            1.0 / 60.0,
        );
        let up = pose.rotation * DVec3::Y;
        assert!(up.abs_diff_eq(DVec3::Y, 1.0e-10));
    }

    #[test]
    fn partial_level_lock_reduces_roll() {
        let rotation = DQuat::from_rotation_z(0.7);
        let mut pose = ViewerPose {
            position: DVec3::ZERO,
            rotation,
        };
        let before = (rotation * DVec3::Y).x.abs();

        settle_level_lock(
            &mut pose,
            LevelLock::new(DVec3::Y, 0.5).unwrap(),
            1.0 / 60.0,
        );

        let after = (pose.rotation * DVec3::Y).x.abs();
        assert!(after < before);
        assert!(after > 0.0);
    }

    #[test]
    fn partial_level_lock_is_frame_rate_independent() {
        let initial = ViewerPose {
            position: DVec3::ZERO,
            rotation: DQuat::from_rotation_z(0.7),
        };
        let lock = LevelLock::new(DVec3::Y, 0.5).unwrap();
        let mut one_step = initial;
        settle_level_lock(&mut one_step, lock, 1.0);

        let mut many_steps = initial;
        for _ in 0..60 {
            settle_level_lock(&mut many_steps, lock, 1.0 / 60.0);
        }

        assert!(one_step.rotation.dot(many_steps.rotation).abs() > 1.0 - 1.0e-12);
    }

    #[test]
    fn apparent_size_authority_uses_documented_angle_band() {
        let body_radius_m = 10.0;
        let full_radius_m = body_radius_m / (60.0_f64.to_radians()).sin();
        let release_radius_m = body_radius_m / (22.5_f64.to_radians()).sin();

        assert_eq!(level_lock_authority(full_radius_m, body_radius_m), 1.0);
        assert_eq!(level_lock_authority(release_radius_m, body_radius_m), 0.0);
        let middle = level_lock_authority((full_radius_m + release_radius_m) * 0.5, body_radius_m);
        assert!(middle > 0.0 && middle < 1.0);
    }

    #[test]
    fn apparent_size_authority_releases_monotonically_with_distance() {
        let authorities =
            [1.1, 1.3, 1.6, 2.0, 2.4, 2.8].map(|radius_m| level_lock_authority(radius_m, 1.0));
        assert!(authorities.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    #[test]
    fn apparent_size_authority_is_scale_invariant() {
        assert_eq!(
            level_lock_authority(2.0, 1.0),
            level_lock_authority(20.0, 10.0)
        );
    }
}
