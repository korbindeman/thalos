//! Shared local-tangent-frame (ENU) geometry for HUD displays.
//!
//! The PFD attitude ladder ([`super::pfd_panel`]) and the MFD navigation
//! display ([`super::mfd::widgets::nav_display`]) both express craft
//! attitude / surface bearings in the local east-north-up frame at the
//! craft, anchored to the dominant body. Sharing one basis construction
//! keeps their headings consistent by definition.

use bevy::math::DVec3;

/// Local ENU basis at `craft_pos` relative to `body_pos`:
/// `up` = radial-out from the body, `north` = world-Y projected onto the
/// tangent plane (X-axis fallback at the poles), `east` = `north × up`.
///
/// Returns `None` only when the craft sits exactly at the body centre
/// (degenerate up).
pub(crate) fn local_enu_basis(
    craft_pos: DVec3,
    body_pos: DVec3,
) -> Option<(DVec3, DVec3, DVec3)> {
    let up = (craft_pos - body_pos).try_normalize()?;
    let mut north = DVec3::Y - DVec3::Y.dot(up) * up;
    if north.length_squared() < 1e-12 {
        north = DVec3::X - DVec3::X.dot(up) * up;
    }
    let north = north.try_normalize()?;
    let east = north.cross(up);
    Some((up, north, east))
}
