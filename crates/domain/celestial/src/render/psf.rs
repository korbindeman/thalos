//! Gaussian point-spread function used for star splatting.
//!
//! Cheap analytical kernel evaluated per-pixel inside a small bounding
//! box. Intentionally simple — a real telescope renderer will want
//! optics-derived Airy disks later, but for a display skybox a
//! Gaussian is indistinguishable below a pixel.

use glam::Vec2;

/// Splat a Gaussian at sub-pixel position `center` with standard
/// deviation `sigma_px` into a buffer.
///
/// The Gaussian is truncated at `3 · sigma` to bound work. `flux_rgb`
/// is multiplied into the kernel so the total integrated flux over
/// the buffer equals `flux_rgb` (assuming the kernel lies fully
/// inside).
pub fn splat_gaussian(
    buffer: &mut [[f32; 4]],
    width: usize,
    height: usize,
    center: Vec2,
    sigma_px: f32,
    flux_rgb: [f32; 3],
) {
    if sigma_px <= 0.0 {
        return;
    }
    let radius = (3.0 * sigma_px).ceil() as i32;
    let cx = center.x;
    let cy = center.y;
    let x0 = (cx.floor() as i32 - radius).max(0);
    let y0 = (cy.floor() as i32 - radius).max(0);
    let x1 = (cx.floor() as i32 + radius).min(width as i32 - 1);
    let y1 = (cy.floor() as i32 + radius).min(height as i32 - 1);
    if x1 < x0 || y1 < y0 {
        return;
    }
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma_px * sigma_px);
    // Normalise so the analytic integral ≈ 1; we then multiply by the
    // flux. This is the continuous normalisation; for small radii the
    // discrete sum is close but not exact. Good enough for a skybox.
    let norm = 1.0 / (std::f32::consts::TAU * sigma_px * sigma_px);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let r2 = dx * dx + dy * dy;
            let g = norm * (-r2 * inv_two_sigma_sq).exp();
            let idx = (y as usize) * width + (x as usize);
            buffer[idx][0] += flux_rgb[0] * g;
            buffer[idx][1] += flux_rgb[1] * g;
            buffer[idx][2] += flux_rgb[2] * g;
        }
    }
}
