//! Celestial coordinate helpers.
//!
//! Thalos's celestial sphere is a game-world absolute frame, decoupled
//! from any specific body. We use right ascension / declination as the
//! angular convention because it matches astronomical tooling players
//! will recognise once the telescope layer exists.

use glam::{DVec3, Vec3};

pub type UnitVector3 = Vec3;

/// Right ascension / declination pair in radians.
///
/// `ra` ∈ [0, 2π), `dec` ∈ [-π/2, π/2].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquatorialCoord {
    pub ra: f32,
    pub dec: f32,
}

impl EquatorialCoord {
    pub fn new(ra: f32, dec: f32) -> Self {
        Self { ra, dec }
    }

    /// Convert to a unit direction vector in the celestial frame.
    ///
    /// Convention: +X at (ra=0, dec=0), +Z along the north pole,
    /// +Y at (ra=π/2, dec=0). Matches a standard right-handed frame
    /// with Z up.
    pub fn to_unit(self) -> UnitVector3 {
        let (sin_ra, cos_ra) = self.ra.sin_cos();
        let (sin_dec, cos_dec) = self.dec.sin_cos();
        Vec3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec)
    }

    pub fn to_unit_f64(self) -> DVec3 {
        let ra = self.ra as f64;
        let dec = self.dec as f64;
        let (sin_ra, cos_ra) = ra.sin_cos();
        let (sin_dec, cos_dec) = dec.sin_cos();
        DVec3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec)
    }
}

/// Uniform sample on the unit sphere from two uniforms in [0, 1).
pub fn uniform_sphere(u1: f32, u2: f32) -> UnitVector3 {
    let z = 1.0 - 2.0 * u1;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = std::f32::consts::TAU * u2;
    let (sin_phi, cos_phi) = phi.sin_cos();
    Vec3::new(r * cos_phi, r * sin_phi, z)
}

/// Galactic plane pole in the celestial frame. Arbitrary for Thalos —
/// chosen so the milky way band runs across a visually interesting
/// direction rather than aligning with the equator.
pub fn galactic_pole() -> UnitVector3 {
    // ~ equivalent to ICRS galactic pole (ra=192.86°, dec=27.13°), kept
    // in-universe as a stylistic choice rather than any claim about
    // Thalos's actual home galaxy.
    EquatorialCoord::new(192.859_5f32.to_radians(), 27.128_2f32.to_radians())
        .to_unit()
        .normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn equatorial_origin_is_plus_x() {
        let v = EquatorialCoord::new(0.0, 0.0).to_unit();
        assert_relative_eq!(v.x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(v.y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(v.z, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn north_pole_is_plus_z() {
        let v = EquatorialCoord::new(0.0, std::f32::consts::FRAC_PI_2).to_unit();
        assert_relative_eq!(v.z, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn uniform_sphere_is_unit_length() {
        for i in 0..100 {
            let u1 = (i as f32 + 0.5) / 100.0;
            let u2 = ((i * 37) % 100) as f32 / 100.0;
            let v = uniform_sphere(u1, u2);
            assert_relative_eq!(v.length(), 1.0, epsilon = 1e-6);
        }
    }
}
