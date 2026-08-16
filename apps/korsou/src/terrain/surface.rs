fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn smooth_fraction(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn hash_unit(x: i32, z: i32, seed: u32) -> f32 {
    let mut value = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (z as u32).wrapping_mul(0x85EB_CA77)
        ^ seed.wrapping_mul(0xC2B2_AE3D);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32 * 2.0 - 1.0
}

fn value_noise(x: f64, z: f64, wavelength: f64, seed: u32) -> f32 {
    let x = x / wavelength;
    let z = z / wavelength;
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = smooth_fraction((x - x.floor()) as f32);
    let tz = smooth_fraction((z - z.floor()) as f32);
    let a = hash_unit(x0, z0, seed);
    let b = hash_unit(x0 + 1, z0, seed);
    let c = hash_unit(x0, z0 + 1, seed);
    let d = hash_unit(x0 + 1, z0 + 1, seed);
    let lower = a + (b - a) * tx;
    let upper = c + (d - c) * tx;
    lower + (upper - lower) * tz
}

/// Canonical visual canopy coverage used by both terrain color and plant scatter.
///
/// This is deliberately a lightweight visual field, not a simulated biome. Keeping
/// one source of truth lets nearby geometry hand off to distant terrain color.
pub(crate) fn canopy_coverage(x: f64, z: f64, height: f32, shore_distance: f32, slope: f32) -> f32 {
    if height <= 0.0 || shore_distance <= 95.0 {
        return 0.0;
    }

    let broad = value_noise(x, z, 560.0, 137) * 0.5 + 0.5;
    let fine = value_noise(x, z, 145.0, 173) * 0.5 + 0.5;
    let glades = value_noise(x, z, 68.0, 211) * 0.5 + 0.5;
    let habitat = broad * 0.72 + fine * 0.28;
    let stands = smoothstep(0.31, 0.66, habitat);
    let glade_weight = 0.58 + smoothstep(0.25, 0.78, glades) * 0.42;
    let slope_weight = 1.0 - smoothstep(0.34, 0.72, slope);
    let shore_weight = smoothstep(95.0, 125.0, shore_distance);

    (stands * glade_weight * slope_weight * shore_weight).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canopy_never_colors_beaches_or_sea() {
        assert_eq!(canopy_coverage(0.0, 0.0, -1.0, 500.0, 0.0), 0.0);
        assert_eq!(canopy_coverage(0.0, 0.0, 12.0, 70.0, 0.0), 0.0);
    }

    #[test]
    fn canopy_coverage_stays_normalized() {
        for z in -10..=10 {
            for x in -10..=10 {
                let coverage = canopy_coverage(
                    f64::from(x) * 100.0,
                    f64::from(z) * 100.0,
                    40.0,
                    500.0,
                    0.12,
                );
                assert!((0.0..=1.0).contains(&coverage));
            }
        }
    }
}
