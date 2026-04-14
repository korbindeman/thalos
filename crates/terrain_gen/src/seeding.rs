//! Hierarchical sub-seeding and a tiny deterministic PRNG.
//!
//! Per spec §Design principles 4: one top-level seed derives per-stage
//! sub-seeds by hashing with a stable stage name, so re-running one stage
//! must not disturb any other stage's randomness.
//!
//! Neither the hash nor the PRNG pulls a dependency — determinism across
//! toolchains and platforms is part of the contract, so we can't rely on
//! `std::hash::Hasher` (whose implementations are allowed to change).  Both
//! primitives below are deliberately simple and byte-stable.

/// FNV-1a 64-bit hash of a byte slice.  Deterministic across platforms.
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Derive a deterministic per-stage sub-seed from the top-level seed and a
/// stable stage identifier.  Two distinct stage names must always produce
/// distinct sub-seeds for the same top seed — mixing in 8 bytes of `top`
/// after the name rather than XORing avoids the trivial collision where a
/// name-derived constant lines up with the top bits.
pub fn sub_seed(top: u64, stage_name: &str) -> u64 {
    let mut buf: Vec<u8> = Vec::with_capacity(stage_name.len() + 9);
    buf.extend_from_slice(stage_name.as_bytes());
    buf.push(0); // domain separator
    buf.extend_from_slice(&top.to_le_bytes());
    fnv1a(&buf)
}

/// SplitMix64 — one step of Vigna's splitmix64.  Used as a finalizer and
/// as the seed expansion for `Rng`.  Tiny, fast, and passes the usual
/// statistical tests well enough for a procedural generator.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic 64-bit LCG-style PRNG seeded via splitmix64.  Good enough
/// for procedural placement and perturbation; not cryptographic.
///
/// Same seed → identical sequence, across machines and runs.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state trap by running the seed through splitmix.
        Self {
            state: splitmix64(seed ^ 0xA5A5_A5A5_5A5A_5A5A),
        }
    }

    /// Next uniformly-distributed u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    /// Uniform f64 in `[0, 1)`.  Uses the top 53 bits so the mantissa is
    /// filled uniformly.
    pub fn next_f64(&mut self) -> f64 {
        // 2^53 = 9007199254740992
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in `[-1, 1)`.
    pub fn next_f64_signed(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }

    /// Uniform f64 in `[lo, hi)`.
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniformly random unit vector on the sphere (Marsaglia's method,
    /// branch-free enough for a small PRNG budget).
    pub fn unit_vector(&mut self) -> glam::DVec3 {
        // z ∈ [-1, 1], φ ∈ [0, 2π)
        let z = self.next_f64_signed();
        let phi = self.next_f64() * std::f64::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let (sp, cp) = phi.sin_cos();
        glam::DVec3::new(r * cp, r * sp, z)
    }

    /// Power-law sample in `[d_min, d_max]` with cumulative exponent
    /// `alpha > 0`, i.e. `N(>D) ∝ D^-alpha`.  Inverse-CDF transform.
    pub fn power_law(&mut self, d_min: f64, d_max: f64, alpha: f64) -> f64 {
        let u = self.next_f64();
        let lo = d_min.powf(-alpha);
        let hi = d_max.powf(-alpha);
        let y = lo + (hi - lo) * u; // in [hi, lo] since hi < lo for alpha>0
        y.powf(-1.0 / alpha)
    }

    /// Two-slope (broken) power-law sample in `[d_min, d_max]` with a slope
    /// break at `d_break`.  Above the break, cumulative exponent is
    /// `alpha_large`; below it, `alpha_small`.  Used for lunar SFDs where
    /// the small-crater tail is steeper than the large-crater slope
    /// (Neukum Production Function piecewise approximation).
    ///
    /// The PDFs in the two branches are joined so there is no density jump
    /// at `d_break`, then normalized, then sampled by inverse CDF.
    pub fn broken_power_law(
        &mut self,
        d_min: f64,
        d_break: f64,
        d_max: f64,
        alpha_small: f64,
        alpha_large: f64,
    ) -> f64 {
        debug_assert!(d_min < d_break && d_break < d_max);
        // Unnormalized cumulative mass in each branch (integral of PDF ∝ D^(-alpha-1))
        let mass_small = d_min.powf(-alpha_small) - d_break.powf(-alpha_small);
        // Large-branch PDF is scaled by k so it matches small-branch PDF at d_break:
        // alpha_small × D^(-alpha_small-1) = k × alpha_large × D^(-alpha_large-1) at D=d_break
        let k = (alpha_small / alpha_large) * d_break.powf(alpha_large - alpha_small);
        let mass_large = k * (d_break.powf(-alpha_large) - d_max.powf(-alpha_large));
        let total = mass_small + mass_large;
        let u = self.next_f64();
        if u * total < mass_small {
            // Sample small branch
            let u2 = (u * total) / mass_small;
            let lo = d_min.powf(-alpha_small);
            let hi = d_break.powf(-alpha_small);
            let y = lo + (hi - lo) * u2;
            y.powf(-1.0 / alpha_small)
        } else {
            // Sample large branch
            let u2 = (u * total - mass_small) / mass_large;
            let lo = d_break.powf(-alpha_large);
            let hi = d_max.powf(-alpha_large);
            let y = lo + (hi - lo) * u2;
            y.powf(-1.0 / alpha_large)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_seed_is_deterministic() {
        assert_eq!(sub_seed(42, "stage_a"), sub_seed(42, "stage_a"));
    }

    #[test]
    fn sub_seed_differs_per_stage_and_top() {
        let a1 = sub_seed(1, "stage_a");
        let b1 = sub_seed(1, "stage_b");
        let a2 = sub_seed(2, "stage_a");
        assert_ne!(a1, b1);
        assert_ne!(a1, a2);
    }

    #[test]
    fn rng_is_deterministic() {
        let mut r1 = Rng::new(123);
        let mut r2 = Rng::new(123);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn different_seeds_produce_different_streams() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let av: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let bv: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_ne!(av, bv);
    }

    #[test]
    fn next_f64_in_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x), "f64 out of range: {x}");
        }
    }

    #[test]
    fn unit_vector_is_unit_length() {
        let mut r = Rng::new(9);
        for _ in 0..1000 {
            let v = r.unit_vector();
            assert!((v.length() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn power_law_stays_within_bounds() {
        let mut r = Rng::new(13);
        for _ in 0..10_000 {
            let d = r.power_law(1.0, 100.0, 2.0);
            assert!(d >= 1.0 && d <= 100.0 + 1e-9, "out of range: {d}");
        }
    }
}
