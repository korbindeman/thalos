//! Cubemap warp advection + final cloud-density bake.
//!
//! Pipeline (matches Wedekind's 4-pass prototype, collapsed to CPU):
//!
//! 1. Build identity warp: each cubemap texel's initial warp vector is
//!    its own home direction on the unit sphere.
//! 2. For `num_iterations` steps:
//!      for each texel:
//!        prev  = normalise(current_warp_at_texel)
//!        next  = prev + step · curl_on_sphere(prev)
//!      ping-pong into a fresh cubemap.
//! 3. Final pass: sample the cover-Worley fBm at `normalise(warp)` for
//!    every texel → `Cubemap<f32>` density in `[0, 1]` (scaled by the
//!    per-octave weights, no coverage threshold applied here — the
//!    shader side does that).
//!
//! The warp field itself is stored as three `Cubemap<f32>` (x/y/z
//! components) so the existing `terrain_gen::Cubemap::sample_bilinear`
//! can do the per-iteration lookup without new generic machinery.
//! Storing as three f32 cubes is also how the GPU path would ship (a
//! single RGB16F cube), so the data layout is portable.

use glam::Vec3;
use rayon::prelude::*;
use thalos_terrain_gen::Cubemap;
use thalos_terrain_gen::cubemap::{CubemapFace, face_uv_to_dir};

use crate::flow::{FlowConfig, curl_on_sphere};
use crate::worley::{WorleyVolume, worley_fbm};

/// Full configuration for a cloud-cover bake.
pub struct CloudBakeConfig {
    /// Cubemap face resolution. `size²` texels per face; `6·size²` total.
    /// Wedekind prototype: 512. Baked CPU side so lower is cheaper.
    pub size: u32,
    /// Flow-field config (potential + both Worley volumes).
    pub flow: FlowConfig,
    /// Pre-baked Worley volume for the cover (density) field.
    pub cover_volume: WorleyVolume,
    /// Per-octave weights for the cover fBm. Wedekind default:
    /// `[0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625]` (6 octaves).
    pub cover_octaves: Vec<f32>,
    /// Rescale applied to sample direction before the cover fBm lookup.
    /// Wedekind default: 2.0 (so `idx = dir * 0.5`).
    pub cover_scale: f32,
    /// Number of Lagrangian advection steps. Wedekind: 50.
    pub num_iterations: u32,
    /// Magnitude of each advection step, in sphere arc-length units.
    /// Wedekind: `1.5e-3 × curl_scale` (≈ 0.006 rad at curl_scale=4).
    /// Total accumulated displacement is roughly
    /// `num_iterations · flow_step`, which sets the dominant spiral
    /// size: `50 · 0.006 ≈ 0.3 rad ≈ 17°`.
    pub flow_step: f32,
    /// Finite-difference epsilon for the gradient of the flow potential.
    /// Wedekind: `1 / worley_size / 2^num_flow_octaves`.
    pub gradient_eps: f32,
}

impl CloudBakeConfig {
    /// Produce a Wedekind-default config keyed off a 64-bit body seed.
    /// Three distinct Worley volumes are baked (north, south, cover)
    /// with deterministic sub-seeds so the same body seed always
    /// produces the same cloud pattern.
    ///
    /// `face_size` is the cubemap face resolution; the Worley volumes
    /// themselves are baked at a fixed internal size of 64³ — this is
    /// Wedekind's `worley-size` default and is what the `gradient_eps`
    /// value assumes.
    pub fn wedekind_defaults(seed: u64, face_size: u32) -> Self {
        let seed_lo = seed as u32;
        let seed_hi = (seed >> 32) as u32;
        // Sub-seeds: deterministic splits from the body seed so the
        // three volumes are independent but reproducible.
        let seed_n = seed_lo.wrapping_add(0xA1C3_7F19);
        let seed_s = seed_hi.wrapping_add(0x4B9D_2C51);
        let seed_c = seed_lo ^ seed_hi.wrapping_mul(0xD37A_B602);

        let worley_size = 64u32;
        let worley_cells = 8u32;
        let curl_scale = 4.0_f32;

        let flow = FlowConfig {
            curl_scale,
            prevailing: 0.1,
            whirl: 1.0,
            flow_octaves: vec![0.5, 0.25, 0.125],
            volume_north: WorleyVolume::bake(worley_size, worley_cells, seed_n),
            volume_south: WorleyVolume::bake(worley_size, worley_cells, seed_s),
        };

        Self {
            size: face_size,
            flow,
            cover_volume: WorleyVolume::bake(worley_size, worley_cells, seed_c),
            cover_octaves: vec![0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625],
            cover_scale: 2.0,
            num_iterations: 50,
            flow_step: 1.5e-3 * curl_scale,
            gradient_eps: 1.0 / worley_size as f32 / 8.0, // 2^3 for 3 flow octaves
        }
    }
}

// ---------------------------------------------------------------------------
// Warp field advection (cubemap iteration)
// ---------------------------------------------------------------------------

/// Sample the x/y/z component cubemaps bilinearly at direction `dir`
/// and assemble the Vec3. Matches Wedekind's
/// `interpolate_vector_cubemap`.
fn sample_warp(
    warp_x: &Cubemap<f32>,
    warp_y: &Cubemap<f32>,
    warp_z: &Cubemap<f32>,
    dir: Vec3,
) -> Vec3 {
    Vec3::new(
        warp_x.sample_bilinear(dir),
        warp_y.sample_bilinear(dir),
        warp_z.sample_bilinear(dir),
    )
}

/// Identity warp: every texel's warp value equals the texel's own
/// home direction on the unit sphere. First iteration input.
fn identity_warp(size: u32) -> (Cubemap<f32>, Cubemap<f32>, Cubemap<f32>) {
    let mut wx = Cubemap::<f32>::new(size);
    let mut wy = Cubemap::<f32>::new(size);
    let mut wz = Cubemap::<f32>::new(size);
    let inv = 1.0 / size as f32;
    for face in CubemapFace::ALL {
        // Process rows in parallel — each row is independent.
        let rows: Vec<(u32, Vec<Vec3>)> = (0..size)
            .into_par_iter()
            .map(|y| {
                let v = (y as f32 + 0.5) * inv;
                let row: Vec<Vec3> = (0..size)
                    .map(|x| {
                        let u = (x as f32 + 0.5) * inv;
                        face_uv_to_dir(face, u, v)
                    })
                    .collect();
                (y, row)
            })
            .collect();
        for (y, row) in rows {
            for (x, dir) in row.into_iter().enumerate() {
                wx.set(face, x as u32, y, dir.x);
                wy.set(face, x as u32, y, dir.y);
                wz.set(face, x as u32, y, dir.z);
            }
        }
    }
    (wx, wy, wz)
}

/// One advection step of the warp cubemap. For each texel's home
/// direction `v`:
///   prev = normalise(current_warp_at_v)
///   next = prev + step · curl(prev)
fn advect_warp(
    size: u32,
    flow_step: f32,
    gradient_eps: f32,
    cfg: &FlowConfig,
    wx: &Cubemap<f32>,
    wy: &Cubemap<f32>,
    wz: &Cubemap<f32>,
) -> (Cubemap<f32>, Cubemap<f32>, Cubemap<f32>) {
    let mut nx = Cubemap::<f32>::new(size);
    let mut ny = Cubemap::<f32>::new(size);
    let mut nz = Cubemap::<f32>::new(size);
    let inv = 1.0 / size as f32;

    for face in CubemapFace::ALL {
        // Parallelise over rows; each row writes its own segment of the
        // output buffers, so no contention.
        let rows: Vec<(u32, Vec<Vec3>)> = (0..size)
            .into_par_iter()
            .map(|y| {
                let v = (y as f32 + 0.5) * inv;
                let row: Vec<Vec3> = (0..size)
                    .map(|x| {
                        let u = (x as f32 + 0.5) * inv;
                        let home_dir = face_uv_to_dir(face, u, v);
                        let current = sample_warp(wx, wy, wz, home_dir);
                        // `current` may have drifted off the sphere over
                        // previous iterations; re-project before the
                        // curl evaluation (Wedekind does the same).
                        let prev = safe_normalize(current, home_dir);
                        let step = curl_on_sphere(prev, gradient_eps, cfg) * flow_step;
                        prev + step
                    })
                    .collect();
                (y, row)
            })
            .collect();
        for (y, row) in rows {
            for (x, v) in row.into_iter().enumerate() {
                nx.set(face, x as u32, y, v.x);
                ny.set(face, x as u32, y, v.y);
                nz.set(face, x as u32, y, v.z);
            }
        }
    }
    (nx, ny, nz)
}

/// Robust normalisation: fall back to `default` if `v` is near zero or
/// non-finite. The warp field starts on the unit sphere and curl steps
/// are small, so this fallback should never trip — but guarding it
/// costs one compare and protects the bake from a rogue NaN.
fn safe_normalize(v: Vec3, default: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > 1e-12 {
        v / len_sq.sqrt()
    } else {
        default
    }
}

// ---------------------------------------------------------------------------
// Top-level bake entry point
// ---------------------------------------------------------------------------

/// Bake a cloud-cover cubemap from a complete config.
///
/// Runs the full Wedekind pipeline: identity → N iterations of warp
/// advection → final Worley fBm lookup at the advected direction.
/// Returns a `Cubemap<f32>` whose values are the raw weighted Worley
/// fBm sum (not yet thresholded — that happens shader-side via
/// `threshold` / `multiplier`, so authors can tune coverage without a
/// re-bake).
pub fn bake_cloud_cover(cfg: &CloudBakeConfig) -> Cubemap<f32> {
    let (mut wx, mut wy, mut wz) = identity_warp(cfg.size);

    for _ in 0..cfg.num_iterations {
        let (nx, ny, nz) = advect_warp(
            cfg.size,
            cfg.flow_step,
            cfg.gradient_eps,
            &cfg.flow,
            &wx,
            &wy,
            &wz,
        );
        wx = nx;
        wy = ny;
        wz = nz;
    }

    // Final pass: at every texel, normalise the accumulated warp vector
    // back to the unit sphere and sample the cover fBm there.
    let mut cover = Cubemap::<f32>::new(cfg.size);
    let inv_scale = 1.0 / cfg.cover_scale;
    let inv = 1.0 / cfg.size as f32;
    for face in CubemapFace::ALL {
        let rows: Vec<(u32, Vec<f32>)> = (0..cfg.size)
            .into_par_iter()
            .map(|y| {
                let v = (y as f32 + 0.5) * inv;
                let row: Vec<f32> = (0..cfg.size)
                    .map(|x| {
                        let u = (x as f32 + 0.5) * inv;
                        let home = face_uv_to_dir(face, u, v);
                        let w = sample_warp(&wx, &wy, &wz, home);
                        let d = safe_normalize(w, home);
                        worley_fbm(&cfg.cover_volume, d * inv_scale, &cfg.cover_octaves)
                    })
                    .collect();
                (y, row)
            })
            .collect();
        for (y, row) in rows {
            for (x, v) in row.into_iter().enumerate() {
                cover.set(face, x as u32, y, v);
            }
        }
    }
    cover
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_iterations_is_identity_warp_density() {
        // With 0 iterations, the warp is identity, so the cover at a
        // direction equals the cover Worley fBm at that same direction —
        // a sanity check that the pipeline plumbing is wired end-to-end.
        let mut cfg = CloudBakeConfig::wedekind_defaults(0xDEAD_BEEF, 16);
        cfg.num_iterations = 0;
        let cube = bake_cloud_cover(&cfg);
        // Spot check one texel.
        let v = (0.5_f32) / 16.0;
        let dir = face_uv_to_dir(CubemapFace::PosX, v, v);
        let expected = worley_fbm(&cfg.cover_volume, dir / cfg.cover_scale, &cfg.cover_octaves);
        let actual = cube.get(CubemapFace::PosX, 0, 0);
        assert!(
            (actual - expected).abs() < 1e-5,
            "zero-iter cube mismatch: got {}, expected {}",
            actual,
            expected,
        );
    }

    #[test]
    fn bake_is_deterministic() {
        let a = bake_cloud_cover(&CloudBakeConfig {
            num_iterations: 5,
            ..CloudBakeConfig::wedekind_defaults(0xCAFEBABE, 16)
        });
        let b = bake_cloud_cover(&CloudBakeConfig {
            num_iterations: 5,
            ..CloudBakeConfig::wedekind_defaults(0xCAFEBABE, 16)
        });
        for face in CubemapFace::ALL {
            let da = a.face_data(face);
            let db = b.face_data(face);
            assert_eq!(da, db, "non-deterministic on face {:?}", face);
        }
    }

    #[test]
    fn bake_produces_finite_values() {
        let cfg = CloudBakeConfig {
            num_iterations: 3,
            ..CloudBakeConfig::wedekind_defaults(0x1234_5678, 16)
        };
        let cube = bake_cloud_cover(&cfg);
        for face in CubemapFace::ALL {
            for &v in cube.face_data(face) {
                assert!(v.is_finite(), "non-finite density on face {:?}: {}", face, v);
                assert!(v >= 0.0, "negative density on face {:?}: {}", face, v);
                assert!(v <= 2.0, "unreasonable density on face {:?}: {}", face, v);
            }
        }
    }

    #[test]
    fn warp_accumulates_over_iterations() {
        // After 50 iterations with the default flow_step, the warp at a
        // generic direction should deviate noticeably from identity.
        let cfg = CloudBakeConfig::wedekind_defaults(0xABCD_EF00, 16);
        // Run the advection loop and introspect the final warp, not the
        // baked density — we want to measure the accumulated displacement.
        let (mut wx, mut wy, mut wz) = identity_warp(cfg.size);
        for _ in 0..cfg.num_iterations {
            let (nx, ny, nz) = advect_warp(
                cfg.size,
                cfg.flow_step,
                cfg.gradient_eps,
                &cfg.flow,
                &wx,
                &wy,
                &wz,
            );
            wx = nx;
            wy = ny;
            wz = nz;
        }
        // Measure the average displacement from identity across all
        // cubemap texels. Wedekind's 50 × 1.5e-3 × 4 = 0.3 suggests an
        // average magnitude on that order.
        let mut total = 0.0_f64;
        let mut count = 0u32;
        for face in CubemapFace::ALL {
            for y in 0..cfg.size {
                for x in 0..cfg.size {
                    let u = (x as f32 + 0.5) / cfg.size as f32;
                    let v = (y as f32 + 0.5) / cfg.size as f32;
                    let home = face_uv_to_dir(face, u, v);
                    let warped = Vec3::new(
                        wx.get(face, x, y),
                        wy.get(face, x, y),
                        wz.get(face, x, y),
                    );
                    let normalised = safe_normalize(warped, home);
                    total += (normalised - home).length() as f64;
                    count += 1;
                }
            }
        }
        let mean = total / count as f64;
        assert!(
            mean > 0.05,
            "warp didn't accumulate enough displacement: mean {} (expected > 0.05)",
            mean,
        );
        assert!(
            mean < 1.0,
            "warp blew up: mean displacement {} (expected < 1.0)",
            mean,
        );
    }
}
