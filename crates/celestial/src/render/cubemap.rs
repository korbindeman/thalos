//! Bake a [`Universe`] into an HDR cubemap for the realtime skybox.
//!
//! Output is a `HdrCubemap`: six square float buffers in
//! `Rgba32Float` layout, one per cube face, using the standard
//! OpenGL / Vulkan face ordering (+X, -X, +Y, -Y, +Z, -Z). The game
//! crate converts this to a `bevy::prelude::Image` without any logic
//! of its own.
//!
//! Rendering strategy for phase 2 (stars only):
//!   * Integrate each star's blackbody spectrum through three
//!     visible-light passbands → linear RGB flux.
//!   * Scale by the magnitude-derived flux factor.
//!   * Pick the dominant cube face for its direction, project to face
//!     UV, splat a small Gaussian PSF.
//!
//! Seam handling: phase 2 splats only to the dominant face. A star
//! whose PSF overlaps a face edge loses the spill. With σ ≈ 1 px this
//! is ~1 sub-pixel leakage per edge star, below perceptible. We
//! revisit when galaxies/nebulae land.

use glam::{Vec2, Vec3};
use rayon::prelude::*;

use crate::Universe;
use crate::spectrum::{Passband, Spectrum, passbands};

use super::psf::splat_gaussian;

/// Six cube face buffers in OpenGL/Vulkan face order.
///
/// Face index → basis:
///   0: +X    1: -X    2: +Y    3: -Y    4: +Z    5: -Z
pub const FACE_POS_X: usize = 0;
pub const FACE_NEG_X: usize = 1;
pub const FACE_POS_Y: usize = 2;
pub const FACE_NEG_Y: usize = 3;
pub const FACE_POS_Z: usize = 4;
pub const FACE_NEG_Z: usize = 5;

/// HDR cubemap output. Each face is `size * size` RGBA float pixels.
#[derive(Debug, Clone)]
pub struct HdrCubemap {
    pub size: usize,
    pub faces: [Vec<[f32; 4]>; 6],
}

impl HdrCubemap {
    pub fn new(size: usize) -> Self {
        let pixels = size * size;
        Self {
            size,
            faces: [
                vec![[0.0; 4]; pixels],
                vec![[0.0; 4]; pixels],
                vec![[0.0; 4]; pixels],
                vec![[0.0; 4]; pixels],
                vec![[0.0; 4]; pixels],
                vec![[0.0; 4]; pixels],
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct BakeParams {
    /// Edge length of each cube face in pixels.
    pub face_size: usize,
    /// PSF Gaussian standard deviation in pixels.
    pub star_sigma_px: f32,
    /// Flux multiplier applied to every star after band integration.
    /// Tunes visual scale; the `Skybox.brightness` uniform in Bevy
    /// then applies final exposure.
    pub flux_scale: f32,
}

impl Default for BakeParams {
    fn default() -> Self {
        Self {
            face_size: 1024,
            // Tight enough to look point-like; wide enough that
            // sub-pixel centers don't alias under bilinear filtering.
            star_sigma_px: 0.7,
            // Tuned so a Sun-analog magnitude-0 star peaks near 50 in
            // the cubemap. Skybox.brightness then scales to final HDR.
            flux_scale: 150.0,
        }
    }
}

/// Map a 3D direction to a cube face index and face-local UV in
/// [0, 1]². UV is origin-at-top-left, matching standard cubemap image
/// conventions.
fn direction_to_face(dir: Vec3) -> (usize, Vec2) {
    let ax = dir.x.abs();
    let ay = dir.y.abs();
    let az = dir.z.abs();

    // Largest-component axis wins. Ties broken by axis order (X, Y, Z).
    let (face, sc, tc, ma) = if ax >= ay && ax >= az {
        if dir.x > 0.0 {
            (FACE_POS_X, -dir.z, -dir.y, ax)
        } else {
            (FACE_NEG_X, dir.z, -dir.y, ax)
        }
    } else if ay >= ax && ay >= az {
        if dir.y > 0.0 {
            (FACE_POS_Y, dir.x, dir.z, ay)
        } else {
            (FACE_NEG_Y, dir.x, -dir.z, ay)
        }
    } else if dir.z > 0.0 {
        (FACE_POS_Z, dir.x, -dir.y, az)
    } else {
        (FACE_NEG_Z, -dir.x, -dir.y, az)
    };

    // sc, tc ∈ [-ma, ma] → u, v ∈ [0, 1]
    let u = 0.5 * (sc / ma + 1.0);
    let v = 0.5 * (tc / ma + 1.0);
    (face, Vec2::new(u, v))
}

/// Bake the entire universe into an HDR cubemap.
///
/// Runs parallel across faces. Each face iterates the full star list
/// and keeps stars whose dominant face matches. Per-star work is
/// tiny; the wasted classification on non-matching stars is cheaper
/// than any sharing scheme for our target counts.
pub fn bake_skybox(universe: &Universe, params: &BakeParams) -> HdrCubemap {
    let [red, green, blue] = passbands::visible_rgb();
    let bands = [red, green, blue];

    // Photometric zero point: a Sun-like blackbody sets the overall
    // brightness scale so that a magnitude-0 Sun-analog star produces
    // peak RGB near 1.0. Real astronomy anchors to Vega (~9600 K);
    // using the Sun is the same idea, adjusted to our taste, and it
    // keeps the brightest-star constant independent of PSF radius and
    // face resolution.
    let reference = Spectrum::Blackbody { temperature_k: 5778.0, scale: 1.0 };
    let reference_flux = bands[1].integrate(&reference).max(1e-30);

    let mut out = HdrCubemap::new(params.face_size);

    // Precompute per-star projected data once to avoid redoing the
    // band integrals six times.
    let projected: Vec<ProjectedStar> = universe
        .stars
        .par_iter()
        .map(|s| project_star(s, &bands, reference_flux, params))
        .collect();

    out.faces
        .par_iter_mut()
        .enumerate()
        .for_each(|(face_idx, buffer)| {
            for p in &projected {
                if p.face != face_idx {
                    continue;
                }
                let center = Vec2::new(
                    p.face_uv.x * params.face_size as f32,
                    p.face_uv.y * params.face_size as f32,
                );
                splat_gaussian(
                    buffer,
                    params.face_size,
                    params.face_size,
                    center,
                    params.star_sigma_px,
                    p.flux_rgb,
                );
            }
            // Alpha = 1 for anything touched so the image format is
            // well-formed. We write 1 uniformly at the end.
            for pix in buffer.iter_mut() {
                pix[3] = 1.0;
            }
        });

    out
}

struct ProjectedStar {
    face: usize,
    face_uv: Vec2,
    flux_rgb: [f32; 3],
}

fn project_star(
    star: &crate::sources::Star,
    bands: &[Passband; 3],
    reference_flux: f32,
    params: &BakeParams,
) -> ProjectedStar {
    let (face, face_uv) = direction_to_face(star.position);
    // Band integrals divided by the Sun-reference green integral
    // produce unit-scale values (~1 for a Sun-like star). Then the
    // apparent-magnitude factor gives absolute brightness. Cooler
    // stars have smaller blue integral, so the red/green/blue ratios
    // naturally reflect temperature — no explicit hue normalisation.
    let mag_flux = star.magnitude_flux();
    let k = mag_flux * params.flux_scale / reference_flux;
    let flux_rgb = [
        bands[0].integrate(&star.spectrum) * k,
        bands[1].integrate(&star.spectrum) * k,
        bands[2].integrate(&star.spectrum) * k,
    ];
    ProjectedStar { face, face_uv, flux_rgb }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{DefaultGenParams, generate_default};

    #[test]
    fn direction_to_face_hits_expected_faces() {
        let (f, _) = direction_to_face(Vec3::X);
        assert_eq!(f, FACE_POS_X);
        let (f, _) = direction_to_face(Vec3::NEG_Y);
        assert_eq!(f, FACE_NEG_Y);
        let (f, _) = direction_to_face(Vec3::Z);
        assert_eq!(f, FACE_POS_Z);
    }

    #[test]
    fn bake_runs_and_writes_nonzero_pixels() {
        let universe = generate_default(&DefaultGenParams {
            seed: 7,
            star_count: 2_000,
            faint_magnitude_limit: 8.0,
            galaxy_count: 0,
            galaxy_faint_magnitude_limit: 12.0,
        });
        let cubemap = bake_skybox(
            &universe,
            &BakeParams { face_size: 128, ..Default::default() },
        );
        let total_lum: f32 = cubemap
            .faces
            .iter()
            .flat_map(|f| f.iter())
            .map(|p| p[0] + p[1] + p[2])
            .sum();
        assert!(total_lum > 0.0, "baked cubemap has no flux");
    }
}
