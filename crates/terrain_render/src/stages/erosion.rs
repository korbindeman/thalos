//! Multi-octave gully erosion via Rune Skovbo Johansen's filter.
//!
//! Carves slope-masked gully patterns into the height cubemap. Designed
//! for the "fluvially eroded" aesthetic on atmospheric bodies — Vaelen
//! foremost (its `ColdDesertFormerlyWet` archetype is the filter's
//! intended target).
//!
//! The filter is 2D, but cubemap face UVs are discontinuous across face
//! seams, so a naïve per-face evaluation produces visible cell-pattern
//! discontinuities at the cubemap edges. To collapse those seams we
//! evaluate the filter in **octahedral coordinates**: a bijective
//! sphere↔[-1,1]² parametrization that's continuous on the upper
//! hemisphere and the equator. The cubemap's twelve face seams (~6πR
//! of seam line total) are replaced by two half-great-circles in the
//! lower hemisphere where the encoding sign-flips at the meridians
//! `x=0` and `y=0` (~2πR of seam line, all below the equator).
//!
//! Net effect for Vaelen: the upper hemisphere is fully seam-free, the
//! lower hemisphere has two crossed meridians of 1-pixel artifacts. If
//! the body has a "hero side" the user normally looks at, place that
//! side in the upper hemisphere (i.e. use a body axis where +Z points
//! at the hero side).
//!
//! Bake/render parity: the same `octahedron_encode` / `octahedron_decode`
//! pair must be used in any future render-time erosion pass so the gully
//! pattern matches across the LOD handoff.
//!
//! Slope threshold note: the filter's upstream `onset` was authored for
//! unit-less heightfields where typical slopes are O(0.5..1). On
//! physical (meters/meters) heightfields, only very steep terrain
//! triggers the gully mask at upstream defaults. The `slope_scale`
//! parameter divides all four onset components uniformly: 1.0 keeps
//! upstream behaviour (gullies on ≳22° slopes), 0.3 spreads gullies
//! onto gentler mountain flanks (≳7° slopes). Fluvial physics says
//! gullies *do* concentrate on steep slopes, so use this as a knob
//! for visual presence, not a correction.

use bevy_erosion_filter::cpu::{ErosionFilterParams, erosion_filter};
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{Cubemap, CubemapFace, face_uv_to_dir};
use crate::stage::Stage;
use crate::surface_field::cube_face_texel_scale_m;

/// Erosion stage parameters. Mirrors the tuneable subset of
/// `bevy_erosion_filter::cpu::ErosionFilterParams`, re-exposed here so
/// it can `Deserialize` from RON without leaking the upstream type into
/// the manifest schema. Non-listed knobs (lacunarity, gain, cell_scale,
/// rounding, …) inherit Shadertoy reference values.
#[derive(Debug, Clone, Deserialize)]
pub struct Erosion {
    /// Horizontal scale of the largest gully cluster, in meters.
    /// Try `mountain_width / 5..10`. Octaves halve from here per the
    /// filter's lacunarity (2.0). The finest octave should land above
    /// the cubemap Nyquist — for Vaelen at ~0.87 km/texel, 30 km × 4
    /// octaves bottoms at 3.7 km, comfortably above.
    pub scale_m: f32,
    /// Per-octave strength, in [0..0.5] roughly. The filter applies
    /// `~strength * scale_m` per octave, summed geometrically (gain=0.5).
    /// The slope mask and `gully_weight` reduce it further; expect total
    /// gully amplitude ≈ `strength * scale_m * 1.9 * gully_weight * 0.5`.
    pub strength: f32,
    /// Visible-gully weight inside the slope band, [0..1]. 0 still
    /// sharpens peaks/valleys but produces no gullies; 1 gives full
    /// gullies but leaves peaks/valleys rounded.
    #[serde(default = "default_gully_weight")]
    pub gully_weight: f32,
    /// Octaves of gully noise (3..6 typical). Cost is roughly linear.
    #[serde(default = "default_octaves")]
    pub octaves: u32,
    /// Multiplier on the filter's accumulated magnitude, added as a
    /// **uniform global** height offset (not slope-masked). The Shadertoy
    /// reference uses `-0.65` to make a single mountain demo look more
    /// carved; on a planet-scale heightfield with mixed terrain it just
    /// drops everything by a constant. Default 0.0 keeps the global
    /// shape; the per-pixel slope-masked carving in `result.delta` does
    /// the actual mountain erosion.
    #[serde(default = "default_height_bias_factor")]
    pub height_bias_factor: f32,
    /// Reference amplitude (m) for `fade_target = clamp(h / mean_amp_m, -1, 1)`.
    /// Biases the filter to preserve extrema in flat masked-out regions
    /// (peaks bias up, troughs bias down). Roughly the body's typical
    /// mountain elevation.
    #[serde(default = "default_mean_amp_m")]
    pub mean_amp_m: f32,
    /// Uniform multiplier on all four `onset` thresholds in
    /// `ErosionFilterParams`. 1.0 = upstream defaults (gullies activate
    /// on ≳22° slopes). Drop toward ~0.3 to spread gullies onto gentler
    /// mountain flanks.
    #[serde(default = "default_slope_scale")]
    pub slope_scale: f32,
}

fn default_gully_weight() -> f32 {
    0.6
}
fn default_octaves() -> u32 {
    4
}
fn default_height_bias_factor() -> f32 {
    0.0
}
fn default_mean_amp_m() -> f32 {
    2_000.0
}
fn default_slope_scale() -> f32 {
    1.0
}

impl Erosion {
    /// Tuning starting point for cold, formerly wet desert bodies. The preset
    /// is currently validated against Vaelen (`ColdDesertFormerlyWet`,
    /// R≈1130 km, default ~2048² faces, ~0.87 km/texel).
    pub fn cold_desert_default() -> Self {
        Self {
            scale_m: 30_000.0,
            strength: 0.04,
            gully_weight: 0.6,
            octaves: 4,
            height_bias_factor: 0.0,
            mean_amp_m: 2_000.0,
            slope_scale: 0.3,
        }
    }

    pub fn vaelen_default() -> Self {
        Self::cold_desert_default()
    }

    fn to_filter_params(&self) -> ErosionFilterParams {
        let defaults = ErosionFilterParams::default();
        ErosionFilterParams {
            scale: self.scale_m,
            strength: self.strength,
            gully_weight: self.gully_weight,
            octaves: self.octaves as i32,
            onset: defaults.onset * self.slope_scale,
            ..defaults
        }
    }
}

impl Stage for Erosion {
    fn name(&self) -> &str {
        "erosion"
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        if self.strength <= 0.0 || self.octaves == 0 || self.scale_m <= 0.0 {
            return;
        }

        let res = builder.cubemap_resolution;
        let radius = builder.radius_m;
        // Map the octahedron unit-square diagonals to one body-radius of
        // arc, so `params.scale` (in meters) is interpreted in the same
        // physical units as before. The octahedron's L1-based mapping
        // distorts area by up to ~2×, but `params.scale` is a soft
        // visual-tuning knob — exact area-preservation isn't needed.
        let world_p_scale = radius;
        // Finite-difference step in `p`-space. Half a cubemap texel of
        // arc length, converted back through the projection. Smaller
        // than the filter's smallest cell (params.scale × cell_scale ≈
        // 21 km) so the gradient resolves the underlying height field
        // rather than the gully pattern itself.
        let eps_p = 0.5 * cube_face_texel_scale_m(radius, res);
        let params = self.to_filter_params();
        let bias = self.height_bias_factor;
        let inv_mean_amp = if self.mean_amp_m > 0.0 {
            1.0 / self.mean_amp_m
        } else {
            0.0
        };

        // Snapshot for read-only cross-face sampling. The in-place pass
        // below mutates `builder.height_contributions.height`, but every
        // gradient and base-height read must see the pre-erosion field.
        let snapshot = builder.height_contributions.height.clone();

        builder
            .height_contributions
            .height
            .faces_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(face_idx, slice)| {
                let face = CubemapFace::ALL[face_idx];
                let res_us = res as usize;
                let res_f = res as f32;
                for y in 0..res {
                    let v = (y as f32 + 0.5) / res_f;
                    for x in 0..res {
                        let u = (x as f32 + 0.5) / res_f;
                        let dir = face_uv_to_dir(face, u, v);

                        // Octahedron coords for this texel; cells in
                        // `phacelle_noise` are placed in this 2D space,
                        // so they're continuous across cubemap faces.
                        let p = octahedron_encode(dir) * world_p_scale;

                        let h = snapshot.sample_bilinear(dir);
                        let h_xp = sample_at_p(&snapshot, p + Vec2::new(eps_p, 0.0), world_p_scale);
                        let h_xn = sample_at_p(&snapshot, p - Vec2::new(eps_p, 0.0), world_p_scale);
                        let h_yp = sample_at_p(&snapshot, p + Vec2::new(0.0, eps_p), world_p_scale);
                        let h_yn = sample_at_p(&snapshot, p - Vec2::new(0.0, eps_p), world_p_scale);
                        let dh_dx = (h_xp - h_xn) / (2.0 * eps_p);
                        let dh_dy = (h_yp - h_yn) / (2.0 * eps_p);

                        let base = Vec3::new(h, dh_dx, dh_dy);
                        let fade_target = (h * inv_mean_amp).clamp(-1.0, 1.0);
                        let result = erosion_filter(p, base, fade_target, &params);

                        // Write only height — the cubemap format doesn't
                        // carry gradient, and the impostor reconstructs
                        // per-fragment normals from the eroded height
                        // field at render time.
                        let new_h = h + result.delta.x + bias * result.magnitude;
                        slice[(y as usize) * res_us + (x as usize)] = new_h;
                    }
                }
            });
    }
}

#[inline]
fn sample_at_p(snapshot: &Cubemap<f32>, p: Vec2, world_p_scale: f32) -> f32 {
    let dir = octahedron_decode(p / world_p_scale);
    snapshot.sample_bilinear(dir)
}

/// Standard octahedron encoding (Cigolle et al., "A Survey of Efficient
/// Representations for Independent Unit Vectors"). Maps unit sphere →
/// `[-1, 1]²`, bijective. C⁰ everywhere; the only non-smooth point is
/// the equator of the input z-axis (the diamond `|x| + |y| = 1` in the
/// output).
#[inline]
fn octahedron_encode(dir: Vec3) -> Vec2 {
    let n = dir / (dir.x.abs() + dir.y.abs() + dir.z.abs()).max(1e-20);
    if n.z >= 0.0 {
        Vec2::new(n.x, n.y)
    } else {
        Vec2::new(
            (1.0 - n.y.abs()) * sign_nonzero(n.x),
            (1.0 - n.x.abs()) * sign_nonzero(n.y),
        )
    }
}

/// Inverse of `octahedron_encode`. Produces a unit-length direction.
#[inline]
fn octahedron_decode(uv: Vec2) -> Vec3 {
    let mut n = Vec3::new(uv.x, uv.y, 1.0 - uv.x.abs() - uv.y.abs());
    if n.z < 0.0 {
        let nx = (1.0 - n.y.abs()) * sign_nonzero(n.x);
        let ny = (1.0 - n.x.abs()) * sign_nonzero(n.y);
        n.x = nx;
        n.y = ny;
    }
    n.normalize_or_zero()
}

#[inline]
fn sign_nonzero(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}
