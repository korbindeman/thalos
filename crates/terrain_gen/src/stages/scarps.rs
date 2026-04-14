use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use super::util::for_face_texels_in_cap;
use crate::body_builder::BodyBuilder;
use crate::cubemap::CubemapFace;
use crate::seeding::Rng;
use crate::stage::Stage;

/// Lobate thrust-fault scarps. Sinuous compressional ridges produced by
/// global radial contraction as the body's interior cools — visible at
/// low sun angles as bright bow-shaped lineations cutting across pre-
/// existing terrain (Lee-Lincoln, Discovery Rupes, etc.). Tidally locked
/// silicate moons like Mira have a permanent stress field, so they're a
/// natural and physically motivated feature class.
///
/// The stage stamps each scarp as a Gaussian ridge along a great-circle
/// polyline whose tangent is randomly perturbed at each step (curvature),
/// producing the characteristic sinuous shape. Heights are tens to a few
/// hundred meters — subtle on the height field but dramatic at terminator
/// lighting.
#[derive(Debug, Clone, Deserialize)]
pub struct Scarps {
    /// Number of scarps to spawn over the body.
    pub count: u32,
    /// Minimum scarp arc length in meters of surface distance.
    pub min_length_m: f32,
    /// Maximum scarp arc length in meters.
    pub max_length_m: f32,
    /// Cross-section width (Gaussian σ) in meters. Real lunar scarps are
    /// typically ~300 m wide; the value here controls the bake footprint.
    pub width_m: f32,
    /// Peak ridge height in meters. ~50–300 m matches measured lunar
    /// scarps; the value gets attenuated by the Gaussian falloff so the
    /// observable peak is roughly this number.
    pub height_m: f32,
    /// How strongly the local tangent rotates per polyline step (radians).
    /// 0 = perfectly straight great circle; 0.1 ≈ gentle sinuous; 0.3 =
    /// strongly bowed.
    #[serde(default = "default_curvature")]
    pub curvature: f32,
}

fn default_curvature() -> f32 {
    0.12
}

impl Stage for Scarps {
    fn name(&self) -> &str {
        "scarps"
    }
    fn dependencies(&self) -> &[&str] {
        &["cratering"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        if self.count == 0 || self.height_m == 0.0 {
            return;
        }
        let mut rng = Rng::new(builder.stage_seed());
        let body_radius = builder.radius_m;
        let res = builder.cubemap_resolution;

        // Step length along the great circle. Set to ~half the cross-
        // section width so adjacent vertex caps overlap and the ridge
        // reads as continuous.
        let step_m = (self.width_m * 0.5).max(400.0);

        // Build all polylines first so the parallel face bake can borrow
        // them immutably.
        struct Scarp {
            vertices: Vec<Vec3>,
        }
        let mut scarps: Vec<Scarp> = Vec::with_capacity(self.count as usize);
        for _ in 0..self.count {
            let length_m =
                self.min_length_m + rng.next_f64() as f32 * (self.max_length_m - self.min_length_m);
            let n_steps = ((length_m / step_m) as usize).max(2);

            let mut pos = rng.unit_vector().as_vec3().normalize();
            // Initial tangent at random azimuth.
            let mut dir = {
                let up = if pos.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                let east = up.cross(pos).normalize();
                let north = pos.cross(east);
                let phi = rng.next_f64() as f32 * std::f32::consts::TAU;
                east * phi.cos() + north * phi.sin()
            };

            let mut vertices = Vec::with_capacity(n_steps + 1);
            vertices.push(pos);
            for _ in 0..n_steps {
                let arc = step_m / body_radius;
                pos = (pos * arc.cos() + dir * arc.sin()).normalize();
                vertices.push(pos);
                // Re-orthogonalize the tangent at the new point and
                // perturb its azimuth.
                let up = if pos.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                let east = up.cross(pos).normalize();
                let north = pos.cross(east);
                let de = dir.dot(east);
                let dn = dir.dot(north);
                let phi = dn.atan2(de) + rng.next_f64_signed() as f32 * self.curvature;
                dir = east * phi.cos() + north * phi.sin();
            }
            scarps.push(Scarp { vertices });
        }

        // Bake parameters. The cap radius covers ~3σ across-track plus the
        // along-track half-window so a single texel can be reached from at
        // most one vertex per scarp (overlapping reach is fine — the
        // along-track triangular falloff blends them).
        let across_sigma_m = self.width_m * 0.5;
        let across_extent_m = across_sigma_m * 3.0;
        let along_half_m = step_m * 1.0; // triangular window half-width
        let cap_radius_m = (across_extent_m * across_extent_m + along_half_m * along_half_m).sqrt();
        let cap_angle = cap_radius_m / body_radius;
        let across_sigma_ang = across_sigma_m / body_radius;
        let along_half_ang = along_half_m / body_radius;
        let height_m = self.height_m;

        let scarps_ref = &scarps;
        builder
            .height_contributions
            .height
            .faces_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(face_idx, slice)| {
                let face = CubemapFace::ALL[face_idx];
                for scarp in scarps_ref {
                    let n = scarp.vertices.len();
                    for i in 0..n {
                        let v = scarp.vertices[i];
                        // Local tangent on the unit sphere — forward
                        // difference at start, backward at end, central
                        // otherwise. Then re-project into the tangent
                        // plane at v to remove any radial component.
                        let raw_t = if i == 0 {
                            scarp.vertices[i + 1] - v
                        } else if i + 1 == n {
                            v - scarp.vertices[i - 1]
                        } else {
                            scarp.vertices[i + 1] - scarp.vertices[i - 1]
                        };
                        let tan_plane = (raw_t - v * raw_t.dot(v)).normalize();
                        let perp = v.cross(tan_plane);

                        for_face_texels_in_cap(face, res, v, cap_angle, |x, y, dir, _ang| {
                            // Project the texel direction into v's tangent
                            // plane (sphere → plane displacement vector).
                            let local = dir - v * dir.dot(v);
                            let along = local.dot(tan_plane);
                            let across = local.dot(perp);
                            // Along-track triangular window. Skip texels
                            // that belong to neighbour vertices.
                            if along.abs() > along_half_ang {
                                return;
                            }
                            let along_w = 1.0 - along.abs() / along_half_ang;
                            // Across-track Gaussian.
                            let g = (-(across / across_sigma_ang).powi(2) * 0.5).exp();
                            let h_delta = height_m * g * along_w;
                            if h_delta < 0.5 {
                                return;
                            }
                            let idx = (y * res + x) as usize;
                            slice[idx] += h_delta;
                        });
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body_builder::BodyBuilder;
    use crate::cubemap::CubemapFace;
    use crate::types::Composition;

    #[test]
    fn scarps_raise_terrain() {
        let comp = Composition::new(0.95, 0.04, 0.0, 0.01, 0.0);
        let mut b = BodyBuilder::new(1_000_000.0, 0xCAFE, comp, 64, 4.5, None, 0.0);
        let s = Scarps {
            count: 4,
            min_length_m: 60_000.0,
            max_length_m: 120_000.0,
            width_m: 4_000.0,
            height_m: 200.0,
            curvature: 0.1,
        };
        s.apply(&mut b);
        let mut max_h: f32 = 0.0;
        for face in CubemapFace::ALL {
            for &h in b.height_contributions.height.face_data(face).iter() {
                if h > max_h {
                    max_h = h;
                }
            }
        }
        assert!(
            max_h > 50.0,
            "scarp ridge should raise terrain, max={max_h}"
        );
    }
}
