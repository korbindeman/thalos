//! DuneSeas stage — rasterizes hand-anchored aeolian regions into the
//! height + albedo cubemaps at the draa scale (~km wavelengths).
//!
//! Per region:
//! - membership: a smooth disc on the sphere (`region_weight`)
//! - placement:  sparse barchanoid/crescent lobes in the local wind frame,
//!               with fbm sand-sheet gating so regions do not read as
//!               fixed-spacing contour lines
//! - profile:    broad sand body plus a curved brink/crest mask
//! - albedo:     direct grading toward darker interdune sand and brighter
//!               active crests
//!
//! See `docs/gen/dunes.md` §B.2-B.3 for the math and §F.1 for placement.
//! The dune-scale band (sub-km) is left for the impostor to synthesize
//! per fragment — bodies of Mira-to-Earth size can't resolve it in a
//! cubemap.

use glam::Vec3;

use crate::aeolian::{asym_ridge, region_weight};
use crate::body_builder::BodyBuilder;
use crate::cubemap::CubemapFace;
use crate::noise::{fbm3, pcg_u32};
use crate::seeding::sub_seed;
use crate::stage::Stage;
use crate::stages::util::for_face_texels_in_cap_rows;
use crate::surface_field::{mix3, smoothstep};
use crate::types::DuneSea;

/// Bake stage that paints draa-scale dune ridges + crest tint into a body
/// from one or more hand-anchored `DuneSea` regions.
pub struct DuneSeas {
    pub regions: Vec<DuneSea>,
}

impl Stage for DuneSeas {
    fn name(&self) -> &str {
        "dune_seas"
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        if self.regions.is_empty() {
            return;
        }

        let res = builder.cubemap_resolution;
        let body_radius_m = builder.radius_m;
        let stage_seed = builder.stage_seed();

        // Snapshot regions so the parallel-bake closure can borrow immutably.
        let regions = &self.regions;

        // Per-face / per-region painting. We loop regions outside the
        // texel iterator so each region's spherical-cap bbox limits work
        // to the texels that can actually receive contribution.
        for face_idx in 0..6 {
            let face = CubemapFace::ALL[face_idx];

            // Borrow both height + albedo accumulator slices for this
            // face. Albedo is RGBA where alpha is the accumulated weight
            // (`finalize_albedo` divides RGB through to recover the
            // weighted-mean color).
            let res_usize = res as usize;
            let height_slice = builder.height_contributions.height.face_data_mut(face);
            let albedo_slice = builder.albedo_contributions.albedo.face_data_mut(face);

            for (region_idx, region) in regions.iter().enumerate() {
                let region_seed =
                    sub_seed(stage_seed ^ region.seed, &format!("region:{region_idx}"));
                let outer_rad = region.influence_radius_rad();
                if outer_rad <= 0.0 {
                    continue;
                }
                // Outer cross-wind axis; `region_weight` already enforces
                // `dir.angle(center) ≤ outer_rad`, so we pass that as the
                // cap half-angle. The bbox path skips faces that don't
                // touch the cap.
                let cross = region.center.cross(region.axis_tangent);
                let cross_unit = cross.try_normalize().unwrap_or(Vec3::Y);

                // Iterate texels on this face inside the cap. The closure
                // can't `&mut` the slices through a captured borrow if we
                // do it in parallel, so use the row-restricted variant
                // and split the row range across rayon workers.
                let row_chunks = (res as usize).max(1);
                // Emit all (x, y, dir) inside the cap for the whole face
                // serially, then process — this is sufficient for v1 and
                // matches `Scarps`. With ~5 regions and bbox-bound work
                // it stays fast.
                for_face_texels_in_cap_rows(
                    face,
                    res,
                    region.center,
                    outer_rad,
                    0,
                    row_chunks as u32,
                    |x, y, dir, _ang| {
                        let weight = region_weight(
                            dir,
                            region.center,
                            region.radius_rad,
                            region.feather_rad,
                        );
                        if weight <= 0.0 {
                            return;
                        }

                        // Cross-wind anisotropic warp: displace the
                        // sample point along `cross` by an fbm-driven
                        // amount, then take the wind-aligned phase at
                        // the displaced point. A slower wavelength jitter
                        // breaks the "radio-wave" regularity without
                        // losing the prevailing wind read.
                        // (See `docs/gen/dunes.md §B.3`.)
                        let warp_seed = sub_seed(region_seed, "draa_warp") as u32;
                        let warp = fbm3(
                            dir.x * region.warp_freq,
                            dir.y * region.warp_freq,
                            dir.z * region.warp_freq,
                            warp_seed,
                            3,
                            0.5,
                            2.02,
                        ) * region.warp_amp_unit;
                        let dir_w = (dir + cross_unit * warp).normalize();

                        let tangent_disp = dir_w - region.center * dir_w.dot(region.center);
                        let along = tangent_disp.dot(region.axis_tangent.normalize_or_zero());
                        let cross_m = tangent_disp.dot(cross_unit) * body_radius_m;
                        let along_m = along * body_radius_m;
                        let (sheet_body, sheet_crest, sand_sheet) = layered_dune_sheet(
                            cross_m,
                            along_m,
                            region.lambda_draa_m,
                            sub_seed(region_seed, "draa_layered_sheet") as u32,
                        );
                        let (anchor_body, anchor_crest) = barchanoid_dune_field(
                            cross_m,
                            along_m,
                            region.radius_rad * body_radius_m,
                            region.lambda_draa_m,
                            sub_seed(region_seed, "draa_barchanoid_lobes") as u32,
                        );
                        let lobe = smoothstep(
                            -0.20,
                            0.52,
                            fbm3(
                                dir.x * region.warp_freq * 0.52,
                                dir.y * region.warp_freq * 0.52,
                                dir.z * region.warp_freq * 0.52,
                                sub_seed(region_seed, "draa_sand_sheet") as u32,
                                4,
                                0.55,
                                2.02,
                            ) * 0.62
                                + sand_sheet * 0.46
                                + weight * 0.10,
                        );
                        let maze = ridge_value(
                            fbm3(
                                dir.x * region.warp_freq * 2.2,
                                dir.y * region.warp_freq * 2.2,
                                dir.z * region.warp_freq * 2.2,
                                sub_seed(region_seed, "draa_irregular_ridge_net") as u32,
                                4,
                                0.54,
                                2.02,
                            ) * 0.86
                                + fbm3(
                                    dir.x * region.warp_freq * 6.8,
                                    dir.y * region.warp_freq * 6.8,
                                    dir.z * region.warp_freq * 6.8,
                                    sub_seed(region_seed, "draa_irregular_ridge_fray") as u32,
                                    3,
                                    0.50,
                                    2.02,
                                ) * 0.18,
                        );
                        let ridge_net = smoothstep(0.58, 0.92, maze) * lobe * weight;
                        let body = (sheet_body * 1.05 + anchor_body * 0.05 + ridge_net * 0.12)
                            .clamp(0.0, 1.0);
                        let crest = (sheet_crest * 0.88 + anchor_crest * 0.03 + ridge_net * 0.08)
                            .clamp(0.0, 1.0);

                        let relief =
                            (body * 1.08 + crest * 0.46 + sand_sheet * 0.12) * (0.54 + lobe * 0.82);
                        let h_delta = region.amplitude_draa_m * relief * weight;
                        let idx = (y as usize) * res_usize + x as usize;
                        height_slice[idx] += h_delta;

                        // Directly grade the existing albedo so this stage
                        // can run after the biome color pass. Interdune
                        // areas are darker/rustier, while crests pull toward
                        // the active-sand color.
                        let tint_profile =
                            (sand_sheet * 0.86 + body * 0.58 + crest * 0.20) * (0.64 + lobe * 0.52);
                        let tint_strength =
                            (region.crest_strength.max(0.0) * 2.65 * tint_profile * weight)
                                .clamp(0.0, 0.68);
                        if tint_strength > 0.0 {
                            let cur = albedo_slice[idx];
                            let alpha = cur[3].max(1.0e-5);
                            let base = [cur[0] / alpha, cur[1] / alpha, cur[2] / alpha];
                            let interdune = [
                                (region.albedo_crest_lin[0] * 0.58).clamp(0.0, 1.0),
                                (region.albedo_crest_lin[1] * 0.45).clamp(0.0, 1.0),
                                (region.albedo_crest_lin[2] * 0.34).clamp(0.0, 1.0),
                            ];
                            let dune_color = mix3(
                                interdune,
                                region.albedo_crest_lin,
                                (crest * 0.22 + body * 0.30 + sand_sheet * 0.30 + lobe * 0.12)
                                    .clamp(0.0, 1.0),
                            );
                            let graded = mix3(base, dune_color, tint_strength);
                            albedo_slice[idx] = [
                                graded[0] * alpha,
                                graded[1] * alpha,
                                graded[2] * alpha,
                                alpha,
                            ];
                        }
                    },
                );
            }
        }

        // Hand the regions to BodyBuilder so they end up on BodyData.
        // The impostor will pull them via `BodyData.dune_seas`.
        builder.dune_seas.extend(regions.iter().cloned());
    }
}

fn barchanoid_dune_field(
    cross_m: f32,
    along_m: f32,
    region_radius_m: f32,
    lambda_m: f32,
    seed: u32,
) -> (f32, f32) {
    const LOBE_COUNT: u32 = 24;

    let mut body = 0.0_f32;
    let mut crest = 0.0_f32;
    let radius_m = region_radius_m.max(lambda_m * 2.0);

    for i in 0..LOBE_COUNT {
        let angle = hash01(seed, i * 11 + 1) * std::f32::consts::TAU;
        let distance = hash01(seed, i * 11 + 2).sqrt() * radius_m * 0.86;
        let (sin_a, cos_a) = angle.sin_cos();
        let center_cross = cos_a * distance;
        let center_along = sin_a * distance;

        let width_m = lambda_m * (0.92 + hash01(seed, i * 11 + 3) * 1.70);
        let length_m = lambda_m * (1.15 + hash01(seed, i * 11 + 4) * 2.15);
        let yaw = (hash01(seed, i * 11 + 5) * 2.0 - 1.0) * 0.38;
        let (sin_y, cos_y) = yaw.sin_cos();

        let dx = cross_m - center_cross;
        let dy = along_m - center_along;
        let x = (dx * cos_y + dy * sin_y) / width_m.max(1.0);
        let y = (-dx * sin_y + dy * cos_y) / length_m.max(1.0);

        // Crescentic footprint: the parabolic y-shift pulls the brink into
        // a barchan-like arc and leaves downwind horns instead of full-width
        // contour lines.
        let crescent_y = y + 0.34 * x * x - 0.10;
        let footprint = 1.0 - ((x.abs() * 0.76).powf(1.55) + crescent_y.abs().powf(1.22));
        let core = smoothstep(-0.12, 0.36, footprint);
        let horns = smoothstep(0.34, 1.06, x.abs())
            * smoothstep(-0.22, 0.50, crescent_y)
            * smoothstep(1.30, 0.48, x.abs());
        let local_body = (core * 0.92 + horns * 0.38).clamp(0.0, 1.0);

        let brink_y = 0.18 + 0.22 * x * x;
        let local_crest = smoothstep(0.16, 0.0, (crescent_y - brink_y).abs())
            * smoothstep(1.18, 0.18, x.abs())
            * (0.45 + core * 0.55);
        let rib_phase = y * (3.2 + hash01(seed, i * 11 + 7) * 2.8)
            + x * (0.55 + hash01(seed, i * 11 + 8) * 1.1)
            + (x * 3.4 + hash01(seed, i * 11 + 9) * std::f32::consts::TAU).sin() * 0.28;
        let rib_t = rib_phase - rib_phase.floor();
        let rib = smoothstep(0.82, 1.0, 1.0 - (rib_t * 2.0 - 1.0).abs())
            * core
            * smoothstep(1.05, 0.18, x.abs())
            * 0.20;

        let amp = 0.62 + hash01(seed, i * 11 + 6) * 0.55;
        body = body.max(local_body * amp);
        crest = crest.max((local_crest + rib).clamp(0.0, 1.0) * amp);
    }

    (body.clamp(0.0, 1.0), crest.clamp(0.0, 1.0))
}

fn layered_dune_sheet(cross_m: f32, along_m: f32, lambda_m: f32, seed: u32) -> (f32, f32, f32) {
    let lambda = lambda_m.max(1.0);
    let u = cross_m / lambda;
    let v = along_m / lambda;

    let sand_sheet = smoothstep(
        -0.42,
        0.46,
        fbm2(
            u * 0.12,
            v * 0.10,
            pcg_u32(seed ^ 0xA173_19D1),
            4,
            0.56,
            2.0,
        ) * 0.76
            + fbm2(
                u * 0.36,
                v * 0.24,
                pcg_u32(seed ^ 0xB8B0_73AF),
                3,
                0.52,
                2.1,
            ) * 0.26,
    );
    if sand_sheet <= 0.001 {
        return (0.0, 0.0, 0.0);
    }

    let wavelength_jitter = (1.0
        + fbm2(
            u * 0.07,
            v * 0.06,
            pcg_u32(seed ^ 0x6D2B_79F5),
            3,
            0.55,
            2.0,
        ) * 0.36)
        .clamp(0.64, 1.58);
    let warp = fbm2(
        u * 0.18,
        v * 0.10,
        pcg_u32(seed ^ 0x31C4_A427),
        4,
        0.56,
        2.0,
    ) * 1.18
        + fbm2(
            u * 0.62,
            v * 0.22,
            pcg_u32(seed ^ 0xC2F3_589D),
            3,
            0.52,
            2.1,
        ) * 0.34;
    let train_mask = smoothstep(
        -0.22,
        0.50,
        fbm2(
            u * 0.22,
            v * 0.15,
            pcg_u32(seed ^ 0x44C1_5B7D),
            4,
            0.55,
            2.0,
        ) * 0.66
            + fbm2(
                u * 0.055,
                v * 0.040,
                pcg_u32(seed ^ 0x72E9_AA13),
                3,
                0.56,
                2.0,
            ) * 0.34
            + sand_sheet * 0.28,
    );

    let phase = u / wavelength_jitter
        + warp
        + fbm2(
            u * 0.34,
            v * 0.78,
            pcg_u32(seed ^ 0x19A8_2C4F),
            3,
            0.54,
            2.0,
        ) * 0.33
        + fbm2(
            u * 1.45,
            v * 0.42,
            pcg_u32(seed ^ 0x59A0_CE31),
            3,
            0.50,
            2.0,
        ) * 0.16;
    let ridge = asym_ridge(phase, 0.84);
    let break_mask = smoothstep(
        -0.18,
        0.54,
        fbm2(
            u * 0.72,
            v * 0.36,
            pcg_u32(seed ^ 0x912A_47E3),
            4,
            0.55,
            2.0,
        ) * 0.72
            + fbm2(
                u * 1.85,
                v * 0.62,
                pcg_u32(seed ^ 0xD756_1F87),
                3,
                0.50,
                2.1,
            ) * 0.20,
    );

    let body = smoothstep(0.22, 0.88, ridge) * sand_sheet * train_mask * (0.42 + break_mask * 0.72)
        + sand_sheet * train_mask * 0.26;
    let crest = smoothstep(0.70, 0.975, ridge) * sand_sheet * train_mask * break_mask * 0.76;

    (body.clamp(0.0, 1.0), crest.clamp(0.0, 1.0), sand_sheet)
}

fn hash01(seed: u32, salt: u32) -> f32 {
    let h = pcg_u32(seed ^ salt.wrapping_mul(0x9E37_79B9));
    (h >> 8) as f32 / 16_777_216.0
}

fn ridge_value(v: f32) -> f32 {
    1.0 - v.abs().clamp(0.0, 1.0)
}

fn fbm2(x: f32, y: f32, seed: u32, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for octave in 0..octaves {
        let octave_seed = pcg_u32(seed.wrapping_add(octave));
        sum += amp * value_noise_2d(x * freq, y * freq, octave_seed);
        norm += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    sum / norm.max(1.0e-6)
}

fn value_noise_2d(x: f32, y: f32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let fx = fade(x - xi as f32);
    let fy = fade(y - yi as f32);

    let c00 = hash2(xi, yi, seed);
    let c10 = hash2(xi + 1, yi, seed);
    let c01 = hash2(xi, yi + 1, seed);
    let c11 = hash2(xi + 1, yi + 1, seed);

    let x0 = c00 + (c10 - c00) * fx;
    let x1 = c01 + (c11 - c01) * fx;
    x0 + (x1 - x0) * fy
}

fn hash2(ix: i32, iy: i32, seed: u32) -> f32 {
    let mut h = pcg_u32(seed);
    h = pcg_u32(h ^ ix as u32);
    h = pcg_u32(h ^ iy as u32);
    let u = (h >> 8) as f32 / 16_777_216.0;
    u * 2.0 - 1.0
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}
