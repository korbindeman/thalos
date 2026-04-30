//! Stage 2 — Coarse Elevation.
//!
//! Computes per-vertex elevation on the icosphere. The *continental
//! mask is independent of the plate partition* — it comes from a
//! warped multi-octave fBm field, thresholded to hit
//! `target_ocean_fraction`. Plate boundaries from Stage 1 drive
//! tectonic features (sutures, rifts, active margins, hotspots) that
//! can fall on continent or ocean without aligning to coastlines.
//! This matches Earth: the Pacific plate has no continent, Eurasia
//! carries Europe and Asia, the Mid-Atlantic Ridge runs underwater,
//! and the Andes happen to sit where a convergent plate boundary
//! crosses continental crust.
//!
//! Contributions, summed per vertex:
//!
//! 1. **Continental mask bias** — continental vs. oceanic baseline
//!    elevation, decided by thresholded noise.
//! 2. **Suture ridges** — age-decayed Gaussian ridge along Suture
//!    vertices.
//! 3. **Rift scars** — narrow linear valleys.
//! 4. **Active margins** — continental-side uplift or oceanic-side
//!    offshore trench. Side is read from the continental mask, not
//!    from plate identity. Along-belt fBm modulation produces
//!    Aleutian-style peaks + gaps rather than a continuous wall.
//! 5. **Hotspot tracks** — small positive bumps (seamounts on ocean,
//!    volcanic chains on land).
//! 6. **Domain-warped fBm noise** — fractal regional variation;
//!    amplitude sized comparable to the coastal step so the coastline
//!    itself gets broken up.
//!
//! See `docs/thalos_terrain_pipeline.md §Stage 2`.

use std::collections::VecDeque;

use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::icosphere::Icosphere;
use crate::noise::fbm3;
use crate::province::{ProvinceDef, ProvinceKind};
use crate::seeding::splitmix64;
use crate::stage::Stage;

/// No serde defaults — every body must tune every field explicitly.
/// An unset field on a CoarseElevation(()) block will fail RON parse at
/// body-load time, which is what we want: silent defaults here produce
/// terrain that looks generic-bad and invite cargo-culting from body to
/// body.
#[derive(Debug, Clone, Deserialize)]
pub struct CoarseElevation {
    // --- Continental mask (noise-derived) -----------------------------------
    /// Fraction of the sphere to be ocean. Threshold is computed by
    /// sorting per-vertex continent-noise samples and picking the value
    /// at this percentile — vertices below threshold are oceanic,
    /// above are continental.
    pub target_ocean_fraction: f32,
    /// Base frequency of the continent-mask fBm. ~1.5–2.5 gives a
    /// handful of continent-scale features at Thalos radius.
    pub continent_noise_frequency: f32,
    pub continent_noise_octaves: u32,
    pub continent_noise_persistence: f32,
    /// Amplitude of the 3D fBm domain warp applied to the continent-
    /// mask sample point, in unit-sphere units. Produces fractal
    /// interlocking coastlines. ~0.3–0.5 is a good range.
    pub continent_warp_amplitude: f32,
    pub continent_warp_frequency: f32,
    /// Anisotropic coordinate scaling on the N–S axis before continent
    /// noise is sampled. `1.0` = isotropic; `>1.0` compresses features
    /// along latitude, making continents read wider E–W than N–S
    /// (Eurasian-style elongation). Use this sparingly — 1.5–2.0
    /// visibly Earth-like; >3 looks striped.
    pub horizontal_stretch: f32,
    /// Signed additive bias on the mask value, shaped as a quadratic
    /// in `sin(lat)`: `bias * (0.5 − sin²(lat))`. Positive values push
    /// more land toward mid/low latitudes and away from the poles
    /// (Earth-like); negative values produce polar continents. `0.0` =
    /// latitude-neutral. Amplitude is in mask-value units (fBm output
    /// is roughly [−1, 1]); 0.1–0.3 is a mild-to-strong bias.
    pub mid_latitude_bias: f32,
    /// Minimum fraction of the sphere a contiguous land region must
    /// occupy to survive component filtering. Regions below this are
    /// flipped to ocean. 0.003 ≈ Madagascar at Thalos radius. Set to
    /// 0 to disable.
    pub min_land_fraction: f32,
    /// Same for ocean components — small enclosed seas below this
    /// fraction get filled in as land. 0 keeps them.
    pub min_ocean_fraction: f32,
    /// Target elevation on continental vertices (mask = true).
    pub continental_bias_m: f32,
    /// Target elevation on oceanic vertices (mask = false).
    pub oceanic_bias_m: f32,

    // --- Suture ridges ------------------------------------------------------
    pub suture_height_m: f32,
    pub suture_falloff_hops: f32,
    /// Age (Myr) at which suture amplitude decays to 1/e. Older
    /// sutures have eroded more.
    pub suture_age_decay_myr: f32,

    // --- Rift scars ---------------------------------------------------------
    pub rift_depth_m: f32,
    pub rift_falloff_hops: f32,

    // --- Active margins (asymmetric via continental mask) -------------------
    pub active_uplift_m: f32,
    pub active_trench_m: f32,
    pub active_falloff_hops: f32,
    /// Hop offset from the margin to the *peak* of the trench.
    /// Subduction trenches sit 100–300 km offshore with a continental
    /// shelf + forearc between; a positive offset reproduces that
    /// geometry instead of making the coast itself a 5 km cliff.
    pub trench_offset_hops: f32,
    pub trench_width_hops: f32,

    // --- Hotspot tracks -----------------------------------------------------
    pub hotspot_height_m: f32,
    pub hotspot_falloff_hops: f32,

    // --- Regional noise -----------------------------------------------------
    pub noise_amplitude_m: f32,
    pub noise_frequency: f32,
    pub noise_octaves: u32,
    pub noise_persistence: f32,
    pub warp_amplitude: f32,
    pub warp_frequency: f32,

    pub debug_paint_albedo: bool,
}

impl Stage for CoarseElevation {
    fn name(&self) -> &str {
        "coarse_elevation"
    }

    fn dependencies(&self) -> &[&str] {
        &["tectonic_skeleton"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let sphere = builder
            .sphere
            .as_ref()
            .expect("CoarseElevation requires sphere from TectonicSkeleton");
        let vertex_provinces = builder
            .vertex_provinces
            .as_ref()
            .expect("CoarseElevation requires vertex_provinces");
        let vertex_craton_provinces = builder
            .vertex_craton_provinces
            .as_ref()
            .expect("CoarseElevation requires vertex_craton_provinces from TectonicSkeleton");
        let provinces = &builder.provinces;

        let n_verts = sphere.vertices.len();
        let seed = builder.stage_seed();

        // 1. Continental mask — warped fBm with latitude bias,
        //    thresholded to hit the target ocean fraction by
        //    percentile. Independent of the plate partition.
        //
        //    The latitude bias is added before thresholding, so after
        //    threshold selection the bias has shifted *where* the
        //    threshold-crossings land but the total ocean fraction
        //    still hits target exactly.
        let mask_seed = splitmix64(seed ^ 0x4E7A_C0FF_EE_AA_BB_01);
        let mask_samples: Vec<f32> = sphere
            .vertices
            .par_iter()
            .map(|v| {
                let n = sample_continent_noise(*v, mask_seed, self);
                // Latitude bias shape: cos²(lat) − 0.5 = 0.5 − sin²(lat).
                // For a unit vector `v`, sin(lat) = v.y, so the shape
                // simplifies to (0.5 − v.y²). Positive at equator,
                // zero at |lat| = 45°, negative at the poles.
                let shape = 0.5 - v.y * v.y;
                n + self.mid_latitude_bias * shape
            })
            .collect();
        let threshold = percentile(&mask_samples, self.target_ocean_fraction);
        let mut is_continental: Vec<bool> = mask_samples.iter().map(|&v| v >= threshold).collect();

        // Remove tiny land/ocean fragments. Connected-component
        // filter on the vertex graph: any region (land or ocean)
        // smaller than its threshold gets flipped. Keeps fractal
        // coastlines from the mask, kills dotted-island noise.
        filter_small_components(
            &mut is_continental,
            &sphere.vertex_neighbors,
            (self.min_land_fraction * n_verts as f32) as u32,
            (self.min_ocean_fraction * n_verts as f32) as u32,
        );

        // 2. BFS distance fields for tectonic features. Cap hops to
        //    a few Gaussian sigmas of the relevant falloff; beyond
        //    that the contribution is negligible.
        let suture_cap = ((self.suture_falloff_hops * 3.0) as u32).max(1);
        let rift_cap = ((self.rift_falloff_hops * 3.0) as u32).max(1);
        let active_cap = ((self
            .active_falloff_hops
            .max(self.trench_offset_hops + self.trench_width_hops * 3.0))
            as u32)
            .max(1);
        let hotspot_cap = ((self.hotspot_falloff_hops * 3.0) as u32).max(1);

        let suture_dist = hop_distance_from_kind(
            sphere,
            vertex_provinces,
            provinces,
            ProvinceKind::Suture,
            suture_cap,
        );
        let rift_dist = hop_distance_from_kind(
            sphere,
            vertex_provinces,
            provinces,
            ProvinceKind::RiftScar,
            rift_cap,
        );
        let active_dist = hop_distance_from_kind(
            sphere,
            vertex_provinces,
            provinces,
            ProvinceKind::ActiveMargin,
            active_cap,
        );
        let hotspot_dist = hop_distance_from_kind(
            sphere,
            vertex_provinces,
            provinces,
            ProvinceKind::HotspotTrack,
            hotspot_cap,
        );

        let nearest_suture_pid = nearest_province_of_kind(
            sphere,
            vertex_provinces,
            provinces,
            ProvinceKind::Suture,
            suture_cap,
        );

        // 3. Per-vertex assembly. Pure function of per-vertex inputs.
        let elevations: Vec<f32> = (0..n_verts)
            .into_par_iter()
            .map(|vi| {
                let pos = sphere.vertices[vi];
                // Continental vertices pick up their home-craton's
                // per-province elevation bias on top of the global
                // continental baseline — creates inter-continent
                // relief variation (plateaus vs lowlands) so the
                // surface reads as more than a single flat sheet.
                // Oceanic vertices ignore craton bias (plate
                // ownership is meaningless below sea level).
                let mut h = if is_continental[vi] {
                    let craton_pid = vertex_craton_provinces[vi] as usize;
                    self.continental_bias_m + provinces[craton_pid].elevation_bias_m
                } else {
                    self.oceanic_bias_m
                };

                // Suture ridge — age-decayed Gaussian.
                if suture_dist[vi] < f32::MAX {
                    let age = nearest_suture_pid[vi]
                        .map(|pid| provinces[pid as usize].age_myr)
                        .unwrap_or(0.0);
                    let age_factor = (-age / self.suture_age_decay_myr).exp();
                    let falloff = gaussian_hops(suture_dist[vi], self.suture_falloff_hops);
                    h += self.suture_height_m * age_factor * falloff;
                }

                // Rift valley.
                if rift_dist[vi] < f32::MAX {
                    let falloff = gaussian_hops(rift_dist[vi], self.rift_falloff_hops);
                    h -= self.rift_depth_m * falloff;
                }

                // Active margin. Continental-side arc uplift,
                // oceanic-side offshore trench. Side picked from
                // the continental mask, not plate identity, so
                // boundaries that happen to cross land give
                // Andes-style arcs; those entirely in ocean give
                // island arcs / trenches. Along-belt fBm cubes
                // amplitude into isolated peaks (Aleutian-style).
                if active_dist[vi] < f32::MAX {
                    let along_belt_noise = fbm3(
                        pos.x * 18.0,
                        pos.y * 18.0,
                        pos.z * 18.0,
                        splitmix64(seed ^ 0xAC71_5E_BE_17_42) as u32,
                        4,
                        0.55,
                        2.0,
                    );
                    let t = (along_belt_noise * 0.5 + 0.5).clamp(0.0, 1.0);
                    let mult = t * t * t * 1.8;
                    if is_continental[vi] {
                        let falloff = gaussian_hops(active_dist[vi], self.active_falloff_hops);
                        h += self.active_uplift_m * falloff * mult;
                    } else {
                        let d_from_peak = (active_dist[vi] - self.trench_offset_hops).abs();
                        let trench_falloff = gaussian_hops(d_from_peak, self.trench_width_hops);
                        h -= self.active_trench_m * trench_falloff * mult.max(0.3);
                    }
                }

                // Hotspot bump.
                if hotspot_dist[vi] < f32::MAX {
                    let falloff = gaussian_hops(hotspot_dist[vi], self.hotspot_falloff_hops);
                    h += self.hotspot_height_m * falloff;
                }

                // Regional fractal noise — breaks the coastal step
                // into a fractal edge at the scale of the noise
                // amplitude.
                h += self.noise_amplitude_m * warped_fbm(pos, seed, self);

                h
            })
            .collect();

        builder.vertex_elevations_m = Some(elevations);

        bake_elevation_to_cubemap(builder);
        if self.debug_paint_albedo {
            paint_elevation_debug_albedo(builder);
        }
    }
}

// ---------------------------------------------------------------------------
// Continental noise mask
// ---------------------------------------------------------------------------

fn sample_continent_noise(pos: Vec3, seed: u64, params: &CoarseElevation) -> f32 {
    // Anisotropic coordinate scaling: multiply the N-S (y) axis by
    // `horizontal_stretch` BEFORE any noise evaluation. At stretch=2,
    // the noise sees twice as much "distance" along latitude per unit
    // of real arc, so features repeat twice as often N-S and are
    // effectively half as tall → continents read elongated E-W.
    // Applied once here so warp and fbm see the same stretched space.
    let stretch = params.horizontal_stretch;
    let p = Vec3::new(pos.x, pos.y * stretch, pos.z);

    let warp_seed = splitmix64(seed ^ 0x21_BC_83_42_5E_A1_00_11) as u32;
    let wf = params.continent_warp_frequency;
    let wx = fbm3(p.x * wf, p.y * wf, p.z * wf, warp_seed, 4, 0.5, 2.0);
    let wy = fbm3(
        p.x * wf + 17.31,
        p.y * wf + 17.31,
        p.z * wf + 17.31,
        warp_seed.wrapping_add(1),
        4,
        0.5,
        2.0,
    );
    let wz = fbm3(
        p.x * wf + 41.17,
        p.y * wf + 41.17,
        p.z * wf + 41.17,
        warp_seed.wrapping_add(2),
        4,
        0.5,
        2.0,
    );
    let warp_amp = params.continent_warp_amplitude;
    let warped = Vec3::new(
        p.x + warp_amp * wx,
        p.y + warp_amp * wy,
        p.z + warp_amp * wz,
    );
    let f = params.continent_noise_frequency;
    fbm3(
        warped.x * f,
        warped.y * f,
        warped.z * f,
        seed as u32,
        params.continent_noise_octaves,
        params.continent_noise_persistence,
        2.0,
    )
}

/// Flip any land component smaller than `min_land_size` to ocean, and
/// any ocean component smaller than `min_ocean_size` to land. BFS
/// flood-fill on the vertex adjacency graph — each vertex is visited
/// once. Preserves the fractal coastlines of surviving components.
fn filter_small_components(
    is_continental: &mut [bool],
    neighbors: &[Vec<u32>],
    min_land_size: u32,
    min_ocean_size: u32,
) {
    let n = is_continental.len();
    let mut component: Vec<i32> = vec![-1; n];
    let mut component_size: Vec<u32> = Vec::new();
    let mut next_cid: i32 = 0;

    for start in 0..n {
        if component[start] != -1 {
            continue;
        }
        let target = is_continental[start];
        let mut queue: VecDeque<u32> = VecDeque::new();
        queue.push_back(start as u32);
        component[start] = next_cid;
        let mut size: u32 = 0;
        while let Some(u) = queue.pop_front() {
            size += 1;
            for &v in &neighbors[u as usize] {
                if component[v as usize] != -1 {
                    continue;
                }
                if is_continental[v as usize] != target {
                    continue;
                }
                component[v as usize] = next_cid;
                queue.push_back(v);
            }
        }
        component_size.push(size);
        next_cid += 1;
    }

    for vi in 0..n {
        let cid = component[vi] as usize;
        let size = component_size[cid];
        let threshold = if is_continental[vi] {
            min_land_size
        } else {
            min_ocean_size
        };
        if size < threshold {
            is_continental[vi] = !is_continental[vi];
        }
    }
}

fn percentile(values: &[f32], p: f32) -> f32 {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (p.clamp(0.0, 1.0) * sorted.len() as f32) as usize;
    sorted[idx.min(sorted.len().saturating_sub(1))]
}

// ---------------------------------------------------------------------------
// BFS distance/ownership fields
// ---------------------------------------------------------------------------

fn hop_distance_from_kind(
    sphere: &Icosphere,
    vertex_provinces: &[u32],
    provinces: &[ProvinceDef],
    target: ProvinceKind,
    max_hops: u32,
) -> Vec<f32> {
    let n = sphere.vertices.len();
    let mut dist: Vec<f32> = vec![f32::MAX; n];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
    for (vi, &pid) in vertex_provinces.iter().enumerate() {
        if provinces[pid as usize].kind == target {
            dist[vi] = 0.0;
            queue.push_back((vi as u32, 0));
        }
    }
    while let Some((u, d)) = queue.pop_front() {
        if d >= max_hops {
            continue;
        }
        for &v in &sphere.vertex_neighbors[u as usize] {
            if dist[v as usize] == f32::MAX {
                dist[v as usize] = (d + 1) as f32;
                queue.push_back((v, d + 1));
            }
        }
    }
    dist
}

fn nearest_province_of_kind(
    sphere: &Icosphere,
    vertex_provinces: &[u32],
    provinces: &[ProvinceDef],
    target: ProvinceKind,
    max_hops: u32,
) -> Vec<Option<u32>> {
    let n = sphere.vertices.len();
    let mut owner: Vec<Option<u32>> = vec![None; n];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
    for (vi, &pid) in vertex_provinces.iter().enumerate() {
        if provinces[pid as usize].kind == target {
            owner[vi] = Some(pid);
            queue.push_back((vi as u32, 0));
        }
    }
    while let Some((u, d)) = queue.pop_front() {
        if d >= max_hops {
            continue;
        }
        let src = owner[u as usize];
        for &v in &sphere.vertex_neighbors[u as usize] {
            if owner[v as usize].is_none() {
                owner[v as usize] = src;
                queue.push_back((v, d + 1));
            }
        }
    }
    owner
}

// ---------------------------------------------------------------------------
// Gaussian falloff in hop space
// ---------------------------------------------------------------------------

fn gaussian_hops(hops: f32, one_over_e_hops: f32) -> f32 {
    let t = hops / one_over_e_hops;
    (-(t * t)).exp()
}

// ---------------------------------------------------------------------------
// Domain-warped multi-octave fbm on the sphere (regional detail)
// ---------------------------------------------------------------------------

fn warped_fbm(pos: Vec3, seed: u64, params: &CoarseElevation) -> f32 {
    let warp_seed = splitmix64(seed ^ 0x9E37_79B9_7F4A_7C15) as u32;
    let freq = params.warp_frequency;
    let wx = fbm3(
        pos.x * freq,
        pos.y * freq,
        pos.z * freq,
        warp_seed,
        4,
        0.5,
        2.0,
    );
    let wy = fbm3(
        pos.x * freq + 17.31,
        pos.y * freq + 17.31,
        pos.z * freq + 17.31,
        warp_seed.wrapping_add(1),
        4,
        0.5,
        2.0,
    );
    let wz = fbm3(
        pos.x * freq + 41.17,
        pos.y * freq + 41.17,
        pos.z * freq + 41.17,
        warp_seed.wrapping_add(2),
        4,
        0.5,
        2.0,
    );

    let warp_amp = params.warp_amplitude;
    let warped = Vec3::new(
        pos.x + warp_amp * wx,
        pos.y + warp_amp * wy,
        pos.z + warp_amp * wz,
    );

    let base_seed = splitmix64(seed) as u32;
    let f = params.noise_frequency;
    fbm3(
        warped.x * f,
        warped.y * f,
        warped.z * f,
        base_seed,
        params.noise_octaves,
        params.noise_persistence,
        2.0,
    )
}

// ---------------------------------------------------------------------------
// Cubemap bake
// ---------------------------------------------------------------------------

fn bake_elevation_to_cubemap(builder: &mut BodyBuilder) {
    let sphere = builder.sphere.as_ref().unwrap();
    let elevations = builder.vertex_elevations_m.as_ref().unwrap();
    let res = builder.cubemap_resolution;
    let inv = 1.0 / res as f32;

    for face in CubemapFace::ALL {
        let data = builder.height_contributions.height.face_data_mut(face);
        data.par_chunks_mut(res as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let mut last_v = 0u32;
                for (x, texel) in row.iter_mut().enumerate() {
                    let u = (x as f32 + 0.5) * inv;
                    let v = (y as f32 + 0.5) * inv;
                    let dir = face_uv_to_dir(face, u, v);
                    let (ti, w) = sphere.barycentric_triangle(dir, last_v);
                    let tri = sphere.triangles[ti as usize];
                    last_v = tri[0];
                    *texel = w[0] * elevations[tri[0] as usize]
                        + w[1] * elevations[tri[1] as usize]
                        + w[2] * elevations[tri[2] as usize];
                }
            });
    }
}

fn paint_elevation_debug_albedo(builder: &mut BodyBuilder) {
    let sphere = builder.sphere.as_ref().unwrap();
    let elevations = builder.vertex_elevations_m.as_ref().unwrap();
    let res = builder.cubemap_resolution;
    let inv = 1.0 / res as f32;

    let mut min_e = f32::MAX;
    let mut max_e = f32::MIN;
    for &e in elevations {
        min_e = min_e.min(e);
        max_e = max_e.max(e);
    }
    let range = (max_e - min_e).max(1.0);

    for face in CubemapFace::ALL {
        let data = builder.albedo_contributions.albedo.face_data_mut(face);
        data.par_chunks_mut(res as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let mut last_v = 0u32;
                for (x, texel) in row.iter_mut().enumerate() {
                    let u = (x as f32 + 0.5) * inv;
                    let v = (y as f32 + 0.5) * inv;
                    let dir = face_uv_to_dir(face, u, v);
                    let (ti, w) = sphere.barycentric_triangle(dir, last_v);
                    let tri = sphere.triangles[ti as usize];
                    last_v = tri[0];
                    let e = w[0] * elevations[tri[0] as usize]
                        + w[1] * elevations[tri[1] as usize]
                        + w[2] * elevations[tri[2] as usize];
                    let t = (e - min_e) / range;
                    let (r, g, b) = if e < 0.0 {
                        let k = t / ((0.0 - min_e) / range).max(0.01);
                        (0.05 + 0.10 * k, 0.12 + 0.25 * k, 0.35 + 0.30 * k)
                    } else {
                        let k = (e - 0.0) / (max_e - 0.0).max(1.0);
                        (0.35 + 0.45 * k, 0.40 + 0.25 * k, 0.25 + 0.10 * k)
                    };
                    *texel = [r, g, b, 1.0];
                }
            });
    }
}
