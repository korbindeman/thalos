//! Stage 3 — Hydrological Carving.
//!
//! Water-driven landscape evolution on the icosphere. Takes the Stage 2
//! coarse-elevation field and lets rivers carve it for a few hundred
//! million years of simulated time: flow routing, stream-power erosion,
//! sediment deposition, and final sea-level selection. This is the
//! character-defining stage per spec — everywhere the surface stops
//! looking "noisy blob" and starts looking like a planet with rivers
//! draining to an ocean is this stage.
//!
//! Sub-steps, run in a loop for `erosion_iterations`:
//!
//! 1. **Pit-fill** (Priority-Flood). Raises interior local minima on
//!    land to their lowest escape route so flow routing has a downhill
//!    path from every vertex. The initial field out of Stage 2 has
//!    noise-induced pits; without this, erosion stalls.
//! 2. **D8 flow routing**. Each vertex points to its steepest-descent
//!    neighbor. Self-reference = sink (ocean or unresolved pit).
//! 3. **Flow accumulation**. Topological pass (elevation-sorted) that
//!    propagates upstream area downstream. Per-vertex value is total
//!    catchment in m².
//! 4. **Stream-power erosion**. `dh = -K · A^m · slope^n · dt` for each
//!    land vertex. Erodes faster where more water flows over steeper
//!    ground, which is what carves dendritic valleys.
//! 5. **Sediment transport** (simplified). The material eroded from a
//!    vertex is accumulated downstream and partially deposited where
//!    slope drops — produces floodplain fill at continent margins and
//!    sinks.
//!
//! After the loop settles, a final pass picks the sea-level elevation
//! at the `target_ocean_fraction` percentile of the eroded heightfield.
//! This matches the spec's "pick sea level last, after the landscape
//! has settled."
//!
//! Outputs committed to `BodyBuilder`:
//! - refined `vertex_elevations_m`
//! - `drainage_graph` (persistent — gameplay consumes rivers)
//! - `vertex_sediment_m` (persistent — Stage 5 uses sediment provenance)
//! - `sea_level_m`
//!
//! See `docs/thalos_terrain_pipeline.md §Stage 3`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use glam::Vec3;
use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::drainage::DrainageGraph;
use crate::icosphere::Icosphere;
use crate::stage::Stage;

#[derive(Debug, Clone, Deserialize)]
pub struct HydrologicalCarving {
    /// Number of outer erosion iterations. Each runs a full
    /// pit-fill + flow-route + accumulate + erode cycle. More
    /// iterations ⇒ more mature landscape (deeper valleys, flatter
    /// uplands). For Thalos's "billion-year-old" aesthetic, 80–120 is
    /// in the right zone.
    #[serde(default = "default_erosion_iterations")]
    pub erosion_iterations: u32,
    /// Stream-power erodibility constant per iteration. Tuned so total
    /// erosion across all iterations lands in the ~100–500 m range,
    /// which is enough to carve visible valleys without sawing
    /// continents in half.
    #[serde(default = "default_erosion_k")]
    pub erosion_k: f32,
    /// Drainage-area exponent in the stream-power law. Classic
    /// detachment-limited value 0.5.
    #[serde(default = "default_stream_power_m")]
    pub stream_power_m: f32,
    /// Slope exponent in the stream-power law. Classic detachment-
    /// limited value 1.0.
    #[serde(default = "default_stream_power_n")]
    pub stream_power_n: f32,
    /// Maximum height (m) any vertex can erode in one iteration — caps
    /// the numerical overshoot on high-drainage / high-slope outliers
    /// and keeps the landscape stable.
    #[serde(default = "default_max_erosion_per_iter_m")]
    pub max_erosion_per_iter_m: f32,
    /// Fraction of eroded material deposited on the *downstream* vertex
    /// per hop (rest continues as fluvial flux). 0.0 = no deposition;
    /// 1.0 = everything deposits one hop down. Low values produce
    /// braided floodplains across long flat lowlands; high values pile
    /// sediment right at the foot of uplands.
    #[serde(default = "default_deposition_rate")]
    pub deposition_rate: f32,
    /// Tiny slope forced during pit-fill to keep drainage direction
    /// unambiguous. A millimeter per hop is enough — doesn't show up
    /// in the final elevation.
    #[serde(default = "default_pit_fill_epsilon_m")]
    pub pit_fill_epsilon_m: f32,
    /// Target fraction of the surface below sea level after erosion.
    /// Spec: Thalos is 65% ocean → 0.65.
    #[serde(default = "default_target_ocean_fraction")]
    pub target_ocean_fraction: f32,
    /// Accumulation threshold (m²) above which a drainage edge is
    /// drawn as a river in the debug visualization. Coarser icosphere
    /// levels need a larger threshold to avoid painting the whole
    /// continent as river-blue.
    #[serde(default = "default_river_accumulation_threshold_m2")]
    pub river_accumulation_threshold_m2: f32,
    /// River perpendicular half-width, in radians of arc — the distance
    /// from a drainage edge within which a cubemap texel gets painted
    /// river-blue. 0.0015 rad ≈ 5 km on Thalos, which from orbit reads
    /// as a thin ribbon.
    #[serde(default = "default_river_half_width_rad")]
    pub river_half_width_rad: f32,
    /// When true, repaint the albedo cubemap with an updated
    /// hypsometric view that shows sea level, rivers, and sediment.
    #[serde(default = "default_true")]
    pub debug_paint_albedo: bool,
}

fn default_erosion_iterations() -> u32 {
    100
}
fn default_erosion_k() -> f32 {
    // Chosen so K · sqrt(area_per_vertex) · typical_slope lands near
    // ~1 m/iteration at moderate drainage — see stage docs for the
    // math. Tunable per body.
    2.0e-4
}
fn default_stream_power_m() -> f32 {
    0.5
}
fn default_stream_power_n() -> f32 {
    1.0
}
fn default_max_erosion_per_iter_m() -> f32 {
    50.0
}
fn default_deposition_rate() -> f32 {
    0.08
}
fn default_pit_fill_epsilon_m() -> f32 {
    0.001
}
fn default_target_ocean_fraction() -> f32 {
    0.65
}
fn default_river_accumulation_threshold_m2() -> f32 {
    // ~25 vertex-areas on level 6 Thalos (≈ 8×10¹⁰ m² catchment).
    // Tunable: drop for more rivers, raise for only major rivers.
    8.0e10
}
fn default_river_half_width_rad() -> f32 {
    0.0015
}
fn default_true() -> bool {
    true
}

impl Stage for HydrologicalCarving {
    fn name(&self) -> &str {
        "hydrological_carving"
    }

    fn dependencies(&self) -> &[&str] {
        &["coarse_elevation"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        let sphere = builder
            .sphere
            .as_ref()
            .expect("HydrologicalCarving requires sphere")
            .clone();
        let mut elevations = builder
            .vertex_elevations_m
            .clone()
            .expect("HydrologicalCarving requires vertex_elevations_m from Stage 2");

        let n_verts = sphere.vertices.len();
        let radius_m = builder.radius_m;

        // Per-vertex area (m²). An icosphere has very-nearly-equal cells,
        // so a single constant is a fine first approximation for stream
        // power. (Level 6 = 40 962 verts → ~3.1×10⁹ m² per cell on
        // Thalos.)
        let vertex_area_m2 = 4.0 * std::f32::consts::PI * radius_m * radius_m / n_verts as f32;

        // Sea level picked from the initial heightfield and held
        // CONSTANT across all iterations. Re-picking by percentile
        // per-iteration created a runaway: pit-fill raises land
        // vertices a tiny bit, shrinking the land distribution →
        // percentile re-pick drops sea level → more vertices count
        // as land next iteration → pit-fill raises more → …
        // Over 100 iterations this drowned the coastline under a
        // hugely negative sea level. Fixing sea level to the initial
        // percentile keeps the erosion target consistent.
        let sea_level = pick_sea_level(&elevations, self.target_ocean_fraction);

        // Per-vertex sediment produced by this whole run. Built up across
        // iterations.
        let mut sediment = vec![0.0f32; n_verts];

        for _iter in 0..self.erosion_iterations {
            // 1. Pit-fill on land. Ocean vertices are open sinks.
            pit_fill(&sphere, &mut elevations, sea_level, self.pit_fill_epsilon_m);

            // 2. D8 flow routing.
            let downstream = flow_routing(&sphere, &elevations, sea_level);

            // 3. Flow accumulation.
            let accumulation = flow_accumulation(&elevations, &downstream, vertex_area_m2);

            // 4+5. Erode and deposit.
            erode_and_deposit(
                &sphere,
                &mut elevations,
                &mut sediment,
                &downstream,
                &accumulation,
                radius_m,
                sea_level,
                self.erosion_k,
                self.stream_power_m,
                self.stream_power_n,
                self.max_erosion_per_iter_m,
                self.deposition_rate,
            );
        }

        // Final pit-fill before publishing the drainage graph. Erosion
        // and deposition in the last iteration can re-introduce small
        // interior minima; without this pass those become sinks in the
        // final graph and cause rivers to dead-end mid-continent
        // instead of reaching the ocean.
        pit_fill(&sphere, &mut elevations, sea_level, self.pit_fill_epsilon_m);

        // Final drainage graph from the settled heightfield.
        let downstream = flow_routing(&sphere, &elevations, sea_level);
        let accumulation = flow_accumulation(&elevations, &downstream, vertex_area_m2);

        builder.vertex_elevations_m = Some(elevations);
        builder.vertex_sediment_m = Some(sediment);
        builder.drainage_graph = Some(DrainageGraph {
            downstream,
            accumulation_m2: accumulation,
        });
        builder.sea_level_m = Some(sea_level);

        if self.debug_paint_albedo {
            paint_hydrology_debug_albedo(
                builder,
                self.river_accumulation_threshold_m2,
                self.river_half_width_rad,
            );
        }

        // Re-bake the elevation into the cubemap height accumulator so
        // the rendered impostor reflects the eroded landscape.
        bake_elevation_to_cubemap(builder);
    }
}

// ---------------------------------------------------------------------------
// Pit filling — Priority-Flood
// ---------------------------------------------------------------------------

/// Raise any interior minimum on land to its lowest escape route so
/// the flow-routing step finds a downhill path from every vertex.
/// Standard Priority-Flood (Barnes et al. 2014): seed the priority
/// queue with every ocean vertex, then pop the lowest open vertex and
/// force every unprocessed neighbor to elevation ≥ popped_elevation +
/// epsilon before pushing it back on the queue. One pass, O(V log V).
fn pit_fill(sphere: &Icosphere, elevations: &mut [f32], sea_level: f32, epsilon_m: f32) {
    let n = sphere.vertices.len();
    let mut processed = vec![false; n];
    // BinaryHeap is a max-heap; we use `Reverse` to get a min-heap, and
    // millimeter-precision i64 keys to compare floats reliably.
    let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();

    for vi in 0..n {
        if elevations[vi] <= sea_level {
            processed[vi] = true;
            heap.push(Reverse((elev_key(elevations[vi]), vi as u32)));
        }
    }

    while let Some(Reverse((_, vi))) = heap.pop() {
        let e = elevations[vi as usize];
        for &nb in &sphere.vertex_neighbors[vi as usize] {
            let nbi = nb as usize;
            if processed[nbi] {
                continue;
            }
            processed[nbi] = true;
            let raised = elevations[nbi].max(e + epsilon_m);
            elevations[nbi] = raised;
            heap.push(Reverse((elev_key(raised), nb)));
        }
    }
}

fn elev_key(e: f32) -> i64 {
    // Millimeter precision; f32 elevations within ±1000 km fit easily.
    (e * 1000.0) as i64
}

// ---------------------------------------------------------------------------
// D8 flow routing
// ---------------------------------------------------------------------------

/// For each vertex, point to the steepest-descent neighbor. If no
/// neighbor is lower (interior sink or ocean after pit-fill), the
/// vertex points to itself. Ocean vertices are treated as sinks too —
/// water that reaches the ocean leaves the terrestrial flow graph.
fn flow_routing(sphere: &Icosphere, elevations: &[f32], sea_level: f32) -> Vec<u32> {
    (0..sphere.vertices.len() as u32)
        .into_par_iter()
        .map(|vi| {
            if elevations[vi as usize] <= sea_level {
                return vi; // ocean sink
            }
            let my_e = elevations[vi as usize];
            let mut best = vi;
            let mut best_e = my_e;
            for &nb in &sphere.vertex_neighbors[vi as usize] {
                let e = elevations[nb as usize];
                if e < best_e {
                    best_e = e;
                    best = nb;
                }
            }
            best
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Flow accumulation
// ---------------------------------------------------------------------------

/// Propagate upstream area downstream. Each vertex starts with its own
/// area; in topological order (decreasing elevation) every vertex adds
/// its running total to its downstream neighbor.
fn flow_accumulation(elevations: &[f32], downstream: &[u32], vertex_area_m2: f32) -> Vec<f32> {
    let mut accum = vec![vertex_area_m2; elevations.len()];
    crate::drainage::accumulate_downstream(elevations, downstream, |from, to| {
        accum[to] += accum[from];
    });
    accum
}

// ---------------------------------------------------------------------------
// Erosion + deposition
// ---------------------------------------------------------------------------

/// Stream-power erosion with simplified per-hop deposition. Applied
/// to every land vertex in one iteration's step: the vertex erodes by
/// `K · A^m · S^n`, the eroded material is passed downstream, and a
/// fraction `deposition_rate` of the incoming downstream flux is
/// deposited on the receiver vertex (raises it and bumps its sediment
/// thickness). The rest continues as fluvial flux to the next hop.
#[allow(clippy::too_many_arguments)]
fn erode_and_deposit(
    sphere: &Icosphere,
    elevations: &mut [f32],
    sediment: &mut [f32],
    downstream: &[u32],
    accumulation: &[f32],
    radius_m: f32,
    sea_level: f32,
    k: f32,
    m: f32,
    n: f32,
    max_erosion_m: f32,
    deposition_rate: f32,
) {
    let n_verts = sphere.vertices.len();

    // Process vertices in order of decreasing elevation so downstream
    // deposition accumulates sediment flux correctly per hop.
    let order = crate::drainage::topological_order(elevations);

    // Per-vertex sediment flux arriving from upstream (m³-per-iteration
    // in m-units × vertex-area; we track as "equivalent height delta
    // worth of sediment at this vertex").
    let mut flux = vec![0.0f32; n_verts];

    for &vi in &order {
        let vi_idx = vi as usize;
        let ds = downstream[vi_idx];

        if elevations[vi_idx] <= sea_level || ds == vi {
            // Ocean vertex or sink — deposit everything arriving here.
            sediment[vi_idx] += flux[vi_idx];
            elevations[vi_idx] += flux[vi_idx];
            flux[vi_idx] = 0.0;
            continue;
        }

        let ds_idx = ds as usize;

        // Slope (dimensionless) to downstream neighbor.
        let dh = (elevations[vi_idx] - elevations[ds_idx]).max(0.0);
        let dist_m = (sphere.vertices[vi_idx] - sphere.vertices[ds_idx]).length() * radius_m;
        let slope = if dist_m > 0.0 { dh / dist_m } else { 0.0 };

        let area = accumulation[vi_idx];
        let erosion = (k * area.powf(m) * slope.powf(n)).min(max_erosion_m);

        // Ensure we don't erode below the downstream neighbor (keeps
        // drainage monotone; Priority-Flood guarantees this at the
        // start of the iteration but erosion could invert).
        let erosion = erosion.min(dh * 0.5);

        elevations[vi_idx] -= erosion;
        let incoming = flux[vi_idx];
        // Total material moving through this vertex this iteration.
        let total_through = incoming + erosion;
        let deposit_here = incoming * deposition_rate;

        // Apply deposition: raise elevation, record sediment.
        elevations[vi_idx] += deposit_here;
        sediment[vi_idx] += deposit_here;

        // Pass remaining flux downstream.
        flux[ds_idx] += total_through - deposit_here;
        flux[vi_idx] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Sea-level selection
// ---------------------------------------------------------------------------

/// Elevation at the (1 − target_ocean_fraction) percentile. Each
/// vertex is an approximately-equal area, so this is a good proxy for
/// "pick the elevation threshold that submerges the desired fraction
/// of the surface."
fn pick_sea_level(elevations: &[f32], target_ocean_fraction: f32) -> f32 {
    let mut sorted: Vec<f32> = elevations.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let target_idx = ((target_ocean_fraction.clamp(0.0, 1.0)) * sorted.len() as f32) as usize;
    let idx = target_idx.min(sorted.len() - 1);
    sorted[idx]
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

/// Hypsometric albedo with rivers drawn as great-circle arc ribbons.
///
/// For each cubemap texel, find the triangle containing its direction
/// and test each of the triangle's three drainage edges. If the texel
/// is within `river_half_width_rad` perpendicular distance of an edge
/// whose upstream accumulation exceeds the threshold, paint it river-
/// blue. This draws rivers as continuous ribbons of fixed physical
/// width rather than per-vertex hex dots that follow mesh topology.
fn paint_hydrology_debug_albedo(
    builder: &mut BodyBuilder,
    river_threshold_m2: f32,
    river_half_width_rad: f32,
) {
    let sphere = builder.sphere.as_ref().unwrap();
    let elevations = builder.vertex_elevations_m.as_ref().unwrap();
    let drainage = builder.drainage_graph.as_ref().unwrap();
    let sea_level = builder.sea_level_m.unwrap_or(0.0);
    let res = builder.cubemap_resolution;
    let inv = 1.0 / res as f32;

    // Ocean depth still uses percentile: the deep-abyss to
    // shallow-shelf ramp is distribution-relative, and seabed
    // variation we want to show is the full range of oceanic
    // elevations.
    let mut ocean_elevs: Vec<f32> = elevations
        .iter()
        .copied()
        .filter(|&e| e <= sea_level)
        .collect();
    ocean_elevs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (ocean_low, ocean_high) = if ocean_elevs.is_empty() {
        (sea_level - 1.0, sea_level)
    } else {
        (ocean_elevs[ocean_elevs.len() / 20], sea_level)
    };

    let accumulation = &drainage.accumulation_m2;
    let downstream = &drainage.downstream;

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

                    // Base terrain color. For ocean cells this paints the
                    // SEABED (dark abyss basalt → lighter continental-shelf
                    // sediment), not the water column — the impostor shader
                    // composes the water BRDF on top, so the baked albedo
                    // under water is effectively "what you'd see if there
                    // were no water", which is the ocean floor.
                    //
                    // Land palette is keyed to *absolute* elevation above
                    // sea level, not percentile — so every elevation band
                    // always gets the same color and the distribution
                    // doesn't collapse the ramp when most vertices cluster
                    // near the bias. Stops (m above sea level):
                    //   0     → dark green (coastal wetland)
                    //   1000  → olive (plains / savanna)
                    //   2500  → tan (hills)
                    //   4000  → warm brown (mountain flanks)
                    //   6000+ → pale rocky gray (peaks)
                    let (br, bg, bb) = if e > sea_level {
                        let h = (e - sea_level).max(0.0);
                        hypsometric_land_color(h)
                    } else {
                        let t = ((e - ocean_low) / (ocean_high - ocean_low)).clamp(0.0, 1.0);
                        (0.06 + 0.18 * t, 0.05 + 0.16 * t, 0.04 + 0.13 * t)
                    };

                    // River overlay: render as arc-distance to each of
                    // the triangle's 3 drainage edges. A drainage edge
                    // v → downstream[v] is a great-circle arc between
                    // the two vertex positions; we paint river-blue
                    // where `dir` is within `river_half_width_rad` of
                    // that arc AND `v`'s accumulation is over the
                    // threshold. This draws rivers as continuous
                    // ribbons at the correct width regardless of mesh
                    // orientation, instead of per-vertex hex dots.
                    let mut river_intensity = 0.0f32;
                    if e > sea_level {
                        for i in 0..3 {
                            let vi = tri[i] as usize;
                            let ds = downstream[vi] as usize;
                            if ds == vi {
                                continue; // sink — no edge
                            }
                            if accumulation[vi] < river_threshold_m2 {
                                continue;
                            }
                            let va = sphere.vertices[vi];
                            let vb = sphere.vertices[ds];
                            let arc_len = va.dot(vb).clamp(-1.0, 1.0).acos();
                            let ang_a = dir.dot(va).clamp(-1.0, 1.0).acos();
                            let ang_b = dir.dot(vb).clamp(-1.0, 1.0).acos();
                            // (ang_a + ang_b − arc_len) ≥ 0, and equals 0
                            // when `dir` lies on the arc. Off-arc,
                            // ≈ perpendicular distance + small end-cap
                            // distance. Clamping against 2× the half-
                            // width captures both "close to the arc"
                            // and "close to one of the endpoints."
                            let off_arc = (ang_a + ang_b - arc_len).max(0.0);
                            if off_arc >= 2.0 * river_half_width_rad {
                                continue;
                            }
                            let k = 1.0 - off_arc / (2.0 * river_half_width_rad);
                            // Smooth the edge of the ribbon and scale
                            // by accumulation so larger rivers read as
                            // darker than small tributaries.
                            let smooth = k * k * (3.0 - 2.0 * k);
                            let accum_factor =
                                ((accumulation[vi].ln() - river_threshold_m2.max(1.0).ln()) / 2.0)
                                    .clamp(0.3, 1.0);
                            river_intensity = river_intensity.max(smooth * accum_factor);
                        }
                    }

                    let (rr, rg, rb) = (0.08, 0.20, 0.55);
                    let (r, g, b) = (
                        br * (1.0 - river_intensity) + rr * river_intensity,
                        bg * (1.0 - river_intensity) + rg * river_intensity,
                        bb * (1.0 - river_intensity) + rb * river_intensity,
                    );

                    *texel = [r, g, b, 1.0];
                }
            });
    }
}

/// Hypsometric land color keyed to absolute elevation above sea level.
/// Stop anchors (meters):
///   0    → dark green (coastal wetland)
///   1000 → olive (plains)
///   2500 → tan (hills)
///   4000 → warm brown (mountain flanks)
///   6000 → pale rocky gray (peaks)
/// Above the last stop, color saturates at the peak stop.
fn hypsometric_land_color(h_m: f32) -> (f32, f32, f32) {
    const STOPS: &[(f32, [f32; 3])] = &[
        (0.0, [0.20, 0.38, 0.18]),
        (1000.0, [0.45, 0.52, 0.26]),
        (2500.0, [0.65, 0.58, 0.35]),
        (4000.0, [0.70, 0.50, 0.32]),
        (6000.0, [0.78, 0.72, 0.58]),
    ];
    if h_m <= STOPS[0].0 {
        let c = STOPS[0].1;
        return (c[0], c[1], c[2]);
    }
    for pair in STOPS.windows(2) {
        let (lo_h, lo_c) = (pair[0].0, pair[0].1);
        let (hi_h, hi_c) = (pair[1].0, pair[1].1);
        if h_m <= hi_h {
            let s = (h_m - lo_h) / (hi_h - lo_h);
            return (
                lo_c[0] + (hi_c[0] - lo_c[0]) * s,
                lo_c[1] + (hi_c[1] - lo_c[1]) * s,
                lo_c[2] + (hi_c[2] - lo_c[2]) * s,
            );
        }
    }
    let c = STOPS.last().unwrap().1;
    (c[0], c[1], c[2])
}

// Keep the Vec3 import used (the Stream-power math doesn't reference
// it directly but bake_elevation_to_cubemap's face_uv_to_dir calls
// rely on the CubemapFace enum only). Sphere.vertices indexing is
// the consumer.
#[allow(dead_code)]
fn _vec3_sentinel() -> Vec3 {
    Vec3::ZERO
}
