//! Vegetation scatter system: the shared placement gate plus per-tile instance
//! scatter for shrubs and trees.
//!
//! Grass ([`crate::ground::vegetation`]) bakes thousands of blades into one
//! mesh per tile; shrubs and trees instead resolve to *discrete instances*
//! that the game-side driver realizes through an LOD cascade (mesh LODs →
//! octahedral impostor → terrain albedo). Both share:
//!
//! - the cube-sphere [`TileLattice`](crate::ground::tile_lattice) and
//!   deterministic hashed placement, and
//! - [`placement_gate`], the single definition of *where vegetation can grow*
//!   (the slope/curvature material-mask gate the terrain shader bakes from,
//!   plus the surface normal), so the layers never disagree.
//!
//! This module is pure data + math (no ECS); the per-tile lifecycle, anchoring,
//! and LOD selection live in `thalos_runtime::rendering::vegetation`.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::math::{DVec3, Quat, Vec3};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use thalos_terrain::TerrainFlatten;

use crate::ground::height_source::HeightSource;
use crate::ground::pipeline::material_masks_from_heights;
use crate::ground::rendered_height::TerrainPatchBasis;
use crate::ground::rock_mesh::RockMeshData;
use crate::ground::tile_lattice::{TileKey, TileLattice, cube_dir, tiles_per_side};
use crate::ground::tree_mesh::TreeMeshData;

/// Vegetation clearance above sea level (m): grass blades, GPU grass, and
/// scattered plants/trees all require `height > sea_level +
/// VEG_BEACH_CLEAR_M`, so the beach strand (the +4 m berm face the generator
/// authors — BL-10) stays bare sand instead of growing a lawn to the
/// waterline. Mirrored by the hardcoded gate in `gpu_grass.wgsl` — change
/// together.
pub const VEG_BEACH_CLEAR_M: f32 = 4.0;

/// LOD sample hint for placement height queries — small enough to engage the
/// full near-field cascade, matching what the player stands on.
const PLACEMENT_LOD_M: f32 = 0.5;

/// Finite-difference step for the per-candidate slope/curvature probe. Matches
/// the lower clamp of the tile baker's mask step so the CPU gate reproduces the
/// shader's near-field grass mask.
const MASK_STEP_M: f64 = 2.0;

/// Neighbourhood radius (in cells) the Poisson elimination scans. With a cell
/// size of one `min_spacing`, any two candidates closer than the spacing fall
/// within ±2 cells, so a 5×5 scan catches every conflict.
const POISSON_HALO: i64 = 2;

/// Safety clamp on the Poisson cell grid scanned per tile per axis, against a
/// pathological tiny-spacing / huge-tile combination at a coarse LOD ring.
const MAX_POISSON_CELLS_PER_AXIS: i64 = 512;

/// Hash salts so the per-cell jitter, priority, accept roll, and per-instance
/// variation are independent draws derived from the one global cell key.
const SALT_JITTER_U: u64 = 0x01;
const SALT_JITTER_V: u64 = 0x02;
const SALT_PRIORITY: u64 = 0x03;
const SALT_ACCEPT: u64 = 0x04;
const SALT_YAW: u64 = 0x05;
const SALT_SCALE: u64 = 0x06;
const SALT_TILT: u64 = 0x07;
const SALT_SPECIES: u64 = 0x08;
const SALT_THIN: u64 = 0x09;

/// How far beyond a flatten pad's ramp the forest stays cleared (metres). The
/// tree/shrub density fades from 0 at the pad edge to full over this margin, so
/// the airfield sits in an open clearing instead of a wall of trees.
const VEG_CLEARING_MARGIN_M: f32 = 320.0;

// ---------------------------------------------------------------------------
// Building-terrain scatter regions
// ---------------------------------------------------------------------------

/// How a built footprint on the surface overrides the *natural* scatter
/// (grass / shrubs / trees / rocks). Derived per frame from the structure
/// registry — the spaceport basin, runway strip, launchpads, buildings, tanks —
/// and threaded into the tile builders so a base reads as *managed* ground (a
/// tidy grass lawn between the structures, bare paving under them) instead of
/// either wild meadow or one blanket dead zone.
///
/// This is the seam for "scatter on the building terrain": a footprint declares
/// its surface treatment here, and every scatter layer (grass today, trees and
/// rocks later) honours it through [`classify_scatter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterTreatment {
    /// Suppress all scatter under the footprint: the runway strip, launchpad
    /// slabs, building footprints, tanks — anything paved or built on.
    Clear,
    /// Managed lawn: force a tidy grass cover (the consuming layer picks the
    /// concrete blade profile) and suppress woody scatter. The spaceport's
    /// flattened grassy ground.
    Lawn,
}

/// One building-terrain scatter footprint: a rectangle on the body surface plus
/// how it treats scatter. The rectangle reuses the flatten pad's tangent-plane
/// SDF ([`TerrainFlatten::weight`]) — only the weight is read here, the
/// `elevation_m` the pad carries is irrelevant — so a structure's clearing and
/// its terrain levelling share one geometry definition. A circular structure
/// (launchpad / tank) is represented by its bounding square: a slightly generous
/// clear under a round pad, which reads fine since grass never wants to meet a
/// hard built edge anyway.
#[derive(Debug, Clone, Copy)]
pub struct ScatterRegion {
    pub footprint: TerrainFlatten,
    pub treatment: ScatterTreatment,
}

/// The scatter class resolved at one candidate direction by
/// [`classify_scatter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterClass {
    /// No building footprint here — use the natural placement gates.
    Natural,
    /// Inside a cleared footprint — place nothing.
    Clear,
    /// Inside a lawn footprint — force the consuming layer's managed cover.
    Lawn,
}

/// Weight above which a clearing suppresses scatter — just inside the footprint
/// ramp, so blades never poke at a paved edge.
const SCATTER_CLEAR_W: f64 = 0.05;
/// Weight above which a lawn forces managed cover — well inside the flat pad, so
/// the natural meadow blends in across the lawn's outer ramp rather than ending
/// at a hard line at the basin edge.
const SCATTER_LAWN_W: f64 = 0.5;

/// Classify one body-fixed unit direction against a base's building-terrain
/// scatter regions. A clearing always wins over a lawn (a building sitting on
/// the lawn clears the grass under it), so every region is checked for a
/// covering clear before a lawn can apply. Empty `regions` → [`ScatterClass::Natural`]
/// everywhere, so off-base terrain is unaffected.
pub fn classify_scatter(regions: &[ScatterRegion], dir: DVec3) -> ScatterClass {
    let mut best_lawn = 0.0;
    let mut in_lawn = false;
    for region in regions {
        let w = region.footprint.weight(dir);
        match region.treatment {
            ScatterTreatment::Clear => {
                if w > SCATTER_CLEAR_W {
                    return ScatterClass::Clear;
                }
            }
            ScatterTreatment::Lawn => {
                if w > SCATTER_LAWN_W && w > best_lawn {
                    best_lawn = w;
                    in_lawn = true;
                }
            }
        }
    }
    if in_lawn {
        ScatterClass::Lawn
    } else {
        ScatterClass::Natural
    }
}

// ---------------------------------------------------------------------------
// Shared placement gate
// ---------------------------------------------------------------------------

/// What the shared placement gate resolves at one candidate direction. Callers
/// apply per-species thresholds (slope limit, altitude band, accept
/// probability) on top of this.
#[derive(Debug, Clone, Copy)]
pub struct PlacementSample {
    /// Terrain height, metres above the body's reference radius.
    pub height_m: f32,
    /// Grass material-mask weight in `[0, 1]` (the terrain shader's grass
    /// channel — high on grassland, low on rock/soil).
    pub grass_w: f32,
    /// Slope magnitude `|∇h|` (rise over run).
    pub slope: f32,
    /// Mean terrain curvature `∇²h` over the gate stencil, `1/m`. **Positive in
    /// concave hollows** (the floor sits below its surroundings — moisture
    /// collects, sheltered), **negative on convex ridges/knolls** (exposed). The
    /// woody-plant terrain coupling reads this so forest correlates with the
    /// landform.
    pub curvature: f32,
    /// Body-fixed terrain normal at the sample.
    pub normal_body: DVec3,
}

/// Evaluate the shared vegetation placement gate at body-fixed unit direction
/// `dir`. Returns `None` when any height probe is missing (off the resident
/// atlas) — callers skip the candidate.
///
/// This is the exact slope/curvature/mask/normal block grass placement uses,
/// factored out so grass, shrubs, and trees share one definition.
pub fn placement_gate(
    source: &dyn HeightSource,
    basis: &TerrainPatchBasis,
    dir: DVec3,
    radius_m: f64,
) -> Option<PlacementSample> {
    let height_m = source.sample_height_m(dir.as_vec3(), PLACEMENT_LOD_M)?;

    let p = dir * radius_m;
    let probe =
        |offset: DVec3| source.sample_height_m((p + offset).normalize().as_vec3(), PLACEMENT_LOD_M);
    let (Some(h_l), Some(h_r), Some(h_d), Some(h_u)) = (
        probe(basis.tangent_x * -MASK_STEP_M),
        probe(basis.tangent_x * MASK_STEP_M),
        probe(basis.tangent_z * -MASK_STEP_M),
        probe(basis.tangent_z * MASK_STEP_M),
    ) else {
        return None;
    };

    let masks = material_masks_from_heights(height_m, h_l, h_r, h_d, h_u, MASK_STEP_M as f32);
    let grass_w = masks[0] as f32 / 255.0;

    let grad_x = (h_r - h_l) / (2.0 * MASK_STEP_M as f32);
    let grad_z = (h_u - h_d) / (2.0 * MASK_STEP_M as f32);
    let slope = (grad_x * grad_x + grad_z * grad_z).sqrt();
    // Discrete Laplacian over the 5-point stencil → mean curvature. Positive
    // when the four neighbours sit above the centre (a hollow), negative on a
    // ridge. Free: the four probes above are already in hand.
    let step = MASK_STEP_M as f32;
    let curvature = (h_l + h_r + h_d + h_u - 4.0 * height_m) / (step * step);
    let normal_body = basis
        .local_to_body_vec(DVec3::new(-grad_x as f64, 1.0, -grad_z as f64))
        .normalize();

    Some(PlacementSample {
        height_m,
        grass_w,
        slope,
        curvature,
        normal_body,
    })
}

// ---------------------------------------------------------------------------
// Species + instances
// ---------------------------------------------------------------------------

/// Which vegetation layer a species belongs to — selects the payload and the
/// far end of its LOD cascade (grass → terrain albedo, shrub → fade out, tree
/// → impostor → fold, rock → short near-only fade).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VegLayer {
    GroundCover,
    Shrub,
    Tree,
    /// Scattered pebbles / rocks. Unlike the woody layers, rocks are placed
    /// **inversely** to grass — dense on bare / rocky ground, thinning under
    /// thick grass (which hides them) to a small floor density — and resolve
    /// only up close (no impostor band).
    Rock,
}

/// Placement parameters for one species. Asset-handle-free, so it can be
/// snapshotted into the async scatter build; the game-side `SpeciesLibrary`
/// pairs each entry (by index) with its mesh-LOD chain, material, and impostor.
#[derive(Debug, Clone, Copy)]
pub struct VegSpeciesPlacement {
    pub layer: VegLayer,
    /// Minimum spacing between instances of this species, metres. Placement is
    /// blue-noise (Poisson-disk) on a body-global hashed cell grid at this
    /// spacing: every instance is at least this far from every other of the same
    /// species, so trunks never interpenetrate, while neighbouring canopies can
    /// still touch (pick spacing above a trunk's width but below a canopy's
    /// diameter for connected-but-distinct groves). Density falls out of the
    /// spacing; the gates (slope / mask / altitude / clump) only thin it further.
    pub min_spacing_m: f32,
    /// Relative abundance of this species *within its layer*. All species of one
    /// `VegLayer` share a single Poisson grid (so no two of them ever
    /// interpenetrate, regardless of species); at each grid point the species is
    /// drawn weighted by `mix_weight`. The layer's grid spacing is the largest
    /// `min_spacing_m` among its members.
    pub mix_weight: f32,
    /// Uniform-scale range `(min, max)` applied per instance.
    pub scale_range: (f32, f32),
    /// Reject candidates on terrain steeper than this (rise/run).
    pub slope_limit: f32,
    /// Altitude bands `(lush_lo, lush_hi, fade_lo, fade_hi)` in metres: full
    /// density up to `fade_lo`, smoothstep out to `fade_hi`.
    pub altitude_band: (f32, f32, f32, f32),
    /// Clumping affinity: 0 = uniform scatter, 1 = tight groves.
    pub clump_affinity: f32,
    /// Require the grass material-mask weight to exceed this (trees tolerate a
    /// little soil; grass wants near-pure grassland).
    pub min_grass_w: f32,
}

/// One placed plant, in the tile's local frame. Appearance variation beyond
/// these spatial fields (tint, wind phase) is derived in-shader from the
/// world-space root, so instances of one species stay auto-batchable.
#[derive(Debug, Clone, Copy)]
pub struct VegInstance {
    pub species: u16,
    /// Root position, body-fixed metres, relative to the tile's surface centre.
    pub root_offset_body_m: Vec3,
    /// Local up (body-fixed terrain normal) for orienting the plant.
    pub up_body: Vec3,
    pub yaw: f32,
    pub scale: f32,
    /// Small lean off local-up (radians), for natural variation.
    pub tilt: f32,
}

/// A finished scatter tile: the instances plus the f64 anchor + staleness
/// metadata, mirroring [`GrassTileMesh`](crate::ground::vegetation::GrassTileMesh).
pub struct VegScatterTile {
    pub center_surface_body_m: DVec3,
    pub instances: Vec<VegInstance>,
    /// `HeightSource::revision()` at build time.
    pub built_revision: u64,
    /// Sampled terrain height at the tile centre, for cheap rebuild gating.
    pub center_height_m: f32,
}

/// Everything [`build_scatter_tile`] needs, snapshotted by the driver so the
/// build runs on the async compute pool without touching ECS.
pub struct VegScatterInput {
    pub key: TileKey,
    pub lattice: TileLattice,
    pub radius_m: f64,
    pub height_source: Arc<dyn HeightSource>,
    /// Placement params for every species this layer scatters.
    pub species: Arc<[VegSpeciesPlacement]>,
    pub seed: u64,
    /// Sea level (m above reference radius); plants require
    /// `height > sea_level + VEG_BEACH_CLEAR_M`. Pass `f32::MIN` for bodies without oceans.
    pub sea_level_m: f32,
    /// Active terrain-flatten pad (e.g. the runway); plants are skipped where
    /// the pad has meaningful weight.
    pub flatten_exclusion: Option<TerrainFlatten>,
    /// Poisson-grid coarsening factor for the clipmap (`1.0` = the species'
    /// authored `min_spacing_m`). Coarse far rings pass `> 1` to scatter fewer,
    /// wider-spaced plants — a *different, coarser* grid — so a 2 km impostor tile
    /// holds a bounded instance count. `< 1` is clamped to 1 (never denser than
    /// authored). Rings that must share positions with a finer ring (see
    /// `keep_fraction`) keep this at `1.0` and thin instead.
    pub spacing_scale: f32,
    /// Nested-subset thinning fraction in `[0, 1]` (`1.0` = keep every survivor).
    /// Applied *after* Poisson elimination as a deterministic per-cell hash gate,
    /// so a ring with a smaller fraction renders a strict **subset** of a finer
    /// ring's trees **at the identical positions** — provided both share the grid
    /// (`spacing_scale = 1`). That is what makes the near↔far handoff keep each
    /// tree *in place* (only the density-delta infill fades in on approach)
    /// instead of dissolving one independent grid into another. Evaluated before
    /// the height gate, so a decimated ring doesn't pay the gate for thinned-out
    /// candidates.
    pub keep_fraction: f32,
}

/// One Poisson candidate: the body-global cell `(face, ci, cj)` hashes to a
/// jittered direction on the unit sphere plus a priority. Pure function of the
/// cell — so neighbouring tiles regenerate identical candidates and agree on
/// every elimination, making placement seamless across tile (and most face)
/// boundaries with no stored state.
#[derive(Clone, Copy)]
struct Candidate {
    dir: DVec3,
    u: f64,
    v: f64,
    priority: f64,
}

/// Deterministic candidate for a global Poisson cell. `cells` is the cell count
/// per cube-face edge for this species (`tiles_per_side(radius, spacing)`).
#[inline]
fn cell_candidate(seed: u64, face: u8, ci: i64, cj: i64, cells_f: f64, grid_id: u64) -> Candidate {
    let key = TileKey { face, x: ci, y: cj };
    let ju = veg_hash(seed, key, grid_id, 0, SALT_JITTER_U);
    let jv = veg_hash(seed, key, grid_id, 0, SALT_JITTER_V);
    let priority = veg_hash(seed, key, grid_id, 0, SALT_PRIORITY);
    let u = -1.0 + (ci as f64 + ju) * 2.0 / cells_f;
    let v = -1.0 + (cj as f64 + jv) * 2.0 / cells_f;
    Candidate {
        dir: cube_dir(face, u, v),
        u,
        v,
        priority,
    }
}

/// Does `cell` survive Poisson elimination? It survives unless a strictly
/// higher-priority candidate of the same species lies within `spacing` metres in
/// the ±[`POISSON_HALO`] cell neighbourhood. Ties break on `(cj, ci)` so two
/// candidates can never mutually eliminate. Evaluated identically from any tile.
#[allow(clippy::too_many_arguments)]
fn survives_elimination(
    seed: u64,
    face: u8,
    ci: i64,
    cj: i64,
    cells: i64,
    cells_f: f64,
    grid_id: u64,
    spacing: f64,
    radius_m: f64,
    me: &Candidate,
) -> bool {
    for dj in -POISSON_HALO..=POISSON_HALO {
        for di in -POISSON_HALO..=POISSON_HALO {
            if di == 0 && dj == 0 {
                continue;
            }
            let (nci, ncj) = (ci + di, cj + dj);
            // Cross-face neighbours are skipped (a small seam artifact, as with
            // the rest of the lattice); within-face cells are global.
            if nci < 0 || ncj < 0 || nci >= cells || ncj >= cells {
                continue;
            }
            let other = cell_candidate(seed, face, nci, ncj, cells_f, grid_id);
            let higher = other.priority > me.priority
                || (other.priority == me.priority && (ncj, nci) > (cj, ci));
            if higher && (me.dir - other.dir).length() * radius_m < spacing {
                return false;
            }
        }
    }
    true
}

/// Stable per-layer grid discriminator, so all species of one layer share one
/// Poisson grid (no intra-layer interpenetration, any species mix) while
/// different layers are independent (shrubs may sit under trees as undergrowth).
fn layer_grid_id(layer: VegLayer) -> u64 {
    match layer {
        VegLayer::GroundCover => 0,
        VegLayer::Shrub => 1,
        VegLayer::Tree => 2,
        VegLayer::Rock => 3,
    }
}

/// Build one scatter tile: blue-noise (Poisson-disk) placement on a body-global
/// hashed cell grid — one grid per [`VegLayer`], sized to the layer's largest
/// `min_spacing_m`, with the species at each point drawn by `mix_weight` — then
/// gated by [`placement_gate`] + per-species slope/altitude/clump, with
/// per-instance variation hashed from the candidate. Pure and deterministic for
/// a given input + source state, and seamless across tile boundaries (the
/// candidate set and every elimination are global functions of the cell, not the
/// tile); intended for `AsyncComputeTaskPool`. Returns `None` when no instance
/// passes the gates.
pub fn build_scatter_tile(input: &VegScatterInput) -> Option<VegScatterTile> {
    let (center_dir, basis) = input.lattice.frame(input.key)?;
    let source = input.height_source.as_ref();
    let built_revision = source.revision();

    let center_height_m = source.sample_height_m(center_dir.as_vec3(), PLACEMENT_LOD_M)?;
    let center_surface_body_m = center_dir * (input.radius_m + center_height_m as f64);

    let (u_lo, u_hi, v_lo, v_hi) = input.lattice.uv_span(input.key);
    let face = input.key.face;

    let mut instances = Vec::new();
    for layer in [VegLayer::Shrub, VegLayer::Tree, VegLayer::Rock] {
        // Members of this layer + the combined grid spacing (the widest member,
        // so even the largest canopy never interpenetrates a neighbour) + the
        // total mix weight for the per-point species draw.
        let members: Vec<usize> = input
            .species
            .iter()
            .enumerate()
            .filter(|(_, s)| s.layer == layer)
            .map(|(i, _)| i)
            .collect();
        if members.is_empty() {
            continue;
        }
        let grid_id = layer_grid_id(layer);
        let spacing = members
            .iter()
            .map(|&i| input.species[i].min_spacing_m as f64)
            .fold(0.0_f64, f64::max)
            .max(0.25)
            * input.spacing_scale.max(1.0) as f64;
        let weight_sum: f32 = members
            .iter()
            .map(|&i| input.species[i].mix_weight.max(0.0))
            .sum::<f32>()
            .max(f32::EPSILON);

        // Global cell grid for this layer: one cell ≈ one `spacing` across, so a
        // cell holds at most one survivor and the grid is identical from any tile
        // observing this region.
        let cells = tiles_per_side(input.radius_m, spacing).max(1);
        let cells_f = cells as f64;

        // Cells whose owned region overlaps this tile, plus the elimination halo.
        let to_cell = |c: f64| ((c + 1.0) * 0.5 * cells_f).floor() as i64;
        let ci0 = (to_cell(u_lo) - POISSON_HALO).max(0);
        let ci1 = (to_cell(u_hi) + POISSON_HALO).min(cells - 1);
        let cj0 = (to_cell(v_lo) - POISSON_HALO).max(0);
        let cj1 = (to_cell(v_hi) + POISSON_HALO).min(cells - 1);
        if (ci1 - ci0) > MAX_POISSON_CELLS_PER_AXIS || (cj1 - cj0) > MAX_POISSON_CELLS_PER_AXIS {
            // Spacing far too fine for this tile size — skip rather than stall the
            // async build. (Tuning should keep near-ring tiles well under this.)
            continue;
        }

        for cj in cj0..=cj1 {
            for ci in ci0..=ci1 {
                let cand = cell_candidate(input.seed, face, ci, cj, cells_f, grid_id);

                // A cell is owned by exactly one tile: the one containing its
                // jittered position. Halo cells (outside the span) act only as
                // elimination blockers, so there is no double-placement or gap.
                if !(cand.u >= u_lo && cand.u < u_hi && cand.v >= v_lo && cand.v < v_hi) {
                    continue;
                }
                if !survives_elimination(
                    input.seed,
                    face,
                    ci,
                    cj,
                    cells,
                    cells_f,
                    grid_id,
                    spacing,
                    input.radius_m,
                    &cand,
                ) {
                    continue;
                }

                let dir = cand.dir;
                let key = TileKey { face, x: ci, y: cj };

                // Nested-subset thinning (before the height gate). A pure hash of
                // the *global* cell → a coarse ring (small `keep_fraction`) drops
                // the same cells every finer ring keeps, so its trees are a strict
                // subset at identical positions. The finer ring only *adds* the
                // dropped ("infill") trees on approach; the shared ones never move.
                if input.keep_fraction < 1.0
                    && veg_hash(input.seed, key, grid_id, 1, SALT_THIN)
                        >= input.keep_fraction as f64
                {
                    continue;
                }

                let rng = |salt: u64| veg_hash(input.seed, key, grid_id, 0, salt);

                // Draw the species at this point (weighted), spatially stable.
                let pick = rng(SALT_SPECIES) as f32 * weight_sum;
                let mut acc = 0.0_f32;
                let mut chosen = members[0];
                for &i in &members {
                    acc += input.species[i].mix_weight.max(0.0);
                    if pick < acc {
                        chosen = i;
                        break;
                    }
                }
                let sp = &input.species[chosen];

                if let Some(flatten) = &input.flatten_exclusion
                    && flatten.weight(dir) > 0.05
                {
                    continue;
                }

                let Some(sample) = placement_gate(source, &basis, dir, input.radius_m) else {
                    continue;
                };
                if sample.height_m <= input.sea_level_m + VEG_BEACH_CLEAR_M {
                    continue;
                }
                if sample.slope > sp.slope_limit || sample.grass_w < sp.min_grass_w {
                    continue;
                }

                let alt = altitude_fade(sample.height_m, sp.altitude_band);
                let mut accept = if sp.layer == VegLayer::Rock {
                    // Rocks are placed *inversely* to grass: `bare = 1 - grass_w`
                    // is ~1 on rock/soil, ~0 on lush grassland. They thin to a
                    // small floor density under thick grass (which would hide
                    // them anyway) and fill in on bare patches, gathered into
                    // loose scree clusters by a medium-scale field.
                    let bare = (1.0 - sample.grass_w).clamp(0.0, 1.0);
                    let density = ROCK_GRASS_FLOOR + (1.0 - ROCK_GRASS_FLOOR) * bare;
                    density * alt * rock_scatter_field(dir, sp.clump_affinity)
                } else {
                    // Woody: grass-mask-correlated, forest-clumped, and tied to
                    // the landform — thinner toward ridges/steeps, denser in
                    // sheltered hollows (the real ecotones, not a noise patch).
                    let clump = clump_field(dir, sp.layer, sp.clump_affinity);
                    let terrain = woody_terrain_factor(sp.layer, &sample, sp.slope_limit);
                    // Biome coupling: fold in the macro landcover the ground
                    // already paints, so woody plants thin out on the dry-tan
                    // steppe and vanish on the bare-soil / sand desert (moisture)
                    // and above the latitude-descended treeline (cold lift),
                    // instead of standing in the desert / at the poles. Mirrors
                    // the ground's `vegetation_color` transfer — one world.
                    let sin_lat = dir.y.abs();
                    let eco_altitude_m =
                        sample.height_m + thalos_terrain::climate_cold_lift_m(sin_lat) as f32;
                    let biome =
                        woody_biome_gate(sp.layer, source.landcover_moisture(dir), eco_altitude_m);
                    sample.grass_w * alt * clump * terrain * biome
                };
                // Clearing around a flatten pad (e.g. the runway / spaceport
                // basin): the *forest* fades in over a margin beyond the pad's
                // ramp so trees don't crowd right up to the strip. Measured with
                // the pad's RECTANGULAR exterior distance (the same tangent frame
                // as `TerrainFlatten::weight`), NOT a circular radius off the
                // diagonal — a wide basin (km across) would otherwise clear a
                // multi-km disc that swallows the whole surrounding view, leaving
                // the base ringed by bare ground instead of forest at its edge.
                // Rocks are ground-level gravel — fine right up to the apron — so
                // they only honour the pad itself (the `flatten_exclusion` gate).
                if sp.layer != VegLayer::Rock
                    && let Some(flatten) = &input.flatten_exclusion
                {
                    let offset = (dir - flatten.center_dir) * flatten.radius_m;
                    let along = offset.dot(flatten.tangent_along).abs()
                        - flatten.half_along_m
                        - flatten.ramp_m;
                    let across = offset.dot(flatten.tangent_across).abs()
                        - flatten.half_across_m
                        - flatten.ramp_m;
                    // Exterior distance beyond the ramped rectangle (0 inside it).
                    let d = along.max(0.0).hypot(across.max(0.0)) as f32;
                    accept *= smoothstep(0.0, VEG_CLEARING_MARGIN_M, d);
                }
                // Thinning only ever *removes* points, so the min-spacing
                // guarantee from elimination is preserved.
                if rng(SALT_ACCEPT) as f32 >= accept {
                    continue;
                }

                // Rocks take any orientation (worn stones lying or half-buried at
                // a jaunty angle); woody plants stand near-upright off the normal.
                let tilt_range = if sp.layer == VegLayer::Rock {
                    0.60
                } else {
                    0.12
                };
                let root_body = dir * (input.radius_m + sample.height_m as f64);
                instances.push(VegInstance {
                    species: chosen as u16,
                    root_offset_body_m: (root_body - center_surface_body_m).as_vec3(),
                    up_body: sample.normal_body.as_vec3(),
                    yaw: (rng(SALT_YAW) * std::f64::consts::TAU) as f32,
                    scale: lerp(sp.scale_range.0, sp.scale_range.1, rng(SALT_SCALE) as f32),
                    tilt: (rng(SALT_TILT) * tilt_range) as f32,
                });
            }
        }
    }

    (!instances.is_empty()).then_some(VegScatterTile {
        center_surface_body_m,
        instances,
        built_revision,
        center_height_m,
    })
}

// ---------------------------------------------------------------------------
// Per-tile mesh combine (one batched mesh per tile, grass-style)
// ---------------------------------------------------------------------------

/// Bake all of a tile's instances into ONE mesh — the same one-mesh-per-tile
/// batching the grass uses, so there's no per-tree ECS entity and forests scale
/// to dense/far. Each instance's species mesh `species_lod[inst.species]`
/// (`None` skips it — e.g. shrubs outside the near band) is transformed by the
/// instance (orient to terrain normal, yaw, lean, scale) and appended.
///
/// Vertices are relative to the tile's surface centre (like the grass tile
/// mesh, for f64 anchoring). The tree's **base** (its root offset from the tile
/// centre) is baked into `UV_0.xy` + `UV_1.x` so the shader can scale-fade each
/// tree about its own root and hash a stable per-tree wind/tint seed from it.
/// `COLOR` carries the species tint (`rgb`) + per-vertex wind weight (`a`).
///
/// Returns `None` if nothing was emitted.
pub fn combine_tree_tile_mesh(
    instances: &[VegInstance],
    species_lod: &[Option<Arc<TreeMeshData>>],
) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut uv0: Vec<[f32; 2]> = Vec::new();
    let mut uv1: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for inst in instances {
        let Some(Some(data)) = species_lod.get(inst.species as usize) else {
            continue;
        };
        let rot = Quat::from_rotation_arc(Vec3::Y, inst.up_body.normalize_or(Vec3::Y))
            * Quat::from_rotation_y(inst.yaw)
            * Quat::from_rotation_x(inst.tilt);
        let base = inst.root_offset_body_m; // tree base, tile-centre-relative
        let base_uv0 = [base.x, base.y];
        let start = positions.len() as u32;
        for i in 0..data.positions.len() {
            let p = Vec3::from_array(data.positions[i]);
            let n = Vec3::from_array(data.normals[i]);
            let wp = base + rot * (p * inst.scale);
            positions.push(wp.to_array());
            normals.push((rot * n).normalize_or_zero().to_array());
            colors.push(data.colors[i]);
            uv0.push(base_uv0);
            // UV_1 = (base.z for the per-tree scale-fade/seed, atlas leaf code).
            uv1.push([base.z, data.leaf_code[i]]);
        }
        indices.extend(data.indices.iter().map(|idx| start + idx));
    }

    if positions.is_empty() {
        return None;
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
}

/// Bake all of a tile's **rock** instances into ONE mesh — the same
/// one-mesh-per-tile batching the trees and grass use. Each instance's species
/// mesh `species_lod[inst.species]` (`None` skips it) is oriented to the terrain
/// normal, yawed, tilted, and scaled, then appended.
///
/// Vertices are relative to the tile's surface centre (for f64 anchoring). The
/// stone's **base** (its root offset from the tile centre) is baked into
/// `UV_0.xy` + `UV_1.x` so [`RockMaterial`](crate::ground::RockMaterial) can
/// scale-grow each stone about its own root across the near-band fade. `COLOR`
/// carries the per-vertex stone albedo × baked cavity-AO / top-bleach.
///
/// Returns `None` if nothing was emitted.
pub fn combine_rock_tile_mesh(
    instances: &[VegInstance],
    species_lod: &[Option<Arc<RockMeshData>>],
) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut uv0: Vec<[f32; 2]> = Vec::new();
    let mut uv1: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for inst in instances {
        let Some(Some(data)) = species_lod.get(inst.species as usize) else {
            continue;
        };
        let rot = Quat::from_rotation_arc(Vec3::Y, inst.up_body.normalize_or(Vec3::Y))
            * Quat::from_rotation_y(inst.yaw)
            * Quat::from_rotation_x(inst.tilt);
        let base = inst.root_offset_body_m; // stone base, tile-centre-relative
        let base_uv0 = [base.x, base.y];
        let start = positions.len() as u32;
        for i in 0..data.positions.len() {
            let p = Vec3::from_array(data.positions[i]);
            let n = Vec3::from_array(data.normals[i]);
            let wp = base + rot * (p * inst.scale);
            positions.push(wp.to_array());
            normals.push((rot * n).normalize_or_zero().to_array());
            colors.push(data.colors[i]);
            uv0.push(base_uv0);
            uv1.push([base.z, 0.0]);
        }
        indices.extend(data.indices.iter().map(|idx| start + idx));
    }

    if positions.is_empty() {
        return None;
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
}

/// Bake a tile's **tree** instances into ONE billboard-quad mesh for the
/// octahedral impostor far band: 4 verts + 2 triangles per tree, no canopy
/// geometry. Per vertex the tree base goes in `POSITION` (all four corners
/// share it — degenerate for the standard prepass, expanded into a camera-facing
/// card by `TreeImpostorMaterial`'s vertex shader), the terrain up in `NORMAL`,
/// the card corner in `UV_0`, `(instance scale, atlas species index)` in `UV_1`,
/// and `(tint.rgb, yaw / TAU)` in `COLOR`.
///
/// `atlas_species[species]` maps a placement species index to its atlas layer,
/// or `None` for species without an impostor (shrubs) — those instances are
/// skipped.
///
/// `grove_scale` enlarges every billboard. It is retained as a knob but the
/// driver passes `1.0` (natural size) on **every** ring: the constant-coverage
/// trick of growing a far element to stand in for the clump it replaces is right
/// for grass (a blade is never individually resolvable) but wrong for trees,
/// where a resolvable tree grown `grove×` reads as a giant tree and snaps size at
/// each ring boundary. Far coverage is carried by density (`spacing_scale`) + the
/// forest-tinted terrain albedo instead. `1.0` = natural size; see the
/// `TreeRing` docs in `game::rendering::vegetation`.
///
/// Returns `None` if nothing was emitted.
pub fn combine_impostor_tile_mesh(
    instances: &[VegInstance],
    atlas_species: &[Option<u32>],
    grove_scale: f32,
) -> Option<Mesh> {
    const CORNERS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut uv0: Vec<[f32; 2]> = Vec::new();
    let mut uv1: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for inst in instances {
        let Some(Some(layer)) = atlas_species.get(inst.species as usize) else {
            continue;
        };
        let base = inst.root_offset_body_m.to_array();
        let up = inst.up_body.normalize_or(Vec3::Y).to_array();
        let yaw01 = (inst.yaw / std::f32::consts::TAU).rem_euclid(1.0);
        let start = positions.len() as u32;
        for corner in CORNERS {
            positions.push(base);
            normals.push(up);
            colors.push([1.0, 1.0, 1.0, yaw01]);
            uv0.push(corner);
            uv1.push([inst.scale * grove_scale.max(0.0), *layer as f32]);
        }
        indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    }

    if positions.is_empty() {
        return None;
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
}

// ---------------------------------------------------------------------------
// Clumping field
// ---------------------------------------------------------------------------

/// Angular frequency of the forest-patch mask. At planet radius this sets the
/// patch scale: `~ radius / FREQ` metres per lattice cell (≈ a few hundred m on
/// a ~3000 km body), so groves read as patches the player can walk between — not
/// the near-constant ~100 km blanket a low frequency gives.
const FOREST_PATCH_FREQ: f64 = 8000.0;

/// Forest-stand window over the raw fBM mask — the centre of the ecotone ramp
/// in [`forest_coverage`] (which widens it by ±0.06/0.12 for a soft transition).
/// Deliberately **high**: only the upper part of the noise carries forest, so
/// groves read as *stands over mostly-open ground* (large grassy plains between
/// them) instead of the near-continuous blanket a centred `smoothstep(0.40,
/// 0.60)` gives. Lower `FOREST_LO` for more forest, raise it for emptier plains.
const FOREST_LO: f32 = 0.52;
const FOREST_HI: f32 = 0.72;

/// Raw domain-warped value-noise fBM forest field (≈ `[0, 1]`, centred near
/// 0.5), before the [`FOREST_LO`]/[`FOREST_HI`] contrast. Warp-then-sample so
/// patch edges aren't grid-aligned. Shared by [`forest_coverage`] and
/// [`clump_field`] so they agree on where the groves are.
fn forest_mask_raw(dir: DVec3) -> f32 {
    let warp = fbm(dir * (FOREST_PATCH_FREQ * 0.45), 2) as f64;
    fbm(dir * FOREST_PATCH_FREQ + DVec3::splat(warp * 5.0), 4)
}

/// Glade noise frequency (multiple of [`FOREST_PATCH_FREQ`]): the medium-scale
/// field that opens internal clearings inside a stand. ~130 m glades on a
/// ~3000 km body — clearing-sized, so the canopy interior isn't a solid fill.
const GLADE_FREQ_MUL: f64 = 3.0;

/// Position-only **canopy potential** in `[0, 1]`: how much closed forest a spot
/// would carry from large-scale climate + stand structure alone, *before* the
/// per-sample terrain-form coupling ([`woody_terrain_factor`]). Two scales give
/// a forest that reads naturally instead of as a stamped patch:
///
/// - a **wide ecotone ramp** on the large-scale stand field (no hard edge, no
///   flat plateau) so density feathers in over a broad transition band, and
/// - a **medium-scale glade field** that carves real internal clearings and
///   breaks the uniform interior.
///
/// Shared by tree/shrub/ground-cover placement *and* by the grass driver's
/// far-ring cull (grass is occluded under closed canopy but should return in the
/// glades), so all four agree on where the forest actually is.
pub fn forest_coverage(dir: DVec3) -> f32 {
    let mask = forest_mask_raw(dir);
    // Wide ecotone: the stand ramps in gradually over a broad band around the
    // FOREST_LO/HI window, so edges thin out instead of cutting off.
    let stand = smoothstep(FOREST_LO - 0.06, FOREST_HI + 0.12, mask);
    // Glades: medium-scale dips open clearings within the stand. Closed canopy
    // is the majority (most of the field is above the upper edge), glades the
    // minority — but they break the solid interior and feather the margins.
    let glade = fbm(dir * (FOREST_PATCH_FREQ * GLADE_FREQ_MUL), 3);
    let canopy = smoothstep(0.30, 0.60, glade);
    stand * canopy
}

/// Forest-patch clumping in `[0, 1]`. Built from [`forest_coverage`] (ecotone +
/// internal glades) — noise is legitimate here because this is placement
/// *breakup*, not visible terrain height/albedo (CLAUDE.md's process-first
/// rule). The per-sample terrain coupling (denser hollows, thinner ridges) is
/// applied separately in [`build_scatter_tile`] via [`woody_terrain_factor`],
/// because it needs the height sample this position-only field doesn't see.
/// Shrubs hug the stand margins; ground cover is full in clearings and sparser
/// under closed canopy.
pub fn clump_field(dir: DVec3, layer: VegLayer, affinity: f32) -> f32 {
    let canopy = forest_coverage(dir);
    match layer {
        // affinity 0 = uniform cover everywhere; 1 = patches only (plains truly
        // clear). Trees run near affinity 1 so the plains stay open; the ecotone
        // in `forest_coverage` supplies the gradual plain→forest falloff.
        VegLayer::Tree => lerp(1.0, canopy, affinity),
        VegLayer::Shrub => {
            // Bushes are mostly lone individuals on the open plain, with a thin
            // fringe at the stand margin (ecotone undergrowth) and very few under
            // closed canopy — shade thins the forest floor. The forest-correlated
            // terms (`edge`, `canopy`) are deliberately small so bush density drops
            // considerably around and inside forests rather than crowding them.
            let edge = 1.0 - (canopy - 0.5).abs() * 2.0;
            let lone = fbm(dir * (FOREST_PATCH_FREQ * 2.5), 2);
            lerp(
                lone * 0.35,
                (edge * 0.45 + canopy * 0.12).clamp(0.0, 1.0),
                affinity,
            )
        }
        VegLayer::GroundCover => 1.0 - 0.5 * canopy,
        // Rocks don't use the forest field (they're placed inversely to grass,
        // not correlated with canopy); their clustering is `rock_scatter_field`.
        VegLayer::Rock => 1.0,
    }
}

/// Floor density for rocks under fully lush grass (`grass_w = 1`): a few stones
/// still poke through a meadow, but rocks gather on the bare / rocky patches.
const ROCK_GRASS_FLOOR: f32 = 0.12;

/// Angular frequency of the rock-scatter clustering field. Finer than the forest
/// patches: `~ radius / FREQ` metres per cell ≈ ~50 m scree clusters on a
/// ~3000 km body, so stones gather in loose patches rather than an even sprinkle.
const ROCK_PATCH_FREQ: f64 = 60_000.0;

/// Medium-scale rock-scatter clustering in `[0, 1]`: loose scree patches instead
/// of a uniform sprinkle. `affinity` 0 = uniform everywhere, 1 = tight clusters.
fn rock_scatter_field(dir: DVec3, affinity: f32) -> f32 {
    let f = fbm(dir * ROCK_PATCH_FREQ, 3);
    let patch = smoothstep(0.34, 0.70, f);
    lerp(1.0, patch, affinity.clamp(0.0, 1.0))
}

/// Per-sample terrain-form density factor for **woody** plants (trees, shrubs):
/// the coupling that ties the forest to the landform rather than floating it on
/// noise alone. Returns `1.0` for ground cover (grass ignores landform here).
///
/// - **Slope**: full density on gentle ground, thinning to zero as the slope
///   approaches the species' limit — ridgelines and steep faces open up.
/// - **Curvature**: concave **hollows** (moisture + shelter) get a boost; convex
///   **ridges/knolls** (exposed, thin soil) get cut.
///
/// The product is what produces ecotones that hug terrain: a stand thins as the
/// ground steepens toward a ridge and thickens down in the sheltered hollow,
/// instead of holding one flat interior density across both.
fn woody_terrain_factor(layer: VegLayer, sample: &PlacementSample, slope_limit: f32) -> f32 {
    if layer == VegLayer::GroundCover {
        return 1.0;
    }
    let slope_factor = 1.0 - smoothstep(SLOPE_THIN_FRAC * slope_limit, slope_limit, sample.slope);
    let concave = (sample.curvature * CURVATURE_GAIN).clamp(-3.0, 3.0).tanh();
    let curve_factor = (1.0 + CURVATURE_STRENGTH * concave).clamp(CURVE_FLOOR, CURVE_CEIL);
    (slope_factor * curve_factor).clamp(0.0, 1.0)
}

/// Biome suitability for **woody** plants (trees / shrubs) in `[0, 1]`, folding
/// in the macro landcover the ground already paints so the woody layers stop
/// ignoring climate (the scatter/biome coupling docs/terrain_macro.md §4,
/// TM-P2r, calls for). Two terms, both mirroring the ground's `vegetation_color`
/// transfer, so trees appear where the terrain reads forest / grass and vanish
/// where it reads desert / tundra — one world:
///
/// - **Moisture** — trees follow the forest ↔ grass ↔ dry-tan ↔ bare-soil ramp:
///   full on wet / temperate ground, thinning to scattered savanna on dry
///   steppe, gone on the bare-soil / sand desert. Shrubs tolerate a little more
///   dryness (scrub margins) but still bare out on true desert.
/// - **Treeline** — via the climate-descended eco altitude (`height_m +
///   climate_cold_lift`, which drops with latitude), woody cover stops where the
///   ground greys to alpine / tundra, so the poles read treeless.
///
/// `moisture` is the macro field in `[-1, 1]`
/// ([`HeightSource::landcover_moisture`](crate::ground::height_source::HeightSource::landcover_moisture));
/// `eco_altitude_m = height_m + climate_cold_lift_m(sin_lat)`. Rocks never call
/// this (they are placed inversely to grass and *want* the bare desert).
fn woody_biome_gate(layer: VegLayer, moisture: f32, eco_altitude_m: f32) -> f32 {
    let dryness = (0.5 - 0.5 * moisture).clamp(0.0, 1.0);
    let moist = match layer {
        // Trees: savanna-tolerant, but gone by the ground's bare-soil threshold.
        VegLayer::Tree => smoothstep(0.82, 0.35, dryness),
        // Shrubs push a little further into the dry margin before baring out.
        VegLayer::Shrub => smoothstep(0.92, 0.42, dryness),
        _ => 1.0,
    };
    // 1 below the lush band, ramping to 0 across the treeline the ground greys to
    // alpine at (same constants as the ground palette). The generous low edge
    // leaves mid-altitude temperate ground unaffected; latitude enters through
    // `eco_altitude_m`, so a low-altitude polar site still crosses the treeline.
    let below_treeline = smoothstep(
        crate::ground::landcover::TREELINE_HI_M,
        crate::ground::landcover::LUSH_HI_M,
        eco_altitude_m,
    );
    moist * below_treeline
}

/// Slope at which woody density starts thinning, as a fraction of the species
/// slope limit (below this, gentle ground carries full density).
const SLOPE_THIN_FRAC: f32 = 0.5;
/// Curvature → concavity gain before the `tanh` soft-clamp. Higher = a gentler
/// hollow registers as fully concave. Tuned for the 2 m gate stencil.
const CURVATURE_GAIN: f32 = 12.0;
/// How far curvature pushes density around 1.0 (± this at full concave/convex).
const CURVATURE_STRENGTH: f32 = 0.45;
/// Clamp on the curvature factor (ridge floor / hollow ceiling).
const CURVE_FLOOR: f32 = 0.45;
const CURVE_CEIL: f32 = 1.45;

/// Value-noise fBM over a direction (treated as a 3D position). Cheap,
/// deterministic, no trig; for placement masks only.
fn fbm(p: DVec3, octaves: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 0.5f32;
    let mut freq = 1.0f64;
    let mut norm = 0.0f32;
    for _ in 0..octaves.max(1) {
        sum += amp * value_noise(p * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm.max(f32::EPSILON)
}

/// Trilinearly-interpolated value noise in `[0, 1]`.
fn value_noise(p: DVec3) -> f32 {
    let pf = p.floor();
    let (ix, iy, iz) = (pf.x as i64, pf.y as i64, pf.z as i64);
    let f = p - pf;
    let s = DVec3::new(smooth01(f.x), smooth01(f.y), smooth01(f.z));
    let c = |dx: i64, dy: i64, dz: i64| lattice_value(ix + dx, iy + dy, iz + dz);
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), s.x as f32);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), s.x as f32);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), s.x as f32);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), s.x as f32);
    let y0 = lerp(x00, x10, s.y as f32);
    let y1 = lerp(x01, x11, s.y as f32);
    lerp(y0, y1, s.z as f32)
}

fn lattice_value(ix: i64, iy: i64, iz: i64) -> f32 {
    // u32 integer hash, kept **WGSL-portable** (no u64): the terrain shader
    // replicates this exactly so the far-ground forest/grass-patch tint lines up
    // with where the trees are placed. Mirror in `body_terrain.wgsl`:
    //   var h = bitcast<u32>(cx)*374761393u + bitcast<u32>(cy)*668265263u
    //         + bitcast<u32>(cz)*2246822519u;
    //   h = (h ^ (h >> 13u)) * 1274126177u; h = h ^ (h >> 16u);
    //   return f32(h) * (1.0 / 4294967296.0);
    let x = (ix as i32) as u32;
    let y = (iy as i32) as u32;
    let z = (iz as i32) as u32;
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263))
        .wrapping_add(z.wrapping_mul(2_246_822_519));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    h as f32 / 4_294_967_296.0
}

fn smooth01(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Altitude density factor: full below `fade_lo`, smoothstep to zero by
/// `fade_hi`. The lush band `(lush_lo, lush_hi)` is consumed by appearance
/// (tint) code, not density, so only the fade band gates count here.
fn altitude_fade(h: f32, band: (f32, f32, f32, f32)) -> f32 {
    let (_lush_lo, _lush_hi, fade_lo, fade_hi) = band;
    1.0 - smoothstep(fade_lo, fade_hi, h)
}

/// Integer-mix hash → `[0, 1)`. Deterministic per (seed, tile, species,
/// candidate, salt); same family as the grass blade hash.
fn veg_hash(seed: u64, key: TileKey, species: u64, cand: u64, salt: u64) -> f64 {
    let mut h = seed
        ^ (key.face as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (key.x as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (key.y as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ species.wrapping_mul(0x2545_F491_4F6C_DD1D)
        ^ cand.wrapping_mul(0xD6E8_FEB8_6659_FD93)
        ^ salt.wrapping_mul(0xA24B_AED4_963E_E407);
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0x000F_FFFF_FFFF_FFFF) as f64 / (1u64 << 52) as f64
}

// WGSL-parity smoothstep, safe for descending edges. Never guard the
// denominator with `.max(EPSILON)` — that inverts descending-edge calls into
// a hard step (INC-0005).
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    if denom.abs() < f32::EPSILON {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ground::height_source::ConstantHeightSource;

    const TEST_RADIUS: f64 = 3_186_000.0;
    const TEST_SPACING: f32 = 12.0;

    fn test_species() -> Arc<[VegSpeciesPlacement]> {
        Arc::from(vec![VegSpeciesPlacement {
            layer: VegLayer::Tree,
            min_spacing_m: TEST_SPACING,
            mix_weight: 1.0,
            scale_range: (0.8, 1.4),
            slope_limit: 0.4,
            altitude_band: (1800.0, 2900.0, 2400.0, 3100.0),
            clump_affinity: 0.0, // uniform, so flat ground reliably scatters
            min_grass_w: 0.3,
        }])
    }

    /// World position of an instance, body-fixed metres.
    fn instance_pos(tile: &VegScatterTile, inst: &VegInstance) -> DVec3 {
        tile.center_surface_body_m + inst.root_offset_body_m.as_dvec3()
    }

    #[test]
    fn scatter_is_deterministic_and_flat_ground_grows_trees() {
        let lattice = TileLattice::for_body(3_186_000.0, 250.0);
        let input = VegScatterInput {
            key: lattice.key_of(DVec3::new(0.2, 0.3, 0.93).normalize()),
            lattice,
            radius_m: 3_186_000.0,
            height_source: Arc::new(ConstantHeightSource::new(2000.0)),
            species: test_species(),
            seed: 7,
            sea_level_m: 0.0,
            flatten_exclusion: None,
            spacing_scale: 1.0,
            keep_fraction: 1.0,
        };
        let a = build_scatter_tile(&input).expect("flat land scatters trees");
        let b = build_scatter_tile(&input).expect("flat land scatters trees");
        assert!(!a.instances.is_empty());
        assert_eq!(a.instances.len(), b.instances.len());
        assert_eq!(
            a.instances[0].root_offset_body_m,
            b.instances[0].root_offset_body_m
        );
    }

    #[test]
    fn no_trees_below_sea_level() {
        let lattice = TileLattice::for_body(3_186_000.0, 250.0);
        let input = VegScatterInput {
            key: TileKey {
                face: 4,
                x: 100,
                y: 100,
            },
            lattice,
            radius_m: 3_186_000.0,
            height_source: Arc::new(ConstantHeightSource::new(-50.0)),
            species: test_species(),
            seed: 7,
            sea_level_m: 0.0,
            flatten_exclusion: None,
            spacing_scale: 1.0,
            keep_fraction: 1.0,
        };
        assert!(build_scatter_tile(&input).is_none());
    }

    fn flat_input(lattice: TileLattice, key: TileKey) -> VegScatterInput {
        VegScatterInput {
            key,
            lattice,
            radius_m: TEST_RADIUS,
            height_source: Arc::new(ConstantHeightSource::new(2000.0)),
            species: test_species(),
            seed: 7,
            sea_level_m: 0.0,
            flatten_exclusion: None,
            spacing_scale: 1.0,
            keep_fraction: 1.0,
        }
    }

    /// Blue-noise placement guarantees a minimum separation: no two trees of a
    /// species sit closer than `min_spacing` — the fix for trees growing into
    /// each other.
    #[test]
    fn min_spacing_is_respected_within_a_tile() {
        let lattice = TileLattice::for_body(TEST_RADIUS, 250.0);
        let key = lattice.key_of(DVec3::new(0.2, 0.3, 0.93).normalize());
        let tile = build_scatter_tile(&flat_input(lattice, key)).expect("flat land scatters trees");
        assert!(tile.instances.len() > 20, "expected a populated tile");

        let pts: Vec<DVec3> = tile
            .instances
            .iter()
            .map(|i| instance_pos(&tile, i))
            .collect();
        let mut min_d = f64::INFINITY;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                min_d = min_d.min((pts[i] - pts[j]).length());
            }
        }
        assert!(
            min_d >= TEST_SPACING as f64 * 0.98,
            "min pairwise distance {min_d:.2} m below spacing {TEST_SPACING} m"
        );
    }

    /// Placement is seamless: two adjacent tiles share the body-global cell grid,
    /// so the min-spacing guarantee holds across their boundary too (no clipping
    /// where tiles meet).
    #[test]
    fn placement_is_seamless_across_adjacent_tiles() {
        let lattice = TileLattice::for_body(TEST_RADIUS, 250.0);
        let key = lattice.key_of(DVec3::new(0.2, 0.3, 0.93).normalize());
        let key_right = TileKey {
            x: key.x + 1,
            ..key
        };
        let a = build_scatter_tile(&flat_input(lattice, key)).expect("tile a");
        let b = build_scatter_tile(&flat_input(lattice, key_right)).expect("tile b");

        let mut pts: Vec<DVec3> = a.instances.iter().map(|i| instance_pos(&a, i)).collect();
        pts.extend(b.instances.iter().map(|i| instance_pos(&b, i)));

        // No duplicate placement at the shared edge, and spacing still holds.
        let mut min_d = f64::INFINITY;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                min_d = min_d.min((pts[i] - pts[j]).length());
            }
        }
        assert!(
            min_d >= TEST_SPACING as f64 * 0.98,
            "cross-tile min distance {min_d:.2} m below spacing {TEST_SPACING} m"
        );
    }
}
