//! CPU mirror of the resident tile heights — the tile path's answer to
//! "how high is the *rendered* ground here, and how refined is it?".
//!
//! This is the direct analogue of udlod's
//! [`GpuAtlasHeightMirror`](crate::ground::GpuAtlasHeightMirror), and it exists
//! for the same reason: everything that must *sit on* the ground — surface
//! scatter (grass / trees / rocks), the local-physics collider patch, the
//! camera terrain floor, HUD altitude — has to read the heights the renderer
//! actually meshed, not a freshly evaluated analytic sample.
//!
//! Sampling the canonical [`SurfaceQuery`](thalos_terrain::SurfaceQuery)
//! directly is *not* equivalent: `SurfaceQuery` is band-limited by its `lod_m`
//! argument, and the tile mesher meshes at the tile's own sample spacing
//! (~6 m at the deepest level on a Thalos-scale body). Scatter placement asks
//! for 0.5 m detail, so it would seat plants on octaves the ground mesh does
//! not carry — trees hovering or half-buried by the amplitude of everything
//! between 0.5 m and the tile spacing. Reading the tile grid instead makes
//! placement exact by construction, exactly as the udlod mirror does.
//!
//! The mirror stores each resident tile's **halo grid verbatim** (the provider
//! payload the mesher consumed), so a lookup is a bilinear tap into the same
//! numbers that produced the vertices. ~18 KB per tile; a ground-level
//! resident set (~1,500 tiles) costs ~27 MB.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use bevy::math::{DVec3, Vec3};
use thalos_terrain::{TerrainPatchBasis, TerrainPatchMesh};

use super::{MIN_LEVEL, SurfaceTile, TILE_HALO, TILE_RES, TileKey, face_uv_of_dir};

/// Shared handle — cloned into the height-source seam and the streaming system.
pub type TileHeightMirrorHandle = Arc<RwLock<TileHeightMirror>>;

/// Resident tile heights, keyed by tile.
///
/// Written only by `stream_tile_terrain` (tiles landing / despawning); read by
/// every ground consumer through
/// [`RenderedGround`](crate::ground::RenderedGround).
pub struct TileHeightMirror {
    radius_m: f64,
    max_level: u8,
    /// Halo grids of the resident tiles (`SurfaceTile::grid_side()` squared).
    tiles: HashMap<TileKey, Arc<Vec<f32>>>,
    /// Bumped on every insert / removal, so consumers that cache geometry
    /// derived from these heights (scatter tiles, colliders) can notice the
    /// ground under them shifted. Mirrors `GpuAtlasHeightMirror::revision`.
    revision: u64,
}

impl TileHeightMirror {
    pub fn new(radius_m: f64, max_level: u8) -> Self {
        Self {
            radius_m,
            max_level,
            tiles: HashMap::new(),
            revision: 0,
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn resident_count(&self) -> usize {
        self.tiles.len()
    }

    pub(super) fn insert(&mut self, key: TileKey, heights_m: Arc<Vec<f32>>) {
        self.tiles.insert(key, heights_m);
        self.revision = self.revision.wrapping_add(1);
    }

    pub(super) fn remove(&mut self, key: TileKey) {
        if self.tiles.remove(&key).is_some() {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    /// Drop every resident grid — the root is handing off to another body, and
    /// a `TileKey` is only unique *within* a body, so keeping these would serve
    /// one body's heights under another's coordinates.
    pub(super) fn clear(&mut self) {
        if !self.tiles.is_empty() {
            self.tiles.clear();
            self.revision = self.revision.wrapping_add(1);
        }
    }

    /// Finest resident tile containing `dir`, with `dir`'s in-tile coordinates.
    fn finest_at(&self, dir: Vec3) -> Option<(TileKey, f64, f64)> {
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }
        let (face, u, v) = face_uv_of_dir(dir.as_dvec3());
        for level in (MIN_LEVEL..=self.max_level).rev() {
            let key = TileKey::containing(face, level, u, v);
            if self.tiles.contains_key(&key) {
                let (u0, v0, u1, v1) = key.uv_rect();
                let s = ((u - u0) / (u1 - u0)).clamp(0.0, 1.0);
                let t = ((v - v0) / (v1 - v0)).clamp(0.0, 1.0);
                return Some((key, s, t));
            }
        }
        None
    }

    /// Rendered ground height (m above the reference radius) at `dir`, or
    /// `None` while no tile covers it (caller falls back to the CPU surface).
    pub fn sample_height_m(&self, dir: Vec3) -> Option<f32> {
        let (key, s, t) = self.finest_at(dir)?;
        let grid = self.tiles.get(&key)?;
        bilinear(grid, s, t)
    }

    /// Sample spacing (m/vertex) of the finest resident tile at `dir` — the
    /// tile path's `best_resident_texel_m`. Callers gate detail work on this so
    /// blades and trunks are never seated on a kilometre-coarse tile.
    pub fn best_resident_texel_m(&self, dir: Vec3) -> Option<f32> {
        let (key, _, _) = self.finest_at(dir)?;
        Some(key.sample_spacing_m(self.radius_m) as f32)
    }

    /// Collider mesh cut from the finest resident tile under `center_dir`, one
    /// vertex per grid sample at the tile's native spacing, each placed at the
    /// exact position the mesher used (`dir_at(s, t) * (radius + h)`) — so the
    /// collider coincides with the drawn surface by construction.
    ///
    /// The window is clamped inside one tile's core grid (no cross-tile
    /// stitching), matching the udlod mirror's contract.
    pub fn build_collider_patch(
        &self,
        center_dir: Vec3,
        max_resolution: u32,
    ) -> Option<TerrainPatchMesh> {
        let (key, s, t) = self.finest_at(center_dir)?;
        let grid = self.tiles.get(&key)?;
        let side = SurfaceTile::grid_side();
        if grid.len() < side * side {
            return None;
        }
        let core = TILE_RES as u32;
        let res = max_resolution.clamp(2, core);
        let step = 1.0 / (TILE_RES - 1) as f64;

        // Core-grid sample nearest the requested direction, then a square
        // window around it clamped to the core extent `[0, TILE_RES)`.
        let center_i = (s * (TILE_RES - 1) as f64).round() as i64;
        let center_j = (t * (TILE_RES - 1) as f64).round() as i64;
        let half = (res as i64 - 1) / 2;
        let i0 = (center_i - half).clamp(0, core as i64 - res as i64);
        let j0 = (center_j - half).clamp(0, core as i64 - res as i64);

        let n = res as usize;
        let mut vertices_body_m = Vec::with_capacity(n * n);
        for j in 0..res {
            let gj = (j0 + j as i64) as usize;
            for i in 0..res {
                let gi = (i0 + i as i64) as usize;
                let h = grid[(gj + TILE_HALO) * side + (gi + TILE_HALO)] as f64;
                vertices_body_m
                    .push(key.dir_at(gi as f64 * step, gj as f64 * step) * (self.radius_m + h));
            }
        }

        let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 2);
        for j in 0..(res - 1) {
            for i in 0..(res - 1) {
                let a = j * res + i;
                let b = a + 1;
                let c = a + res;
                let d = c + 1;
                indices.push([a, c, b]);
                indices.push([b, c, d]);
            }
        }

        let center_index = (res / 2) as usize * n + (res / 2) as usize;
        let center_surface_body_m = vertices_body_m[center_index];
        // `res >= 2`, so the first two vertices (adjacent in i) always exist.
        let texel_spacing_m = (vertices_body_m[1] - vertices_body_m[0]).length();
        let half_extent_m = texel_spacing_m * (res as f64 - 1.0) * 0.5;

        Some(TerrainPatchMesh {
            vertices_body_m,
            indices,
            center_surface_body_m,
            basis: TerrainPatchBasis::from_normal(center_dir.normalize_or_zero().as_dvec3()),
            half_extent_m,
        })
    }

    /// Body-fixed position of the rendered surface at `dir`, or `None` while
    /// nothing is resident there.
    pub fn surface_position(&self, dir: DVec3) -> Option<DVec3> {
        let h = self.sample_height_m(dir.as_vec3())? as f64;
        Some(dir.normalize() * (self.radius_m + h))
    }
}

/// Bilinear tap of a tile's core grid at in-tile `(s, t) ∈ [0, 1]²`. The core
/// grid starts at `TILE_HALO` in both axes (see `build_tile_mesh`, which reads
/// the same offsets), so this returns exactly the mesher's vertex height at
/// grid points and the triangle-plane value between them to within the
/// bilinear/triangulation difference.
fn bilinear(grid: &[f32], s: f64, t: f64) -> Option<f32> {
    let side = SurfaceTile::grid_side();
    if grid.len() < side * side {
        return None;
    }
    let last = (TILE_RES - 1) as f64;
    let x = (s * last).clamp(0.0, last);
    let y = (t * last).clamp(0.0, last);
    let i0 = x.floor() as usize;
    let j0 = y.floor() as usize;
    let i1 = (i0 + 1).min(TILE_RES - 1);
    let j1 = (j0 + 1).min(TILE_RES - 1);
    let fx = (x - i0 as f64) as f32;
    let fy = (y - j0 as f64) as f32;
    let at = |i: usize, j: usize| grid[(j + TILE_HALO) * side + (i + TILE_HALO)];
    let h0 = at(i0, j0) + (at(i1, j0) - at(i0, j0)) * fx;
    let h1 = at(i0, j1) + (at(i1, j1) - at(i0, j1)) * fx;
    Some(h0 + (h1 - h0) * fy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_grid(h: f32) -> Arc<Vec<f32>> {
        let side = SurfaceTile::grid_side();
        Arc::new(vec![h; side * side])
    }

    #[test]
    fn samples_the_finest_resident_tile() {
        let radius = 4_035_000.0;
        let mut mirror = TileHeightMirror::new(radius, 14);
        let dir = DVec3::new(0.3, 0.5, 0.81).normalize();
        let coarse = TileKey::containing_dir(dir, MIN_LEVEL);
        let fine = TileKey::containing_dir(dir, 10);
        mirror.insert(coarse, flat_grid(100.0));
        assert_eq!(mirror.sample_height_m(dir.as_vec3()), Some(100.0));
        mirror.insert(fine, flat_grid(250.0));
        assert_eq!(mirror.sample_height_m(dir.as_vec3()), Some(250.0));
        // The finer tile also decides the reported refinement.
        let texel = mirror.best_resident_texel_m(dir.as_vec3()).unwrap();
        assert!((texel - fine.sample_spacing_m(radius) as f32).abs() < 1e-3);
        // Removal falls back to the coarse tile, not to nothing.
        mirror.remove(fine);
        assert_eq!(mirror.sample_height_m(dir.as_vec3()), Some(100.0));
    }

    #[test]
    fn empty_mirror_reports_nothing() {
        let mirror = TileHeightMirror::new(4_035_000.0, 14);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        assert_eq!(mirror.sample_height_m(dir), None);
        assert_eq!(mirror.best_resident_texel_m(dir), None);
        assert!(mirror.build_collider_patch(dir, 33).is_none());
    }

    /// The mirror must reproduce the mesher's vertex positions, since the
    /// collider and the drawn ground have to coincide.
    #[test]
    fn collider_patch_lands_on_the_mesh_vertices() {
        let radius = 4_035_000.0;
        let mut mirror = TileHeightMirror::new(radius, 14);
        let dir = DVec3::new(0.1, 0.2, 0.97).normalize();
        let key = TileKey::containing_dir(dir, 12);
        let side = SurfaceTile::grid_side();
        // Ramp in i so a wrong axis order or halo offset shows up.
        let mut grid = vec![0.0f32; side * side];
        for j in 0..side {
            for i in 0..side {
                grid[j * side + i] = i as f32;
            }
        }
        mirror.insert(key, Arc::new(grid));
        let patch = mirror.build_collider_patch(dir.as_vec3(), 5).unwrap();
        assert_eq!(patch.vertices_body_m.len(), 25);
        for v in &patch.vertices_body_m {
            let h = v.length() - radius;
            // Every vertex height must be one of the integer ramp values.
            assert!((h - h.round()).abs() < 1e-3, "height {h} is off-grid");
        }
        // Adjacent-in-i vertices differ by exactly one ramp step.
        let d = (patch.vertices_body_m[1].length() - patch.vertices_body_m[0].length()).abs();
        assert!((d - 1.0).abs() < 1e-3, "ramp step {d}");
    }
}
