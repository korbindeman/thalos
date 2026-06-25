//! Cube-sphere tile lattice — the one tiling shared by every vegetation layer.
//!
//! A body's surface is tiled on a cube-sphere lattice: six faces, each an
//! `N×N` grid where `N = tiles_per_side`. A tile at a face centre is
//! ~`tile_size_m` across; the cube projection shrinks tiles laterally toward
//! face corners (down to ~1/2 side), so metric extents are computed per tile
//! ([`TileLattice::tile_extents_m`]).
//!
//! Grass ([`crate::ground::vegetation`]), the shrub/tree scatter system
//! ([`crate::ground::scatter`]), and their LOD clipmap rings all key off a
//! [`TileLattice`]. Each LOD ring is a coarser lattice (smaller
//! `tiles_per_side`, larger `tile_size_m`); the math lives here exactly once so
//! the layers cannot drift.

use bevy::math::DVec3;

use crate::ground::rendered_height::TerrainPatchBasis;

/// One tile on a body's cube-sphere lattice. Faces are
/// `0..6 = +X, -X, +Y, -Y, +Z, -Z`; `x, y` index the face's tile grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub face: u8,
    pub x: i64,
    pub y: i64,
}

/// A cube-sphere tiling at a fixed resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileLattice {
    pub tiles_per_side: i64,
}

impl TileLattice {
    /// Lattice sized so a tile at a cube-face centre is ~`tile_size_m` across.
    pub fn for_body(radius_m: f64, tile_size_m: f64) -> Self {
        Self {
            tiles_per_side: tiles_per_side(radius_m, tile_size_m),
        }
    }

    /// Tile containing a body-fixed unit direction.
    pub fn key_of(&self, dir: DVec3) -> TileKey {
        let (face, u, v) = cube_face_uv(dir.normalize());
        let n = self.tiles_per_side as f64;
        let to_index =
            |c: f64| (((c + 1.0) * 0.5 * n).floor() as i64).clamp(0, self.tiles_per_side - 1);
        TileKey {
            face,
            x: to_index(u),
            y: to_index(v),
        }
    }

    /// Centre direction + tangent basis of a tile. Returns `None` for keys
    /// outside the face grid (callers enumerate raw neighbour offsets and
    /// distance-check each candidate).
    pub fn frame(&self, key: TileKey) -> Option<(DVec3, TerrainPatchBasis)> {
        if key.face > 5
            || key.x < 0
            || key.y < 0
            || key.x >= self.tiles_per_side
            || key.y >= self.tiles_per_side
        {
            return None;
        }
        let n = self.tiles_per_side as f64;
        let u = -1.0 + (key.x as f64 + 0.5) * 2.0 / n;
        let v = -1.0 + (key.y as f64 + 0.5) * 2.0 / n;
        let center = cube_dir(key.face, u, v);
        Some((center, TerrainPatchBasis::from_normal(center)))
    }

    /// Face-uv corner span of a tile (`u_lo, u_hi, v_lo, v_hi`).
    pub fn uv_span(&self, key: TileKey) -> (f64, f64, f64, f64) {
        let n = self.tiles_per_side as f64;
        let u_lo = -1.0 + key.x as f64 * 2.0 / n;
        let v_lo = -1.0 + key.y as f64 * 2.0 / n;
        (u_lo, u_lo + 2.0 / n, v_lo, v_lo + 2.0 / n)
    }

    /// Metric lateral extents `(u_span_m, v_span_m)` of a tile, accounting for
    /// cube distortion, so placement density stays uniform per square metre.
    pub fn tile_extents_m(&self, key: TileKey, radius_m: f64) -> (f64, f64) {
        let (u_lo, u_hi, v_lo, v_hi) = self.uv_span(key);
        let u_mid = (u_lo + u_hi) * 0.5;
        let v_mid = (v_lo + v_hi) * 0.5;
        let ext_u =
            (cube_dir(key.face, u_hi, v_mid) - cube_dir(key.face, u_lo, v_mid)).length() * radius_m;
        let ext_v =
            (cube_dir(key.face, u_mid, v_hi) - cube_dir(key.face, u_mid, v_lo)).length() * radius_m;
        (ext_u, ext_v)
    }
}

/// Tiles along one cube-face edge so a centre tile is ~`tile_size_m` across
/// (`u = tan θ`, metric `≈ R·du` at the face centre).
pub fn tiles_per_side(radius_m: f64, tile_size_m: f64) -> i64 {
    ((2.0 * radius_m) / tile_size_m.max(1.0)).ceil().max(1.0) as i64
}

/// Cube face + face uv (each in `[-1, 1]`) of a body-fixed unit direction.
pub fn cube_face_uv(dir: DVec3) -> (u8, f64, f64) {
    let a = dir.abs();
    if a.x >= a.y && a.x >= a.z {
        if dir.x >= 0.0 {
            (0, -dir.z / a.x, dir.y / a.x)
        } else {
            (1, dir.z / a.x, dir.y / a.x)
        }
    } else if a.y >= a.x && a.y >= a.z {
        if dir.y >= 0.0 {
            (2, dir.x / a.y, -dir.z / a.y)
        } else {
            (3, dir.x / a.y, dir.z / a.y)
        }
    } else if dir.z >= 0.0 {
        (4, dir.x / a.z, dir.y / a.z)
    } else {
        (5, -dir.x / a.z, dir.y / a.z)
    }
}

/// Inverse of [`cube_face_uv`]: unit direction of face uv coordinates.
pub fn cube_dir(face: u8, u: f64, v: f64) -> DVec3 {
    let d = match face {
        0 => DVec3::new(1.0, v, -u),
        1 => DVec3::new(-1.0, v, u),
        2 => DVec3::new(u, 1.0, -v),
        3 => DVec3::new(u, -1.0, v),
        4 => DVec3::new(u, v, 1.0),
        _ => DVec3::new(-u, v, -1.0),
    };
    d.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_key_roundtrip() {
        let lattice = TileLattice::for_body(3_186_000.0, 25.0);
        for &dir in &[
            DVec3::new(0.3, 0.8, -0.5).normalize(),
            DVec3::X,
            DVec3::new(-0.6, 0.1, 0.79).normalize(),
        ] {
            let key = lattice.key_of(dir);
            let (center, _) = lattice.frame(key).unwrap();
            assert_eq!(lattice.key_of(center), key);
            let max_angle = 2.0 * 25.0 / 3_186_000.0;
            assert!(center.angle_between(dir) < max_angle);
        }
    }

    #[test]
    fn extents_positive_at_corner() {
        let lattice = TileLattice::for_body(3_186_000.0, 25.0);
        // A tile near a cube-face corner still has positive metric extents.
        let key = TileKey {
            face: 4,
            x: 0,
            y: 0,
        };
        let (eu, ev) = lattice.tile_extents_m(key, 3_186_000.0);
        assert!(eu > 0.0 && ev > 0.0);
    }
}
