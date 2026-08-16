use std::collections::HashSet;

use bevy::math::DVec3;

use super::data::{COVERAGE_LAND, COVERAGE_WATER, TerrainDataset};

const SPLIT_ERROR_PX: f64 = 1.10;
const MERGE_ERROR_PX: f64 = 0.72;
const SPLIT_GRID_PX: f64 = 24.0;
const MERGE_GRID_PX: f64 = 17.0;

pub const TILE_CELLS: usize = 64;
pub const RTIN_TOLERANCE_M: [f32; 7] = [32.0, 16.0, 8.0, 4.0, 2.0, 0.65, 0.12];

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TileKey {
    pub level: u8,
    pub x: u32,
    pub z: u32,
}

impl TileKey {
    pub const ROOT: Self = Self {
        level: 0,
        x: 0,
        z: 0,
    };

    pub fn parent(self) -> Option<Self> {
        (self.level > 0).then_some(Self {
            level: self.level.saturating_sub(1),
            x: self.x / 2,
            z: self.z / 2,
        })
    }

    pub fn children(self) -> [Self; 4] {
        let level = self.level + 1;
        let x = self.x * 2;
        let z = self.z * 2;
        [
            Self { level, x, z },
            Self { level, x: x + 1, z },
            Self { level, x, z: z + 1 },
            Self {
                level,
                x: x + 1,
                z: z + 1,
            },
        ]
    }

    pub fn is_ancestor_of(self, other: Self) -> bool {
        if self.level >= other.level {
            return false;
        }
        let shift = other.level - self.level;
        other.x >> shift == self.x && other.z >> shift == self.z
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgeStitch(pub u8);

impl EdgeStitch {
    pub const SOUTH: u8 = 1;
    pub const NORTH: u8 = 2;
    pub const WEST: u8 = 4;
    pub const EAST: u8 = 8;

    pub fn has(self, edge: u8) -> bool {
        self.0 & edge != 0
    }
}

pub fn select_leaves(
    dataset: &TerrainDataset,
    camera: DVec3,
    focal_length_px: f64,
    previous: &HashSet<TileKey>,
) -> HashSet<TileKey> {
    let mut previous_internal = HashSet::new();
    for leaf in previous {
        let mut cursor = *leaf;
        while let Some(parent) = cursor.parent() {
            previous_internal.insert(parent);
            cursor = parent;
        }
    }

    let max_level = dataset.metadata.quadtree.visual_max_level;
    let coast_level = max_level;
    let mut leaves = HashSet::new();
    let mut stack = vec![TileKey::ROOT];
    while let Some(key) = stack.pop() {
        let coverage = dataset.quadtree_coverage(key.level, key.x, key.z);
        if coverage & COVERAGE_LAND == 0 {
            continue;
        }
        let was_split = previous_internal.contains(&key);
        let bounds = dataset.quadtree_bounds(key.level, key.x, key.z);
        let distance = distance_to_node(camera, bounds, dataset.max_height_m());
        let tolerance = RTIN_TOLERANCE_M[key.level.min(6) as usize];
        let error_m = f64::from(dataset.quadtree_error_m(key.level, key.x, key.z) + tolerance);
        let projected_error = error_m * focal_length_px / distance;
        let spacing_m = (bounds[2] - bounds[0]) / TILE_CELLS as f64;
        let projected_spacing = spacing_m * focal_length_px / distance;
        let coast_requires_split = coverage & COVERAGE_WATER != 0 && key.level < coast_level;
        let error_threshold = if was_split {
            MERGE_ERROR_PX
        } else {
            SPLIT_ERROR_PX
        };
        let grid_threshold = if was_split {
            MERGE_GRID_PX
        } else {
            SPLIT_GRID_PX
        };
        let split = key.level < max_level
            && (coast_requires_split
                || projected_error > error_threshold
                || projected_spacing > grid_threshold);
        if split {
            stack.extend(key.children());
        } else {
            leaves.insert(key);
        }
    }

    balance_2_to_1(dataset, &mut leaves);
    leaves
}

pub fn edge_stitch(key: TileKey, leaves: &HashSet<TileKey>) -> EdgeStitch {
    let side = 1i64 << key.level;
    let mut mask = 0;
    for (edge, dx, dz) in [
        (EdgeStitch::NORTH, 0, -1),
        (EdgeStitch::SOUTH, 0, 1),
        (EdgeStitch::WEST, -1, 0),
        (EdgeStitch::EAST, 1, 0),
    ] {
        let x = i64::from(key.x) + dx;
        let z = i64::from(key.z) + dz;
        if x < 0 || z < 0 || x >= side || z >= side {
            continue;
        }
        if let Some(neighbour) = covering_leaf(
            leaves,
            TileKey {
                level: key.level,
                x: x as u32,
                z: z as u32,
            },
        ) && neighbour.level + 1 == key.level
        {
            mask |= edge;
        }
    }
    EdgeStitch(mask)
}

fn balance_2_to_1(dataset: &TerrainDataset, leaves: &mut HashSet<TileKey>) {
    for _ in 0..16 {
        let mut split = HashSet::new();
        for leaf in leaves.iter().copied() {
            let side = 1i64 << leaf.level;
            for (dx, dz) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
                let x = i64::from(leaf.x) + dx;
                let z = i64::from(leaf.z) + dz;
                if x < 0 || z < 0 || x >= side || z >= side {
                    continue;
                }
                let probe = TileKey {
                    level: leaf.level,
                    x: x as u32,
                    z: z as u32,
                };
                if let Some(coarse) = covering_leaf(leaves, probe)
                    && coarse.level + 1 < leaf.level
                {
                    split.insert(coarse);
                }
            }
        }
        if split.is_empty() {
            return;
        }
        for coarse in split {
            leaves.remove(&coarse);
            for child in coarse.children() {
                if dataset.quadtree_coverage(child.level, child.x, child.z) & COVERAGE_LAND != 0 {
                    leaves.insert(child);
                }
            }
        }
    }
    debug_assert!(false, "planar quadtree did not reach 2:1 balance");
}

fn covering_leaf(leaves: &HashSet<TileKey>, mut probe: TileKey) -> Option<TileKey> {
    loop {
        if leaves.contains(&probe) {
            return Some(probe);
        }
        probe = probe.parent()?;
    }
}

fn distance_to_node(camera: DVec3, bounds: [f64; 4], max_height_m: f64) -> f64 {
    let dx = if camera.x < bounds[0] {
        bounds[0] - camera.x
    } else if camera.x > bounds[2] {
        camera.x - bounds[2]
    } else {
        0.0
    };
    let dz = if camera.z < bounds[1] {
        bounds[1] - camera.z
    } else if camera.z > bounds[3] {
        camera.z - bounds[3]
    } else {
        0.0
    };
    let dy = (camera.y - max_height_m).max(1.0);
    dx.hypot(dz).hypot(dy).max(1.0)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn real_dataset_selection_is_balanced() {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        for camera in [
            DVec3::new(-30_000.0, 46_000.0, 40_000.0),
            DVec3::new(-17_000.0, 620.0, -4_000.0),
        ] {
            let leaves = select_leaves(&dataset, camera, 2_100.0, &HashSet::new());
            assert!(!leaves.is_empty());
            assert!(leaves.len() < 1_000);
            for leaf in leaves.iter().copied() {
                let coverage = dataset.quadtree_coverage(leaf.level, leaf.x, leaf.z);
                if coverage == (COVERAGE_LAND | COVERAGE_WATER) {
                    assert_eq!(leaf.level, dataset.metadata.quadtree.visual_max_level);
                }
                let side = 1i64 << leaf.level;
                for (edge, dx, dz) in [
                    (EdgeStitch::NORTH, 0, -1),
                    (EdgeStitch::SOUTH, 0, 1),
                    (EdgeStitch::WEST, -1, 0),
                    (EdgeStitch::EAST, 1, 0),
                ] {
                    let x = i64::from(leaf.x) + dx;
                    let z = i64::from(leaf.z) + dz;
                    if x < 0 || z < 0 || x >= side || z >= side {
                        continue;
                    }
                    let probe = TileKey {
                        level: leaf.level,
                        x: x as u32,
                        z: z as u32,
                    };
                    if let Some(neighbour) = covering_leaf(&leaves, probe) {
                        assert!(leaf.level <= neighbour.level + 1);
                        if leaf.level == neighbour.level + 1 {
                            assert!(edge_stitch(leaf, &leaves).has(edge));
                        }
                    }
                }
            }
        }
    }
}
