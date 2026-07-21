//! This module contains the implementation of the Uniform Distance-Dependent Level of Detail (UDLOD).
//!
//! This algorithm is responsible for approximating the terrain geometry.
//! Tiny mesh tiles are refined in a tile_tree-like manner — on the **CPU**, in
//! [`TileTree::compute_draw_set`](crate::terrain_data::tile_tree::TileTree::compute_draw_set),
//! because the balanced 2:1 LOD constraint across cube-face seams needs global
//! awareness of the tile set that upstream's per-tile-independent GPU predicate
//! could not provide. The resulting draw set is uploaded and drawn with a single
//! indirect call, morphed together into one continuous surface.

pub mod terrain_bind_group;
pub mod terrain_material;
pub mod terrain_view_bind_group;
