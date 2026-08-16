//! Topology-independent vegetation appearance payloads.
//!
//! This module owns the procedural woody-species library, foliage atlases, and
//! the shared impostor rendering mechanism. Callers own placement, streaming,
//! spatial precision, and tile topology: Thalos adapts the mechanism to
//! cube-sphere scatter tiles, while Kòrsou adapts it to projected planar cells.

mod atlas;
mod impostor;
mod mesh;

pub use atlas::{
    ATLAS_N, BARK_CELL_COUNT, BARK_CELL_FIRST, GRASS_CARD_VARIANTS, LEAF_CELL_COUNT,
    LEAF_CELL_FIRST, NEEDLE_CELL, atlas_uv, build_foliage_atlas, build_foliage_material_atlas,
    build_grass_card_atlas,
};
pub use impostor::{
    BakeParams, FoliageImpostorBakePlugin, FoliageImpostorExtension, FoliageImpostorMaterial,
    FoliageImpostorMaterialPlugin, IMPOSTOR_MAX_SPECIES, ImpostorAtlas, ImpostorAtlasLayout,
    ImpostorBakeConfig, ImpostorBakeRig, ImpostorInstance, ImpostorParams, ImpostorViewParams,
    TreeBakeMaterial, combine_impostor_mesh, despawn_impostor_bake_rig, foliage_impostor_material,
    hemioct_decode, impostor_bake_rotation, make_impostor_atlas, recenter_tree_mesh,
    spawn_impostor_bake_rig, tree_bounding_sphere,
};
pub use mesh::{CanopyStyle, TreeMeshData, TreeMeshParams, build_tree_mesh, build_tree_mesh_data};
