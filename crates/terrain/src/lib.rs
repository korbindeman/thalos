#![allow(clippy::too_many_arguments)]

/// WGSL source for the mid-frequency-detail GPU compute kernel.
pub const MID_FREQ_DETAIL_WGSL: &str = include_str!("../shaders/mid_freq_detail.wgsl");

pub mod aeolian;
pub mod aging_oceanic_field;
pub mod biome_mask;
pub mod body_builder;
pub mod cache;
pub mod cold_desert_field;
pub(crate) mod crater_profile;
pub mod cubemap;
pub mod feature_compiler;
pub(crate) mod feature_compositor;
pub mod field_surface;
pub mod generic_terrestrial_field;
pub mod height_generator;
pub mod icosphere;
pub mod noise;
pub mod pipeline;
pub mod procedural;
pub mod query;
pub mod sample;
pub mod seeding;
pub mod spatial_index;
pub mod stage;
pub mod stages;
pub mod static_surface;
pub mod surface_color;
pub mod surface_field;
pub mod tectonics;
pub mod terrain_config;
pub mod types;
pub mod vaelen_field;

pub use aging_oceanic_field::AgingOceanicField;
pub use biome_mask::*;
pub use body_builder::BodyBuilder;
pub use cold_desert_field::*;
pub use cubemap::{Cubemap, CubemapAccumulator, CubemapFace, default_resolution};
pub use feature_compiler::*;
pub use field_surface::FieldSurface;
pub use generic_terrestrial_field::*;
pub use height_generator::*;
pub use icosphere::Icosphere;
pub use procedural::{
    GENERATOR_VERSION, MacroBiome, ProceduralSurface, climate_cold_lift_m, climate_warmth,
};
pub use query::{
    BakedSurface, FlattenHandle, FlattenRegion, FlattenedSurface, Region, SurfaceQuery, SurfaceRef,
    TerrainFlatten, flatten_handle, nearest_flatten, surface_height_m, surface_height_range_m,
    surface_normal, surface_sample,
};
pub use sample::{
    SurfaceSample, apply_dynamic_surface_layers, sample_static_surface, sample_surface,
};
pub use seeding::{Rng, sub_seed};
pub use spatial_index::{FeatureRef, IcoBuckets};
pub use stage::Stage;
pub use stages::*;
pub use static_surface::{PlanetSurface, StaticSurfaceData};
pub use surface_color::*;
pub use surface_field::*;
pub use tectonics::{
    Boundary, BoundaryKind, Plate, PlateId, PlateKind, SphericalMesh, TectonicActivity,
    TectonicConfig, TectonicFields, TectonicSample, TectonicSystem,
};
pub use terrain_config::*;
pub use types::*;
pub use vaelen_field::*;
