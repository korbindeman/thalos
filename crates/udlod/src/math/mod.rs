mod coordinate;
mod ellipsoid;
mod terrain_model;

pub use crate::math::{
    coordinate::{Coordinate, TileCoordinate},
    terrain_model::{
        TerrainModel, TerrainModelApproximation, generate_terrain_model_approximation,
    },
};

/// The square of the parameter c of the algebraic sigmoid function, used to convert between uv and st coordinates.
const C_SQR: f64 = 0.87 * 0.87;
