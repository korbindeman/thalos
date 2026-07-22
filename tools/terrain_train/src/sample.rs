use serde::{Deserialize, Serialize};

use crate::grid::Grid;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Validation,
    Holdout,
}

#[derive(Clone, Debug, Serialize)]
pub struct Parameters {
    pub crater_density: f32,
    pub mare_fraction: f32,
    pub gardening: f32,
    pub rim_sharpness: f32,
    pub crater_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Provenance {
    pub source_id: String,
    pub split: Split,
    pub metres_per_pixel: f32,
    pub synthetic: bool,
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub height: Grid,
    pub mare_mask: Grid,
    pub parameters: Parameters,
    pub provenance: Provenance,
}

impl Sample {
    pub fn scale_condition(&self) -> f32 {
        ((self.provenance.metres_per_pixel / 250.0).log2() / 4.0).clamp(-1.0, 1.0)
    }
}
