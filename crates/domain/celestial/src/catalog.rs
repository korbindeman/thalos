//! The `Universe` catalog — typed layers of celestial sources.
//!
//! Typed storage beats `Vec<Box<dyn Source>>` because each layer can
//! pick its own traversal strategy during rendering (stars: scan all,
//! galaxies: bounding tests, nebulae: ray intersection) without
//! virtual dispatch in the inner loop.

use crate::sources::{Galaxy, NebulaField, Star};

/// A deterministic, regeneratable snapshot of the celestial sphere.
///
/// Generators append to the layers; renderers read them. Nothing in
/// here mutates after generation.
#[derive(Debug, Clone, Default)]
pub struct Universe {
    pub stars: Vec<Star>,
    pub galaxies: Vec<Galaxy>,
    pub nebulae: Vec<NebulaField>,
    pub seed: u64,
}

impl Universe {
    pub fn new(seed: u64) -> Self {
        Self {
            stars: Vec::new(),
            galaxies: Vec::new(),
            nebulae: Vec::new(),
            seed,
        }
    }

    pub fn star_count(&self) -> usize {
        self.stars.len()
    }
}
