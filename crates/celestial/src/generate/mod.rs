//! Procedural catalog generators.
//!
//! Each submodule is a pure function `(&mut Universe, seed)` —
//! analogous to terrain_gen stages. Generators are additive and
//! order-independent where possible, so a caller can mix and match.

pub mod galaxies;
pub mod stars;

use crate::catalog::Universe;

/// Parameters for the default sky build.
#[derive(Debug, Clone)]
pub struct DefaultGenParams {
    pub seed: u64,
    pub star_count: usize,
    pub faint_magnitude_limit: f32,
    pub galaxy_count: usize,
    pub galaxy_faint_magnitude_limit: f32,
}

impl Default for DefaultGenParams {
    fn default() -> Self {
        Self {
            seed: 0x00C0_FFEE_DEAD_u64,
            // Past naked-eye (~6k to mag 6) to give the sky a dusting
            // of sub-pixel background points. The diffuse milky way
            // glow itself still belongs to the (future) nebula layer.
            star_count: 50_000,
            faint_magnitude_limit: 8.5,
            // Far fewer extragalactic sources; most land below 1 px
            // and just add texture, a handful resolve as soft blobs.
            galaxy_count: 2_500,
            galaxy_faint_magnitude_limit: 13.0,
        }
    }
}

/// Convenience: build a `Universe` with the default content layers
/// enabled for the current development phase.
pub fn generate_default(params: &DefaultGenParams) -> Universe {
    let mut universe = Universe::new(params.seed);
    stars::populate(
        &mut universe,
        &stars::StarGenParams {
            seed: params.seed ^ 0x0A57_A1ED,
            count: params.star_count,
            faint_magnitude_limit: params.faint_magnitude_limit,
            bright_magnitude_floor: -1.5,
        },
    );
    galaxies::populate(
        &mut universe,
        &galaxies::GalaxyGenParams {
            seed: params.seed ^ 0x6A1A_0C1E,
            count: params.galaxy_count,
            faint_magnitude_limit: params.galaxy_faint_magnitude_limit,
            bright_magnitude_floor: 4.0,
        },
    );
    universe
}
