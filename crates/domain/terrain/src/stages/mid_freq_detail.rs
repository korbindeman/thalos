//! Mid-frequency detail: the procedural relief layer that lives between
//! the cubemap's continental-scale features (≥ 5 km wavelength) and the
//! runtime tile cascade's local detail (< 600 m wavelength).
//!
//! This module owns the **parameters** and the **runner type alias**.
//! The actual computation does not live here: it's a compute kernel that
//! runs on the GPU during the bake (see `bake_dump/src/gpu.rs`). The
//! pipeline takes a `MidFreqRunner` closure and invokes it once per body,
//! handing it the height accumulator + body radius + these params.
//!
//! Why a closure instead of a stage: the kernel must run on GPU, but
//! `thalos_terrain` is pure Rust with no wgpu dependency. Inversion
//! of control: the *caller* (bake_dump) wires the GPU dispatch, the
//! library only describes the data and the schedule.
//!
//! For consumers that don't want mid-freq detail (planet editor preview,
//! tests, anything before GPU plumbing lands in that consumer), pass
//! `None` to `compile_static_terrain_config` — the pipeline skips the
//! call entirely.

use serde::{Deserialize, Serialize};

use crate::cubemap::Cubemap;

/// Tunables for the mid-frequency detail kernel.
///
/// The cascade evaluates fBm at wavelengths covering the gap between the
/// cubemap's resolution-limited Nyquist (~5 km for a 2048² Thalos
/// cubemap; ~1.5 km for 4096²) and the runtime tile-provider cascade's
/// upper edge (~600 m, where `PipelineTileProvider` engages today). The
/// kernel additively perturbs the height accumulator with this band.
///
/// Defaults are tuned for Thalos's Scottish-hills target. Per-body
/// overrides can come later via the body RON if more bodies need
/// different character.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MidFreqDetailParams {
    /// Wavelength (metres) of octave 0. Octave `k` runs at
    /// `base_wl_m / 2^k` with `lacunarity = 2`.
    pub base_wl_m: f32,
    /// Peak one-sided amplitude (metres) of the noise contribution.
    /// Mid-freq sits between continental scale and local cascade; tune
    /// to taste so ridges read at the right size from low orbit.
    pub noise_amp_m: f32,
    /// Number of fBm octaves to evaluate.
    pub octaves: u32,
    /// fBm persistence (amplitude multiplier per octave). 0.5 is the
    /// textbook default.
    pub persistence: f32,
    /// fBm lacunarity (frequency multiplier per octave). 2.0 is the
    /// textbook default — keep it 2 so octave indices line up cleanly
    /// with the runtime cascade.
    pub lacunarity: f32,
    /// Hash seed. Held distinct from the body generator's seed so
    /// changing terrain gen doesn't reshuffle mid-freq detail and vice
    /// versa.
    pub seed: u32,
    /// Fade-in distance (metres) above sea level over which the noise
    /// amplitude ramps from 0 to full. Keeps coastlines clean — we
    /// don't want fBm pushing the surface above sea level mid-ocean.
    pub sea_fade_m: f32,
}

impl Default for MidFreqDetailParams {
    fn default() -> Self {
        // First-cut tuning. `base_wl_m = 30_000` keeps octave 0 clearly
        // above the 512² preview's Nyquist (~20 km on a ~3000 km body)
        // so the cascade is visible in `--preview` PNGs; with
        // `lacunarity = 2` and 4 octaves we reach 1.9 km — still well
        // above where the runtime cascade picks up. Tune down once the
        // full-resolution pipeline is producing the look we want.
        Self {
            base_wl_m: 30_000.0,
            noise_amp_m: 800.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
            seed: 0x4D_4D_46_44, // "MMFD"
            sea_fade_m: 200.0,
        }
    }
}

/// Type erased one-shot runner the pipeline calls with the height
/// accumulator. Consumers (bake_dump, eventually the editor) supply a
/// closure that uploads the accumulator to GPU, dispatches the WGSL
/// kernel, and downloads the result.
///
/// `Send` so it can cross rayon worker threads in `bake_dump`'s
/// `par_iter` bake-all loop. `FnOnce` because each body's pipeline calls
/// it exactly once.
pub type MidFreqRunner =
    Box<dyn FnOnce(&mut Cubemap<f32>, f32, &MidFreqDetailParams) -> Result<(), String> + Send>;
