#![recursion_limit = "256"]
//! Backend-generic learned terrain models and diffusion contracts.
//!
//! This crate intentionally has no Bevy dependency. Offline training and a
//! future optional authoring runtime use the same model source; normal gameplay
//! continues to consume immutable terrain packages.

mod diffusion;
mod model;

pub use diffusion::{DiffusionPrediction, DiffusionSchedule, DiffusionScheduleError};
pub use model::{
    AirlessDenoiser, AirlessDenoiserConfig, CONDITION_CHANNELS, TimeConditioning, Upsampling,
};
