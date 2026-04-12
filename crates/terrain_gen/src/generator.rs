//! Body file → pipeline glue.
//!
//! `GeneratorParams` is the deserialized form of a body file's `generator`
//! block. It carries the body-level fields (seed, composition, age,
//! resolution) and an ordered list of `StageDef`s — each variant maps 1:1
//! to a stage param struct in `crate::stages`.
//!
//! Loaders (the physics parser) deserialize this directly from RON; the
//! editor and game then walk `pipeline` and call `StageDef::into_stage`
//! to construct the runtime `Stage` trait objects.

use serde::Deserialize;

use crate::stage::Stage;
use crate::stages::{Cratering, Differentiate, MareFlood, Megabasin, Regolith, SpaceWeather};
use crate::types::Composition;

/// One stage in a body's pipeline. Variants are 1:1 with stage param structs.
///
/// Sphere is implicit (every body in scope is spherical) and is not listed.
#[derive(Debug, Clone, Deserialize)]
pub enum StageDef {
    Differentiate(Differentiate),
    Megabasin(Megabasin),
    Cratering(Cratering),
    MareFlood(MareFlood),
    Regolith(Regolith),
    SpaceWeather(SpaceWeather),
}

impl StageDef {
    pub fn into_stage(self) -> Box<dyn Stage> {
        match self {
            StageDef::Differentiate(s) => Box::new(s),
            StageDef::Megabasin(s)     => Box::new(s),
            StageDef::Cratering(s)     => Box::new(s),
            StageDef::MareFlood(s)     => Box::new(s),
            StageDef::Regolith(s)      => Box::new(s),
            StageDef::SpaceWeather(s)  => Box::new(s),
        }
    }
}

/// Body-level generator block. Single source of truth for everything the
/// pipeline needs to run a body, parsed directly from a body file.
#[derive(Debug, Clone, Deserialize)]
pub struct GeneratorParams {
    pub seed: u64,
    pub composition: Composition,
    pub cubemap_resolution: u32,
    pub body_age_gyr: f32,
    pub pipeline: Vec<StageDef>,
}
