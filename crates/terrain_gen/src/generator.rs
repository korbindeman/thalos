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
use crate::stages::{
    Biomes, Climate, CoarseElevation, Cratering, Differentiate, HydrologicalCarving, MareFlood,
    Megabasin, OrogenDla, PaintBiomes, Plates, Regolith, Scarps, SpaceWeather, SurfaceMaterials,
    TectonicSkeleton, Tectonics, Topography,
};
use crate::types::Composition;

/// One stage in a body's pipeline. Variants are 1:1 with stage param structs.
///
/// Sphere is implicit (every body in scope is spherical) and is not listed.
#[derive(Debug, Clone, Deserialize)]
pub enum StageDef {
    Differentiate(Differentiate),
    Biomes(Biomes),
    Climate(Climate),
    CoarseElevation(CoarseElevation),
    HydrologicalCarving(HydrologicalCarving),
    Megabasin(Megabasin),
    Cratering(Cratering),
    MareFlood(MareFlood),
    OrogenDla(OrogenDla),
    PaintBiomes(PaintBiomes),
    Plates(Plates),
    Regolith(Regolith),
    Scarps(Scarps),
    SpaceWeather(SpaceWeather),
    SurfaceMaterials(SurfaceMaterials),
    TectonicSkeleton(TectonicSkeleton),
    Tectonics(Tectonics),
    Topography(Topography),
}

impl StageDef {
    pub fn into_stage(self) -> Box<dyn Stage> {
        match self {
            StageDef::Differentiate(s) => Box::new(s),
            StageDef::Biomes(s) => Box::new(s),
            StageDef::Climate(s) => Box::new(s),
            StageDef::CoarseElevation(s) => Box::new(s),
            StageDef::HydrologicalCarving(s) => Box::new(s),
            StageDef::Megabasin(s) => Box::new(s),
            StageDef::Cratering(s) => Box::new(s),
            StageDef::MareFlood(s) => Box::new(s),
            StageDef::OrogenDla(s) => Box::new(s),
            StageDef::PaintBiomes(s) => Box::new(s),
            StageDef::Plates(s) => Box::new(s),
            StageDef::Regolith(s) => Box::new(s),
            StageDef::Scarps(s) => Box::new(s),
            StageDef::SpaceWeather(s) => Box::new(s),
            StageDef::SurfaceMaterials(s) => Box::new(s),
            StageDef::TectonicSkeleton(s) => Box::new(s),
            StageDef::Tectonics(s) => Box::new(s),
            StageDef::Topography(s) => Box::new(s),
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

impl GeneratorParams {
    /// Multiplies `total_count` on every `Cratering` stage by `factor`.
    /// Used by editor/game in dev builds to cut iteration time — the
    /// `Cratering` and `SpaceWeather` stages both scale linearly with
    /// crater count, and those two together dominate the Mira bake.
    ///
    /// Release builds should leave the authored counts alone.
    pub fn scale_crater_count(&mut self, factor: f32) {
        for stage in &mut self.pipeline {
            if let StageDef::Cratering(c) = stage {
                c.total_count = ((c.total_count as f32 * factor).round() as u32).max(1);
            }
        }
    }
}
