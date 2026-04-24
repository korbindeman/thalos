//! Tectonic province types — the output of Stage 1 (TectonicSkeleton) and
//! the input skeleton that every subsequent Thalos stage builds on.
//!
//! A province is a typed chunk of the planet's surface representing one
//! major tectonic feature: a craton interior, a suture belt, a rift scar,
//! and so on. Each icosphere vertex belongs to exactly one province, and
//! the province table carries per-province attributes (age, elevation
//! bias) that later stages consume.

use serde::{Deserialize, Serialize};

/// Kind of tectonic province. Drives downstream elevation, material
/// assignment, and erosion behavior.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ProvinceKind {
    /// Ancient stable continental core. Broad, low relief. Most of the
    /// continental surface of an old world.
    Craton,
    /// Worn collisional mountain belt between ancient cratons — the
    /// planet's fossilized Appalachians.
    Suture,
    /// Failed or fossilized rift — linear basin where continents once
    /// pulled apart. Low elevation, often flooded.
    RiftScar,
    /// Eroded volcanic arc associated with fossil subduction. Curved band
    /// parallel to an old suture.
    ArcRemnant,
    /// Active convergent or divergent margin — the rare place on an old
    /// world where sharp young relief is allowed.
    ActiveMargin,
    /// Hotspot track: linear chain of volcanic seamounts or islands
    /// trailing behind a mantle plume. Mostly oceanic.
    HotspotTrack,
    /// Ocean basin — default for everything that isn't continental or
    /// otherwise tagged. Negative elevation bias.
    OceanicBasin,
}

/// One province in the tectonic skeleton.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvinceDef {
    /// Stable ID within this body, referenced by `vertex_provinces`.
    pub id: u32,
    pub kind: ProvinceKind,
    /// Myr since this province was last tectonically active. Older
    /// provinces have more erosion applied downstream.
    pub age_myr: f32,
    /// Target mid-elevation for this province in meters. Stage 2 builds
    /// the coarse elevation field from these biases.
    pub elevation_bias_m: f32,
}

/// Sentinel for "no province assigned". Vertices with this ID should not
/// appear after `TectonicSkeleton` runs; it exists so default-initialized
/// `vertex_provinces` storage is distinguishable from real assignments.
pub const PROVINCE_NONE: u32 = u32::MAX;
