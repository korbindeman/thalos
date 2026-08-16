//! Shared per-body terrain flattening state.
//!
//! Gameplay authors flatten regions through this registry; every ground
//! renderer reads the same handle. Keeping the state outside either renderer
//! prevents a renderer selection from changing authored terrain edits.

use std::collections::HashMap;

use bevy::prelude::*;
use thalos_terrain::{FlattenHandle, flatten_handle};
use thalos_world::BodyId;

/// Bodies whose rendered terrain must be rebuilt after the flatten set changes.
#[derive(Resource, Default)]
pub struct TerrainRebuildRequest {
    pub(crate) bodies: std::collections::HashSet<BodyId>,
}

impl TerrainRebuildRequest {
    /// Queue a body for a renderer rebuild. Idempotent within a frame.
    pub fn request(&mut self, body_id: BodyId) {
        self.bodies.insert(body_id);
    }
}

/// Per-body shared flatten handles for local terrain edits such as runways.
///
/// Handles survive renderer residency churn and are created lazily, so writers
/// and readers share the same object regardless of which runs first.
#[derive(Resource, Default)]
pub struct TerrainFlattenRegistry {
    handles: HashMap<BodyId, FlattenHandle>,
}

impl TerrainFlattenRegistry {
    pub fn handle(&mut self, body_id: BodyId) -> FlattenHandle {
        self.handles
            .entry(body_id)
            .or_insert_with(flatten_handle)
            .clone()
    }

    pub fn get(&self, body_id: BodyId) -> Option<&FlattenHandle> {
        self.handles.get(&body_id)
    }
}
