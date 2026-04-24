//! Drainage graph produced by Stage 3 (HydrologicalCarving).
//!
//! Every icosphere vertex has exactly one downstream pointer (single-
//! flow D8 routing). Sinks — ocean vertices and land vertices that
//! couldn't drain anywhere after pit-filling — point to themselves.
//! Accumulated upstream catchment area is stored alongside in m², so
//! the renderer and gameplay can classify rivers by flow magnitude
//! (a river is a downstream chain of vertices with accumulation above
//! some threshold).
//!
//! This is a *persistent* output — the spec calls out navigable rivers
//! and settlements at confluences as gameplay consumers, so the graph
//! must survive to `BodyData`, not just be Stage 3's scratch state.

use serde::{Deserialize, Serialize};

/// Per-vertex drainage graph on the icosphere.
///
/// Indexed 1:1 with the sphere vertex array. Sinks are encoded as a
/// self-reference: `downstream[i] == i as u32` means vertex `i` has no
/// downstream neighbor (ocean floor, closed-basin lake bed, or pit the
/// filler couldn't resolve).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrainageGraph {
    /// `downstream[i]` is the neighbor vertex that water at `i` flows
    /// toward. `downstream[i] == i` flags a sink.
    pub downstream: Vec<u32>,
    /// `accumulation[i]` is the upstream catchment area in m² — the
    /// total area of every vertex whose eventual drainage path passes
    /// through `i`, including `i` itself.
    pub accumulation_m2: Vec<f32>,
}

impl DrainageGraph {
    pub fn is_sink(&self, vi: u32) -> bool {
        self.downstream[vi as usize] == vi
    }
}

/// Sort vertex indices by decreasing elevation. A valid topological
/// order for a single-flow drainage tree: each vertex's downstream
/// neighbor is strictly lower (or equal, at sinks), so processing in
/// this order guarantees every upstream contributor is visited before
/// its receiver.
pub(crate) fn topological_order(elevations: &[f32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..elevations.len() as u32).collect();
    order.sort_by(|&a, &b| {
        elevations[b as usize]
            .partial_cmp(&elevations[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Walk every non-sink vertex in topological order, calling `f(from, to)`
/// for each upstream→downstream vertex pair. The caller owns whatever
/// flux vectors are being propagated; this helper only provides the
/// traversal.
pub(crate) fn accumulate_downstream<F: FnMut(usize, usize)>(
    elevations: &[f32],
    downstream: &[u32],
    mut f: F,
) {
    for vi in topological_order(elevations) {
        let ds = downstream[vi as usize];
        if ds != vi {
            f(vi as usize, ds as usize);
        }
    }
}
