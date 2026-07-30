//! Shared finishing step every part mesher owes its output.
//!
//! One canonical path, so a new part type cannot quietly ship geometry the
//! raytracing scene refuses.

use bevy::mesh::Mesh;
use bevy::mesh::VertexAttributeValues;

/// Give a part mesh the TANGENT attribute that `bevy_solari`'s BLAS gate
/// requires, so the visible mesh can *be* the raytraced mesh.
///
/// Solari accepts a mesh only when its attribute set is exactly
/// `[POSITION, NORMAL, UV_0, TANGENT]` with `Indices::U32` — an equality on the
/// whole sequence, not a subset test — and it skips a mesh that misses
/// **silently**. Part meshers already emit the first three plus U32 indices, so
/// this call is the entire difference between a hull that reflects the world
/// and one that reflects only sky. `thalos_body_render::rt` owns the predicate
/// and the tests that hold both sides to it.
///
/// Sharing the visible mesh means one BLAS and no duplicated geometry. Terrain
/// tiles cannot do this — they carry COLOR and UV_1 for the layer stack — which
/// is why they get a separate RT-only mesh and craft parts do not.
///
/// Call **after** normals are final: mikktspace derives the tangent frame from
/// POSITION, NORMAL and UV_0 together.
///
/// # These tangents are gate filler, and today they are all placeholders
///
/// Part meshers write a constant `[0, 0]` UV_0 — parts are shaded by procedural
/// panel/rivet functions in `ship_part.wgsl`, not by textures — so every UV
/// triangle has zero area and mikktspace has nothing to work from. The
/// placeholder branch below is therefore the *normal* path, not an edge case.
///
/// That is harmless precisely because nothing reads the tangent: Solari
/// consults it only when a hit material carries a normal map, and the hull's
/// RT-side `StandardMaterial` has none. Give parts real UVs and a normal map
/// and this has to be revisited — the tangent frame would go from unused to
/// load-bearing in one step, with no compile error to mark the moment.
pub fn add_raytracing_tangents(mesh: &mut Mesh) {
    if mesh.generate_tangents().is_ok() && tangents_are_finite(mesh) {
        return;
    }
    // No tangent attribute means no BLAS means the part is invisible to
    // reflection rays — a far worse artifact than an arbitrary tangent on a
    // material that never consults one. A non-finite frame is equally
    // disqualifying, and mikktspace can return `Ok` with one.
    let verts = mesh.count_vertices();
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_TANGENT,
        VertexAttributeValues::Float32x4(vec![[1.0, 0.0, 0.0, 1.0]; verts]),
    );
}

/// What every part mesher must produce for its mesh to enter the raytracing
/// scene: exactly `[POSITION, NORMAL, UV_0, TANGENT]`, triangle list, U32
/// indices.
///
/// Deliberately duplicated from `thalos_body_render::rt::is_raytracing_eligible`
/// rather than imported — this crate is a *dependency* of the renderer, and a
/// mesh-format obligation the meshers owe belongs next to the meshers. The two
/// are pinned to each other by an agreement test in `rt.rs`; change one and
/// that test fails.
pub fn is_raytracing_ready(mesh: &Mesh) -> bool {
    use bevy::mesh::{Indices, PrimitiveTopology};

    mesh.primitive_topology() == PrimitiveTopology::TriangleList
        && matches!(mesh.indices(), Some(Indices::U32(..)))
        && mesh.attributes().map(|(attr, _)| attr.id).eq([
            Mesh::ATTRIBUTE_POSITION.id,
            Mesh::ATTRIBUTE_NORMAL.id,
            Mesh::ATTRIBUTE_UV_0.id,
            Mesh::ATTRIBUTE_TANGENT.id,
        ])
}

fn tangents_are_finite(mesh: &Mesh) -> bool {
    match mesh.attribute(Mesh::ATTRIBUTE_TANGENT) {
        Some(VertexAttributeValues::Float32x4(t)) => {
            t.iter().all(|v| v.iter().all(|c| c.is_finite()))
        }
        _ => false,
    }
}
