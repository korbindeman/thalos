//! Raytracing-scene eligibility: what Solari will and will not put in a BLAS.
//!
//! [ADR-20260724T224242Z](../../../../docs/adr/20260724T224242Z-solari-scene-half-not-lighting-half.md)
//! takes Solari's `RaytracingScenePlugin` and never its lighting half, so RT is
//! a ray service inside `thalos::lighting` rather than a second lighting model.
//! This module owns the one thing that decision depends on and that nothing
//! else in the tree checks: **whether a mesh is even visible to the RT scene.**
//!
//! # The gate, and why it is a trap
//!
//! `bevy_solari`'s `is_mesh_raytracing_compatible` (scene/blas.rs) requires:
//!
//! - `PrimitiveTopology::TriangleList`,
//! - `Indices::U32`,
//! - and a vertex attribute set that **exactly equals**
//!   `[POSITION, NORMAL, UV_0, TANGENT]`.
//!
//! That last one is an equality on the whole attribute sequence, not a subset
//! test — an *extra* attribute disqualifies a mesh just as surely as a missing
//! one. `Mesh` stores attributes in a `BTreeMap` keyed by attribute id
//! (POSITION 0, NORMAL 1, UV_0 2, UV_1 3, TANGENT 4, COLOR 5), so the sequence
//! is sorted by id and the gate is well-defined.
//!
//! A mesh that fails is **silently skipped**: no warning, no BLAS, and every
//! ray simply misses it. The failure mode is therefore an image that looks
//! plausible — a hull reflecting sky with no ground in it — and the wrong
//! conclusion ("RT didn't help") drawn from it. That is why eligibility is
//! asserted in tests here rather than discovered on a GPU.
//!
//! # Consequence for the two Thalos surfaces
//!
//! - **Craft parts qualify on the visible mesh.** The shipyard meshers emit
//!   POSITION + NORMAL + UV_0 + U32 indices, so one `generate_tangents()` call
//!   lands them exactly on the gate. The RT proxy can share the visible mesh
//!   handle: one BLAS, no duplicated geometry.
//! - **Terrain cannot share the raster mesh with RT.** Raster entities use a
//!   shared address-only patch mesh and fetch exact positions plus material
//!   channels from an array atlas selected through `MeshTag`. Solari builds a
//!   BLAS from static vertex buffers; it neither executes vertex displacement
//!   nor resolves a per-instance atlas layer. Terrain therefore still needs a
//!   **separate, RT-only mesh asset**, extracted from the same CPU tile payload
//!   so raster and traced geometry agree. Its duplicated geometry cost keeps
//!   near-radius-only RT proxies mandatory.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, MeshVertexAttributeId, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;

/// The attribute id sequence `bevy_solari` requires, in `BTreeMap` order.
///
/// Written as the ids themselves rather than a hand-copied `[0, 1, 2, 4]` so
/// the comparison is exactly the one Solari makes and survives a Bevy bump that
/// renumbers an attribute.
pub const RT_ATTRIBUTE_IDS: [MeshVertexAttributeId; 4] = [
    Mesh::ATTRIBUTE_POSITION.id,
    Mesh::ATTRIBUTE_NORMAL.id,
    Mesh::ATTRIBUTE_UV_0.id,
    Mesh::ATTRIBUTE_TANGENT.id,
];

/// GPU bytes per vertex of an RT-eligible mesh: POSITION 12 + NORMAL 12 +
/// UV_0 8 + TANGENT 16.
///
/// The denominator for any RT-geometry budget: a *count* of proxies looks
/// harmless and silently means gigabytes.
pub const RT_VERTEX_BYTES: usize = 12 + 12 + 8 + 16;

/// Mirror of `bevy_solari::scene::blas::is_mesh_raytracing_compatible`, minus
/// the `Mesh::enable_raytracing` flag (a runtime opt-out, not a property of the
/// geometry).
///
/// This exists so eligibility is a **tested property of our meshers** instead
/// of something discovered as a missing reflection on hardware we may not have
/// in front of us. Keep it in step with the upstream predicate; a Bevy bump
/// that changes the required attribute set must change this and fail its tests.
pub fn is_raytracing_eligible(mesh: &Mesh) -> bool {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return false;
    }
    if !matches!(mesh.indices(), Some(Indices::U32(..))) {
        return false;
    }
    mesh.attributes()
        .map(|(attr, _)| attr.id)
        .eq(RT_ATTRIBUTE_IDS)
}

/// The mesh's vertex attribute *names* in storage order. For assertion
/// messages, where "which attributes does it actually have" is the whole
/// diagnosis and `Vertex_Color` says it faster than an id does.
pub fn attribute_names(mesh: &Mesh) -> Vec<&'static str> {
    mesh.attributes().map(|(attr, _)| attr.name).collect()
}

/// Build the RT-only twin of a terrain tile from the visible mesh's geometry.
///
/// Takes the raster mesh's own arrays rather than re-deriving the surface, so
/// the traced ground is the rasterised ground by construction — a reflection
/// can never disagree with the terrain under it about where the terrain is.
/// Skirts are kept: at LOD boundaries they are what stops a reflection ray
/// slipping through a crack and returning sky from under the surface.
///
/// `tangents` are **gate filler**, not shading data. Solari reads the tangent
/// only when a hit material carries a normal map, and the tile RT proxy's
/// `StandardMaterial` has none; they exist because the attribute set must match
/// exactly. Anyone adding a normal map to that proxy has to revisit them —
/// UV_0 here is a planar body-fixed projection, not a surface parameterisation.
pub fn build_rt_tile_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uv0: Vec<[f32; 2]>,
    indices: Vec<u32>,
) -> Mesh {
    let vert_count = positions.len();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_indices(Indices::U32(indices));

    // mikktspace is the right answer where UV_0 is a real parameterisation and
    // degenerate where it is not — which, on a planar projection, it sometimes
    // is not (a tile face seen edge-on in the projected plane). Fall back to a
    // unit tangent rather than dropping the attribute, because dropping it
    // fails the gate and takes the tile out of the RT scene silently.
    if mesh.generate_tangents().is_err() {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_TANGENT,
            VertexAttributeValues::Float32x4(vec![[1.0, 0.0, 0.0, 1.0]; vert_count]),
        );
    }
    debug_assert!(
        is_raytracing_eligible(&mesh),
        "RT tile mesh missed Solari's gate; attributes are {:?}",
        attribute_names(&mesh)
    );
    mesh
}

/// Convert a built terrain tile's visible mesh into its RT-only twin.
///
/// Returns `None` if the visible mesh does not carry the three attributes to
/// derive from — including after its CPU data has been released to the render
/// world, which is why this must run at tile-build time and not later from
/// `Assets<Mesh>` (tile meshes are `RenderAssetUsages::RENDER_WORLD`).
pub fn rt_twin_of_tile(visible: &Mesh) -> Option<Mesh> {
    let positions = visible.attribute(Mesh::ATTRIBUTE_POSITION)?.as_float3()?;
    let normals = visible.attribute(Mesh::ATTRIBUTE_NORMAL)?.as_float3()?;
    let uv0 = match visible.attribute(Mesh::ATTRIBUTE_UV_0)? {
        VertexAttributeValues::Float32x2(uv) => uv.clone(),
        _ => return None,
    };
    let indices = triangle_list_indices(visible)?;
    Some(build_rt_tile_mesh(
        positions.to_vec(),
        normals.to_vec(),
        uv0,
        indices,
    ))
}

/// Return the mesh's triangles as the `U32` list Solari requires.
///
/// Raster terrain uses a restart-delimited `U16` strip to avoid carrying the
/// same shared vertices three times in its index buffer. The RT scene cannot:
/// Solari's compatibility gate requires both a triangle list and `U32`
/// indices. Expanding here keeps the visible path compact without making the
/// traced surface disagree with it.
fn triangle_list_indices(mesh: &Mesh) -> Option<Vec<u32>> {
    match (mesh.primitive_topology(), mesh.indices()?) {
        (PrimitiveTopology::TriangleList, Indices::U32(indices)) => Some(indices.clone()),
        (PrimitiveTopology::TriangleList, Indices::U16(indices)) => {
            Some(indices.iter().copied().map(u32::from).collect())
        }
        (PrimitiveTopology::TriangleStrip, Indices::U16(indices)) => {
            Some(expand_triangle_strip(indices, u16::MAX))
        }
        (PrimitiveTopology::TriangleStrip, Indices::U32(indices)) => {
            Some(expand_triangle_strip(indices, u32::MAX))
        }
        _ => None,
    }
}

fn expand_triangle_strip<T>(indices: &[T], restart: T) -> Vec<u32>
where
    T: Copy + Eq + Into<u32>,
{
    let mut triangles = Vec::new();
    for strip in indices.split(|index| *index == restart) {
        for (triangle_index, window) in strip.windows(3).enumerate() {
            let [a, b, c] = [window[0].into(), window[1].into(), window[2].into()];
            if a == b || b == c || a == c {
                continue;
            }
            if triangle_index.is_multiple_of(2) {
                triangles.extend_from_slice(&[a, b, c]);
            } else {
                triangles.extend_from_slice(&[b, a, c]);
            }
        }
    }
    triangles
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×2 grid, the smallest thing with a triangle in it.
    fn grid() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let normals = vec![[0.0, 1.0, 0.0]; 4];
        let uv0 = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let indices = vec![0, 2, 1, 1, 2, 3];
        (positions, normals, uv0, indices)
    }

    /// The load-bearing one: what we build for the RT scene actually enters it.
    /// If this fails, rays miss the ground and the image still renders.
    #[test]
    fn rt_tile_mesh_passes_solaris_gate() {
        let (p, n, uv, i) = grid();
        let mesh = build_rt_tile_mesh(p, n, uv, i);
        assert!(
            is_raytracing_eligible(&mesh),
            "attributes {:?}",
            attribute_names(&mesh)
        );
        assert_eq!(
            attribute_names(&mesh),
            [
                "Vertex_Position",
                "Vertex_Normal",
                "Vertex_Uv",
                "Vertex_Tangent"
            ]
        );
    }

    /// The gate is an equality, not a subset test — pin that, because it is the
    /// non-obvious half and the reason tiles need a separate mesh at all.
    #[test]
    fn an_extra_attribute_disqualifies_a_mesh() {
        let (p, n, uv, i) = grid();
        let mut mesh = build_rt_tile_mesh(p, n, uv, i);
        assert!(is_raytracing_eligible(&mesh));
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0, 1.0, 1.0, 1.0]; 4]);
        assert!(
            !is_raytracing_eligible(&mesh),
            "COLOR must disqualify: Solari compares the whole attribute sequence"
        );
    }

    /// U16 indices are as fatal as a wrong attribute set, and just as silent.
    #[test]
    fn u16_indices_disqualify_a_mesh() {
        let (p, n, uv, i) = grid();
        let mut mesh = build_rt_tile_mesh(p, n, uv, i);
        mesh.insert_indices(Indices::U16(vec![0, 2, 1, 1, 2, 3]));
        assert!(!is_raytracing_eligible(&mesh));
    }

    /// Raster strips expand to the exact list Solari expects, including
    /// primitive-restart boundaries resetting strip parity.
    #[test]
    fn rt_twin_expands_restart_delimited_u16_strips() {
        let (p, n, uv, _) = grid();
        let mut raster = Mesh::new(
            PrimitiveTopology::TriangleStrip,
            RenderAssetUsages::RENDER_WORLD,
        );
        raster.insert_attribute(Mesh::ATTRIBUTE_POSITION, p);
        raster.insert_attribute(Mesh::ATTRIBUTE_NORMAL, n);
        raster.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
        raster.insert_indices(Indices::U16(vec![0, 2, 1, u16::MAX, 1, 2, 3]));

        let twin = rt_twin_of_tile(&raster).expect("strip converts");
        assert!(is_raytracing_eligible(&twin));
        assert_eq!(twin.indices(), Some(&Indices::U32(vec![0, 2, 1, 1, 2, 3])));
    }

    /// The shipyard states the same contract in its own words
    /// (`part_mesh::is_raytracing_ready`) because it is a *dependency* of this
    /// crate and cannot import from it. This test is what keeps the two copies
    /// honest — on a mesh that passes and one that does not.
    #[test]
    fn the_shipyard_predicate_agrees_with_this_one() {
        let (p, n, uv, i) = grid();
        let eligible = build_rt_tile_mesh(p, n, uv, i);
        assert!(is_raytracing_eligible(&eligible));
        assert!(thalos_shipyard::part_mesh::is_raytracing_ready(&eligible));

        let mut ineligible = eligible.clone();
        ineligible.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0, 1.0, 1.0, 1.0]; 4]);
        assert!(!is_raytracing_eligible(&ineligible));
        assert!(!thalos_shipyard::part_mesh::is_raytracing_ready(
            &ineligible
        ));
    }

    /// A real craft mesh, through the real shipyard mesher, reaches the RT
    /// scene. The shipyard's own tests assert the same thing locally; this one
    /// asserts it against the predicate the renderer will actually use.
    #[test]
    fn a_shipyard_mesh_passes_this_crates_gate() {
        let cockpit = thalos_shipyard::build_cockpit_mesh(2.0, 3.0);
        assert!(
            is_raytracing_eligible(&cockpit),
            "attributes {:?}",
            attribute_names(&cockpit)
        );
    }

    /// The byte denominator must equal what the mesher uploads, for the reason
    /// `tiles::TILE_MESH_BYTES` records: an under-counting budget reports
    /// headroom that does not exist.
    #[test]
    fn rt_vertex_bytes_matches_the_built_mesh() {
        let (p, n, uv, i) = grid();
        let verts = p.len();
        let mesh = build_rt_tile_mesh(p, n, uv, i);
        assert_eq!(
            mesh.get_vertex_buffer_size(),
            verts * RT_VERTEX_BYTES,
            "RT_VERTEX_BYTES disagrees with the built mesh — an attribute changed"
        );
    }
}
