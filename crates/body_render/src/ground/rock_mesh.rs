//! Procedural stylized rock / pebble mesh generation for the scatter layer.
//!
//! Built around the recipe every good procedural-rock generator converges on
//! (Blender's `add_mesh_rocks` / stylized-rock add-ons, the 80.lv Houdini
//! breakdown): **not** a noise-displaced sphere (which reads as a lumpy potato),
//! but a blob **carved by random planes into flat facets of varied size**, with
//! **Worley (cellular) displacement** for angular rock character, then **flat
//! shading** so each facet reads as a clean plane with sharp edges. The pipeline:
//!
//! 1. **Icosphere → ellipsoid** — a subdivided icosphere squashed by a per-rock
//!    ellipsoid (flattened, slightly elongated worn stones).
//! 2. **Form + cells** — a low-frequency gradient-noise warp for an irregular
//!    silhouette, plus a **Worley `F2−F1`** term that bulges each cell and grooves
//!    the boundaries (the "rock crackle"), so the pre-cut blob is already angular.
//! 3. **Plane cuts** — `cuts` random planes (golden-spiral directions + per-rock
//!    jitter) each flatten the cap of the blob beyond them. Varied cut depths give
//!    **facets of varied size** and the plane intersections give **sharp edges**.
//!    The result is a faceted convex-ish polytope — a stylized rock.
//! 4. **Flat shading + baked AO** — per-face normals (coplanar facet triangles
//!    share a normal → one flat face), with a cavity AO from how far each vertex
//!    was carved in, plus a sun-bleached top, baked onto the vertex colour.
//!
//! Authored **+Y up, seated slightly below the origin** so the scatter driver
//! orients +Y to the terrain normal and the stone embeds (no floating). A small
//! library of these is generated once and scattered with per-instance variation
//! (scale, rotation), so all instances of a species share a `(Mesh, Material)`
//! and batch into one mesh per tile. `subdivisions` is the LOD (0=20,1=80,2=320,
//! 3=1280 base tris); rocks resolve only up close, so the chain is short.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::math::Vec3;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};

/// Parameters for one procedurally generated stylized rock / pebble.
#[derive(Debug, Clone, Copy)]
pub struct RockMeshParams {
    /// Mean radius (m) before the per-rock ellipsoid + per-instance scale.
    pub radius_m: f32,
    /// Ellipsoid axis scales `(x, y, z)`. Pebbles flatten in **Y** (e.g.
    /// `(1.0, 0.6, 0.85)`) and elongate slightly laterally.
    pub axes: Vec3,
    /// Low-frequency form-warp amplitude (fraction of radius) — the broad,
    /// asymmetric irregularity of the overall silhouette.
    pub form_amp: f32,
    /// Low-frequency form-warp spatial frequency.
    pub form_freq: f32,
    /// **Worley `F2−F1`** cell-bulge amplitude (fraction of radius): bulges each
    /// cellular zone and grooves the boundaries, the angular "rock crackle".
    pub cell_amp: f32,
    /// Worley cell spatial frequency (≈ number of cells across the rock).
    pub cell_freq: f32,
    /// Number of random **plane cuts** that carve the blob into flat facets. More
    /// cuts = more of the surface is flat planes (more stylized / angular); 0 = an
    /// uncut faceted-noise blob.
    pub cuts: u32,
    /// Cut-plane offset as a fraction of the blob's extent along the cut normal,
    /// `(min, max)`. A shallow cut (→ 1.0) shaves a small facet; a deep cut
    /// (→ lower) carves a big flat face. The spread gives **varied facet sizes**.
    pub cut_depth: (f32, f32),
    /// High-frequency surface-grit amplitude (fraction of radius) added after the
    /// cuts — keep small so the facets stay readable.
    pub detail_amp: f32,
    /// High-frequency surface-grit spatial frequency.
    pub detail_freq: f32,
    /// Base albedo (linear). Light natural stone; a cavity-AO + top-bleach
    /// gradient is baked on top.
    pub color: Vec3,
    /// Deterministic shape seed.
    pub seed: u64,
    /// Icosphere subdivision level: 0 = 20 tris, 1 = 80, 2 = 320, 3 = 1280.
    pub subdivisions: u32,
}

impl Default for RockMeshParams {
    fn default() -> Self {
        Self {
            radius_m: 0.10,
            axes: Vec3::new(1.0, 0.62, 0.84),
            form_amp: 0.22,
            form_freq: 1.4,
            cell_amp: 0.12,
            cell_freq: 3.0,
            cuts: 14,
            cut_depth: (0.70, 0.96),
            detail_amp: 0.018,
            detail_freq: 8.0,
            // Light, faintly warm natural stone (linear). Real limestone/granite
            // is far brighter than soil — a low albedo renders near-black on grass.
            color: Vec3::new(0.48, 0.45, 0.40),
            seed: 0,
            subdivisions: 2,
        }
    }
}

/// Raw CPU mesh arrays for one rock species at one LOD. Kept on the CPU so the
/// scatter driver can *combine* many rocks into one batched per-tile mesh.
/// Always **flat-shaded** (per-face normals), so vertices are unshared (3 per
/// triangle). `colors[i].w` is unused (mirrors the tree mesh's `COLOR` slot).
#[derive(Clone, Default)]
pub struct RockMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

/// Build the raw CPU mesh arrays for one rock species.
pub fn build_rock_mesh_data(params: &RockMeshParams) -> RockMeshData {
    let (dirs, faces) = icosphere(params.subdivisions);

    let seed_form = params.seed as u32 ^ 0x52_4f_43_4b; // "ROCK"
    let seed_cell = (params.seed >> 7) as u32 ^ 0x43_45_4c_4c; // "CELL"
    let seed_detail = (params.seed >> 17) as u32 ^ 0x47_52_49_54; // "GRIT"
    let seed_cut = params.seed ^ 0xC0_07_5E_ED;

    // 1+2. Ellipsoid base + low-freq form warp + Worley cell bulge.
    let mut verts: Vec<Vec3> = dirs
        .iter()
        .map(|d| {
            let form = params.form_amp * fbm3(*d * params.form_freq, 4, seed_form);
            let cell = params.cell_amp * worley_f2_f1(*d * params.cell_freq, seed_cell);
            let r = (1.0 + form + cell).max(0.40);
            (*d * (params.radius_m * r)) * params.axes
        })
        .collect();

    // 3. Plane cuts: each plane flattens the cap of the blob beyond it into a
    // facet. Cut normals are spread over the sphere (golden spiral) + per-rock
    // jitter; depths vary so facets vary in size; intersections give sharp edges.
    for k in 0..params.cuts {
        let d = cut_plane_dir(k, params.cuts, seed_cut);
        // Current max extent along the cut normal (auto-adapts to the ellipsoid).
        let mut proj_max = f32::MIN;
        for v in &verts {
            proj_max = proj_max.max(v.dot(d));
        }
        if proj_max <= 0.0 {
            continue;
        }
        let frac = lerp(
            params.cut_depth.0,
            params.cut_depth.1,
            hash_u01(seed_cut ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) as f32,
        );
        let t = frac * proj_max;
        for v in &mut verts {
            let dp = v.dot(d) - t;
            if dp > 0.0 {
                *v -= d * dp;
            }
        }
    }

    // 4. Small surface grit (kept tiny so facets stay flat-ish).
    if params.detail_amp > 0.0 {
        for (i, v) in verts.iter_mut().enumerate() {
            let n = params.detail_amp
                * params.radius_m
                * fbm3(dirs[i] * params.detail_freq, 2, seed_detail);
            *v += dirs[i] * n;
        }
    }

    // Seat the stone so its lowest point sits a little below the origin (the
    // scatter driver puts the origin on the terrain → the base embeds).
    let min_y = verts.iter().fold(f32::INFINITY, |m, v| m.min(v.y));
    let sink = 0.16 * params.radius_m * params.axes.y;
    let dy = -sink - min_y;
    for v in &mut verts {
        v.y += dy;
    }

    // Cavity AO: how far in each vertex was carved (smaller radial extent = a
    // cut-in face, more occluded → darker). Normalize across the mesh.
    let radii: Vec<f32> = verts.iter().map(|v| v.length()).collect();
    let (mut r_min, mut r_max) = (f32::INFINITY, f32::MIN);
    for &r in &radii {
        r_min = r_min.min(r);
        r_max = r_max.max(r);
    }
    let r_span = (r_max - r_min).max(1.0e-5);

    // Per-rock hue jitter so a field of one species still varies in tone.
    let tint = 0.90 + 0.20 * hash_u01(params.seed ^ 0xA53C_19E7) as f32;

    // Flat shading: each triangle gets its own three vertices + its face normal,
    // so coplanar facet triangles read as one flat face with sharp edges.
    let mut data = RockMeshData::default();
    for f in &faces {
        let (ia, ib, ic) = (f[0] as usize, f[1] as usize, f[2] as usize);
        let (a, b, c) = (verts[ia], verts[ib], verts[ic]);
        let n = (b - a).cross(c - a).normalize_or(Vec3::Y);
        let start = data.positions.len() as u32;
        for &(p, ri) in &[(a, ia), (b, ib), (c, ic)] {
            let ao_t = (radii[ri] - r_min) / r_span;
            push_vert(&mut data, p, n, params.color, ao_t, tint);
        }
        data.indices
            .extend_from_slice(&[start, start + 1, start + 2]);
    }
    data
}

/// Build a standalone `Mesh` for one rock — for the preview and any direct
/// (non-tiled) spawn. Inserts zeroed `UV_0`/`UV_1` so the layout matches
/// [`RockMaterial`](crate::ground::RockMaterial) (whose vertex shader reads the
/// per-rock base from those slots for the scale-grow fade); with a zero base and
/// a full-on fade band, the rock just renders at full size.
pub fn build_rock_mesh(params: &RockMeshParams) -> Mesh {
    let data = build_rock_mesh_data(params);
    let n = data.positions.len();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, data.colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0_f32, 0.0]; n]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.0_f32, 0.0]; n]);
    mesh.insert_indices(Indices::U32(data.indices));
    mesh
}

/// Push one finished vertex with the baked cavity-AO + sun-bleached-top tint.
fn push_vert(data: &mut RockMeshData, pos: Vec3, normal: Vec3, base: Vec3, ao_t: f32, tint: f32) {
    // Carved-in faces (low `ao_t`) darken; the rounded outer bulges stay light.
    let ao = 0.58 + 0.46 * ao_t.clamp(0.0, 1.0);
    // Sun-bleached, dust-settled tops: the upward-facing crown lightens a touch.
    let top = 0.88 + 0.22 * normal.y.clamp(0.0, 1.0);
    let c = base * (ao * top * tint);
    data.positions.push(pos.to_array());
    data.normals.push(normal.normalize_or(Vec3::Y).to_array());
    data.colors.push([c.x, c.y, c.z, 0.0]);
}

/// Direction of plane cut `k` of `cuts`: a golden-spiral point on the sphere
/// (even coverage from all sides) jittered per rock so no two rocks share the
/// same facet layout.
fn cut_plane_dir(k: u32, cuts: u32, seed: u64) -> Vec3 {
    let n = cuts.max(1) as f32;
    let i = k as f32 + 0.5;
    let y = 1.0 - 2.0 * i / n; // -1 → 1
    let r = (1.0 - y * y).max(0.0).sqrt();
    const GOLDEN_ANGLE: f32 = 2.399_963_2;
    let theta = GOLDEN_ANGLE * i;
    let base = Vec3::new(r * theta.cos(), y, r * theta.sin());
    let jitter = Vec3::new(
        hash_u01(seed ^ (k as u64 * 0x11 + 1)) as f32 - 0.5,
        hash_u01(seed ^ (k as u64 * 0x13 + 2)) as f32 - 0.5,
        hash_u01(seed ^ (k as u64 * 0x17 + 3)) as f32 - 0.5,
    ) * 0.7;
    (base + jitter).normalize_or(Vec3::Y)
}

// ---------------------------------------------------------------------------
// Icosphere
// ---------------------------------------------------------------------------

/// A subdivided unit icosphere: unit-direction vertices + triangle indices.
fn icosphere(subdivisions: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let mut verts: Vec<Vec3> = [
        Vec3::new(-1.0, t, 0.0),
        Vec3::new(1.0, t, 0.0),
        Vec3::new(-1.0, -t, 0.0),
        Vec3::new(1.0, -t, 0.0),
        Vec3::new(0.0, -1.0, t),
        Vec3::new(0.0, 1.0, t),
        Vec3::new(0.0, -1.0, -t),
        Vec3::new(0.0, 1.0, -t),
        Vec3::new(t, 0.0, -1.0),
        Vec3::new(t, 0.0, 1.0),
        Vec3::new(-t, 0.0, -1.0),
        Vec3::new(-t, 0.0, 1.0),
    ]
    .iter()
    .map(|v| v.normalize())
    .collect();

    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for _ in 0..subdivisions {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut next: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
        for f in &faces {
            let a = midpoint(f[0], f[1], &mut verts, &mut cache);
            let b = midpoint(f[1], f[2], &mut verts, &mut cache);
            let c = midpoint(f[2], f[0], &mut verts, &mut cache);
            next.push([f[0], a, c]);
            next.push([f[1], b, a]);
            next.push([f[2], c, b]);
            next.push([a, b, c]);
        }
        faces = next;
    }

    (verts, faces)
}

/// Shared, normalized edge midpoint (cached so the two faces sharing an edge
/// reuse the same new vertex — a watertight icosphere).
fn midpoint(a: u32, b: u32, verts: &mut Vec<Vec3>, cache: &mut HashMap<(u32, u32), u32>) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&i) = cache.get(&key) {
        return i;
    }
    let m = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
    let i = verts.len() as u32;
    verts.push(m);
    cache.insert(key, i);
    i
}

// ---------------------------------------------------------------------------
// 3-D gradient (Perlin) noise + 3-D Worley (cellular) noise — deterministic,
// seedable. Range of the Perlin fBm ~[-1, 1]; Worley `F2−F1` ~[0, 1].
// ---------------------------------------------------------------------------

/// 12 standard improved-Perlin gradient directions (edge midpoints of a cube).
const GRADS: [Vec3; 12] = [
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(-1.0, 1.0, 0.0),
    Vec3::new(1.0, -1.0, 0.0),
    Vec3::new(-1.0, -1.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(-1.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, -1.0),
    Vec3::new(-1.0, 0.0, -1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(0.0, -1.0, 1.0),
    Vec3::new(0.0, 1.0, -1.0),
    Vec3::new(0.0, -1.0, -1.0),
];

fn fbm3(p: Vec3, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for o in 0..octaves.max(1) {
        sum += amp * perlin3(p * freq, seed.wrapping_add(o.wrapping_mul(0x9E37_79B1)));
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm.max(f32::EPSILON)
}

fn perlin3(p: Vec3, seed: u32) -> f32 {
    let pi = p.floor();
    let f = p - pi;
    let (xi, yi, zi) = (pi.x as i32, pi.y as i32, pi.z as i32);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // quintic fade, per-axis

    let g = |dx: i32, dy: i32, dz: i32| -> f32 {
        let h = ihash(xi + dx, yi + dy, zi + dz, seed);
        let grad = GRADS[(h % 12) as usize];
        let o = f - Vec3::new(dx as f32, dy as f32, dz as f32);
        grad.dot(o)
    };

    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(g(0, 0, 0), g(1, 0, 0), u.x);
    let x10 = lerp(g(0, 1, 0), g(1, 1, 0), u.x);
    let x01 = lerp(g(0, 0, 1), g(1, 0, 1), u.x);
    let x11 = lerp(g(0, 1, 1), g(1, 1, 1), u.x);
    let y0 = lerp(x00, x10, u.y);
    let y1 = lerp(x01, x11, u.y);
    lerp(y0, y1, u.z)
}

/// 3-D Worley (cellular) noise, returning `F2 − F1` (the distance gap between the
/// two nearest jittered cell points). It's ~0 along cell boundaries and largest
/// at cell interiors, so adding it to the radius **bulges each cell and grooves
/// the boundaries** — the angular "rock crackle".
fn worley_f2_f1(p: Vec3, seed: u32) -> f32 {
    let pi = p.floor();
    let (xi, yi, zi) = (pi.x as i32, pi.y as i32, pi.z as i32);
    let mut f1 = f32::INFINITY;
    let mut f2 = f32::INFINITY;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cell = Vec3::new((xi + dx) as f32, (yi + dy) as f32, (zi + dz) as f32);
                let feat = cell + cell_point(xi + dx, yi + dy, zi + dz, seed);
                let d = (feat - p).length();
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
    }
    (f2 - f1).clamp(0.0, 1.0)
}

/// Jittered feature point inside cell `(x, y, z)`, components in `[0, 1)`.
fn cell_point(x: i32, y: i32, z: i32, seed: u32) -> Vec3 {
    Vec3::new(
        u01(ihash(x, y, z, seed ^ 0x68_95_1A_3D)),
        u01(ihash(x, y, z, seed ^ 0x2C_1B_3F_57)),
        u01(ihash(x, y, z, seed ^ 0x9E_10_77_C1)),
    )
}

/// 3-D integer hash → `u32`.
fn ihash(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed
        ^ (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (y as u32).wrapping_mul(0x85EB_CA77)
        ^ (z as u32).wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_F491);
    h ^= h >> 13;
    h
}

/// `u32` hash → `[0, 1)`.
fn u01(h: u32) -> f32 {
    (h & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}

/// Scalar hash of a `u64` → `[0, 1)`, for per-rock tint / jitter.
fn hash_u01(x: u64) -> f64 {
    let mut h = x ^ 0x9E37_79B9_7F4A_7C15;
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    (h & 0x000F_FFFF_FFFF_FFFF) as f64 / (1u64 << 52) as f64
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_subdivision_counts() {
        let (v0, f0) = icosphere(0);
        assert_eq!(v0.len(), 12);
        assert_eq!(f0.len(), 20);
        let (_, f1) = icosphere(1);
        assert_eq!(f1.len(), 80);
        let (_, f2) = icosphere(2);
        assert_eq!(f2.len(), 320);
        for v in &v0 {
            assert!((v.length() - 1.0).abs() < 1.0e-4);
        }
    }

    #[test]
    fn rock_mesh_is_deterministic_and_well_formed() {
        let p = RockMeshParams {
            seed: 0xDEAD_BEEF,
            ..Default::default()
        };
        let a = build_rock_mesh_data(&p);
        let b = build_rock_mesh_data(&p);
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.normals, b.normals);
        assert!(!a.positions.is_empty());
        assert_eq!(a.indices.len() % 3, 0);
        for &i in &a.indices {
            assert!((i as usize) < a.positions.len());
        }
        for n in &a.normals {
            let v = Vec3::from_array(*n);
            assert!(v.is_finite() && (v.length() - 1.0).abs() < 1.0e-3);
        }
        // Seated: the lowest vertex sits just below the origin.
        let min_y = a.positions.iter().fold(f32::INFINITY, |m, v| m.min(v[1]));
        assert!(min_y < 0.0, "rock base should seat below origin, got {min_y}");
    }

    #[test]
    fn flat_shaded_expands_vertices() {
        // Always flat-shaded → 3 unshared verts per face. Subdiv 1 = 80 faces.
        let p = RockMeshParams {
            subdivisions: 1,
            ..Default::default()
        };
        let d = build_rock_mesh_data(&p);
        assert_eq!(d.positions.len(), 80 * 3);
        assert_eq!(d.indices.len(), 80 * 3);
    }

    #[test]
    fn plane_cuts_create_flat_facets() {
        // With cuts, some adjacent triangles must end up coplanar (sharing a face
        // normal) — the flat facets. A pure noise blob almost never does.
        let p = RockMeshParams {
            seed: 0x1234,
            subdivisions: 2,
            cuts: 14,
            ..Default::default()
        };
        let d = build_rock_mesh_data(&p);
        // Count face normals that repeat (two+ triangles sharing a normal ⇒ a flat
        // facet spanning multiple triangles).
        let mut shared = 0;
        let faces = d.normals.len() / 3;
        for i in 0..faces {
            let ni = Vec3::from_array(d.normals[i * 3]);
            for j in (i + 1)..faces {
                let nj = Vec3::from_array(d.normals[j * 3]);
                if ni.dot(nj) > 0.9995 {
                    shared += 1;
                    break;
                }
            }
        }
        assert!(
            shared > 5,
            "expected several multi-triangle flat facets, found {shared}"
        );
    }
}
