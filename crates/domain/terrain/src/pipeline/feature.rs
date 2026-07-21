//! Feature types, instances, generators, and the [`FeatureCatalog`]
//! (spec §3–4).
//!
//! Features are the discrete, individually-addressable contributions to a
//! planet (craters, volcanoes, boulders, …), as opposed to the continuous
//! intent [`super::field`]s. There are two layers:
//!
//! - **Feature *types*** ([`FeatureType`]) are declared schemas — a name, a
//!   [`FeatureKind`] that routes the contribution, a parameter schema, and (for
//!   terrain-modifying features) a composition declaration. New types are added
//!   by declaration; nothing is hard-coded.
//! - **Feature *instances*** ([`FeatureInstance`]) are placements of a type.
//!   *Explicit* instances (authored or promoted) are stored; *procedural*
//!   instances are computed on demand by [`ScatterGenerator`]s and not stored.
//!   Querying a region returns the union, with promoted procedural instances
//!   removed via the generator's exclusion index.
//!
//! Generation is deterministic from `(seed, cell)`, so the same query returns
//! the same instances regardless of order — the property the renderer and
//! collider rely on. Promotion captures a procedural instance's parameters into
//! an explicit one and excludes its originating cell, so it survives reshuffles
//! and can be edited individually (spec §4).
//!
//! This increment lands the data model + a density-gated scatter generator +
//! promotion. Compositing terrain-modification features into the heightfield
//! (the influence-radius range query and ordered operator application, spec §7)
//! is the detail stage, built in migration phase P2.

use std::collections::{HashMap, HashSet};

use glam::Vec3;

use crate::pipeline::cubesphere::face_uv_to_dir;
use crate::pipeline::field::CompositionOp;
use crate::pipeline::planet::Planet;
use crate::pipeline::stamp::Falloff;
use crate::query::Region;

/// How a feature's contribution is routed (spec §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// Contributes to the heightfield via the detail stage.
    TerrainModification,
    /// Exposed as a scatter instance placement.
    SurfaceInstance,
    /// Texture-domain overlay (deferred).
    SurfaceDecal,
}

/// One declared parameter of a feature type.
#[derive(Debug, Clone)]
pub struct FeatureParam {
    pub name: String,
    pub default: f32,
}

impl FeatureParam {
    pub fn new(name: impl Into<String>, default: f32) -> Self {
        Self {
            name: name.into(),
            default,
        }
    }
}

/// Influence radius as a function of an instance's parameters (spec §3). Drives
/// the tile range query during compositing (P2).
#[derive(Debug, Clone)]
pub enum InfluenceRadius {
    /// A fixed radius in metres.
    Const(f32),
    /// `param * factor` metres (e.g. `radius_km * 1000`).
    FromParam { param: String, factor: f32 },
}

impl InfluenceRadius {
    pub fn radius_m(&self, params: &FeatureParams) -> f32 {
        match self {
            InfluenceRadius::Const(r) => *r,
            InfluenceRadius::FromParam { param, factor } => params.get(param) * factor,
        }
    }
}

/// How a `TerrainModification` feature composes into the accumulated terrain
/// (spec §7). Stored on the type; consumed by the detail stage in P2.
#[derive(Debug, Clone)]
pub struct FeatureComposition {
    pub op: CompositionOp,
    pub influence_radius: InfluenceRadius,
    pub falloff: Falloff,
    /// Parameter the instances are ordered by (e.g. `age`); younger overprints
    /// older. `None` means order is irrelevant for this type.
    pub ordering_key: Option<String>,
}

/// A declared feature type — a schema, not a placement.
#[derive(Debug, Clone)]
pub struct FeatureType {
    pub name: String,
    pub kind: FeatureKind,
    pub params: Vec<FeatureParam>,
    /// Required for `TerrainModification`; `None` otherwise.
    pub composition: Option<FeatureComposition>,
}

impl FeatureType {
    /// Look up a parameter's declared default.
    pub fn param_default(&self, name: &str) -> Option<f32> {
        self.params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.default)
    }
}

/// Committed parameter values of a feature instance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FeatureParams(pub HashMap<String, f32>);

impl FeatureParams {
    pub fn get(&self, name: &str) -> f32 {
        self.0.get(name).copied().unwrap_or(0.0)
    }

    pub fn set(&mut self, name: impl Into<String>, value: f32) {
        self.0.insert(name.into(), value);
    }
}

/// Stable identifier for an instance. Procedural instances derive theirs
/// deterministically from `(generator, cell)`; explicit instances get a fresh
/// id from the catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureInstanceId(pub u64);

/// Identifies a generator within a catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GeneratorId(pub u32);

/// Where a feature instance came from (spec §3–4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureOrigin {
    /// Authored from scratch.
    Authored,
    /// Computed on demand by a generator (not stored).
    Procedural { generator: GeneratorId, cell: u64 },
    /// Promoted from a procedural instance: stored, survives reshuffles.
    PromotedFrom { generator: GeneratorId, cell: u64 },
}

/// A placement of a feature type.
#[derive(Debug, Clone)]
pub struct FeatureInstance {
    pub id: FeatureInstanceId,
    pub type_name: String,
    /// Unit direction on the sphere.
    pub position: Vec3,
    pub params: FeatureParams,
    pub origin: FeatureOrigin,
}

/// A deterministic density-gated scatter generator.
///
/// The sphere is covered by a cube-sphere cell grid (6 faces × `cells_per_axis`
/// per side). Each cell deterministically hashes to a candidate position
/// (jittered within the cell) and parameter values. A candidate is kept when a
/// per-cell uniform draw falls under the local density (a field sampled at the
/// candidate, in `[0, 1]`) or, absent a density field, under `base_rate`.
/// Excluded cells (promoted) are skipped.
#[derive(Debug, Clone)]
pub struct ScatterGenerator {
    pub id: GeneratorId,
    pub feature_type: String,
    /// Field gating placement (sampled in `[0, 1]`). `None` ⇒ use `base_rate`.
    pub density_field: Option<String>,
    /// Placement probability per cell when `density_field` is `None`.
    pub base_rate: f32,
    pub seed: u32,
    pub cells_per_axis: u32,
    /// Per-parameter `[lo, hi]` ranges, sampled uniformly per instance.
    pub param_ranges: Vec<(String, f32, f32)>,
    /// Cells promoted to explicit instances — skipped during generation.
    exclusions: HashSet<u64>,
}

impl ScatterGenerator {
    pub fn new(
        id: GeneratorId,
        feature_type: impl Into<String>,
        seed: u32,
        cells_per_axis: u32,
    ) -> Self {
        Self {
            id,
            feature_type: feature_type.into(),
            density_field: None,
            base_rate: 1.0,
            seed,
            cells_per_axis: cells_per_axis.max(1),
            param_ranges: Vec::new(),
            exclusions: HashSet::new(),
        }
    }

    pub fn with_density_field(mut self, field: impl Into<String>) -> Self {
        self.density_field = Some(field.into());
        self
    }

    pub fn with_base_rate(mut self, rate: f32) -> Self {
        self.base_rate = rate;
        self
    }

    pub fn with_param_range(mut self, name: impl Into<String>, lo: f32, hi: f32) -> Self {
        self.param_ranges.push((name.into(), lo, hi));
        self
    }

    /// Append procedural instances within `region` to `out` (excluded cells
    /// skipped). Deterministic in `(seed, cell)`.
    pub fn generate_in_region(
        &self,
        planet: &Planet,
        region: Region,
        lod_m: f32,
        out: &mut Vec<FeatureInstance>,
    ) {
        let n = self.cells_per_axis;
        for face in 0..6u32 {
            for i in 0..n {
                for j in 0..n {
                    let cell = cell_id(face, i, j);
                    if self.exclusions.contains(&cell) {
                        continue;
                    }
                    let h = hash_cell(self.seed, face, i, j);

                    let u = (i as f32 + unit(pcg(h ^ 0x9E37_79B1))) / n as f32;
                    let v = (j as f32 + unit(pcg(h ^ 0x85EB_CA77))) / n as f32;
                    let pos = face_uv_to_dir(face, u, v);

                    if !in_region(pos, region) {
                        continue;
                    }

                    let u_exist = unit(pcg(h ^ 0xC2B2_AE3D));
                    let keep = match &self.density_field {
                        Some(field) => {
                            let d = planet.sample_field(field, pos, lod_m).unwrap_or(0.0);
                            u_exist < d.clamp(0.0, 1.0)
                        }
                        None => u_exist < self.base_rate.clamp(0.0, 1.0),
                    };
                    if !keep {
                        continue;
                    }

                    let mut params = FeatureParams::default();
                    for (k, (name, lo, hi)) in self.param_ranges.iter().enumerate() {
                        let up = unit(pcg(
                            h ^ (0xA24B_AED5u32.wrapping_add(k as u32 * 2_654_435_761))
                        ));
                        params.set(name.clone(), lo + (hi - lo) * up);
                    }

                    out.push(FeatureInstance {
                        id: FeatureInstanceId(stable_instance_id(self.id, cell)),
                        type_name: self.feature_type.clone(),
                        position: pos,
                        params,
                        origin: FeatureOrigin::Procedural {
                            generator: self.id,
                            cell,
                        },
                    });
                }
            }
        }
    }
}

/// Holds feature types, explicit instances, and generators; answers region
/// queries and supports promotion/demotion.
#[derive(Debug, Clone, Default)]
pub struct FeatureCatalog {
    types: HashMap<String, FeatureType>,
    generators: Vec<ScatterGenerator>,
    explicit: Vec<FeatureInstance>,
    next_explicit_id: u64,
}

impl FeatureCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn declare_type(&mut self, ty: FeatureType) {
        self.types.insert(ty.name.clone(), ty);
    }

    pub fn feature_type(&self, name: &str) -> Option<&FeatureType> {
        self.types.get(name)
    }

    pub fn add_generator(&mut self, generator: ScatterGenerator) {
        self.generators.push(generator);
    }

    /// Add an authored (explicit) instance, assigning it a fresh id.
    pub fn add_authored(&mut self, mut instance: FeatureInstance) -> FeatureInstanceId {
        let id = FeatureInstanceId(self.alloc_explicit_id());
        instance.id = id;
        instance.origin = FeatureOrigin::Authored;
        self.explicit.push(instance);
        id
    }

    /// All feature instances in `region`: procedural (minus exclusions) ∪
    /// explicit, sorted by the type's ordering key (ascending; ties broken by
    /// id) so terrain-modification compositing is order-stable.
    pub fn query_in_region(
        &self,
        planet: &Planet,
        region: Region,
        lod_m: f32,
    ) -> Vec<FeatureInstance> {
        let mut out = Vec::new();
        for generator in &self.generators {
            generator.generate_in_region(planet, region, lod_m, &mut out);
        }
        for instance in &self.explicit {
            if in_region(instance.position, region) {
                out.push(instance.clone());
            }
        }
        out.sort_by(|a, b| {
            let ka = self.ordering_value(a);
            let kb = self.ordering_value(b);
            ka.partial_cmp(&kb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.0.cmp(&b.id.0))
        });
        out
    }

    /// Filter a region query by kind (e.g. `SurfaceInstance` for the scatter
    /// stream, `TerrainModification` for the heightfield).
    pub fn query_in_region_by_kind(
        &self,
        planet: &Planet,
        region: Region,
        lod_m: f32,
        kind: FeatureKind,
    ) -> Vec<FeatureInstance> {
        self.query_in_region(planet, region, lod_m)
            .into_iter()
            .filter(|inst| self.types.get(&inst.type_name).map(|t| t.kind) == Some(kind))
            .collect()
    }

    /// Promote a procedural instance to an explicit one (spec §4): capture its
    /// parameters, mark the origin `PromotedFrom`, and exclude its originating
    /// cell so the generator no longer emits it. Returns the new explicit id,
    /// or `None` if the instance isn't procedural or its generator is unknown.
    pub fn promote(&mut self, instance: &FeatureInstance) -> Option<FeatureInstanceId> {
        let FeatureOrigin::Procedural { generator, cell } = instance.origin else {
            return None;
        };
        let slot = self.generators.iter_mut().find(|g| g.id == generator)?;
        slot.exclusions.insert(cell);

        let id = FeatureInstanceId(self.alloc_explicit_id());
        let mut promoted = instance.clone();
        promoted.id = id;
        promoted.origin = FeatureOrigin::PromotedFrom { generator, cell };
        self.explicit.push(promoted);
        Some(id)
    }

    /// Remove an explicit instance. If it was promoted, also drop its exclusion
    /// so the position returns to procedural control. Returns whether anything
    /// was removed.
    pub fn demote(&mut self, id: FeatureInstanceId) -> bool {
        let Some(pos) = self.explicit.iter().position(|i| i.id == id) else {
            return false;
        };
        let removed = self.explicit.remove(pos);
        if let FeatureOrigin::PromotedFrom { generator, cell } = removed.origin
            && let Some(slot) = self.generators.iter_mut().find(|g| g.id == generator)
        {
            slot.exclusions.remove(&cell);
        }
        true
    }

    fn ordering_value(&self, instance: &FeatureInstance) -> f32 {
        self.types
            .get(&instance.type_name)
            .and_then(|t| t.composition.as_ref())
            .and_then(|c| c.ordering_key.as_deref())
            .map(|key| instance.params.get(key))
            .unwrap_or(0.0)
    }

    fn alloc_explicit_id(&mut self) -> u64 {
        // Explicit ids live in the high half so they never collide with the
        // hash-derived procedural ids.
        let id = self.next_explicit_id | 0x8000_0000_0000_0000;
        self.next_explicit_id += 1;
        id
    }
}

// ---------------------------------------------------------------------------
// Cube-sphere cell addressing + hashing
// ---------------------------------------------------------------------------

fn in_region(pos: Vec3, region: Region) -> bool {
    let cos_limit = region.angular_radius_rad.cos();
    pos.normalize_or_zero()
        .dot(region.center.normalize_or_zero())
        >= cos_limit
}

fn cell_id(face: u32, i: u32, j: u32) -> u64 {
    ((face as u64) << 48) | ((i as u64) << 24) | (j as u64)
}

fn stable_instance_id(generator: GeneratorId, cell: u64) -> u64 {
    let mut h = cell.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (generator.0 as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    // Keep procedural ids in the low half (explicit ids set the top bit).
    h & 0x7FFF_FFFF_FFFF_FFFF
}

fn pcg(x: u32) -> u32 {
    let state = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

fn hash_cell(seed: u32, face: u32, i: u32, j: u32) -> u32 {
    let mut h = face.wrapping_mul(73_856_093);
    h ^= i.wrapping_mul(19_349_663);
    h ^= j.wrapping_mul(83_492_791);
    h = pcg(h);
    h ^= seed;
    pcg(h)
}

fn unit(x: u32) -> f32 {
    x as f32 / 4_294_967_296.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::expr::Expr;
    use crate::pipeline::field::Field;
    use crate::pipeline::planet::{Planet, PlanetPhysical};

    fn whole_sphere() -> Region {
        Region {
            center: Vec3::X,
            angular_radius_rad: std::f32::consts::PI,
        }
    }

    fn planet_with_density(density: f32) -> Planet {
        Planet::new(
            PlanetPhysical { radius_m: 1000.0 },
            7,
            vec![Field::scalar("d", Expr::Const(density))],
        )
        .unwrap()
    }

    fn crater_catalog(generator: ScatterGenerator) -> FeatureCatalog {
        let mut catalog = FeatureCatalog::new();
        catalog.declare_type(FeatureType {
            name: "crater".into(),
            kind: FeatureKind::TerrainModification,
            params: vec![
                FeatureParam::new("age", 0.0),
                FeatureParam::new("radius_m", 100.0),
            ],
            composition: Some(FeatureComposition {
                op: CompositionOp::SmoothMin { k: 50.0 },
                influence_radius: InfluenceRadius::FromParam {
                    param: "radius_m".into(),
                    factor: 2.0,
                },
                falloff: Falloff::Smoothstep,
                ordering_key: Some("age".into()),
            }),
        });
        catalog.add_generator(generator);
        catalog
    }

    #[test]
    fn generation_is_deterministic() {
        let planet = planet_with_density(1.0);
        let scatter = ScatterGenerator::new(GeneratorId(1), "crater", 99, 8)
            .with_density_field("d")
            .with_param_range("age", 0.0, 4.0);
        let catalog = crater_catalog(scatter);
        let a = catalog.query_in_region(&planet, whole_sphere(), 1.0);
        let b = catalog.query_in_region(&planet, whole_sphere(), 1.0);
        assert_eq!(a.len(), b.len());
        assert!(a.len() > 0, "density 1.0 should place instances");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.position, y.position);
            assert_eq!(x.params, y.params);
        }
    }

    #[test]
    fn density_gates_placement() {
        let make_scatter =
            || ScatterGenerator::new(GeneratorId(1), "crater", 99, 8).with_density_field("d");
        let none = crater_catalog(make_scatter()).query_in_region(
            &planet_with_density(0.0),
            whole_sphere(),
            1.0,
        );
        let all = crater_catalog(make_scatter()).query_in_region(
            &planet_with_density(1.0),
            whole_sphere(),
            1.0,
        );
        assert!(none.is_empty(), "zero density places nothing");
        assert!(all.len() > none.len(), "full density places more");
    }

    #[test]
    fn instances_lie_within_region() {
        let planet = planet_with_density(1.0);
        let region = Region {
            center: Vec3::Z,
            angular_radius_rad: 0.3,
        };
        let scatter =
            ScatterGenerator::new(GeneratorId(1), "crater", 5, 24).with_density_field("d");
        let catalog = crater_catalog(scatter);
        let instances = catalog.query_in_region(&planet, region, 1.0);
        assert!(!instances.is_empty());
        for inst in &instances {
            assert!(
                in_region(inst.position, region),
                "instance outside the queried cap"
            );
        }
    }

    #[test]
    fn promotion_excludes_cell_and_keeps_count_stable() {
        let planet = planet_with_density(1.0);
        let scatter = ScatterGenerator::new(GeneratorId(1), "crater", 99, 8)
            .with_density_field("d")
            .with_param_range("age", 0.0, 4.0);
        let mut catalog = crater_catalog(scatter);

        let before = catalog.query_in_region(&planet, whole_sphere(), 1.0);
        let target = before[0].clone();
        let new_id = catalog
            .promote(&target)
            .expect("procedural instance promotes");

        let after = catalog.query_in_region(&planet, whole_sphere(), 1.0);
        // Same total count: the procedural one is gone, the explicit one present.
        assert_eq!(after.len(), before.len());
        // The promoted instance is present with its new id at the same position.
        let promoted = after
            .iter()
            .find(|i| i.id == new_id)
            .expect("promoted present");
        assert_eq!(promoted.position, target.position);
        assert!(matches!(
            promoted.origin,
            FeatureOrigin::PromotedFrom { .. }
        ));
        // The original procedural cell is no longer emitted.
        assert!(
            !after.iter().any(|i| i.origin == target.origin),
            "originating cell must be excluded after promotion"
        );

        // Demotion returns the position to procedural control.
        assert!(catalog.demote(new_id));
        let restored = catalog.query_in_region(&planet, whole_sphere(), 1.0);
        assert_eq!(restored.len(), before.len());
        assert!(restored.iter().any(|i| i.origin == target.origin));
    }

    #[test]
    fn results_sorted_by_ordering_key() {
        let planet = planet_with_density(1.0);
        let scatter = ScatterGenerator::new(GeneratorId(1), "crater", 3, 10)
            .with_density_field("d")
            .with_param_range("age", 0.0, 4.0);
        let catalog = crater_catalog(scatter);
        let instances = catalog.query_in_region(&planet, whole_sphere(), 1.0);
        for pair in instances.windows(2) {
            assert!(
                pair[0].params.get("age") <= pair[1].params.get("age"),
                "instances must be ordered by age (ascending)"
            );
        }
    }
}
