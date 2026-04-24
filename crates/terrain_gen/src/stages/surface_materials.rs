//! Stage 5 — Surface Materials.
//!
//! Assigns a discrete material ID to every icosphere vertex based on the
//! *history* recorded in earlier stages: which province the vertex sits
//! on, how much sediment Stage 3 deposited on it, how much water passes
//! through, and — crucially — where that sediment came from.
//!
//! The "iron palette" of Thalos is the visible consequence of
//! *provenance*: floodplains downstream of an ActiveMargin or
//! HotspotTrack are iron-rich because Stage 3's drainage network routes
//! mafic-derived sediment to them. Cratons and old Suture belts
//! weathered in place get a paler, less-saturated look. The rare active
//! provinces are near-black basalt. Together this tells the story the
//! spec asks for — an old world with fossil iron in its sediments,
//! active relief only at a few exceptional places — without a shader
//! tint sitting on top of an otherwise uniform surface.
//!
//! Algorithm, per vertex, first-match-wins:
//!
//! 1. Submerged shallow (within `shelf_depth_m` of sea level):
//!    → iron-sand beach if upstream iron-fraction is high, else
//!    continental shelf silt.
//! 2. Submerged deep: abyssal seabed.
//! 3. Active volcanic (ActiveMargin / HotspotTrack, age ≤ threshold):
//!    → fresh volcanic rock.
//! 4. Floodplain (high drainage accumulation + high sediment thickness):
//!    → iron-rust sediment if iron-rich, else peat/wet soil.
//! 5. Bare scoured upland (high drainage accumulation + low sediment +
//!    high elevation): → bare rock.
//! 6. Ancient stable (Craton or Suture, low sediment):
//!    → banded iron formation on iron-rich old Sutures,
//!    weathered granite elsewhere.
//! 7. Default continental: regolith / weathered granite.
//!
//! Provenance ("iron-fraction") is computed by running a flow-
//! accumulation pass over Stage 3's drainage graph, identical in shape
//! to Stage 3's area accumulation but with a per-vertex *iron source
//! weight* as the starting quantity. After accumulation,
//! `iron_flux[v] / area_flux[v]` is the fraction of the upstream
//! catchment that contributed iron-rich source rock to the sediment
//! passing through v.
//!
//! See `docs/thalos_terrain_pipeline.md §Stage 5`.

use rayon::prelude::*;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cubemap::{CubemapFace, face_uv_to_dir};
use crate::icosphere::Icosphere;
use crate::province::{ProvinceDef, ProvinceKind};
use crate::stage::Stage;
use crate::types::Material;

/// Material palette indices. Exported so tests and future consumers can
/// reference them without reaching into the materials vector by position.
///
/// The indices below are the order materials are pushed into
/// `builder.materials` at the start of `apply()`. A future stage that
/// needs to introduce more material IDs should either extend this
/// palette or push after the SurfaceMaterials block and document the
/// offset.
pub const MAT_CONTINENTAL_REGOLITH: u8 = 0;
pub const MAT_WEATHERED_GRANITE: u8 = 1;
pub const MAT_BANDED_IRON: u8 = 2;
pub const MAT_IRON_RUST_FLOODPLAIN: u8 = 3;
pub const MAT_PEAT_WETLAND: u8 = 4;
pub const MAT_BARE_SCOURED_ROCK: u8 = 5;
pub const MAT_FRESH_VOLCANIC: u8 = 6;
pub const MAT_IRON_SAND_BEACH: u8 = 7;
pub const MAT_COASTAL_SHELF_SILT: u8 = 8;
pub const MAT_ABYSSAL_SEABED: u8 = 9;

/// No serde defaults — every Thalos body file must tune every field
/// explicitly. Matching the CoarseElevation policy: silent defaults on
/// material-assignment thresholds produce muddy output and invite
/// cargo-culting between bodies with different spec profiles.
#[derive(Debug, Clone, Deserialize)]
pub struct SurfaceMaterials {
    // --- Coastal / submerged band ------------------------------------------
    /// Depth (m) below sea level within which a vertex reads as "shelf"
    /// instead of deep abyss. Beaches form in the upper part of this
    /// band; shelf silt fills the rest. ~500 m is a realistic continental
    /// shelf break.
    pub shelf_depth_m: f32,

    // --- Floodplain rule ---------------------------------------------------
    /// Log10 accumulation threshold above which a land vertex reads as
    /// part of a drainage trunk (floodplain candidate). Expressed as
    /// log10 m² so a single number covers the 10⁸–10¹² m² range Stage 3
    /// produces. Lower values catch tributary-scale rivers; higher
    /// values restrict to continental-trunk mains.
    pub floodplain_log10_accumulation_m2: f32,
    /// Sediment thickness (m) above which a floodplain candidate counts
    /// as depositional. Stage 3 deposits tens of meters on major delta
    /// vertices, fractions of a meter on uplands.
    pub floodplain_min_sediment_m: f32,
    /// Maximum elevation above sea level a vertex can have to still
    /// count as "floodplain" rather than upland. 1000 m cuts off around
    /// the foothill / plain transition.
    pub floodplain_max_elevation_m: f32,

    // --- Bare-rock rule ----------------------------------------------------
    /// Elevation (m above sea level) above which a river-carved vertex
    /// with exposed bedrock reads as "bare scoured upland."
    pub bare_rock_min_elevation_m: f32,
    /// Sediment thickness (m) strictly below which a vertex counts as
    /// scoured (bare bedrock exposed). Distinct from the floodplain and
    /// ancient-stable thresholds so each rule can be tuned
    /// independently.
    pub bare_rock_max_sediment_m: f32,

    // --- Ancient-stable rule -----------------------------------------------
    /// Sediment thickness (m) strictly below which a Craton or Suture
    /// vertex reads as weathered granite / banded iron rather than
    /// regolith. Higher than `bare_rock_max_sediment_m` because
    /// weathered-in-place stable terrain still carries a thin soil
    /// blanket (and `ancient_stable_max_sediment_m ≥
    /// bare_rock_max_sediment_m` should always hold).
    pub ancient_stable_max_sediment_m: f32,

    // --- Fresh-volcanic rule -----------------------------------------------
    /// Maximum province age (Myr) for ActiveMargin / HotspotTrack
    /// vertices to still read as fresh basalt rather than weathered
    /// volcanic residue. The spec's "rare active exceptions" are
    /// recent enough (tens of Myr) to still expose fresh rock.
    pub fresh_volcanic_max_age_myr: f32,

    // --- Iron provenance ---------------------------------------------------
    /// Iron-source weight assigned to ActiveMargin vertices — their
    /// mafic arc-volcanic contribution to downstream sediment.
    pub iron_source_active_margin: f32,
    /// Iron-source weight for HotspotTrack vertices (ocean-island
    /// basalt, oversampled as source rock compared to area).
    pub iron_source_hotspot_track: f32,
    /// Iron-source weight for Suture vertices — the "eroded arc
    /// remnant" of the spec. Not as iron-rich per unit area as live
    /// arcs but contributes over wide belts.
    pub iron_source_suture: f32,

    /// Iron-fraction threshold above which a floodplain reads as
    /// rust-red rather than peat, and a coastal vertex reads as iron-
    /// sand beach rather than silt shelf. Normalized to [0, 1] —
    /// iron_flux / area_flux.
    pub iron_fraction_threshold: f32,
    /// Iron-fraction threshold above which an *old Suture* (high age,
    /// low sediment) reads as banded iron formation rather than
    /// weathered granite. Lower than the floodplain threshold because
    /// the signal on an old belt is concentration rather than flux.
    pub iron_fraction_bif_threshold: f32,

    /// When true, overwrite the albedo accumulator with a material-
    /// colored debug paint so `just bake Thalos` shows the material
    /// assignments. Turn off once downstream stages paint real albedo.
    pub debug_paint_albedo: bool,
}

impl Stage for SurfaceMaterials {
    fn name(&self) -> &str {
        "surface_materials"
    }

    fn dependencies(&self) -> &[&str] {
        &["hydrological_carving"]
    }

    fn apply(&self, builder: &mut BodyBuilder) {
        // Register the palette. Order matches the MAT_* constants above.
        builder.materials = palette();

        let sphere = builder
            .sphere
            .as_ref()
            .expect("SurfaceMaterials requires sphere from earlier stages")
            .clone();
        let elevations = builder
            .vertex_elevations_m
            .as_ref()
            .expect("SurfaceMaterials requires vertex_elevations_m from Stage 3")
            .clone();
        let sediment = builder
            .vertex_sediment_m
            .as_ref()
            .expect("SurfaceMaterials requires vertex_sediment_m from Stage 3")
            .clone();
        let drainage = builder
            .drainage_graph
            .as_ref()
            .expect("SurfaceMaterials requires drainage_graph from Stage 3")
            .clone();
        let sea_level = builder
            .sea_level_m
            .expect("SurfaceMaterials requires sea_level_m from Stage 3");
        let provinces = builder.provinces.clone();
        let vertex_provinces = builder
            .vertex_provinces
            .as_ref()
            .expect("SurfaceMaterials requires vertex_provinces from Stage 1")
            .clone();

        // 1. Compute per-vertex iron-source weight from the province
        //    kind. This is the quantity that gets propagated downstream
        //    by flow accumulation to produce per-vertex provenance.
        let iron_source: Vec<f32> = vertex_provinces
            .par_iter()
            .map(|&pid| {
                let kind = provinces[pid as usize].kind;
                self.source_weight(kind)
            })
            .collect();

        // 2. Flow-accumulate the iron source alongside the area flux.
        //    Downstream iron-fraction = iron_flux / area_flux. Both
        //    tracks are seeded with the per-vertex quantity, sorted by
        //    decreasing elevation (valid topological order for a
        //    single-flow drainage tree), and pushed downstream.
        let iron_fraction = compute_iron_fraction(&elevations, &drainage.downstream, &iron_source);

        // 3. Per-vertex material assignment. Pure function of per-vertex
        //    inputs — parallelizable, deterministic.
        let accumulation = &drainage.accumulation_m2;
        let materials: Vec<u8> = (0..sphere.vertices.len())
            .into_par_iter()
            .map(|vi| {
                let e = elevations[vi];
                let sed = sediment[vi];
                let accum = accumulation[vi];
                let iron = iron_fraction[vi];
                let prov = &provinces[vertex_provinces[vi] as usize];

                self.classify(e, sea_level, sed, accum, iron, prov.kind, prov.age_myr)
            })
            .collect();

        // 4. Bake the material-ID cubemap. Classifying at
        //    per-texel resolution (rather than taking the nearest
        //    vertex's category) lets the coastline and other
        //    elevation-thresholded boundaries follow the smoothly-
        //    interpolated height cube at texel sharpness, killing
        //    the L8 Voronoi-cell stair-stepping that nearest-vertex
        //    rasterization produced along every shoreline. Discrete
        //    categorical fields (province kind, age) still take the
        //    nearest vertex — an enum can't be interpolated.
        bake_materials_to_cubemap(
            builder,
            &sphere,
            &elevations,
            &sediment,
            &drainage.accumulation_m2,
            &iron_fraction,
            sea_level,
            &provinces,
            &vertex_provinces,
            self,
        );

        if self.debug_paint_albedo {
            paint_materials_debug_albedo(
                builder,
                &sphere,
                &materials,
                &iron_fraction,
                self.iron_fraction_threshold,
            );
        }

        // Write debug counts to stderr for the bake iteration loop.
        // Cheap (just a histogram); useful for confirming rule coverage
        // without spinning up a full visual comparison.
        #[cfg(debug_assertions)]
        {
            let mut counts = [0u32; PALETTE_SIZE];
            for &m in &materials {
                if (m as usize) < counts.len() {
                    counts[m as usize] += 1;
                }
            }
            eprintln!(
                "SurfaceMaterials: regolith={} granite={} BIF={} rust={} peat={} bare={} volcanic={} ironsand={} shelf={} abyss={}",
                counts[MAT_CONTINENTAL_REGOLITH as usize],
                counts[MAT_WEATHERED_GRANITE as usize],
                counts[MAT_BANDED_IRON as usize],
                counts[MAT_IRON_RUST_FLOODPLAIN as usize],
                counts[MAT_PEAT_WETLAND as usize],
                counts[MAT_BARE_SCOURED_ROCK as usize],
                counts[MAT_FRESH_VOLCANIC as usize],
                counts[MAT_IRON_SAND_BEACH as usize],
                counts[MAT_COASTAL_SHELF_SILT as usize],
                counts[MAT_ABYSSAL_SEABED as usize],
            );
        }
    }
}

impl SurfaceMaterials {
    /// Per-province source-rock iron contribution. ActiveMargin and
    /// HotspotTrack are mafic/young; Suture is the old eroded-arc
    /// remnant with residual banded-iron signature; everything else
    /// (Craton, RiftScar, oceanic basin) doesn't contribute.
    fn source_weight(&self, kind: ProvinceKind) -> f32 {
        match kind {
            ProvinceKind::ActiveMargin => self.iron_source_active_margin,
            ProvinceKind::HotspotTrack => self.iron_source_hotspot_track,
            ProvinceKind::Suture | ProvinceKind::ArcRemnant => self.iron_source_suture,
            _ => 0.0,
        }
    }

    /// First-match-wins rule walk. See module-level comment for the
    /// full ladder. Pure function of the per-vertex arguments — keeps
    /// the rule logic compact and trivially auditable.
    #[allow(clippy::too_many_arguments)]
    fn classify(
        &self,
        elevation_m: f32,
        sea_level_m: f32,
        sediment_m: f32,
        accumulation_m2: f32,
        iron_fraction: f32,
        kind: ProvinceKind,
        age_myr: f32,
    ) -> u8 {
        let height_above_sea = elevation_m - sea_level_m;

        // Submerged.
        if height_above_sea <= 0.0 {
            let depth = -height_above_sea;
            if depth <= self.shelf_depth_m {
                if iron_fraction >= self.iron_fraction_threshold {
                    return MAT_IRON_SAND_BEACH;
                }
                return MAT_COASTAL_SHELF_SILT;
            }
            return MAT_ABYSSAL_SEABED;
        }

        // Active volcanic provinces — narrow window, caught before any
        // sediment/drainage classification so that the rare active
        // exceptions aren't masked by river-carving logic.
        if (kind == ProvinceKind::ActiveMargin || kind == ProvinceKind::HotspotTrack)
            && age_myr <= self.fresh_volcanic_max_age_myr
        {
            return MAT_FRESH_VOLCANIC;
        }

        // Drainage-conditioned rules. `high_accumulation` ⇒ a river
        // reaches this vertex; combined with sediment/elevation it
        // splits depositional lowland from scoured upland.
        let high_accumulation =
            accumulation_m2.log10() >= self.floodplain_log10_accumulation_m2;

        if high_accumulation
            && sediment_m >= self.floodplain_min_sediment_m
            && height_above_sea <= self.floodplain_max_elevation_m
        {
            if iron_fraction >= self.iron_fraction_threshold {
                return MAT_IRON_RUST_FLOODPLAIN;
            }
            return MAT_PEAT_WETLAND;
        }

        if high_accumulation
            && sediment_m < self.bare_rock_max_sediment_m
            && height_above_sea >= self.bare_rock_min_elevation_m
        {
            return MAT_BARE_SCOURED_ROCK;
        }

        // Ancient-stable: old Suture belts with iron-fraction signal
        // above a low threshold read as banded iron formation;
        // otherwise Cratons and aged Sutures blend into the weathered-
        // granite continental default. Uses its own sediment threshold
        // so stable terrain can carry a thin weathered blanket and
        // still count as "bedrock at surface."
        if kind == ProvinceKind::Suture
            && sediment_m < self.ancient_stable_max_sediment_m
            && iron_fraction >= self.iron_fraction_bif_threshold
        {
            return MAT_BANDED_IRON;
        }

        if (kind == ProvinceKind::Craton || kind == ProvinceKind::Suture)
            && sediment_m < self.ancient_stable_max_sediment_m
        {
            return MAT_WEATHERED_GRANITE;
        }

        // Default: continental regolith — everywhere without a stronger
        // signal. Weathered blanket, warm pale gray.
        MAT_CONTINENTAL_REGOLITH
    }
}

// ---------------------------------------------------------------------------
// Material palette
// ---------------------------------------------------------------------------

/// Number of materials registered by this stage. Keeps the debug-count
/// histogram and palette len in sync.
#[cfg(debug_assertions)]
const PALETTE_SIZE: usize = 10;

fn palette() -> Vec<Material> {
    // Order MUST match the MAT_* constants. Linear RGB, pre-sRGB —
    // `finalize_albedo` handles the sRGB conversion at bake time.
    //
    // Design: desaturated earthy tones on the spectrum of stone and
    // soil, with one chromatic signal (iron-red) that reads strongly
    // only where provenance justifies it. The spec's "Thalos is a rust
    // world" target is driven by the *distribution* of these materials,
    // not the saturation of any one tint.
    vec![
        // 0 MAT_CONTINENTAL_REGOLITH — warm pale gray, default blanket.
        Material {
            albedo: [0.42, 0.38, 0.32],
            roughness: 0.85,
        },
        // 1 MAT_WEATHERED_GRANITE — slightly lighter, cooler. Craton
        // interiors and aged Sutures without iron provenance.
        Material {
            albedo: [0.52, 0.48, 0.42],
            roughness: 0.82,
        },
        // 2 MAT_BANDED_IRON — dark brown-red. Old arc remnants (Suture)
        // with residual iron provenance. Heavily weathered.
        Material {
            albedo: [0.28, 0.16, 0.12],
            roughness: 0.70,
        },
        // 3 MAT_IRON_RUST_FLOODPLAIN — signature Thalos color. Warm
        // iron-oxide brown, *not* saturated red — on Earth, heavily
        // iron-stained ground reads as rusty earth tones rather than
        // painted lines. Desaturated so the debug albedo doesn't
        // look cartoonish when this material fires along rivers.
        Material {
            albedo: [0.38, 0.26, 0.20],
            roughness: 0.75,
        },
        // 4 MAT_PEAT_WETLAND — dark brown-black. Depositional lowland
        // without iron provenance; boggy mud-and-organics.
        Material {
            albedo: [0.10, 0.08, 0.05],
            roughness: 0.65,
        },
        // 5 MAT_BARE_SCOURED_ROCK — cool mid-gray. Recently scoured
        // upland with no sediment blanket.
        Material {
            albedo: [0.36, 0.36, 0.38],
            roughness: 0.80,
        },
        // 6 MAT_FRESH_VOLCANIC — near-black basalt. Active margins and
        // youngest hotspot vertices.
        Material {
            albedo: [0.08, 0.08, 0.08],
            roughness: 0.60,
        },
        // 7 MAT_IRON_SAND_BEACH — very dark with red cast. Mafic-mineral
        // concentration at iron-source river mouths.
        Material {
            albedo: [0.14, 0.09, 0.07],
            roughness: 0.55,
        },
        // 8 MAT_COASTAL_SHELF_SILT — tan-gray. Continental shelf with
        // mixed sediment, no iron signature.
        Material {
            albedo: [0.30, 0.28, 0.22],
            roughness: 0.65,
        },
        // 9 MAT_ABYSSAL_SEABED — very dark cool. Deep ocean basalt.
        Material {
            albedo: [0.08, 0.08, 0.10],
            roughness: 0.60,
        },
    ]
}

// ---------------------------------------------------------------------------
// Iron provenance via drainage-accumulated source weight
// ---------------------------------------------------------------------------

/// Compute per-vertex iron fraction by propagating the per-vertex
/// iron-source weight downstream through the drainage graph, in
/// lockstep with the area flux.
///
/// Returns `iron_flux / area_flux` clamped to [0, 1]. A vertex with no
/// upstream iron source sees 0; a vertex directly downstream of an
/// ActiveMargin vertex sees roughly `iron_source_active_margin /
/// (n_vertices_upstream + 1)`.
///
/// Uses decreasing-elevation topological order identical to Stage 3's
/// `flow_accumulation`, so the result is deterministic under the same
/// ordering rules.
fn compute_iron_fraction(
    elevations: &[f32],
    downstream: &[u32],
    iron_source: &[f32],
) -> Vec<f32> {
    let n = elevations.len();

    // Every vertex starts with its own unit "area flux" and its own
    // iron source. The sum of these over the catchment yields the
    // upstream catchment area and upstream iron source respectively.
    // Fraction = iron / area. Using unit area here (rather than
    // vertex_area_m2) keeps the ratio dimensionally correct without
    // threading radius_m through.
    let mut area_flux = vec![1.0f32; n];
    let mut iron_flux = iron_source.to_vec();

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        elevations[b as usize]
            .partial_cmp(&elevations[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &vi in &order {
        let ds = downstream[vi as usize];
        if ds != vi {
            area_flux[ds as usize] += area_flux[vi as usize];
            iron_flux[ds as usize] += iron_flux[vi as usize];
        }
    }

    (0..n)
        .map(|i| {
            let a = area_flux[i];
            if a > 0.0 {
                (iron_flux[i] / a).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Material cubemap bake (nearest-vertex) + debug albedo paint
// ---------------------------------------------------------------------------

/// Walk every cubemap texel and classify each one at texel resolution.
///
/// The earlier per-vertex `materials` array is still computed (debug
/// albedo and the histogram consume it), but the cubemap — which the
/// shader reads — is produced by re-running `classify` at each texel
/// against *interpolated* continuous inputs. Continuous fields
/// (elevation, sediment, accumulation, iron fraction) blend
/// barycentrically; categorical fields (province kind, age) take the
/// nearest of the triangle's three vertices because enums can't be
/// interpolated.
///
/// The key payoff is at elevation-threshold boundaries, particularly
/// the coastline: the height cube is already smoothly interpolated,
/// so classifying per-texel makes the land/ocean transition follow
/// the actual `height == sea_level` isoline at texel sharpness
/// instead of painting each vertex's ~38-texel Voronoi cell with a
/// uniform material ID.
///
/// Province-kind discontinuities still alias at province boundaries
/// (suture belts, active margins) — enum interpolation is undefined —
/// but those boundaries are soft gradients in the downstream
/// continuous fields anyway, so the visible aliasing is much reduced.
#[allow(clippy::too_many_arguments)]
fn bake_materials_to_cubemap(
    builder: &mut BodyBuilder,
    sphere: &Icosphere,
    elevations: &[f32],
    sediment: &[f32],
    accumulation: &[f32],
    iron_fraction: &[f32],
    sea_level: f32,
    provinces: &[ProvinceDef],
    vertex_provinces: &[u32],
    params: &SurfaceMaterials,
) {
    let res = builder.cubemap_resolution;
    let inv = 1.0 / res as f32;

    for face in CubemapFace::ALL {
        let data = builder.material_cubemap.face_data_mut(face);
        data.par_chunks_mut(res as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let mut last_v = 0u32;
                for (x, texel) in row.iter_mut().enumerate() {
                    let u = (x as f32 + 0.5) * inv;
                    let v = (y as f32 + 0.5) * inv;
                    let dir = face_uv_to_dir(face, u, v);
                    let (ti, w) = sphere.barycentric_triangle(dir, last_v);
                    let tri = sphere.triangles[ti as usize];
                    last_v = tri[0];

                    let i0 = tri[0] as usize;
                    let i1 = tri[1] as usize;
                    let i2 = tri[2] as usize;

                    // Continuous fields: straight barycentric blend.
                    let elevation_m =
                        w[0] * elevations[i0] + w[1] * elevations[i1] + w[2] * elevations[i2];
                    let sediment_m =
                        w[0] * sediment[i0] + w[1] * sediment[i1] + w[2] * sediment[i2];
                    let iron =
                        w[0] * iron_fraction[i0] + w[1] * iron_fraction[i1] + w[2] * iron_fraction[i2];

                    // Accumulation spans ~1 to 1e12 m². Interpolate
                    // in log space so the smooth ramp from "upland
                    // divide" to "trunk river" matches the log-space
                    // threshold the classifier uses, rather than
                    // collapsing to the dominant vertex's value.
                    let a0 = accumulation[i0].max(1.0);
                    let a1 = accumulation[i1].max(1.0);
                    let a2 = accumulation[i2].max(1.0);
                    let accum_ln = w[0] * a0.ln() + w[1] * a1.ln() + w[2] * a2.ln();
                    let accumulation_m2 = accum_ln.exp();

                    // Categorical: pick the nearest vertex of the
                    // three by largest barycentric weight.
                    let nearest_idx = if w[0] >= w[1] && w[0] >= w[2] {
                        i0
                    } else if w[1] >= w[2] {
                        i1
                    } else {
                        i2
                    };
                    let prov = &provinces[vertex_provinces[nearest_idx] as usize];

                    *texel = params.classify(
                        elevation_m,
                        sea_level,
                        sediment_m,
                        accumulation_m2,
                        iron,
                        prov.kind,
                        prov.age_myr,
                    );
                }
            });
    }
}

/// Paint the debug albedo by barycentric-blending per-vertex colors
/// across each triangle. Two reasons not to use per-texel nearest-
/// vertex lookup here:
///
/// - Nearest-vertex sampling prints the icosphere hex-cell topology
///   as hard seams wherever neighbour vertices carry different
///   material IDs — visible as Voronoi-like edges on the continents.
/// - It forces the rust-iron floodplain material to read as thin
///   painted lines along the drainage graph, because each river
///   vertex flips to a saturated rust colour against its neighbours.
///
/// Instead we pre-compute a per-vertex *final colour* (with the
/// iron-rust overlay already mixed into peat vertices proportional
/// to their iron-fraction), then barycentric-blend the three vertex
/// colours per texel. This gives:
///   - smooth material transitions across triangles (no hex seams), and
///   - iron staining that fades in continuously along rivers rather
///     than flipping on a hard threshold boundary.
///
/// The `material_cubemap` the shader reads still uses discrete per-
/// texel nearest-vertex material IDs — that's correct for downstream
/// shader logic which selects per-material BRDF parameters. The
/// smooth-colour treatment is strictly a debug-albedo thing.
fn paint_materials_debug_albedo(
    builder: &mut BodyBuilder,
    sphere: &Icosphere,
    materials: &[u8],
    iron_fraction: &[f32],
    iron_fraction_threshold: f32,
) {
    let palette = builder.materials.clone();
    let peat = palette[MAT_PEAT_WETLAND as usize].albedo;
    let rust = palette[MAT_IRON_RUST_FLOODPLAIN as usize].albedo;

    // Iron overlay fade window: fully-peat at 0 iron-fraction, fully-
    // rust around 2.5× the classifier threshold. This gives a smooth
    // ramp so mildly-iron-stained floodplains read as warm peat and
    // heavily-stained ones read as saturated rust, with no visible
    // edge at the classifier's binary cutoff.
    let iron_ramp_top = iron_fraction_threshold * 2.5;

    // Pre-compute per-vertex final colour. For peat/rust vertices we
    // mix peat → rust by smoothstepped iron-fraction; every other
    // material pulls its flat palette entry. This keeps the "iron
    // staining" visually continuous rather than a binary flip along
    // the drainage graph.
    let vertex_colors: Vec<[f32; 3]> = materials
        .iter()
        .enumerate()
        .map(|(vi, &mid)| {
            if mid == MAT_PEAT_WETLAND || mid == MAT_IRON_RUST_FLOODPLAIN {
                let t = smoothstep(0.0, iron_ramp_top, iron_fraction[vi]);
                [
                    peat[0] + (rust[0] - peat[0]) * t,
                    peat[1] + (rust[1] - peat[1]) * t,
                    peat[2] + (rust[2] - peat[2]) * t,
                ]
            } else {
                palette[mid as usize].albedo
            }
        })
        .collect();

    let res = builder.cubemap_resolution;
    let inv = 1.0 / res as f32;

    for face in CubemapFace::ALL {
        let data = builder.albedo_contributions.albedo.face_data_mut(face);
        data.par_chunks_mut(res as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let mut last_v = 0u32;
                for (x, texel) in row.iter_mut().enumerate() {
                    let u = (x as f32 + 0.5) * inv;
                    let v = (y as f32 + 0.5) * inv;
                    let dir = face_uv_to_dir(face, u, v);
                    let (ti, w) = sphere.barycentric_triangle(dir, last_v);
                    let tri = sphere.triangles[ti as usize];
                    last_v = tri[0];

                    let c0 = vertex_colors[tri[0] as usize];
                    let c1 = vertex_colors[tri[1] as usize];
                    let c2 = vertex_colors[tri[2] as usize];

                    *texel = [
                        w[0] * c0[0] + w[1] * c1[0] + w[2] * c2[0],
                        w[0] * c0[1] + w[1] * c1[1] + w[2] * c2[1],
                        w[0] * c0[2] + w[1] * c1[2] + w[2] * c2[2],
                        1.0,
                    ];
                }
            });
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    if denom.abs() < 1e-6 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
