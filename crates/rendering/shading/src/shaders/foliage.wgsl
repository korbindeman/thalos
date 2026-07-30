// Shared foliage MATERIAL model — the single source of truth for a tree/shrub
// fragment's intrinsic (view/light-independent) surface colour.
//
// This is the material analogue of `thalos::lighting::shade_foliage`: that
// function is the one *lighting* routine every foliage representation shares;
// this is the one *albedo* routine they share. Every tree representation derives
// its leaf/bark colour from here —
//
//   * the near mesh trees   (`tree.wgsl`        fragment), and
//   * the octahedral impostor bake (`tree_bake.wgsl` fragment),
//
// so the impostor captures EXACTLY the colour the near trees show and the
// mesh→impostor handoff cannot drift. Change the look here and both the near
// canopy and the (startup-rebaked) impostor band move together — that is the
// systematic guarantee that impostors track tree updates. Keep this function
// free of any view/sun term: directional brightening is the lighting model's
// job (`shade_foliage`), and anything sun-dependent here would bake fake
// lighting into the impostor atlas.

#define_import_path thalos::foliage

// Dynamic canopy grade: a multiplier on the atlas leaf colour driven by an
// exposure term — toward a deep shaded green (exposure 0, the recesses between
// lobes) or a bright yellow-green highlight (exposure 1, the sunlit top/outer
// surface). Grading the atlas colour (rather than replacing it) keeps each
// leaf's tone variation, so the leaf cards still break up instead of reading as
// flat sheets.
// The lit end stays bright on purpose — a sunlit leaf really is a high-albedo
// surface, and that highlight is what reads as foliage rather than paint. What
// was wrong is the SHADED end: at 0.40/0.54/0.45 the darkest recess of a crown
// sat only ~1.8× below the sunlit top, so a crown had almost no interior depth
// (it read as a flat blob — the "canopies look thin" report) and its *mean*
// landed at the brightness of open meadow. Measured against the `forest-stand`
// preset, stand-vs-open-ground luma at matched distance ran 0.97 near / 1.01 mid
// / 1.04 far — i.e. canopy the same as or BRIGHTER than the meadow it should be
// clearly darker than, inverting further with distance. Real closed canopy is
// darker than grassland (albedo ~0.05–0.10 vs ~0.15–0.20) because most of what a
// pixel integrates is shadowed interior and gaps, and that only strengthens as
// the crown subtends less. Deepening the recess restores both the intra-crown
// depth and the aggregate value, in ONE place — the near mesh and the
// startup-baked impostor atlas both read this, so they move together.
fn canopy_grade(e: f32) -> vec3<f32> {
    let shaded = vec3<f32>(0.14, 0.20, 0.16); // deep shadowed interior (cool recess)
    let lit = vec3<f32>(1.38, 1.26, 0.80);    // gentle brighten + mild yellow (sunlit)
    return mix(shaded, lit, e);
}

// Per-leaf exposure from the mesh-baked occlusion carried in the vertex-colour
// green channel. The tree species set `canopy_color.g == 1.0`, so the baked
// `color.g` is pure AO (≈ height-in-crown × cluster form × occlusion). The env
// folding constant (0.92) keeps the canopy's mean brightness where the old
// sun-facing `env` term left it; the *directional* sunlit-leaf pop the env term
// used to fake now comes from the shared `shade_foliage` wrap-diffuse, so this
// stays view-independent and the impostor bake reproduces it exactly.
fn foliage_exposure(vcolor_g: f32) -> f32 {
    let baked = clamp((vcolor_g - 0.42) / 0.58, 0.0, 1.0);
    return clamp(baked * 0.92, 0.0, 1.0);
}

// Per-instance hue jitter moved CPU-side (2026-07-30): it is folded into the
// instance landcover tint at scatter time (`scatter.rs`, `SALT_HUE`), because
// only the scatter generator knows the body-global Poisson cell — the shaders
// hashed the TILE-relative root, which gave the same tree two different hues
// while both clipmap rings drew it during the complementary cross-fade. Still
// never baked into the atlas, so a re-baked impostor inherits no stamped tint.

// The one foliage albedo function. Returns the intrinsic, view/light-independent
// base colour of a foliage fragment. `leaf_flag` 1 = translucent foliage (leaf /
// needle), 0 = opaque shell / bark. `atlas_rgb` is the foliage-atlas albedo
// sample; `vcolor_g` is the baked AO in vertex-colour green; `seed` drives only
// the opaque brightness jitter (foliage hue is applied separately, per
// instance, folded into the instance tint CPU-side — see the note above).
// Both the near mesh and the impostor bake call THIS,
// on the same atlas sample + vertex colour, so their base colour is pixel-equal.
fn foliage_base_albedo(
    atlas_rgb: vec3<f32>,
    vcolor_g: f32,
    leaf_flag: f32,
    seed: f32,
) -> vec3<f32> {
    if (leaf_flag < 0.5) {
        // Opaque shell / bark: full painterly atlas colour (decoupled from the
        // dark trunk tint) with a subtle per-instance brightness jitter.
        return atlas_rgb * (0.94 + 0.12 * seed);
    }
    // Foliage: grade the atlas leaf colour by the baked exposure, then naturalize
    // — pull the vivid green toward a muted olive (desaturate + warm the grey
    // point) so the canopy reads as real foliage, not cartoon neon.
    let leaf = atlas_rgb * canopy_grade(foliage_exposure(vcolor_g));
    let luma = dot(leaf, vec3<f32>(0.30, 0.59, 0.11));
    let olive = vec3<f32>(luma * 1.07, luma * 0.99, luma * 0.64);
    return mix(olive, leaf, 0.62);
}
