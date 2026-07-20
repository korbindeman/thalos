# INC-0004: Distant terrain went grey, shiny, and pixelated — mask stencil step clamped below coarse texel spacing

- **Status:** Fixed
- **Date:** 2026-07-20 (observed) / 2026-07-20 (fixed)
- **Severity:** visual
- **Surface:** any high-altitude / orbital view of a vegetated body (`just screenshot spaceport-aerial` with `THALOS_SCREENSHOT_DISTANCE` ≳ 50 km; user screenshots from orbit)

## Summary

From ~50 km up, Thalos degraded into a grey vermicular maze with a broad dappled
tan sun-glint, and the look of any given point changed with camera distance. The
material-mask bake (`populate_material_masks` in
`crates/body_render/src/ground/pipeline.rs`) computed slope and curvature from
neighbouring texels but divided by a step clamped to **250 m**, while coarse-tile
texels really span up to ~20 km — inflating slope/laplacian by `tile_lod_m / 250`
(up to ~80×). The saturated **rock** mask painted the planet grey, and the
saturated **wetness** mask tightened specular roughness (0.92 → ~0.41) into a
km-scale glint mottle. Fix: divide by the true texel spacing.

## Symptoms

- Grey dominates the land at high altitude; green only survives near-surface.
- A dappled, shiny tan mottle around the sun's specular region from orbit.
- The mottle/maze pattern is "pixelated" — its feature scale is the tile texel
  grid (km-scale blobs) — and shifts as tiles change LOD with camera distance,
  so one surface point is not consistently coloured.
- Repro: `THALOS_SCREENSHOT_DISTANCE=100000..1800000 just screenshot`.

## Evidence

- `material_masks_from_heights` receives `step_m = tile_lod_m.clamp(2.0, 250.0)`
  while the height stencil taps are `tile_lod_m` apart. `tile_lod_m` at LOD 0 on
  Thalos (R = 3186 km, detail grid ~256 texels/tile) ≈ 20 km/texel → gradients
  divided by 500 m instead of 40 km → slope ~80× too steep;
  `smoothstep(0.20, 0.75, slope)` saturates → `rock ≈ 0.82+`; the laplacian
  `((Σ/4 − h)/step)` inflates identically → `hollow`/`wetness` saturate in every
  km-scale concavity.
- Shader side (`body_terrain.wgsl`): `wetness` tightens GGX roughness
  `mix(roughness, roughness * 0.45, wetness)` — base roughness is a flat 0.92
  (`ProceduralSurface::sample_d`), so the sheen could only come from the wet
  mask; the grey could only come from `rock_w` (shader `SOIL/ROCK_STRENGTH` are
  0 and the eco-band greys are altitude-gated, LOD-stable).
- After the fix, screenshot brackets at 8 km / 100 km / 400 km / 1800 km show
  green plains + altitude-banded rock/snow ranges, no maze, no glint; the 8 km
  near field (always < 250 m/texel) is pixel-identical in character.

## Hypotheses considered

1. **Shader eco-bands read the render-LOD height** (user's initial hypothesis —
   "coloring based on in-shader height that shifts with LOD"). Partially real
   but ruled out as the driver: `ProceduralSurface` height is LOD-invariant by
   design (octave fades are small-amplitude), and the band thresholds are
   hundreds of metres wide — coarse height error can't flip whole plains grey.
2. **Baked macro albedo grey band** (`albedo_at` mixes toward rock above 900 m).
   Ruled out as driver: the shader blends macro tint at only 10 %.
3. **Specular/Fresnel at grazing angles.** Ruled out: base roughness 0.92 is
   matte, and specular-AA widens it further at range; only the wet-tightened
   0.41 lobe can glint, pointing back at the wetness mask.
4. **Mask stencil step vs texel spacing mismatch.** Confirmed by arithmetic
   (above) and by the screenshot A/B: the artifact scale matches the texel
   grid, and every affected LOD is exactly the set with `tile_lod_m > 250 m`.

## Root cause

`populate_material_masks` divided texel-spaced height differences by a step
clamped to 250 m, overestimating slope and curvature by `tile_lod_m / 250` on
every tile coarser than 250 m/texel. The masks are re-baked per LOD, so the
error grew with distance and re-patterned on every LOD swap.

## Fix

`step_m = tile_lod_m.max(1.0)` — the divisor is the real tap spacing. Coarse
tiles now measure genuinely coarse (smoother) slopes.

Follow-up (same day): a raw coarse-baseline slope *under*-reports fine-scale
steepness on fractal terrain (RMS slope ∝ `L^(H−1)`), which left mountain
flanks reading green from orbit. The **rock** response therefore reads a
statistically compensated slope — `slope × (step_m / 30 m)^0.35`, ratio clamped
≥ 1 so tiles at/below the reference spacing (the near field the thresholds were
tuned on, and the grass-placement callers) are exact no-ops
(`SLOPE_REF_STEP_M` / `SLOPE_SPECTRUM_EXP` in `pipeline.rs`). Soil and
curvature (wetness) stay uncompensated — boosting soil repainted the plains
with a brown mottle, and boosting wetness was the glint mottle itself; both are
fine-scale phenomena that should fade at range. `GENERATOR_VERSION` bumped
5 → 8 across the iteration (masks are part of the cached tile payload).

## Prevention & recurrence signals

- **Standing rule:** any stencil over baked tile texels must derive its metric
  step from the tile's actual texel spacing (`tile_lod_m`), never a fixed
  constant — and any per-LOD-baked quantity should converge toward the filtered
  limit of its finer LODs, or a point's appearance will depend on camera
  distance.
- **Tell:** artifact features whose spatial scale tracks the tile texel grid
  (km-scale blobs that re-pattern as tiles swap LOD) point at a per-LOD bake
  input, not at the shader; check the mask/attachment bake before the shader.
- Related: INC-0003 used the same distance-bracket screenshot probe; the
  `THALOS_SCREENSHOT_DISTANCE` bracket sweep is the standard falsifier for
  LOD-dependent appearance bugs.
