# INC-20260724T221348Z-standin-detail-had-no-near-field-rolloff: tile-terrain stand-in detail was a low-pass, so it peaked where it should vanish

- **Status:** Fixed
- **Date:** 2026-07-24 (observed) / 2026-07-24 (fixed)
- **Severity:** visual
- **Surface:** low-altitude flight over any vegetated tile-rendered ground — user's ship view at ~700 m AGL over the spaceport, reproduced headlessly by `just screenshot runway-atmosphere`

## Summary

Flying low over the spaceport on the NTR-X tile renderer, the near-field ground read as
"deep-fried": house-sized dark/light blobs across ground that should be flat grass, with a
bright specular sheen off the low sun. Distant terrain looked fine. The cause was not the
terrain content, the mesh, or the lighting: every NTR-X4 material layer gates its detail
through `footprint_fade`, which is a **low-pass** — it retires a detail wavelength once the
fragment footprint grows past it, but returns `1.0` all the way down. So detail that exists
purely as a *stand-in for geometry we do not have yet* (the forest canopy stipple standing in
for scatter trees) sat at **maximum strength exactly where the real thing would be resolved**,
growing from legitimate 2 px/cell grain at 5 km to ~90 px/cell blobs at 150 m. The fix adds
the missing half of the window — a `footprint_band` with a near roll-off — and gives
vegetation its own, much lower, specular reflectance.

## Symptoms

- Near half of the frame covered in ~10–25 m irregular blobs with strong light/dark contrast,
  reading as violent micro-relief on flat ground; a bright sheen off the same ground at a low
  sun. Mid and far field unaffected and (per the accepted NTR-X4 rounds) good.
- Repro: `THALOS_TERRAIN=diffusion THALOS_TILE_RENDERER=1 just screenshot runway-atmosphere`
  (73 m above the flattened basin). Baseline captured to
  `artifacts/visual/runs/ntr-x4-nearfield/b0_baseline.png`.

## Evidence

All in `artifacts/visual/runs/ntr-x4-nearfield/`.

| capture | change | near field |
|---|---|---|
| `b0_baseline.png` | — | blobby + sheen |
| `b1_no_normal.png` | `normal_offset = 0`, all layer albedo intact | **clean, matte** |
| `b2_no_canopy.png` | canopy layer off, all normals intact | **clean, flat** |
| `fp_runway.png` | footprint false-colour | near field is < 4 m/px, reaching < 0.5 m at the frame edge |
| `fp_massif_ridge.png` / `fp_massif_valley.png` | footprint false-colour | accepted showcase framings bottom out at ~2 m / ~4 m per pixel |

The two ablations bracket it: killing the normal perturbation alone removes the artefact while
leaving every layer's albedo, and killing the canopy layer alone removes it while leaving every
other layer's normals. The only term in both sets is the canopy normal dimple.

The footprint probe is what turned the fix from a guess into a calibration: it showed the
problem zone and the accepted showcase framings occupy **disjoint** footprint ranges, so a
near roll-off closing under 4 m/px clears the former and barely touches the latter.

## Hypotheses considered

- **Terrain content / mesh too rough at high LOD** (`SPLIT_FACTOR` 6 meshes the near field at
  ~1.5 m, and `octaves_for_lod` fades in fine octaves) — *ruled out*: `b1_no_normal` renders
  the identical mesh perfectly smooth. The relief was shading, not geometry.
- **Layer albedo (canopy stipple / meadow mottle) painting the blobs** — *ruled out*:
  `b1_no_normal` keeps all layer albedo and shows no blobbing; the albedo contribution at this
  range is a faint tone shift.
- **Rock / scree layers misfiring on flat ground** (a slope-classification bug) — *ruled out*:
  `b2_no_canopy` leaves rock and scree fully enabled and is clean; the basin is flat, low, and
  below the treeline, so neither layer claims it.
- **Photometry (NTR-X5's exposure/ambient bridge)** — *ruled out as the cause of the crunch*:
  it is a per-pixel pattern, not a level shift, and it survives every brightness variation. The
  sheen is downstream of the same normals (see below), not a separate lighting defect.
- **Canopy normal dimple with no near-field roll-off** — *confirmed*, and it explains both
  halves of the report: an 18 m Perlin gradient at amplitude 0.30 perturbs the shading normal
  by up to ~17°, which both sculpts the fake relief and swings enough microfacets at the low
  sun to light up the stock dielectric F0.

## Root cause

`footprint_fade(f, on, off) = 1 - smoothstep(on, off, f)` is a one-sided window. It correctly
retires detail the footprint can no longer resolve (anti-aliasing, the udlod faded-fBm
discipline it was modelled on), but it is `1.0` for every footprint below `on_m`.

That is right for detail describing a **real surface property at every scale** (rock strata,
gully striation, canopy colour). It is wrong for detail that is a **stand-in for absent
geometry**: the canopy stipple exists to give aerial forest per-tree grain "long before real
scatter trees load", the meadow mottle stands in for ground-cover variety, the rock/scree grain
for micro-relief. A stand-in must *dissolve* as the real feature comes within resolution;
instead each grew to fill the screen, because nothing in the model knew the difference between
"too far to resolve" and "close enough that the real thing should have taken over".

The NTR-X4 layer stack was tuned entirely against 5–22 km aerial framings, where the
distinction never surfaces — those framings never sample below ~2 m/px.

## Fix

In `assets/shaders/tile_terrain.wgsl`:

- New `footprint_band(f, on_m, off_m)` = `smoothstep(STANDIN_OFF_M, STANDIN_ON_M, f)` × the
  existing far fade, with `STANDIN_OFF_M = 1.0` / `STANDIN_ON_M = 4.0` m per pixel — the
  measured gap between the near-field problem zone and the showcase framings.
- Stand-in terms moved onto the band: canopy stipple (tone + normal dimple), meadow mottle,
  rock isotropic grain normal, the whole scree grain layer. Structure that is real at any
  scale — rock strata banding, gully striation, canopy *colour* — keeps the plain far-only
  `footprint_fade`.
- The canopy tone was reformulated as `1.0 + stipple · (…)` so that when the band closes it
  collapses to `1.0` (flat canopy colour) instead of the previous quarter-strength blotch that
  the `max(canopy_fade, 0.25)` albedo floor would otherwise have kept painting.
- Per-class specular: `VEG_REFLECTANCE = 0.32` (F0 ≈ 0.016) for vegetated ground, versus the
  stock `0.5` (F0 = 0.04) that rock and snow keep. Vegetation scatters far more than it
  mirrors; at a low sun the stock value turned every perturbed normal into glitter.

Verified: `runway-atmosphere` near field is clean and matte (`f1_fixed_runway.png`);
`massif-aerial`, `massif-ridge`, `massif-valley`, `spaceport-aerial` keep their accepted
character. Measured cost at `massif-valley` (band on vs band neutralised, same host, same
framing): **mean |Δ| 2.3/255 (~1 %)**, p50 = 1, concentrated in the nearest rows — a slight
softening of the finest near grain, no structural change. Sky rows differ by ~0.2, which is
also the capture's repeatability floor.

## Prevention & recurrence signals

- **Standing rule (added to `docs/reference/showcase_patch_prompt.md`):** a detail term that
  stands in for geometry the renderer does not have yet needs a **band**, not a low-pass —
  it must fade out at *both* ends of the footprint range. Ask of every new layer: "what is
  this a stand-in for, and what happens when the real thing is in reach?" Terms describing a
  genuine surface property at every scale keep the one-sided fade.
- **Tune against the framing's footprint, not its distance.** Before picking a threshold,
  false-colour `length(fwidth(p))` and read the actual range each framing samples; the
  showcase presets and a low-altitude flight view differ by more than an order of magnitude.
- **Recurrence tell:** detail whose apparent cell size *grows* as the camera approaches —
  texture that "zooms in" with the view instead of resolving into finer structure. If a
  material looks right from 5 km and deep-fried from 200 m, suspect a missing near roll-off
  before suspecting terrain content, mesh LOD, or lighting.
