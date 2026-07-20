# INC-0003: Black continents from orbit + dotted land-through-water coast speckle

- **Status:** Fixed
- **Date:** 2026-07-19 (observed) / 2026-07-19 (fixed)
- **Severity:** visual
- **Surface:** ship view of Thalos from orbital distance (any scenario, camera ≳ 13,000 km out); coast speckle visible from any high vantage, worst near the limb

## Summary

Two independent bugs that compounded into "the planet looks broken from orbit".
(1) Beyond ~13,000 km camera distance every **land** pixel of Thalos went pure
black while the ocean and atmosphere limb stayed lit — the planet read as an
atmosphere ring around a black hole. Cause: `body_sky.wgsl` classified the
body's *own terrain* as a "far background celestial body" past a fixed
`atmos_top_r * 4` distance cutoff and applied the star-crushing sky-pixel
treatment to it. Fixed by classifying "near geometry" as *anything inside the
ray's atmosphere-shell segment* (`t_scene <= shell exit`), which is scale-free.
(2) Coastlines speckled with dotted land-through-water texels that shifted with
camera distance, smearing into streaks near the limb. Cause: the ocean-side
hypsometric transfer opened with a zero-derivative `smoothstep`, leaving tens of
kilometres of shelf only ~1–3 m below sea level — inside the renderer's f32
depth-error floor — so seabed texels z-fought the analytic sea sphere. Fixed by
giving the shelf shoulder a quadratic ease-out (full slope at the waterline), so
depth clears the error floor within about a coarse-LOD texel of the shore
(`GENERATOR_VERSION` 3 → 4).

## Symptoms

- From ~13,000 km out (≈ 4 × the atmosphere-shell radius) to the
  terrain↔impostor swap at ~531,000 km — essentially the whole orbital envelope
  — continents rendered as flat black silhouettes; ocean (analytic water) and
  the atmosphere limb rendered normally, giving a "second atmosphere around a
  black hole" look.
- A halftone-dot fringe along the black/lit boundary (the `atmos_top_r * 4`
  iso-distance contour wiggling across texel-scale terrain bumps).
- Independently of the threshold: dotted land-coloured speckle along coasts
  over what should be open water, worst near the limb where grazing incidence
  stretches each poked texel into a streak; pale shallow blobs hugging every
  coast. Pattern changed with camera distance / tile LOD.
- Repro: `THALOS_SCREENSHOT_DISTANCE=20000000 THALOS_SCREENSHOT_ELEVATION=35
  just screenshot` (black land + limb speckle), `..._DISTANCE=4000000` (clean
  inside the threshold, pre-fix).

## Evidence

- Headless captures `tools/screenshots/diag_far_baseline.png` (20,000 km:
  black continents, dotted limb fringe) vs `diag_near_baseline.png` (4,000 km:
  clean) — bracketing the computed `atmos_top_r * 4` = (3,186 km + 80 km) × 4 ≈
  13,064 km threshold.
- After the shader fix, `diag_far_fixed.png`: land shades correctly at
  20,000 km, but land-coloured dot streaks remain over coastal water near the
  limb — isolating the second, independent bug.
- `crates/terrain/examples/shelf_breach_probe.rs` (new): walks transects
  offshore from fine-LOD waterlines. Pre-fix, offshore shelf depth averaged
  **−3 m for 40 km** on the probed coast with 0% of samples above sea at any
  LOD — the generator never breaches; the speckle had to be *renderer* error
  (f32 depth reconstruction at 20,000 km range ≈ 2–4 m > shelf depth). Post-fix
  the same coast averages −15…−18 m.

## Hypotheses considered

1. **Terrain↔impostor swap misfiring** (terrain hidden, impostor absent) —
   ruled out: swap distance is ~531,000 km, far beyond the failure onset, and
   the black shape tracked the continent outline exactly (terrain *was*
   rendering and writing depth; only its colour was black).
2. **Coarse-LOD relief breaching the shelf cap into skerry fields** (the dot
   fields as real +14 m islets aliasing with LOD) — initially ruled out (the
   first probe's coasts happened to show 0% breach), then **confirmed as a
   third mechanism** once top-down captures survived the first two fixes: on
   shallow shelves elsewhere the `SHELF_BREACH_CAP_M` exp-cap turned relief
   peaks into flat +14 m mesa fields pocked with circular noise-dip water
   holes, and the breach coverage was LOD-dependent (probe: 40% at LOD 0 →
   13% fine) because the relief cascade is LOD-faded. A probe that samples
   only a few coasts can miss a regional mechanism — bracket visually too.
3. **Sky-pass misclassification at the `atmos_top_r * 4` cutoff** — confirmed:
   the black onset distance matches the computed threshold, and pixels beyond
   it take the sky-pixel path (full-column in-scatter integration through the
   planet interior + the perceptual luminance opacity boost that crushes
   stars → opacity ≈ 1 over near-zero in-scatter = black), while ocean pixels
   take the independent analytic-water branch and stay lit.
4. **Sub-error-floor shelf z-fighting the analytic sea sphere** — confirmed by
   the probe's −3 m/40 km shelf plus the streak geometry (grazing-angle
   amplification near the limb is the signature of metre-scale height ties).

## Root cause

Two independent mechanisms:

1. `body_sky.wgsl` distinguished "this body's surface" from "far background
   body seen through the atmosphere" by a **fixed distance cutoff**
   (`t_scene <= atmos_top_r * 4.0`). The cutoff encodes an assumption ("the
   camera is never more than ~4 shell radii from the body whose sky pass this
   is") that every orbital view violates. Beyond it, the body's own terrain got
   the far-body treatment: in-scatter integrated over the whole shell segment
   (mostly below ground → near zero) with the sky-luminance opacity boost →
   land composited to black.
2. The ocean hypsometric shoulder used `smoothstep`, whose derivative is zero
   at the waterline. Combined with the flat local continentalness gradient and
   the coastal relief fade (`COAST_BAND_M`), this produced vast foreshores
   1–3 m deep — shallower than the unavoidable f32 depth-reconstruction error
   at orbital camera ranges, so the water/land depth test (`t_ocean <=
   scene_t`) resolved pseudo-randomly per texel.

## Fix

- `body_sky.wgsl`: near/far geometry classification is now **membership in the
  ray's atmosphere-shell segment** (`t_scene <= t_exit * 1.001`, `t_exit` still
  the shell far exit at that point) — scale-free, correct at any camera
  distance; a background moon behind the shell still gets the far-body
  treatment (its hit lies past the shell exit).
- `procedural.rs`: shelf shoulder `smoothstep` → quadratic ease-out
  `x(2 − x)` (full slope at the waterline, flat at the shelf edge; same total
  depth/width, C¹ at the shelf edge). Foreshore now deepens immediately, so
  seabed clears the renderer's error floor within ~a coarse-LOD texel.
  `GENERATOR_VERSION` bumped 3 → 4 (disk tile cache re-keys).
- `procedural.rs` (second pass): **offshore relief never breaches sea level
  any more.** The `SHELF_BREACH_CAP_M` +14 m islet cap is replaced by an
  awash-reef fold (`AWASH_REEF_DEPTH_M`): would-be breaches shoal
  asymptotically to 2 m *below* the surface. Sea-level crossings now belong
  exclusively to the LOD-invariant macro field, so no waterline — mainland or
  islet — can move with camera distance. Probe post-fix: 0% breach at every
  LOD on every coast. (`GENERATOR_VERSION` → 5.)

**Not a bug (identified during verification):** one conspicuous dot field
survived every fix unchanged — a coastal plain pocked with circular ~50 km
water holes in a lattice-ish arrangement (top-down captures). An
azimuth-rotation capture proved it is surface-fixed, and it is identical at
every distance: these are **real, LOD-invariant macro lagoons** — the
continentalness detail fBm meandering along its land/sea threshold over a low
coastal zone, rendered correctly. It *looks* halftone-mechanical because value
noise arranges blobs on its lattice; that is authored-geography character for
the terrain iteration / plate-tectonics seam to address (backlog BL-6), not a
water-rendering defect.
- `body_sky.wgsl` (second pass): the deeper foreshore alone could not fix
  regions where terrain *legitimately* hovers within metres of sea level over
  wide areas (tidal-flat plains where the continentalness field meanders along
  its threshold) — top-down captures still showed halftone speckle hugging
  those coasts. The shoreline feather is now **error-aware**: the water/land
  branch runs across a view-distance-scaled tie band (`ocean_depth_err_m =
  5e-7 × t_ocean`, ~centimetres in flight, ~metres from orbit) and
  `shore_cov` ramps over the *signed* sphere-vs-depth gap widened by that
  band, so terrain inside the renderer's own error floor composites as a
  smooth translucent shallow wash instead of a per-texel hard land/water
  call.

## Prevention & recurrence signals

- **Never encode a camera-distance assumption as a fixed cutoff in a fullscreen
  body pass.** Classify geometry by *geometric relationship to the body/shell
  along the ray* (inside/outside the shell segment), which holds at any
  distance. (This is the same class of bug as a "4 × radius" LOD constant —
  Slice 6 already replaced those with screen-size tests.)
- **Seabed must out-run the depth-error floor.** Any terrain intended to read
  as underwater has to drop below sea level faster than the renderer's
  worst-case height error grows with viewing distance; a transfer curve with a
  zero-derivative start at the waterline violates this by construction.
- **Every sea-level crossing must be LOD-invariant** — this generalizes the
  `COAST_BAND_M` lesson from the mainland shoreline to *all* waterlines:
  LOD-faded relief may shoal (awash reefs) but never cross 0, or the islets it
  makes appear/move with camera distance.
- Recurrence tells: continents (not ocean) going black at exactly one camera
  distance → a distance cutoff in the sky pass; land-coloured dot streaks
  radiating toward the limb over coastal water → sub-error-floor shelf depth.
  Probe with `shelf_breach_probe.rs` and bracketing `THALOS_SCREENSHOT_DISTANCE`
  captures.
