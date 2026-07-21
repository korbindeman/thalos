# INC-0009: Mira's opaque horizon looked transparent — unresolved regolith normals aliased through Hapke

- **Status:** Fixed
- **Date:** 2026-07-20 (observed) / 2026-07-21 (fixed)
- **Severity:** visual
- **Surface:** `just game mira-eva`; canonical headless repro `just screenshot mira-eva`

## Summary

Mira's distant eye-level ridges broke into bright/dark single-pixel stipple that
looked like transparent terrain against space. The mesh was fully opaque and
rasterised. Mira's procedural regolith mottle, speckle, and micro-normal were
faded only by camera distance, so a grazing ridge could compress metre-scale
octaves below one pixel while still inside the fade band. Hapke's strong
incidence/emission response magnified those unresolved normal changes. The fix
feeds the fragment's body-space pixel footprint into the shared per-octave fBm
filters, allowing each octave to converge to its mean before it becomes
subpixel.

## Symptoms

- Nearby regolith was solid, but the distant horizon was a dense screen-door
  pattern whose dark samples visually merged with the black sky.
- Thalos did not show the same failure because its vegetated dielectric terrain
  path did not amplify the same airless micro-normal stack.
- Repro: `THALOS_TILE_CACHE=0 just screenshot mira-eva` after the preset's
  1,200-frame terrain warm-up.

## Evidence

The first screenshot framing reused an aerial orbit-boom camera and could place
the camera inside crater relief. `mira-eva` was corrected to use the canonical
EVA direction, live atlas-backed surface height, a 1.7 m eye height, and tangent
look. That faithful probe reproduced the user's solid foreground and stippled
ridge.

Controlled captures then established:

```text
fullbright terrain        ridge fully solid (opaque raster coverage present)
two-sided rasterisation   stipple remains / worsens
flat or analytic provider not a faithful surface match; package validates
self-shadow off           stipple remains
external shadows off      stipple remains
height-normal mip filter  stipple remains
geometric normal only     stipple disappears
footprint-filtered fBm    stipple disappears; close texture remains
```

The retained matched comparison is
`tools/agent_scratch/screenshots/comparisons/mira-eva/terrain-regolith-filter/`: 2048×1280
legacy-unfiltered versus footprint-filtered captures at identical camera/world
state, with a labelled contact sheet, wipe, diff, and manifest. Mean absolute
RGB change is 1.724/255 across the frame; 42.7% of pixels change because the
filter also correctly averages unresolved fine albedo detail.

The package profiler found a separate compatibility-producer defect while the
geometry hypothesis was being tested: f32 `acos(dot)` quantised exact crater
distance near `dot = 1`. Keeping the exact distance in f64 reduced the eight-way
30 km profile's worst adjacent 10 m height jump from 69.576 m to 5.377 m. This
improved the package, but did not remove the stipple, so it was not the root
cause.

## Hypotheses considered

- **Actual alpha/discard or missing triangles:** ruled out by the opaque terrain
  pipeline, no discard path, and a fully solid fullbright capture.
- **Back-face culling:** ruled out by a two-sided terrain pipeline comparison.
- **Missing/stale package tiles or bad resident-ancestor lookup:** package
  validation, cache-disabled repros, provider experiments, and lookup guards did
  not track the final artifact. Several early tests were invalidated by the
  aerial camera being inside relief.
- **Terrain self-shadow or shared object shadows:** separately bypassing each
  left the stipple unchanged.
- **Height-atlas relief normals:** selecting derivative-driven height mips is a
  valid general anti-aliasing improvement, but the canonical ridge still
  stippled.
- **Procedural regolith normal stack:** selected by the geometric-normal capture;
  the matched legacy/footprint A/B then confirmed it.

## Root cause

`regolith_detail` generated colour fBm down to fine dust scales and a roughly
1.25 m procedural micro-normal. Its only visibility control was a radial
camera-distance fade ending at 1.8 km. Screen frequency also depends on view
angle: at a grazing ridge the projected body-space pixel footprint spans many
of those octaves well before the radial fade ends. Sampling them once per
fragment aliases both colour and, more severely, the normal. Hapke converts
small normal changes near grazing incidence/emission into large radiance
changes, producing the stipple. The black samples resemble holes, but terrain
alpha remains one.

## Fix

- Pass the already-computed body-space fragment footprint to
  `regolith_detail`.
- Generate regolith mottle/speckle with `fbm3_value_faded` and micro-normal with
  `fbm3_perlin_grad_faded`, fading each octave according to its physical
  wavelength and the fragment footprint.
- Keep `legacy-regolith` only as the capture-side before variant for
  `just compare mira-eva terrain-regolith-filter`.
- Select a derivative-driven decoded height mip for height-atlas normals while
  retaining mip 0 for vertex geometry, preventing the same alias class in
  resolved relief.
- Keep exact crater angular distance in f64 and expose `terrain_baker diagnose`
  so future package-height discontinuities can be measured independently of
  shading.

## Prevention & recurrence signals

- Every procedural colour/roughness/normal octave must be footprint/Nyquist
  filtered; distance-only fades are insufficient at grazing view angles. This
  invariant is recorded in `mira_airless_mvp.md` and the WGSL/Bevy skill.
- Preserve `mira-eva` as the eye-level, live-height regression framing and the
  typed `terrain-lighting` / `terrain-regolith-filter` axes.
- A fully solid fullbright ridge plus a broken lit ridge is the fast recurrence
  tell. If geometric-normal is also clean, inspect procedural and height normal
  bandwidth before tile coverage or alpha.
