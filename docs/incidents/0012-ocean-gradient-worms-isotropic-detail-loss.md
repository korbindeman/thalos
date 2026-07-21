# INC-0012: Ocean gradient worms and premature detail loss

- **Status:** Fixed
- **Date:** 2026-07-21 (observed) / 2026-07-21 (fixed)
- **Severity:** visual
- **Surface:** `just screenshot ocean`; low-altitude open-water views, clearest outside the sun road

## Summary

The first BL-12 ocean field showed repeated closed C-shaped specular contours
in the foreground and became smooth far too early. These were two independent
failures: direct derivatives of sparse scalar gradient noise exposed the
closed neighborhoods around its extrema as “worms,” while a scalar grazing
footprint removed an entire wave band in every direction as soon as the
foreshortened view axis became subpixel. The fix replaces the shader noise with
a dense mipmapped broadband slope texture and samples it with explicit major
and minor surface gradients through a 16× anisotropic sampler. Resolved
cross-wave structure now survives toward the horizon; genuinely unresolved
slope variance transfers into GGX roughness.

## Symptoms

- Foreground highlights formed many similarly sized closed hooks / C-shapes,
  visible even in dark water away from direct sun glare.
- Wave structure crossed an obvious distance boundary into a smooth, broad
  GGX reflection while the reference retained visible ripples much farther.
- A stale `ocean_side.png` from a discarded carrier experiment also showed a
  separate crosshatch; it predated the current screenshot and was not evidence
  about the live field.
- Repro: `just screenshot ocean`, then inspect the foreground outside the sun
  road and the non-glare path from foreground to horizon.

## Evidence

- User crop on 2026-07-21 isolated the repeated foreground hooks; the full
  capture showed the simultaneous mid/far flattening.
- Setting every choppiness coefficient to zero produced
  `ocean_no_chop.png`: the hooks were unchanged. Crest sharpening was ruled
  out as their source.
- Extending the old footprint fade from 0.34λ to 0.75λ produced
  `ocean_long_filter.png`: detail reached materially farther, but the hooks
  were unchanged. This separated distance filtering from field topology.
- Finite low-component carrier probes removed the hooks but exposed straight
  line families (`ocean_spectrum_probe.png` through
  `ocean_nested_drag_strong.png`), showing that swapping one sparse basis for
  another was not sufficient.
- Final deterministic capture `ocean_final.png`: no closed hook population,
  no white foam scars, and continuous visible structure through the mid/far
  field and horizon band.

## Hypotheses considered

1. **Crest/choppiness shaping folded the normal into loops.** Ruled out by the
   zero-choppiness capture; topology was identical.
2. **Post-processing or atmospheric haze invented the contours and blur.**
   Ruled out because controlled changes to only the wave evaluator changed the
   contour basis, and extending only its footprint cutoff moved only the
   detail-loss boundary.
3. **The footprint cutoff was responsible for both symptoms.** Half true:
   extending it fixed the reach but not the hooks, proving two causes.
4. **Sparse scalar gradient-noise derivatives exposed extrema contours.**
   Confirmed by the unchanged no-chop result and by replacing the basis: the
   hooks disappeared immediately, although insufficiently dense Fourier probes
   then revealed their own carrier lines.
5. **A longer isotropic cutoff was the production fix for the horizon.**
   Rejected: the analytic surface footprint grows as `1 / |ray·normal|` at
   grazing incidence. It describes a long, thin ellipse, not a large disc;
   retaining everything would alias, while isotropically discarding everything
   erased still-resolvable cross-view waves.

## Root cause

Two mechanisms compounded:

1. The old normal was the analytic derivative of a small number of periodic
   scalar gradient-noise bands. Around every scalar extremum the gradient turns
   through a closed neighborhood. A low-roughness water BRDF maps those
   repeated gradient orientations into bright closed contours, making the
   noise construction legible as similarly sized “worms.”
2. `ocean_fp_m` correctly represented the **major** surface distance swept by
   a pixel along the projected view ray, including its grazing-angle
   `1 / |ray·normal|` growth. The wave filter incorrectly treated that scalar
   as an isotropic footprint and faded each whole band. Near the horizon, the
   major axis can be hundreds of times the minor axis, so the code erased
   cross-view frequencies the screen still resolves.

## Fix

- Added `ground/ocean_slope.rs`: a deterministic 256² RGBA8 slope texture with
  two independent 128-mode directional spectra. Integer, non-duplicated wave
  vectors make it periodic without collapsing the lowest octave into a few
  carriers; a CPU-authored 2×2-average chain supplies every mip.
- `BodySkyMaterial` binds that texture once for all oceans. `body_sky.wgsl`
  samples it over four overlapping body-fixed physical domains (8192, 1024,
  128, and 16 m), preserving the existing f64-reduced world phase.
- The analytic sea intersection now computes both pixel-ellipse axes and the
  major tangent direction. `textureSampleGrad` plus a 16× anisotropic repeat
  sampler filters the foreshortened axis while preserving cross-wave detail.
- Mip-omitted slope variance feeds GGX alpha, so filtered geometry becomes
  statistical glitter energy rather than a flat mirror.
- Open-water foam thresholds now require exceptional resolved slopes; the
  initial retune's pale diagonal scars are not accepted as wave detail.

## Prevention & recurrence signals

- The standing contract is [ocean.md invariant 4](../ocean.md#1-invariants):
  ocean LOD is directional at grazing incidence. Never drive a 2-D normal
  field from the major footprint alone.
- Do not expose a sparse scalar-noise derivative directly as a low-roughness
  water normal. Inspect the BRDF output, not just a grayscale height preview;
  repeated hooks reveal extrema topology, while repeated straight lines reveal
  an under-populated carrier spectrum.
- Every ocean screenshot acceptance pass must inspect a non-glare foreground
  crop and an unbroken foreground-to-horizon path. The sun road alone hides
  both recurrence signatures.
