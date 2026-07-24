# INC-20260724T170955Z-diffusion-window-north-south-mirror: diffusion detail window rendered vertically mirrored

- **Status:** Fixed
- **Date:** 2026-07-24 (observed) / 2026-07-24 (fixed)
- **Severity:** visual (terrain content wrong everywhere in the 553 km window except its center)
- **Surface:** any Thalos view under `THALOS_TERRAIN=diffusion`; found by the NTR-X4
  `massif-aerial` showcase preset (`just screenshot massif-aerial`)

## Summary

`DiffusionSurface`'s tangent drape rendered the 553 km / 90 m detail window
**north-south mirrored about the site**: its "north" basis vector was computed
as `site_dir.cross(east)`, which points *south* (ENU north is `east × up`).
Every raster row rendered at the mirrored latitude, so the exported massif at
+50 km N appeared at −50 km S, and the residual band silently fought the
chart it was conditioned on. Invisible at the window center — the flip's fixed
point — which is exactly where the spaceport site sits and where all NTR-X2a
verification framed. Fixed by correcting the basis (`east.cross(site_dir)`)
and bumping `GENERATOR_VERSION` 17→18 (generation output changed; cached
tiles would otherwise keep the mirrored ground).

## Symptoms

- The new NTR-X4 showcase preset `massif-aerial`, framed at the 5799 m peak
  location derived from the raster (lat 8.5015 / lon 178.3756), captured low
  rolling ~800 m terrain — no massif, no snowline.
- Repro: `THALOS_TERRAIN=diffusion just screenshot massif-aerial` before the
  fix; the fixed-site log line reports `height 841 m` where the raster says
  5799 m.

## Evidence

- Instrumented `fixed_site_context` (a one-line INFO in the capture log):

```
fixed site (8.5015, 178.3756): height 841 m, diffusion=true
```

  Diffusion active, site resolved as authored — but the sampled height is the
  value the raster holds at the *mirrored* row.

- Orientation ground truth: the window residual was conditioned on the global
  chart, so only one row convention can correlate. Downsampling the window to
  64² and correlating against chart samples at each cell's lat/lon:

```
row0=north: corr(chart, window) = +0.950
row0=south: corr(chart, window) = -0.130
```

  The raster is row-0-north (standard image convention, matching the export
  sidecar and `chart_px`'s own `0.5 − lat/π` row mapping); the engine's drape
  was the flipped side.

## Hypotheses considered

1. **Diffusion backing not active in the capture host** — ruled out by the
   instrumented log line (`diffusion=true`; earlier confusion came from a
   parallel user-run capture host started without the env var).
2. **Fixed-site context frame mismatch** (surface vs ephemeris frame,
   INC-20260724T001023Z class) — ruled out: the same
   `surface_orientation_authored` path serves the verified dry-belt/ocean
   contexts, and the logged lat/lon matched the authored values.
3. **LOD starvation (massif not yet streamed)** — ruled out: even the 23 km/px
   chart renders the massif as a ~4 km dome; the captured relief was ~100 m.
4. **Wrong peak coordinates from the offline raster scan** — ruled out by the
   chart-correlation test above: the scan's row-0-north reading is the
   correct one; the engine disagreed with both the raster *and* the chart.

## Root cause

In `DiffusionSurface::load` (crates/domain/terrain/src/diffusion_surface.rs),
the drape tangent frame computed `north = site_dir.cross(east)`. With
`up × east` ordering that vector points geographic **south** (ENU north is
`east × up`). `detail_px` maps `dy = side/2 − dir·north·R / 90`, so every
sample landed on the row mirrored about the window's horizontal centerline.
The window center — the authored spaceport site — is the mirror's fixed
point, so site-anchored verification (flatten, base build, "looks gorgeous"
fly-through) could not catch it; the mis-placed geography read as plausible
terrain in every free-flight framing.

## Fix

- `north = east.cross(site_dir)` (true ENU north), with a comment pinning the
  convention.
- `GENERATOR_VERSION` 17 → 18: the disk tile cache keys generation identity
  through it, and the content fingerprint (chart payload + detail length)
  does not change under this fix — without the bump, cached runs would
  silently keep rendering the mirrored ground.

## Prevention & recurrence signals

- **Rule:** a tangent-frame basis is ENU only as `east × up = north`
  (equivalently `north = east.cross(up)`); `up.cross(east)` is south. When
  building any drape/anchor frame, verify with a known off-center landmark,
  not the anchor point — the anchor is the fixed point of exactly the
  mirror/rotation errors the frame can hide.
- **Tell:** an off-center feature whose raster-derived coordinates and
  in-engine location disagree in latitude sign about the window center; or a
  conditioned band that anti-correlates with its conditioning chart on one
  axis. The NTR-X4 fixed-site log line (`fixed site (lat, lon): height H`)
  vs the raster's value at that lat/lon is a one-look check.
