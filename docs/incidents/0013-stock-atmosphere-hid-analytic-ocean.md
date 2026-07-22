# INC-0013: Stock atmosphere hid the analytic ocean

- **Status:** Fixed
- **Date:** 2026-07-21 (observed) / 2026-07-21 (fixed)
- **Severity:** visual
- **Surface:** `just screenshot ocean`, `just screenshot ocean-slopes`, normal ship view with Bevy atmosphere enabled

## Summary

The default Bevy-atmosphere renderer displayed a flat olive seabed and no wave
field because its double-atmosphere guard hid the whole `BodySky` entity, which
still owned the analytic ocean. Water was moved to one dedicated compositor
that stays visible under either atmosphere backend; `BodySky` is now atmosphere-
only.

## Symptoms

- `just screenshot ocean` showed no dark water volume, sun road, or directional
  wave detail—only the terrain/seabed behind the missing surface.
- `just screenshot ocean-slopes` produced the same production-colour image
  instead of the expected red/green slope and blue variance diagnostic.
- Both runs exited zero and logged no shader/pipeline error.

## Evidence

The production and slope-diagnostic frames were visually identical even though
`OceanDebugSettings::slope_view` was set and propagated into
`BodySkyExtra::ocean_spectrum.w`. Code inspection then showed the decisive
visibility conflict:

```text
sync_body_render_lod: BodySky -> Visibility::Inherited
suppress_body_sky_for_stock_atmosphere: active BodySky -> Visibility::Hidden
body_sky.wgsl: the sole ship-view analytic-ocean branch
```

The diagnostic flag was correct; the material containing the branch never
rendered.

## Hypotheses considered

- **Ocean slope texture/binding was broken by the merge.** Ruled down because
  the production and diagnostic outputs did not merely lack slope variation;
  the false-colour branch was absent altogether.
- **The spectral BRDF was too dark.** Ruled out by the unchanged diagnostic,
  which bypasses the BRDF and sun road.
- **A WGSL or bind-group validation failure removed the pass.** Ruled out by
  clean process output and successful pipeline creation.
- **Atmosphere visibility suppressed the physical surface owner.** Confirmed by
  the render-LOD/suppressor ordering and `BodySky`'s sole ownership of water.

## Root cause

Atmosphere and ocean were separate physical responsibilities implemented by one
entity. Promoting Bevy atmosphere to canonical correctly hid the superseded sky
renderer, but entity-level hiding also deleted the unrelated analytic-ocean
projection. The atmosphere toggle therefore changed world topology.

## Fix

`BodyOceanMaterial` now owns the analytic-sphere projection independently. It
reuses the canonical signed-field/spectral binding and shader implementation in
ocean-only mode; `BodySkyMaterial` compiles the same source atmosphere-only.
The ocean follows terrain residency and receives current body/sun/phase/tile
state through one sync path. Explicit ordering places ocean after the legacy
atmosphere and before clouds.

## Prevention & recurrence signals

- A renderer-backend toggle may replace only that backend's projection; it must
  not own visibility for unrelated physical surfaces.
- Every ocean change must pass both `ocean` and `ocean-slopes` with the canonical
  Bevy atmosphere. If the two frames become identical, first inspect whether
  the dedicated ocean entity rendered before debugging spectrum math.
- See ADR-20260721T050036Z-ocean-composite-independent-of-atmosphere and
  [ocean.md](../rendering/ocean.md).
