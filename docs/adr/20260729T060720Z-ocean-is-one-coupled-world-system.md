# ADR-20260729T060720Z-ocean-is-one-coupled-world-system: Ocean weather, waves, physics, and gameplay form one world system

- **Status:** Accepted
- **Date:** 2026-07-29

## Context

Thalos already has a stable analytic planet-scale ocean, one authored coastline,
filtered open-water slopes, shallow-water optics and a first dispersion-aware
visual tracer. It does not yet have displaced waves, shared wave physics,
persistent foam, moving storm systems or ocean traversal. Extending those as
separate rendering, vehicle and scenario features would make a storm's visible
waves disagree with buoyancy and let clouds, wind and hazards describe different
weather.

The product direction resolves the fork: oceans become a major traversal and
gameplay domain, with heavenly beaches and large offshore storms, and Pelagos
eventually depends on that foundation.

## Decision

- One evolving per-body weather authority owns storm identity and surface
  forcing. The existing `CloudWeatherField` evolves behind this broader seam;
  the ocean does not invent an independent storm map. Clouds, rain, visibility,
  local wind and ocean forcing project the same weather.
- One dynamic sea-state field responds to weather, bathymetry and currents.
  Rendering, buoyancy, wakes and hazards query that same state; there is no
  visual-only wave function beside a physics-only one.
- ADR-20260720T185954Z and ADR-20260720T185958Z remain in force: the exact global
  water surface is analytic and the signed sea-height field alone owns the
  permanent coastline. Displacement and shallow-water simulation are bounded
  local layers.
- The program is proved end to end on Thalos with one vessel, one showcase beach
  and one moving storm before Pelagos receives bespoke content. Pelagos is the
  culmination of the shared system, not its prototype.
- The first weather producer remains authored/procedural and deterministic.
  Global fluid or climate simulation may later replace a producer behind the
  same authorities; it is not a prerequisite for playable high-fidelity oceans.

## Alternatives

- **Continue as rendering polish only** — rejected because larger waves that do
  not move vehicles or alter decisions still read as animated wallpaper.
- **Separate visual, physics and gameplay sea presets** — rejected because they
  drift in phase, scale and severity and make storm bugs impossible to diagnose
  from one state.
- **Build a global fluid/climate simulation first** — rejected because it
  spends a very large simulation and validation budget before proving the
  beach-to-storm player experience. Structured weather fields can exercise the
  same downstream contracts.
- **Prototype directly on Pelagos** — rejected because the narrative destination
  would carry untested rendering, weather, vehicle, coast and underwater stacks
  at once. Thalos offers existing terrain, weather, infrastructure and faster
  iteration.
- **Replace the analytic ocean with a planet-scale water mesh** — rejected by
  ADR-20260720T185954Z; local displacement does not reopen the facet-sag and
  map-depth failure already settled there.

## Consequences

The implementation needs a deterministic wave-query boundary that both
rendering and simulation can consume, plus explicit observability for
render/query agreement, wave energy, weather forcing and vessel response.
Weather work must coordinate with the existing cloud authority instead of
creating an ocean-owned duplicate. Local spectral and shallow-water layers need
careful handoffs to the analytic surface and a measured GPU/memory budget.

In return, every later improvement strengthens one coherent ocean: a storm seen
on the horizon is the storm producing the rain, wind, swell, control difficulty
and eventual Pelagos operating constraint.
