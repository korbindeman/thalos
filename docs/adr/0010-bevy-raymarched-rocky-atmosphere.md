# ADR-0010: Bevy raymarching is the canonical rocky-body atmosphere

- **Status:** Accepted
- **Date:** 2026-07-20

## Context

Thalos had two atmosphere renderers: the shipping `BodySky` fullscreen
single/multi-scatter integration and an experimental Bevy 0.19 atmosphere
toggle. The custom path produced an oversized white limb and weakly coupled
surface haze in the Earth-reference orbital view. The original Bevy experiment
was also not a valid comparison: its atmosphere lived in the planet grid while
the camera used a floating render origin, moving the raymarched shell away from
the visible terrain.

The visual target is an ISS-like Earth photograph: a narrow saturated-blue
limb, readable surface beneath distance-dependent aerial perspective, and one
continuous solution from ground to orbit. Bevy 0.19 provides that solution as
`AtmosphereMode::Raymarched`; `bevy_fs` demonstrated the same renderer around a
spherical BigSpace planet.

## Decision

Bevy's `AtmosphereMode::Raymarched` is the canonical rocky-body atmosphere.

- The active render-view body is projected onto one camera-local `Atmosphere`
  proxy. Its center is `camera_global + body_center_relative_to_camera`, with
  the relative term computed from the f64 body-fixed `ViewAnchor`, so
  BigSpace/floating-origin offsets cancel exactly (INC-0007).
- Each camera with the atmosphere uses `AtmosphereSettings` in raymarched mode.
- Authored Rayleigh scale heights and spectral optical depths feed a Bevy
  `ScatteringMedium`; the Earth projection adds Bevy's Mie split and ozone.
- A single `0.1` density calibration applies to scattering and extinction while
  Thalos terrain remains in the spine's arbitrary scene-flux units. F7's shared
  photometric scene/atmosphere binding is the place to retire that adapter.
- The custom `BodySky` atmosphere remains only behind the
  `legacy_body_sky` debug comparison setting and the deterministic screenshot
  override. It is not a second normal rendering path.
- Clouds are deliberately outside this decision. Non-atmospheric fullscreen
  composites that still live in `BodySky` migrate as their own backlog slices.

## Alternatives

- **Keep repairing the custom raymarch** — rejected because it duplicates a
  maintained Bevy 0.19 renderer and the Earth-reference capture still showed a
  broad white halo after substantial bespoke tuning.
- **Attach `Atmosphere` directly to every BigSpace body grid** — rejected
  because Bevy extracts ordinary `GlobalTransform`s while the view and terrain
  are floating-origin projections; the matched capture exposed an off-planet
  atmosphere wedge. One view-local proxy is the precise N-body projection.
- **Keep a permanent user-facing choice between two atmosphere renderers** —
  rejected because the consolidation sprint requires one canonical operation.
  The legacy checkbox is explicitly diagnostic and can be deleted after visual
  verification.
- **Adopt `AtmosphereEnvironmentMapLight` from `bevy_fs` immediately** —
  deferred because the camera already owns a `GeneratedEnvironmentMapLight`
  supplied by the F3/F4 atmosphere-derived reflection probe; installing both
  creates two writers for the same generated environment component.

## Consequences

The sky, limb, and opaque-scene aerial perspective now come from one maintained
raymarch and share Bevy's directional light, exposure, and depth. The active
atmosphere follows every camera mode through `ViewAnchor`, and the fixed 3:2
`earth-reference` capture makes future tuning reproducible.

The temporary density calibration records a real photometric mismatch rather
than hiding it in several artistic knobs. It must be revisited when F7 puts the
terrain and Bevy-lit surfaces in one radiance unit system. Daylight star
suppression and the retained non-atmospheric `BodySky` composites remain
separate follow-up work; neither justifies restoring a second sky renderer.
