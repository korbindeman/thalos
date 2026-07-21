# ADR-20260721T185221Z-custom-rocky-atmosphere: The shared custom raymarch is the canonical rocky-body atmosphere

- **Status:** Accepted
- **Date:** 2026-07-21
- **Supersedes:** ADR-20260721T032343Z-bevy-raymarched-rocky-atmosphere

## Context

ADR-20260721T032343Z promoted Bevy 0.19's `AtmosphereMode::Raymarched`
after an orbital comparison showed a narrower limb than Thalos's `BodySky`
raymarch. The integration required a camera-local atmosphere proxy because
Bevy extracts ordinary f32 transforms while Thalos renders through a
floating-origin, N-body `ViewAnchor`. It also required separate suppression
and material-layout paths around the custom terrain, impostor, ocean, and
cloud projections.

Matched `runway-atmosphere / atmosphere` and `earth-reference / atmosphere`
captures on 2026-07-21 reversed the visual result: Bevy laid a nearly uniform
blue veil over distant surface geometry and erased most orbital surface
contrast, while the custom renderer retained natural distance structure.
Live surface-to-distance use additionally showed the Bevy atmosphere
disappearing or failing to cover parts of the composed world. These are
projection, lifecycle, and media-ordering failures rather than coefficient
tuning. The custom optical model already drives the sky-view LUT, environment,
terrain aerial recession, ocean, and analytic cloud fallback, so Bevy also
created a second atmosphere authority inside the one-world lighting system.

## Decision

The shared Thalos atmosphere raymarch is the sole rocky-body atmosphere.

- `BodySkyMaterial` renders the atmosphere for every resident terrain view,
  clipped against the shared opaque scene depth.
- `thalos::atmosphere` and its CPU `SkyViewLut` remain the one optical model
  for sky, aerial perspective, environment lighting, ocean, and cloud
  coupling.
- The Bevy `Atmosphere` proxy, `AtmosphereSettings`, Bevy-LUT cloud bindings,
  terrain atmosphere-layout specialization, screenshot backend override, and
  `atmosphere` comparison axis are deleted rather than retained as a parallel
  fallback.
- The deterministic `runway-atmosphere` and `earth-reference` presets remain
  the surface and orbital regression probes. The custom limb's width/colour
  is tuned inside the shared model, not by adding another renderer.

## Alternatives

- **Continue tuning Bevy's scattering medium** — rejected. Density, aerosols,
  ozone, shell height, and sample count can change colour and smoothness, but
  cannot repair camera-anchor disappearance, omitted render layers, or the
  foreground/background media-order split.
- **Keep Bevy as an orbital-only renderer** — rejected. A range-dependent
  backend switch would create a visible handoff and two optical authorities,
  directly opposing the consolidation sprint.
- **Keep Bevy as a permanent diagnostic A/B** — rejected. The matched
  evidence has resolved the fork; retaining dead proxy/layout/bind-group
  machinery would keep causing integration failures and maintenance cost.

## Consequences

Rocky atmospheres again have one renderer and one radiometric model from the
surface through orbit. The camera-local proxy and all backend-switch state are
gone, so atmosphere visibility follows the existing body render-LOD lifecycle
instead of a second extracted-component lifecycle.

The custom orbital limb still needs focused width/white-point calibration.
That is a bounded tuning task within the shared raymarch. Clouds temporarily
use their existing analytic atmosphere coupling rather than Bevy's private
LUT resources; CLOUD-4 can bind the shared Thalos LUT explicitly when its
foreground/background media ordering is completed.
