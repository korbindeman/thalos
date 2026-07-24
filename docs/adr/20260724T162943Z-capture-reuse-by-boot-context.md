# ADR-20260724T162943Z-capture-reuse-by-boot-context: Reuse capture hosts by boot context

- **Status:** Accepted
- **Date:** 2026-07-24

## Context

The persistent capture host originally treated the named framing as its process
identity. Moving from `spaceport-aerial` to `runway-atmosphere`, for example,
restarted the complete renderer even though both use the same Thalos runway
world and differ only in camera pose. Agent verification often needs several
framings, so restart-per-preset repeatedly pays GPU initialization, authored
world setup, terrain streaming, and temporal convergence.

Not every framing is interchangeable. Target body, spawn scenario, the hub
route, and viewport-sized render resources are app-builder inputs. Pretending
they can change live would require a general world teardown/rebuild mechanism
inside the capture runtime and create a larger state-leak surface than the
saved startup warrants.

## Decision

The persistent host is compatible by the explicit boot-context tuple:

`(target body, spawn scenario, hub mode, viewport, startup-override fingerprint)`.

Protocol v2 publishes the canonical presets compatible with the current host.
Camera framing and request-scoped diagnostic resources may change in-process;
a different boot context performs the existing managed restart. The
`thalos_capture shot` command accepts multiple scenes and walks those
boundaries automatically, grouping the remaining compatible scenes before
crossing into the next context. Comparison is a subcommand of the same
controller binary and uses the same lifecycle and validation path.

Every live override must be reversible when the next request omits it. A preset
that introduces startup-only state either makes that state request-scoped or
declares a distinct boot context. The initial startup fingerprint covers target
extent, terrain backing/renderer/culling, tile-cache mode, fixed runway site,
and wgpu backend.

## Alternatives

- **Restart for every named preset** — rejected because a framing name is not a
  world identity and needlessly multiplies the dominant startup cost.
- **Boot one universal superset world** — rejected because hub/no-craft,
  runway, EVA, orbit, and per-body scenes have materially different startup
  state; the superset would no longer be canonical evidence.
- **Implement arbitrary in-process world teardown/rebuild now** — rejected
  because it recreates game-mode orchestration inside the capture tool and adds
  state-leak risk without avoiding the renderer-resource work of a true context
  change.
- **Parallel cameras or render targets** — rejected by the one-camera visual
  testing contract: viewport-dependent LOD, shadows, SSAO, and temporal state
  would no longer be held constant.

## Consequences

Multi-framing runs amortize one stable world/GPU boot without experimental
compiler or runtime features. Compatibility is visible and testable rather
than inferred by the controller. Adding a preset requires keeping the protocol
catalog and runtime catalog aligned.

The tuple is intentionally conservative. A preset with a different default
viewport restarts even if an explicit override could make it fit; this trades a
possible extra boot for reliable viewport-resource ownership. Future
generalized scene transitions must supersede this ADR rather than silently
weakening the tuple.
