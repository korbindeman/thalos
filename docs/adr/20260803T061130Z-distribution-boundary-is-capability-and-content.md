# ADR-20260803T061130Z-distribution-boundary-is-capability-and-content: Distribution is bounded by capability and content, not by removing every developer tool

- **Status:** Accepted
- **Date:** 2026-08-03

## Context

Thalos is beginning to produce player distributions while many planet pipelines
are intentionally incomplete. At the same time, several tools commonly labelled
"debug" are useful inside the game itself: F3 makes performance legible and
teleportation makes a planetary sandbox navigable during pre-alpha. Other
development facilities reach outside the game, emit investigation-scale data,
or depend on checkout-only content such as saved viewpoints and test crafts.
Using Cargo's debug/release distinction as the product boundary would remove the
useful first group while still failing to define which assets and external
interfaces belong in a distribution.

## Decision

Player distributions are defined by an explicit capability and content
allowlist, not by compiling out everything associated with development.

- Self-contained in-game facilities may ship when they remain useful to a
  player or support diagnosis. F3 and teleportation are explicitly in this
  group.
- External authoring/control surfaces, checkout-facing integration, and
  investigation-scale debug dumping do not ship in the player application.
- Developer content does not ship merely because it is under `assets/` or
  `ships/`. Saved viewpoints and debug/test crafts are excluded; distributions
  carry an explicit stock-craft and runtime-content set.
- Bounded production diagnostics, build identity, and install verification are
  product capabilities rather than debug affordances.
- Pre-alpha planet acceptance may satisfy an incomplete stage through a
  declared fallback such as the existing solid-sphere representation. A
  deliberate fallback is valid content; an undeclared missing, corrupt, or stale
  artifact remains a release failure.
- Later acceptance profiles tighten per-planet requirements without changing
  the fallback mechanism or silently reinterpreting old packages.

Cargo's optimized profile remains a performance choice. It is not, by itself,
proof that a binary or content tree is a player distribution.

## Alternatives

- **Compile every developer-labelled facility out of release builds.** Rejected
  because it removes useful, self-contained in-game capabilities such as F3 and
  teleportation while saying nothing about developer-only assets or external
  interfaces.
- **Ship the complete runtime checkout and hide unwanted facilities.** Rejected
  because hidden authoring content, test craft, viewpoints, and external control
  surfaces are still part of the product and can drift into player-visible
  behavior.
- **Require every planet pipeline to be complete before any pre-alpha release.**
  Rejected because pre-alpha is specifically allowed to expose incomplete world
  production behind honest fallback representations.
- **Fall back whenever an asset is absent.** Rejected because it makes a broken
  package indistinguishable from an intentional maturity decision. Fallbacks
  must be declared and accepted by the active release policy.

## Consequences

The distribution workflow needs a versioned runtime-content manifest and a
capability audit instead of copying whole source directories. Planet recipes and
release policy must name allowed fallbacks. Install verification must test the
extracted allowlisted package, including the absence of forbidden content, while
runtime observability must distinguish bounded product diagnostics from opt-in
investigation streams.
