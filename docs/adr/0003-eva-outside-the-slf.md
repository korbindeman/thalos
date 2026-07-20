# ADR-0003: EVA stays a kinematic body-fixed controller, outside the SLF

- **Status:** Accepted
- **Date:** 2026-06 (recorded 2026-07-18; design detail in `surface_local.md` §10)

## Context

The surface-local frame (SLF) unification moved ship near-surface physics onto a body-fixed
tangent frame with a solid ground collider and Avian integration. EVA (the on-foot player) is the
obvious next candidate to "unify" — it also walks on terrain — and future agents keep being
tempted to fold it in for consistency.

## Decision

EVA is a **deliberately separate kinematic path**: `player_controller::step_eva_controller` runs
a character controller directly in the body-fixed frame, tracking a body-fixed position +
surface-relative velocity with its own grounded/airborne state machine. It has **no collider**
and does not integrate in the SLF. Canonical authority while walking is pinned by the regime
resolver; the controller owns the capsule pose outright.

## Alternatives

- **Make EVA an SLF citizen (Avian capsule + contact solver)** — rejected: EVA gains nothing
  from the SLF's contact-solver stability (a walking capsule needs no solver rest, no gear
  raycasts, no impact model), while inheriting all its costs — collider streaming dependence,
  re-anchor churn, integrator/warp coupling. The body-fixed kinematic form is what fixed the
  historical failure modes: surface velocity is walking speed (m/s) rather than the inertial
  co-rotation `ω×r` (hundreds of m/s), warp can't explode the integrator, and a missing-tile
  height sample holds altitude instead of teleporting to the reference radius.
- **Unify "later" by default** — rejected as a standing assumption: any future unification must
  be justified by a concrete need (e.g. jetpack-EVA contact physics) **and** walk-tested on foot
  before landing (`surface_local.md` §10).

## Consequences

- Two ground-interaction paths exist knowingly (ship SLF, EVA kinematic); the regime resolver is
  the single place that knows which applies.
- EVA remains cheap and warp-stable; jetpack/boarding features extend the controller, not the SLF.
- Agents must not "clean up" this split as accidental duplication — it is a recorded decision.
