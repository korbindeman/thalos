# ADR-20260730T003005Z-damage-state-is-canonical-not-solver-owned: Damage state is canonical; the solver only produces load

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

Thalos wants BeamNG-flavoured craft destruction: graded and survivable, with a
part's *function* failing before its *structure*, so a clipped wing is a flight
you fight home rather than a game over. Today damage is a single whole-craft
`hull_destroyed` bool tripped by a ~12 m/s approach-speed tolerance — coarser
even than KSP's per-part model.

[physics.md](../simulation/physics.md) §1.2 already names node-beam soft-body as
the endgame and the reason to own `thalos_physics`, with node-beam as Phase 3
behind a solver rewrite (Phase 1 deferred, Avian still live). Read literally,
that sequencing means **no destruction gameplay at all** until a multi-million-token
solver program completes. The question was whether to accept that, and — if not —
how to build the gameplay now without creating a second damage authority that the
eventual node-beam layer would have to fight or replace.

Two facts from the code shaped the answer. Detaching an assembly is *already*
solved: the graph cut, canonical vessel creation, resource partition, and impulse
application all landed (ADR-20260724T230226Z). And the aero model is whole-body —
`AeroConfig` has one reference area and one lift slope, with roll moment only
from control deflection — so it has **no term that can express asymmetry**, which
makes it, not the physics substrate, the real blocker for the flagship
"fly home on one wing" case.

## Decision

Three commitments.

**1. Damage state is canonical.** `VesselDamage` lives in `physics_canonical`
beside the rest of `VesselRecord` — Bevy-free, `serde`, keyed by blueprint part
index. The physics backend produces *load events*; it never owns, stores, or
interprets damage. Consequences reach flight through one rebuild path
(`rebuild_damaged_aggregates`) that recomputes derived aggregates from
(blueprint ∪ damage); no consumer reads damage state to decide behaviour.

**2. Staged substrate.** A structural graph over the existing aggregate rigid
body ships now; node-beam later becomes a *better producer* of the same damage
state, plus real deformation geometry. Destruction gameplay does not wait on the
solver rewrite, and the solver rewrite does not have to re-litigate the damage
model.

**3. Per-panel aero is a prerequisite.** `evaluate_aero` becomes a genuine panel
sum over the already-per-panel `WingAeroPanel` geometry, gated by a parity test
against the current whole-body model at the Meridian's cruise point.

Damage persists across flights and is repaired only at a base facility.

## Alternatives

- **Wait for node-beam (physics Phase 3 first)** — rejected because its
  prerequisites (Phase 1 ~1–2M tokens, Phase 2 ~0.5–1M) deliver *zero*
  destruction gameplay before Phase 3 itself (~2–4M), and because ~80% of the
  felt experience for aircraft is asymmetric moment, lost thrust, and leaking
  fuel — none of which needs crumple geometry. Node-beam's visual payoff is also
  weakest at Thalos's usual camera distances, where BeamNG's is a close
  third-person car body.
- **Put damage state in the solver** (the natural home if node-beam were the
  substrate) — rejected because damage must survive regime handoff, warp,
  collapse-to-rails, save/load, and the local-scene hydrate/collapse cycle. The
  solver exists only in the surface regime, so solver-owned damage would need a
  mirror, and mirrors are the reconciliation debt physics.md §1.1 is explicitly
  trying to delete.
- **Per-part health with explode thresholds (KSP+)** — rejected as the stated
  non-goal: it cannot express function-before-structure or partial degradation.
- **Keep whole-body aero and bolt on asymmetry deltas** (`roll_moment_asym` etc.
  computed from the damaged-vs-pristine panel difference) — rejected because it
  is a second aero model to keep consistent with the first, and it still cannot
  express a partially degraded or jammed individual panel. The panel sum
  *removes* tuned coefficients (roll damping, roll authority become geometry)
  rather than adding them.
- **A debris-only visual path for shed parts** — rejected for the same reason
  ADR-20260724T230226Z rejected it for stage separation: it creates a second,
  non-persistent answer to "where is this object?".

## Consequences

- **The Meridian's handling gets re-tuned.** Per-panel forces change felt
  behaviour across every flight scenario. This is why the parity test is part of
  the slice rather than a follow-up: it converts "does it still feel right" into
  a bounded, testable job. The whole-body model survives only as that test's
  fixture — not as a runtime path.
- **Replay safety becomes a real correctness requirement.** Plastic accumulation
  is history-dependent and the fleet replays to the earliest contact time on a
  collision, so load events must be staged per epoch and committed once at
  interval close. This is the subtlest part of the feature and it gets its own
  test.
- **Failure is deterministic + seeded**, never runtime RNG — forced by the same
  replay behaviour, and by the requirement that save/load not re-roll a marginal
  part.
- **`is_destroyed` becomes a derived verdict** behind the existing accessor, so
  `control_bus`, `fuel`, `bridge`, and `scenario_menu` are untouched.
- **physics.md §7 Phase 3 is narrowed** to what only node-beam can provide —
  crumple geometry, suspension ruin, mesh-to-node skinning — instead of owning
  the damage model.
- Full design and delivery slices: [damage.md](../simulation/damage.md).
