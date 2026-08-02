# ADR-20260802T210926Z-campaign-revisions-and-session-loading: campaigns own history; sessions are loaded projections

- **Status:** Accepted
- **Date:** 2026-08-02

## Context

Thalos currently has several independent ways to construct a playable world:
process boot, the start screen, developer scenarios, destruction respawn,
shipyard launch, and the space-center loader. They mutate live resources and
append entities directly. Consequently, the same nominal situation can select a
different craft or setup path, and requesting the space center twice can create
two coincident copies. This cannot be repaired with more caller-side guards:
the product also needs multiple player-built space centers, quicksave/retry
inside a flight attempt, FMRS-style focused runs, parallel missions on one
clock, and future automated mission plans.

## Decision

A **Campaign** is the durable aggregate. It owns one shared simulation clock,
the fleet, bases and structures, mission state, and an immutable graph of
complete campaign revisions. Manual saves, quicksaves, and autosaves are
bookmarks pointing to revisions. Continuing from an older revision creates a
child revision; it never mutates history in place.

A **runtime session** is a disposable projection of one checked-out revision.
Loading is one generation-stamped transaction: validate and migrate the source,
replace authoritative session state, then reconcile ECS/render/physics
projections. Feature entry points request a load; they do not spawn campaign
objects themselves. Stable domain identity, not entity presence or optional
resources, enforces uniqueness. In particular, bases and structures have
campaign-stable IDs, and materializing the same base twice is an idempotent
reconciliation, not a second registration.

Developer scenarios are bundled, versioned **scenario fixtures**. A fixture is
loaded through the same transaction as a future disk-backed campaign revision,
but is hosted by an ephemeral campaign adapter and discarded by default. New
Game creates a durable campaign through that same seam.

Focused flight retries fork the complete campaign revision because clock,
collision, resources, and the rest of the fleet remain shared. A run may commit
only its declared craft outcomes; shared campaign mutations are forbidden.
Every craft-topology split records a run opportunity internally, while UI
offers only craft with structural `ControlCapability`. Temporary inability to
command one is separate `ControlAvailability` with a reason. Exactly one run is
accepted into canonical history; other attempts remain addressable.

Parallel missions are concurrent activity in one authoritative campaign and
clock, not history branches or time subspaces. Operator focus is local UI state,
not campaign authority. Future mission plans are executable intent graphs with
explicit contingencies; world state remains authoritative. Failure or diversion
terminates future automation but does not erase craft or roll back the campaign.

## Alternatives

- **Keep scenarios as procedural entry modes and add guards** — rejected
  because callers can still bypass a guard, every new entry point forks setup
  behavior, and there is no durable identity on which to base save/load.
- **One save file per quicksave or attempt** — rejected because it hides the
  history relationship, duplicates campaign metadata, and makes accepting one
  focused outcome an ad hoc file merge.
- **Per-vessel timelines or multiplayer time subspaces** — rejected because
  collisions, launch windows, resources, structures, and coordinated missions
  require one shared world clock. Alternate histories and concurrent missions
  are different concepts.
- **Replay input to reconstruct an attempt** — rejected in favor of the
  recorded-state replay decision in ADR-20260730T212556Z. Input replay is not a
  deterministic or sufficiently complete save primitive.
- **Make the Bevy `World` the save authority** — rejected because entity IDs,
  renderer caches, local-physics bodies, and presentation resources are runtime
  projections with incompatible lifecycles.

## Consequences

The loader and revision schema become production compatibility surfaces and
must be versioned, validated, observable, and independently testable. Runtime
systems must separate authoritative records from projections and tolerate
reconciliation after checkout. A load may reuse process-level GPU/content
services, but it must replace session-level state as one generation.

The first vertical slice introduces the session source/generation seam, routes
New Game and all developer starts through it, makes base identity registry-owned,
and deletes append-style space-center construction. Disk persistence, full
revision/bookmark storage, focused-run commit, and mission orchestration follow
behind the same interfaces; they do not get parallel loaders.
