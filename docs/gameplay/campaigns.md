# Campaigns, revisions, and runtime sessions

This is the authoritative product and loading model for playable state. Domain
terms are summarized in `CONTEXT.md`; the decision and rejected alternatives are
recorded in ADR-20260802T210926Z.

## 1. Ownership model

The durable root is a `Campaign`, not a scene and not one flat save file. It
owns:

- one shared simulation clock and world configuration;
- N bodies, N bases, N structures, and N vessels with stable IDs;
- per-vessel control capability, control availability, programs, resources,
  damage, and authoritative dynamics state;
- mission state and future mission-plan execution state;
- an immutable revision graph and named bookmarks.

The Bevy world is a projection. Meshes, material handles, local rigid bodies,
cameras, selected panels, and streamed terrain are never persisted as authority.
They are reconciled from the checked-out revision and may be destroyed and
rebuilt without changing the campaign.

`active craft` is operator-local focus: which craft receives manual input and
drives camera/HUD. It is not a campaign-wide ownership singleton and is not a
substitute for `CraftId`.

## 2. Revisions and bookmarks

Each campaign revision is immutable and complete at one epoch. A revision can
have multiple children when the player retries or deliberately explores an
alternate continuation. There is one accepted head for ordinary continuation,
but rejected attempts remain reachable until explicitly pruned.

A bookmark is a named pointer to a revision:

- Save creates a revision and moves/creates a player-named bookmark.
- Quicksave creates a revision and moves the campaign's quicksave bookmark.
- Autosave uses the same primitive with retention policy.
- Quickload checks out the bookmark target. Continuing records a new child; it
  never overwrites the former future.

An exported `.thalos` artifact will contain the campaign metadata, revision
graph, bookmarks, schema/version metadata, and content references. A single
revision export may be supported for sharing, but is not the in-process model.

## 3. The session-load transaction

Every entry point submits a `SessionLoadRequest` with a source and a monotonically
increasing `SessionGeneration`:

1. Read the source through an adapter (new campaign, bundled fixture, or disk).
2. Validate identity, references, invariants, and content compatibility.
3. Migrate older schemas to the current in-memory schema.
4. Atomically replace all authoritative session resources.
5. Reconcile ECS/render/local-physics projections by stable identity.
6. Reveal the requested operator context, then publish the active generation
   once required craft replacement/projection work is idle.

Failure before step 4 leaves the old session intact. Projection failures after
step 4 keep the load screen up, report the generation and failed subsystem, and
are retryable. Systems must reject work stamped for an older generation.

Process services such as content catalogs, renderer pipelines, and terrain
packages may survive a checkout. Campaign state, structure registries, craft
roots, flight plans, transient requests, and local-physics worlds do not leak
across generations.

## 4. Scenario fixtures

A developer scenario is a bundled, versioned campaign situation. It is not a
game mode and has no scenario-only gameplay implementation. Fixtures use an
ephemeral campaign adapter; closing or replacing one discards its revisions by
default. They may still use ordinary bookmarks during a debugging session.

The fixture catalog may supply compact authored source data, but it must compile
to the same validated current snapshot accepted by disk loading. Runtime
placement, craft construction, engine state, base materialization, controls,
and UI read only the loaded snapshot and stable domain records. No consumer may
branch on “came from dev scenario.”

The current compatibility slice keeps `SpawnSituation` inside the fixture
adapter while existing singleton state is migrated. It is source vocabulary,
not a second loader. It must disappear from gameplay consumers as those fields
move into complete craft/base snapshot records.

## 5. Bases and structures

`BaseId` identifies a base within a campaign. `StructureId` identifies one
structure and carries exactly one owning `BaseId`; child structures cannot be
registered without a valid parent base. Multiple bases on one body and multiple
space centers are ordinary data.

Authored fixtures reserve explicit IDs. Player construction allocates IDs from
the campaign. Reconciliation is an upsert by identity. Asking to materialize an
already-present base returns the existing record and ensures its projections;
it cannot append another logical base. An optional `RunwaySite` resource is a
view/cache and never the uniqueness authority.

## 6. Focused flight runs

Every craft topology split (decoupling, docking separation, future undocking)
records a `RunOpportunity` at the source revision. The simulation retains every
resulting craft regardless of UI eligibility.

The UI offers a focused run only when a separated craft has structural
`ControlCapability` (crew, guidance computer, remote-control hardware, and so
on). `ControlAvailability` answers whether that capability can act now and, if
not, why (power, link, damage, program lock). Temporary unavailability disables
an opportunity; it does not erase it.

A retry checks out the opportunity revision and creates a sibling flight run.
The entire campaign is simulated in each run. The run declares the `CraftId`
outcomes it may commit; shared mutations and undeclared craft outcomes cannot be
merged. Accepting one run advances canonical history with those validated
outcomes. Quicksaves inside a run are ordinary revision bookmarks scoped to
that run and can create further child attempts.

## 7. Parallel missions and future automation

Parallel missions coexist in one campaign at one epoch. Background vessels may
coast analytically or execute per-vessel programs, but they advance under the
same clock and can affect the same world. Multiplayer therefore shares the
campaign clock; Thalos does not create per-player time subspaces.

A future mission planner produces a graph of intended operations with
preconditions, success conditions, explicit contingency edges, and terminal
states. Its executor sends demands through the same control path as a human or
autopilot. The world decides what happened:

- a satisfied precondition advances the plan;
- a missed window or rendezvous follows an authored contingency or marks the
  plan failed;
- a manual diversion cancels the remaining plan unless the graph explicitly
  rejoins;
- failure/cancellation stops future automation but leaves all craft and world
  state intact, from which a new plan can be authored.

## 8. Delivery slices

1. **Session foundation:** source/generation types, one runtime request
   consumer, New Game and fixture routing, stable base ownership, idempotent
   space-center reconciliation, regression tests.
2. **Complete snapshot adapter:** separate serializable authoritative snapshot
   records from propagators/runtime caches; fixture compiler and in-memory
   adapter both produce that schema.
3. **Campaign store:** immutable revisions, bookmarks, atomic disk writes,
   migration/validation diagnostics, checkout/branch tests.
4. **Focused runs:** topology-split opportunities, capability/availability,
   scoped commit validation, attempt UI.
5. **Parallel mission execution:** per-vessel programs, scheduling and
   observability on one clock, then the mission-plan graph.
