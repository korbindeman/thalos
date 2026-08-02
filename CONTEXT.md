# Thalos domain language

## Campaign

**Definition:** One durable player universe: its shared clock, bases, fleet,
mission state, immutable revision graph, and named bookmarks.

**Aliases:** save slot (UI only)

**Avoid:** save file, when referring to one exported revision rather than the
whole graph.

**Example dialogue:**

> A quicksave adds a bookmark inside the current campaign; it does not create a
> second campaign.

## Campaign revision

**Definition:** An immutable, complete campaign state at one epoch, with zero or
more parent revisions. Checking out an older revision and continuing creates a
new child revision.

**Aliases:** checkpoint

**Avoid:** save, snapshot, branch (unless the graph relationship is the point)

**Example dialogue:**

> Retrying the booster landing checks out the separation revision and records a
> new child revision.

## Bookmark

**Definition:** A mutable name pointing at a campaign revision. Manual saves,
autosaves, and quicksaves differ by bookmark policy, not by file format.

**Aliases:** normal save, quicksave, autosave (policy-specific UI terms)

**Avoid:** copy of the world

**Example dialogue:**

> Quicksave moves the `quicksave` bookmark to the new revision.

## Runtime session

**Definition:** The currently checked-out campaign revision projected into the
running Bevy world, plus operator-local presentation state. It is disposable;
the campaign revision is authoritative.

**Aliases:** session

**Avoid:** world, when the distinction between authoritative state and ECS
projection matters.

**Example dialogue:**

> Loading replaces the runtime session from one validated revision; it never
> appends a second copy of the revision's bases.

## Scenario fixture

**Definition:** A bundled, versioned campaign situation used for development.
It enters the same loader as a durable campaign revision, but uses an ephemeral
campaign adapter and is discarded by default.

**Aliases:** dev scenario, fixture

**Avoid:** game mode

**Example dialogue:**

> The runway fixture and a future disk save both hydrate through the session
> loader, so the aircraft cannot receive scenario-only behavior.

## Flight run

**Definition:** One attempted continuation from a campaign revision, usually
scoped to a declared set of controllable craft outcomes. A retry is a sibling
run; accepting one run advances the campaign's canonical history.

**Aliases:** attempt

**Avoid:** timeline, unless discussing the whole revision graph

**Example dialogue:**

> The booster guidance unit makes its separated stage eligible for a focused
> flight run; debris without control capability is simulated but not offered.

## Mission plan

**Definition:** A future executable graph of intended operations,
preconditions, explicit contingencies, and termination rules. The plan commands
the world but never overrides the world's authoritative outcome.

**Aliases:** mission program

**Avoid:** script, when it implies success is forced or world state is private

**Example dialogue:**

> Missing the rendezvous precondition follows a declared contingency or fails
> the mission plan; the craft and campaign continue from the actual state.
