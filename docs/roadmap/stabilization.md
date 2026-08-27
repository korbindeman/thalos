# Stabilization and distribution (`stab`)

**Status:** near-future program · **Started:** 2026-08-03
**Decision:** [ADR-20260803T061130Z-distribution-boundary-is-capability-and-content](../adr/20260803T061130Z-distribution-boundary-is-capability-and-content.md)
**Cross-ref prefix:** `stab §N`

## §1 Thesis

Stabilization means making incomplete systems explicit, inspectable, and safe
to distribute. It does not mean pretending every planet pipeline is finished.

A planet pipeline may be incomplete, but its state must never be ambiguous. A
pre-alpha release may deliberately render a body with the existing solid sphere
or another declared fallback. It may not silently substitute that fallback
because a required artifact is missing, corrupt, stale, or was omitted from the
archive.

The same distinction applies to the application. F3 and teleportation are
useful parts of a planetary sandbox and may ship. Viewpoint authoring, capture
control, bulk investigation dumps, debug craft, and checkout-facing facilities
are development surfaces and do not belong in the player distribution. The
boundary is capability and content, not the vague label "debug" and not Cargo's
optimization profile.

## §2 Fixed decisions

- **Offline planet packages remain the shipping authority.** The runtime
  consumes validated immutable artifacts behind the terrain/package contract;
  it does not run planetary neural generation or hydrology for the player.
- **One small typed build graph.** The graph orchestrates coarse artifacts and
  their dependencies. It is not a revival of the archived per-sample field DAG
  and does not become a plugin framework before a second implementation needs
  one.
- **Every stage is content-addressed.** A result binds its input artifacts,
  configuration, implementation/model identity, output hashes, validation
  receipt, and completion outcome. Changing an input makes downstream state
  observably stale.
- **Final hydrology agrees with final height.** Base terrain and neural/detail
  bands compose before authoritative drainage. If a geomorphic pass carves or
  deposits height, drainage is re-solved before packaging; the shipped height,
  reach graph, catchment, and discharge cannot describe different terrain.
- **Fallback is policy, not error handling.** Pre-alpha profiles may accept a
  named fallback per planet/stage. Missing or invalid content never opts into a
  fallback by accident.
- **Backlog status and planet status are different facts.** `docs/backlog.jsonl`
  remains the only authority for development work (`just queue`). Planet status is derived
  from recipes, hashes, artifacts, and acceptance receipts.
- **The distribution is allowlisted.** Stock craft and runtime assets are named
  explicitly. Copying all of `assets/` or `ships/` is not a release contract.
- **Self-contained in-game tools may ship.** F3, teleportation, bounded
  diagnostics, build identity, and install verification survive. External
  authoring/control surfaces, saved viewpoints, debug/test craft, capture
  infrastructure, and investigation-scale data emission do not.
- **Optimized is not synonymous with distributed.** `cargo --release` remains
  valid for offline tools, profiling, and developer experiments. The canonical
  distribution capability/content manifest is the product boundary.

## §3 Four records, four questions

| Record | Authority | Question answered |
|---|---|---|
| Planet recipe | tracked source | What should be built for this body, and which fallbacks may this release profile accept? |
| Stage receipt | generated build evidence | What ran, from which exact inputs, what did it emit, and did local validation pass? |
| Planet-package manifest | immutable runtime content | What exact height and attachment authority does the game consume? |
| Release policy and distribution manifest | tracked product contract | Is this set of packages, capabilities, craft, and assets sufficient to publish? |

None replaces the backlog. A stage implementation can be unfinished while the
current artifact is valid through a declared fallback; conversely, completed
code can have a stale planet artifact.

## §4 Planet build graph

The first graph is deliberately small. Parallel derivations are allowed where
their data permits it, but every dependency is explicit.

1. **Authored source** — body identity, seed, physical constants, lore
   constraints, permanent interventions, model/config locks, and the release
   profile's fallback declarations.
2. **Base terrain and conditioning** — signed macro height, coastline authority,
   climate/landform conditioning, and the continuous global parent band.
3. **Neural height** — hierarchical residual bands and authored detail windows,
   each conditional on its parent and reconciled across package seams.
4. **Height-mutating post-process** — only processes that alter immutable static
   height. Each emits a recoverable delta rather than hiding the source surface.
5. **Hydrology** — the solve/carve/re-solve contract in
   [drainage.md](../world/drainage.md): topology and catchment remain separate
   from climate-weighted discharge, and final products bind final height.
6. **Surface attachments** — biomes, landcover/material weights, roughness,
   horizon data, landmarks, scatter conditioning, and other derived channels
   that do not own height.
7. **Package assembly** — immutable runtime payloads, indices, fallbacks,
   provenance closure, checksums, and declared reconstruction error.
8. **Planet acceptance** — independent product checks over the assembled
   artifact. Stages do not certify their own suitability for release.

### Stage contract

Every stage has one stable id and declares:

- typed input and output channels;
- required and optional inputs;
- configuration, tool, algorithm, model, and dataset identity;
- input and output content hashes;
- deterministic or explicitly bounded-nondeterministic behavior;
- stage-local validation metrics and verdict;
- phase timings, peak resource use, outcome, and failure reason through the
  shared tool diagnostics lane;
- downstream invalidation edges;
- the fallback artifact, if the active release profile permits one.

Stage state is derived as `not-applicable`, `missing`, `stale`, `built`,
`validated`, `accepted`, or `fallback-accepted`. A process-local `running`
state may be reported while a tool owns the build, but it is never persisted as
success.

## §5 Release maturity and fallbacks

Requirements are versioned by release profile and planet. They are not inferred
from SemVer, body class, or whatever files happen to exist.

The initial `pre-alpha` profile is permissive about completeness and strict
about intent:

- every distributed body has a valid recipe and a runtime representation;
- an incomplete stage names the exact fallback used, including a solid sphere;
- fallback acceptance is visible in `planet status`, the release receipt, and
  build information;
- required present artifacts pass schema, checksum, provenance, bounds, seam,
  and reconstruction validation;
- the archive contains every referenced runtime byte and no checkout-relative
  dependency can satisfy a missing file;
- a fallback cannot hide a stale or corrupt artifact that the recipe selected.

Later profiles tighten the per-planet matrix. They may require neural coverage,
hydrology, surface attachments, landing-scale height/collision parity, or
specific visual gates without changing how the runtime addresses a planet.

## §6 Development experience

One maintained CLI should answer both humans and CI:

```text
just planet status [body] [--json]
just planet build <body> [--from <stage>]
just planet check <body|all> --profile <profile> [--json]
just distribution verify
```

`status` derives freshness from current inputs and artifacts; nobody manually
marks a stage built. `build` skips fresh stages, invalidates downstream outputs
by hash, records every phase through `thalos_diagnostics::ToolRun`, and never
promotes a partial output. `check` emits one machine-readable acceptance receipt
and a concise human table. CI consumes exit status and structured output rather
than scraping prose.

The initial implementation wraps the current procedural/macro source, neural
payloads, 2 km hydrology preview, Mira package baker, and package validator. It
does not rewrite those producers merely to put them behind the graph.

## §7 Distribution boundary

| Ships in the player application/package | Excluded from the player distribution |
|---|---|
| F3 performance/diagnostic view | Saved/scripted viewpoint catalog and F8/F9 authoring flows |
| Teleportation and other self-contained sandbox navigation | Headless capture server/client control and checkout-facing protocols |
| Bounded production diagnostics, build identity, install verification | Investigation-scale continuous dumps and opt-in experiment traces |
| Declared planet representations, including accepted solid-sphere fallbacks | Unreferenced intermediate bakes, training data, checkpoints, and tool artifacts |
| Explicit stock craft and player-facing scenarios | Debug/test craft and authoring-only scenarios/content |
| Runtime licenses and player documentation | Offline bakers, trainers, map exporters, and developer documentation |

This is an audience/capability classification, not a mandate to force every
item into a Cargo feature. Separate binaries already exclude most offline
tools. Player-application code that crosses the boundary should be assembled at
one explicit composition seam, while content inclusion is enforced by a
versioned distribution manifest.

Production observability remains because release-only failures must be
diagnosable. The release policy limits cadence, volume, retention, and sensitive
content; it does not turn the player binary into a black box.

## §8 CI and release gates

The canonical publishing lane performs, in order:

1. Resolve one immutable source revision and canonical capability set.
2. Validate the distribution manifest and reject forbidden/unclassified files.
3. Run deterministic planet acceptance for the active profile; publish its
   per-body stage/fallback summary into the workflow report.
4. Build the player application on each target platform.
5. Assemble only allowlisted runtime content and licenses.
6. Extract into an unrelated directory and run install verification without a
   checkout fallback.
7. Verify build identity, capability set, planet-package hashes, stock craft,
   absence rules, and release receipt from the extracted artifact.
8. Publish only if every platform proves the same source, policy, and content
   identities.

Deterministic data/package gates run on ordinary CI. GPU visual gates continue
through the existing capture lane and become a release-candidate requirement
only on controlled hardware; a random hosted GPU is not allowed to define
planet correctness.

## §9 Work order

- **STAB-1 — Planet pipeline contract and status.** Define the recipe, typed
  stage receipt, freshness derivation, acceptance receipt, and read-only
  `planet status` surface against current artifacts.
- **STAB-2 — Wrap current producers.** Put base/macro, neural payload,
  hydrology preview, Mira bake, and package validation behind the graph without
  changing their algorithms; make interruption and downstream invalidation
  testable.
- **STAB-3 — Pre-alpha planet acceptance.** Check in the first per-planet
  requirement/fallback matrix and gate deliberate solid-sphere fallbacks,
  stale-artifact rejection, provenance closure, and package invariants in CI.
- **STAB-4 — Distribution content manifest.** Replace whole-directory copying
  with an explicit stock-craft/runtime-asset/license manifest and positive plus
  negative archive verification.
- **STAB-5 — Player capability boundary.** Keep F3 and teleportation; remove
  viewpoint authoring, capture/external control, bulk debug output, and
  debug-content dependencies from the distributed application composition.
- **STAB-6 — Canonical publish lane.** Make every platform build, assemble,
  extract, verify, and report the same immutable release identity and planet
  acceptance profile.
- **STAB-7 — Tightening profiles.** Raise requirements planet by planet as
  neural, hydrology, attachments, collision, and visual evidence become product
  promises; no blanket date- or SemVer-derived switch.

## §10 Open product choices

- Which existing craft, if any, graduate into the first named stock-craft set.
- Whether F3 and teleportation are directly discoverable in pre-alpha or live
  behind an in-game advanced/developer setting. Both remain inside the shipped
  application either way.
- The first later-than-pre-alpha requirement matrix: which bodies become
  landing-capable promises and which remain intentionally representation-only.
- Platform packaging beyond portable archives: installer shape, signing,
  notarization, update channel, crash-report consent, and support-bundle UX.
