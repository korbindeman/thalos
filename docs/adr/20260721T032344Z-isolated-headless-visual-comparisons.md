# ADR-20260721T032344Z-isolated-headless-visual-comparisons: Visual comparisons run as isolated headless captures

- **Status:** Superseded by ADR-20260721T192218Z-persistent-visual-iteration
- **Date:** 2026-07-21

## Context

Graphics debugging needs fast, controlled A/B tests and multi-channel views.
The tempting implementation is an in-game split screen with two or more cameras,
each rendering a different configuration. Thalos's renderer, however, deliberately
has one canonical render view: `ViewAnchor`, scene depth, SSAO, atmosphere, terrain
streaming, and the shadow rig follow the one `ShipCamera`. Several of those systems
own viewport-sized resources or enforce that single view with `.single()`.

A live split also changes the viewport dimensions that drive screen-space effects,
LOD, antialiasing, and shadow coverage. It therefore changes more than the factor
under test and can manufacture or conceal the symptom being diagnosed.

The existing headless screenshot path already provides deterministic presets,
fixed framing, capture-only overrides, full-resolution output, and the real game
camera/render stack. What is missing is orchestration that holds every input fixed,
changes one declared factor, and presents the captures as one comparison artifact.

## Decision

Visual A/B and multi-tests use **isolated sequential headless captures**, not a
second live render-view architecture.

- `just compare <preset> <axis>` is the canonical comparison entry point.
- An axis is typed in the comparison runner and declares one capture-only setting
  plus two or more labelled variants. A run changes only that axis.
- Every variant boots as a separate game process. Camera/framing, world inputs,
  resolution, warm-up, graphics preferences, and all inherited screenshot
  overrides remain identical.
- The runner preserves every full-resolution capture and emits a labelled contact
  sheet, baseline-relative pixel diffs, wipe images, and a JSON manifest containing
  the revision, invariant inputs, variant override, output paths, and diff metrics.
- Comparison overrides never write the user's persisted graphics settings.
- Debug channels (AO, shadow factor/cascade, depth, normals, material/LOD IDs,
  lighting lobes) extend the same typed-axis mechanism as they become available.
- An interactive wipe may later display already-rendered A/B images. It must not
  create a second production camera, `ViewAnchor`, or set of shared render resources.

## Alternatives

- **Live multi-camera split screen** — rejected because it conflicts with the
  one-view architecture, duplicates per-view render resources, doubles GPU work,
  and changes viewport-dependent behavior during the test.
- **A shader branch selected by screen quadrant** — rejected as the general A/B
  mechanism because both sides would still share upstream depth, shadows, AO,
  atmosphere, and post state. It is valid only for a local debug visualization of
  data already computed by one render.
- **Toggle variants sequentially inside one game process** — rejected because A can
  warm caches, temporal histories, streamed terrain, or global render resources
  before B. Separate processes make isolation explicit.
- **Continue taking manual screenshots** — rejected because filenames, framing,
  warm-up, and settings drift easily, and the absence of a manifest makes a visual
  conclusion difficult to reproduce.

## Consequences

Comparisons pay startup and warm-up once per variant, so an N-way test is slower
than N simultaneous viewports. In exchange, each image is a real full-resolution
render with no split-screen confound and no parallel render architecture to keep in
sync.

Adding an axis is intentionally small and explicit. Adding a new debug channel may
still require shader instrumentation, but capture, presentation, provenance, and
comparison metrics remain one shared path.
