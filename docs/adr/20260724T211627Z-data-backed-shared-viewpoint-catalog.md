# ADR-20260724T211627Z-data-backed-shared-viewpoint-catalog: Saved viewpoints are shared authored data

- **Status:** Superseded by ADR-20260724T221858Z-unified-viewpoint-registry
- **Date:** 2026-07-24
- **Supersedes:** the overwrite-only saved-perspective portion of BL-24

## Context

The original F8 handoff writes one `latest_perspective.json` diagnostic artifact.
It supports a player showing one camera pose to an agent, but it has no identity,
history, discovery, editing, or round trip: the next save destroys the previous
one, agents cannot add a durable view to a list, and the game cannot browse or
apply a view authored outside the running process.

Scripted capture presets solve a different problem. Many search for a dynamic
site or synthesize a diagnostic framing and therefore remain executable capture
logic. A saved camera point is authored state and should not require a Rust enum
variant or rebuild.

## Decision

Store named saved viewpoints in the versioned, human-readable
`assets/viewpoints.json` catalog. The portable capture protocol owns the Serde
schema and validation; it stays free of Bevy and filesystem policy.

Each viewpoint records a stable slug, display name and description, canonical
boot context (body, spawn, hub mode), provenance time, viewport and vertical
FOV, plus camera position and orientation in the body's authored surface-fixed
frame.

Both interfaces are projections of that catalog:

- F8 opens an in-game egui manager for create, read, apply, update, rename and
  delete operations. Opening/reloading the manager rereads the file so an agent
  can edit it while the game is running.
- The capture CLI accepts a viewpoint slug (or `viewpoint:<slug>`) alongside
  scripted preset names. Headless replay loads the catalog at request time and
  uses the real `ShipCamera`. `latest` resolves to the newest catalog entry for
  continuity with the old handoff.

Scripted presets remain code when their framing is computed, searched, or tied
to a diagnostic state machine. They do not get copied into the viewpoint file.

## Alternatives

- **Keep one latest file and add UI around it** — rejected because it preserves
  overwrite semantics and gives agents no stable names to exchange or version.
- **Turn every viewpoint into a Rust preset** — rejected because authored camera
  data would still require code edits, compilation, and duplicate catalogs.
- **Store viewpoints only under generated artifacts or per-user settings** —
  rejected because agents and developers need one inspectable, versionable
  collaboration surface.
- **Convert computed diagnostic presets to data immediately** — rejected because
  site searches, temporal probes, and diagnostic resource setup are behavior,
  not static camera points.

## Consequences

Viewpoint edits can round-trip game → file → agent → game without recompiling.
The asset is developer-authored source data, so an installed/read-only build may
show a clear write error rather than silently falling back to user settings.
Applying a viewpoint restores geographic framing and lens, not a full simulation
snapshot; its recorded boot context remains the recipe for canonical headless
replay.
