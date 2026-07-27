# ADR-20260724T221858Z-unified-viewpoint-registry: Saved and agent-scripted views share one public catalog

- **Status:** Accepted
- **Date:** 2026-07-25
- **Supersedes:** ADR-20260724T211627Z-data-backed-shared-viewpoint-catalog

## Context

The first data-backed manager separated exact saved camera poses from the
existing agent capture presets. That distinction was mechanically accurate:
ocean/desert searches, cloud temporal motion, plume setup, and Mira landmark
selection are executable behavior rather than static transforms. But it left
two public viewpoint lists. Developers could browse only saved poses in F8,
while agents continued to know a parallel Rust/CLI preset catalog. A viewpoint
manager that omits the views agents already use is not unified.

Flattening those procedural views into captured transforms would also be wrong.
They would stop following regenerated terrain, sun geometry, landmark size, or
the diagnostic state they exist to exercise.

## Decision

`assets/viewpoints.json` is the one public registry for both forms:

- `viewpoints` contains exact body-fixed saved poses.
- `scripted_viewpoints` contains stable id, display metadata, and a `driver`
  capability name. The runtime driver performs the procedural focus/framing and
  diagnostic setup; it is an executor, not a second user-facing catalog.

F8 displays both in one list. It can view either form, edit metadata, rename,
and delete entries. “Replace from current” on a scripted view deliberately
converts it to an exact saved pose under the same identity.

The capture CLI resolves public ids through the catalog. Existing agent views
retain their ids, so commands remain stable while the source of discovery and
identity moves from the Rust constant to JSON. Runtime driver names remain
validated against compiled capabilities so data cannot request arbitrary code.

Live “View” applies a scripted view's camera focus and framing. Capture-only
diagnostic state (for example false-colour ocean slopes, forced plume pressure,
or a temporal camera slew) remains in the headless executor and is reported as
such; the manager does not pretend that posing a camera enables those modes.

## Alternatives

- **Show compiled presets as a second read-only panel** — rejected because it
  preserves two registries and prevents agents/developers from editing the
  common list.
- **Convert every procedural preset to one raw pose** — rejected because it
  destroys search/lighting/diagnostic semantics and becomes stale as authored
  terrain changes.
- **Encode every diagnostic algorithm directly in JSON** — rejected because
  data should select and parameterize capabilities, not become an untyped
  programming language.

## Consequences

The manager, source file, and capture command surface now enumerate the same
views. Adding a genuinely new procedural behavior still requires one compiled
driver, but adding aliases, curating metadata, removing a public view, or
turning a procedural result into a fixed composition does not require another
Rust enum entry.

The compiled driver set remains temporarily named by `CAPTURE_PRESETS` for
protocol validation and internal compatibility scheduling. It must not be used
as the developer-facing viewpoint list.
